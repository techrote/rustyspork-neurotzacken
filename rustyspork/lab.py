"""Content-addressed runs and atomic checkpoint publication."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
import uuid
from pathlib import Path

from . import __version__
from .gguf import GGUF, GGUFError, open_gguf
from .surgery import Budget, Prepared, prepare, verify_candidate, write_candidate

RUN_SCHEMA = "rsn.experiment.v1"
PLAN_LIMIT = 1024 * 1024


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise GGUFError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise GGUFError(f"Unsupported non-finite JSON constant {value}")


def load_json(path: Path, max_bytes: int) -> dict:
    with path.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise GGUFError(f"{path.name} exceeds the {max_bytes}-byte JSON input limit")
    try:
        value = json.loads(data, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    except (ValueError, RecursionError) as error:
        raise GGUFError(f"{path.name}: invalid JSON ({error})") from error
    if not isinstance(value, dict):
        raise GGUFError(f"{path.name} must contain a JSON object")
    return value


def write_json(path: Path, value: dict) -> None:
    data = canonical_json(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def environment() -> dict:
    checkout = Path(__file__).parent
    modules = ("adapters.py", "gguf.py", "lab.py", "surgery.py")
    return {
        "rustyspork": __version__, "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.system(), "machine": platform.machine(),
        "engine_sha256": digest({
            name: hashlib.sha256((checkout / name).read_bytes()).hexdigest() for name in modules
        }),
    }


def report(run: Path) -> dict:
    run = run.resolve()
    receipt = load_json(run / "receipt.json", 32 * 1024 * 1024)
    if (receipt.get("schema_version") != RUN_SCHEMA or receipt.get("state") != "mechanically_verified"
            or receipt.get("run_id") != run.name or digest(receipt.get("inputs")) != run.name
            or not (run / "candidate.gguf").is_file()):
        raise GGUFError("Directory does not hold a completed RustySpork run")
    return receipt


def compact(model: GGUF, plan: dict, root: Path, budget: Budget) -> dict:
    plan_bytes = canonical_json(plan)
    if len(plan_bytes) > PLAN_LIMIT:
        raise GGUFError(f"Plan exceeds the {PLAN_LIMIT}-byte limit")
    root.mkdir(parents=True, exist_ok=True)
    inputs = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "engine": environment(), "limits": budget.receipt()}
    run_id = digest(inputs)
    final = root / run_id
    stage = None
    phase = "validate"
    try:
        prepared = prepare(model, plan, budget)
        if prepared.kind == "probe":
            raise GGUFError("Probe is read-only; use the probe command")
        if final.exists():
            phase = "verify"
            return reuse_run(prepared, final, budget)
        needed = prepared.candidate_bytes + len(plan_bytes)
        if shutil.disk_usage(root).free < needed:
            raise GGUFError(f"At least {needed} free bytes are required at the output location")
        phase = "write"
        stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.", suffix=".partial", dir=root))
        model_path = stage / "candidate.gguf"
        with model.open() as source, model_path.open("xb") as out:
            write_candidate(out, source, prepared, budget)
            model.assert_unchanged(source)
            out.flush()
            os.fsync(out.fileno())
        phase = "verify"
        budget.check()
        evidence = verify_candidate(prepared, open_gguf(model_path, check=budget.check), budget)
        receipt = {
            "schema_version": RUN_SCHEMA, "run_id": run_id, "state": "mechanically_verified",
            "inputs": inputs, "runtime_evaluation": None, "dataset": None, "verification": evidence,
        }
        write_json(stage / "plan.json", plan)
        write_json(stage / "receipt.json", receipt)
        phase = "publish"
        budget.check()
        try:
            stage.rename(final)
        except OSError:
            if not final.exists():
                raise
            return reuse_run(prepared, final, budget)
        return receipt
    except (ValueError, OSError, KeyboardInterrupt) as error:
        failure = {
            "schema_version": RUN_SCHEMA, "run_id": run_id, "state": "cancelled"
            if isinstance(error, KeyboardInterrupt) else "blocked" if phase == "validate" else "failed",
            "phase": phase, "reason": str(error) or type(error).__name__, "inputs": inputs,
            "claimed_source_sha256": plan.get("source_sha256"),
        }
        try:
            write_json(root / f"{run_id}.failure-{uuid.uuid4().hex}.json", failure)
        except OSError as journal_error:
            error.add_note(f"Unable to write failure receipt: {journal_error}")
        raise
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def reuse_run(prepared: Prepared, path: Path, budget: Budget) -> dict:
    receipt = report(path)
    if load_json(path / "plan.json", PLAN_LIMIT) != prepared.plan:
        raise GGUFError("Stored plan differs from the exact requested plan")
    evidence = verify_candidate(prepared, open_gguf(path / "candidate.gguf", check=budget.check), budget)
    if evidence != receipt.get("verification"):
        raise GGUFError("Stored run fails its recorded checkpoint verification")
    return receipt
