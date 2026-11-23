"""Command-line entry point for the Windows and headless lab."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .adapters import inspect_moe, research_arms
from .fixtures import create_fixture
from .gguf import GGUFError, open_gguf
from .lab import PLAN_LIMIT, compact, load_json, report, write_json
from .surgery import OPERATIONS, Budget, make_plan, prepare, verify_candidate


def comma_indices(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Enter comma-separated integer source indices") from error


def budgets(command: argparse.ArgumentParser) -> None:
    command.add_argument("--max-output-bytes", type=int, default=64 * 1024 * 1024, help="maximum candidate bytes")
    command.add_argument("--chunk-bytes", type=int, default=1024 * 1024, help="maximum streamed I/O chunk (up to 8 MiB)")
    command.add_argument("--max-seconds", type=int, default=60, help="cooperative operation deadline in seconds")


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(prog="rustyspork", description="CPU-first GGUF model surgery and verification.")
    main.add_argument("--version", action="version", version=__version__)
    subs = main.add_subparsers(dest="command", required=True)
    inv = subs.add_parser("inventory", help="inspect GGUF metadata, storage, decoder types and MoE layout")
    inv.add_argument("model", type=Path)
    inv.add_argument("--header-only", action="store_true", help="skip the full source SHA-256 read")
    inv.add_argument("-o", "--output", type=Path)
    budgets(inv)
    fixture = subs.add_parser("fixture", help="write a deterministic lfm2moe checkpoint")
    fixture.add_argument("--experts", type=int, choices=(4, 8), default=4)
    fixture.add_argument("-o", "--output", type=Path, required=True)
    plan = subs.add_parser("plan", help="resolve a selection to a source-bound plan")
    plan.add_argument("model", type=Path)
    plan.add_argument("--operation", choices=sorted(OPERATIONS), default="gouge")
    plan.add_argument("--keep", type=comma_indices, help="ascending original survivor IDs for every routed layer")
    plan.add_argument("--tensor", help="exact tensor name from inventory")
    plan.add_argument("--indices", type=comma_indices, help="ascending logical flat indices")
    plan.add_argument("--scale", type=float, help="scoop scale in [0, 1]")
    plan.add_argument("--amplitude", type=float, help="scrape amplitude in [0, 1]")
    plan.add_argument("--seed", type=int, help="unsigned 64-bit scrape seed")
    plan.add_argument("-o", "--output", type=Path)
    budgets(plan)
    for name, help_text in (
        ("validate", "validate a source-bound plan"),
        ("dry-run", "resolve a plan and predict exact storage changes"),
        ("probe", "measure the logical values selected by a Probe plan"),
        ("compact", "write, verify and publish a new candidate checkpoint"),
        ("verify", "independently reopen and verify a candidate against its donor and plan"),
    ):
        command = subs.add_parser(name, help=help_text)
        command.add_argument("model", type=Path)
        command.add_argument("plan", type=Path)
        if name == "compact":
            command.add_argument("--output", type=Path, required=True, help="parent directory for content-addressed runs")
        else:
            if name == "verify":
                command.add_argument("candidate", type=Path)
            command.add_argument("-o", "--output", type=Path)
        budgets(command)
    report_command = subs.add_parser("report", help="read a completed run receipt")
    report_command.add_argument("run", type=Path)
    subs.add_parser("arms", help="list KAT and LFM research arms and their authorization gates")
    demo = subs.add_parser("demo", help="run the end-to-end CPU demonstration")
    demo.add_argument("--output", type=Path, required=True, help="a fresh demonstration directory")
    demo.add_argument("--experts", type=int, choices=(4, 8), default=4)
    budgets(demo)
    return main


def inventory(model, budget: Budget, *, header_only: bool = False) -> dict:
    receipt = model.inventory(include_sha256=not header_only, chunk_size=budget.chunk_size, check=budget.check)
    layout = inspect_moe(model)
    receipt["moe"] = asdict(layout) if layout else None
    budget.check()
    return receipt


def demonstration(output: Path, experts: int, budget: Budget) -> dict:
    output.mkdir(parents=True)
    model = open_gguf(create_fixture(output / "donor.gguf", experts), check=budget.check)
    write_json(output / "inventory.json", inventory(model, budget))
    keep = list(range(0, experts, 2))
    plan = make_plan(model, budget, operation="gouge", keep=keep)
    write_json(output / "plan.json", plan)
    dry_run = prepare(model, plan, budget).summary()
    write_json(output / "dry-run.json", dry_run)
    receipt = compact(model, plan, output / "runs", budget)
    run_dir = output / "runs" / receipt["run_id"]
    recorded = report(run_dir)
    write_json(output / "report.json", recorded)
    return {"state": recorded["state"], "run_directory": str(run_dir.resolve()), "receipt": recorded}


def execute(args: argparse.Namespace) -> dict:
    command = args.command
    if command == "fixture":
        path = create_fixture(args.output, args.experts)
        return {"path": str(path.resolve()), "experts": args.experts, "file_bytes": path.stat().st_size}
    if command == "report":
        return report(args.run)
    if command == "arms":
        return research_arms()
    budget = Budget(args.max_output_bytes, args.chunk_bytes, args.max_seconds)
    if command == "demo":
        return demonstration(args.output, args.experts, budget)
    model = open_gguf(args.model, check=budget.check)
    if command == "inventory":
        return inventory(model, budget, header_only=args.header_only)
    if command == "plan":
        return make_plan(
            model, budget, operation=args.operation, keep=args.keep, tensor_name=args.tensor,
            selected=args.indices, scale=args.scale, amplitude=args.amplitude, seed=args.seed,
        )
    plan = load_json(args.plan, PLAN_LIMIT)
    if command == "compact":
        return compact(model, plan, args.output, budget)
    prepared = prepare(
        model, plan, budget, enforce_output=command == "dry-run", require_compaction=command == "verify",
    )
    if command == "verify":
        return verify_candidate(prepared, open_gguf(args.candidate, check=budget.check), budget)
    if command == "probe":
        if prepared.probe is None:
            raise GGUFError("Probe command requires a Probe plan")
        return prepared.probe
    return prepared.summary()


def main() -> int:
    args = parser().parse_args()
    try:
        result = execute(args)
        if args.command in {"inventory", "plan", "validate", "dry-run", "probe", "verify"} and args.output:
            write_json(args.output, result)
        else:
            print(json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2))
    except (ValueError, OSError) as error:
        print(json.dumps({"state": "error", "reason": str(error)}, ensure_ascii=True), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('{"state":"cancelled"}', file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
