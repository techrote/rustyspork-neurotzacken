"""Run a bounded CPU loader and forward-pass check on tiny local GGUFs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


DEFAULT_MODEL_LIMIT = 64 * 1024 * 1024
DEFAULT_TIMEOUT = 30
HTTP_LIMIT = 64 * 1024
REQUEST = {
    "prompt": [1, 3, 4, 5],
    "n_predict": 1,
    "n_probs": 5,
    "temperature": -1.0,
    "seed": 42,
    "ignore_eos": True,
    "return_tokens": True,
    "stream": False,
}
CPU_ARGS = [
    "--device", "none", "--gpu-layers", "0", "--no-kv-offload", "--no-op-offload",
    "--ctx-size", "128", "--threads", "1", "--threads-batch", "1",
    "--threads-http", "1", "--parallel", "1", "--n-predict", "1",
    "--check-tensors", "--no-webui",
]


def identify(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest}


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def request_json(opener, url, timeout, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        payload = response.read(HTTP_LIMIT + 1)
        if len(payload) > HTTP_LIMIT:
            raise ValueError("response exceeded the smoke-test byte budget")
    return json.loads(payload)


def completion_evidence(response, alias):
    if response.get("model") != alias:
        raise ValueError("completion did not identify the launched model alias")
    tokens = response.get("tokens", [])
    probabilities = response.get("completion_probabilities", response.get("probs", []))
    if len(tokens) != 1 or type(tokens[0]) is not int or len(probabilities) != 1:
        raise ValueError("expected one generated token with probability evidence")
    prediction = probabilities[0]
    top = prediction.get("top_logprobs", prediction.get("probs", []))
    if not top or len(top) > REQUEST["n_probs"]:
        raise ValueError("expected a bounded prefix of token scores")
    scores = []
    for item in top:
        if "logprob" in item:
            score_type = "logprob"
        else:
            score_type = "prob"
        score = item.get(score_type)
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("a reported token score was absent or nonfinite")
        scores.append({"id": item.get("id"), "token": item.get("token", item.get("tok_str")), score_type: score})
    return {"token_ids": tokens, "top_scores": scores}


def run_model(server, model, identity, env, timeout):
    port = free_port()
    alias = "rsn-smoke-" + uuid4().hex
    endpoint = f"http://127.0.0.1:{port}"
    args = [str(server), "--model", str(model), "--alias", alias,
            "--host", "127.0.0.1", "--port", str(port), *CPU_ARGS]
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    observed = {"model": identity, "status": "failed"}
    with tempfile.TemporaryFile() as log:
        worker = subprocess.Popen(
            args, cwd=server.parent, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            while time.monotonic() < deadline:
                if worker.poll() is not None:
                    raise ValueError(f"runtime exited during load with code {worker.returncode}")
                try:
                    health = request_json(opener, endpoint + "/health", 0.25)
                except (HTTPError, URLError, TimeoutError):
                    time.sleep(0.1)
                    continue
                if health.get("status") == "ok":
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("runtime did not become ready before the deadline")
            response = request_json(
                opener, endpoint + "/completion", max(0.01, deadline - time.monotonic()), REQUEST,
            )
            observed["forward"] = completion_evidence(response, alias)
            if identify(model) != identity:
                raise ValueError("checkpoint identity changed during runtime verification")
            observed["status"] = "runtime_verified"
        except (ValueError, OSError, AttributeError, TypeError) as error:
            observed["reason"] = str(error)
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=5)
        if observed["status"] == "failed":
            size = log.seek(0, os.SEEK_END)
            log.seek(max(0, size - 4096))
            observed["log_tail"] = log.read().decode("utf-8", errors="replace")
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True, help="path to llama-server executable")
    parser.add_argument("donor", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--no-op", type=Path, help="optional byte-identical no-op checkpoint")
    parser.add_argument("--max-model-bytes", type=int, default=DEFAULT_MODEL_LIMIT,
                        help="per-checkpoint file-byte limit (default 64 MiB)")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_TIMEOUT,
                        help="per-checkpoint runtime deadline (default 30 seconds)")
    args = parser.parse_args()
    try:
        if args.max_model_bytes < 1 or args.max_seconds < 1:
            raise ValueError("file and time limits must be positive")
        server = args.server.resolve(strict=True)
        models = {"donor": args.donor.resolve(strict=True),
                  "candidate": args.candidate.resolve(strict=True)}
        if args.no_op is not None:
            models["no_op"] = args.no_op.resolve(strict=True)
        for model in models.values():
            if not model.is_file() or not 0 < model.stat().st_size <= args.max_model_bytes:
                raise ValueError(f"{model.name} must be a regular checkpoint at most {args.max_model_bytes} bytes")
        identities = {role: identify(path) for role, path in models.items()}
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith("LLAMA_ARG_")}
        version = subprocess.run(
            [str(server), "--version"], cwd=server.parent, env=env,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.decode("utf-8", errors="replace").strip()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))

    evidence = {
        role: run_model(server, path, identities[role], env, args.max_seconds)
        for role, path in models.items()
    }
    passed = all(item["status"] == "runtime_verified" for item in evidence.values())
    no_op_equivalent = None
    if "no_op" in evidence:
        no_op_equivalent = (passed and identities["donor"]["sha256"] == identities["no_op"]["sha256"]
                            and evidence["donor"]["forward"] == evidence["no_op"]["forward"])
        passed = passed and no_op_equivalent
    print(json.dumps({
        "schema": "rsn.runtime-smoke.v1",
        "status": "runtime_verified" if passed else "failed",
        "measurement": "one CPU forward pass per checkpoint",
        "runtime": {**identify(server), "version": version, "arguments": CPU_ARGS},
        "limits": {"model_bytes": args.max_model_bytes, "seconds_per_model": args.max_seconds},
        "prompt_and_sampling": REQUEST,
        "checkpoints": evidence,
        "exact_no_op_equivalent": no_op_equivalent,
    }, indent=2, allow_nan=False))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
