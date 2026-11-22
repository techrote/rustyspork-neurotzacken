# Deterministic Greyhound / RustySpork Laboratory

**Engineering blueprint — no scheduler, backend, GPU pipeline or benchmarks implemented by this documentation commit.**

## Contents

1. [Purpose and threat model](#1-purpose-and-threat-model) — keep all expensive or destructive control outside language models
2. [Experiment graph](#2-experiment-graph) — frozen stages with explicit receipts and reusable outputs
3. [Manifest and data contracts](#3-manifest-and-data-contracts) — pin checkpoint, quantizer, masks, runtime and corpora
4. [Execution and recovery](#4-execution-and-recovery) — bounded jobs, resource reservations and deterministic resume
5. [Optional Nemotron adviser](#5-optional-nemotron-adviser) — proposal-only constrained intelligence
6. [Acceptance tests and minimum slice](#6-acceptance-tests-and-minimum-slice) — smallest honest end-to-end implementation

## 1. Purpose and threat model

The prior unattended worm schedules explored model behaviour but conflated experimental design, execution, evaluation, interpretation and continuation. A new controller must be a **deterministic, versioned experiment machine**. Its job is to prevent invalid baselines, missing pinned dependencies, accidental repeated large downloads, unbounded inference, silent routing changes, overwritten donors and cherry-picked results.

Initial platform: local Windows machine with **24 GiB Titan RTX**, CPU/RAM and disk staging. Optional secondary GPU and Linux host are future **measured** capabilities; do not schedule against imaginary resources. Do not allocate a nearly full Titan to an orchestration model during an LFM test.

**Controller never:**
- mutates a donor;
- infers a model's identity from a convenient short name;
- automatically downloads multi-GB weights merely because a PR landed;
- launches paid/external inference by default;
- lowers evaluation thresholds or changes hidden holdout tasks after seeing results;
- promotes a candidate from provisional to accepted without an explicit owner-reviewed result.

## 2. Experiment graph

A job is an immutable DAG with explicit dependencies, not an LLM conversation:

| Node | Input | Artifact / blocking gate |
| --- | --- | --- |
| 0. Inventory | Local path + approved HF locator | SHA-256, file size, tensor types/shapes, metadata, tokenizer/template; licence receipt |
| 1. Baseline | Frozen exact donor + runtime/decoding recipe | No-op inference, output traces, VRAM and timing; rejected if malformed |
| 2. Calibration | Corpus digest + source + trace adapter | Aggregated expert saliency, router-bias/coverage, confidence; no eval leakage |
| 3. Plan | Fixed candidate quota/method/seed | Exact survivor indices and source→destination remap; predicted bytes |
| 4. Dry run | Model topology + plan | Schema/routing compatibility; disk and RAM reservations; exact untouched tensors |
| 5. Surgery | Immutable donor + validated plan | Separate candidate, manifest, byte inventory, content hashes; atomic output |
| 6. Mechanical smoke | Candidate + patched/pinned runtime | Load, forward pass, top-k, cache reset/state correctness, NaN checks; failures retained |
| 7. Evaluation | Locked held-out tasks + matched baselines | Case-level outputs, metrics, context ladder, OOM data, latency/VRAM |
| 8. Decision | Predeclared gates + metric/uncertainty packet | GO / HOLD / NO-GO with reasons; no fabricated optimism |

The **same candidate** can be tested by multiple pinned runtimes, but results must not be mixed without labels. Histories are append-only; an invalid node stops its descendants and records why.

### Proposed CLI, not implemented

    rustyspork inventory --model <local-file> --output <receipt-dir>
    rustyspork calibrate --source <receipt> --corpus <frozen-corpus> --plan <config>
    rustyspork plan --saliency <receipt> --remove-per-layer 12 --method reap
    rustyspork dry-run --plan <plan> --source <receipt>
    rustyspork compact --plan <plan> --output <new-file>
    rustyspork smoke --candidate <receipt> --runtime <pinned-build>
    rustyspork evaluate --candidate <receipt> --suite <locked-suite>
    rustyspork report --run <run-id>

The names illustrate the command boundary; they are **not runnable commands**.

## 3. Manifest and data contracts

The controller must use canonical, deterministic serialization for run IDs and receipts. At minimum:

```json
{
  "schema_version": "rsn.experiment.v1-proposal",
  "run_id": "content-addressed-after-canonicalization",
  "source": {
    "hub_repo": "DavidAU/LFM2.5-24B-A2B-Turbo-Brilliance-50X-NEO-MAX-Imatrix-GGUF",
    "revision": "PIN_REQUIRED",
    "filename": "PIN_REQUIRED.gguf",
    "sha256": "PIN_REQUIRED",
    "file_bytes": null,
    "quant_inventory_sha256": "PIN_REQUIRED",
    "tokenizer_sha256": "PIN_REQUIRED",
    "chat_template_sha256": "PIN_REQUIRED",
    "license_id": "VERIFY_AND_PIN"
  },
  "environment": {
    "host_os": "Windows 11",
    "gpu": "Titan RTX",
    "vram_physical_gib": 24,
    "driver": "PIN_REQUIRED",
    "runtime_repo": "ggml-org/llama.cpp",
    "runtime_commit": "PIN_REQUIRED",
    "build_options": {},
    "kv_cache_k": "PIN_REQUIRED",
    "kv_cache_v": "PIN_REQUIRED",
    "context_tokens": 32768,
    "batch": 1,
    "gpu_offload": "PIN_REQUIRED"
  },
  "operation": {
    "family": "reap-structured-expert-prune",
    "method_version": "PIN_REQUIRED",
    "seed": 42,
    "expert_quota_per_moe_layer": 12,
    "survivor_manifest_sha256": "PIN_REQUIRED",
    "source_tensor_layout_sha256": "PIN_REQUIRED",
    "quant_policy": "preserve-when-block-aligned"
  },
  "corpora": {
    "calibration_sha256": "PIN_REQUIRED",
    "evaluation_sha256": "PIN_REQUIRED",
    "calibration_eval_overlap": 0
  },
  "limits": {
    "allow_network_download": false,
    "allow_gpu_surgery": false,
    "max_disk_bytes": null,
    "max_host_ram_bytes": null,
    "max_gpu_memory_mib": null,
    "max_wall_seconds": null
  }
}
```

**PIN_REQUIRED** is an invalid sentinel, not an automatic resolver. The real schema should reject absent hashes, unbounded resource limits and unauthorized network access; null example limits above must be filled before execution. Inputs, versions, numeric parameters and policies contribute to the run identity. Runtime command line and environment variables are preserved with secrets redacted.

Outputs append as separate records: \`inventory.json\`, \`calibration.json\`, \`plan.json\`, \`candidate.json\`, \`smoke.json\`, \`evaluation.jsonl\`, \`performance.jsonl\`, \`verdict.json\`. A proof packet includes donor hash and destination hash, measured file/mapped/GPU bytes and signatures or checksums of critical untouched tensors.

A transformation is not scientifically "the same" simply because its displayed quant label, expert count or seed matches. File exactness, ordering, runtime and sampling choices matter.

## 4. Execution and recovery

**State machine:** \`PLANNED → INVENTORIED → BASELINED → CALIBRATED → READY → RUNNING → MECHANICAL_PASS → EVALUATED → REVIEWED\`, with terminal \`FAILED\`, \`BLOCKED\`, \`CANCELLED\` and \`INCOMPLETE\` states. Transitions require validated output hashes. \`REVIEWED\` does not imply accepted/merged.

**Atomicity:** write candidates under a new content-addressed staging directory; fsync where applicable; validate size and hash; atomically promote to the candidate store. A crash must never replace the donor or make a partial file appear valid. Quarantine stale staging output and offer explicit resume only when input hashes match.

**Resource reservation:** preflight disk space for donor + scratch + complete candidate + logs, host RAM, peak GPU target and user desktop headroom. Prefer operating on one target model at a time. The local inference process and WebGPU viewer should normally be sequential until a co-residency measurement passes. Cache or reuse immutable inventory/calibration outputs only when all defining hashes match.

**Reproducibility:** pin seeds in Python/NumPy/PyTorch/CUDA/llama.cpp where supported, exact prompt templates and decoder settings; note that GPU reductions and kernels may remain nondeterministic. For stochastic samples, use repeated paired trials with confidence intervals rather than assuming one same-seed generation is bit-identical across altered graph shapes.

**Fast failure:** CLI dry run catches unsupported tensor rank/stride, quant block alignment, expert-id remap, missing bias path, non-topk-compatible expert counts, cache-state mismatch, insufficient disk space, unsupported architecture and untrusted metadata before any large export. Abort and record raw traceback/exit status; **do not** automatically retry destructive operations in a loop.

**Evidence durability:** Git holds small text receipts and review summaries. Local checkpoint store holds large binaries; result paths and sha256s remain in receipts. Do not rely exclusively on expiring CI artifacts or push copyrighted/unlicensed weight files to GitHub.

## 5. Optional Nemotron adviser

An NVIDIA Nemotron model, or another constrained model, may:
- review **anonymized/aggregated** metrics and failure clusters;
- generate candidate **proposals** within an operator-defined allowed set;
- explain uncertainty, recommend tests and cluster unusual mutations;
- draft a human-readable report citing exact receipt IDs.

It may **not**: access arbitrary paths, modify evaluation criteria, install dependencies, choose secret holdouts, initiate expensive inference, mark GO, or execute an unvalidated command. Every proposal is parsed as data, schema-validated and passed through dry-run and resource gates. The experiment grid, candidate IDs and stop conditions remain deterministic and auditable with or without the adviser.

Prefer hosted inference for large adviser models **only with explicit networking/privacy/cost approval**. If local, it runs on spare CPU/GPU resources or in a separate time slot. No inference provider availability or rate is assumed.

## 6. Acceptance tests and minimum slice

**First implementable slice (no large weights required):**

1. Deterministic inventory for a tiny synthetic GGUF-like fixture and one legal real small model.
2. Content-addressed run directory; explicit no-download, max-memory, max-disk and max-duration enforcement.
3. Expert survivor manifest \`old_id → new_id\`; duplicate/out-of-range/empty-selection rejection; stable deterministic ordering.
4. No-op surgery: unchanged tensor values/allowed metadata, provable source hash preservation and functional equivalence.
5. Deliberately invalid mutant: correctly marked BLOCKED/FAILED without partial promotion.
6. Resume an interrupted job without rerunning already validated stages; archived negative trial.
7. Compare pinned baseline and mutant case-level task traces, with a real zero-result/failed-result pathway.
8. Generate viewer-compatible, exact tensor selection IDs without relying on a GPU renderer.

**Only after these pass** should the system attempt complete LFM Q6 compaction or expensive long-context tests. This order prevents the earlier unattended-work problems from resurfacing at 24B scale.

See [compactor design](LFM_EXACT_Q6_COMPACTOR.md), [FRANKENFIELD contract](FRANKENFIELD_IMPLEMENTATION_CONTRACT.md), [qualification](QUALIFICATION_GATES.md) and [reconciliation](RECONCILIATION_AND_DECISIONS.md).
