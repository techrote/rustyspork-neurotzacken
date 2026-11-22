# Experimental design — reproducible neural surgery trials

**Protocol sketch, 22 November 2026. Not an executed trial or a completed test harness.** This governs the future GGUF pruning/FRANKENFIELD experiments; [RSN-001](https://github.com/techrote/rustyspork-neurotzacken/issues/1) retains its separate causal reblitteration protocol.

## 1. Research questions

- Can calibrated expert pruning or mixed precision recover **3–5 GB or more of real loaded weight memory** from the exact LFM2.5-24B-A2B Turbo-Brilliance Q6 specimen?
- Which tradeoffs appear in task quality, long-context reliability, perplexity, tool use, output style, throughput, OOM rates and model-specific prompt modes?
- Does removing expert weight storage translate into enough *actual device headroom* for 250K cache allocation and ordinary desktop use? Can useful retrieval/reasoning be demonstrated above the model's 131,072 published native context?
- Can the 3D atlas select/modify precise tensor entries and show real, reproducible effects without confusing shader effects with scientific evidence?
- How do graph surgery (Greyhound/Branchwitch), scalar weight perturbations and frozen-feature upstream recruitment differ from structural expert removal?

## 2. Baselines and comparison arms

| Arm | Purpose | Status |
| --- | --- | --- |
| LFM exact Q6 donor, unmodified | Only valid primary stock reference for the local target | Owner-used; this repo has no baseline logs |
| KAT-Coder-V2.5 REAP-50 Q6 | Existing successfully pruned external reference, separate architecture | Published; not a head-to-head quality baseline for LFM |
| LFM random expert deletion (fixed seed) | Negative control for saliency value | Proposed |
| LFM simple router-frequency pruning | Baseline to compare with REAP ranking | Proposed |
| LFM REAP remove 8/64 experts/layer | Conservative ~2.35 GB estimate | Proposed |
| LFM REAP remove 12/64 experts/layer | ~3.53 GB target | Proposed |
| LFM REAP remove 16/64 experts/layer | Primary ~4.71 GB target | Proposed |
| LFM REAP remove 20/64 experts/layer | Conditional ~5.88 GB stress arm | Proposed |
| LFM mixed precision without pruning | Equal-size compression control | Proposed |
| LFM routing suppression without storage reduction | Behaviour-only diagnostic: not a memory win | Proposed |
| Official parent BF16→prune→GGUF | Pruning methodology comparison; different checkpoint risk | Proposed, **not source-exact** without proof |
| LFM quantised/no-prune with lower KV cache precision | Practical low-risk capacity control | Proposed |
| Structure skip/repeat mutants | Separate runtime-surgery research | Proposed only after model-specific graph safety proof |

**No results for proposed LFM arms exist in this repository as of this note.** Match exact quant choice (regular NEO vs MAX NEO) and runtime before attributing any behavioural difference to surgery.

## 3. Source custody and frozen evidence

Every run has:
- Source Hub repo/revision, file/digest inventory, loaded model/template/tokenizer metadata, engine build commit and patched graph (if any).
- Host OS, CUDA/NVIDIA driver and GPU, browser/graphics coexistence, available/total VRAM, CPU RAM, process/GPU memory peaks, cache type, context allocation, number of slots and batch size.
- Frozen calibration and evaluation datasets with SHA-256, item lists, classification/subject mix and random seed; evaluation is never chosen by inspecting candidates.
- Precise algorithm config (REAP score, expert quota, ranking, masks, bias and router rewrite, quantisation, output tensor exception, sampling config), baseline file hashes and derivative hashes.
- Prompt + original/decoded token counts, logprobs if available, tool-call validity, completion status, errors, latency (p50/p95), input/output token rates and benchmark case-level outcomes.
- Results archived even for load failures, malformed outputs, immediate STOP, all-branch bias, looping, invalid attention/conv states and OOM.

Never modify the stock file or overwrite previous mutated checkpoints. Run reproducibly with independent initial-world or prompt seeds as relevant. Keep equivalent decoding settings when comparing throughput/quality; stochastic comparisons require repeated sampling and uncertainty rather than one anecdote.

## 4. Calibration vs evaluation

**Calibration:** contain representative coding, instruction following, JSON/tool calling, multilingual, domain-specific long-tail prompts, short/long contexts and Turbo-Brilliance mode prompts if those are relevant in local use. Log per-layer expert coverage and underrepresented experts. Avoid implicitly biasing against experts that are rare but useful.

**Evaluation:** disjoint prompts/tasks; independently established before inspecting results. Candidate categories:
- Code: small deterministic programming tasks with unit tests; optional established HumanEval+/MBPP+ harness if properly pinned.
- Tools: schema fidelity, valid arguments, multi-step tool selection and recovery, BFCL-like tasks when harness is available.
- General reasoning: exact factual questions, compositional reasoning, math, hard negatives; style cannot substitute for correctness.
- Instruction following: constraints, formatting, abstention and long multi-part instructions.
- Modes/character: stock vs mutant responses under documented Turbo-Brilliance mode activation and neutral prompts.
- Long context: needle-in-a-haystack *plus* multi-needle recall, cross-document synthesis and distractors, with exact token counts and data crossing context windows; same truncation policy.
- Degeneration: repetition, premature stop, answer loops, mode leakage, refusal change and unexpected special tokens.

Compare both per-task metrics and statistically supported aggregate outcomes; no single benchmark, throughput or qualitative first impression proves retention.

## 5. Context ladder and occupancy measurement

Test **model's published 131,072 token limit** as a distinct reliability boundary. Proposed allocation ladder: 8K, 32K, 64K, 128K (at/near official), then **150K and 250K are explicitly experimental extrapolation**. For every level record successful allocation, true prompt tokens, first-token latency, throughput, KV type, retained quality and observed errors. Separate “server accepted context allocation” from “model genuinely used that faraway information correctly.”

For VRAM:
1. Idle baseline with desktop and viewer separately; identify process-level model allocation and shared compositor/driver demand.
2. Source/candidate at a fixed quant and same context/KV settings: cold load, warmed inference, long prefill peak, decoding peak, cleanup/reload. Include fragmentation and memory freed after exit.
3. Realistic co-residency: Windows desktop, Firefox active, optionally a streaming video tab; record actual OOM probability over repeat trials. Do not intentionally crash unsaved work.
4. Same maximum-safe-load experiment for candidates; ensure any savings were actually available to the KV cache and not consumed by runtime staging/buffers.
5. Repeat with optional lightweight FRANKENFIELD inspector open, with measured explicit graphics memory budget.

Derive max safe context *empirically* and report native/experimental extrapolation separately. The idealised FP16 attention KV calculation is in [MODEL_SPECIMENS.md](MODEL_SPECIMENS.md).

## 6. Quality and selection gates (provisional)

**Mechanical pass required before behavioural testing:** new model loads through the target llama.cpp build, no invalid tensor names/sizes or nonfinite values, router selects only surviving expert IDs, correct top-k, exact smoke-test prompt completion and no-op equivalence. Track unmodified tensor hash/value checks and recorded exceptions.

**Scientific comparison:** every candidate paired with the same frozen stock tasks and engine/hardware settings. Publish absolute scores, paired deltas, per-category regressions, uncertainty and negative cases. Set numeric retention thresholds before selection, rather than fitting success criteria after seeing performance.

**Practical promotion:** file AND loaded-memory savings measured, safe operation with desired context under realistic graphics co-residency, no unacceptable coding/tool-use regressions, and no evidence of severe looping/collapse. The 12/64 vs 16/64 winner depends on measured tradeoffs. Do not announce 250K usable simply because it fits.

**Reblitteration special gate:** frozen target tensor verification, causal feature proof, upstream-only permitted delta masks, increased *appropriate* recruitment with false-positive penalty, no harmful unrelated drift; follow [issue #1](https://github.com/techrote/rustyspork-neurotzacken/issues/1).

## 7. FRANKENFIELD validation, separate from inference quality

- Golden small tensors test every tile summary, exact coarse/fine coverage, picking/mask projection and edit accounting.
- Percentages/colour legend checked against known signed/outlier distributions; preserve zeros and opposite-sign cancellation rather than only averages.
- Validate zero/no-op operation produces an unchanged file/function result (allowing for explicitly documented non-byte-identical serializer metadata).
- Brush/sphere/box/threshold selection exactly maps to tensor coordinates, including edge and partially visible tiles; 3D/world coordinates are not claimed to be causal.
- Editing large models must not lock the browser, freeze during disk scans, or allocate full-model buffers in WebGPU.
- Separate GPU headroom budgets and fail gracefully on device loss; prove viewer and inference can be run sequentially or together within measured budget.
- Regression tests verify every edit is reversible via preserved donor + patch manifest, rather than unsafe in-place undo.

## 8. Evidence packet schema (proposal)

Suggested per-run directory:
- manifest.json: source, exact revisions/files/checksums, runtime/environment, operation and expected footprints
- calibration/: corpus receipt and expert-rank measurements
- baseline/: pinned stock task outcomes and raw replies
- candidate/: operation patches, transformed-checkpoint receipt, actual size and GGUF inventory
- eval/: case-level JSONL, aggregate CSV/JSON, uncertainty and task verdict
- performance/: prompt/decode throughput, VRAM snapshots, OOM and context outcomes
- notes.md: observations, failed cases, caveats and a GO/HOLD/NO-GO statement

No model binaries or unlicensed redistributions should be uploaded to git by default. GitHub Actions artifacts alone are temporary, so durable result receipts need a controlled repository policy.

## 9. Relation to historical DRIV3R experiments

The [D1 Frankenlayers record](https://github.com/techrote/DRIV3R/blob/main/experiments/driv3r/frankenlayers/results-2026-11-15.md) reports real pinned inference on a **different 2B decision graph**: GREYHOUND 13/24 layers with paired ~1.91× p50 speedup and 45/48 BRANCH choices; BRANCHWITCH 20 execution steps (layer 10 twice), paired ~1.17× speedup and 42/48 BRANCH; SCRAMBLED EGGS immediately STOPped in three initial families. Those were same-seed W0 trials with known generalisation limitations; they must not be read as evidence of expected 24B utility.

Reimplement graph experiments as separate architecture-specific controlled tests and preserve failure attractors as first-class outcomes.

See [research programme](RESEARCH_ROADMAP.md), [model analysis](MODEL_SPECIMENS.md), [REAP/GGUF notes](REAP_AND_QUANTIZATION.md), [interface](FRANKENFIELD_INTERFACE.md), and [references](REFERENCES.md).

## 10. Reconciled acceptance gates and deferrals

The [qualification gates](QUALIFICATION_GATES.md) turn this protocol into a staged G0–G6 acceptance plan with a Windows/Firefox co-residency matrix, exact file/VRAM measurement definitions, context-tier evidence, failure taxonomy and owner-held GO/HOLD/NO-GO decisions. Their proposed numeric screening thresholds are **not** retrospectively owner-approved facts; register policy before seeing candidate outputs.

**Priority order:** KAT-REAP Q6 as reference; DavidAU LFM Turbo-Brilliance Q6 as primary 3–5 GB candidate; defer KAT REAP eight-bit further pruning until disk space and a second GPU are in place. The public REAP GGUF lists **Q8_0**, not a verified Q8_K. Native LFM 131,072 remains distinct from allocated 150K/250K extrapolation, and raw local Q6 receipts are not yet archived here.

The [exact-Q6 exporter investigation](LFM_EXACT_Q6_COMPACTOR.md) and [deterministic controller](DETERMINISTIC_LAB.md) require source-equivalence/quant-block/no-op tests before expensive LFM mutant evaluation.
