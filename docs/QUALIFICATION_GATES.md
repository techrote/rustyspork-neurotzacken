# Model-surgery qualification matrix, acceptance gates and owner runbook

**Proposed acceptance contract for future implementation; no large-model trials performed by this document.** The user reports a locally impressive KAT-Coder REAP-50 Q6 and a fast but VRAM-tight LFM Turbo-Brilliance Q6. The raw local test data has **not** been imported into this repository; maintain that distinction even if a model has already been manually judged useful.

## 1. Objectives and specimens

| Role | Exact family | Status / constraint |
| --- | --- | --- |
| Reference | **KAT-Coder-V2.5-Dev REAP-50 Q6_K**, published by Ttimms, ~15.3 GB | Owner-selected local reference; known already-pruned topology; not a proxy score for different LFM tasks |
| Primary | **LFM2.5-24B-A2B Turbo-Brilliance-50X NEO/ MAX-NEO Imatrix Q6_K**, DavidAU | Preserve fast decode and specific style/modes; recover **3–5 GB** of true model memory |
| Negative controls | Exact donor, zero-removal round trip, random/frequency pruning, lower-precision/no-prune options | Pair against identical tasks/runtime |
| Deferred | **KAT REAP-50 eight-bit** (published **Q8_0**; locally mentioned Q8_K unverified) | Only after additional HDD space and secondary GPU; new baseline and source receipt required |

The target GPU is a **24 GiB Titan RTX** under Windows with Firefox active. Its nominal capacity is not necessarily free VRAM; desktop compositor, video, model staging, graph buffers and selected KV type count.

## 2. Qualification phases

| Phase | Purpose | Entry/exit |
| --- | --- | --- |
| **G0: Source freeze** | Complete donor exact-revision/sha, tokenizer, template, quant tensor inventory, runtime build and legal/source receipt | Missing exact MAX/regular Q6 file identity = HOLD |
| **G1: Format proof** | Tiny synthetic fixture, no-op GGUF round trip, expert-axis/block proof, router/bias remap and target-runtime loader | Any silent value corruption or unsupported graph = NO-GO |
| **G2: Baseline receipt** | Stock LFM performance & quality under normal and constrained contexts; reference KAT receipt separately | Without case-level baseline metrics = HOLD |
| **G3: Expert calibration** | Disjoint, representative activation/router scoring with confidence and long-tail coverage | Missing mode/tool/coding or bias coverage = HOLD |
| **G4: Candidate smoke** | Valid loader, positive/finite outputs, correct top-4 routing, cache/prefill/decode, tokenizer/modes | Any mechanical failure = NO-GO |
| **G5: Paired pilot** | Brief fixed coding/tool/instruction/degeneration battery | No severe regression enables full eval, never automatic GO |
| **G6: Full evaluation** | Held-out quality, robust real-world VRAM and speed, long-context ladder | Publish GO/HOLD/NO-GO with uncertainty; human promotion only |

A candidate that fails G4 is never sent into expensive benchmarks. A file that shrinks but does not release **resident** VRAM is a research result, not a delivered capacity win.

## 3. Tests and measurement definitions

**Mechanical fixtures:** zero removal; remove first/last/middle expert; bad tensor strides; quant block boundary across selected expert; missing expert bias; duplicate/out-of-range IDs; top-k > survivors; count metadata mismatch; payload/trailer corruption; unsupported loader; and model cache-reset correctness. Each deliberately invalid fixture must be rejected. All unaffected tensor hashes or dequantized-equivalence receipts are mandatory.

**Paired quality battery:** freeze an evaluation manifest *before candidate results*. Include deterministic coding tasks with executable unit tests, multi-step structured tools/function arguments, constraint/instruction adherence, math/reasoning, multilingual and long-tail probes, and **Turbo-Brilliance mode prompts vs neutral prompts**. Record case-level correct/incorrect, tool schema failures, format drift, repetition, early stop and special-token leakage. Include adversarial and low-frequency domain tasks where pruning rare experts could matter. Use fixed decoding for a deterministic pilot and repeated seeded draws for stochastic use.

**Suggested operating sizes** (research proposals, not approved success criteria): 30–50 pilot prompts stratified across use categories; ~200+ held-out cases for a first full comparison, supplemented by task-specific suites and confidence intervals. Never tune expert ranking on the same evaluation tasks. Freeze the corpus SHA, harness version, scoring rubric, test-case IDs and exposure of any hidden holdout.

**Metrics:** exact file bytes (GB decimal), mapped/resident model weights (MiB/GiB), whole CUDA process peak and desktop baseline (MiB/GiB), CPU RAM, disk staging, time to load, prompt prefill tok/s, decode tok/s (p50/p95), time-to-first-token, quality per category, utility per **measured** model GiB, and OOM incidence at each context/batch/cache configuration. Report quantization errors and output differences separately from pruning errors when possible.

**Windows occupancy scenarios:**
1. Inference only (desktop compositor background).
2. Firefox normal browsing.
3. Firefox streaming video or similarly memory-intensive workload.
4. Viewer open only (no inference).
5. Viewer + inference **only after measured resource budget passes**; refuse if unsafe.

Take actual readings, not merely \`24 - file_size\`; record every loaded tensor offloaded to CPU before calling the remaining GPU space a win. A saving achieved solely by making CPU inference slower is not equivalent to the user's goal.

## 4. Context ladder and interpretation

Official LFM context: **131,072 tokens**. Compare stock and mutant at **8K, 32K, 64K and 128K**, then **150K and 250K only as experimental extrapolation**. At each level measure (i) accepted allocation, (ii) actual input tokens, (iii) memory peak, (iv) performance, (v) multi-needle retrieval and cross-document synthesis with known answers, (vi) long-range distraction errors, and (vii) context-dependent output changes.

Keep FP16 and quantized KV-cache runs separate. Hybrid convolution/recurrent state may have non-KV allocations that vary with implementation. Distinguish OOM at *model load* from OOM during prefill or decode. A runtime setting that accepts 250K without crashing **does not verify** 250K quality.

For the published KAT REAP reference, a user-reported ~250K local context should be documented as a **user-confirmed experience pending its archived measurement receipt**; do not substitute an external KAT NVFP4/Q4 benchmark for local Q6 quality.

## 5. Decision gates and failure taxonomy

**Hard invariant gates (not negotiable):**
- Donor immutability and SHA preservation.
- Exact donor/decoder/template/runtime identity and stable candidate identification.
- No malformed GGUF/tensor/router/bias or nonfinite inference.
- No false reporting of runtime bypass as disk reduction.
- No source-vs-candidate quality comparison across unlabelled quantization changes.
- No hidden or revised holdout tasks after seeing candidate outcomes.
- Archive partial/failed/negative experiment data.

**Quantitative proposal — must be ratified before testing:** measurable **≥3 GB** real loaded-model memory recovered as a minimum useful LFM outcome, with **~5 GB** as stretch; median decode/prefill throughput no worse than a predefined fraction of paired baseline (for example ≥90% as an initial screening threshold), and no material degradation on critical tool-use/coding categories. For quality, pre-register category-specific allowed paired deltas and confidence bounds; **do not declare a 3-percentage-point change acceptable solely because an aggregate average looks good**. Rare-task and mode-specific regressions can veto even a strong aggregate. These example thresholds are not owner-authorized promotion policy.

Each report must separate:
- **GO (research candidate):** mechanical, quality and real-memory tests passed under a frozen policy. Still Draft/HOLD until the owner accepts the candidate.
- **HOLD:** missing evidence, low statistical power, ambiguous checkpoint provenance, runtime unsupported on the intended desktop, or unresolved memory/quality tradeoff.
- **NO-GO:** corrupt/incompatible export, strong behavioural collapse, severe regressions, privacy/licence violation, or unacceptable operational instability.

**Failure classes:** FORMAT, ROUTER, NUMERIC, GRAPH_STATE, SOURCE_MISMATCH, CALIBRATION_BIAS, QUALITY, MODES, CONTEXT, VRAM_OOM_LOAD, VRAM_OOM_PREFILL, VRAM_OOM_DECODE, THROUGHPUT, VIEWER_RESOURCE, EVIDENCE_MISSING. Record exact triggering cases and executable reproducer where legally possible.

## 6. Execution runbook (future)

1. Choose exactly one local GGUF candidate and pin its **actual file SHA and named quant**. Snapshot donor, tokenizer/template, runtime commit, Windows/NVIDIA driver, device, host RAM, output directory and hard resource limits.
2. Verify enough storage for donor + stage + candidate + evidence, and ensure the Titan is clear before a large model test. Never start the deferred KAT eight-bit programme implicitly.
3. Run G0/G1/G2 baseline. Establish stock's supported native-context and extrapolated-context reports independently; keep source and candidate runtime flags matched.
4. Prepare calibration corpus with labeled categories and **separate sealed evaluation data**. Run bounded saliency observation; archive expert-frequency and activation distributions, coverage and uncertainty.
5. Create plans for **8, 12 and 16 experts removed per layer**; 20/64 only if useful and feasible. Include seeded random/frequency controls, same-size mixed-precision control, and a no-op zero-removal arm.
6. Perform dry-run on target GGUF and runtime compatibility proof. Export one model at a time to a new staged path. Abort on first mechanical mismatch and retain evidence.
7. Run smoke and pilot. Only if valid and useful, run full paired quality, realistic desktop memory and context ladder. Record all failures, false positives, raw outputs and precise throughput/VRAM telemetry.
8. Publish a results bundle with SHA-256 hashes, case-level outcomes, uncertainty, actual model-bytes saved, real device capacity recovered and clear verdict. Keep source model and every mutant independently addressable; do not merge/promote without owner review.

**No work in this document asserts that any of the steps above has already run.** See [experiment protocol](EXPERIMENT_PROTOCOL.md) for detailed categories, [compactor](LFM_EXACT_Q6_COMPACTOR.md), [deterministic controller](DETERMINISTIC_LAB.md), [FRANKENFIELD](FRANKENFIELD_IMPLEMENTATION_CONTRACT.md).
