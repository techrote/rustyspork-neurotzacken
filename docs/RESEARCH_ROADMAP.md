# Investigation map — RustySpork / FRANKENFIELD

**Recorded:** 2026-11-22. **Status:** hypotheses and design candidates; not implementation results.

## 1. Programme intent

Develop a deliberately experimental but scientifically auditable way to inspect, perturb, structurally prune, splice and evaluate open-weight LLMs from roughly 4B to 40B. Complement this with a visually impressive, navigable 3D neural anatomy GUI that never confuses artistic representation with literal network geometry. Keep weights, runtime control flow, routing statistics, features and displayed graphics conceptually separate.

Immediate practical challenge: reduce the local VRAM footprint of [DavidAU LFM2.5-24B-A2B Turbo-Brilliance Q6](MODEL_SPECIMENS.md) by several GB to afford larger KV-cache allocations and ordinary desktop activity, without sacrificing its strong responsiveness. [KAT-Coder-V2.5 REAP-50 Q6](MODEL_SPECIMENS.md) serves as a published precedent and comparison, not proof that equivalent pruning quality transfers to LFM.

## 2. Parallel research tracks

| ID | Investigation | Why it matters | First bounded question | Gate |
| --- | --- | --- | --- | --- |
| A | **FRANKENFIELD multiresolution atlas** | Inspect 4–40B weights with bounded GPU memory | Can one map real tensor addresses to deterministic coarse/fine atlas tiles and select exact indices? | Golden tensor subset matches selected indices bit-for-bit |
| B | **REAP expert pruning for LFM2.5-24B** | Recover ~2–5 GB of Q6 storage | Which 8/12/16 experts per 64, per layer, can be removed with tolerable quality loss? | Paired accuracy and strict memory accounting |
| C | **Quantisation-aware GGUF compaction** | Preserve the exact Turbo-Brilliance derivative | Can compatible expert slices be copied/repacked without a full BF16 reconstitution? | Runtime loads and untouched tensor hashes or value-equivalence proven |
| D | **Mixed-precision importance maps** | Potentially save memory without structural deletion | Are lower-saliency expert matrices safe at lower bit width with all experts retained? | Quality/VRAM compared to same-size pruning |
| E | **Layer-graph surgery: Greyhound / Branchwitch** | Revisit historical layer skips, repeats and rearrangement | Are such runtime graphs valid and performant for a chosen modern model architecture? | Distinguish valid output from useful output and state-cache integrity |
| F | **Rusty Spoon tensor interventions** | Explore quirky behaviours and circuit sensitivity | Can masked scoop, scrape, noise, clipping and zeroing map from GUI to exact tensor deltas? | Reversible patch and identical no-op inference |
| G | **Reblitteration / Funnel** | Recruit an already useful detector on appropriate inputs | Can upstream low-rank edits improve conditional activation without changing the detector? | Follow [RSN-001](https://github.com/techrote/rustyspork-neurotzacken/issues/1); causal controls |
| H | **Grafting / stitching** | Test cross-checkpoint modularity | Can shape-compatible, tokenizer-consistent tensors or low-rank regions be transplanted safely? | Matched compatibility audit and held-out evaluations |
| I | **Activation/circuit overlays** | Make atlas scientifically more useful than a weight heatmap | Can measured saliency or causal contributions be overlaid and identified separately from weight statistics? | Provenance and uncertainty available per overlay |
| J | **Behavioural vivisection** | Connect edits to their consequences | Can the same prompt/task corpus compare untouched, placebo and edited variants side by side? | Paired logs, drift measures and retained failures |

Paths B, C, D and E must not be conflated: pruning experts changes parameter shapes and routers; graph surgery changes executed blocks; quantisation changes representation precision; weight patching modifies numeric content.

## 3. Proposed sequence (not committed milestones)

**P0 — frozen specimens & cost ceilings.** Record exact Hub repo, revision/commit, GGUF filename, byte size, SHA-256, architecture metadata, tokenizer and chat template, active runtime version/build, GPU driver, context and cache type. Keep originals read-only. Establish baseline workloads and VRAM readings, including desktop/video competing allocations and OOM outcomes. Verify models' licences and downstream redistribution terms.

**P1 — read-only atlas PoC.** Start with synthetic tensors and a real ~1–4B model before whole-model 24B scans. Implement a tile-reduced tensor index and fixed hierarchical levels. Visualise mean, mean absolute magnitude, variance, extremes, zero fraction, outliers, sign ratios, quantisation errors and selected activation statistics. Clicking a cell must resolve to a verifiable tensor/index set.

**P2 — REAP saliency intake.** For LFM2.5, use the existing LFM adapter in [reap-cuda](https://github.com/egesabanci/reap-cuda), then solve final-stage memory constraints (layerwise calibration alone does not eliminate the whole-model reload). Preserve per-layer expert rankings, router statistics, dataset digest and saliency uncertainty. Compare conservative 8, 12 and 16 experts removed out of 64, with 20/64 as conditional exploratory arm. Evaluate nonuniform per-layer quotas in a later phase, after fixed-quota proof.

**P3 — first real compact checkpoint.** Produce an executable derivative only after confirming correct weight packing, expert ordering, router index remapping, expert-bias slicing, quant metadata and llama.cpp runtime support. Maintain a full BF16→prune→GGUF path as scientific control when hardware/host resources permit. A 24GB GPU alone does not hold a 24B BF16 model.

**P4 — controlled edits in the UI.** Implement read-only selection first; preview patch → validate target and budget → apply to isolated overlay → export new derivative → paired smoke/evaluation. Then enable weight scoop, local mask, noise and expert prune views; add graph surgery and grafts only with architecture-specific runtime adapters.

**P5 — expansion.** Test cross-model comparison, out-of-core checkpoints, additional 4–40B architectures, activation/circuit overlays, and reblitteration integration once RSN-001 earns a positive independent verdict.

## 4. Owner-observed priorities, not lab results

- KAT REAP-50 Q6 perceived impressively capable; owner can allocate ~250K context on their machine.
- LFM2.5-24B-A2B Turbo-Brilliance Q6 perceived extremely fast; owner reports ~150K allocation leaves too little VRAM for incidental desktop activity.
- Saving ~3–5 GB from LFM Q6 is the goal. These are configuration/perception reports, not independently benchmarked or guarantees of fidelity at over-native context.

## 5. Non-negotiable research controls

- Donors immutable; every operation logged as a versioned declarative edit with exact parameter/index mask, algorithm, seed, numeric precision and code/runtime hashes. Preview is not applied surgery.
- Do not automatically launch costly model downloads, GPU jobs or scheduled inference on normal code pushes.
- Keep actual model edits separate from rendering: shader noise, glows and visual filaments do not become claims about real synapses or causal influence.
- Causal recruitment tests must include appropriate non-engagement and false-positive penalties. A frozen feature detector is not a proof of unchanged activations.
- Record invalid loading, output collapse, looping, quality regressions, speed regressions, OOM and negative findings.
- Do not merge experiments into CyberSand, alter DRIV3R historical artifacts or assume XenoField shaders can simply be copied without checking ownership/licence and dependencies.

## 6. Outstanding decisions

- Exact installed model files/quant details and local baseline receipts are still needed from a real run.
- Whether the exact Turbo-Brilliance checkpoint has a corresponding recoverable full-precision donor is unresolved; the ordinary LiquidAI BF16 donor is not necessarily source-exact.
- REAP saliency corpus: coding/tool-use and real long-context workloads, not only short generic prompts.
- Fine-grained importance/quantisation: what runtime can actually load mixed GGUF types and reduced LFM expert counts?
- Is local WebGPU rendering compatible with concurrent 24GB model inference without increasing OOM risk? Choose strict graphics budget and fallback.
- Are graph edits even semantically valid across architectures with convolution/recurrent state? Gate per runtime.
- What retained-quality, long-context and memory thresholds constitute a useful winner? Define before evaluating candidates.

See [model dossier](MODEL_SPECIMENS.md), [REAP workplan](REAP_AND_QUANTIZATION.md), [UI specification](FRANKENFIELD_INTERFACE.md), [experiment protocol](EXPERIMENT_PROTOCOL.md), and [references](REFERENCES.md).

## 7. Reconciled work packages and review evidence (22 November 2026)

The owner has now **selected** the KAT REAP-50 Q6 reference and **prioritized** the exact DavidAU LFM Turbo-Brilliance Q6 3–5 GB reduction. Earlier 4–40B paths remain relevant research, but **no other specimen takes precedence**. Replace unattended conversational worm scheduling with a [bounded deterministic laboratory](DETERMINISTIC_LAB.md). An optional NVIDIA adviser may suggest experiments only, never own the run or verdict.

| Package | Deliverable before expensive inference | Release restriction |
| --- | --- | --- |
| RSN-LAB-01 | Frozen-source inventory + no-op fixture + CPU/disk limit enforcement | No live donor mutation |
| RSN-LFM-01 | Exact Q6 tensor layout, router/bias remap, source-exact compactor feasibility report | BLOCK if GGUF round-trip or loader fails |
| RSN-LFM-02 | Expert saliency with mode/coding/tool/long-tail coverage and source hashes | Keep holdout entirely separate |
| RSN-LFM-03 | 8/12/16 expert reduction candidate receipts and matched quality/VRAM assessment | No automatic merge or model promotion |
| RSN-ATLAS-01 | Exact picking/selection and signed statistical atlas on synthetic/small source | WebGPU viewer is not an inference engine |
| RSN-001 | Existing reblitteration causal/upstream PoC | Independent Draft/HOLD issue |

The [reconciliation](RECONCILIATION_AND_DECISIONS.md), [exact-Q6 engineering](LFM_EXACT_Q6_COMPACTOR.md), [FRANKENFIELD contract](FRANKENFIELD_IMPLEMENTATION_CONTRACT.md), [numeric surgery contracts](SURGERY_OPERATIONS.md) and [qualification gates](QUALIFICATION_GATES.md) define these packages. A second GPU and disk clearance permit the **deferred** eight-bit KAT project only after a new source/baseline receipt.
