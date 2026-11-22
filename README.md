# RustySpork Neurotzacken — neural surgery research

A standalone research home for **reversible LLM weight/graph surgery**, expert pruning, conditional circuit recruitment (**reblitteration**), and the prospective **FRANKENFIELD** WebGPU/WebGL neural atlas and surgery workstation.

**Status (22 November 2026): research programme and source index, not an implemented surgery application.** The existing [RSN-001](https://github.com/techrote/rustyspork-neurotzacken/issues/1) defines the first reblitteration proof of concept. Documentation here preserves hypotheses, target models, mathematical estimates, experiment gates and links for future work. Nothing has been pruned, exported, benchmarked or validated in this repository by these notes.

## Read this first

| Document | Purpose |
| --- | --- |
| [Decisions and reconciliation](docs/RECONCILIATION_AND_DECISIONS.md) | Reconstructed design decisions, resolved conflicts, accepted specimens and scope boundaries |
| [Deterministic laboratory](docs/DETERMINISTIC_LAB.md) | Reproducible execution DAG, state machine, evidence schemas, resource gates and optional Nemotron adviser |
| [LFM exact-Q6 compactor](docs/LFM_EXACT_Q6_COMPACTOR.md) | Technical engineering plan for saliency on exact Q6, quantized expert packing and safe structural export |
| [FRANKENFIELD implementation contract](docs/FRANKENFIELD_IMPLEMENTATION_CONTRACT.md) | UI wireframe, exact tensor picking and selection schemas, bounded WebGPU rendering and golden tests |
| [Surgery operation contracts](docs/SURGERY_OPERATIONS.md) | Scoop, scrape, gouge, REAP, Greyhound, Branchwitch, grafting and Funnel semantics |
| [Qualification gates](docs/QUALIFICATION_GATES.md) | Actual memory/quality/speed/long-context acceptance matrix and execution runbook |
| [Research roadmap](docs/RESEARCH_ROADMAP.md) | Candidate investigation paths, priorities, phases, boundaries and unresolved questions |
| [Model specimens and memory](docs/MODEL_SPECIMENS.md) | KAT REAP-50 Q6 and LFM2.5-24B Turbo-Brilliance Q6; exact identities, pruning arithmetic, KV/cache and context caveats |
| [REAP, quantisation and compaction](docs/REAP_AND_QUANTIZATION.md) | Expert saliency, quantisation controls, GGUF-preserving and BF16 workflows, 24 GB VRAM constraints |
| [FRANKENFIELD interface](docs/FRANKENFIELD_INTERFACE.md) | GPU multiresolution atlas, honest 3D coordinate mapping, interactive surgery tools, XenoField reuse boundaries |
| [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) | Frozen source and evaluation plan, negative controls, headroom/OOM measurements, acceptance gates |
| [Reference index](docs/REFERENCES.md) | Model cards, upstream code, renderer architecture, historical DRIV3R results, research papers |

## Guiding principles

- **Inspect cheaply, edit exactly.** Render statistical summaries of tensors rather than billions of individual points; all selected regions must resolve deterministically to exact model/tensor/parameter addresses.
- **Never overwrite a donor checkpoint.** Pin source revision, file hashes, tokenizer, chat template, runtime and quantisation; use immutable donors and reversible patches or independently produced derivatives.
- **Keep structures distinct.** MoE expert *removal*, runtime layer *skipping/duplication*, tensor *perturbation*, and upstream-feature *recruitment* are different interventions with different risks.
- **Make performance claims measurable.** Retain failed trials and paired stock/mutant evidence; compare quality, VRAM, decode/prefill, context recall and behavioural drift.
- **4–40B is the engineering horizon.** A small 1–4B causal test for reblitteration and a bounded 4–40B metadata/atlas prototype can precede expensive full-model transformations.
- **Separation of ownership.** This repository owns surgery research and any later workstation; [DRIV3R](https://github.com/techrote/DRIV3R) owns its historical worm-policy experiments; [XenoField](https://github.com/techrote/XenoField) is a renderer reference, not a codebase to absorb wholesale. Nothing here changes CyberSand.

## Two anchor specimens

1. **KAT-Coder-V2.5-Dev REAP-50 Q6_K** — [published GGUF](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-GGUF), a useful existing pruning reference (256 → 128 experts; model card lists 15.3 GB Q6_K). Owner reports local ~250K-context allocation with this model; treat that as a local observation, not a published long-context quality guarantee.
2. **LFM2.5-24B-A2B Turbo-Brilliance-50X NEO-MAX Imatrix Q6_K** — [published GGUF](https://huggingface.co/DavidAU/LFM2.5-24B-A2B-Turbo-Brilliance-50X-NEO-MAX-Imatrix-GGUF), the proposed memory-recovery target (model card lists Q6_K ≈19.7–20 GB). Owner reports very fast inference but little VRAM margin at ~150K allocated context. **Hypothesis:** a calibrated 12.5–25% reduction of MoE experts may recover roughly 2.35–4.71 GB of Q6 expert storage while retaining useful behaviour. No measured result yet.

## Adopted immediate priorities (22 November 2026)

**Reference:** KAT-Coder-V2.5-Dev-REAP-50 **Q6_K**, the owner-confirmed useful local specimen. The recorded ~250K context experience is retained, but raw machine-readable proof has not been imported to this repository. **Primary:** the exact DavidAU LFM2.5-24B-A2B Turbo-Brilliance **Q6_K**, with a **3–5 GB actual memory reduction** goal and explicit protection for throughput, model modes and useful task capability.

Execution sequence: **(1)** freeze exact local Q6 bytes and inference baseline, **(2)** prove quantized tensor layout, no-op round trip and loader support, **(3)** calibrate experts and construct 8/12/16-of-64 candidate plans, **(4)** export serially with a bounded out-of-core compactor, **(5)** qualify saved file bytes, loaded VRAM, quality and context, **(6)** pursue the read-only FRANKENFIELD atlas independently. The published KAT eight-bit file is **Q8_0**; a locally described Q8_K is unverified and further KAT surgery is **deferred until disk space and secondary GPU are available**.

See the [reconciled decision record](docs/RECONCILIATION_AND_DECISIONS.md), [lab architecture](docs/DETERMINISTIC_LAB.md), [exact Q6 exporter workplan](docs/LFM_EXACT_Q6_COMPACTOR.md) and [qualification gates](docs/QUALIFICATION_GATES.md).

## Current boundaries

Documentation-only initialisation. No model downloads, automated expensive inference, direct pushes to XenoField/DRIV3R, checkpoint transformations, or training. Implementations should be isolated and reviewed before promotion; follow the Draft/HOLD practice in [RSN-001](https://github.com/techrote/rustyspork-neurotzacken/issues/1).

See [references](docs/REFERENCES.md) for primary sources and verification status.
