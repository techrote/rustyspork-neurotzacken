# RustySpork Neurotzacken — neural surgery research

A standalone research home for **reversible LLM weight/graph surgery**, expert pruning, conditional circuit recruitment (**reblitteration**), and the prospective **FRANKENFIELD** WebGPU/WebGL neural atlas and surgery workstation.

**Status (22 November 2026): research programme and source index, not an implemented surgery application.** The existing [RSN-001](https://github.com/techrote/rustyspork-neurotzacken/issues/1) defines the first reblitteration proof of concept. Documentation here preserves hypotheses, target models, mathematical estimates, experiment gates and links for future work. Nothing has been pruned, exported, benchmarked or validated in this repository by these notes.

## Read this first

| Document | Purpose |
| --- | --- |
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

## Current boundaries

Documentation-only initialisation. No model downloads, automated expensive inference, direct pushes to XenoField/DRIV3R, checkpoint transformations, or training. Implementations should be isolated and reviewed before promotion; follow the Draft/HOLD practice in [RSN-001](https://github.com/techrote/rustyspork-neurotzacken/issues/1).

See [references](docs/REFERENCES.md) for primary sources and verification status.
