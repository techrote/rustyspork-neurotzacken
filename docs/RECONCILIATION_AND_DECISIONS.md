# Design Neural Surgery Tool — reconciliation and decisions

**22 November 2026 · Design record / owner-priority reconciliation · No implementation claimed**

This document reconciles the recovered design discussion, the existing RustySpork research package and the subsequent two-specimen decision. It is **not a verbatim transcript**: the recovered design requirements and existing repository material were reviewed together. Assumptions and source-reported observations are labelled separately from measurements made by this repository.

## Decision summary

| Decision | Status | Reason / explicit constraint |
| --- | --- | --- |
| Repository: \`techrote/rustyspork-neurotzacken\` | Fixed | Sole home for neural surgery, FRANKENFIELD and eventual instrument; DRIV3R remains historical |
| **Reference specimen:** KAT-Coder-V2.5-Dev **REAP-50 Q6_K** | Owner-selected | Known useful **externally pruned** model; local owner-confirmed capability and long-context use; raw local acceptance packet not yet imported |
| **Primary candidate:** DavidAU **LFM2.5-24B-A2B Turbo-Brilliance-50X NEO / MAX NEO Imatrix Q6_K** | Owner-selected family; exact file unresolved | Speed and behaviour are the asset to preserve; target **3–5 GB** recovered, not simply a smaller number in a filename |
| 24 GiB Titan RTX | Current limiting device | Keep model's VRAM for inference; CPU/disk-backed surgery, viewer budget, staged comparison |
| Future KAT REAP eight-bit reduction | Deferred | Wait for disk space and second GPU; published REAP-50 GGUF is **Q8_0**, not a verified Q8_K file |
| Inspect a full model without rendering every weight | Fixed | Screen pixels are the visualization budget, not parameter count; streaming statistics and deterministic LOD |
| Impressive 3D instrument with an actual numerical selection backend | Fixed | Luminous/neural-cosmos artistic forms are allowed; apparent 3D geometry is **not** literal biological placement or claimed causal connectivity |
| XenoField-inspired WebGPU frontend | Preferred | Reuse appropriate renderer concepts, masks, LUTs and controls; *do not* import its entire engine or confuse effects with data |
| Structural **expert compaction** first, not layer bypass first | Working choice | Recover file and resident bytes; historical Greyhound run-graph bypass did not physically delete checkpoint weights |
| Deterministic experiment controller | Fixed | No unattended conversational exploration; bounded jobs, pinned inputs, reproducible negative results |
| NVIDIA Nemotron adviser | Optional, non-authoritative | May propose or interpret experiments; cannot change donors, acceptance thresholds or execute unapproved expensive actions |
| Reblitteration / Funnel | Separate issue [RSN-001](https://github.com/techrote/rustyspork-neurotzacken/issues/1) | Protect validated target feature; change **upstream pathways**, measure conditional recruitment and false positives |

## What the initial conversation actually established

The owner requested a sophisticated **WebGL/WebGPU** neural-surgery interface with intuitive 3D navigation and direct selection of layers, experts, tensor regions and "rusty spoon" gouges. In whole-model view, the owner explicitly preferred a quantised visual abstraction over attempting to render every parameter. The interface should take inspiration from **XenoField's data-pattern shaders**, with zoomable structural islands, density/colour statistics, lasso, brush, volume/plane slices, selections and dramatic visual surgery effects.

The software boundary is non-negotiable: clicking a bright object or dragging a sphere must yield exact tensor/index/mask coordinates. The rendered cosmos can be partly editorial; the **backend operation cannot be inferred from unspecified spatial proximity**. A decorative filament, pseudo-synapse or glow is never sufficient evidence for a neural connection. See [FRANKENFIELD interface](FRANKENFIELD_INTERFACE.md) and [implementation contract](FRANKENFIELD_IMPLEMENTATION_CONTRACT.md).

The owner also proposed **reblitteration**: identify a beneficial feature or circuit, freeze it, and modify surrounding/upstream weights to recruit it in more *appropriate* situations. The target is not to force the desired neuron permanently on. Issue RSN-001 retains its own small-model causality and false-positive gates; it is not a prerequisite for model-size surgery.

## Reconciling the prior Greyhound experiments

Historical DRIV3R Greyhound/Branchwitch work proved that **changing executed layer graphs** can alter behaviour and speed. It did not demonstrate useful general model retention, nor does merely skipping layers reduce the size of a stored GGUF. Some variants exhibited single-action/looping attractors. Treat those as lessons in validity and degeneration testing, **not** as a reason to put worms in charge of a 24B experiment.

There are three distinct intervention families:

1. **Graph edits:** runtime layer skip/repeat/reorder; may reduce executed compute but need not save loaded weight memory. Dangerous around LFM convolution state and attention/KV semantics.
2. **Structural edits:** physically remove routed experts, heads or compatible blocks; change tensor dimensions, metadata and graph/runtime; these can genuinely reduce memory.
3. **Numeric edits:** retain shapes but change values, including scooping, perturbation, grafting, mixed-precision transforms and upstream low-rank reblitteration.

Report metrics and failure types separately. "Greyhound treatment" may describe an experimental family, but the memory-release requirement specifically demands **structural compaction**.

## Why LFM is harder than simply reproducing KAT-REAP

KAT has an already published 256→128 expert pruning path and BF16 pruned source. LFM has 64 experts/top-4 in its MoE layers, hybrid convolution/attention execution, an **expert-bias** mapping, and an exact DavidAU derivative that cannot be silently exchanged for LiquidAI's official model. The important unsolved technical challenge is a **source-exact quantised-GGUF compactor or a verifiably equivalent full-precision reconstruction**, not identifying a pruning percentage.

The 3–5 GB target translates into **12–16 of 64** experts pruned per affected MoE layer under a simplified all-Q6 tensor assumption: ~3.53–4.71 GB of expert tensor storage. A 20/64 removal (~5.88 GB) is a stress arm, not a default. Estimates do not establish capability retention, loaded VRAM savings, or safe 250K context.

The official LFM model config advertises **131,072** positions. The owner's ~150K allocation and possible ~250K ambition are deliberately marked as *runtime configurations requiring independent long-context task evidence*. No statement here converts context allocation into certified recall/reasoning quality.

## Explicit scientific and product boundaries

- **Protect source identity.** Pin the selected GGUF's file, sha256, HF revision, tokenizer, chat template, any mode prompts and MAX/regular variant. Official BF16 base is **not** a source-exact substitute without equality evidence.
- **Do not overwrite donors.** Apply declarative reversible edits, always export separate candidates; preserve failures and exact negative controls.
- **No fake causal claims.** Weight magnitude maps, activation heatmaps, routing frequencies and causally validated features are different classes of evidence. Every overlay carries source method and confidence.
- **No forced co-residency.** Simultaneous browser WebGPU and a near-full Titan RTX is optional; a separate viewer-only session is a legitimate workflow. A future secondary GPU changes the resource budget only after measuring it.
- **No weight binaries in Git.** Receipts, schemas, tests, scripts and summaries belong here; checkpoints stay in managed local storage with licensing respected.
- **No unattended GPU experiments or autonomous merges.** Research candidates remain Draft/HOLD unless owner explicitly approves promotion.

## Execution order

**First:** source custody and exact tensor census, deterministic evaluation harness, no-op GGUF/atlas fixtures. **Second:** REAP-derived saliency and demonstrably correct source-exact exporter on tiny synthetic and tractable real models. **Third:** LFM 8/12/16 expert reductions and matched actual VRAM/quality measurements. **Fourth:** optional nonuniform quotas, mixed precision and highly constrained deeper surgery. **In parallel at low cost:** FRANKENFIELD read-only atlas and exact selection tests. **Later:** KAT REAP eight-bit secondary pruning, broader weight surgery and RSN-001 Funnel integration.

See [deterministic lab](DETERMINISTIC_LAB.md), [source-exact LFM compactor](LFM_EXACT_Q6_COMPACTOR.md), [qualification matrix](QUALIFICATION_GATES.md) and [roadmap](RESEARCH_ROADMAP.md).

### Decision changes need their own provenance

Future changes must record **date, owner decision, reason, affected specimens and acceptance impact** in a dated note or PR. Do not overwrite historical premises and then describe a new hypothesis as if it had been proven from the beginning.
