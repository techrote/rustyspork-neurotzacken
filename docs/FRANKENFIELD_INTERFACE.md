# FRANKENFIELD — 3D neural atlas and surgical workstation

**Design record, 2026-11-22. Status: proposed UI and backend, not an implemented application.**

The graphical responsibilities in this historical design have transferred to [Xenovection](https://github.com/techrote/Xenovection). RustySpork retains canonical tensor addresses, selection-mask validation, numerical statistics, provenance and verified checkpoint edits. See the [current repository boundary](../README.md#repository-boundary).

## 1. Product hypothesis

A dedicated neural anatomy/surgery laboratory for open-weight models: inspect a 4–40B model as a navigable, statistically faithful structure; select tensor coordinates and graph components intuitively; plan and preview deliberately destructive experiments; execute them against an immutable donor; compare behavioural consequences; and export reproducible mutants.

Dual aesthetic: scientific instrument with a deliberately theatrical Rusty Spoon surgery presentation. The theatre may be abstract. The **selection and mutation addresses must be exact**.

Names: **FRANKENFIELD** (working UI title); **RustySpork Neurotzacken** remains the repository and scientific programme name.

## 2. Why 4–40B is renderable (but not fully resident)

1080p (1920 × 1080) has exactly **2,073,600 pixels**. Even if all pixels depicted model weights, 4B and 40B parameter sets imply approximately 1,929 and 19,290 weights per screen pixel respectively. Depth occlusion, interface panels, colour quantisation and 3D surfaces further reduce visual information. Therefore full individual-weight drawing is unnecessary and generally misleading.

Separate **model storage** (tens of GB), **GPU visual cache** (bounded statistical tiles), **visible pixels** (screen-limited), and **actual inference residency** (large, device-dependent). Rendering the aggregate atlas does *not* imply the model fits into WebGPU memory, and simultaneous inference plus graphics may increase OOM risk.

A hierarchy of deterministic statistical summaries is the right data structure, rather than a particle per parameter.

## 3. Multiresolution atlas (out-of-core)

At ingest, enumerate model architecture, ordered blocks, tensors, shape, element type, quant type and stable byte/tensor offsets. Build a signed manifest and a tile pyramid per tensor or per expert. At all zoom levels, a rendered cell corresponds to a documented index-range or list of disjoint index ranges, never to an invented 'neuron position'.

Candidate per-tile payload:
- count and validity/NaN counts; min/max; sum and signed mean;
- mean absolute weight, variance, standard deviation, L2 and selected magnitude quantiles;
- sign counts; zero/near-zero fraction; saturation, scale statistics and quantisation error where measurable;
- compact signed/magnitude histograms and rare extreme/outlier counts (avoid hiding a +X/-X region behind mean≈0);
- optional expert router frequency, activation/saliency and confidence, with source run ID and held-out caveats.

Use a deterministic tiled 2D matrix layout for each tensor, spatially arranged as layer stacks or per-expert columns, with explicit drill-down and axis labels. Derived 3D volumes can interpolate statistics for presentation; do not map that interpolation directly to an edit without recovering the exact underlying index mask.

Use streaming/windowed decode and tile caches; separately bound CPU RAM, disk cache and WebGPU texture/buffer budgets. Prefer screen-space error/coverage tests to choose LOD; visibility and camera changes should not force full-model rasterization or readback.

**Picking path:** pixel/raycast hit → atlas surface/tile ID → tensor name + full shape + exact element subset → preview highlight with parameter counts → serialized edit selection. A selection on coarse cells must either explicitly include **all** weights in the referenced tile ranges or invoke refinement. Never silently sample only the displayed aggregate.

## 4. Visual modes

| View | Purpose | What is real |
| --- | --- | --- |
| Model cosmos | Whole architecture and relative layout | Block order, tensor sizes and connectivity from verified computational graph; otherwise label inferred/decorative links |
| Layer canyon | Groups of attention, convolution, MLP/MoE projections | Actual module/tensor structure, layer numbering and expert counts |
| Matrix microscope | Per-tensor heatmaps and sections | Aggregated or individual numerical weights at stated resolution |
| MoE garden | Expert survival, per-layer routing/saliency maps | Calibration-driven statistics, including dataset and uncertainty |
| Activation overlay | Response to prompts/contrasts | Measured activation/causal evidence, not assumed weight-to-behaviour equivalence |
| Surgical preview | Gouges, grafts, attenuation and predicted footprint | Selected indices, requested transformation and projected bytes; glowing debris is symbolic |
| Difference theatre | Stock vs mutant per-tensor or per-behaviour delta | Measured numerical/behavioural differences with source receipts |

Suggested colour-routing signals: signed mean, abs magnitude, variance, outliers, sparsity, quantisation bin errors, saliency, selection mask and before/after changes. Legends must show actual scale, normalization, threshold and aggregation count.

## 5. Selection grammar and surgery tools

**Selection:** layer/expert/tensor tree, exact submatrix rectangle, 3D box, sphere, brush, lasso, slab/slice, plane clipping, numeric threshold, outlier selector, histogram band and set operators (union/intersect/subtract). A 3D spherical cut only means something after defining a stable coordinate-to-tensor mapping; permit mapping debug mode.

**Non-mutating tools first:** pan/orbit/zoom; probe exact indices/statistics; extract selected masks; dry-run affected element count/bytes; compare two specimens; restore saved selections.

| Named operation | Proposed numerical / graph meaning | Required constraints |
| --- | --- | --- |
| Scoop | Multiply selected weights by gain in [0,1] or zero them | Explicit dtype/quant re-encoding; reversible patch; finite checks |
| Gouge | Remove structured blocks/experts/heads, not arbitrary tensor bytes | Exact topology and loader support |
| Scrape | Bounded noise, clipping or rescaling | Seed, distribution, per-tensor norm limits |
| Fracture | Permute selected compatible weights/rows/channels | Preserve dimensionality, detect invalid permutations |
| Stitch | Blend compatible donor and receiver tensors | Shape, architecture, tokenizer and embedding alignment check |
| Graft | Replace a compatible expert/layer or subset | Routing and interface validation; no speculative semantic promises |
| Greyhound | Skip ordered executed layers at runtime | Graph and recurrent/conv cache safety; may not reduce model-file bytes |
| Branchwitch | Repeat/reorder selected layer executions | Context/state semantics, duplicate state reuse and separate runtime graph receipt |
| REAP | Rank and structurally drop selected MoE experts | Calibration and router/bias remapping; compact export required for file savings |
| Funnel / reblitteration | Freeze validated detector, alter upstream pathway weights | Defer scientific test to [RSN-001](https://github.com/techrote/rustyspork-neurotzacken/issues/1) |

A GUI stroke or lasso initially creates an **operation plan**; it does not secretly write bytes to the donor.

## 6. Reuse from XenoField, with a hard boundary

XenoField [V4.4a architecture](https://github.com/techrote/XenoField/blob/main/docs/V4.4a-ARCHITECTURE.md) already contains a native WebGPU render graph, depth/normal auxiliary passes, compact cached 3D fields, low-resolution volume raymarching and upsampling, independent temporal histories, masks, quality budgeting, diagnostic views and a docked typed-parameter UI. [V4.4b clouds](https://github.com/techrote/XenoField/blob/main/docs/V4.4b-PROCEDURAL-CLOUDS.md) add world-space volume placement plus MOVE/SCALE, axis constraints and pointer ownership. Those are reference-quality parts for exploration/prototyping.

Borrow or adapt **only appropriate modules** after dependency and licence review:
- WebGPU device/surface lifecycle, depth/picking infrastructure and resource-budget planner;
- volume cache and low-resolution compositing for *derived* atlas statistics;
- camera manipulation / axis-constrained 3D boxes;
- LUT/colour operators, histogram/variance shader patterns and carefully restricted optional artistic effects;
- quality/budget telemetry, device-loss recovery planning and UI parameter controls.

Do not wholesale embed the XenoField planet/cloud/FFT/temporal pipeline, or run expensive volumes by default. Its cached channels are general procedural fields, not a ready-made lossless tensor/selection engine. Build the atlas format and exact selection backend separately, then feed GPU-friendly summaries into rendering components.

**WebGPU over WebGL:** XenoField is currently a native WebGPU project. Prefer native WebGPU where supported on the intended Windows/Firefox setup; probe actual adapter and browser capabilities. WebGL2/2D fallback can provide inspection when WebGPU is unavailable, with reduced features. Avoid pretending the browser supports 24B model surgery as a single in-page GPU allocation.

## 7. Application split

1. **Browser/viewer:** WebGPU renderer with strict VRAM budget and UI; all expensive model I/O asynchronous or incremental to preserve interaction; GPU texture cache only for visible LOD.
2. **Local atlas service:** CPU/disk-backed GGUF and/or safetensors reader, architecture adapters, tile reductions, metadata and exact-mask resolver. Does not need inference to open a model.
3. **Surgery engine:** validates declarative patch against donor SHA + tensors, stages modified artifacts, supplies file/graph translators, logs every operation and supports undo by discarding overlays.
4. **Inference harness:** holds model(s) in a separate process, measures real VRAM/cache and runs paired tests. Can launch sequentially to avoid two 20GB specimens co-resident in VRAM.
5. **Evidence store:** manifests, input hashes, masks, scalar metrics, traces, run recipes, screenshots and invalid outcomes; never commit multi-GB weights into source control by default.

Network posture for an initial workstation: localhost-only, no third-party token/telemetry by default. A minimal authenticated/loopback backend for file access is preferable to granting arbitrary local filesystem paths to untrusted web content.

## 8. First genuinely useful PoC

A runnable visualiser for synthetic matrices and one actual 1–4B checkpoint, then a read-only LFM2.5-24B atlas if out-of-core ingest succeeds:

- browse block→tensor→tile→exact index with bounded GPU memory;
- shader/LUT modes for distribution patterns with tooltips on actual numerical ranges;
- 3D brush or box selection serialized to exact deterministic tensor indices;
- verify brush/box selection against known synthetic golden masks;
- preview a small reversible scooping/zeroing patch on a disposable checkpoint; export a new derivative only when format permits;
- inspect LFM's 38×64 expert architecture and eventual REAP rankings; no promise of instant 24B edit/export;
- demonstrate stock vs mutant comparison for a small tractable model before quality claims at 24B.

This is a **proposed implementation plan**, not current source functionality.

See [research tracks](RESEARCH_ROADMAP.md), [REAP](REAP_AND_QUANTIZATION.md), [evaluation](EXPERIMENT_PROTOCOL.md) and [references](REFERENCES.md).

## 9. Implementable viewer-to-surgery contract

The [FRANKENFIELD implementation contract](FRANKENFIELD_IMPLEMENTATION_CONTRACT.md) extends this aesthetic/interaction design with a concrete instrument layout, canonical tensor-coordinate and quant-block model, pick→tile→mask→preview→validated transform chain, declarative selection sample, GPU cache cap, device-loss/fallback strategy and golden synthetic tests. **Rendered imagery is not the source of truth**: the mask resolver and tensor inventory own the exact addresses.

[Operation contracts](SURGERY_OPERATIONS.md) distinguish numeric scoops from structural gouges and runtime skips. [Deterministic lab](DETERMINISTIC_LAB.md) specifies how real edits and tests occur away from the renderer, preserving the 24 GiB Titan for the model where possible. No FRANKENFIELD app or large-model surgery has been implemented in these documents.
