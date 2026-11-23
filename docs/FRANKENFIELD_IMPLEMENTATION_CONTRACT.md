# FRANKENFIELD — implementable interface-to-tensor contract

**Proposed technical specification.** Extends [the visual concept](FRANKENFIELD_INTERFACE.md) without claiming an existing renderer, GGUF editor or LLM execution backend.

This is a historical design. [Xenovection](https://github.com/techrote/Xenovection) owns graphical rendering, interaction and presentation. RustySpork owns exact tensor coordinates and masks, deterministic statistics, provenance and validation of all proposed edits. The payload sketches below retain their design context. See the [implemented formats](IMPLEMENTATION_STATUS.md) for the current interfaces.

## Contents

1. [Visual interaction and instrument layout](#1-visual-interaction-and-instrument-layout) — a theatrical but interpretable 3D neural cosmos
2. [Atlas ingestion and coordinate identity](#2-atlas-ingestion-and-coordinate-identity) — bounded data, stable addressability
3. [Picking, masks and previews](#3-picking-masks-and-previews) — every UI gesture resolves to exact indices
4. [Backend boundaries and user safety](#4-backend-boundaries-and-user-safety) — independent rendering, immutable donors and local security
5. [Golden tests and first proof](#5-golden-tests-and-first-proof) — how to validate that the view represents real data

## 1. Visual interaction and instrument layout

The original **Design Neural Surgery Tool** request was for an impressive, intuitive WebGL/WebGPU workspace where an entire 4–40B network can be navigated and "scooped" with a rusty spoon. The user explicitly agreed that **not every weight needs a rendered primitive**: whole-model detail should be limited by the pixel budget. The intended visual treatment includes luminous filaments, coloured high-density fields, branching neural silhouettes and a structural model cosmos, inspired by XenoField's procedural shader/volume presentation. These are aesthetic references, *not* an assertion about anatomical or causal connectivity.

Suggested UI arrangement:

    ┌────────────────────────────────────────────────────────────┐
    │ Model / revision / quant / donor SHA · view LOD · GPU cap  │
    ├────────────┬───────────────────────────────┬───────────────┤
    │ Anatomy    │     3D structure / LOD atlas  │ Probe / legend│
    │ tree       │     brush / lasso / cut plane │ tensor/shape  │
    │ layers     │     exact selection overlay   │ stats/source  │
    │ experts    │     optional neural-cosmos FX  │ active masks  │
    ├────────────┴───────────────────────────────┴───────────────┤
    │ Operation timeline · selected elements/bytes · patch plan │
    │ Stock ↔ preview ↔ candidate · evidence · validation      │
    └────────────────────────────────────────────────────────────┘

**Navigation:** model cosmos → block/layer → MoE garden or tensor sheet → matrix microscope → exact element selection. Orbit/translate/zoom and section planes operate on *view* until the user deliberately creates a selection. Support region bookmark, slice/cross-section, isolate/hide, legend inspection, signed distribution, routing/activation overlay, and paired stock/candidate "difference theatre".

**Selection toolbox:** tree ID, rectangle, 3D box, sphere, slab/plane, lasso, brush, histogram threshold, outliers, expert group, and set algebra (union / intersect / subtract). Section and visibility masks must not accidentally become destructive edit masks. A default **inspect-only** mode and explicit preview/apply split are essential.

**Visual truth indicators:** every atlas surface shows aggregation count, metric name, normalization and scale; overlays identify real weight statistics vs calibrated routing vs measured activations vs causal interventions vs decoration. Model-space filaments may depict verified graph edges only if backed by a specific graph source; otherwise label them "decorative / inferred". Colours are not a substitute for real signed and numeric labels.

## 2. Atlas ingestion and coordinate identity

### Source and metadata

A local CPU/disk service inventories GGUF or safetensors **without loading the entire model into GPU memory**. Each tensor record carries model donor hash, original byte source, architecture, tensor name and type, shape, axis labels, stable layout convention, quantization block boundaries, and exact byte/element length.

Two coordinate spaces stay separate:

1. **Canonical tensor coordinates:** source tensor + shape + ordered integer index tuple; operations only target these.
2. **Display coordinates:** model/layer/tensor scene transforms, volumes or 2D tiles, which are **mapping metadata**, never the storage identity.

Define the flattening convention from actual tensor strides. Axis order, mixed transposition and fused expert axes are a common source of silent errors. A 3D "neuron position" has no native meaning for many LLM weights; represent it as **scene geometry encoding a documented tensor index map**.

### Multiresolution pyramid

At ingest build a hierarchy of deterministic, stable tile reductions; streaming over quantized blocks is permitted if the block decoder is pinned and tested. Suggested 64×64 logical tiles for 2D matrix faces, configurable by tensor geometry; exact cell membership (including ragged final tiles) is recorded. Multi-tensor/whole-layer nodes aggregate child statistics but **never lose the child's exact index range**.

Per tile:
- finite count, NaN/Inf count, zero/near-zero counts and positive/negative counts;
- min/max, sum, sum-of-abs, sum-of-squares, mean, variance and selected magnitude quantiles;
- coarse signed/magnitude histograms, rare-outlier statistics, representative quant scales;
- optional quantization error vs **a valid known unquantized source**, not fabricated from the current quant alone;
- separate activation/routing statistics containing corpus digest, run ID, sample coverage and uncertainty.

Statistics must avoid cancellation hiding strong mixed-sign regions. Accurate reductions require explicit accumulator type/overflow treatment. Precompute selected LODs out of core; persist with a manifest linking donor hash, decoder build, tile version and checksum.

**Budget model:** display cache and inference/model residency are disjoint. Prefer initial renderer target of **128 MiB** bounded resources, optional 256 MiB cap only if measured safe. The budget controls atlas textures, geometry, staging buffers and history, but is an **application estimate**, not exact physical VRAM accounting. Do not allocate a giant GPU buffer for all parameters.

At 1080p the complete screen contains ~2.07 million pixels. A 24B checkpoint has >11,500 parameters per pixel **even before UI panels and occlusion**. The only honest full-model representation is aggregate, with local drill-down.

## 3. Picking, masks and previews

### Canonical picking path

    pointer/ray → viewport primitive → atlas tile/surface ID
      → source tensor / shape / stride / exact tile membership
      → immutable selection mask in canonical tensor coordinates
      → affected element counts & quant block accounting
      → human-readable operation preview
      → signed/hashed patch plan (still non-mutating)
      → separate backend validation and new derivative export

At a coarse LOD, selecting a tile must do **exactly one** of:
- select all underlying indices (explicitly state count and range); or
- request a finer LOD before creating the selection.

Never silently select a "representative sample" and present it as the whole region. Highlighted pixels should render a mask **from canonical selected indices**, not independently approximate the intended edit. Undo of a preview discards selection/patch state; undo of applied surgery rebuilds from the immutable donor and prior patches, not untrusted reverse arithmetic.

### Proposed selection payload

```json
{
  "schema_version": "rsn.selection.v1-proposal",
  "donor_sha256": "REQUIRED",
  "tensor_inventory_sha256": "REQUIRED",
  "tensor": "EXACT_NAME_REQUIRED",
  "shape": [64, 1536, 2048],
  "flatten_order": "verified-source-layout",
  "scope": "canonical-index-ranges",
  "ranges": [{"start": 0, "stop_exclusive": 256}],
  "range_semantics": "all-indices-in-range",
  "source_view": {
    "mode": "matrix-microscope",
    "atlas_version": "REQUIRED",
    "lod": 2,
    "view_to_tensor_mapping_sha256": "REQUIRED"
  },
  "selected_elements": 256
}
```

This is a **toy example**, not a real LFM tensor identifier or an implemented schema. Validator must reject missing source hashes, wrong shape, impossible flatten order, overlapping/out-of-range intervals unless canonicalized, nonempty mask in read-only state, and quant-block-unaware mutations. Index ranges should be sorted, merged and immutable after preview approval.

Selection sets must distinguish **source indices** from **candidate indices** after structural pruning. A survivor mapping is required; no implicit edit of "expert 17" after experts have been reindexed.

### Edit preview and UX

Display:
- donor hash/revision and fully qualified tensor path;
- operation, exact source-selected weights, affected structural units and quantized block count;
- predicted *file* change versus measured *device memory* (only available post-export);
- whether the transform is quant-block preserving, requires decode/requantization, or is a runtime graph-only change;
- known model-graph hazards, estimated disk/CPU/GPU budget, evaluation work and validation stage;
- explicit state **Inspect / Plan / Dry-run / Applied to new candidate / Tested**.

A theatrical scoop may fracture voxels or dissolve filaments visually, but **only the validated patch manifest causes numeric effects**. Flashy VFX must be disableable in memory-constrained mode.

## 4. Backend boundaries and user safety

| Subsystem | Resource owner / contract | Forbidden shortcut |
| --- | --- | --- |
| Browser/WebGPU viewer | GPU textures, LOD atlas, camera, picking, UI | Reading/loading raw 24B weights into JS/WebGPU |
| Atlas host service | CPU/disk tensor inventory, tile streaming, exact mask resolver | Assuming rendered LOD tile maps to one individual weight |
| Surgery engine | Canonical edits, GGUF/safetensors compaction, staged derivative | Editing donor in place or raw byte cuts without quant alignment |
| Inference harness | Separate llama.cpp/other pinned runtime process | Assuming file size equals resident VRAM or a loaded context is correct |
| Evidence store | Receipts, calibration provenance, manifests, data retention | Relabelling unmeasured visualization as causal evidence |

Network posture: bind local host only by default; validate file-path allowlist, protect against symbolic-link escape and arbitrary upload, use session-scoped capability for destructive operations, and avoid silent remote telemetry. An untrusted page must not request arbitrary filesystem surgery.

**WebGPU first** (matching the reviewed [XenoField architecture](https://github.com/techrote/XenoField/blob/main/docs/V4.4a-ARCHITECTURE.md)); feature-detect actual Windows/Firefox support. **WebGL2/Canvas2D read-only fallback** if needed; never imply that XenoField already has a selectable 4–40B tensor atlas. Adapt bounded cache, signed LUT/masks, axis manipulation, diagnostics and device-loss recovery concepts, subject to source/licence review. Do not embed its FFT, procedural weather, temporal FX histories and other heavy passes by default.

## 5. Golden tests and first proof

The first UI milestone is **read-only**, not a finished 24B workstation:

1. Index known synthetic tensors with positive, negative, zero, NaN and outlier entries; verify every tile statistic against CPU reference, including odd dimensions and ragged edges.
2. Display a layered structure and shader statistics; selecting a coarse cell recovers **all and only** the exact source indices.
3. Test sphere, box, brush, lasso, plane and threshold masks with known synthetic golden sets; verify set algebra, LOD transitions and viewport resizing do not change selected indices.
4. Resolve one small real GGUF through the exact path→tensor→tile→indices chain; no guessing from display coordinates.
5. Produce a preview patch, verify parent SHA and index budget; **do not export** if the format-specific backend is unavailable.
6. Run a bounded no-op patch on a disposable small model and prove functional equivalence and source immutability.
7. Test WebGPU device loss, Firefox unsupported adapter, viewer-only CPU mode, and repeated camera/zoom/selection without full-model stalls or unbounded cached texture growth.
8. Only then ingest **LFM metadata and 38×64 MoE structure** read-only; show sample coverage and real REAP ranking provenance before enabling surgical edits.

End criterion: a visually striking instrument with an **honest, exact atlas** and bounded GPU consumption, not a merely animated model-shaped sculpture. The full surgery tool will be viable when the small-model end-to-end exporter and paired evaluation are independently validated.

See [operations vocabulary](SURGERY_OPERATIONS.md), [laboratory controller](DETERMINISTIC_LAB.md), [pruning exporter](LFM_EXACT_Q6_COMPACTOR.md) and [FRANKENFIELD concept](FRANKENFIELD_INTERFACE.md).
