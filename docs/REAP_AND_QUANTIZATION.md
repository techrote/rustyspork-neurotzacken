# REAP, quantisation, and source-exact model compaction

**Research workplan, 2026-11-22; no checkpoint exported or tested here.** Practical objective: preserve the strengths of DavidAU's LFM2.5-24B-A2B Turbo-Brilliance GGUF while recovering roughly 3–5 GB in a configuration used on a 24GB NVIDIA GPU.

## 1. Existing tools and what they actually establish

| Tool/reference | Verified capability or limitation | Use here |
| --- | --- | --- |
| [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) and [REAP paper](https://arxiv.org/abs/2510.13999) | Router/activation-based expert-pruning research | Saliency definition and baseline scientific method |
| [egesabanci/reap-cuda](https://github.com/egesabanci/reap-cuda) | PyTorch/CUDA REAP pruning/merging with full and layerwise calibration | Candidate reference implementation |
| [Lfm2MoeModelAdapter source](https://github.com/egesabanci/reap-cuda/blob/main/src/reap/model_adapters.py) and [adapter documentation](https://github.com/egesabanci/reap-cuda/blob/main/docs/model-adapters.md) | Specific LFM2.5 fused expert tensors, router and expert-bias selection | Learn safe expert ordering/slicing; verify against exact LFM2.5-24B |
| [reap-cuda README: memory modes](https://github.com/egesabanci/reap-cuda#memory-modes) | Layerwise observes one block at a time; README expressly warns that final prune/mutate/save reloads full model via gpu_full plan | **Blocker** for a 24B BF16 model on 24GB VRAM unless extended |
| [t-timms/kat-coder-nvfp4](https://github.com/t-timms/kat-coder-nvfp4) | KAT REAP-50 pipeline, router-renormalisation discussion, paired benchmarks and GGUF export | Published precedent; not interchangeable architecture |
| [konic-labs LFM2.5-8B-A1B-REAP-50](https://huggingface.co/konic-labs/LFM2.5-8B-A1B-REAP-50) | Real smaller LFM2.5 derivative; performance loss is documented | Quality-risk anchor, not a 24B guarantee |
| [konic-labs AWQ INT4 derivative](https://huggingface.co/konic-labs/LFM2.5-8B-A1B-REAP-50-AWQ-INT4) | Combined structural expert pruning + quantisation demonstration | Independent compression technique |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml) | GGUF loading, quantisation and architecture-specific model graphs | Runtime and exact tensor-format research |
| [HF safetensors](https://github.com/huggingface/safetensors) | Weight-file format and indexing | BF16 research/CPU/disk streaming |

Check the current revisions, licence and live code before using any tool; this index is not a frozen dependency lockfile.

## 2. Approach alternatives — keep all under investigation

**A. Official BF16 → REAP → GGUF (scientific control).** Use the BF16 parent, calibrate, prune actual experts with router/expert-bias updates and export GGUF. Provides a tractable, well understood control but may not preserve DavidAU's exact Turbo-Brilliance tuning (which may differ in weights/metadata). The full BF16 24B source is approximately 48GB of weights alone. The user's GPU is unsuitable for full-GPU finalisation without a streaming change or external sufficient-memory environment.

**B. Quantised GGUF direct expert compaction (preferred source-preservation research).** Read a frozen exact Q6 GGUF, select surviving expert IDs per layer, map fused expert weight arrays and their quant blocks, repack/copy supported quantised slices, modify matching routers/biases and architecture metadata, write new GGUF and check llama.cpp compatibility. This is *not* equivalent to simply deleting contiguous bytes; quant blocks, alignment, tensor shape, per-expert scale/bias and loader expectations require an exact implementation. No direct-GGUF working path is established yet.

**C. Out-of-core BF16/CPU/disk-backed compaction.** Reuse REAP's adapter and expert rankings but load only the tensor slices needed at a time on CPU (or host pinned chunks), write destination shards incrementally, and quantise only after a viable compact FP source exists. Requires host RAM, disk and temporary workspace capacity; not 24GB VRAM only.

**D. Expert merging instead of pruning.** Cluster similar experts and combine weights as an alternative; potential less drastic expert-count reductions, more compute/quantisation work and greater risk of distribution drift. Do not assume merging protects speciality without experiments.

**E. Mixed-precision expert preservation.** Keep all 64 experts but allocate lower bit widths to low-saliency matrices, higher precision to sensitive projections or output tensor. Compare at equal checkpoint bytes to structural removal, accounting for runtime kernels and effective speed; lower disk size does not automatically imply lower resident GPU memory.

**F. Runtime expert suppression only (diagnostic control).** Disable routing to chosen experts without rewriting the GGUF to establish effect; **does not actually save their stored weight bytes** unless loader and tensor format are changed. Useful negative/causal control, not a successful memory project.

## 3. Pruning quotas to test, not to declare successful

Test fixed per-layer budgets first: **8, 12, 16** experts removed out of 64 (12.5%, 18.75%, 25%) and optionally **20** (31.25%). From [model size estimates](MODEL_SPECIMENS.md): under pure Q6_K expert packing, expected savings are **2.35, 3.53, 4.71, 5.88 GB** respectively. These are model-weight estimates before the broader VRAM allocator and serialization effects.

Next-stage work: global budget allocation by layer-specific sensitivity, expert similarity, router load, calibration coverage and outlier quality. An expert rarely selected by the calibration corpus may nonetheless be crucial for an uncommon domain. Calibrate on a mixture of coding, tools, general instruction, multilingual, unusual cases and long contexts; hold out independent evaluation suites.

Keep router top-k fixed at four unless separately controlled; after pruning, ensure surviving IDs, gate dimensions, expert_bias, routing softmax/normalisation and tensor serialization are consistent with the model's loader. KAT's published router-renormalisation fix is a reminder to measure, not a patch to copy blindly into LFM.

## 4. Boundary checklist for an exact Turbo-Brilliance derivative

1. Obtain *actual* MAX-NEO or regular NEO Q6 filename, byte count, SHA-256 and HF repository revision before all experiments.
2. Read GGUF metadata, template, tokenizer, quant mapping, tensor layouts, output tensor type and routing configuration. Archive a content manifest, not redistributable full weights in this repo.
3. Compare the actual GGUF's tensor inventory with LiquidAI base/post-trained models. Do not assert that official base BF16 and DavidAU tuned Q6 represent the same checkpoint.
4. Determine whether expert tensors are individually addressable or packed by layer and whether expert dimension is quantised contiguously. Require a format-aware reader/serializer and a byte-preserving proof for untouched quant blocks when possible.
5. Determine which model/runtime versions support a non-default LFM expert count and exact top-k routing. If runtime modifications are needed, pin/patch them separately from weights and verify all code paths.
6. Apply selected experts with correct source → destination reindexing, graph shape changes, quant tensor counts, bias and gate parameters. Validate no dangling IDs, invalid strides, nonfinite outputs or unsupported quant types.
7. Produce a fresh derivative file, never overwrite donor; calculate whole-file and surviving-tensor hashes, actual byte savings and peak resident VRAM at loading and inference.
8. For scientific comparisons, distinguish **pruned, source-exact GGUF** from **pruned, requantised BF16 parent**; expose template/weight differences and quantisation drift.

## 5. Memory modes and hardware risks

REAP-CUDA's layerwise *observation* reduces calibration VRAM, but its documented **final mutate/save** stage still reloads the full model, so it needs engineering work for this user. A GPU with 24GB VRAM can hold an approximately 20GB Q6 inference file with some cache budget; it cannot safely host a ~48GB BF16 24B checkpoint entirely in device memory. CPU memory-mapping, offload or streaming is a different resource equation.

Running the final model and an active WebGPU neural atlas simultaneously on the same GPU competes for VRAM. A dedicated low-overhead viewer mode (e.g. 128–256 MiB target, measured) and graphics-device-loss recovery is preferable to importing XenoField's highest-quality temporal/volume defaults.

Potential partial alternative while studying pruning: decrease the KV cache precision and/or use shorter active context to protect interactive desktop headroom. Unlike structural pruning, such a change does not alter model file bytes.

## 6. Failure modes to document

- Router collapse: formerly inactive expert sets abruptly dominate or specialty experts are removed.
- Retained validation/format validity but degraded task ability, repetitive outputs or single-token attractors.
- Prune amount measured on file but not on actual GPU, due to offloading, buffers and allocation rounding.
- A compactor that writes a file but silently invalidates ordering, routing probabilities, or a special BF16 output tensor.
- Requantised BF16 source losing Turbo-Brilliance behaviour, with losses wrongly attributed to expert pruning.
- Apparent 250K success from allocated cache alone despite an official 131,072 context training limit.
- Overinterpreting KAT NVFP4 performance as a Q6 result or extrapolating 8B LFM results to 24B.

## 7. Minimum engineering deliverables (future)

- Read-only model/GGUF inventory and source receipt generator.
- Portable per-layer expert-ranking manifest plus corpus/seed/source hashes.
- Deterministic compact-export proof on a tiny synthetic MoE test fixture, then a small real LFM.
- Exact router/bias remapping tests, nonfinite check, identical no-op/stock output regression and reserialization tests.
- An executable LFM derivative under constrained VRAM with a documented reproduction path.
- Paired task evaluation and memory/performance report, including all failed arms, archived independently of model weights.

Go to [evaluation protocol](EXPERIMENT_PROTOCOL.md), [model budget](MODEL_SPECIMENS.md) or [reference index](REFERENCES.md).
