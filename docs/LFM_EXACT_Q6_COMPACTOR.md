# LFM2.5-24B Turbo-Brilliance — source-exact Q6 compaction engineering

**Technical feasibility specification, not a working exporter.** No stock weights have been transformed or downloaded into this repository.

## Contents

1. [Custody and exact subject](#1-custody-and-exact-subject) — do not accidentally replace the prized derivative
2. [How much can be saved](#2-how-much-can-be-saved) — quantitative storage budget and VRAM caveats
3. [Two-plane observation and export](#3-two-plane-observation-and-export) — calibration from exact inference, CPU/disk compaction
4. [Quantized tensor repacking](#4-quantized-tensor-repacking) — alignment, dimensions and metadata traps
5. [Research ladder and negative controls](#5-research-ladder-and-negative-controls) — bounded candidate matrix
6. [Mechanical and behavioural gates](#6-mechanical-and-behavioural-gates) — prove load, equivalence, savings and survival

## 1. Custody and exact subject

**Owner-selected target:** [DavidAU LFM2.5-24B-A2B Turbo-Brilliance-50X NEO-MAX Imatrix GGUF](https://huggingface.co/DavidAU/LFM2.5-24B-A2B-Turbo-Brilliance-50X-NEO-MAX-Imatrix-GGUF), at **Q6_K**. This repository exposes more than one Q6 style. The model card indicates a **regular NEO** and a **MAX-NEO** Q6 variant, the latter retaining a BF16 output tensor. Different files have different bytes and must be treated as **distinct donors**. The family's reported sizes are ~19.7/20.0 GB decimal; neither is an actual user's pinned local pathname/hash yet.

The reference [official LFM configuration](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/config.json) reports \`lfm2_moe\`, \`num_hidden_layers=40\`, \`num_dense_layers=2\`, \`num_experts=64\`, \`num_experts_per_tok=4\`, \`hidden_size=2048\`, \`moe_intermediate_size=1536\`, **10 attention + 30 conv layers** and \`max_position_embeddings=131072\`. The **actual GGUF tensor shapes and ordering supersede assumptions** drawn from this config.

**Crucial provenance rule:** an official \`LiquidAI/LFM2.5-24B-A2B\` BF16 checkpoint is not automatically equivalent to a named DavidAU Turbo-Brilliance Q6 derivative. The modes/templates may influence apparent behaviour, and there may be tuning differences. Compare exact tokenizers, chat templates and matched outputs/logits before calling any BF16→GGUF reconstruction "source-exact". Do not assume the quantized derivative can be perfectly reconstructed from a different published parent.

The later KAT experiment is independent: [KAT-Coder REAP-50 Q6_K](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-GGUF) is the reference. Its GGUF list documents **Q8_0**, not a known Q8_K file. A hypothetical future Q8_K should not contaminate the first LFM plan.

## 2. How much can be saved

The 38 expected MoE blocks contain, under the simple fused-three-matrix assumption, **3 × 2048 × 1536 = 9,437,184** weights per routed expert. For Q6_K, 256 scalar weights / 210 bytes gives **0.8203125 bytes/weight**. Under ideal uniform all-Q6 packing:

    one expert in one MoE layer ≈ 7,741,440 bytes
    remove one expert from each of 38 layers ≈ 294,174,720 bytes = 0.294175 GB
    remove N per layer ≈ 0.294175 × N GB of expert matrices

| Remove per layer | Remain | Approx. expert bytes saved | Intended arm |
| ---: | ---: | ---: | --- |
| 0 | 64 | 0 | Frozen baseline / no-op export |
| 8 | 56 | 2.35 GB | Low-risk entry |
| **12** | **52** | **3.53 GB** | First 3–5 GB candidate |
| **16** | **48** | **4.71 GB** | Main 3–5 GB candidate |
| 20 | 44 | 5.88 GB | Conditional stress |
| 32 | 32 | 9.41 GB | Diagnostic only; no quality assumption |

**These are projections for expert tensor storage**, not measured GGUF shrinkage, retained ability, available GPU memory or training-context extension. Real donor tensor types, metadata, padding, alignment and output-tensor exceptions must first be inventoried. Global quota per layer may eventually improve retention at an equal byte budget; first establish a uniform baseline.

For context arithmetic: official LFM 10 full-attention layers × K+V × 8 KV heads × head_dim 64 × FP16 bytes 2 = **20,480 bytes per token** for just the ordinary attention KV tensors. This suggests ~2.50 GiB at 131,072 tokens, ~2.86 GiB at 150K and ~4.77 GiB at 250K before conv state, other model/runtime buffers, cache quant, batching or fragmentation. Native context is **131,072**; >131K is experimental extrapolation, not guaranteed correctness.

## 3. Two-plane observation and export

Separate **saliency measurement** and **weight-file transformation** so neither requires the entire BF16 24B checkpoint in Titan VRAM.

**Observation plane, preference order:**

A. **Exact donor, target-runtime calibration (preferred scientifically):** run the pinned DavidAU Q6 GGUF on a pinned llama.cpp build, instrument router top-k assignments/probabilities and **actually routed expert output norms** at a bounded set of token positions. Aggregate per-layer/per-expert sums, routed counts, feature/mode strata, prompts, quantization, and confidence. A router-frequency-only trace is a useful cheap baseline but is **not identical to REAP**. Instrumentation must not alter the model outputs except measured probe overhead; match no-probe control logits/decoding at a tolerance defined in advance.

B. **Architecture-compatible REAP-CUDA layerwise observation:** adapt [Lfm2MoeModelAdapter](https://github.com/egesabanci/reap-cuda/blob/main/src/reap/model_adapters.py) to the exact feasible source format or another verified-equivalent checkpoint. It understands fused LFM-style experts and expert bias, but the README warns that **layerwise final pruning/save reloads the entire model** in \`gpu_full\` mode. That is a known 24 GiB blocker; **observation support does not imply viable compaction**.

C. **Proxy BF16 official-parent ranks:** use only as an explicitly labelled *transfer experiment*. Never attribute a high score measured on different weights/templates to the exact Turbo donor without corroboration.

**Transformation plane (CPU/disk):** accept a strictly validated survivor/ranking manifest. Open exact Q6 GGUF with a format-aware reader; stream quantized slices per layer; remap all affected router/bias indices; write a new GGUF and aligned metadata; validate independently. The GPU need not hold BF16 donor weights for bytewise safe quant-block moves.

No REAP implementation here currently collects all required exact-GGUF activations. **Building or validating the observation probe is a separate engineering deliverable.**

## 4. Quantized tensor repacking

GGUF is not a simple flat array of independent "experts." The executor needs a **per-architecture map** of tensor name, shape, quant type, byte stride, block size, expert axis and expert-internal layout. In many GGUF implementations fused expert matrices have an expert axis inside a quantized 3D tensor; prove the exact axis order from the **actual donor's headers and llama.cpp loader**, not from Python safetensors names alone.

### Fast-path feasibility proof

1. Read all GGUF headers, tensor dimensions, alignment and metadata; record a canonical donor inventory.
2. Build a tiny synthetic LFM-MoE shaped fixture with **distinct known per-expert values** (and, where possible, the actual fused pack layout). Include values around quant block boundaries and tail/expert boundaries.
3. Quantize at the **exact source tensor type**. Determine whether a whole expert can be copied as aligned blocks **without mixing its bytes with any neighbour**. Check 256-value Q6_K grouping relative to the actual quantized first dimension and expert stride. For unaligned or interleaved formats, do **not** raw-slice bytes.
4. Copy surviving expert ranges in stable original-ID order, or use a bounded dequantise→requantise path for incompatible tensor layouts. Report the requantized subset and any non-bit-identical survivor values.
5. Slice/rebuild **router weights, router bias, \`expert_bias\`, expert-count metadata and graph-expected dimensions** consistently; preserve top-k=4 unless separately tested. Produce \`old_expert_id → new_expert_id\` mappings for *each MoE layer*. Never leave a router targeting pruned IDs.
6. Recheck GGUF offsets/alignment, tensor name count, shape/product/byte accounting, model architecture metadata, tokenizer/chat template/special tokens and MAX-NEO output-tensor dtype.
7. Load candidate using a **pinned runtime with demonstrated support for nonstandard LFM expert counts**. If llama.cpp assumes 64 in architecture-specific paths, changes to graph/runtime may be necessary and must be versioned/tested separately.
8. Compare unaffected tensors against donor by raw byte hashes where possible, otherwise dequantized numerical equivalence; archive explicit exceptions and error bounds.

**A zero-removal round trip comes first.** It should reproduce every tensor's decoded value within the chosen exactness standard, preserve tokenizer/mode behaviour, and pass fixed-prompt inference. If a serializer cannot round-trip an untouched donor correctly, it must not be trusted to do 12/64 pruning.

### Suggested survivor receipt (illustrative, not executable)

```json
{
  "schema_version": "rsn.expert-map.v1-proposal",
  "source_gguf_sha256": "REQUIRED",
  "source_tensor_inventory_sha256": "REQUIRED",
  "model_architecture": "lfm2moe",
  "model_num_experts_old": 64,
  "model_num_experts_new": 52,
  "router_top_k": 4,
  "layers": [
    {
      "layer_index": 2,
      "old_expert_ids_sorted": [0, 1, 3, 4],
      "note": "example is deliberately incomplete; validation must require exactly 52 unique survivors here"
    }
  ],
  "saliency_receipt_sha256": "REQUIRED",
  "calibration_sha256": "REQUIRED",
  "quant_block_policy": "copy_only_when_alignment_proven",
  "non_preserved_tensors": []
}
```

The real validator must demand **38** MoE-layer records, each with **exactly 52 unique valid IDs for this particular arm**, and reject examples such as the deliberately incomplete one. No element-level quant-byte cut-and-paste without formal coverage proof.

## 5. Research ladder and negative controls

**Phase A: no weights changed.** Frozen exact file inventory, user-use inference profile and tokenizer/mode/control receipt. Measure runtime graph/topology against metadata, loaded VRAM, per-token cache bytes, browser video pressure, OOM incidence and throughput.

**Phase B: format proof.** Synthetic expert-axis fixtures, exact zero-prune GGUF round trip, malformed router/bias regression, valid target-runtime forward and cache-state round trips. A small real LFM model may qualify mechanics but **not** target quality.

**Phase C: saliency coverage.** Separate calibration categories: coding, tools, dialogue, reasoning, multilingual, unusual/specialist tasks and Turbo-Brilliance mode prompts. Record not only average score but routed frequency, high-tail activations, rare-expert usage and confidence. Hold out disjoint tasks for evaluation.

**Phase D: bounded candidate grid.** For each 8, 12 and 16 experts removed (20 only conditionally), compare REAP to (i) random seeded pruning, (ii) router frequency and (iii) equal-byte mixed quant or quantized-KV controls, without tuning holdout cases. Run candidates serially to avoid co-residency or GPU exhaustion.

**Phase E: only after a useful structural winner.** Explore uneven per-layer quotas, merge/replace redundant experts, selective per-matrix quantization, and optional light restoration/LoRA. Preserve a matched no-restoration branch to distinguish pruning outcome from recovery.

Expert suppression by disabling router choices **without actually removing parameter bytes is a control, not a memory win**. Graph skipping/repeated LFM layers must be an independent research track due to convolutional and recurrent/state-specific hazards.

## 6. Mechanical and behavioural gates

**BLOCK immediately:** donor hash mismatch; unsupported quant or expert axis; wrong tensor strides/block alignment; missing router/expert bias remap; nonfinite value; candidate unable to load; automatic runtime silently offloading most weights when a "GPU saving" is claimed; stock/no-op output mismatch not explained by tracked serialization.

**Mechanical PASS only when:** candidate loads and executes representative prefill/decode, top-4 routes exclusively to surviving experts, graph/cache reset and long-context staging work, unmodified tensors are proven unchanged under stated exactness, and source metadata/template fidelity is documented.

**Practical GO requires:** actual **3–5 GB** model/loaded memory reduction at relevant inference settings, paired held-out coding/tool-use/reasoning/mode quality retention, no severe output collapse, acceptable speed and robust headroom in realistic Windows + Firefox use. No benchmark number may be substituted for a missing local trial. Reaching 250K allocated context is **not** a promotion criterion by itself.

A future **KAT REAP-50 Q8_0** memory experiment should repeat the same controls after more disk space and a second GPU are available. **Do not begin it now**, and first resolve whether the actual local eight-bit specimen is Q8_0 or a separately produced Q8_K derivative.

Related: [specimen math](MODEL_SPECIMENS.md), [REAP discussion](REAP_AND_QUANTIZATION.md), [deterministic lab](DETERMINISTIC_LAB.md), [qualification gates](QUALIFICATION_GATES.md).
