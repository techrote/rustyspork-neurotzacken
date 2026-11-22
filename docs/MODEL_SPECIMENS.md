# Model specimens, topology and memory budget

**Recorded:** 2026-11-22. Published figures and architecture are identified as source-reported; all local hardware and performance claims below are owner observations. No specimen has been downloaded or modified by this documentation task.

## 1. Reference: KAT-Coder-V2.5-Dev REAP-50

- Published GGUF: https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-GGUF
- Source/pipeline: https://github.com/t-timms/kat-coder-nvfp4
- Pruned BF16 source: https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-bf16
- Parent: https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev
- Published model-card summary: Qwen3.5-derived hybrid MoE, **256 → 128 experts**, ~19B remaining parameters, ~3B active; vision tower stripped; router-renormalisation correction. The Q6_K GGUF is listed at **15.3 GB** (decimal display).
- Critical distinction: source-reported benchmark and speed numbers for NVFP4 / Q4 are not automatically valid measurements for this user's Q6_K runtime. KAT's working pruning result is encouraging, but cannot independently prove LFM2.5 will retain its quality.
- User-observed local load: Q6 fits with about **250K allocated context**, not independently verified as reliable long-context reasoning.
- Potential uses: compare tensor/statistical distributions, expert surviving fractions, router correction, before/after quality and size, and atlas interaction against a known compact checkpoint.

## 2. Primary candidate: LFM2.5-24B-A2B Turbo-Brilliance

- Exact intended quant repository: https://huggingface.co/DavidAU/LFM2.5-24B-A2B-Turbo-Brilliance-50X-NEO-MAX-Imatrix-GGUF
- Model/card reports **Q6_K ≈ 19.7 GB** for regular NEO and **≈ 20 GB** for MAX-NEO variant (BF16 output tensor); actual filename and bytes must be pinned before editing.
- Official architecture/base references: https://huggingface.co/LiquidAI/LFM2.5-24B-A2B and https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/config.json
- Official config: model type lfm2_moe, hidden_size **2048**, moe_intermediate_size **1536**, num_hidden_layers **40**, num_dense_layers **2**, num_experts **64**, num_experts_per_tok **4**, num_attention_heads **32**, num_key_value_heads **8**, max_position_embeddings **131072**. Layer-type schedule contains **10 full-attention** and **30 convolution** layers.
- Assuming each of the 38 remaining layers has a standard LFM MoE expert with up, gate and down dense projections: **3 × 2048 × 1536 = 9,437,184 weights per expert**. Verify exact packed implementation and tensor types for the actual target file before applying estimates.
- The variant's claimed Turbo-Brilliance prompts/modes, metadata, tokenizer and fine-tuning provenance are part of the specimen. Do **not** silently substitute ordinary LiquidAI BF16 as equivalent to DavidAU's tuned/quantised Q6. Diff metadata and any available raw checkpoint before designing an exporter.
- User-observed local load: very fast; Q6 at ~150K allocated context is marginal for desktop co-residency.

## 3. Analytical pruning budget, Q6_K-only simplifying assumption

Q6_K commonly packs a 256-weight block into 210 bytes (6.5625 bits/weight, 0.8203125 bytes/weight). This is a **theoretical tensor-only** estimate, not a measurement of GGUF size reduction. For uniform removal of N experts from each of 38 layers:

    saved_bytes ≈ 38 × N × (3 × 2048 × 1536) × 210 / 256
    ≈ 294,174,720 × N bytes
    ≈ 0.294175 × N GB decimal

| Removed / 64 per MoE layer | Share of experts | Approx. Q6 tensor storage saved | Nominal 20.0 GB file minus estimate |
| ---: | ---: | ---: | ---: |
| 8 | 12.5% | 2.35 GB | 17.65 GB |
| 12 | 18.75% | 3.53 GB | 16.47 GB |
| 16 | 25% | 4.71 GB | 15.29 GB |
| 20 | 31.25% | 5.88 GB | 14.12 GB |
| 32 | 50% | 9.41 GB | 10.59 GB |

The total 24B model is *not* reduced by those percentages: embeddings, shared/dense components, routers, attention, conv weights and other tensors stay. GGUF metadata, alignment, selective quantisation and retained formats affect actual bytes. A 25% expert removal produces 48 remaining experts/layer and **does not prove** retained ability; REAP ranking/calibration/evaluation is mandatory.

## 4. Context and cache calculation

Official max_position_embeddings: **131,072 tokens**. Above that is extrapolation; successful memory allocation does not imply stable recall, exact retrieval, reasoning or good tool use. No training-context extension is implied by pruning.

For the attention layers alone and FP16 K *and* V, per-token cache is:

    10 attention layers × 2 (K+V) × 8 KV heads × (2048 / 32 = 64 head dim) × 2 bytes
    = 20,480 bytes per token

| Allocated context | Idealised attention KV bytes, decimal | GiB |
| ---: | ---: | ---: |
| 131,072 | 2.684 GB | 2.50 GiB |
| 150,000 | 3.072 GB | 2.86 GiB |
| 250,000 | 5.120 GB | 4.77 GiB |

Going from 150K to 250K costs about **2.048 GB more** of idealised FP16 attention KV, leaving nominal **~2.66 GB** from the 4.71 GB expert-saving hypothesis before other allocations, if all else is equal. Runtime cache paging, padded lengths, type-specific caching, convolutional state, intermediate/scratch, logits, graph buffers, prompt batches, fragmentation and other GPU applications still count. Actual runtime memory can differ materially; KV quantisation and implementation also change the arithmetic.

**Do not turn this arithmetic into a promise of 250K-quality inference.** Measure context accuracy against stock at each supported/experimental length.

## 5. A model-source receipt is required before experiments

For every donor/candidate record a manifest containing:
- HF repository and exact snapshot revision; licence; filenames, byte counts and SHA-256 (individual shards or GGUF).
- GGUF field/tensor list (names, shape, quant type, count); architecture, expert ordering and router configuration; any output tensor precision exceptions.
- Tokenizer revision and hashes; special tokens; chat template and any prompt-mode activation strings.
- Inference engine commit and build flags, GPU driver, GPU/host RAM, offload policy, cache dtype, batch limits and exact context allocation.
- Baseline model-only VRAM, whole-process peak VRAM, desktop/Firefox/YouTube co-residency impact, prompt eval/decode throughput and errors.
- Experimental run seeds, corpus hashes and JSON/CSV task results; never replace lost traces with recreated guesses.

## 6. Related pruning evidence

- Smaller LFM2.5 precedent: https://huggingface.co/konic-labs/LFM2.5-8B-A1B-REAP-50 (published REAP-50 BF16 derivative; card reports MATH500 88.76 → 77.0%, BFCLv3 64.79 → 59.07%). These are not quality forecasts for LFM2.5-24B.
- KAT's BF16 REAP-50 donor and original pipeline explain why calibration, router correction, validation, and model-specific conversion all matter.

Next: [pruning/compaction investigation](REAP_AND_QUANTIZATION.md) and [paired protocol](EXPERIMENT_PROTOCOL.md).
