# KAT-GREYHOUND-MOE-001 — 24 GiB structural surgery and REAP comparative laboratory

**Decision record: 2026-11-23. Status: implementation plan, not experimental results.** This is a KAT-specific, bounded workstream within RustySpork. It extends the existing LFM Q6 and FRANKENFIELD programmes; it does not replace or modify them, and it does not authorize a merge or an unattended expensive run.

**Mission:** demonstrate whether targeted MoE expert pruning and later architecture-safe Greyhound surgery can preserve useful KAT-Coder-V2.5-Dev agentic coding behaviour while reducing **actual GPU-resident weight bytes** enough for meaningful KV cache and Windows desktop headroom on the owner's Titan RTX (nominal 24 GiB).

## Contents

1. [Decision and success definition](#1-decision-and-success-definition) — establish what constitutes a useful compact coding agent
2. [Source ancestry and exact controls](#2-source-ancestry-and-exact-controls) — distinguish published checkpoints from genuinely matched experiments
3. [Why surgery instead of full retraining](#3-why-surgery-instead-of-full-retraining) — resource economics, limitations and realistic alternatives
4. [Candidate matrix and capacity](#4-candidate-matrix-and-capacity) — bounded expert-count experiments and provisional storage figures
5. [Engineering sequence](#5-engineering-sequence) — source freeze, format validation, calibration, export, smoke, evaluate
6. [Surgery method contracts](#6-surgery-method-contracts) — REAP, Greyhound, Branchwitch and numeric edits remain distinct
7. [Measurement and release gates](#7-measurement-and-release-gates) — paired agentic tests, VRAM/context and regression controls
8. [Data custody, resource limits and reproducibility](#8-data-custody-resource-limits-and-reproducibility) — receipt and storage contract
9. [First implementation slice](#9-first-implementation-slice) — small, immediately actionable engineering deliverables
10. [Dependencies, blockers and boundaries](#10-dependencies-blockers-and-boundaries) — links, explicit non-claims and GO/HOLD criteria

## 1. Decision and success definition

- **Primary KAT objective:** remove whole **routed experts** rather than blindly compressing every weight below Q4; preserve the full 40-layer execution graph in the first serious compact candidate. Candidate **192 routed experts/layer (25% removed)** is a starting hypothesis, not a declared winner. Compare 224 and 160, and the already published 128-expert REAP-50.
- **Quality target:** retain coding ability, repository editing, tool-call correctness, predictable instruction adherence and long-context retrieval; evaluate failures and specialty loss, not merely the ability to produce plausible code.
- **Hardware target:** native Windows 11, NVIDIA Titan RTX nominal 24 GiB, with Firefox and other desktop allocations. Report model-resident bytes, CUDA peaks and actual allocatable context; do not equate GGUF file size with free VRAM.
- **Model target:** official Kwaipilot KAT-Coder-V2.5-Dev (35B total / ~3B activated, 40 layers, 256 routed experts with 8 active per token, hybrid attention/state architecture); independently verify layer/expert inventory and text-only serialization from the pinned file.
- **Win condition:** a physically smaller runnable checkpoint with a verified source ancestry, matched baseline, credible non-regression evidence and measurable usable GPU headroom. Runtime layer skipping without unloading weights **does not** satisfy the capacity goal.
- **Priority relationship:** this is an additional KAT-specific REAP/Greyhound experiment. The existing DavidAU LFM2.5 Q6 3–5 GB goal in [LFM_EXACT_Q6_COMPACTOR.md](LFM_EXACT_Q6_COMPACTOR.md) remains separately controlled; the earlier deferral of **KAT Q8_0 follow-on surgery** remains in force pending disk and second-GPU clearance.

## 2. Source ancestry and exact controls

| Role | Authoritative location | Required caution |
| --- | --- | --- |
| Original full-precision parent | [Kwaipilot/KAT-Coder-V2.5-Dev](https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev) | Pin exact snapshot SHA; include architecture/config/tokenizer/template hashes |
| Convenient stock quant | [bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF](https://huggingface.co/bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF) | Different quantization pipeline from Ttimms; useful practical baseline, **not** isolated pruning ablation |
| Published pruned BF16 source | [Ttimms/KAT-Coder-V2.5-Dev-REAP-50-bf16](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-bf16) | 256 to 128 experts, with router renormalization; text/vision and MTP handling differ |
| Published GGUF comparison | [Ttimms/KAT-Coder-V2.5-Dev-REAP-50-GGUF](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-GGUF) | Q4_K_M / Q5_K_M / Q6_K / Q8_0; do not invent a published Q8_K |
| Procedure and known traps | [t-timms/kat-coder-nvfp4](https://github.com/t-timms/kat-coder-nvfp4) | Inspect frozen commit; router renormalization fix, calibration data, conversion --no-mtp and vision stripping |
| REAP foundations | [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) and [reap-cuda](https://github.com/egesabanci/reap-cuda) | Pin architecture adapter and license; an observation-only memory mode does not guarantee low-memory final export |

**Critical A/B rule:** compare published stock GGUF against published REAP-50 GGUF for **practical end-to-end utility**, but label it **pruning + potentially different conversion/quantization/vision-processing**. A causal pruning delta requires the same unpruned parent, a matched text-only topology/conversion/quantization pipeline and a pruned derivative produced by that identical pipeline. Run a 0%-prune conversion control first. Pin the exact REAP upstream revision before reproducing anything; “same repository name” is insufficient.

The author reports **52.0% on a 50-case SWE-bench Verified agentic pilot** for an NVFP4A16 variant; this is not a validated Q6 GGUF result or a matched comparison to the original model's reported 69.4%. No retention percentage is claimed here.

## 3. Why surgery instead of full retraining

- Approximate full BF16 weights for 35B parameters alone: **70 GB decimal**, before activations, workspace, gradients and optimizer states. Classic mixed-precision Adam-style full fine-tuning often uses **roughly 12–20 bytes/parameter** or more in aggregate depending on configuration: hundreds of GB of total accelerator/host storage are unsurprising. This is a **training** budget, not an inference or structural-pruning requirement.
- Structured expert ranking can be calibrated using representative workloads, then expert tensors can be **repacked** (preferably CPU/disk streaming). Neither stage intrinsically needs full-model gradient optimization. They still require model access, compute, disk space and potentially substantial host memory.
- GPU/CPU offloading, sharded optimizers, checkpointing, LoRA/QLoRA and adapters can train or adapt parts of a model with much less VRAM. They are valid controls or later repair options; claiming full training is literally impossible on 24 GiB would be incorrect.
- GPU inference and GPU surgery are different budgets. Avoid simultaneously holding original and pruned checkpoints on the Titan. A no-op or quantized GGUF can be inspected from disk/CPU without pretending all weights are in device RAM.

## 4. Candidate matrix and capacity

Published sizes below are **decimal GB checkpoint sizes**, not device-VRAM benchmarks. New candidates use only rough interpolation from the published Q4 endpoints; actual quant packing, offsets and untouched tensor formats determine output bytes.

| Arm | Experts/layer | Purpose | Q4 checkpoint bytes / status |
| --- | ---: | --- | --- |
| KAT-STOCK-Q4 | 256 | Practical stock baseline | **21.39 GB** published (Bartowski) |
| KAT-STOCK-Q3 | 256 | Equal-memory lower-bit control | Exact file/variant to pin; performance unmeasured |
| KAT-REAP-224 | 224 | Conservative prune, 12.5% expert reduction | **~18.9 GB** hypothesis |
| **KAT-REAP-192** | **192** | Primary 25% expert reduction | **~16.4 GB** hypothesis |
| KAT-REAP-160 | 160 | 37.5% reduction stress arm | **~13.9 GB** hypothesis |
| KAT-REAP-128-Q4 | 128 | Published 50% reference | **11.4 GB** published (Ttimms) |
| **KAT-REAP-128-Q6** | **128** | Published higher-precision compact control | **15.3 GB** published (Ttimms) |
| KAT-NULL-ROUNDTRIP | 256 | Same-converter, same-quantizer 0%-prune control | Measure; **not** presumed byte-identical |

At 24 GiB physical capacity, 21.39 GB stock Q4 is tight once Windows, runner buffers and KV cache are included. A 15.3 GB REAP Q6 leaves substantially more **nominal** capacity but still needs measured end-to-end residency. For new candidates, the experiment goal is **at least ~4 GiB usable extra context/desktop headroom versus stock Q4**, conditional on retention; **do not translate decimal storage savings directly into GiB of allocatable GPU memory**.

Only pursue extra precision (Q8_0, not an invented Q8_K) when disk space and a second GPU or an approved out-of-core procedure make it reasonable. Native context reported by KAT config is 262,144 tokens; allocation alone is not proof of retrieval or tool-use quality at that length.

## 5. Engineering sequence

| Gate | Work | Mandatory evidence | Stop condition |
| --- | --- | --- | --- |
| K0 — intake | Resolve exact HF commits, file SHAs, GGUF inventory, tokenizer/chat template, licensing, local resources | Immutable source receipt; tested compatible llama.cpp version | Missing parent identity or unsupported loader |
| K1 — donor/no-op | Inspect MoE expert axes, router, shared expert and full/linear attention; build tiny synthetic 0%-prune fixture | Byte-level or value-equivalent no-op, changed tensor set, deterministic smoke | Unexplained output/metadata drift |
| K2 — practical baselines | On one pinned runtime test stock Q4, published REAP-50 Q4 and Q6 and the selected stock Q3 | Same prompts/context/batch/KV dtype; raw per-case outputs, VRAM/CPU peaks | Unstable/OOM baseline without a documented fallback |
| K3 — saliency | Calibrate expert importance using code, tools, instruction, multilingual, rare-language and long-tail cases | Per-layer ranked experts, routing, sample/seed hashes; separate untouched eval | Rare capability uncovered or missing renormalization |
| K4 — compact | Export 224/192 first; 160 only after K4 smoke; validate all router IDs, top-k=8, shared experts, state graph, packed tensor sizes | Source-to-survivor map, hashes, loader + deterministic forward pass | Invalid tensor stride/quant block, state-cache faults or NaNs |
| K5 — pilot | 30–50 fixed stratified cases, source-matched and same runtime; minimal edit/test loop | Case-level correct/incorrect, compiler/test traces, degeneration diagnostics | Tool schema break, looping, language drift or severe regressions |
| K6 — qualified eval | >=200 held-out cases across task strata, multiple seeds for stochastic settings; 32K/64K/128K context ladder then larger as permitted | Uncertainty, actual memory and throughput, failure archive | No verified capacity gain or unacceptable task loss |
| K7 — decision | Compare Pareto frontier (retained usefulness, usable VRAM, speed, CPU spill) | GO / HOLD / NO-GO with owner acceptance procedure | Never auto-promote or merge |

**Resource preflight before K3/K4:** measure host RAM, SSD available, swap/WSL limits, CUDA device allocation and donor availability. The upstream REAP author cites a WSL environment with ~78 GB usable RAM and ~250 GB disk; these are **not known capacities of this owner's machine**. Implement out-of-core finalization or stop with an actionable blocker; do not assume its published runtimes translate to a Titan RTX.

## 6. Surgery method contracts

**Method A — REAP expert excision (first).** Rank routed experts per layer using actual output-activation/router saliency. Preserve shared experts initially; keep 40-layer backbone and top-8 routing. Select survivor original IDs; update router columns and renormalization, expert biases if present, packed expert tensors, metadata and exact runtime graph. Calibrate away from evaluation holdouts. Do not simply blank router outputs: suppression alone frees no storage.

**Method B — source-exact GGUF compaction.** Prefer lossless quantized block copying when expert slices and block alignment are proved. If expert layout or packed quant format prevents safe copying, use memory-bounded BF16/safetensors streaming + matched requantization and record all nonselected numerical drift. No successful direct-GGUF compactor is established merely by this document.

**Method C — Greyhound depth surgery (later).** First test *runtime-only* ordered layer bypass with correct attention/linear-attention state semantics. Track behaviour, latency, cache validity and separate GPU residency. Physically omit blocks **only after** a validated exporter/runtime handles the changed layer schedule and associated state. Native KAT hybrid linear/full attention makes naive arbitrary block deletion especially hazardous; depth surgery must not be the first capacity experiment.

**Method D — numeric scoop/scrape/graft and reblitteration (separate).** These normally do not reduce dense checkpoint bytes. Preserve exact param masks, quant-block collateral damage receipts and frozen targets; refer to [SURGERY_OPERATIONS.md](SURGERY_OPERATIONS.md) and [RSN-001](https://github.com/techrote/rustyspork-neurotzacken/issues/1). Optional tiny adapter repair after pruning must have its own control and inference memory budget, never be conflated with a pure REAP result.

## 7. Measurement and release gates

Pair by test case **and** inference settings; start new model servers sequentially, never compare stock cold startup to pruned warm cache without labeling it. Capture:
- **Ability:** executable coding tests/pass rate, patch applies/tests pass, tool/function JSON/schema correctness, multi-file repairs, language and format drift, refusal/overcompliance, looping, short/long task regressions. Distinguish standalone coding questions from autonomous repository repair.
- **Performance:** prompt prefill and decode tok/s, TTFT, p50/p95 latency, wall-clock per successful fix, context prefix-cache effects, batch and offload overhead. Multiple process restarts for variance; report median/range.
- **Memory:** GGUF exact bytes (decimal GB), resident model bytes, total allocated/reserved CUDA peak (MiB/GiB), Windows desktop/Firefox occupancy, CPU RAM/swapping, KV K/V precision and size, maximum stable context with margin, fragmentation/OOM frequency. GPU savings without comparable task quality is not an automatic success.
- **Context:** fixed 4K/32K/64K/128K ladder, plus 256K as a separate stress experiment if capacity allows, with actual long-context retrieval/agent replay rather than allocate-only success.
- **Controls:** source-matched 0%-prune; random expert removal at matched quota; router-frequency baseline; REAP; Q3 unpruned at comparable bytes; REAP Q4 vs Q6; optional equal-byte mixed precision. Detect and report score changes driven by quantization separately from pruning.
- **Safety and validity:** tokenizer/special tokens/template unchanged when intended; no nonfinite logits, invalid routing indices, stale attention/linear state, repeated-token attractors, unsupported quant kernels or silent CPU offload masquerading as a GPU win.

**Provisional acceptance target:** >=4 GiB *measured* extra usable GPU headroom at a fixed supported context and KV precision versus stock Q4, with no material coding/tool-use regressions on pre-registered held-out tasks and no critical mechanical failures. Define quantitative quality noninferiority margins and statistical power **before viewing results**; small benchmark differences are inconclusive, not automatically wins. If quality is unacceptable, report a negative result and retain the candidate receipts.

## 8. Data custody, resource limits and reproducibility

Every run records:
- Source model repo/revision, exact filenames/SHA-256/bytes, config/tokenizer/template/license; upstream REAP, converter, quantizer, runtime commits; Windows/NVIDIA driver, environment and CUDA build.
- Seed, algorithm version, calibration dataset sha256 and sample ID split; expert saliency/rankings; per-layer original-to-survivor ID maps; frozen shared/unchanged tensor inventories.
- Structured surgery plan and predicted touched/re-encoded quant blocks; output SHA-256 and bytes; loader/inference/VRAM receipts; case-level evaluation and uncertainty estimates; failure classifications.
- Explicit CPU RAM, GPU VRAM, disk and wall-clock budgets with a **preflight stop**, plus an owner-approved download/compute budget for large model trials. Automatic GitHub CI must **not** fetch model weights or run large GPU jobs.
- Staged output into new paths, write/flush/hash then atomic rename. Donor paths read-only, never overwritten; no multi-GB checkpoints committed to Git. Immutable JSON/CSV manifests and links are sufficient.
- Restart-safe, content-addressed DAG with negative outcomes kept; no self-scheduling autonomous mutation. Follow [DETERMINISTIC_LAB.md](DETERMINISTIC_LAB.md) and [QUALIFICATION_GATES.md](QUALIFICATION_GATES.md).

## 9. First implementation slice

**KAT-GREYHOUND-S0: inexpensive, CPU-first, no downloads without approval.**

1. Implement read-only HF snapshot/source receipt and local GGUF tensor inventory CLI; parser should work on header/tensor metadata without loading weights into CUDA.
2. Build synthetic fused-MoE fixtures (4/8 experts, top-k < count) with quantized block edge cases. Provide no-op, exact survivor-copy and explicitly invalid router maps, with golden assertions.
3. Write an expert-survivor manifest validator: sorted unique original IDs, layer-specific cardinality, original-to-new mapping, top-k bounds, shared-expert preservation, axes and quant alignment checks.
4. Generate reproducible baseline invocation recipes for stock Q4, published REAP-50 Q4/Q6 and original Q3, but do not auto-run expensive model tests.
5. Produce the first **K0/K1 GO/HOLD** receipt and a bounded proposal for the first actual Titan RTX baseline run. Only then implement K3 calibration/K4 exporter.

**Expected source structure (proposal, none exists yet):**
- scripts/kat/inventory.py — source/model metadata receipt, hashes and shape census.
- scripts/kat/validate_survivors.py — architecture-aware surgeon-plan validator.
- tests/fixtures/kat_tiny_moe/ — nonproprietary synthetic quant + router fixtures.
- experiments/kat_greyhound/arms.yaml — versioned cases/constraints, not large weights.
- results/kat_greyhound/<run-id>/ — ignored generated artifacts, manually archived evidence.

Review the source and tests in a Draft/HOLD PR, but **do not fork another independent REAP implementation** if the pinned upstream pipeline can be adapted.

## 10. Dependencies, blockers and boundaries

- Existing roadmap: [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md); separate LFM target: [LFM_EXACT_Q6_COMPACTOR.md](LFM_EXACT_Q6_COMPACTOR.md); operation definitions: [SURGERY_OPERATIONS.md](SURGERY_OPERATIONS.md); qualification: [QUALIFICATION_GATES.md](QUALIFICATION_GATES.md).
- Exact parent Hugging Face revision of Ttimms' pruning run remains **to be proven from pinned pipeline/config/run artifacts**, not guessed from the current official repo head.
- Full original BF16 + pruned BF16 + GGUF staging can exceed local disk resources. KAT's parent includes vision-related assets not carried into the text-only pruned GGUF; control this explicitly.
- KAT's published Q6 result is a checkpoint **format**, not an independent published quality benchmark on this owner's Titan. A very long allocated cache is not proof of equivalent long-horizon reasoning.
- This plan documents methods and acceptance only. **No new candidate has been pruned, trained, exported, locally benchmarked, or verified by the creation of this file.** No CyberSand, DRIV3R or XenoField files are modified; no automatic merge authority.

**Next decision:** implement KAT-GREYHOUND-S0, validate format and receipts, then run published baselines and a carefully bounded REAP-192 candidate if K0/K1 pass.
