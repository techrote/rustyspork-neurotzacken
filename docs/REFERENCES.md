# Primary sources, tools, precedents and verification checklist

**Indexed:** 2026-11-22. This is a live-reference collection, not a pinned dependency lock or proof of compatibility. Statements noted as verified were checked against the linked public documentation/source on this date; local quality/inference must be reproduced separately. Revise upstream commit hashes and publication metadata when initiating a concrete experiment.

## 1. Project ownership and historical research

| Reference | Why it matters | Confidence / use |
| --- | --- | --- |
| [RustySpork repository](https://github.com/techrote/rustyspork-neurotzacken) | Owner's intended home for neural surgery/reblitteration/FRANKENFIELD | Project destination |
| [RSN-001: reblitteration upstream recruitment PoC](https://github.com/techrote/rustyspork-neurotzacken/issues/1) | Defines frozen feature, low-rank upstream deltas, causal tests, baselines and Draft PR gate | Existing active research scope, **not yet a positive result** |
| [DRIV3R historical Frankenlayers programme](https://github.com/techrote/DRIV3R/blob/main/experiments/driv3r/frankenlayers/README.md) | Exact GREYHOUND / BRANCHWITCH / SCRAMBLED EGGS execution layers and graph-only surgery scope | Source documentation reviewed |
| [DRIV3R 2026-11-15 results](https://github.com/techrote/DRIV3R/blob/main/experiments/driv3r/frankenlayers/results-2026-11-15.md) | Paired speed/behaviour reports on D1-2B; captures STOP and BRANCH attractors | Limited same-seed W0 evidence; not large-model prediction |
| [DRIV3R runtime patch](https://github.com/techrote/DRIV3R/blob/main/experiments/driv3r/frankenlayers/patch_runtime.py) | Example of surgically changing executed graph while Q8 donor weights remain untouched | Different architecture, not a plug-in for LFM2.5-24B |
| [DRIV3R wider repo](https://github.com/techrote/DRIV3R) | Historical worm/agent-policy experiments and other neural research | Preserve repo separation |
| [Neurounner](https://github.com/techrote/neurounner) | Prior small-model research and some historical surgery staging | Reference only; not FRANKENFIELD destination |

## 2. Primary model specimens

| Model/checkpoint | Link | Important distinction |
| --- | --- | --- |
| KAT REAP-50 GGUF | https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-GGUF | Q6_K 15.3 GB listed, REAP 256→128 experts, not the parent |
| KAT REAP-50 BF16 | https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-bf16 | Full-precision **pruned** source for independent quant conversion |
| KAT upstream parent | https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev | Parent, materially different parameter count |
| KAT pruning/conversion pipeline | https://github.com/t-timms/kat-coder-nvfp4 | Source for pipeline/renormalisation, quality and known negative experiments |
| LFM exact proposed Turbo-Brilliance GGUF | https://huggingface.co/DavidAU/LFM2.5-24B-A2B-Turbo-Brilliance-50X-NEO-MAX-Imatrix-GGUF | Q6_K ≈19.7–20 GB; confirm selected MAX-NEO vs regular NEO, actual file SHA |
| Official post-trained LFM2.5-24B | https://huggingface.co/LiquidAI/LFM2.5-24B-A2B | Architecture reference but **do not assume the weights match Turbo-Brilliance** |
| Official LFM model config | https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/config.json | 40 layers, two dense, 64 experts, top-4, 131,072 native context; inspect actual model metadata |
| Official LiquidAI release article | https://huggingface.co/blog/liquidai/lfm25-24b-a2b | Release context, evaluation settings and architecture/context exposition |
| Earlier LFM2-24B-A2B original | https://huggingface.co/LiquidAI/LFM2-24B-A2B | **Earlier LFM2, not LFM2.5**; not an interchangeable donor |
| LFM2.5-8B-A1B REAP-50 | https://huggingface.co/konic-labs/LFM2.5-8B-A1B-REAP-50 | Smaller pruned reference with quantified quality loss |
| LFM2.5-8B-A1B REAP-50 AWQ INT4 | https://huggingface.co/konic-labs/LFM2.5-8B-A1B-REAP-50-AWQ-INT4 | Distinct expert pruning + quantisation precedent |

**Unverified lead:** An earlier conversation mentioned a preexisting *LFM2-24B* REAP derivative. No exact checkpoint/link has been independently substantiated in this source pass. Do not claim a ready-made LFM2.5-24B REAP release exists; investigate the exact repo/revision separately before relying on it.

## 3. Expert pruning, quantisation and format engineering

| Source/tool | Link | Intended role |
| --- | --- | --- |
| REAP original research | https://github.com/CerebrasResearch/reap | Expert-pruning methodology |
| REAP paper | https://arxiv.org/abs/2510.13999 | Scientific context for activation/router saliency |
| REAP CUDA | https://github.com/egesabanci/reap-cuda | Existing CLI, layerwise observation, expert prune/merge, multiple adapters |
| LFM adapter source | https://github.com/egesabanci/reap-cuda/blob/main/src/reap/model_adapters.py | Fused expert layout and expert_bias slicing; adapt/test for 24B |
| REAP CUDA adapter docs | https://github.com/egesabanci/reap-cuda/blob/main/docs/model-adapters.md | Architecture-specific adapter contract |
| REAP CUDA README memory modes | https://github.com/egesabanci/reap-cuda#memory-modes | **Known limitation:** layerwise final mutate/save still reloads full model |
| REAP CUDA reports | https://github.com/egesabanci/reap-cuda/tree/main/reports | Real 8B compression/quality/benchmark/format research |
| llama.cpp | https://github.com/ggml-org/llama.cpp | GGUF runtime, model graph, quantised inference |
| ggml | https://github.com/ggml-org/ggml | Underlying tensor formats, quant layouts and GGUF tooling |
| Hugging Face safetensors | https://github.com/huggingface/safetensors | Streaming/metadata access to BF16 research parents |
| Hugging Face Transformers | https://github.com/huggingface/transformers | Official/model-adapter reference implementation; pin compatible LFM version |
| PyTorch | https://github.com/pytorch/pytorch | Activation tracing, low-rank updates, numerical controls |

REAP CUDA README directly distinguishes (a) layerwise calibration/observation from (b) the full-model final mutation/save reload. A 24GB consumer GPU is not sufficient for whole-model 24B BF16 residency. Implement and verify a streaming exporter or use appropriate larger-memory resources; do not claim layerwise calibration alone solves the final export.

## 4. Visual-rendering references and technology

| Reference | Link | Specific relevance |
| --- | --- | --- |
| XenoField repository | https://github.com/techrote/XenoField | Native WebGPU synthesis engine, owner-developed |
| XenoField V4.4a architecture | https://github.com/techrote/XenoField/blob/main/docs/V4.4a-ARCHITECTURE.md | Render pipeline, volume tiles/raymarch, masks, depth, quality planner and diagnostics |
| XenoField V4.4b clouds | https://github.com/techrote/XenoField/blob/main/docs/V4.4b-PROCEDURAL-CLOUDS.md | World-space AABB editing, axis constraints, scene-space volume placement |
| XenoField README | https://github.com/techrote/XenoField/blob/main/README.md | Native WebGPU/Node 20+ local server and runtime expectations |
| WebGPU specification | https://www.w3.org/TR/webgpu/ | Target GPU graphics/compute API; browser support must be probed |
| WGSL specification | https://www.w3.org/TR/WGSL/ | Shader compute/render language for GPU statistical presentations |

The four image references supplied during concept discussion are *aesthetic sketches*: luminous cyan neural filaments, bilateral orange/blue computational fields, dense coloured network structures, and sparse pale star-like branching forms. None proves a particular neural connectivity. The images themselves were not copied into this repository by this documentation change.

**Important correction of terminology:** Xenofield currently uses **WebGPU**, not simply WebGL. WebGL2 fallback is an engineering option to investigate, not a verified feature of the proposed workstation. Also, XenoField's volume renderer is a visual-effect engine, not a ready-made exact LLM tensor explorer.

## 5. Evaluation and context references

| Reference | Link | Caution |
| --- | --- | --- |
| Official LFM2.5 config | https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/blob/main/config.json | Native max_position_embeddings=131072; 150K/250K require extrapolation study |
| LiquidAI model release/results | https://huggingface.co/blog/liquidai/lfm25-24b-a2b | Benchmark values refer to specified model, hardware, engine and settings; not automatically to derivative Q6 |
| KAT pipeline results | https://github.com/t-timms/kat-coder-nvfp4 | Many figures are NVFP4A16 or specific Q4 runs, **not** a Q6 guarantee |
| Smaller LFM REAP results | https://huggingface.co/konic-labs/LFM2.5-8B-A1B-REAP-50 | MATH500 and BFCLv3 losses illustrate risk; not a predicted 24B delta |
| DRIV3R graph-results record | https://github.com/techrote/DRIV3R/blob/main/experiments/driv3r/frankenlayers/results-2026-11-15.md | Valid output and measured speedups did not imply diverse or generalised behaviour |

## 6. Immediate source audit tasks before code

- Pin all dependency and model revision hashes rather than using 'latest'.
- Confirm exact Q6 GGUF files, quant tensor inventories and tokenizer/template preservation. MAX-NEO and regular NEO are different test specimens.
- Check whether a true full-precision DavidAU Turbo-Brilliance parent is available and licence-compatible.
- Inspect LFM adapter on a tiny 24B-shaped synthetic model and test expert bias/router shape updates.
- Verify llama.cpp at a pinned commit can consume a pruned LFM with nonstandard expert counts; test against small synthetic GGUF before expensive export.
- Determine how to stream compact Q6 expert tensor blocks (and which tensors need dequantise→requantise).
- Bench GPU memory pressure under concurrent native Firefox/WebGPU viewer and inference, including device-loss recovery.
- Archive raw results, including unsuccessful runs; no benchmark result in this index substitutes for our own paired experiment.

## 7. Newly reconciled project references and technical caveats (22 November 2026)

| Source | Link | Why included |
| --- | --- | --- |
| Design/owner decision reconciliation | [RECONCILIATION_AND_DECISIONS.md](RECONCILIATION_AND_DECISIONS.md) | Exact adopted LFM and KAT specimens; retained visual intent and engineering boundaries |
| Deterministic research controller | [DETERMINISTIC_LAB.md](DETERMINISTIC_LAB.md) | Content-addressed DAG, pins, failure capture, optional constrained adviser |
| LFM source-exact exporter study | [LFM_EXACT_Q6_COMPACTOR.md](LFM_EXACT_Q6_COMPACTOR.md) | Actual-Q6 observation, fused quant blocks, router-bias rewrite and runtime blocker |
| FRANKENFIELD UI/backend contract | [FRANKENFIELD_IMPLEMENTATION_CONTRACT.md](FRANKENFIELD_IMPLEMENTATION_CONTRACT.md) | Exact tensor masks behind multiresolution WebGPU visualization |
| Surgery operations taxonomy | [SURGERY_OPERATIONS.md](SURGERY_OPERATIONS.md) | Reversible numerical/structural/graph interventions and Funnel |
| Qualification matrix | [QUALIFICATION_GATES.md](QUALIFICATION_GATES.md) | Measured VRAM/quality/speed/context and future owner acceptance |
| NVIDIA Nemotron Lightning model | https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 | **Optional adviser research**, not a third primary experimental subject |
| Nemotron Lightning quant inventory | https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF | Q4_0 (~19.1 GB) vs Q4_K_M (~25.5 GB) illustrates why GGUF **tensor-level** types matter more than filename quant labels |
| Nemotron Q4_K fallback analysis | https://huggingface.co/bottlecapai/ThinkingCap-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/blob/main/README.md | Expert widths not divisible by 256 trigger mixed fallback quantization in conventional llama.cpp pipelines; counterexample relevant to LFM compaction audits |
| Original Turbo-Brilliance GGUF | https://huggingface.co/DavidAU/LFM2.5-24B-A2B-Turbo-Brilliance-50X-NEO-MAX-Imatrix-GGUF | Actual family lists regular Q6 and MAX-NEO Q6; hash the **chosen** one |
| Public KAT-REAP eight-bit file | https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-GGUF | The public eight-bit GGUF is **Q8_0**, not evidence of a Q8_K file |

**Do not extrapolate the Nemotron fallback to LFM:** it is an existence proof for per-tensor format audits, not a measurement of LFM tensor types. Do not assume hosted adviser access or prices until separately verified.

**End of sources index.** Go to [roadmap](RESEARCH_ROADMAP.md), [specimens](MODEL_SPECIMENS.md), [compaction](REAP_AND_QUANTIZATION.md), [FRANKENFIELD](FRANKENFIELD_INTERFACE.md), [experiment protocol](EXPERIMENT_PROTOCOL.md).
