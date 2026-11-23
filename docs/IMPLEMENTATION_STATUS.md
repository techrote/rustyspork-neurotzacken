# RSN-IMPLEMENT-001 verification record

This first delivery targets CPU-only model inspection, exact miniature surgery and independently checkable results on Windows. Review status is Draft/HOLD.

## Capability status

| Capability | Status | Scope and evidence |
| --- | --- | --- |
| GGUF inventory | **Implemented · Mechanically verified** | Little-endian v2/v3 parser with bounded metadata and streamed hashing. Both local production GGUFs below were inventoried read-only. Tensor geometry, logical shape and physical byte spans are recorded separately. |
| Gouge and zero-removal | **Implemented · Mechanically verified · Runtime verified** | Generated 4- and 8-expert `lfm2moe` GGUFs using actual Q6_K blocks. Compaction remaps the router, expert bias and all three expert matrices. Zero removal copies the original bytes exactly. |
| Trained small GGUF round trip | **Mechanically verified · Runtime verified** | The local Sprocket-500M Chat Q4_K_M file and its identity Scoop output matched byte for byte and through the CPU smoke request. |
| Probe | **Implemented · Mechanically verified** | Bounded explicit indices into F32, F16 and Q6_K tensors. Exports selected logical values, statistics and selection coverage. |
| Scoop and Scrape | **Implemented · Mechanically verified** | Exact scalar edits to F32/F16 selections. Scale/zero and seeded bounded perturbation produce a new checkpoint and per-index evidence. Quantized numerical editing requires a separately verified encoder. |
| Experiment controller | **Implemented · Mechanically verified** | Source-bound plans, dry-run storage predictions, bounded CPU I/O, a cooperative deadline, new staged outputs, independently reopened verification, content-addressed run directories and generated receipts. |
| Neutral export and selection input | **Implemented · Mechanically verified** | Versioned JSON inventories, exact-index samples, validated plans, expert remaps and numerical delta records. Selections are bounded and resolved in RustySpork. |
| KAT reference topology and manifests | **Implemented · Mechanically verified** | Local 128-expert reference census and survivor validation for recognized layouts. The 224/192/160 cases remain **Proposed** pending a pinned 256-expert common parent, matched quantization, calibration and runtime qualification. |
| LFM2.5 production compaction | **Blocked** | The supplied 24B GGUF is IQ4_XS. Exact DavidAU Turbo-Brilliance NEO/MAX-NEO Q6_K custody, model-scale calibration and target-runtime qualification remain open. The miniature Q6_K pathway establishes the block-preserving mechanism. |
| REAP, Greyhound, Branchwitch, Graft and Funnel | **Proposed** | Activation-calibrated ranking, state-aware execution-graph adapters, cross-checkpoint pairing and causally validated frozen targets are separate research steps. |
| Model-scale inference and evaluation | **Proposed** | The miniature runtime smoke supplies loader and a bounded forward-pass result. Coding/tool-use retention, context integrity, throughput, full activation traces and resident VRAM recovery require paired production trials. |

## Evidence and external selections

The CLI writes UTF-8 JSON. `inventory` binds metadata and tensor descriptors to separate digests and optionally streams the exact file for its SHA-256. `--header-only` leaves the full source digest `null`. The numerical coordinate order is GGML axis 0 fastest. Descriptors include the type's logical block width and serialized byte size. Q6_K stores 256 logical values in 210 bytes. [Pinned llama.cpp quant implementations](https://github.com/ggml-org/llama.cpp/blob/1f8baae28d096624bd47d33d2d179ec479d47720/ggml/src/ggml-quants.c)

The plan format `rsn.surgery-plan.v1` binds to `source_sha256`. For a numeric operation it names the exact tensor, shape and GGML type and supplies sorted, unique zero-based logical indices. The engine validates the source digest and descriptors before resolving any byte range. For a structural operation it supplies source-ordered routed layers and sorted original survivor IDs. Each layer may pick different experts while the architecture's global count requires a common survivor cardinality. Verification records the per-layer old-to-new map with removed IDs mapped to `null`.

The inventory includes offsets for inspection. The plan accepts logical indices and structural IDs. External tools can generate a reference plan with `plan`, submit the same schema to `validate`, and inspect `dry-run` before asking RustySpork to create a derivative. The input schema accepts one operation and up to 4,096 explicit numeric indices. The prototype uses bounded JSON, with chunked binary data appropriate for future large activation exports.

```powershell
.\rustyspork.cmd plan .\donor.gguf --operation probe --tensor blk.0.ffn_gate_exps.weight --indices 0,128,256 -o .\probe-plan.json
.\rustyspork.cmd probe .\donor.gguf .\probe-plan.json -o .\measurements.json
.\rustyspork.cmd plan .\donor.gguf --operation scoop --tensor blk.0.ffn_gate_inp.weight --indices 0,128 --scale 0 -o .\scoop-plan.json
.\rustyspork.cmd plan .\donor.gguf --operation scrape --tensor blk.0.ffn_gate_inp.weight --indices 0,128 --amplitude 0.1 --seed 42 -o .\scrape-plan.json
```

Probe and candidate verification are measured. Dry-run bytes are predictions until export. A compaction receipt contains file and tensor checksums, byte accounting, modified shapes and exact mappings. Runtime evaluation and dataset fields are `null` in mechanical receipts. The `arms` output defines unrun comparisons. Resident memory requires a separate runtime measurement under matched settings.

The default candidate limit is 64 MiB, with 1 MiB streamed I/O chunks and a cooperative 60-second deadline. Operation commands expose `--max-output-bytes`, `--chunk-bytes` and `--max-seconds`. GGUF metadata bounds apply independently. Validated candidates publish from a hidden `.partial` staging directory. A recorded failure leaves a reason without promoting that directory. A completed run reused for the same input is reverified. The separate `verify` command checks the candidate against the source and plan.

## Miniature mechanical and runtime results

Generated weights have known expert values. Both fixtures use two LFM2MoE blocks, one convolutional and one attention, with top-2 routing. They are small standard GGUF checkpoints readable by the independent llama.cpp loader.

| Source → candidate | Survivors per layer | Source bytes | Candidate bytes | File bytes saved |
| --- | --- | ---: | ---: | ---: |
| 4 → 2 | `0,2` | 3,678,080 | 3,028,864 | 649,216 |
| 8 → 4 | `0,2,4,6` | 4,976,512 | 3,678,080 | 1,298,432 |

| Checkpoint | SHA-256 |
| --- | --- |
| 4-expert donor and zero-removal candidate | `6497d9ed8112af46da57546745d560d9dafcfb253fbb0bec337bf9babe6606a2` |
| 2-expert physical candidate | `a5465db4c46b46ab19afbabb7d7611a5ecc77e156daabc7f57769e1809c34ca8` |
| 8-expert donor and zero-removal candidate | `7e2a91b9c891b638fbbb351be3c8886ace5d7d7fcdc5963a071c6be109ad6f30` |
| 4-expert physical candidate | `c3dd18c63394a6df93e18d3f729a5f310d457ffb06d14dfa0b1f9d971932b942` |

Windows `llama.cpp-win-x86_64-avx2@2.55.0`, reporting `0.6.0-dev` build 1 commit `fc9ce6b`, loaded all six files using CPU only. Each run evaluated the same four input token IDs and returned token ID 5 and five finite top log probabilities. The two no-op pairs matched both full-file SHA-256 and all reported scores. The first reported log probability was `-3.589961051940918` in the 4-expert donor and `-3.5899035930633545` in its candidate. The 8-expert pair yielded `-3.5905089378356934` and `-3.5904152393341064`. The result covers this prompt and runtime build.

### CPU runtime smoke

After running the 4-expert demo, select a local `llama-server.exe` with the native completion endpoint. This optional script uses only loopback HTTP and runs one request per checkpoint with one CPU worker thread and a 128-token context. It defaults to 64 MiB per file and 30 seconds per load and forward pass. Use `--max-model-bytes` and `--max-seconds` to set those limits explicitly. The script closes each temporary server before continuing.

```powershell
$Server = 'C:\path\to\llama-server.exe'
$CompactRun = Get-ChildItem .\rsn-demo\runs -Directory | Select-Object -First 1
.\rustyspork.cmd plan .\rsn-demo\donor.gguf --keep 0,1,2,3 -o .\rsn-demo\no-op-plan.json
.\rustyspork.cmd compact .\rsn-demo\donor.gguf .\rsn-demo\no-op-plan.json --output .\rsn-demo\no-op-runs
$NoOpRun = Get-ChildItem .\rsn-demo\no-op-runs -Directory | Select-Object -First 1
python scripts/runtime_smoke.py --server $Server .\rsn-demo\donor.gguf `
  (Join-Path $CompactRun.FullName 'candidate.gguf') --no-op (Join-Path $NoOpRun.FullName 'candidate.gguf')
```

The script emits a `rsn.runtime-smoke.v1` JSON receipt, including the runtime identification, prompt, result and per-file digests. The matching no-op check uses both the source bytes and the reported token scores. [llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

### Local trained GGUF round trip

The installed `sprocket-500m-chat-q4_k_m.gguf` passed a scale-1 Scoop on `output_norm.weight[0]`. Verification preserved the exact SHA-256 `edc2ac323a2513fa6f8e3e54adf120243de0f7a6b8cc11b349b3e6ad61494a09`, 310,279,360 bytes and all 236 tensors. The CPU runtime independently loaded the donor and output and returned token ID 1422 with identical top-five scores for the fixed probe. This qualifies identity serialization for one trained checkpoint.

```powershell
$Models = Join-Path $env:USERPROFILE '.lmstudio\models'
$Sprocket = Join-Path $Models 'kandivault\sprocket-500m\sprocket-500m-chat-q4_k_m.gguf'
.\rustyspork.cmd plan $Sprocket --operation scoop --tensor output_norm.weight --indices 0 --scale 1 -o .\sprocket-no-op.json
.\rustyspork.cmd compact $Sprocket .\sprocket-no-op.json --max-output-bytes 320000000 --output .\sprocket-runs
$SprocketRun = Get-ChildItem .\sprocket-runs -Directory | Select-Object -First 1
$RoundTrip = Join-Path $SprocketRun.FullName 'candidate.gguf'
.\rustyspork.cmd verify $Sprocket .\sprocket-no-op.json $RoundTrip
python scripts/runtime_smoke.py --server $Server --max-model-bytes 320000000 $Sprocket $RoundTrip
```

## Local source evidence

The two GGUF v3 files were inspected and hashed read-only from the supplied LM Studio model directory on 23 November 2026.

| Local checkpoint | Exact bytes | SHA-256 | Measured structure |
| --- | ---: | --- | --- |
| Ttimms `kat-reap50-Q6_K.gguf` | 15,260,151,680 | `0f174f7f27f2af145bd54229c29b344961f72a3f1a0f812dee60f2db8a0e64fa` | `qwen35moe`, 733 tensors, 40 routed blocks, 128 experts, top-8. F32 router `(2048,128)`, Q6_K gate/up `(2048,512,128)` and down `(512,2048,128)`. Shared experts have separate tensors. |
| DavidAU `LFM2.5-24B-A2B-TBrill-NEO-IQ4_XS.gguf` | 12,837,660,224 | `a1562db04b263e5406ac8412e07be6ea92941ea653653e9f678ab13f430e59b1` | `lfm2moe`, 428 tensors, 38 routed blocks after two dense blocks, 64 experts, top-4. F32 router `(2048,64)` and F32 expert bias `(64,)`, IQ4_XS gate/up `(2048,1536,64)` and down `(1536,2048,64)`. |

The installed KAT file already has 128 routed experts. Its ancestry and the specific pruning-run revision need to be proven before attributing comparisons to pruning. The target LFM donor is specifically NEO or MAX-NEO Q6_K. The installed IQ4_XS inventory establishes topology for those local bytes and leaves the Q6_K target's actual tensor types and bytes unresolved. Both production models remain on HOLD for compaction. The 3–5 GB target concerns actual recovered model memory under matched runtime and context settings. [KAT comparison plan](KAT_GREYHOUND_REAP_PLAN.md) · [LFM Q6 plan](LFM_EXACT_Q6_COMPACTOR.md) · [Qualification gates](QUALIFICATION_GATES.md)

One format diagnostic validated even original IDs `0,2,…,126` as 64 survivors in every routed layer of the 128-expert KAT reference. The validated plan predicted 6,627,000,320 fewer serialized bytes. The production exporter stayed on HOLD. This diagnostic has no candidate artifact, saliency basis or quality evidence and sits outside the proposed KAT comparison arms.

The architecture layouts were checked against pinned llama.cpp loader sources. [Qwen3.5 MoE](https://github.com/ggml-org/llama.cpp/blob/1f8baae28d096624bd47d33d2d179ec479d47720/src/models/qwen35moe.cpp) · [LFM2 MoE](https://github.com/ggml-org/llama.cpp/blob/1f8baae28d096624bd47d33d2d179ec479d47720/src/models/lfm2moe.cpp)
