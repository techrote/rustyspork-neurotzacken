# Rusty Spoon operations — semantic, numeric and structural contracts

**Research design, not a library of implemented operations.** Every edit is a declarative transform against a pinned, immutable donor. Visual brush strokes only become model edits after resolution to exact tensor coordinates and explicit validation.

## Operation taxonomy and expected physical effect

| Tool | Object edited | Intended transform | Disk / VRAM reduction? | Main hazard |
| --- | --- | --- | --- | --- |
| **Probe** | None | Read a finite set/statistical tile | No | Mistaking correlation for causal importance |
| **Scoop** | Selected numeric weights | Multiply by scalar \`a\` in [0,1], zero at \`a=0\` | No unless subsequently compressed with a compatible sparse layout | Quant re-encode may affect *nonselected* weights in a block |
| **Scrape** | Selected weights | Seeded bounded noise, clamp, sign-aware rescaling | No | Large outliers / distribution shift |
| **Fracture** | Compatible rows/channels | Explicit one-to-one permutation | Usually no | Interface semantics and shape mismatch |
| **Gouge** | Whole experts / heads / compatible structural units | Physically remove and repack, update graph/router | **Yes, if actually exported and loaded** | Dangling indexes, KV/conv state and unsupported runtime |
| **REAP** | MoE routed experts | Score via router-weighted output activations; remove selected experts | **Yes, after source-exact compaction** | Rare specialists lost; biased calibration |
| **Greyhound** | Executed block graph | Skip/bypass selected layers | No by itself | Changed recurrence/state and behavioural collapse |
| **Branchwitch** | Executed block graph | Repeat or reorder compatible layer calls | No by itself | Reusing/duplicating state and direction-of-information changes |
| **Graft / Stitch** | Shape- and semantics-compatible tensor region | Replace or blend from separately hashed donor | Not necessarily | Tokenizer, representation alignment, unknown ability |
| **Funnel / reblitteration** | Upstream low-rank weight delta | Increase appropriate conditional recruitment of an existing frozen feature | Typically no | False positives, context-insensitive activation |

**Runtime graph surgery is not physical checkpoint compression.** The DRIV3R Greyhound variant that executes fewer layers did not free the stored skipped layer tensors. For the owner's current 24 GiB memory problem, prioritize actual MoE expert removal, then compare equal-size mixed-precision controls.

## Patch algebra and custody

A patch plan specifies: donor sha256; exact tensor name/shape/quant type; sorted canonical index mask or whole-structural-unit identifier; transform type; parameters; deterministic seed; source/existing quant decoder; expected changed tensors; strict parent revision; and a predicted bytes/requantization report.

An operation is **purely specified** until the backend performs a dry run. Surviving GGUF tensor and router indices after previous structural edits are not the same identifiers as original expert IDs; patches must bind to the named graph revision and expose explicit source→target coordinate mappings. Reject conflicting operations unless an ordered composition is recorded.

For numeric edits, quant-block coupling is fundamental: changing one logical weight may require re-encoding a block of 256 or another format-dependent size, changing untouched neighbours numerically. Thus:
- display *requested selected elements* and *actual re-encoded elements/blocks* separately;
- preserve original quant bytes for unaffected aligned blocks whenever proven safe;
- hash, decode and compare all advertised unaffected data, with tolerance/exception receipts;
- measure file size separately from resident VRAM; zeros alone rarely make dense GGUF tensors smaller.

**No-op control:** applying the identity transform produces an unchanged value-domain model and source file untouched; if output metadata/byte layout changes, record it and establish functional equivalence before enabling destructive operations.

## Structured expert surgery (primary model-size path)

Each MoE layer needs a verified input expert count \`E\`, active-per-token routing \`k\`, list of unique survivor IDs, mapping \`old→new\`, actual tensor layout, router/gate and expert-bias matrices, quant-block alignment, and loader-support proof. KAT REAP-50 demonstrates 256→128 on one architecture. For LFM2.5 the expected 64 routed experts and top-4 are **subject to the target GGUF's actual metadata**.

Calibration should include both frequency and activation/output-based scoring, prompt and mode strata, long-tail expert hits and uncertainty. Compare one-step deletion with optional mixed-precision/no-prune at **equal measured file bytes**, not just a stated quant label. Freeze scoring before examining held-out evaluations.

Where whole-expert blocks are contiguous and aligned, prefer source-exact copying. Where not, use a verified bounded decode/requantize pathway and report which untouched logical weights changed. A compactor must not be allowed to silently reinterpret fused matrix axes or remove an expert which the graph/router still addresses.

## Greyhound and Branchwitch research (secondary)

Historical layer bypass, duplicate and reorder studies are useful **mechanistic perturbations**. They are not automatically portable to models mixing attention with Gated-DeltaNet, convolution or other recurrent state. Each architecture adapter must model:
- layer input/output type and residual dimension;
- attention and recurrent/conv cache state ownership and reset semantics;
- whether an execution call may repeat/reorder without corrupting sequence state;
- exact runtime graph deployed (not merely a README saying a layer is absent);
- actual compute and latency after warmup, with paired behavioural quality and degeneration checks.

Physically removing a layer is a distinct exporter operation: remove its weights and change graph metadata/load paths. A runtime skip baseline cannot be reported as a checkpoint-size improvement.

## Reblitteration (Funnel) — frozen-feature upstream recruitment

The owner-defined hypothesis is **not** "modify the good neuron" or "force the desired neuron to activate more." A target that already produces desirable engagements and non-engagements remains frozen. Alter *upstream* weights so that it is recruited **more often in appropriate missed-opportunity contexts**, while retaining correct suppression in negatives.

1. Establish a target that is causally relevant to a measured desirable behaviour, possibly a distributed residual direction, SAE/transcoder feature or validated circuit—not necessarily a literal neuron.
2. Build disjoint positive, missed-opportunity, appropriate non-engagement and hard-negative groups. Identify target activity and actual downstream task quality for each.
3. Freeze detector and target-circuit weights by hash and allowlist. Perform small rank-1–4 upstream updates: \`W' = W + αuvᵀ\` is only an **example**, not a guarantee of functional direction.
4. Optimize conditional recruitment on missed opportunities while penalizing incorrect activation on negatives, drift on already-correct cases, output KL, large update norms and unrelated regressions. Avoid constant activation offsets that simply change all contexts.
5. Compare unchanged donor, random-matched edit, naive scale, inference-time steering and parameter-matched LoRA. Use causal patching/ablation or equivalent interventions to distinguish genuine target mediation from coincident style shifts.
6. Show a **Funnel** overlay only after measured feature validity; provenance should include discovery/causal-test run, condition coverage and confidence. Display "unknown" otherwise.

This is governed by [issue RSN-001](https://github.com/techrote/rustyspork-neurotzacken/issues/1), initially on a tractable ~1–4B checkpoint. A 24B graph atlas can later host its results, but the 24B primary expert-pruning objective **does not depend on completing RSN-001**.

## Failure and refusal cases

An operation must be refused, not improvised, if any donor SHA/architecture mismatch, invalid tensor selection, unverified quant packing, inability to update router bias, missing target-runtime support, assumed layer-state compatibility, unresolved license restrictions, absent causal target for Funnel, or resource limit violation occurs. Failures and null results are legitimate outputs, recorded alongside successful mutants.

**Decision rule:** a visually impressive mutation without an exact numerical receipt is a demonstration effect, **not a completed experiment**. A smaller file that fails load or destroys useful behaviour is a failed compression candidate, not an optimized model.

See [interface contract](FRANKENFIELD_IMPLEMENTATION_CONTRACT.md), [LFM compactor](LFM_EXACT_Q6_COMPACTOR.md), [deterministic orchestration](DETERMINISTIC_LAB.md), and [evaluation gates](QUALIFICATION_GATES.md).
