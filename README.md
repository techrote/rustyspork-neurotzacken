# RustySpork Neurotzacken

RustySpork is a headless model-surgery laboratory. It inventories local GGUFs, binds plans to exact donor bytes, repacks supported expert tensors, and verifies the resulting checkpoints. The first implementation includes a CPU-only demonstration with generated miniature Mixture-of-Experts models. See the [capability and verification record](docs/IMPLEMENTATION_STATUS.md).

## Windows quick start

Use Python 3.11 or newer and run from the repository checkout in Command Prompt or PowerShell. The laboratory uses the Python standard library. The demonstration generates its own weights and runs without CUDA or network access.

```powershell
.\rustyspork.cmd --help
.\rustyspork.cmd demo --output .\rsn-demo --experts 4
```

Choose a new output directory. Open `inventory.json` for the original tensors and expert layout, `plan.json` for the survivor IDs, `dry-run.json` for expected storage changes, and `report.json` for the run result. The verified derivative, exact plan and receipt are under `runs/<run-id>/`. Use `--experts 8` in a second output directory to exercise the larger fixture.

To run each stage explicitly in PowerShell:

```powershell
.\rustyspork.cmd fixture --experts 4 -o .\donor.gguf
.\rustyspork.cmd inventory .\donor.gguf -o .\inventory.json
.\rustyspork.cmd plan .\donor.gguf --keep 0,2 -o .\plan.json
.\rustyspork.cmd validate .\donor.gguf .\plan.json
.\rustyspork.cmd dry-run .\donor.gguf .\plan.json
.\rustyspork.cmd compact .\donor.gguf .\plan.json --output .\runs
$Run = Get-ChildItem .\runs -Directory | Select-Object -First 1
.\rustyspork.cmd verify .\donor.gguf .\plan.json (Join-Path $Run.FullName 'candidate.gguf')
.\rustyspork.cmd report $Run.FullName
```

Use `python -m unittest discover -s tests` to run the mechanical tests. For a separate CPU loader and forward-pass check using an installed `llama-server`, see the [runtime smoke instructions](docs/IMPLEMENTATION_STATUS.md#cpu-runtime-smoke).

## Inspect a local model

```powershell
$Models = Join-Path $env:USERPROFILE '.lmstudio\models'
$Kat = Join-Path $Models 'Ttimms\KAT-Coder-V2.5-Dev-REAP-50-GGUF\kat-reap50-Q6_K.gguf'
.\rustyspork.cmd inventory $Kat --header-only -o .\kat-layout.json
.\rustyspork.cmd inventory $Kat -o .\kat-source.json
.\rustyspork.cmd arms
```

The first inventory reads metadata and tensor descriptors. The full inventory streams the file to compute SHA-256. Both use CPU only. `arms` describes proposed KAT and LFM comparison cases. See [local source evidence](docs/IMPLEMENTATION_STATUS.md#local-source-evidence) for the specimen findings and prerequisites for model-scale experiments.

## Repository boundary

RustySpork owns model computation, canonical tensor coordinates, validation of exact selections, checkpoint transformations, instrumentation, evidence and experiment control. External tools can consume versioned JSON output or submit declarative selections for RustySpork to validate against a pinned donor.

[Xenovection](https://github.com/techrote/Xenovection) owns graphical presentation, tensor atlases, visual interaction, shaders and artistic synthesis. The projects run independently. The [FRANKENFIELD design](docs/FRANKENFIELD_INTERFACE.md) and its [interface-to-tensor contract](docs/FRANKENFIELD_IMPLEMENTATION_CONTRACT.md) remain as historical references for the scientific requirements.

## Research programme

- **KAT Greyhound/REAP.** The local Ttimms KAT-Coder-V2.5-Dev REAP-50 Q6_K is the compact reference. The 224, 192 and 160 survivor arms belong to a common 256-expert parent and matched quantization workflow. The first stock experiments favour Q4. The Q8_0 follow-on remains deferred. [KAT plan](docs/KAT_GREYHOUND_REAP_PLAN.md) · [RSN-KAT-003](https://github.com/techrote/rustyspork-neurotzacken/issues/3)
- **Source-exact LFM2.5.** The selected target is DavidAU LFM2.5-24B-A2B Turbo-Brilliance-50X NEO/MAX-NEO Imatrix Q6_K. The proposed 8, 12 and 16 removals per 64 experts target 3–5 GB of measured model-memory recovery pending loader, behaviour and memory qualification. [LFM plan](docs/LFM_EXACT_Q6_COMPACTOR.md) · [Model specimens](docs/MODEL_SPECIMENS.md)
- **Reblitteration.** Frozen-feature upstream recruitment is an independent causal research track with explicit tests for appropriate engagement and non-engagement. [RSN-001](https://github.com/techrote/rustyspork-neurotzacken/issues/1)

The [roadmap](docs/RESEARCH_ROADMAP.md), [operation contracts](docs/SURGERY_OPERATIONS.md), [laboratory blueprint](docs/DETERMINISTIC_LAB.md), [experiment protocol](docs/EXPERIMENT_PROTOCOL.md) and [qualification gates](docs/QUALIFICATION_GATES.md) set out subsequent research. The [decision record](docs/RECONCILIATION_AND_DECISIONS.md), [quantization notes](docs/REAP_AND_QUANTIZATION.md) and [references](docs/REFERENCES.md) preserve the original design and evidence.
