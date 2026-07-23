---
name: e3-miu-gnn
description: Inspect, run, validate, and couple E3-miu-GNN atomistic checkpoints through the safe Python, ASE, JSON, evaluation, relaxation, and phonon interfaces. Use when an agent must work with E3-miu-GNN `.pt` or `.pth` checkpoints, predict an ASE-readable CIF/POSCAR/extXYZ structure, evaluate a Neo HDF5 dataset, run a fixed-cell relaxation or phonon calculation, export the ground-only SevenNet interface, or prepare deterministic model tasks for humans, scripts, workflow engines, or LLM tools.
---

# E3-miu-GNN Operations

Use the repository's public `e3mu` interface. Do not import private model classes
from `E3_miu_GNN.py` in generated task code.

## Operating Sequence

1. Locate the repository and activate its native environment when available.
2. Run `scripts/preflight_e3mu.py` for the current checkpoint and requested device.
3. Inspect the checkpoint before prediction. Read its recommended mode, element
   table, recommended outputs, and limitations.
4. Select one action: inspect, predict, relax, evaluate, phonon, or
   export-sevennet.
5. Create an `e3mu-task-v1` JSON request when deterministic or repeated execution
   is useful.
6. Run `scripts/run_e3mu_task.py` and retain its versioned JSON result.
7. Report the checkpoint digest, selected inference mode, units, convergence or
   evaluation status, and any unsupported capability.

## Preflight

Run from the skill directory or pass an explicit repository:

```bash
python scripts/preflight_e3mu.py \
  --repo /path/to/E3-miu-GNN \
  --checkpoint /path/to/model.pt \
  --device auto \
  --pretty
```

Stop before computation if the checkpoint cannot be safely reconstructed, its
element table does not cover the structure, or a required optional dependency is
missing.

## Deterministic Task Execution

```bash
python scripts/run_e3mu_task.py request.json \
  --repo /path/to/E3-miu-GNN \
  --output result.json \
  --pretty
```

Use `--validate-only` before a long phonon or evaluation task. Relative paths in
the task are resolved from the process working directory.

## Routing

- Read [references/API.md](references/API.md) for task fields and result schemas.
- Read [references/CAPABILITIES.md](references/CAPABILITIES.md) before selecting
  response, spin, stress, or SevenNet outputs.
- Read [references/WORKFLOWS.md](references/WORKFLOWS.md) for complete task-file
  examples.

## Safety and Scientific Boundaries

- Keep `model_mode=auto` unless the user explicitly requests a controlled
  ground-only/full-coupled comparison.
- Use only the manifest's `recommended` outputs for scientific conclusions.
- Never fabricate stress, virial, response, spin, convergence, or accuracy data.
- Do not perform cell relaxation because native virial stress is unavailable.
- Treat SevenNet TorchScript export as Layer-1 ground energy and forces only.
- Require `spin_policy=required` only when the structure supplies a spin state
  and the checkpoint enables a trained spin layer.
- Reject unknown elements instead of substituting another species.
- Keep unsafe pickle loading off. Enable it only for a trusted local legacy file
  after explicitly stating the code-execution risk.
- Preserve source structures and checkpoints. Write outputs to new paths.
- Do not claim production accuracy without an appropriate held-out evaluation.
