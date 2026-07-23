# Capability Rules

## Checkpoint Stage

The public API resolves `model_mode=auto` from checkpoint evidence:

1. `recommended_inference_mode` metadata;
2. `training_mode=base`, `response`, or `joint`;
3. a legacy base-checkpoint filename;
4. response loss weights;
5. complete-model behavior if no stage evidence exists.

Base checkpoints recommend `ground_only`. Response and joint checkpoints
recommend `full_coupled`.

## Physics Mapping

| Configuration | Additional potential outputs | Required context |
| --- | --- | --- |
| QEq or PME | charges and electrostatic energy diagnostics | total charge; periodic cell for PME |
| DEQ polarization | induced/atomic dipoles and atomic polarizability | trained response labels |
| D4 | C6 and dispersion energy | installed D4 dependency |
| Spin | exchange, anisotropy, magnetic moments, effective field | supplied spin state for spin energy |
| DMI | DMI vectors | spin, parity, and DMI supervision |
| FiLM | coupled feedback in total energy | full-coupled mode |

Configured means the module exists. Recommended means checkpoint metadata shows
the relevant training surface. Do not equate the two.

## Always Unsupported

- native virial stress;
- stress-driven cell relaxation;
- a full Layer-2/Layer-3 SevenNet export;
- verified LAMMPS deployment from the current TorchScript artifact.

## Device Rules

- `auto` selects CUDA, then MPS, then CPU.
- MPS uses float32 when a checkpoint requests float64.
- `torch.compile` is disabled on MPS because Metal compilation can abort the
  process instead of raising a recoverable Python exception.
- Safe checkpoints load on CPU before being moved to the selected device.

## Units

- energy: eV
- forces: eV/Angstrom
- positions and cell: Angstrom
- electric field: V/Angstrom
- dipole: elementary charge times Angstrom
- polarizability: checkpoint `polarizability_unit`, normally Angstrom cubed

## Accuracy Claims

Inference being finite is a functional check, not an accuracy benchmark. Use a
held-out canonical dataset with `evaluate` and report its source, split, masks,
checkpoint digest, and metrics before discussing accuracy.
