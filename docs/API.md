# Public API Reference

The `e3mu` package is the stable, headless entry point for using checkpoints
created by `E3_miu_GNN.py`. It follows the same practical pattern as atomistic
model packages such as SevenNet: load a checkpoint, attach an ASE calculator,
or call a command-line interface that emits structured results.

## Installation

From a source checkout:

```bash
python -m pip install -e .
```

Install optional integrations only when needed:

```bash
python -m pip install -e '.[physics,phonon]'
```

The existing `requirements.txt` remains the complete research environment.

## Checkpoint Inspection

Always inspect an unfamiliar checkpoint before inference:

```python
from e3mu import inspect_checkpoint

manifest = inspect_checkpoint("model.pt")
print(manifest["inference"]["recommended_mode"])
print(manifest["inference"]["outputs"]["recommended"])
```

`inspect_checkpoint` returns an `e3mu-model-manifest-v1` object containing:

- checkpoint format, schema version, SHA-256 digest, and size;
- model class, complete configuration, element table, cutoff, and parameter count;
- configured physics layers;
- recommended inference mode and the reason for that choice;
- potential, recommended, and unsupported outputs.

Safe state-dict checkpoints load with PyTorch `weights_only=True`. A trusted
legacy pickled module requires the explicit `allow_unsafe_legacy=True` opt-in.
The loader supports native mixed-granularity bundles, native dual-layer bundles,
and reconstructable state-dict bundles; future configuration fields are ignored
by older readers while known fields retain their typed values.

## ASE Calculator

`E3MUCalculator` is the primary coupling surface:

```python
from ase.io import read
from e3mu import E3MUCalculator

atoms = read("POSCAR")
atoms.calc = E3MUCalculator(
    "model.pt",
    device="auto",
    model_mode="auto",
    total_charge=0.0,
    electric_field=(0.0, 0.0, 0.0),
    spin_policy="auto",
)

energy_eV = atoms.get_potential_energy()
forces_eV_per_A = atoms.get_forces()
stress_eV_per_A3 = atoms.get_stress(voigt=True)  # fully periodic structures
```

Constructor parameters:

| Parameter | Values | Meaning |
| --- | --- | --- |
| `device` | `auto`, `cpu`, `mps`, `cuda` | Runtime backend. `auto` selects CUDA, then MPS, then CPU. |
| `model_mode` | `auto`, `ground_only`, `full_coupled` | Hamiltonian surface. `auto` follows checkpoint training metadata. |
| `total_charge` | finite float | Structure charge in elementary-charge units. |
| `electric_field` | three finite floats | External field in V/Angstrom. |
| `spin_policy` | `auto`, `off`, `required` | Read `e3mu_spins`, `spins`, or ASE initial magnetic moments. |
| `compile_inference` | boolean | Optional PyTorch compile path. It is disabled on MPS because Metal compilation can abort the process. |
| `allow_unsafe_legacy` | boolean | Permit pickle loading only for a trusted local legacy checkpoint. |

The calculator provides conservative energy, forces, and three-dimensional
Cauchy stress. Stress is computed as the symmetric homogeneous cell-strain
derivative of the same total energy, divided by the undeformed volume. The ASE
Voigt order is `xx, yy, zz, yz, xz, xy`, the unit is eV/Angstrom^3, and positive
values are tensile. Stress requires periodicity in all three directions and a
finite, nonsingular cell.

The output is technically available for every native checkpoint, but the
manifest recommends it only when training metadata records `w_stress > 0`.
Variable-cell relaxation is scientifically appropriate only after stress and
elastic response have passed a target-domain held-out validation; a finite
stress from an older energy/force-only checkpoint is not evidence of accuracy.

`stress_l1`, `stress_l2`, `stress_l3`, and `stress_dispersion` return full
3x3 tensors whose sum equals total stress. `piezoelectric` returns a 3x3x3
tensor in C/m$^2$. `magnetoelastic_stress` requires target spins plus an
`Atoms` array named `e3mu_reference_spins` or `reference_spins`; it returns the
full target-minus-reference stress tensor in eV/Angstrom$^3$.

## Prediction API

Use `predict` with an ASE `Atoms` object:

```python
from e3mu import predict

result = predict(
    "model.pt",
    atoms,
    properties=("energy", "forces", "stress", "dipole", "polarizability"),
)
```

Use `predict_file` for any ASE-readable structure file:

```python
from e3mu import predict_file

result = predict_file("model.pt", "structures.extxyz", index=":")
```

A single structure returns `e3mu-prediction-v1`. Multiple structures return
`e3mu-prediction-batch-v1`. Arrays are converted to ordinary JSON arrays, and
NaN or infinity causes an error instead of invalid JSON.

Supported result names include:

- native conservative derivatives: `energy`, `forces`, periodic `stress`, and
  `stress_l1`, `stress_l2`, `stress_l3`, `stress_dispersion`;
- response branch: `dipole`, `polarizability`, `bec`, `piezoelectric`;
- enabled domain layers: `charges`, `atomic_dipoles`,
  `atomic_polarizability`, `c6`;
- enabled spin layer: `Jij`, `Di`, `DMIij`, `magnetic_moments`,
  `effective_field`, and paired `magnetoelastic_stress`.

The manifest's `recommended` list is authoritative for scientific use. A base
checkpoint may contain initialized response modules without having trained
them; those outputs remain technically computable but are not recommended.

## Fixed-Cell Relaxation

```python
from e3mu import relax

report = relax(
    "model.pt",
    atoms,
    optimizer="FIRE",
    fmax=0.05,
    steps=200,
    trajectory="relax.traj",
    output_structure="relaxed.extxyz",
)
```

`relax` optimizes positions at a fixed cell with FIRE or BFGS. The result records
convergence, final energy, maximum force, final geometry, and calculator
metadata. For cell optimization, use the ASE calculator directly with an ASE
cell filter and only a checkpoint whose manifest recommends stress; the public
`relax` helper intentionally remains fixed-cell.

## Dataset Evaluation

```python
from e3mu import evaluate

metrics = evaluate(
    "model.pt",
    "neo_standard_l1_l2_l3.h5",
    split="test",
    batch_size=4,
    device="auto",
)
```

This calls the canonical mask-aware evaluator in `E3_miu_GNN.py`. HDF5 inputs
retain their streaming behavior.

## Phonon Workflow

```python
from e3mu import run_phonon

result = run_phonon(
    "model.pt",
    "POSCAR",
    device="auto",
    model_mode="auto",
    supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    dos_mesh=(10, 10, 10),
    thermal_temperatures_K=None,
)
```

This is a lazy bridge to the verified finite-displacement implementation in
`Verify_Program_Phonon.py`. The phonon workflow and direct ASE use share the
same public calculator.

## SevenNet-Compatible Export

```python
from e3mu import export_sevennet

report = export_sevennet("model.pt")
```

The export is intentionally scoped to Layer-1 ground energy and forces. It is
useful for SevenNet-style TorchScript consumers, but it does not serialize QEq,
PME, DEQ polarization, D4, spin, FiLM feedback, response tensors, or stress.
The export report states this scope explicitly.

## Task API

`execute_task` accepts the same `e3mu-task-v1` mapping used by LLM and workflow
scripts:

```python
from e3mu import execute_task

result = execute_task({
    "schema": "e3mu-task-v1",
    "action": "predict",
    "checkpoint": "model.pt",
    "structure": "POSCAR",
    "options": {"device": "cpu", "properties": ["energy", "forces"]},
})
```

Schemas are stored in [`coupling/`](../coupling/README.md).
