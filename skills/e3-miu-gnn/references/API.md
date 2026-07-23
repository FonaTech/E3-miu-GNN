# Task and Result API

## Task Envelope

Every deterministic request uses:

```json
{
  "schema": "e3mu-task-v1",
  "action": "inspect",
  "checkpoint": "/absolute/or/working-directory-relative/model.pt",
  "options": {}
}
```

Valid actions and required paths:

| Action | Required fields | Common options |
| --- | --- | --- |
| `inspect` | `checkpoint` | `allow_unsafe_legacy` |
| `predict` | `checkpoint`, `structure` | `index`, `device`, `model_mode`, `properties`, `total_charge`, `electric_field`, `spin_policy` |
| `relax` | `checkpoint`, `structure` | `device`, `model_mode`, `fmax`, `steps`, `optimizer`, `trajectory`, `output_structure` |
| `evaluate` | `checkpoint`, `dataset` | `split`, `batch_size`, `device`, `output_json` |
| `phonon` | `checkpoint`, `structure` | `device`, `model_mode`, `supercell_matrix`, `dos_mesh`, `displacement_amplitude_A`, `band_npoints` |
| `export_sevennet` | `checkpoint` | `output` |

## Prediction Properties

Start with `energy` and `forces`. Add properties only when checkpoint inspection
lists them as recommended:

```json
"properties": ["energy", "forces", "dipole", "polarizability"]
```

Possible native names are `energy`, `forces`, `dipole`, `polarizability`, `bec`,
`charges`, `atomic_dipoles`, `atomic_polarizability`, `c6`, `Jij`, `Di`,
`DMIij`, `magnetic_moments`, and `effective_field`.

`stress` and `virial` are rejected.

## Inference Options

- `device`: `auto`, `cpu`, `mps`, or `cuda`.
- `model_mode`: `auto`, `ground_only`, or `full_coupled`.
- `total_charge`: finite charge in elementary-charge units.
- `electric_field`: three finite values in V/Angstrom.
- `spin_policy`: `auto`, `off`, or `required`.
- `index`: ASE index expression such as `0` or `:`.
- `allow_unsafe_legacy`: false unless the user trusts the local pickle source.

## Result Schemas

- checkpoint inspection: `e3mu-model-manifest-v1`
- single prediction: `e3mu-prediction-v1`
- batch prediction: `e3mu-prediction-batch-v1`
- relaxation: `e3mu-relaxation-v1`
- SevenNet export: `e3mu-sevennet-export-v1`
- runtime failure: `e3mu-error-v1`

The repository schemas are in `coupling/`. Results contain ordinary JSON values;
NaN and infinity are rejected.

## Direct CLI

The task runner calls the same API as:

```bash
python -m e3mu --pretty inspect model.pt
python -m e3mu predict model.pt POSCAR --properties energy,forces
python -m e3mu run-task request.json --output result.json
```
