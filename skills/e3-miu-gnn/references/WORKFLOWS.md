# Workflow Examples

## Inspect

```json
{
  "schema": "e3mu-task-v1",
  "action": "inspect",
  "checkpoint": "model.pt",
  "options": {}
}
```

## Predict Energy and Forces

```json
{
  "schema": "e3mu-task-v1",
  "action": "predict",
  "checkpoint": "model.pt",
  "structure": "POSCAR",
  "options": {
    "device": "auto",
    "model_mode": "auto",
    "properties": ["energy", "forces"],
    "total_charge": 0.0,
    "electric_field": [0.0, 0.0, 0.0],
    "spin_policy": "auto"
  }
}
```

## Predict a Multi-Frame extXYZ

Use `index=":"`:

```json
{
  "schema": "e3mu-task-v1",
  "action": "predict",
  "checkpoint": "model.pt",
  "structure": "structures.extxyz",
  "options": {
    "index": ":",
    "device": "cpu",
    "properties": ["energy", "forces"]
  }
}
```

## Fixed-Cell Relaxation

```json
{
  "schema": "e3mu-task-v1",
  "action": "relax",
  "checkpoint": "model.pt",
  "structure": "input.cif",
  "options": {
    "device": "auto",
    "model_mode": "auto",
    "optimizer": "FIRE",
    "fmax": 0.05,
    "steps": 200,
    "trajectory": "relax.traj",
    "output_structure": "relaxed.extxyz"
  }
}
```

## Held-Out Evaluation

```json
{
  "schema": "e3mu-task-v1",
  "action": "evaluate",
  "checkpoint": "model.pt",
  "dataset": "neo_standard_l1_l2_l3.h5",
  "options": {
    "split": "test",
    "batch_size": 4,
    "device": "auto",
    "output_json": "test_metrics.json"
  }
}
```

## Phonon Calculation

```json
{
  "schema": "e3mu-task-v1",
  "action": "phonon",
  "checkpoint": "model.pt",
  "structure": "POSCAR",
  "options": {
    "device": "auto",
    "model_mode": "auto",
    "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
    "dos_mesh": [10, 10, 10],
    "displacement_amplitude_A": 0.01,
    "band_npoints": 101,
    "thermal_temperatures_K": null
  }
}
```

## SevenNet Ground Export

```json
{
  "schema": "e3mu-task-v1",
  "action": "export_sevennet",
  "checkpoint": "model.pt",
  "options": {
    "output": "model_ground_compat_sevennet.pt"
  }
}
```
