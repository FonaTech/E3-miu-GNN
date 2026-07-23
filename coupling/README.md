# Coupling Contracts

This directory contains versioned, machine-readable contracts for connecting
E3-miu-GNN to ASE, workflow engines, services, and LLM tool runtimes.

- `model_manifest.schema.json` validates `e3mu inspect` output.
- `prediction_result.schema.json` validates single-structure predictions.
- `task_request.schema.json` defines the deterministic task-file interface.
- `llm_tools.json` provides function declarations suitable for an LLM tool layer.
- `ase_example.py` is the minimal direct ASE integration.
- `task_example.json` can be executed with `e3mu run-task`.

Run the examples from the repository root:

```bash
python coupling/ase_example.py model_base.pt mp-2998_BaTiO3.cif --device cpu
python -m e3mu run-task coupling/task_example.json --output prediction.json
```

The native API does not advertise stress because the current implementation
does not derive virial stress. The SevenNet TorchScript export contains only the
Layer-1 ground energy and forces; it is not a serialization of the coupled
Layer-2/Layer-3 Hamiltonian.
