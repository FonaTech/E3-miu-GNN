# Contributing

Contributions are welcome when they preserve the physical and provenance
contracts of the project.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
python Dual_Layer_Atomic_E3_GNN.py self-test
```

## Required checks

Before opening a pull request:

1. Run `pytest -q` and the deterministic `self-test` command.
2. Add focused tests for changed symmetry, solver, dataset, or checkpoint logic.
3. Keep executable implementation changes in `Dual_Layer_Atomic_E3_GNN.py`.
4. Do not commit raw datasets, HDF5 corpora, checkpoints, VASP outputs, POTCAR
   files, local absolute paths, credentials, proxy settings, or generated
   training artifacts.
5. Do not synthesize missing physical labels. Extend the canonical HDF5 masks
   and provenance records when introducing a new target or source.
6. Document units, electronic-structure reference, grouping key, split policy,
   source checksum, license, and redistribution status for every new dataset.

## Scientific changes

A new energy term must document its Hamiltonian contribution and demonstrate
conservative forces when positions are differentiated. A new tensor or vector
output must include the relevant rotation, reflection, parity, or time-reversal
test. A new long-range solver must report convergence and stability residuals.

## Licensing

Contributions to original source code and documentation are accepted under the
project's MIT license. Dataset records and copied third-party material retain
their upstream terms; do not submit them under MIT unless you own the rights to
do so.
