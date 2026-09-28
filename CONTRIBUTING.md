# Contributing

Contributions are welcome when they preserve the physical and provenance
contracts of the project.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
python E3_miu_GNN.py self-test
```

## Required checks

Before opening a pull request:

1. Run `pytest -q` and the deterministic `self-test` command.
2. Add focused tests for changed symmetry, solver, dataset, or checkpoint logic.
3. Keep executable implementation changes in `E3_miu_GNN.py`.
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

By intentionally submitting original source code or documentation for
inclusion, you offer that contribution under the project's
[E3-miu-GNN Research and Non-Commercial License v1.0](LICENSE), unless a
separate written agreement specifies otherwise. Confirm that you have the
right to grant that license, including any necessary employer or institutional
permission. You retain your copyright; submission is not a copyright transfer
or permission for maintainers to commercially relicense your work.

Preserve the project attribution to Yufeng Zhan and Guohua Hu, existing
copyright notices, and accurate credits for your changes. The project is
source-available, not GNU GPL or OSI open source. Its non-commercial terms
cannot be imposed on a GPL-covered combination. Check copied code and linked
dependencies, including PyQt6, before contributing; see [NOTICE.md](NOTICE.md).

Dataset records, models, and third-party material retain their upstream terms;
do not label them with the project license unless the relevant rights permit
it. Historical MIT grants remain valid for the material they cover. Read the
[licensing rationale](docs/LICENSE_RATIONALE.md) for scope and migration details.
