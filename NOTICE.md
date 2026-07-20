# Licensing and Release Notice

The root [MIT license](LICENSE) applies to the original source code and
original project documentation in this repository unless a file states
otherwise. It does **not** relicense datasets, model checkpoints, third-party
software, or third-party scientific results.

## Dataset boundary

The Neo corpus is an aggregate of independently licensed sources. Its public
metadata must use `license: other`, and every upstream license and attribution
continues to apply. In particular:

- MPtrj is distributed under MIT terms.
- JARVIS-DFT, QM7-X, SO3LR, and SCFNN data are distributed under CC BY 4.0.
- The identifiable DeepSPIN component is subject to GPL-3.0 conditions.
- The transformed `BEC/H2O`, `BEC/MAPbI3`, and `BEC/dimer` records remain
  blocked from public redistribution until archive-level rights are confirmed.

See [Datasets/Neo/LICENSES_AND_ATTRIBUTION.md](Datasets/Neo/LICENSES_AND_ATTRIBUTION.md)
and [docs/DATASETS.md](docs/DATASETS.md) before publishing any dataset binary.

## Files intentionally excluded from the software repository

Canonical HDF5 corpora, raw source archives, extXYZ aggregations, checkpoints,
VASP outputs, POTCAR files, Hugging Face staging binaries, and local validation
artifacts are not part of the MIT-licensed source release. The root
`.gitignore` is a publication guard, not a statement that excluded files have
no scientific value.

## Dependencies

Runtime libraries are imported as external dependencies and retain their own
licenses. No dependency is relicensed by this repository. Review dependency
terms before producing a redistributed binary environment.

## Research status

This is research software. Symmetry, differentiability, data integrity, and
small validation workflows are tested, but the repository does not claim a
converged universal interatomic potential or production accuracy for every
chemical, electric, and magnetic domain.
