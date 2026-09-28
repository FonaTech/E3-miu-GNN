# Licensing and Release Notice

**E3-miu-GNN, by Yufeng Zhan and Guohua Hu.**

Copyright (c) 2026 Yufeng Zhan (Fona). Existing third-party and contributor
copyright notices continue to apply. Author attribution does not assign
copyright or establish ownership of another person's contribution.

## Current license and scope

Original project source code and original project documentation in this
release are offered under the [E3-miu-GNN Research and Non-Commercial License,
version 1.0](LICENSE), unless a file-specific or third-party notice applies.
License reference: `LicenseRef-E3-miu-GNN-Research-NonCommercial-1.0`.
This is a custom reference, not an SPDX-listed or OSI-approved license.

The project is **source-available, not GNU GPL or OSI open-source software**.
Qualifying non-commercial research, education, study, evaluation, modification,
and sharing are permitted without a license fee. Commercial use requires
prior written authorization from the relevant rightsholders. Enterprise
internal R&D, commercial model training, hosted commercial services, paid
consulting, and software sales are included in the commercial-use definition,
even when no program copy is distributed.

Preserve the attribution to **Yufeng Zhan and Guohua Hu** when distributing
copies or substantial portions. Modified distributions must identify changes,
provide complete corresponding source, and use the same license. See the
operative [LICENSE](LICENSE) for the complete definitions and conditions;
publication citation is requested, not an additional license condition.

The license does **not** relicense datasets, model checkpoints, third-party
software, or independent scientific results. It grants rights only to the
extent each licensor is authorized to grant them.

## Commercial authorization

To request the appropriate licensing contact, use the maintainers' published
contact details on the [project repository](https://github.com/FonaTech/E3-miu-GNN)
or open a [licensing inquiry](https://github.com/FonaTech/E3-miu-GNN/issues).
Identify the intended user or organization, version, use, distribution or
hosting model, and third-party components. Keep confidential information out
of public issues. A request, silence, or an informal discussion is not
authorization; obtain an express written agreement from all relevant
rightsholders or their authorized representatives before commercial use.
No commercial agreement or third-party permission is supplied by this notice.

## Historical MIT grants

The local license declaration changed on 17 September 2026. The previous
MIT text and copyright notice are preserved verbatim in
[MIT-HISTORICAL.txt](docs/licensing/MIT-HISTORICAL.txt). Valid earlier MIT
grants and rights in material lawfully available under those grants are not
revoked. The historical file is not a new offer to license all later code
under MIT. See [the history record](docs/licensing/README.md) and the
[Chinese licensing rationale](docs/LICENSE_RATIONALE.md).

## Dataset boundary

The Neo corpus is an aggregate of independently licensed sources. Its public
metadata must use `license: other`, and every upstream license and attribution
continues to apply. In particular:

- MPtrj is distributed under MIT terms.
- JARVIS-DFT, QM7-X, SO3LR, and SCFNN data are distributed under CC BY 4.0.
- The identifiable DeepSPIN component is subject to GPL-3.0 conditions.
- The transformed `BEC/H2O`, `BEC/MAPbI3`, and `BEC/dimer` records remain
  blocked from public redistribution until archive-level rights are confirmed.

These are existing project provenance records, not a fresh archive-level
rights audit. See [docs/DATASETS.md](docs/DATASETS.md) and the upstream notices
for the exact data version before publishing any dataset binary. The older
`Datasets/Neo/LICENSES_AND_ATTRIBUTION.md` ledger referenced by project
documentation is not included in this local snapshot.

## Data, checkpoints, and output files

This snapshot contains `datasets/neo_tiny_l1_l2_l3.h5`, model checkpoints, and
results under `models/`. Their presence does not grant redistribution rights
or place them under the root software license. Raw source archives, HDF5
corpora, checkpoints, VASP outputs, POTCAR files, and other external material
require their own applicable rights and notices. No new permission for those
materials is granted here.

## Dependencies

Runtime libraries retain their own licenses. External imports do not
automatically exempt a combined work from those licenses.

In particular, [PyQt6 is offered under GPLv3 or Riverbank commercial terms,
not LGPL](https://www.riverbankcomputing.com/software/pyqt/intro). A combination
subject to GPL cannot also be distributed with this project's non-commercial
restriction. Resolve the applicable permissions or replace incompatible
components before distributing such a combination. This change does not
establish that the current GUI or a bundled environment has been cleared for
distribution. A project commercial agreement cannot grant Riverbank's rights.

## Research status

This is research software. Symmetry, differentiability, data integrity, and
small validation workflows are tested, but the repository does not claim a
converged universal interatomic potential or production accuracy for every
chemical, electric, and magnetic domain.
