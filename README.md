# Mixed-Granularity E(3)-mu-GNN

An E(3)-equivariant atomistic graph neural network that couples local chemical
interactions, differentiable electrostatic and polarization physics, and a
time-reversal-aware spin Hamiltonian in one trainable model.

> **Research status.** The three-layer architecture, canonical data pipeline,
> training system, PyQt6 interface, and deterministic physics tests are
> implemented. Reported dataset and short-run metrics are functional validation,
> not a claim of a converged universal interatomic potential.

![Mixed-granularity architecture](docs/assets/proposal/mixed-granularity-core.png)

*Figure 1. Completed E(3)-mu-GNN portion of the research-proposal diagram. The
crop excludes the proposal's unimplemented agent, reinforcement-learning, and
LoRA workflow.*

## What is implemented

- **Layer 1 - local atomic representation:** scalar, polar-vector,
  axial-vector, symmetric-traceless $L=2$, and optional $L=3$ channels with
  explicit O(3) parity handling.
- **Layer 2 - domain response:** constrained differentiable QEq, periodic
  Ewald/PME through `torch-pme`, Thole-damped self-consistent polarization,
  molecular DFT-D4, dipoles, polarizabilities, charges, C6, and Born effective
  charges.
- **Layer 3 - magnetic response:** geometry-conditioned Heisenberg exchange,
  traceless single-ion anisotropy, optional Dzyaloshinskii-Moriya interaction,
  magnetic moments, and effective spin fields.
- **Cross-granularity feedback:** charge, electrostatic potential, and spin
  invariants modulate atomic message passing through bounded FiLM gates.
- **Training and evaluation:** mask-aware mixed-label losses, group-safe fixed
  splits, staged base/response/joint training, normalized multi-task model
  selection, live plots, memory diagnostics, safe checkpoints, and a
  dataset-aware Auto Research search space.
- **Data tooling:** canonical ragged HDF5, deterministic tier construction,
  strict validation, provenance records, source-specific masking, and
  rights-aware Hugging Face staging.

## Effective Hamiltonian

The implemented model assembles

$$
E_{\mathrm{tot}} =
E_{\mathrm{short}} + E_{\mathrm{QEq}} + E_{\mathrm{PME}}
+ E_{\mathrm{D4}} + E_{\mathrm{spin}} + E_{\mathrm{resp}},
$$

with electric response

$$
E_{\mathrm{resp}}
= -\boldsymbol{\mu}\cdot\boldsymbol{\mathcal E}
- \frac{1}{2}\boldsymbol{\mathcal E}^{\mathsf T}
\boldsymbol{\alpha}\boldsymbol{\mathcal E}.
$$

Forces and spin fields remain derivatives of the same energy:

$$
\mathbf F_i=-\frac{\partial E_{\mathrm{tot}}}{\partial \mathbf R_i},
\qquad
\mathbf H_i^{\mathrm{eff}}=-\frac{\partial E_{\mathrm{spin}}}{\partial \mathbf S_i},
\qquad
Z^{*}_{i,\alpha\beta}=\frac{\partial \mu_\alpha}{\partial R_{i\beta}}.
$$

## Execution graph

```mermaid
flowchart LR
    A[Atomic numbers, positions, cell, field, spins] --> G[Neighbor graph]
    G --> L1[Layer 1: parity-aware O(3) message passing]
    L1 --> PES[Short-range energy]
    L1 --> R[Response tensor heads]
    R --> Q[Layer 2: QEq and PME]
    R --> P[Layer 2: polarization and D4]
    R --> S[Layer 3: J, Di, and DMI]
    Q --> C[Charge and potential condition]
    S --> C2[Spin-invariant condition]
    C --> F[FiLM feedback]
    C2 --> F
    F --> L1
    PES --> H[Effective Hamiltonian]
    Q --> H
    P --> H
    S --> H
    H --> O[Energy, forces, response tensors, spin field]
```

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

pytest -q
python Dual_Layer_Atomic_E3_GNN.py self-test
python Dual_Layer_Atomic_E3_GNN.py gui
```

The GUI is the default research workflow for dataset inspection, architecture
selection, training, live plots, solver residuals, memory monitoring, and
Auto Research.

![PyQt6 research studio](docs/assets/gui/qt-research-studio.png)

## Command-line workflows

Inspect and validate a canonical dataset:

```bash
python Dual_Layer_Atomic_E3_GNN.py dataset-summary path/to/data.h5
python Dual_Layer_Atomic_E3_GNN.py dataset-validate path/to/data.h5 --output validation.json
```

Train and evaluate:

```bash
python Dual_Layer_Atomic_E3_GNN.py train \
  --dataset path/to/data.h5 \
  --mode joint \
  --device auto \
  --epochs 50 \
  --out-ckpt model.pt

python Dual_Layer_Atomic_E3_GNN.py evaluate \
  model.pt path/to/data.h5 \
  --split test \
  --output test_metrics.json
```

Convert an extXYZ file into the canonical schema:

```bash
python Dual_Layer_Atomic_E3_GNN.py dataset-extxyz \
  input.extxyz.gz output.h5
```

Run `python Dual_Layer_Atomic_E3_GNN.py --help` for the complete dataset,
training, evaluation, VASP, self-test, and GUI command set.

## Dataset policy

Large data files are intentionally excluded from this GitHub repository. Neo
uses the `e3mu-hdf5-v1` schema with explicit label masks, units, provenance,
physical parent groups, and fixed train/validation/test splits. Missing labels
are never fabricated, and incompatible absolute energy references are not
silently mixed.

| Neo tier | Structures | Approximate size | Intended use |
| --- | ---: | ---: | --- |
| Tiny | 5,575 | 21.3 MB | Fast functional checks |
| Small | 15,221 | 52.8 MB | Intermediate experiments |
| Standard | 46,414 | 135.1 MB | Portable mixed-granularity training |
| Large | 613,267 | 1.23 GB | Trajectory-rich training |

Public redistribution of the current Neo binaries is **not yet authorized**:
the transformed `BEC/H2O`, `BEC/MAPbI3`, and `BEC/dimer` records require
archive-level rights confirmation or removal. See
[Dataset and licensing](docs/DATASETS.md) before publishing any binary.

## Verified behavior

The current source tree passes 44 regression tests and the deterministic
physics self-test. The checked invariants include:

- rotation and reflection behavior of energy, force, dipole, and
  polarizability;
- invariance of spin energy and odd transformation of the effective spin field
  under simultaneous spin reversal;
- graph-wise charge conservation and QEq stationarity;
- conservative forces against finite differences;
- differentiable QEq, PME, polarization, D4, FiLM, and Layer-3 losses;
- HDF5 mask semantics, group-safe splits, checkpoint round trips, and VASP
  magnetic mapping.

![Physics self-test margins](docs/assets/generated/physics-self-tests.png)

Short benchmark values, dataset limitations, and memory measurements are
reported in [Training and validation](docs/TRAINING_AND_VALIDATION.md). They
must not be interpreted as production accuracy claims.

## Paper and technical documentation

The Markdown manuscript follows the completed E(3)-GNN chapters of the research
proposal. Each paper part maps to a deeper technical document:

| Manuscript part | Technical document |
| --- | --- |
| Full paper | [Paper](docs/PAPER.md) |
| Scientific background and scope | [Scientific background](docs/SCIENTIFIC_BACKGROUND.md) |
| Three-layer network and coupling | [Architecture](docs/ARCHITECTURE.md) |
| Hamiltonians and physical mechanisms | [Physics](docs/PHYSICS.md) |
| Neo composition, schema, and rights | [Datasets](docs/DATASETS.md) |
| Optimization, validation, and measured results | [Training and validation](docs/TRAINING_AND_VALIDATION.md) |
| Installation and reproducibility | [Reproducibility](docs/REPRODUCIBILITY.md) |
| Proposal-to-code equation audit | [Formula crosswalk](docs/FORMULAE.md) |

The original proposal is retained as
[Research_Proposal Mixed-Granularity-Aware Graph Neural Network V5.1_Comp.pdf](Research_Proposal%20Mixed-Granularity-Aware%20Graph%20Neural%20Network%20V5.1_Comp.pdf).
The manuscript includes only the implemented E(3)-GNN scope; later proposal
material is intentionally excluded.

## Repository layout

```text
Dual_Layer_Atomic_E3_GNN.py   Single executable implementation
tests/                        Regression and physics tests
docs/PAPER.md                 Markdown manuscript
docs/*.md                     Technical sections and reproducibility notes
docs/assets/                  Proposal figures and measured plots
Datasets/Neo/*.md             Dataset card, schema, provenance, and rights docs
requirements.txt              Runtime and test dependencies
```

## Citation

Use [CITATION.cff](CITATION.cff) when citing the software. Until a versioned
archival release or journal DOI exists, cite the repository commit used in the
experiment together with the dataset versions and checksums.

## License

Original software and project documentation are released under the
[MIT License](LICENSE). Datasets, checkpoints, dependencies, and identifiable
third-party records are not relicensed by MIT. Read [NOTICE.md](NOTICE.md) and
[Datasets/Neo/LICENSES_AND_ATTRIBUTION.md](Datasets/Neo/LICENSES_AND_ATTRIBUTION.md)
before redistribution.
