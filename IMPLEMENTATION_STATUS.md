# Mixed-Granularity E(3)-mu-GNN Implementation Status

Status date: 2026-07-27

This page is a concise boundary between implemented software, verified
behavior, and remaining scientific validation. The complete manuscript is in
[`docs/PAPER.md`](docs/PAPER.md).

## Current implementation

All executable project logic is merged into
`E3_miu_GNN.py`. The default GUI command launches PyQt6;
`gui-tk` retains the legacy Tk interface.

| Layer | Implemented components | Available outputs |
| --- | --- | --- |
| Layer 1: local atomic | parity-aware O(3) scalar, polar, axial, $L=2$, optional $L=3$ message passing; continuous chemistry; conservative energy | short-range energy, total energy, forces |
| Layer 2: electric/domain | constrained differentiable QEq; periodic Ewald/PME; exact Thole-damped polarization equilibrium; molecular D4 | charges, electrostatic potential, dipoles, polarizabilities, C6, BEC, named energies and residuals |
| Layer 3: spin | time-reversal-even Heisenberg exchange, traceless single-ion anisotropy, optional axial DMI | $J_{ij}$, $D_i$, DMI, spin energy, magnetic moments, effective spin field |
| Auxiliary electronic | fixed-dimension real-symmetric Hamiltonian head and reference-eigenspace WALoss | aligned orbital Hamiltonian, diagonal/off-diagonal loss and MAE diagnostics |
| Cross-granularity coupling | bounded FiLM from charge, potential, and spin invariants to local equivariant features | coupling residual and refined energy/response outputs |

The active Hamiltonian is

```math
E_{\mathrm{tot}}=
E_{\mathrm{short}}+E_{\mathrm{QEq}}+E_{\mathrm{PME}}
+E_{\mathrm{D4}}+E_{\mathrm{spin}}+E_{\mathrm{resp}}.
```

Forces, stress, BEC, and spin effective fields are derivatives of the assembled
energy, not post-processing corrections. The auxiliary electronic Hamiltonian
does not enter this energy and is separate from the Layer-3 spin Hamiltonian.

## Training and GUI

- Base, response, and joint modes share one `TrainConfig` and trainer.
- Mask-aware task weights cover the observable families plus optional paired
  WALoss Hamiltonian/eigenvector supervision.
- PyQt6 architecture switches are filtered by HDF5 masks, periodicity, and
  physical dependencies.
- Disabled controls retain explanatory hover documentation without exposing
  irrelevant editable fields.
- Physics solver parameters and Auto Research dimensions follow the selected
  architecture.
- Auto Research runs a baseline plus random/surrogate-guided trials on one
  deterministic subset and split; `Apply Best` is explicit and invalidated by
  dataset changes.
- The live pane displays regression, MAE, solver-residual, and memory views.
- Epoch artifacts include safe checkpoints, plots, and JSON histories.

## Numerical stability and memory

- Apple MPS QEq uses an analytic Helmert neutral basis and reduced Cholesky
  solve, avoiding unsupported QR and unreliable indefinite KKT backward.
- Its smallest-curvature diagnostic is evaluated on CPU float64 with a
  differentiable on-device Rayleigh quotient and a Gershgorin fallback.
- `tad-dftd4` runs as a differentiable CPU sublayer when the main model is on
  MPS because its reference data require float64.
- Global gradient clipping uses a scale-normalized norm to prevent float32
  reduction overflow.
- Non-finite outputs, losses, and gradients stop training with structure and
  parameter diagnostics; no unvalidated epoch-0 checkpoint is saved.
- Edge-balanced MPS batches constrain higher-order force/BEC graph size.
- Parsed data, autograd references, optimizer gradients, plots, and reclaimable
  MPS cache blocks are released explicitly.
- A measured five-epoch MPS run grew RSS by 16.3 MiB from epoch 1 to 5; active
  MPS memory remained near 30.8 MiB after cleanup and no sustained-growth
  warning fired.

## Neo data status

Canonical `e3mu-hdf5-v1` files store packed geometry, explicit target masks,
source/method/system/group metadata, fixed group-safe splits, units, and
provenance. Composite `e3mu-composite-hdf5-v1` files embed a complete canonical
response payload and add packed OMat24 foundation arrays, exact selection
provenance, and atomic-reference statistics. Tiny and Small are deterministic
nested subsets of Standard; Large is built under a separate trajectory-rich
policy.

| Tier | Schema | Embedded response | OMat24 foundation | Total structures | Total atoms | Size |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | canonical | 5,780 | 0 | 5,780 | 394,755 | 20.011 MiB |
| Small | canonical | 16,703 | 0 | 16,703 | 1,069,318 | 51.218 MiB |
| Standard | canonical | 46,414 | 0 | 46,414 | 2,316,736 | 0.120772 GiB |
| SE | composite | Standard: 46,414 | 559,279 | 605,693 | 12,767,209 | 0.690297 GiB |
| Large | canonical | 613,267 | 0 | 613,267 | 17,760,024 | 1.219244 GiB |
| Plus | composite | Large: 613,267 | 25,206,004 | 25,819,271 | 488,227,614 | 38.237181 GiB |
| Max | composite | Large: 613,267 | 100,670,282 | 101,283,549 | 1,899,323,661 | 128.559024 GiB |

The portable tiers contain Layer-1 labels, electric-response labels, spins,
magnetic moments, and 100 effective spin-field records. They do **not** contain
active direct $J$, $D_i$, or DMI aggregate labels. The Layer-3 architecture is
functionally and symmetry validated, but paper-grade magnetic calibration
still requires compatible collected calculations.

The schema and trainer support optional paired `orbital_hamiltonian` and
`orbital_eigenvectors` labels for WALoss. All seven current tiers omit those
arrays and the optional `wavefunction_dim` attribute. Existing targets and old
checkpoints remain supported, but no released Neo file currently supervises
the electronic Hamiltonian head.

Absolute energies from unrelated electronic-structure methods are masked from
the shared aggregate energy objective. Missing labels are never fabricated.

## Verified behavior

At the time of this status update:

- The regression suite covers physical invariants, dataset contracts, training,
  GUI behavior, and checkpoint compatibility.
- Conservative-force finite-difference error: $8.15\times10^{-12}$
  eV/angstrom.
- QEq stationarity residual: $9.39\times10^{-12}$.
- Rotation, reflection, and simultaneous spin-reversal errors are at numerical
  precision.
- Periodic PME agrees with a direct `torch-pme` Ewald reference to numerical
  precision.
- D4 energy and C6 agree with the official `dftd4` H2 reference.
- QEq, PME, polarization, D4, FiLM, and Layer-3 supervised paths retain
  gradients in focused tests.
- WALoss tests cover reference-basis invariance, separate diagonal/off-diagonal
  normalization, differentiability without a predicted eigensolver, paired
  masks, checkpoint warm starts, and separation from the spin Hamiltonian.
- HDF5 masks, group splits, checkpoint round trips, GUI capability filtering,
  Auto Research write-back, and VASP magnetic mapping are regression-tested.

Short held-out functional baselines are:

| Dataset | Result |
| --- | --- |
| QM7-X | energy 1.907 eV/system; dipole 0.1313 e angstrom/component; polarizability 0.7217 angstrom3/component; charge 0.0949 e/atom |
| BEC | 0.2156 e/component over four validation cells |
| SCFNN | dipole 2.435 e angstrom/component versus zero baseline 2.972 |

These short experiments verify data flow and optimization. They are not
production-accuracy claims. In particular, the short QM7-X QEq model required
a mean test stability shift of 14.39 eV, indicating incomplete hardness
calibration.

## Publication status

- Original code and project documentation: MIT.
- Root README, manuscript, technical sections, contribution guide, citation
  metadata, notice, figures, and formula audit: prepared.
- Canonical data binaries: intentionally excluded from GitHub.
- Neo Hugging Face technical staging: prepared and validated locally.
- Public Neo binary release: **blocked** until redistribution rights for the
  supplied `BEC/H2O`, `BEC/MAPbI3`, and `BEC/dimer` archive are confirmed or
  those records are removed and all tiers are rebuilt.

Dataset components retain MIT, CC BY 4.0, GPL-3.0, or unresolved source terms;
the software MIT license does not relicense them.

## Remaining scientific work

- Run and collect converged independent magnetic calculations with direct
  $J$, $D_i$, DMI, and held-out spin-configuration validation.
- Collect a fixed-subspace, gauge-aligned electronic Hamiltonian/eigenvector
  dataset and establish held-out WALoss orbital-energy and coupling accuracy.
- Calibrate QEq hardness and solver shifts on larger independent domains.
- Perform converged phonon and long molecular-dynamics stability studies.
- Evaluate periodic dispersion with an explicitly periodic backend before
  enabling D4 for crystals.
- Establish production accuracy separately for each source/method domain.

Proposal-only reinforcement learning, Bayesian optimization for reaction
paths, grand-canonical sampling, and agent workflows remain outside the
implemented E(3)-GNN scope.

## Documentation map

- [Scientific background](docs/SCIENTIFIC_BACKGROUND.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Physical mechanisms](docs/PHYSICS.md)
- [Datasets](docs/DATASETS.md)
- [Training and validation](docs/TRAINING_AND_VALIDATION.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Formula crosswalk](docs/FORMULAE.md)
