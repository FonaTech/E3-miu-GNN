# Training, Auto Research, and Validation

This document expands Sections 4.3-5 of the [paper](PAPER.md). Reported values
are reproducible functional checks of the current source tree. They are not a
claim of a converged universal interatomic potential.

## Training contract

The trainer consumes either a canonical `e3mu-hdf5-v1` file, a self-contained
SE/Plus/Max `e3mu-composite-hdf5-v1` file, or the retained legacy
static/response extXYZ pair. HDF5 is preferred because it preserves target
masks, source identity, physical groups, units, and fixed splits explicitly.
Composite files keep the complete Standard or Large response payload at the
root and stream their selected OMat24 foundation from
`sources/omat24/packed/`; no external Parquet tree is required.

```mermaid
flowchart LR
    D[Canonical or Composite data and masks] --> B[Neighbor graphs and batches]
    B --> M[MixedGranularityE3GNN]
    M --> H[Total Hamiltonian]
    H --> O[Predictions and derivatives]
    O --> L[Mask-aware multi-task loss]
    L --> G[Finite-gradient check and clipping]
    G --> U[Optimizer update]
    U --> V[Fixed-split validation]
    V --> C[Safe checkpoint and artifacts]
```

## Mask-aware objective

For target family $`t`$, prediction $`\widehat{\mathbf{y}}_{t,k}`$, reference
$`\mathbf{y}_{t,k}`$, mask or sample weight $`m_{t,k}`$, and component count $`d_t`$,
the implemented multi-task objective is

```math
\mathcal L(\theta)=
\sum_{t\in\mathcal T}w_t
\frac{\sum_k m_{t,k}
\left\|\widehat{\mathbf y}_{t,k}-\mathbf y_{t,k}\right\|_2^2}
{\sum_k m_{t,k}d_t}.
```

Active targets can include energy, forces, dipole, molecular and atomic
polarizability, charges, atomic dipoles, C6, Born effective charge, magnetic
moments, effective spin field, $`J`$, $`D_i`$, and DMI. A target contributes only
when all three conditions hold:

1. its loss weight is positive;
2. the dataset mask is active for at least one item in the batch; and
3. the selected architecture produces the required output.

Energy errors are normalized per atom before aggregation so that large cells
do not dominate solely through atom count. Expensive derivative outputs are
constructed only when their active mask and loss weight require them.

### Wavefunction alignment objective

When `model.enable_waloss=True`, a separate electronic Hamiltonian head reads
the shared Response scalar features and predicts a fixed $`K\times K`$
real-symmetric matrix. For paired `orbital_hamiltonian` and
`orbital_eigenvectors` labels, the trainer transforms prediction and reference
with the reference eigenvectors and adds

```math
\mathcal L_{\mathrm{WALoss}}
=w_{\mathrm{waloss}}
\left(
\lambda_{\mathrm d}\mathcal L_{\mathrm{orbital\ energy}}
+\lambda_{\mathrm o}\mathcal L_{\mathrm{orbital\ coupling}}
\right).
```

Diagonal and strict-upper-triangle terms have separate element-count
normalization. `waloss_diagonal_weight` and `waloss_off_diagonal_weight` control
their internal balance; `w_waloss` controls the contribution to the complete
training objective. The prediction side is never diagonalized during training,
so no gradient crosses an eigensolver.

WALoss is accepted only in Response or Joint training (and the corresponding
Full Chain stages). `waloss_dim=0` means infer $`K`$ from canonical data; a
positive configured value must match the file. Train and validation splits must
both contain paired active labels. The GUI and Auto Research hide or reject the
controls when the data capability scan finds no such pair.

## Training modes

| Mode | Trainable scope | Intended use |
| --- | --- | --- |
| `base` | ground-state Layer-1 branch | establish local energy and force representation |
| `response` | response branch and optional electronic WALoss head above a base checkpoint | train electric/electronic response while retaining a frozen ground model |
| `joint` | active Layer-1, Layer-2, Layer-3, and FiLM parameters | coupled fine-tuning under a shared objective |

The full-chain GUI workflow can freeze the ground branch during response
warmup, assign separate ground and response learning rates, ramp response
weights, and finish with one or more joint stages. The command-line interface
exposes the same `TrainConfig` through JSON presets.

## Architecture and data compatibility

The GUI scans canonical masks and periodicity before enabling switches or loss
fields. The enforced dependency graph is:

```mermaid
flowchart TD
    PME[PME / Ewald] --> Q[QEq]
    D4[D4] --> Q
    DMI[DMI] --> S[Spin]
    DMI --> P["O(3) parity"]
    L3[L=3 tensor] --> P
    FILM[FiLM] --> P
    FILM --> DOMAIN[At least one active domain]
    MASKS[Dataset labels and periodicity] --> PME
    MASKS --> D4
    MASKS --> S
```

The current D4 backend is molecular, so periodic datasets disable D4. PME is
meaningful only for periodic records and requires QEq. Direct $`J`$, $`D_i`$, or
DMI losses remain unavailable for the current portable Neo tiers because their
masks are false. WALoss likewise remains unavailable until paired aligned
Hamiltonian/eigenvector labels are supplied.

The checkpoint loader preserves pre-WALoss behavior. A legacy native checkpoint
can continue training or inference with the head disabled. When a compatible
WALoss configuration deliberately warm-starts from it, matching tensors are
reused and only the new electronic-head parameters are initialized; this leaves
the legacy ground energy and force path unchanged before further training. An
initialized head is not a trained physical output, and source versions that
predate the head are not guaranteed to read new WALoss checkpoints.

## Stable optimization safeguards

The trainer refuses to save an unvalidated epoch-0 checkpoint. Each optimizer
step checks model outputs, loss, and every parameter gradient for finite values.
The global gradient norm is evaluated after scale normalization so a float32
sum of squares cannot overflow merely because individual gradients are large.
A failed step reports structure IDs, atom and edge counts, and affected
parameter names before stopping.

QEq and induced-polarization solvers expose both residuals and stability
shifts. A finite solve with a large curvature shift is retained as a diagnostic
rather than presented as a calibrated physical result.

## Validation score and checkpoint selection

Loss weights are optimization choices and must not control model ranking.
Checkpoint selection and Auto Research therefore use

```math
S_{\mathrm{val}}=
\frac{1}{|\mathcal T_{\mathrm{active}}|}
\sum_{t\in\mathcal T_{\mathrm{active}}}
\frac{\mathrm{MAE}_t}{s_t},
```

where $`s_t`$ is a fixed characteristic scale. Current scales are 1 for energy,
force, dipole, polarizability, and magnetic moment; 0.1 for charge, atomic
dipole, atomic polarizability, BEC, and stress in eV/Angstrom$^3$; 10 for C6;
and 0.01 eV for effective spin field and spin-Hamiltonian parameters. WALoss
orbital-energy and orbital-coupling MAEs use a separate 0.1 eV characteristic
scale. A candidate cannot appear better by reducing its own loss coefficient.

The electromechanical scale is 0.5 C/m$^2$ and the paired magnetoelastic scale
is 0.02 eV/Angstrom$^3$. `w_piezoelectric` and `w_magnetoelastic` remain zero
unless their explicit masks are present; a total magnetic stress label does
not silently activate the paired magnetoelastic objective.

Stress loss is evaluated over the six independent symmetric components. `mse`
and scaled `huber` are supported; the default Huber threshold is 0.05
eV/Angstrom$^3$. Only fully periodic, nonsingular cells with active stress masks
contribute. The predicted tensor is never an independent regressor: it is the
cell-strain derivative of the same total energy used for force training.
The piezoelectric objective differentiates the model dipole with respect to the
same strain variable. The magnetoelastic objective differentiates the complete
target-minus-reference coupled energy, and therefore requires co-located
`spins`, `reference_spins`, and `magnetoelastic_stress` masks.

With `label_aware_coupling=True`, each Joint batch activates only the electric,
polarization, dispersion, and spin mechanisms supported by its labels or
explicit field/charge/spin state. Batches with only L1 energy, force, and stress
targets skip the response core entirely. Optional BEC acoustic-sum-rule and
electric/spin FiLM residual losses add explicit physical constraints without
turning missing L2/L3 labels into zero targets.

## Auto Research

Auto Research first evaluates the current GUI configuration as a baseline. It
then combines random exploration with a small Gaussian-process surrogate. The
same deterministic subset and split are reused across candidates.

```mermaid
sequenceDiagram
    participant U as User architecture
    participant D as Dataset capability scan
    participant A as Auto Research
    participant T as Short training trial
    participant G as GUI
    U->>D: lock selected switches
    D->>A: remove unsupported targets and solver dimensions
    A->>T: run baseline
    loop exploration and surrogate proposals
        A->>T: candidate parameters
        T-->>A: normalized validation score
    end
    A-->>G: retained best parameter set
    G->>G: Apply Best only on explicit click
```

Search levels progressively add active loss weights, optimizer/backbone
parameters, staged fine-tuning parameters, and solver parameters that remain
meaningful for the selected architecture. The selected architecture itself is
locked by default. A dataset change invalidates the retained result so values
from an earlier capability scan cannot be applied to a different corpus.

## Live artifacts

With epoch artifacts enabled, training writes under
`<checkpoint parent>/train/<checkpoint stem>/`:

- safe per-epoch checkpoints;
- full and clipped energy/force parity plots plus stress MAE history;
- force-norm plots;
- loss and MAE histories;
- active auxiliary-task MAEs, including orbital-energy and orbital-coupling MAE
  when WALoss labels are present;
- QEq, polarization, and separate electric/spin FiLM residual histories;
- memory histories and machine-readable JSON; and
- the best validated checkpoint at the requested output path.

The PyQt6 GUI displays the latest regression, MAE, solver-residual, and memory
views after every validation epoch.

![PyQt6 live research interface](assets/gui/qt-research-studio.png)

## Deterministic physical validation

The float64 self-test evaluates complete transformed forward passes and finite
differences. With the documented seed 7, the current maximum errors are:

| Check | Maximum error |
| --- | ---: |
| Rotation: energy | 0 |
| Rotation: force | $`3.47\times10^{-18}`$ |
| Rotation: dipole | $`2.61\times10^{-15}`$ |
| Rotation: polarizability | $`2.22\times10^{-16}`$ |
| Reflection: energy, force, dipole, polarizability | 0 |
| Time reversal: spin energy and effective field | 0 |
| Charge conservation | 0 e |
| QEq stationarity residual | $`9.45\times10^{-12}`$ |
| Conservative-force finite difference | $`2.02\times10^{-11}`$ eV/angstrom |
| Conservative-stress finite difference | $`1.87\times10^{-14}`$ eV/angstrom$^3$ |
| Stress symmetry | 0 |

![Deterministic validation margins](assets/generated/physics-self-tests.png)

The regression suite covers affine-strain stress finite differences, stress
rotation covariance and nonperiodic masking, O(3) channels, QEq on Apple MPS,
PME and D4 reference behavior, polarization gradients, label-aware L1/L2/L3
routing, spin losses, checkpoint safety, HDF5 masks and splits, dataset-aware
GUI state, WALoss basis invariance and differentiability, and VASP magnetic
mapping.

## Short held-out benchmarks

These experiments validate data flow and trainability over small budgets.

| Dataset and held-out split | Training scope | Held-out result |
| --- | --- | --- |
| QM7-X, 8 test molecules | 12 epochs; energy, dipole, polarizability, charge, atomic polarizability | energy 1.907 eV/system; dipole 0.1313 e angstrom/component; polarizability 0.7217 angstrom3/component; charge 0.0949 e/atom; atomic polarizability 0.3089 angstrom3/component |
| BEC, 4 validation cells / 768 atoms | 2 epochs | BEC MAE 0.2156 e/component |
| SCFNN, 4 validation cells / 768 atoms | 20 epochs | dipole MAE 2.435 e angstrom/component; zero baseline 2.972 e angstrom/component |

The QM7-X force and C6 loss weights were zero, so their evaluator outputs are
not trained-accuracy results. The short QEq run required a mean test stability
shift of 14.39 eV, which indicates that its learned raw hardness was not yet
physically calibrated.

## Memory behavior

Force, stress, and BEC losses require higher-order autograd graphs. On Apple MPS,
batches are packed by edge count rather than structure count; graph references,
optimizer gradients, plotting figures, and reclaimable allocator blocks are
released after their useful lifetime. Every epoch reports process RSS, active
MPS tensors, driver allocation, and reclaimable cache.

Canonical and Composite HDF5 training and checkpoint evaluation stream by
default. The in-memory state contains the structure/atom index, group split,
label masks, and element table. Plus and Max read OMat24 structures from
lossless packed arrays in contiguous batch spans; no per-row Arrow conversion
occurs in the training path. Exact neighbor indices and periodic shift vectors
are kept in a source-, selection-, cutoff-, and backend-keyed disk cache. The
cache is written once, then opened read-only as contiguous memory-mapped arrays;
it is never rewritten between epochs. Cached `edge_counts` also drive the MPS
edge-budget sampler while preserving packed selector order. Coordinates and
labels are read only when a batch is requested, and a bounded two-batch CPU
thread prefetch overlaps HDF5 assembly with device execution.
`stream_hdf5=False` (or the CLI option
`--no-stream-hdf5`) retains the former whole-corpus materialization path for
debugging. Legacy extXYZ input is still materialized during parsing.

The following isolated-process measurement uses Neo Tiny at a 5 angstrom
cutoff and batch size 8. The selected train/validation corpus contains 4,971
structures, 332,336 atoms, and 13,495,436 directed edges. Incremental RSS is
measured relative to the same imported-runtime baseline in each fresh ARM64
process.

| Data path | Preparation | Data-only epoch | Peak incremental RSS | Total peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Materialized configurations and graphs | 36.55 s | 0.592 s | 918.4 MiB | 1,352.1 MiB |
| Streamed, cold topology cache | 30.44 s | 3.707 s | 372.3 MiB | 807.0 MiB |
| Streamed, reused topology cache | 0.0246 s | 3.733 s | 120.3 MiB | 552.0 MiB |

The reusable exact topology cache is 43.62 MiB. Local atom indices use an
adaptive safe unsigned integer width and are restored to `int64` before model
execution. Periodic shifts use a per-structure, bitwise-exact dictionary: the
5,542,544 nonzero Tiny shift rows contain 78,758 unique `float64[3]` bit
patterns, with at most 80 patterns in one structure. The dictionary values and
`uint8` codes occupy about 7.09 MiB; exact local nonzero-row indices use
`uint16`. Compared with the preceding 173.92 MiB sparse cache, the complete
cache is 130.30 MiB smaller. Reconstruction performs only integer lookup and
copying, with no floating-point recomputation.

The file-backed memory-map pages are reclaimable by the operating system; the
reported RSS includes pages touched during the complete digest and epoch scans.
The warm data-only epoch remains about 3.7 seconds, without repeated cache
writes. Materialized and streamed graph digests are identical across all
`AtomicData` labels and weights, atom types, coordinates, edge indices,
metadata, and periodic shifts. Transferring the same measured batch to MPS also
produced identical allocations in both modes: 2.460 MiB active and 40.453 MiB
driver memory. Streaming therefore changes host storage and I/O behavior, not
the graph, numerical precision, cutoff, batch tensor, or accelerator-resident
model calculation. Coordinate and label HDF5 arrays remain streamed, while the
largest repeated topology payload is memory-mapped and batch assembly is
prefetched. For production force training, the remaining data-only overhead is
also overlapped with model forward/backward time.

The machine-readable report and reproducer are
`Validation/StreamingBenchmark/tiny_streaming_comparison.json` and
`Validation/StreamingBenchmark/benchmark_hdf5_memory.py`.

A measured five-epoch MPS run with energy, force, dipole, and polarizability
losses increased RSS by 16.3 MiB between epochs 1 and 5. Post-cleanup active
MPS allocation stayed near 30.8 MiB and driver allocation settled near
116.2 MiB. No sustained-growth warning was triggered.

![Measured memory profile](assets/generated/memory-profile.png)

This short bounded result is evidence against an epoch-to-epoch retained-graph
leak in that workflow. It is not a universal peak-memory bound; peak memory
still depends on edge count, active derivative targets, feature width, and
solver configuration.

## Interpretation limits

- Symmetry and derivative tests establish structural correctness, not
  predictive coverage across all elements.
- The portable corpus has no direct active $`J`$, $`D_i`$, or DMI labels.
- The current D4 implementation does not provide periodic lattice dispersion.
- No converged phonon spectrum or production molecular-dynamics stability
  study is reported.
- Neo's source composition and current BEC rights blocker are described in
  [Datasets](DATASETS.md).

Use [Reproducibility](REPRODUCIBILITY.md) for exact setup and command examples.
