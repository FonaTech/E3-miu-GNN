# Mixed-Granularity-Aware E(3)-Equivariant Graph Neural Network for Coupled Atomic, Electrostatic, and Spin Information

**Yufeng Zhan (Fona)**  
Implementation-aligned manuscript, July 2026

This manuscript documents the atomic, domain, and spin layers implemented in
`E3_miu_GNN.py`, together with FiLM coupling, wavefunction-aligned electronic
Hamiltonian supervision, the two Neo HDF5 layouts, and verified training
behavior.

## Abstract

Machine-learning interatomic potentials usually compress an atomistic system
into a local geometric representation and learn a scalar potential energy.
This construction is efficient, but local geometry alone is not an adequate
state description for systems in which charge redistribution, long-range
electrostatics, induced polarization, dispersion, or spin order changes the
energy landscape. We present an implementation of a mixed-granularity
E(3)-equivariant graph neural network, E(3)-mu-GNN, that separates these
mechanisms into three coupled levels. Layer 1 is a parity-aware O(3) atomic
network with scalar, polar, axial, and symmetric-traceless tensor channels.
Layer 2 predicts electronic response parameters and drives differentiable
charge equilibration, Ewald/PME electrostatics, Thole-damped polarization, and
molecular DFT-D4. Layer 3 parameterizes a time-reversal-even spin Hamiltonian
containing Heisenberg exchange, single-ion anisotropy, and an optional
Dzyaloshinskii-Moriya term. Charge, electrostatic potential, and spin
invariants feed back into Layer 1 through bounded feature-wise linear
modulation. A separate real-symmetric electronic Hamiltonian head supports a
wavefunction alignment loss (WALoss) that resolves diagonal orbital-energy
error from off-diagonal reference-state mixing without diagonalizing the
prediction. All energy components are assembled before differentiating forces,
Born effective charges, and effective spin fields, preserving their relation to
a common Hamiltonian. Two mask-aware HDF5 layouts permit partially labelled,
multi-source supervision and self-contained OMat24/response compositions
without inventing absent targets or mixing incompatible energy references. The
current Neo tiers contain no paired WALoss matrices, so no orbital-accuracy
result is claimed. Deterministic tests verify E(3)/O(3)
transformation behavior, time reversal, charge conservation, differentiability,
and conservative forces. Short data benchmarks establish functional behavior,
while also showing that converged multi-domain and magnetic accuracy remains a
separate scientific validation problem.

## 1. Scientific background for the research

### 1.1 From electronic structure to an atomistic potential

The many-electron problem is reduced in Kohn-Sham density-functional theory
(DFT) to auxiliary one-electron equations [2,3]:

```math
\left[-\frac{\hbar^2}{2m_e}\nabla^2 + V_{\mathrm{eff}}(\mathbf r)\right]
\phi_i(\mathbf r)=\epsilon_i\phi_i(\mathbf r),
\qquad
n(\mathbf r)=\sum_i\left|\phi_i(\mathbf r)\right|^2,
```

where

```math
V_{\mathrm{eff}}(\mathbf r)
=V_{\mathrm n}(\mathbf r)+V_{\mathrm H}(\mathbf r)
+V_{\mathrm{xc}}(\mathbf r),
\qquad
V_{\mathrm{xc}}(\mathbf r)
=\frac{\delta E_{\mathrm{xc}}[n]}{\delta n(\mathbf r)}.
```

The exchange-correlation approximation controls a major part of the accuracy
and cost trade-off. One expression of its limitation is the difference between
the Kohn-Sham and quasiparticle gap,

```math
E_g^{\mathrm{QP}}=I-A=E_g^{\mathrm{KS}}+\Delta_{\mathrm{xc}}.
```

Strongly localized states are often treated with an additional on-site
correction,

```math
E_{\mathrm{DFT}+U}
=E_{\mathrm{DFT}}
+\frac{U_{\mathrm{eff}}}{2}\sum_\sigma
\mathrm{Tr}\!\left[\mathbf n_\sigma
\left(\mathbf I-\mathbf n_\sigma\right)\right],
\qquad U_{\mathrm{eff}}=U-J.
```

These equations provide physical background but are not embedded as an explicit
DFT or DFT+U solver in E(3)-mu-GNN. The implemented model instead learns an
effective atomistic Hamiltonian from labelled calculations while enforcing
geometric and physical structure in its representation and solver layers.

### 1.2 Why a local scalar model is insufficient

An ML interatomic potential approximates the Born-Oppenheimer potential energy
surface $`E(\mathbf R)`$ and evaluates forces by differentiation. A strictly
local decomposition,

```math
E_{\mathrm{local}}(\mathbf R)=\sum_i \varepsilon_i
\left(\mathcal N_i^{r_c}\right),
```

is effective when interactions outside the cutoff $`r_c`$ are screened or can be
absorbed into local environments. It becomes incomplete when the state depends
on a global charge constraint, reciprocal-space electrostatics, collective
polarization, or spin order. Treating all such effects as an unconstrained
correction to one scalar network also obscures which symmetry and conservation
law each quantity must satisfy.

The central research question is therefore:

> Can local atomic geometry, domain-scale electric response, and subatomic spin
> information be represented at their natural physical granularities while
> remaining differentiable parts of one energy model?

## 2. Purpose, scientific significance, and originality

The objective is a unified atomistic model with three explicit representation
levels:

1. a local E(3)/O(3)-equivariant atomic potential;
2. an electric domain layer with constrained and long-range solvers; and
3. a time-reversal-aware spin Hamiltonian.

The implementation makes five concrete contributions.

**Symmetry-resolved local features.** Polar and axial vectors are kept
separate under inversion, and higher-order geometry is represented through
fixed real symmetric-traceless bases. This is required to distinguish a
displacement from a magnetic axial vector.

**Physical solvers inside autograd.** QEq, PME, induced polarization, and D4
are energy-producing sublayers rather than post-processing corrections.
Forces therefore include their positional derivatives.

**Interpretable spin energy.** Exchange, anisotropy, and DMI parameters are
predicted from geometric features and assembled into a Hamiltonian that is
exactly even under simultaneous spin reversal.

**Physics-aligned electronic supervision.** An optional fixed-subspace
Hamiltonian head is compared in the reference eigenspace. WALoss separately
normalizes orbital-energy and orbital-coupling residuals, so matrix size does
not silently change their balance and no gradient crosses a predicted
eigendecomposition.

**Feedback instead of independent addition.** Layer-2 and Layer-3 invariants
condition subsequent atomic messages through FiLM. Electronic state can thus
alter the learned short-range representation before the final energy is
formed.

![Implemented system overview](assets/proposal/system-overview-core.png)

*Figure 1. Implemented E(3)-GNN system with three physical layers and their
electronic coupling network.*

## 3. Research method and implemented architecture

### 3.1 Inputs, graph, and transformation contract

For each structure $`g`$, the model receives atomic numbers $`z_i`$, Cartesian
positions $`\mathbf R_i`$, cell $`\mathbf A_g`$, periodic flags, total charge
$`Q_g`$, external field $`\boldsymbol{\mathcal E}_g`$, and optional unit spin
vectors $`\mathbf S_i`$. Directed edges connect neighbors inside a cutoff,

```math
\mathbf r_{ij}=\mathbf R_j+\mathbf t_{ij}-\mathbf R_i,
\qquad
r_{ij}=\|\mathbf r_{ij}\|,
\qquad
\widehat{\mathbf r}_{ij}=\frac{\mathbf r_{ij}}{r_{ij}},
```

where $`\mathbf t_{ij}`$ is the periodic image shift. Translation invariance
follows from relative positions. The learned energy is invariant under O(3),
while vector and tensor outputs transform in their corresponding
representations.

```mermaid
flowchart TB
    I[Structure and physical state] --> N[Exact cutoff neighbor graph]
    N --> A[Atomic scalar, polar, axial, L2 and optional L3 features]
    A --> R[Response parameters]
    R --> E[Electric-domain solvers]
    R --> W[Electronic Hamiltonian head]
    W --> WA[WALoss, training only]
    A --> M[Spin Hamiltonian]
    E --> F[FiLM condition]
    M --> F
    F --> A
    A --> H[Total Hamiltonian]
    E --> H
    M --> H
    H --> D[Autograd observables]
```

![Three physical granularities](assets/proposal/mixed-granularity-core.png)

*Figure 2. Atomic, domain, spin, and cross-granularity feedback components.*

### 3.2 Layer 1: parity-aware atomic E(3)-GNN

The equivariant message-passing layer is expressed as a tensor-product expansion,

```math
\mathbf m_{ij}^{L_{\mathrm{out}}}
=\sum_{L_{\mathrm{in}},L_{\mathrm{edge}}}
W_{L_{\mathrm{in}},L_{\mathrm{edge}}\rightarrow L_{\mathrm{out}}}
(r_{ij})
\left[
\mathbf h_j^{L_{\mathrm{in}}}\otimes
\mathbf Y^{L_{\mathrm{edge}}}(\widehat{\mathbf r}_{ij})
\right]_{L_{\mathrm{out}}}.
```

The implementation realizes the selected products directly in a real
Cartesian basis. Node state contains

```math
\mathbf{h}_i=\bigl(
\mathbf{s}_i,\mathbf{v}_i,\mathbf{a}_i,
\mathbf{T}_i^{(2)},\mathbf{T}_i^{(3)}
\bigr).
```

with scalar $`\mathbf s_i`$, polar vector $`\mathbf v_i`$, axial vector
$`\mathbf a_i`$, five-component symmetric-traceless $`L=2`$ tensor
$`\mathbf T_i^{(2)}`$, and optional seven-component symmetric-traceless $`L=3`$
tensor $`\mathbf T_i^{(3)}`$. Examples of explicit parity-preserving channels
are

```math
\mathbf v_j\cdot\widehat{\mathbf r}_{ij}\rightarrow 0e,
\quad
\mathbf v_j\times\widehat{\mathbf r}_{ij}\rightarrow 1e,
\quad
\mathbf a_j\times\widehat{\mathbf r}_{ij}\rightarrow 1o,
\quad
\mathrm{ST}\!\left(
\mathbf v_j\otimes\widehat{\mathbf r}_{ij}
\right)\rightarrow 2e.
```

Radial filters use fixed Gaussian, trainable Gaussian, or Bessel bases and a
cosine cutoff. Aggregation is a mean over incoming edges; update gates depend
only on parity-even invariants such as $`\|\mathbf v\|^2`$,
$`\|\mathbf a\|^2`$, and tensor norms.

The short-range energy is

```math
E_{\mathrm{short}}
=\sum_i\left[E_{z_i}^{\mathrm{ref}}
+f_E(\mathbf s_i)\right].
```

The reference atomic energies are obtained from a regularized least-squares
fit over the active training set. The Hessian and dynamical matrix remain
derivatives of the same scalar surface,

```math
H_{i\alpha,j\beta}
=\frac{\partial^2 E_{\mathrm{tot}}}
{\partial R_{i\alpha}\partial R_{j\beta}},
\qquad
D_{\alpha\beta}^{ab}(\mathbf q)
=\frac{1}{\sqrt{m_a m_b}}
\sum_{\mathbf T}H_{0a\alpha,\mathbf T b\beta}
e^{i\mathbf q\cdot\mathbf T}.
```

These Hessian and dynamical-matrix expressions state the
automatic-differentiation interface. The current
validation suite tests first-derivative force consistency; it does not report a
converged phonon benchmark.

### 3.3 Field-response parameterization

Under the Born-Oppenheimer approximation, a static electric field is treated as
a perturbation
$`\widehat V_{\mathrm{ext}}=-\widehat{\boldsymbol\mu}\cdot\boldsymbol{\mathcal E}`$.
Retaining second order gives

```math
E(\mathbf R,\boldsymbol{\mathcal E})
=E_{\mathrm{PES}}(\mathbf R)
-\boldsymbol\mu^{(0)}(\mathbf R)\cdot\boldsymbol{\mathcal E}
-\frac{1}{2}\boldsymbol{\mathcal E}^{\mathsf T}
\boldsymbol\alpha(\mathbf R)\boldsymbol{\mathcal E}
+\mathcal O(\|\boldsymbol{\mathcal E}\|^3).
```

The response network reads scalar, polar, and $`L=2`$ features. It predicts raw
charges, permanent atomic dipoles, electronegativities, positive hardnesses,
C6 scaling, and atomic polarizabilities. The latter are decomposed into an
isotropic and symmetric-traceless part,

```math
\boldsymbol\alpha_i
=\mathrm{softplus}(a_i)\mathbf I
+\sum_{k=1}^{5}c_{ik}\mathbf B_k^{(2)},
\qquad
\boldsymbol\alpha=\sum_i\boldsymbol\alpha_i.
```

The total dipole combines permanent, charge-displacement, and induced terms,

```math
\boldsymbol\mu
=\sum_i\boldsymbol\mu_i^{\mathrm{perm}}
+\sum_i q_i(\mathbf R_i-\mathbf R_c)
+\sum_i\mathbf p_i^{\mathrm{ind}}.
```

For non-periodic structures $`\mathbf R_c`$ is the geometric center. Periodic
relative positions use the minimum-image finite-cell convention recorded in
the data metadata.

Here $`\boldsymbol\mu^{(0)}`$ contains the permanent and charge-displacement
terms. The reported finite-field dipole is
$`-\partial E/\partial\boldsymbol{\mathcal E}
=\boldsymbol\mu^{(0)}+\boldsymbol\alpha\boldsymbol{\mathcal E}`$ (including the
configured unit factor), so the induced contribution is not counted again in
the linear energy term.

### 3.4 Layer 2: charge and long-range domain physics

The total energy separates local and domain contributions,

```math
E_{\mathrm{tot}}=E_{\mathrm{short}}+E_{\mathrm{domain}}+E_{\mathrm{spin}}.
```

#### Differentiable charge equilibration

For one graph, QEq minimizes

```math
E_{\mathrm{QEq}}(\mathbf q)
=\boldsymbol\chi^{\mathsf T}\mathbf q
+\frac{1}{2}\mathbf q^{\mathsf T}
\left[\mathrm{diag}(\boldsymbol\eta)+\mathbf K\right]\mathbf q
+\boldsymbol\phi_{\mathrm{ext}}^{\mathsf T}\mathbf q,
\qquad
\mathbf 1^{\mathsf T}\mathbf q=Q.
```

The direct non-periodic kernel is softened at short range,

```math
K_{ij}=\frac{k_e}{\sqrt{r_{ij}^2+\sigma^2}},\qquad i\ne j.
```

Periodic graphs obtain $`\mathbf K`$ from an Ewald calculator. The reciprocal
contribution has the familiar form

```math
E_{\mathrm{rec}}
=\frac{1}{2\Omega}\sum_{\mathbf k\ne 0}
\frac{4\pi k_e}{\|\mathbf k\|^2}
e^{-\|\mathbf k\|^2/(4\alpha_E^2)}
\left|S(\mathbf k)\right|^2.
```

Rather than solve an indefinite KKT system, the implementation eliminates the
constraint. Let $`\mathbf B`$ be an analytic Helmert basis satisfying
$`\mathbf 1^{\mathsf T}\mathbf B=0`$ and
$`\mathbf B^{\mathsf T}\mathbf B=\mathbf I`$. With
$`\mathbf q=\mathbf q_0+\mathbf B\mathbf z`$ and
$`\mathbf 1^{\mathsf T}\mathbf q_0=Q`$,

```math
\left(\mathbf B^{\mathsf T}\mathbf H\mathbf B\right)\mathbf z
=-\mathbf B^{\mathsf T}\left(\mathbf H\mathbf q_0+\mathbf b\right),
\qquad
\mathbf H=\mathrm{diag}(\boldsymbol\eta)+\mathbf K.
```

A differentiable stability shift makes the reduced Hessian positive definite;
Cholesky and triangular solves then work on CPU, CUDA, and Apple MPS. The model
reports stationarity, charge, and stability residuals.

#### Self-consistent induced polarization

The induced-dipole interaction uses Thole damping [9]. Define

```math
u_{ij}=\frac{r_{ij}}{(\alpha_i\alpha_j)^{1/6}},
\quad
f_3=1-e^{-a u_{ij}^3},
\quad
f_5=1-(1+a u_{ij}^3)e^{-a u_{ij}^3},
```

and

```math
\mathbf T_{ij}
=\frac{k_e}{r_{ij}^3}
\left(3f_5\widehat{\mathbf r}_{ij}
\widehat{\mathbf r}_{ij}^{\mathsf T}-f_3\mathbf I\right).
```

The fixed point $`\mathbf p=\mathbf A(\mathbf E_{\mathrm{drv}}+\mathbf T\mathbf p)`$
is linear. The implementation solves its symmetric transformed system exactly,

```math
\left(\mathbf I-\mathbf A^{1/2}\mathbf T\mathbf A^{1/2}\right)\mathbf x
=\mathbf A^{1/2}\mathbf E_{\mathrm{drv}},
\qquad
\mathbf p=\mathbf A^{1/2}\mathbf x,
```

with a reported positive-definiteness shift. This is the implemented
deep-equilibrium response: it avoids retaining a long unrolled iteration graph
while preserving the equilibrium derivative.

#### Molecular DFT-D4

For non-periodic structures, the D4 sublayer delegates the charge-dependent
dispersion energy and atomic C6 coefficients to `tad-dftd4` [10]. Its
two-body damping form is schematically

```math
E_{\mathrm{D4}}^{(2)}
=-\frac{1}{2}\sum_{A\ne B}\sum_{n\in\{6,8\}}
s_n\frac{C_n^{AB}}
{R_{AB}^{n}+\left(a_1R_0^{AB}+a_2\right)^n}.
```

The current backend is molecular. Periodic structures receive no D4 energy,
and the GUI disables the switch for a dataset containing periodic structures.

### 3.5 Auxiliary electronic Hamiltonian head

The optional `WavefunctionHamiltonianHead` pools shared Response scalar
features graph by graph and predicts the $`K(K+1)/2`$ independent coefficients
of a fixed $`K\times K`$ real-symmetric matrix. A fixed symmetric basis expands
those coefficients into $`\widehat H_g`$, so Hermiticity is structural rather
than an approximate loss constraint.

This matrix represents a user-declared aligned orbital or Wannier subspace. It
is neither the total interatomic energy nor the Layer-3 classical-spin
Hamiltonian. It therefore does not enter $`E_{\mathrm{tot}}`$ and is never
reinterpreted as $`J_{ij}`$, $`D_i`$, or DMI. Its gradients can still improve
the shared Response representation when compatible labels activate WALoss.

### 3.6 Layer 3: time-reversal-aware spin Hamiltonian

A spin is an axial vector: under an orthogonal spatial transform $`\mathbf Q`$,

```math
\mathbf S_i\mapsto \det(\mathbf Q)\mathbf Q\mathbf S_i,
```

while time reversal maps $`\mathbf S_i\mapsto-\mathbf S_i`$. The implemented
spin energy is

```math
E_{\mathrm{spin}}
=-\sum_{i\lt j}J_{ij}\,\mathbf{S}_i\cdot\mathbf{S}_j
+\sum_i\mathbf{S}_i^{\mathsf{T}}\mathbf{D}_i\mathbf{S}_i
+\sum_{i\lt j}\mathbf{D}_{ij}^{\mathrm{DMI}}\cdot
\bigl(\mathbf{S}_i\times\mathbf{S}_j\bigr).
```

$`J_{ij}`$ is a scalar pair readout. $`\mathbf D_i`$ is symmetric and explicitly
made traceless. $`\mathbf D_{ij}^{\mathrm{DMI}}`$ is an axial vector assembled
from axial features and cross products of polar channels. Every spin-energy
term is even under simultaneous spin reversal. The predicted magnetic
moment is

```math
\mathbf m_i=\mathrm{softplus}(f_m(\mathbf s_i))\mathbf S_i,
```

and the effective field is the energy derivative

```math
\mathbf H_i^{\mathrm{eff}}
=-\frac{\partial E_{\mathrm{spin}}}{\partial\mathbf S_i}.
```

### 3.7 Cross-granularity FiLM coupling

The first domain/spin pass produces a four-component condition at every atom,

```math
\mathbf c_i=\left[
\tanh(q_i),
\tanh(\phi_i/10),
\|\mathbf S_i\|^2,
\underset{j\in\mathcal N(i)}{\mathrm{mean}}
(\mathbf S_i\cdot\mathbf S_j)
\right].
```

Each atomic interaction block maps $`\mathbf c_i`$ to scalar scale, scalar bias,
and tensor scale. The actual bounded modulation is

```math
\mathbf s_i\leftarrow
\left[1+0.25\tanh\boldsymbol\gamma_i^{(s)}\right]\odot\mathbf s_i
+\boldsymbol\beta_i^{(s)},
```

```math
\mathbf X_i^{(L)}\leftarrow
\left[1+0.25\tanh\boldsymbol\gamma_i^{(L)}\right]
\odot\mathbf X_i^{(L)},
\quad
\mathbf X^{(L)}\in
\{\mathbf v,\mathbf a,\mathbf T^{(2)},\mathbf T^{(3)}\}.
```

The coupled forward pass recomputes atomic, response, QEq, and spin quantities
for a bounded number of outer iterations. It stops early when the graph-wise
mean charge change is below the coupling tolerance.

```mermaid
sequenceDiagram
    participant L1 as Atomic layer
    participant R as Response heads
    participant L2 as QEq / polarization
    participant L3 as Spin Hamiltonian
    participant F as FiLM generator
    L1->>R: scalar, polar, axial, tensor features
    R->>L2: chi, hardness, alpha, permanent dipoles
    R->>L3: geometry-conditioned features
    L2-->>F: charge and electrostatic potential
    L3-->>F: spin invariants
    F-->>L1: bounded scale and bias
    L1->>L1: refine local representation
```

### 3.8 Energy assembly and derivative observables

The complete implemented Hamiltonian is

```math
E_{\mathrm{tot}}
=E_{\mathrm{short}}+E_{\mathrm{QEq}}+E_{\mathrm{PME}}
+E_{\mathrm{D4}}+E_{\mathrm{spin}}+E_{\mathrm{resp}}.
```

The dipole-field term is not double-counted: when QEq is active, the charge
coupling to the field is already contained in the QEq linear potential.
Observables are differentiated after assembling this total energy:

```math
\mathbf F_i=-\frac{\partial E_{\mathrm{tot}}}{\partial\mathbf R_i},
\qquad
\boldsymbol\sigma=\frac{1}{V}\mathrm{sym}
\frac{\partial E_{\mathrm{tot}}}{\partial\boldsymbol\varepsilon},
\qquad
Z^{*}_{i,\alpha\beta}
=\frac{\partial\mu_\alpha}{\partial R_{i\beta}},
\qquad
\mathbf H_i^{\mathrm{eff}}
=-\frac{\partial E_{\mathrm{spin}}}{\partial\mathbf S_i}.
```

## 4. Data generation and training strategy

### 4.1 Mixed-label HDF5 representations

Neo uses two self-contained HDF5 layouts. In canonical `e3mu-hdf5-v1`, atomic
arrays are concatenated and indexed by `structures/atom_ptr`; `labels/` stores
dense or packed physical tensors, `masks/` states which rows are valid, and
`metadata/` stores source, method, group-safe split, and provenance. The mask,
not a placeholder value, determines whether a target contributes to training.

Composite `e3mu-composite-hdf5-v1` retains those four root groups for an
embedded complete Standard or Large response payload. It additionally stores a
deterministic OMat24 foundation under `sources/omat24/packed/`, its selected
source shard and row under `selection/`, and OMat24 atomic-reference statistics
under `atomic_reference/`. Positions, cells, energies, forces, and stress retain
their source float64 values; an external Parquet tree is not required at
training time.

```mermaid
flowchart LR
    U[Upstream datasets] --> C[Source-specific conversion]
    C --> V[Units and finite-value validation]
    V --> G[Physical parent grouping]
    G --> S[Fixed train / val / test split]
    S --> M[Label masks and energy-domain policy]
    M --> H[e3mu-hdf5-v1]
    H --> T[Tiny, Small, Standard, Large]
    T --> X[e3mu-composite-hdf5-v1]
    O[Packed OMat24 foundation] --> X
    X --> P[SE, Plus, Max]
```

The complete seven-tier materialized inventory is:

| Tier | Schema | Embedded Neo response | OMat24 foundation | Total structures | Total atoms | On-disk size |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Tiny | canonical | 5,780 | 0 | 5,780 | 394,755 | 20.011 MiB |
| Small | canonical | 16,703 | 0 | 16,703 | 1,069,318 | 51.218 MiB |
| Standard | canonical | 46,414 | 0 | 46,414 | 2,316,736 | 0.120772 GiB |
| SE | composite | Standard: 46,414 | 559,279 | 605,693 | 12,767,209 | 0.690297 GiB |
| Large | canonical | 613,267 | 0 | 613,267 | 17,760,024 | 1.219244 GiB |
| Plus | composite | Large: 613,267 | 25,206,004 | 25,819,271 | 488,227,614 | 38.237181 GiB |
| Max | composite | Large: 613,267 | 100,670,282 | 101,283,549 | 1,899,323,661 | 128.559024 GiB |

Tiny and Small are deterministic nested subsets of Standard. Large follows a
separate trajectory-rich policy. SE combines the complete Standard response
corpus with an exact `1/180` Max OMat24 selection; Plus and Max embed complete
Large. Max contains 100,670,282 unique OMat24 configurations after 154,252
duplicate configuration IDs are removed.

Mask-derived response coverage is not inferred from file size:

| Target family | Tiny | Small | Standard | Large |
| --- | ---: | ---: | ---: | ---: |
| Energy and forces | 1,912 | 8,131 | 22,761 | 505,736 |
| Cauchy stress | 2,024 | 8,243 | 22,873 | 505,848 |
| Field and total charge | 3,768 | 8,472 | 23,553 | 107,431 |
| Dipole | 3,145 | 7,810 | 22,891 | 106,769 |
| Charges and atomic dipoles | 2,011 | 4,844 | 18,130 | 101,993 |
| Molecular/atomic polarizability and C6 | 302 | 406 | 4,060 | 43,430 |
| Born effective charge | 623 | 662 | 662 | 662 |
| Clamped-ion piezoelectric tensor | 112 | 112 | 112 | 112 |
| Spins and magnetic moments | 1,074 | 4,320 | 12,100 | 73,029 |
| Effective spin field | 100 | 100 | 100 | 100 |
| Stress + spins | 974 | 4,220 | 12,000 | 72,929 |
| Paired magnetoelastic stress | 0 | 0 | 0 | 0 |
| Paired WALoss matrices | 0 | 0 | 0 | 0 |

Composite response rows inherit exactly the Standard or Large coverage shown
above; OMat24 foundation rows add only compatible energy, force, and stress.
The seven current files contain no `orbital_hamiltonian` or
`orbital_eigenvectors` arrays and omit the optional `wavefunction_dim`
attribute. WALoss is therefore an implemented data-contract extension, not a
label claimed for the current Neo release.

Direct $`J`$, $`D_i`$, and DMI aggregate labels are absent from the portable tiers;
their masks remain false. This is not interpreted as a zero physical value.
The repository includes the Tiny tier, while the larger canonical and
Composite tiers and their release metadata are distributed through the
[project dataset on Hugging Face](https://huggingface.co/datasets/FonaTech/E3-miu-GNN).

![Dataset tiers](assets/generated/dataset-tiers.png)

### 4.2 Source and energy-domain policy

Neo aggregates MPtrj, JARVIS-DFT, QM7-X, SO3LR families, SCFNN, DeepSPIN NiO,
and locally supplied BEC calculations. These sources do not share one absolute
electronic-structure reference. The mixed corpus therefore activates the
shared energy/force loss only for compatible MPtrj records and retains other
sources for response- or spin-specific labels. Related trajectory frames,
conformers, field variants, or magnetic blocks share one group and cannot cross
split boundaries.

### 4.3 Mask-aware objective

For target $`t`$, mask $`m_{t,k}`$, prediction $`\widehat{\mathbf{y}}_{t,k}`$, and
reference $`\mathbf{y}_{t,k}`$, the training objective is

```math
\mathcal L
=\sum_{t\in\mathcal T}w_t
\frac{\sum_k m_{t,k}
\left\|\widehat{\mathbf y}_{t,k}-\mathbf y_{t,k}\right\|_2^2}
{\sum_k m_{t,k}\,d_t},
```

where $`d_t`$ is the number of components per labelled item. The implemented
target set includes energy, forces, conservative Cauchy stress, dipole, molecular and atomic
polarizability, charges, atomic dipoles, C6, BEC, magnetic moments, effective
spin fields, and available $`J/D_i`$/DMI targets. Energy loss is evaluated per
atom so large cells do not dominate solely by size. Stress uses only fully
periodic nonsingular cells and the six independent symmetric components.

Checkpoint selection and Auto Research use a weight-independent normalized
score,

```math
S_{\mathrm{val}}
=\frac{1}{|\mathcal T_{\mathrm{active}}|}
\sum_{t\in\mathcal T_{\mathrm{active}}}
\frac{\mathrm{MAE}_t}{s_t},
```

with fixed characteristic scales $`s_t`$. A candidate cannot improve its ranking
merely by reducing its own loss weight.

### 4.4 Wavefunction-aligned Hamiltonian supervision

Uniform raw-matrix MSE does not distinguish perturbations that shift reference
orbital energies from those that mix reference states. This distinction is
especially important near a small gap because first-order state mixing contains
the inverse energy denominator

```math
c_{j\leftarrow i}\simeq
\frac{\langle u_j^*|\Delta H|u_i^*\rangle}
{\epsilon_i^*-\epsilon_j^*}.
```

For each active graph $`g`$, WALoss transforms prediction and reference with the
columns of the supplied reference eigenvector matrix $`U_g^*`$:

```math
\widetilde H_g=(U_g^*)^\dagger\widehat H_gU_g^*,
\qquad
\widetilde H_g^*=(U_g^*)^\dagger H_g^*U_g^*
=\mathrm{diag}(\epsilon_{g,1}^*,\ldots,\epsilon_{g,K}^*).
```

Its two independently normalized terms are

```math
\mathcal L_{\mathrm{diag}}
=\frac{\sum_gm_g\sum_i
|\widetilde H_{g,ii}-\widetilde H_{g,ii}^*|^2}
{\sum_gm_gK},
```

```math
\mathcal L_{\mathrm{off}}
=\frac{\sum_gm_g\sum_{i\lt j}
|\widetilde H_{g,ij}-\widetilde H_{g,ij}^*|^2}
{\sum_gm_gK(K-1)/2},
\qquad
\mathcal L_{\mathrm{WA}}
=\lambda_{\mathrm d}\mathcal L_{\mathrm{diag}}
+\lambda_{\mathrm o}\mathcal L_{\mathrm{off}}.
```

The strict upper triangle prevents symmetric double counting; separate
normalization prevents the $`O(K^2)`$ coupling population from overwhelming the
$`O(K)`$ energy population. The complete trainer adds
$`w_{\mathrm{waloss}}\mathcal L_{\mathrm{WA}}`$ only for paired active masks in
Response or Joint training. It never diagonalizes $`\widehat H_g`$, so gradients
do not traverse an eigensolver.

Reference ingestion verifies finite values, symmetry, orthonormality, and that
$`U_g^*`$ diagonalizes $`H_g^*`$ within a scale-aware tolerance. These numerical
checks do not create a physical gauge. All rows in a training domain must share
one ordered orbital/Wannier subspace, phase or degenerate-subspace convention,
spin and k-point convention, energy zero, and electronic-structure method. The
current Neo tiers have zero paired WALoss records, so this manuscript reports
implementation and deterministic loss tests but no WALoss production-accuracy
result.

### 4.5 Optimization modes

The trainer supports a ground-state base stage, a response stage, and joint
fine-tuning. The full-chain workflow can freeze the ground branch during
response warmup, assign separate branch learning rates, ramp response weights,
and progressively reduce the joint learning rate. On Apple MPS, batches are
packed by edge count because force, stress, and BEC supervision require higher-order
autograd graphs whose memory cost follows edges more closely than structure
count.

Auto Research locks the user-selected architecture. Dataset masks and
periodicity remove meaningless loss and solver dimensions before search. A
random exploration phase is followed by a lightweight Gaussian-process
surrogate; the winning searched values are applied to the GUI only after an
explicit user action.

## 5. Validation and results

### 5.1 Deterministic physical tests

The float64 self-test compares transformed predictions and finite-difference
derivatives. With the documented seed 7, the current maximum errors are:

| Check | Maximum error |
| --- | ---: |
| Rotation: energy | 0 |
| Rotation: force | $`3.47\times10^{-18}`$ |
| Rotation: dipole | $`2.61\times10^{-15}`$ |
| Rotation: polarizability | $`2.22\times10^{-16}`$ |
| Reflection: energy/force/dipole/polarizability | 0 |
| Time reversal: spin energy/effective field | 0 |
| Charge conservation | 0 e |
| QEq stationarity residual | $`9.39\times10^{-12}`$ |
| Conservative-force finite difference | $`8.15\times10^{-12}`$ eV/A |

![Physics validation](assets/generated/physics-self-tests.png)

The repository regression suite additionally checks MPS-specific QEq solves,
differentiable PME and D4 references, DEQ gradients, Layer-3 supervised
gradients, checkpoint safety, HDF5 invariants, dataset-aware GUI state, and
magnetic VASP mapping. WALoss-focused tests cover diagonal/off-diagonal
normalization, common-basis invariance, differentiability without a predicted
eigensolver, paired-mask validation, checkpoint warm starts, and its separation
from the classical-spin Hamiltonian.

### 5.2 Small held-out benchmarks

These experiments are deliberately short functional baselines.

| Dataset and split | Training scope | Held-out result |
| --- | --- | --- |
| QM7-X, 8 test molecules | 12 epochs; energy, dipole, polarizability, charge, atomic polarizability | energy 1.907 eV/system; dipole 0.1313 eA/component; polarizability 0.7217 A3/component; charge 0.0949 e/atom; atomic polarizability 0.3089 A3/component |
| BEC, 4 validation cells / 768 atoms | 2 epochs | BEC MAE 0.2156 e/component |
| SCFNN, 4 validation cells / 768 atoms | 20 epochs | dipole MAE 2.435 eA/component versus zero baseline 2.972 eA/component |

The QM7-X force and C6 heads had zero loss weight and are not reported as
trained accuracy. The QEq model required a mean test stability shift of
14.39 eV in that short run, indicating that its learned raw hardness was not
yet physically calibrated.

### 5.3 Memory behavior

A five-epoch Apple MPS run with energy, force, dipole, and polarizability
losses increased process RSS by 16.3 MiB from epoch 1 to epoch 5. Active MPS
tensors remained near 30.8 MiB after cleanup, and no sustained-growth warning
was triggered.

![Memory profile](assets/generated/memory-profile.png)

## 6. Discussion and limitations

The tests establish structural validity: the selected representation channels
transform correctly, charges obey the graph constraint, spin energy respects
time reversal, and force/BEC/spin derivatives remain connected to the model
energy. They do not establish uniform predictive accuracy over all 94 supported
elements or every source domain.

Several boundaries are material when interpreting results:

- The current D4 backend is molecular and is not a periodic dispersion model.
- Direct $`J`$, $`D_i`$, and DMI labels are not present in the portable Neo tiers;
  the Layer-3 Hamiltonian is functionally and symmetry validated but does not
  yet have a paper-grade cross-material calibration result.
- WALoss and its real-symmetric auxiliary head are implemented, but the current
  Neo tiers contain no aligned Hamiltonian/eigenvector pairs. No orbital-energy
  or orbital-coupling accuracy is claimed. The fixed-$`K`$ head is not a
  variable-band, k-resolved, real-space-wavefunction, or strong-correlation
  solver.
- Absolute energies from different electronic-structure methods remain
  separately masked instead of being forced into a common zero.
- Born effective charge tensors are retained as published; the dataset reports
  acoustic-sum residuals but does not silently project labels.
- Hessians are available through automatic differentiation of the energy, but
  this work does not present a converged phonon-spectrum benchmark.
- Current accuracy tables are small smoke benchmarks and must not be described
  as production potential performance.

## 7. Conclusion

E(3)-mu-GNN is an energy-based, mixed-granularity atomistic model. Its atomic
layer provides explicit O(3)
parity channels; its domain layer turns predicted electronic parameters into
constrained electrostatic, polarization, and dispersion energies; its spin
layer supplies a time-reversal-consistent magnetic Hamiltonian; and FiLM
allows domain and spin state to refine local messages. Optional WALoss adds a
reference-eigenspace electronic objective without conflating that auxiliary
matrix with total or spin energy. A mask-aware two-schema dataset contract and
derivative-based validation keep the physical meaning of each target visible.
The implementation is therefore a complete research platform
for controlled L1-L3 experiments, while its short benchmarks and unresolved
data redistribution item set clear limits on current accuracy and release
claims.

## Code, data, and licensing statement

The implementation is in `E3_miu_GNN.py` and is released under
MIT terms. Neo dataset binaries are not covered by the software license.
MPtrj, JARVIS-DFT, QM7-X, SO3LR, SCFNN, and DeepSPIN retain their respective
upstream terms. The binaries are hosted in the project Hugging Face dataset;
the standalone redistribution terms of the supplied BEC archive remain under
review. Large-scale pretraining is in progress, and validated pretrained
checkpoints are planned for a later project release. Exact source declarations
and transformations are in `Datasets/Neo/SOURCES_AND_PROCESSING.md`.

## References

1. Y. Zhan, *E(3)-mu-GNN*, open-source software repository (2026), <https://github.com/FonaTech/E3-miu-GNN>.
2. P. Hohenberg and W. Kohn, "Inhomogeneous Electron Gas," *Physical Review* **136**, B864-B871 (1964), <https://doi.org/10.1103/PhysRev.136.B864>.
3. W. Kohn and L. J. Sham, "Self-Consistent Equations Including Exchange and Correlation Effects," *Physical Review* **140**, A1133-A1138 (1965), <https://doi.org/10.1103/PhysRev.140.A1133>.
4. I. Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields," *NeurIPS* (2022), <https://doi.org/10.48550/arXiv.2206.07697>.
5. S. Batzner et al., "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials," *Nature Communications* **13**, 2453 (2022), <https://doi.org/10.1038/s41467-022-29939-5>.
6. A. K. Rappe and W. A. Goddard III, "Charge equilibration for molecular dynamics simulations," *The Journal of Physical Chemistry* **95**, 3358-3363 (1991), <https://doi.org/10.1021/j100161a070>.
7. T. Darden, D. York, and L. Pedersen, "Particle mesh Ewald: An N log(N) method for Ewald sums in large systems," *The Journal of Chemical Physics* **98**, 10089 (1993), <https://doi.org/10.1063/1.464397>.
8. U. Essmann et al., "A smooth particle mesh Ewald method," *The Journal of Chemical Physics* **103**, 8577 (1995), <https://doi.org/10.1063/1.470117>.
9. B. T. Thole, "Molecular polarizabilities calculated with a modified dipole interaction," *Chemical Physics* **59**, 341-350 (1981), <https://doi.org/10.1016/0301-0104(81)85176-2>.
10. E. Caldeweyher et al., "A generally applicable atomic-charge dependent London dispersion correction," *The Journal of Chemical Physics* **150**, 154122 (2019), <https://doi.org/10.1063/1.5090222>.
11. E. Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," *AAAI* (2018), <https://doi.org/10.48550/arXiv.1709.07871>.
12. B. Deng et al., "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling," *Nature Machine Intelligence* **5**, 1031-1041 (2023), <https://doi.org/10.1038/s42256-023-00716-3>.
13. J. Hoja et al., "QM7-X, a comprehensive dataset of quantum-mechanical properties spanning the chemical space of small organic molecules," *Scientific Data* **8**, 43 (2021), <https://doi.org/10.1038/s41597-021-00812-2>.
14. A. Gao and R. C. Remsing, "Self-consistent determination of long-range electrostatics in neural network potentials," *Nature Communications* **13**, 1572 (2022), <https://doi.org/10.1038/s41467-022-29243-2>.
15. T. Yang et al., "Screening Spin Lattice Interaction Using Deep Learning Approach," arXiv (2023), <https://doi.org/10.48550/arXiv.2304.09606>.
