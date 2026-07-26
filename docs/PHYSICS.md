# Physical Mechanisms

This document expands the Hamiltonians and solver details in Sections 3.3-3.7
of the [paper](PAPER.md).

## Units and conventions

| Quantity | Internal/public unit |
| --- | --- |
| Position | angstrom |
| Energy | eV |
| Force | eV/angstrom |
| Cauchy stress | eV/angstrom$^3$, tensile-positive |
| Charge and BEC | elementary charge $`e`$ |
| Dipole | $`e\,\mathrm{angstrom}`$ |
| Polarizability | $`\mathrm{angstrom}^3`$ |
| C6 | eV $`\mathrm{angstrom}^6`$ |
| Electric field | V/angstrom |
| Magnetic moment | $`\mu_B`$ |
| Spin Hamiltonian parameters | eV |

The implementation uses

```math
k_e=14.3996454784255\ \mathrm{eV\,angstrom}/e^2
```

and converts a polarizability volume to field-response energy with

```math
c_\alpha=0.06944615422483141
\ \frac{\mathrm{eV}}{\mathrm{angstrom}^3
(\mathrm{V}/\mathrm{angstrom})^2}.
```

## Electric perturbation response

For a static field $`\mathcal E`$, the implemented second-order response is

```math
E_{\mathrm{resp}}
=-\mu_{\mathrm{permanent}}\cdot\mathcal E
-\frac{1}{2}c_\alpha\mathcal E^{\mathsf T}\alpha\mathcal E.
```

The reported field-dependent dipole is the exact conjugate response,

```math
\mu_{\mathrm{reported}}
=-\frac{\partial E_{\mathrm{resp}}}{\partial\mathcal E}
=\mu_{\mathrm{permanent}}+c_\alpha\alpha\mathcal E.
```

Using the reported dipole in the linear energy term would double count the
induced response, so the permanent and induced contributions remain explicit.

The response head predicts permanent atomic dipoles and polarizabilities from
equivariant channels. Total dipole includes charge displacement and, when the
polarization solver is active, induced dipoles:

```math
\mu=\sum_i\mu_i^{\mathrm{perm}}
+\sum_iq_i(R_i-R_c)+\sum_ip_i^{\mathrm{ind}}.
```

The factor $`c_\alpha`$ is omitted only when a configuration explicitly declares
that polarizability is already in energy-per-field-squared units.

## Charge equilibration

### Variational model

The learned electronegativity $`\chi_i`$ and positive hardness $`\eta_i`$ define

```math
E(q)=\sum_i\left(\chi_iq_i+\frac{1}{2}\eta_iq_i^2
+\phi_i^{\mathrm{ext}}q_i\right)
+\frac{1}{2}\sum_{i\ne j}q_iK_{ij}q_j,
```

subject to

```math
\sum_iq_i=Q_{\mathrm{graph}}.
```

Without periodic PME,

```math
K_{ij}=\frac{k_e}{\sqrt{r_{ij}^2+\sigma^2}},\quad K_{ii}=0.
```

The external scalar potential is $`\phi_i^{\mathrm{ext}}=-R_i\cdot\mathcal E`$
after centering non-periodic coordinates.

### Exact constrained solve

Let $`H=\mathrm{diag}(\eta)+K`$, $`b=\chi+\phi^{\mathrm{ext}}`$,
and let $`B`$ span the neutral subspace. The exact constrained variable is

```math
q=q_0+Bz,\qquad
(B^{\mathsf T}HB)z=-B^{\mathsf T}(Hq_0+b).
```

`DifferentiableQEq` constructs $`B`$ analytically as a Helmert basis. This avoids
QR on Apple MPS and avoids the indefinite backward path of a KKT/LU solve. A
minimum-eigenvalue check is evaluated robustly on CPU float64. Its eigenvector
defines an on-device Rayleigh quotient so first derivatives remain connected
to the original tensor. Gershgorin's lower bound is the fallback if the
eigensolver fails.

The applied stability shift is

```math
\delta=\max(0,\lambda_{\mathrm{floor}}
-\lambda_{\min}(B^{\mathsf T}HB)).
```

The reported residual is the maximum of stationarity error and charge error.
A large stability shift is a diagnostic that the learned hardness/kernel is
not physically calibrated, even when the final solve is numerically finite.

## Periodic Ewald/PME

When PME is active and at least one cell axis is periodic, the charge response
kernel is built by applying the Ewald calculator to identity basis charges. It
therefore represents the full linear charge-to-potential operator rather than
only evaluating one current charge vector. The kernel is symmetrized before
the QEq solve.

The reciprocal term follows

```math
E_{\mathrm{rec}}
=\frac{1}{2\Omega}\sum_{k\ne0}
\frac{4\pi k_e}{k^2}e^{-k^2/(4\alpha_E^2)}|S(k)|^2.
```

`qeq_pme_smearing` controls the Ewald splitting and
`qeq_pme_lr_wavelength` controls reciprocal resolution. The implementation
uses `torch-pme`; its tested reference agrees with a direct `torch-pme` Ewald
calculation to numerical precision.

## Thole-damped polarization equilibrium

### Short-range damping

Bare point polarizabilities can diverge at short separation. For isotropic
volumes $`\alpha_i`$, the dimensionless separation and damping factors are

```math
u_{ij}=\frac{r_{ij}}{(\alpha_i\alpha_j)^{1/6}},
\quad f_3=1-e^{-au_{ij}^3},
\quad f_5=1-(1+au_{ij}^3)e^{-au_{ij}^3}.
```

The interaction tensor is

```math
T_{ij}=\frac{k_e}{r_{ij}^3}
\left(3f_5\widehat r_{ij}\widehat r_{ij}^{\mathsf T}-f_3I\right).
```

The driving field includes the applied field, the damped charge field, and the
field of permanent atomic dipoles.

### Symmetric exact equilibrium

With block-diagonal polarizability $`A`$, solve

```math
(I-A^{1/2}TA^{1/2})x=A^{1/2}E_{\mathrm{drv}},
\qquad p=A^{1/2}x.
```

The code calls this the DEQ polarization layer because it evaluates the fixed
point and its implicit equilibrium derivative. The linear form is solved once
with Cholesky instead of unrolling up to `deq_max_iter`. The `deq_iterations`
diagnostic is consequently one for each solved graph. `deq_tol` and
`deq_max_iter` remain checkpoint/config compatibility fields, while residual
and stability shift are the operative quality diagnostics of the exact solve.

## DFT-D4 dispersion

The molecular D4 layer obtains atomic-charge-dependent dispersion from
`tad-dftd4`. Its conceptual two-body energy is

```math
E_{\mathrm{D4}}^{(2)}
=-\frac{1}{2}\sum_{A\ne B}\sum_{n=6,8}
s_n\frac{C_n^{AB}}{R_{AB}^n+f_{\mathrm{damp},n}^{AB}}.
```

Coordinates are converted from angstrom to bohr and energy from Hartree to eV.
On Apple MPS, D4 runs as a differentiable CPU sublayer because its reference
tables require float64; explicit transfers preserve first- and second-order
gradients. The current molecular API does not include lattice images, so
periodic D4 is deliberately inactive rather than physically misrepresented.

## Spin Hamiltonian

The Layer-3 energy is

```math
E_{\mathrm{spin}}
=-\sum_{i\lt j}J_{ij}S_i\cdot S_j
+\sum_iS_i^{\mathsf T}D_iS_i
+\sum_{i\lt j}D_{ij}^{\mathrm{DMI}}\cdot(S_i\times S_j).
```

### Heisenberg exchange

$`J_{ij}`$ is a scalar readout of symmetric pair features:

```math
x_{ij}=[s_i+s_j,|s_i-s_j|,\mathrm{RBF}(r_{ij})],
\qquad J_{ij}=f_J(x_{ij}).
```

### Single-ion anisotropy

An $`L=2`$ readout is mapped back to Cartesian form, symmetrized, and made
traceless:

```math
D_i\leftarrow\frac{D_i+D_i^{\mathsf T}}{2}
-\frac{\mathrm{tr}D_i}{3}I.
```

### Dzyaloshinskii-Moriya interaction

The DMI vector must be axial. It is assembled from learned axial features and
cross products of learned polar features. DMI is therefore allowed only with
O(3) parity and explicit DMI activation.

### Time reversal

Under $`S_i\mapsto-S_i`$,

```math
S_i\cdot S_j\mapsto S_i\cdot S_j,
\quad
S_i^{\mathsf T}D_iS_i\mapsto S_i^{\mathsf T}D_iS_i,
\quad
S_i\times S_j\mapsto S_i\times S_j.
```

Thus $`E_{\mathrm{spin}}`$ is even, while
$`H_i^{\mathrm{eff}}=-\partial E_{\mathrm{spin}}/\partial S_i`$ is odd. Both
properties are exact in the deterministic self-test.

## Wavefunction alignment and electronic Hamiltonian

### Why a matrix-element loss is not enough

Let a reference electronic Hamiltonian in a fixed $`K`$-orbital subspace be
$`H^*`$, and let $`\widehat H=H^*+\Delta H`$ be the model prediction. A uniform
MSE over raw matrix entries treats a shift of an orbital energy and a coupling
between two reference eigenstates as interchangeable numerical errors. They
are not interchangeable physically.

For Hermitian perturbations, Weyl's bound gives

```math
|\widehat\epsilon_i-\epsilon_i^*|
\leq \|\Delta H\|_2
\leq \|\Delta H\|_{\mathrm F}.
```

If many entries each carry an error of scale $`\delta`$, the Frobenius norm can
grow as $`O(K\delta)`$. Therefore small-looking element errors do not by
themselves guarantee small orbital-energy errors as the aligned subspace grows.
More importantly, first-order nondegenerate perturbation theory gives the
mixing of reference state $`i`$ into state $`j`$ as

```math
c_{j\leftarrow i}
\simeq
\frac{\langle u_j^*|\Delta H|u_i^*\rangle}
{\epsilon_i^*-\epsilon_j^*}.
```

An off-diagonal residual can consequently rotate an eigenspace strongly when
the energy gap is small even if its absolute magnitude is modest. This is the
physical motivation for separating orbital-energy and orbital-mixing errors.

### Implemented aligned objective

For graph $`g`$, the columns of $`U_g^*`$ are the orthonormal reference
eigenvectors of $`H_g^*`$. The model's optional
`WavefunctionHamiltonianHead` predicts a real-symmetric $`K\times K`$ matrix
$`\widehat H_g`$. Prediction and reference are rotated into the same reference
eigenspace:

```math
\widetilde H_g=(U_g^*)^\dagger\widehat H_gU_g^*,
\qquad
\widetilde H_g^*=(U_g^*)^\dagger H_g^*U_g^*
=\mathrm{diag}(\epsilon_{g,1}^*,\ldots,\epsilon_{g,K}^*).
```

The diagonal term measures orbital-energy error directly,

```math
\mathcal L_{\mathrm{diag}}
=\frac{\sum_gm_g\sum_i
|\widetilde H_{g,ii}-\widetilde H_{g,ii}^*|^2}
{\sum_gm_gK},
```

and the strict upper triangle measures residual coupling between distinct
reference states,

```math
\mathcal L_{\mathrm{off}}
=\frac{\sum_gm_g\sum_{i\lt j}
|\widetilde H_{g,ij}-\widetilde H_{g,ij}^*|^2}
{\sum_gm_gK(K-1)/2}.
```

The complete auxiliary objective is

```math
\mathcal L_{\mathrm{WA}}
=\lambda_{\mathrm d}\mathcal L_{\mathrm{diag}}
+\lambda_{\mathrm o}\mathcal L_{\mathrm{off}}.
```

Using only the strict upper triangle avoids counting the symmetric element
twice. Independent normalization is also essential: without it, the
$`K(K-1)/2`$ coupling entries would increasingly dominate the $`K`$ diagonal
entries as $`K`$ grows.

This construction is physics-informed in the PINN sense that an eigenproblem
determines the residual coordinates and their physical interpretation. It is
not a discretized Schrödinger-equation residual, a Kohn-Sham solver, or a claim
to recover the many-electron wavefunction. The prediction is not diagonalized
during training. Gradients pass through the fixed products with $`U_g^*`$ and
the symmetric head, avoiding eigensolver derivatives and their ambiguity at
degeneracy.

### Gauge, degeneracy, and scope

A common unitary change of raw basis applied consistently to $`\widehat H_g`$,
$`H_g^*`$, and $`U_g^*`$ leaves the aligned matrices unchanged. This invariance
does not repair inconsistent dataset gauges. All structures in one training
domain must use the same fixed orbital/Wannier subspace, orbital order, energy
zero, spin channel, k-point convention, and electronic-structure method. Phase
choices must be consistent; near an exact or numerical degeneracy, the whole
degenerate subspace should be aligned before assigning individual eigenvectors
or a large off-diagonal weight.

The current implementation has these deliberate boundaries:

- one fixed graph-level real-symmetric matrix dimension $`K`$ per dataset and
  checkpoint, rather than a variable-band or k-resolved container;
- a separate electronic auxiliary head, never a reinterpretation of the
  time-reversal spin $`J/D_i`$/DMI Hamiltonian;
- no direct contribution from $`\widehat H_g`$ to $`E_{\mathrm{tot}}`$, forces,
  or stress; WALoss influences shared learned features only through training;
- paired masks: a sample supplies both `orbital_hamiltonian` and
  `orbital_eigenvectors`, or neither; and
- no active WALoss labels in the current Neo Tiny, Small, Standard, SE, Large,
  Plus, or Max files.

The last point is a data boundary, not an implementation gap. WALoss becomes a
scientific training target only after a compatible gauge-aligned electronic
dataset is collected and independently validated.

## Conservative derivative observables

All enabled components are summed before differentiation:

```math
E_{\mathrm{tot}}
=E_{\mathrm{short}}+E_{\mathrm{QEq}}+E_{\mathrm{PME}}
+E_{\mathrm{D4}}+E_{\mathrm{spin}}+E_{\mathrm{resp}}.
```

The derivative contract is

```math
F_i=-\frac{\partial E_{\mathrm{tot}}}{\partial R_i},
\qquad
\boldsymbol\sigma=\frac{1}{V}\mathrm{sym}
\frac{\partial E_{\mathrm{tot}}}{\partial\boldsymbol\varepsilon},
\qquad
Z^*_{i,\alpha\beta}=\frac{\partial\mu_\alpha}{\partial R_{i\beta}},
\qquad
H_i^{\mathrm{eff}}=-\frac{\partial E_{\mathrm{spin}}}{\partial S_i}.
```

Force and stress training require differentiating these first derivatives with
respect to model parameters. Positions, cell, and periodic image shifts receive
the same affine strain, and stress is masked for cells that are not fully
periodic and nonsingular. BEC supervision similarly requires a higher-order graph.
This explains why MPS batches are limited by edge count and why graph references
are released immediately after every optimizer step.

## Solver diagnostics

| Output | Interpretation |
| --- | --- |
| `qeq_residual` | stationarity or total-charge error |
| `qeq_stability_shift` | curvature added in the neutral charge subspace |
| `deq_residual` | residual of the stabilized induced-dipole linear system |
| `deq_stability_shift` | curvature added to the polarization Hessian |
| `deq_iterations` | number of equilibrium solves, currently one |
| `coupling_residual_electric` | graph mean charge change between FiLM passes |
| `coupling_residual_spin` | graph mean magnetic-moment change between FiLM passes |
| `coupling_residual` | maximum of active electric and spin residuals |

A finite output is necessary but not sufficient for physical calibration.
Residual and stability histories are therefore exposed in checkpoints, JSON
artifacts, and the live GUI.
