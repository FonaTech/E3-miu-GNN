# Physical Mechanisms

This document expands the Hamiltonians and solver details in Sections 3.3-3.7
of the [paper](PAPER.md).

## Units and conventions

| Quantity | Internal/public unit |
| --- | --- |
| Position | angstrom |
| Energy | eV |
| Force | eV/angstrom |
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
=-\mu\cdot\mathcal E
-\frac{1}{2}c_\alpha\mathcal E^{\mathsf T}\alpha\mathcal E.
```

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
=-\sum_{i<j}J_{ij}S_i\cdot S_j
+\sum_iS_i^{\mathsf T}D_iS_i
+\sum_{i<j}D_{ij}^{\mathrm{DMI}}\cdot(S_i\times S_j).
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
Z^*_{i,\alpha\beta}=\frac{\partial\mu_\alpha}{\partial R_{i\beta}},
\qquad
H_i^{\mathrm{eff}}=-\frac{\partial E_{\mathrm{spin}}}{\partial S_i}.
```

Force training requires differentiating these first derivatives with respect
to model parameters. BEC supervision similarly requires a higher-order graph.
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
| `coupling_residual` | graph mean charge change between FiLM passes |

A finite output is necessary but not sufficient for physical calibration.
Residual and stability histories are therefore exposed in checkpoints, JSON
artifacts, and the live GUI.
