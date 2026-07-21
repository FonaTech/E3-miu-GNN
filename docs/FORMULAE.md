# Mathematical Specification and Code Map

This document collects the equations used to explain E(3)-mu-GNN and maps each
implemented relation to its code location. Background electronic-structure
equations are identified explicitly and are not presented as neural-network
solvers.

## Rendering and verification

- Display equations use GitHub's fenced `math` syntax so Markdown list parsing
  cannot split continuation lines beginning with `+` or `-`.
- Symbols, tensor order, vector products, constraints, and equation numbers
  were checked against the implementation and the manuscript.
- Only GitHub-supported math commands are used; unsupported macros such as
  `operatorname` are avoided.

![E(3)-mu-GNN architecture](assets/proposal/mixed-granularity-core.png)

## Formula map

| Formula | Subject | Paper equation(s) | Project status | Primary code location |
| ---: | --- | --- | --- | --- |
| 1 | many-body, Kohn-Sham, density, XC, LDA/GGA | (1)-(2) | scientific background only | training-label provenance, no DFT solver |
| 2 | Kohn-Sham versus quasiparticle gap | (3) | scientific background only | no electronic gap solver |
| 3 | DFT+U correction | (4) | dataset-method context | VASP job metadata; no DFT+U solver |
| 4 | Hessian and dynamical matrix | (11) | derivative interface implemented | `MixedGranularityE3GNN`, autograd outputs |
| 5 | equivariant tensor-product message | (7)-(9) | implemented in selected real Cartesian O(3) products | `FastEquivariantCoreO3` |
| 6-8 | field perturbation and second-order energy expansion | (12) | implemented through second order | response-energy assembly |
| 9 | effective Hamiltonian, dipole, polarizability | (12)-(14) | implemented with explicit unit conversion | response heads and total-energy assembly |
| 10-11 | force and BEC derivatives under field | (32) | implemented by autograd | `MixedGranularityE3GNN.forward` |
| 12 | reciprocal electrostatics and equilibrium polarization | (18), (20)-(22) | implemented with `torch-pme` and a Thole solve | `DifferentiableQEq`, `SelfConsistentPolarization` |
| 13 | mask-aware response loss | (33) | implemented for available targets | `train_dual_layer` |
| 14 | short- and long-range energy separation | (15), (31) | implemented with named components | `MixedGranularityE3GNN` |
| 15 | electronegativity equilibrium and charge conservation | (16)-(19) | constrained differentiable QEq | `DifferentiableQEq` |
| 16a-b | pair readout and spin Hamiltonian | (25)-(27) | implemented with optional DMI | `TimeReversalSpinHamiltonian` |
| 17-18 | charge/spin conditioning and FiLM modulation | (28)-(30) | implemented as bounded feedback | FiLM condition builder, `FastEquivariantCoreO3` |
| 19 | comprehensive weighted loss | (33)-(34) | implemented for the active masked target set | `TrainConfig`, `train_dual_layer` |

## Formula 1: electronic-structure foundation

The electronic-structure background uses the many-electron equation,
Kohn-Sham equations, density, exchange-correlation derivative, and LDA/GGA
forms. The central relations are

```math
\widehat H_{\mathrm{tot}}
\Psi(\mathbf r_1,\ldots,\mathbf r_N;
\mathbf R_1,\ldots,\mathbf R_M)
=E\Psi(\mathbf r_1,\ldots,\mathbf r_N;
\mathbf R_1,\ldots,\mathbf R_M),
```

```math
\left[-\frac{\hbar^2}{2m_e}\nabla^2+V_{\mathrm{eff}}(\mathbf r)\right]
\phi_i(\mathbf r)=\epsilon_i\phi_i(\mathbf r),
\qquad
n(\mathbf r)=\sum_i|\phi_i(\mathbf r)|^2,
```

```math
V_{\mathrm{eff}}=V_{\mathrm n}+V_{\mathrm H}+V_{\mathrm{xc}},
\qquad
V_{\mathrm{xc}}(\mathbf r)=
\frac{\delta E_{\mathrm{xc}}[n]}{\delta n(\mathbf r)},
```

```math
E_{\mathrm{xc}}^{\mathrm{LDA}}[n]
=\int n(\mathbf r)\,
\epsilon_{\mathrm{xc}}^{\mathrm{HEG}}(n(\mathbf r))\,d\mathbf r,
\qquad
E_{\mathrm{xc}}^{\mathrm{GGA}}[n]
=\int f(n(\mathbf r),\nabla n(\mathbf r))\,d\mathbf r.
```

These are motivation and source-method context. The model predicts an effective
atomistic energy; it does not solve for orbitals or electron density.

## Formula 2: fundamental gap

```math
E_g^{\mathrm{KS}}
=\epsilon_{\mathrm{CBM}}-\epsilon_{\mathrm{VBM}},
\qquad
E_g^{\mathrm{QP}}=I-A=E_g^{\mathrm{KS}}+\Delta_{\mathrm{xc}}.
```

This background equation is not a model output in the current implementation.

## Formula 3: DFT+U

```math
E_{\mathrm{DFT}+U}
=E_{\mathrm{DFT}}
+\frac{U_{\mathrm{eff}}}{2}\sum_\sigma
\mathrm{Tr}\!\left[
\mathbf n_\sigma(\mathbf I-\mathbf n_\sigma)
\right],
\qquad U_{\mathrm{eff}}=U-J.
```

The local VASP workflow records PBE+U settings for Ni-bearing calculations,
but the neural network is not itself a DFT+U implementation.

## Formula 4: Hessian and dynamical matrix

```math
H_{i\alpha,j\beta}
=\frac{\partial^2E_{\mathrm{MLIP}}}
{\partial R_{i\alpha}\partial R_{j\beta}},
```

```math
D_{\alpha\beta}^{ab}(\mathbf q)
=\frac{1}{\sqrt{m_am_b}}
\sum_{\mathbf T}
\frac{\partial^2E}
{\partial u_{0a\alpha}\partial u_{\mathbf T b\beta}}
e^{i\mathbf q\cdot\mathbf T}.
```

The energy is differentiable to this order. Current validation checks first
derivatives against finite differences; no phonon-spectrum accuracy claim is
made.

## Formula 5: equivariant message passing

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

The code realizes the selected products in fixed real Cartesian bases with
explicit parity channels rather than delegating the entire expression to a
general Clebsch-Gordan library.

## Formula 6: Born-Oppenheimer field perturbation

```math
\Psi(\mathbf r,\mathbf R)
\approx\psi_{\mathrm e}(\mathbf r;\mathbf R)\chi_{\mathrm n}(\mathbf R),
\qquad
\widehat V_{\mathrm{ext}}
=-\widehat{\boldsymbol\mu}\cdot\boldsymbol{\mathcal E},
```

```math
\widehat H(\boldsymbol{\mathcal E})
=\widehat H_0+\widehat V_{\mathrm{ext}}
=\widehat H_0-
\widehat{\boldsymbol\mu}\cdot\boldsymbol{\mathcal E}.
```

The implementation uses the resulting effective energy expansion; it does not
represent the electronic wavefunction factor explicitly.

The field-coupling convention defines
$`\widehat V_{\mathrm{ext}}=-\widehat{\boldsymbol\mu}\cdot\boldsymbol{\mathcal E}`$
but later prints
$`\widehat H=\widehat H_0-\widehat V_{\mathrm{ext}}=\widehat H_0-\widehat{\boldsymbol\mu}\cdot\boldsymbol{\mathcal E}`$,
whose two
equalities have inconsistent signs. The normalized equation above and the code
use the physically consistent relation
$`\widehat H=\widehat H_0+\widehat V_{\mathrm{ext}}`$.

## Formula 7: second-order expansion

```math
E(\mathbf R,\boldsymbol{\mathcal E})
=E^{(0)}(\mathbf R)
+E^{(1)}(\mathbf R,\boldsymbol{\mathcal E})
+E^{(2)}(\mathbf R,\boldsymbol{\mathcal E})
+\mathcal O(\|\boldsymbol{\mathcal E}\|^3).
```

The current response energy stops at second order.

## Formula 8: perturbation terms

```math
E^{(0)}(\mathbf R)=E_{\mathrm{PES}}(\mathbf R),
\qquad
E^{(1)}=-\boldsymbol\mu\cdot\boldsymbol{\mathcal E},
```

```math
E^{(2)}
=-\frac{1}{2}
\boldsymbol{\mathcal E}^{\mathsf T}
\boldsymbol\alpha
\boldsymbol{\mathcal E}.
```

The usual sum-over-states expression motivates
$`\boldsymbol\alpha`$. The network directly parameterizes the response tensor
instead of learning excited-state wavefunctions.

## Formula 9: effective field Hamiltonian and response heads

```math
E_{\mathrm{eff}}(\mathbf R,\boldsymbol{\mathcal E})
=E_{\mathrm{PES}}(\mathbf R)
-\sum_\beta\mu_\beta(\mathbf R)\mathcal E_\beta
-\frac{1}{2}\sum_{\beta\gamma}
\mathcal E_\beta\alpha_{\beta\gamma}(\mathbf R)
\mathcal E_\gamma,
```

```math
\boldsymbol\mu
=\sum_i\boldsymbol\mu_i^{\mathrm{atomic}}
+\sum_iq_i(\mathbf R_i-\mathbf R_c)
+\sum_i\mathbf p_i^{\mathrm{ind}},
\qquad
\boldsymbol\alpha=\sum_i\boldsymbol\alpha_i.
```

The implementation multiplies the polarizability term by the documented unit
conversion when $`\alpha`$ is stored as a volume in angstrom cubed.

## Formula 10: force under electric field

```math
\mathbf F_i
=-\frac{\partial E_{\mathrm{eff}}}{\partial\mathbf R_i}
=-\frac{\partial E_{\mathrm{PES}}}{\partial\mathbf R_i}
+\left(\frac{\partial\boldsymbol\mu}{\partial\mathbf R_i}\right)^{\mathsf T}
\boldsymbol{\mathcal E}
+\frac{1}{2}\nabla_{\mathbf R_i}
\left(
\boldsymbol{\mathcal E}^{\mathsf T}
\boldsymbol\alpha
\boldsymbol{\mathcal E}
\right).
```

The code differentiates the assembled scalar energy rather than manually
adding these force components after the fact.

## Formula 11: Born effective charge form

Polarization and displacement are related through

```math
Z^*_{i,\alpha\beta}
=\Omega\frac{\partial P_\alpha}{\partial u_{i\beta}}.
```

For the canonical molecular and finite-cell convention used by the code,

```math
Z^*_{i,\alpha\beta}
=\frac{\partial\mu_\alpha}{\partial R_{i\beta}},
```

and the first-order field force follows by contraction with
$`\boldsymbol{\mathcal E}`$. The exact stored BEC convention remains part of each
source's metadata.

## Formula 12: long-range field and equilibrium response

The project uses a reciprocal-space long-range term and an equilibrium fixed
point. The implementation gives both a concrete numerical definition. Its
periodic reciprocal contribution is

```math
E_{\mathrm{rec}}
=\frac{1}{2\Omega}\sum_{\mathbf k\ne0}
\frac{4\pi k_e}{\|\mathbf k\|^2}
e^{-\|\mathbf k\|^2/(4\alpha_E^2)}
|S(\mathbf k)|^2,
```

and the Thole-damped polarization equilibrium is solved as

```math
\left(\mathbf I-
\mathbf A^{1/2}\mathbf T\mathbf A^{1/2}\right)\mathbf x
=\mathbf A^{1/2}\mathbf E_{\mathrm{drv}},
\qquad
\mathbf p=\mathbf A^{1/2}\mathbf x.
```

This is the implemented Ewald/equilibrium response system; it is not a separate
learned Poisson latent solver.

## Formula 13: first response loss

A general unmasked objective has the schematic form

```math
\mathcal L
=\lambda_E\|\widehat E-E\|^2
+\lambda_F\|\widehat{\mathbf F}-\mathbf F\|^2
+\lambda_{\mathrm{BEC}}\|\widehat{\mathbf Z}^*-\mathbf Z^*\|^2
+\lambda_\mu\|\widehat{\boldsymbol\mu}-\boldsymbol\mu\|^2
+\lambda_\alpha\|\widehat{\boldsymbol\alpha}-\boldsymbol\alpha\|^2.
```

The implemented version adds explicit masks, component normalization, and the
additional physical targets listed under Formula 19.

## Formula 14: short-/long-range separation

```math
E_{\mathrm{tot}}
=E_{\mathrm{short}}+E_{\mathrm{long}}
=E_{\mathrm{GNN}}^{\mathrm{short}}
+E_{\mathrm{physics}}^{\mathrm{long}}.
```

The code refines this schematic into named QEq, PME, D4, spin, and external
field contributions, all evaluated before force differentiation.

## Formula 15: charge equilibrium

The equal-electronegativity condition is

```math
\frac{\partial U_{\mathrm{electron}}}{\partial q_i}
=\chi_i+\sum_jJ_{ij}q_j=\overline\chi,
\qquad
\sum_iq_i=Q_{\mathrm{tot}}.
```

The implemented variational form is

```math
E_{\mathrm{QEq}}(\mathbf q)
=\boldsymbol\chi^{\mathsf T}\mathbf q
+\frac{1}{2}\mathbf q^{\mathsf T}\mathbf H\mathbf q
+\boldsymbol\phi_{\mathrm{ext}}^{\mathsf T}\mathbf q,
\qquad
\mathbf 1^{\mathsf T}\mathbf q=Q.
```

An analytic Helmert neutral basis eliminates the equality constraint before a
positive-definite solve.

## Formula 16a-b: spin-pair readout and Hamiltonian

The spin-pair readout is

```math
J_{ij}=\mathrm{Linear}
\left(\mathrm{TENN\_feature}(i,j)\right).
```

The second is the spin Hamiltonian. The implementation uses unique pairs,
explicit tensor contraction, and an optional DMI extension:

```math
E_{\mathrm{spin}}
=-\sum_{i\lt j}J_{ij}\mathbf S_i\cdot\mathbf S_j
+\sum_i\mathbf S_i^{\mathsf T}\mathbf D_i\mathbf S_i
+\sum_{i\lt j}\mathbf D_{ij}^{\mathrm{DMI}}\cdot
(\mathbf S_i\times\mathbf S_j).
```

Every term is even under simultaneous $`\mathbf S_i\mapsto-\mathbf S_i`$.

## Formula 17: conditioned message update

```math
\mathbf h_i^{(l+1)}
=\mathrm{Update}^{(l)}\!\left(
\mathbf h_i^{(l)},
\bigoplus_{j\in\mathcal N(i)}
\mathrm{Message}^{(l)}\!\left(
\mathbf h_i^{(l)},\mathbf h_j^{(l)},\mathbf r_{ij},
q_i,q_j,\mathbf S_i,\mathbf S_j
\right)
\right).
```

The code does not concatenate raw spin vectors into an invariant scalar gate.
It constructs charge, potential, and time-reversal-even spin invariants, then
uses those values as the FiLM condition.

## Formula 18: FiLM modulation

The FiLM affine form is

```math
\mathbf m_i^{\mathrm{mod}}
=\boldsymbol\gamma(q_i,\mathbf S_i)
\odot\mathbf m_i^{\mathrm{orig}}
+\boldsymbol\beta(q_i,\mathbf S_i).
```

The implemented scalar update bounds the scale perturbation,

```math
\mathbf s_i\leftarrow
\left[1+0.25\tanh\boldsymbol\gamma_i^{(s)}\right]
\odot\mathbf s_i+\boldsymbol\beta_i^{(s)},
```

while polar, axial, $`L=2`$, and optional $`L=3`$ tensors receive bounded
multiplicative modulation without an equivariance-breaking tensor bias.

## Formula 19: comprehensive objective

A broad multi-target objective can be written as

```math
\mathcal L_{\mathrm{schematic}}
=w_E\mathcal L_E+w_F\mathcal L_F+w_S\mathcal L_{\mathrm{stress}}
+w_H\mathcal L_{\mathrm{Hessian}}+w_M\mathcal L_{\mathrm{magmom}}
+w_W\mathcal L_{\mathrm{wave}}+w_{\mathrm{Ha}}\mathcal L_{\mathrm{Hamiltonian}}.
```

Stress, wavefunction, and a direct Hessian loss are not implemented targets.
The actual objective is

```math
\mathcal L_{\mathrm{implemented}}
=\sum_{t\in\mathcal T_{\mathrm{available}}}w_t
\frac{\sum_km_{t,k}
\|\widehat{\mathbf y}_{t,k}-\mathbf y_{t,k}\|_2^2}
{\sum_km_{t,k}d_t},
```

where the implemented target set may contain

```math
\mathcal T_{\mathrm{available}}\subseteq
\{E,\mathbf F,\boldsymbol\mu,\boldsymbol\alpha,
q,\boldsymbol\mu_i,\boldsymbol\alpha_i,C_6,Z^*,
\mathbf m,\mathbf H^{\mathrm{eff}},J,\mathbf D,\mathbf D^{\mathrm{DMI}}\}.
```

Availability is determined by both the selected architecture and the dataset
mask. This is the formula used in the manuscript and code.
