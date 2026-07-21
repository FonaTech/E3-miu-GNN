# Scientific Background and Research Scope

This document expands Sections 1 and 2 of the [paper](PAPER.md) and explains
the scientific motivation for the E(3)-GNN system implemented in
[`E3_miu_GNN.py`](../E3_miu_GNN.py).

## From electronic structure to an atomistic energy model

For fixed nuclei, the Born-Oppenheimer approximation reduces the electronic
problem to an energy surface parameterized by nuclear coordinates. In
Kohn-Sham density-functional theory (DFT), auxiliary one-electron orbitals obey

```math
\left[-\frac{\hbar^2}{2m_e}\nabla^2+V_{\mathrm{eff}}(\mathbf r)\right]
\phi_i(\mathbf r)=\epsilon_i\phi_i(\mathbf r),
\qquad
n(\mathbf r)=\sum_i\left|\phi_i(\mathbf r)\right|^2,
```

with

```math
V_{\mathrm{eff}}=V_{\mathrm n}+V_{\mathrm H}+V_{\mathrm{xc}},
\qquad
V_{\mathrm{xc}}(\mathbf r)=
\frac{\delta E_{\mathrm{xc}}[n]}{\delta n(\mathbf r)}.
```

The exchange-correlation functional is approximate. One formal expression of
the missing contribution is the derivative discontinuity in the fundamental
gap,

```math
E_g^{\mathrm{QP}}=I-A=E_g^{\mathrm{KS}}+\Delta_{\mathrm{xc}}.
```

Localized correlated orbitals are often treated with a DFT+$U$ correction,

```math
E_{\mathrm{DFT}+U}=E_{\mathrm{DFT}}
+\frac{U_{\mathrm{eff}}}{2}\sum_\sigma
\mathrm{Tr}\!\left[\mathbf n_\sigma
(\mathbf I-\mathbf n_\sigma)\right].
```

These equations motivate the training labels and physical structure of the
model. The software does not contain a Kohn-Sham or DFT+$U$ solver. It learns
an effective atomistic Hamiltonian from calculations performed upstream.

```mermaid
flowchart LR
    DFT[Electronic-structure calculations] --> Y[Energy, force, response and spin labels]
    Y --> G["E(3)-equivariant atomistic model"]
    G --> H[Differentiable effective Hamiltonian]
    H --> O[Energy and derivative observables]
```

## Local machine-learning interatomic potentials

A conventional local potential decomposes the energy into atomic
contributions,

```math
E_{\mathrm{local}}(\mathbf R)
=\sum_i\varepsilon_i\!\left(\mathcal N_i^{r_c}\right),
```

where $\mathcal N_i^{r_c}$ is the neighborhood inside a finite cutoff. This
form is efficient and is appropriate when distant interactions are screened or
can be represented through local correlations. It is not a complete state
description when the energy depends explicitly on a global charge constraint,
reciprocal-space electrostatics, collective induced dipoles, or magnetic order.

The limitation is not only range. A scalar invariant representation discards
the transformation type of intermediate information. Electric dipoles are
polar vectors, spins are axial vectors, polarizabilities and anisotropies are
rank-2 tensors, and magnetic effective fields are odd under time reversal.
Those contracts must be preserved before a final scalar energy is formed.

## Why E(3) and O(3) equivariance matter

For an orthogonal transformation $\mathbf Q\in O(3)$ and translation
$\mathbf t$, positions transform as

```math
\mathbf R_i' = \mathbf Q\mathbf R_i+\mathbf t.
```

A physically consistent energy, force, polar vector, axial vector, and rank-2
tensor satisfy

```math
E'=E,
\qquad
\mathbf F_i'=\mathbf Q\mathbf F_i,
\qquad
\boldsymbol\mu'=\mathbf Q\boldsymbol\mu,
```

```math
\mathbf a'=\det(\mathbf Q)\mathbf Q\mathbf a,
\qquad
\boldsymbol\alpha'=\mathbf Q\boldsymbol\alpha\mathbf Q^{\mathsf T}.
```

The determinant is essential under reflection: a polar displacement changes
sign under inversion while an axial spin does not. The implementation therefore
uses separate polar and axial channels when O(3) parity is enabled. Its
deterministic tests apply rotations, reflections, and simultaneous spin
reversal directly to complete model inputs and outputs.

## Natural physical granularities

The research hypothesis is that each mechanism should be represented at the
smallest granularity that retains its governing constraints.

| Level | State and scale | Physical role | Implemented mechanism |
| --- | --- | --- | --- |
| Layer 1 | local atoms and edges | short-range chemical environment | parity-aware equivariant message passing and atomic energy |
| Layer 2 | molecular or periodic domain | charge, electric field, polarization, dispersion | constrained QEq, Ewald/PME, Thole equilibrium, molecular D4 |
| Layer 3 | spin-bearing sites and pairs | exchange and spin-lattice energy | $J_{ij}$, $\mathbf D_i$, DMI, moment and effective-field heads |
| Coupling | atom-wise shared state | electronic and magnetic feedback | bounded FiLM modulation of Layer-1 features |

![Implemented three-layer architecture](assets/proposal/system-overview-core.png)

The architecture is not three independent predictors. Layer-2 charge and
potential and Layer-3 spin invariants condition subsequent Layer-1 messages.
All active energy terms are then assembled before force differentiation.

## Energy as the common physical interface

The implemented Hamiltonian is

```math
E_{\mathrm{tot}}=
E_{\mathrm{short}}+E_{\mathrm{QEq}}+E_{\mathrm{PME}}
+E_{\mathrm{D4}}+E_{\mathrm{spin}}+E_{\mathrm{resp}}.
```

Observable derivatives are taken from this common scalar,

```math
\mathbf F_i=-\frac{\partial E_{\mathrm{tot}}}{\partial\mathbf R_i},
\qquad
Z^*_{i,\alpha\beta}=
\frac{\partial\mu_\alpha}{\partial R_{i\beta}},
\qquad
\mathbf H_i^{\mathrm{eff}}=-
\frac{\partial E_{\mathrm{spin}}}{\partial\mathbf S_i}.
```

This energy-first design makes conservative-force and symmetry checks
meaningful across learned and analytic components. A finite prediction alone
does not establish physical calibration; solver residuals and held-out errors
must also be reported.

## Implemented research question

The completed work addresses the following bounded question:

> Can local atomic geometry, constrained electric-domain physics, and a
> time-reversal-aware spin Hamiltonian be coupled in one differentiable
> E(3)/O(3)-consistent training system without inventing missing labels?

The implementation and regression suite answer the architectural and numerical
parts of this question. The current short benchmarks do not establish a
universal production potential, converged phonon spectra, or cross-material
magnetic accuracy.

## Further reading

- [Paper](PAPER.md) for the manuscript narrative and results.
- [Architecture](ARCHITECTURE.md) for representation and code structure.
- [Physical mechanisms](PHYSICS.md) for QEq, PME, polarization, D4, and spin
  equations.
- [Formula reference](FORMULAE.md) for mathematical definitions and their code
  locations.
