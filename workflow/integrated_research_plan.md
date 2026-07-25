# E3-miu-GNN Stress-Coupled Dataset Plan

- Plan status: current

## Research Decision And Scope

This work decides whether the Neo Tiny, Small, Standard, and Large tiers can
train conservative stress together with identifiable L2 electromechanical and
L3 magnetoelastic response, without relying on OMat24 as the only stress
source. The active claims are C01-C03. The scope includes source conversion,
tensor conventions, model derivatives, paired-spin VASP inputs, canonical
HDF5 rebuilding, validation, API exposure, and training presets. It does not
claim a trained stress-accurate checkpoint or completed new VASP calculations.

## Control Snapshot

- Objective: Build non-OMat24 stress coverage for Neo Tiny-Large and connect strain response explicitly to L2 electric and L3 spin physics.
- Active claim: C03
- Current gate: Obtain licensed POTCAR provenance and resource approval for a bounded constrained-spin VASP convergence pilot.
- Next action: Run only the bounded constraint-lambda/cutoff/k-point pilot after those prerequisites are documented; do not launch the 1,080-job campaign.
- Stop rule: Keep w_magnetoelastic=0 and make no L3 differential-stress accuracy claim until same-geometry, same-method spin pairs pass DFT convergence and grouped holdout gates.

## Scientific Rationale And Scale Decision

The minimum sufficient model is an atomistic conservative Hamiltonian plus
periodic DFT reference labels. For configuration, strain, electric field, and
spin state, one scalar generalized energy is used:

```text
G(R, strain, E, S) = E_L1 + E_QEq/PME + E_polarization/field
                     + E_dispersion + E_spin + coupled FiLM feedback.
```

For a fully periodic cell, total stress is `sym(dG/dstrain)/V`. L2 is linked to
strain through the VASP clamped-ion mixed derivative
`e_i,jk=(1/V)d(mu_i)/dstrain_jk`. L3 differential magnetoelastic response is
`d[G(S)-G(S_ref)]/(V dstrain)` and therefore requires target/reference spin
states at identical geometry and DFT settings. MPtrj magnetic stress provides
total spin-conditioned stress but cannot identify that differential by itself.

No MD or continuum scale is required for this data/model gate. Variable-cell
MD and continuum constitutive fitting are deferred until a stress-trained
checkpoint passes equation-of-state, elastic, piezoelectric, and magnetic
strain challenge sets.

## Evidence, Models, And Assumptions

- MPtrj v2022.9 contains 1,580,395 VASP trajectory frames with raw stress in
  compressive-positive kBar. The only accepted conversion is multiplication by
  `-0.0006241509074460763` to tensile-positive eV/Angstrom^3.
- The local 120-archive JARVIS DFPT selection contains VASP stress, per-ion BEC,
  and electronic clamped-ion piezoelectric output. The existing BEC sanity gate
  accepts 112 records without altering their tensors.
- SCFNN, QM7-X, molecular response sources, SO3LR, and DeepSPIN do not carry
  valid three-dimensional Cauchy stress and receive none.
- Assumption A1: MPtrj and JARVIS source methods remain explicit rather than
  treated as one fidelity. Risk: method heterogeneity can dominate stress loss.
  Control: provenance fields, source-stratified metrics, Huber loss, and
  method-specific challenge reporting.
- Assumption A2: the VASP electronic piezoelectric tensor is compared with the
  model's energy-conjugate `(1/V0)d(mu)/dstrain` convention. Ionic-relaxed
  contributions are excluded.
- Assumption A3: constrained-spin VASP stress is usable only when the final
  constraint penalty is below 1e-3 eV/atom and the calculation is otherwise
  converged. A lambda/k-point/cutoff pilot must test this bound.
- No local paired VASP OUTCAR is currently complete. The generated job tree is
  a launch-ready scaffold without POTCAR files, not evidence.

## Stage Roadmap

### S01: Source and convention audit

Status: validated. Claim IDs: C01, C02. Inspect raw MPtrj, JARVIS, SCFNN,
QM7-X, SO3LR, and DeepSPIN for periodicity, stress availability, units, sign,
method, and license. Reject molecular virial synthesis and model-generated
labels. Output: source policy and conversion contract.

### S02: Conservative coupled model

Status: validated for software. Claim IDs: C01, C03. Derive total and component
stress from scalar energies, derive VASP-compatible piezoelectric response from
the model dipole, and derive paired magnetoelastic stress from the complete
target-minus-reference coupled energy. Controls: CPU float64 finite difference,
rotation/symmetry, time reversal, component sum, and backward gradients.

### S03: Raw-label enrichment

Status: validated. Claim ID: C02. Stream MPtrj once to update the static,
magnetic, and trajectory-rich canonical shards transactionally. Rebuild JARVIS
DFPT from the 120 ZIP archives. Outputs retain source, method, response family,
perturbation, stress convention, masks, and checksums.

### S04: Tiny-Large tier rebuild

Status: validated. Claim ID: C02. Rebuild Standard from the mixed policy, derive
Tiny and Small with the existing coverage-stratified selector, and rebuild
Large from the trajectory policy. Acceptance requires valid masks, finite and
symmetric tensors, no response-family split leakage, all evaluation elements
seen in train, and reported L1/L2/L3 mechanism coverage.

### S05: Paired L3 VASP reference campaign

Status: ready, not submitted. Claim ID: C03. The scaffold contains bcc Fe, AFM
NiO, and an Fe/NiO interface; systematic +/- normal, shear, and hydrostatic
strains; and eight spin states per strained geometry. Full 3D stress is
collected only for bulk cells. Controls include train/val/test strain-series
replicas, non-SOC/SOC reference pairing, time reversal, cutoff/k-point checks,
constraint penalty, and POTCAR provenance. The interface remains an
energy/force/spin branch because its declared periodicity is only 2D.

### S06: Training and challenge validation

Status: planned. Claim IDs: C01-C03. Compare otherwise identical checkpoints
with and without stress/piezoelectric losses. Evaluate grouped source-specific
energy, force, stress, EOS, elastic, BEC, piezoelectric, and magnetic-strain
holdouts. Magnetoelastic training remains disabled until S05 passes.

## Stage Gates And Deliverables

- G1: MPtrj sign/unit conversion and JARVIS Voigt expansion pass exact tests.
- G2: autograd stress/piezoelectric/magnetoelastic tensors agree with affine
  finite differences; component stresses sum to total; global spin reversal is
  invariant.
- G3: every strain-response label is finite, symmetric in the required axes,
  fully 3D periodic, method/convention tagged, and split by response family.
- G4: Standard and Large report nonzero L1 stress, L2 stress+BEC+piezoelectric,
  and L3 stress+spin coverage; paired L3 coverage may remain zero and must be
  stated as a negative result.
- G5: VASP paired labels require completed outputs, matching geometry/method,
  acceptable constraint penalty, and no partial-periodic 3D stress.
- Deliverables: model/API code, source converters, canonical datasets, reports,
  tests, presets, job scaffold, and synchronized workflow records.

## Resource-Aware Execution Order

Use existing raw labels first: JARVIS rebuild, one MPtrj stream, Standard, then
portable tiers and Large. Run CPU float64 derivative tests before MPS training.
The VASP branch starts with a small constraint-lambda/cutoff/k-point pilot, not
all 1,080 scaffolded jobs. Production submission requires licensed local
POTCAR resolution and an explicit compute allocation. Plus/Max rematerialization
is deferred because the active request is Tiny-Large and those files are much
larger than the validated source-layer change.

## Risks, Negative Results, And Escalation

- Heterogeneous PBE(+U) and optB88-vdW stress can create method bias. Report by
  source and escalate to multi-fidelity conditioning if errors separate by
  method.
- Sparse JARVIS piezoelectric coverage can overfit. Preserve all 112 records in
  portable tiers, group splits, and use multi-seed learning curves.
- MPtrj spin+stress correlation does not prove L3 attribution. Do not relabel
  it as paired magnetoelastic data.
- Constrained-spin penalty can contaminate stress. Reject failed penalties and
  run a lambda sensitivity pilot; do not correct stress with an invented term.
- Higher derivatives increase memory. Use sparse response batches and MPS edge
  budgets rather than weakening the conservative derivative.
- If VASP references cannot meet the penalty/convergence gate, C03 remains
  inconclusive and `w_magnetoelastic` stays zero.

## Research Spine Synchronization

The objective, active claim, gate, next action, and stop rule match
`workflow/research_spine.md` and `workflow/experiment_manifest.json`. C01-C03,
S01-S06, branch B01, lineage records, and queue action A01 are mirrored in the
claim matrix, branch register, data lineage, decision log, and next-action
queue. Status distinguishes validated conversion, running tier rebuild, ready
DFT scaffold, and unrun training.

## Definition Of Done

This data-design task is complete: Tiny, Small, Standard, and Large are rebuilt
and strictly validated; code/tests/presets/docs agree on tensor conventions;
source and mechanism coverage is reported; and the paired VASP campaign is
reproducibly scaffolded with its launch blockers recorded. C03 is not
scientifically complete until real paired DFT outputs and grouped model
holdouts pass, and no final model-accuracy claim is part of this task.
