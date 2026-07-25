# Research Spine

- Decision: Determine whether Neo Tiny-Large can train conservative stress with identifiable L2 electric and L3 spin coupling without OMat24-only stress dependence.
- Objective: Build non-OMat24 stress coverage for Neo Tiny-Large and connect strain response explicitly to L2 electric and L3 spin physics.
- Active claim: C03
- Decisive observable: Source-stratified stress coverage and held-out stress, piezoelectric, elastic, and magnetic-strain error.
- Evidence tier: Converted public DFT labels and software finite-difference validation; no new paired VASP result or trained accuracy claim.
- Current conclusion: MPtrj supplies L1 and spin-conditioned L3 total stress; JARVIS supplies matched L2 stress+BEC+piezoelectric; paired L3 differential stress is still absent.
- Current gate: Obtain licensed POTCAR provenance and resource approval for a bounded constrained-spin VASP convergence pilot.
- Next action: Run only the bounded constraint-lambda/cutoff/k-point pilot after those prerequisites are documented; do not launch the 1,080-job campaign.
- Stop rule: Keep w_magnetoelastic=0 and make no L3 differential-stress accuracy claim until same-geometry, same-method spin pairs pass DFT convergence and grouped holdout gates.
- Non-goals: Plus/Max rematerialization, production training, MLMD, continuum fitting, and Ultra/Ultimate expansion.
- Last reviewed: 2026-07-25T06:44:48+08:00 by Codex.
