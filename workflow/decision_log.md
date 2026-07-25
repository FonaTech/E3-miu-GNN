# Decision Log

## 2026-07-25 - Stress source and coupling design

- Accepted MPtrj raw VASP stress as the main non-OMat24 L1 and spin-conditioned
  L3 total-stress source after one explicit kBar-to-eV/Angstrom^3 conversion.
- Accepted JARVIS DFPT stress+BEC+electronic piezoelectric tensors for matched
  L2 electromechanical supervision.
- Rejected synthetic stress for molecular sources and DeepSPIN.
- Rejected treating ordinary MPtrj magnetic frames as paired magnetoelastic
  labels; `w_magnetoelastic` remains zero.
- Promoted the complete coupled target-minus-reference energy derivative as the
  magnetoelastic model contract.
- Deferred VASP execution until POTCAR provenance, resource approval, and a
  constraint/convergence pilot are available.

## 2026-07-25 - Tiny-Large stress gate closed

- Validated Tiny, Small, Standard, and Large with no tensor, provenance,
  response-family leakage, or element-coverage warning.
- Confirmed exact `Tiny subset Small subset Standard` sample-ID nesting.
- Recorded stress coverage of 2,024 / 8,243 / 22,873 / 505,848 and matched
  spin-conditioned coverage of 974 / 4,220 / 12,000 / 72,929.
- Kept the paired magnetoelastic count at zero and promoted C02 to supported.
- Moved the active gate to the bounded VASP pilot; the 1,080-job scaffold
  remains unsubmitted.
