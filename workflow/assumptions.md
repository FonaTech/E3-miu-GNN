# Assumptions

- A1: MPtrj raw stress is compressive-positive kBar; conversion and sign are
  fixed at ingestion. Risk: source convention error. Resolution: source docs,
  ASE/VASP convention tests, and EOS spot checks.
- A2: JARVIS OUTCAR electronic piezoelectric output matches the model's
  reference-volume mixed electric-enthalpy derivative. Risk: proper/improper or
  ionic contribution mismatch. Resolution: retain the exact VASP convention
  and validate against direct DFPT examples.
- A3: constrained-spin stress is acceptable when `E_p` is below 1e-3 eV/atom.
  Risk: residual constraint stress despite small energy. Resolution: lambda,
  cutoff, and k-point sensitivity pilot.
- A4: method heterogeneity is learnable with source-aware validation. Risk:
  systematic PBE(+U)/optB88 bias. Resolution: stratified metrics or explicit
  multi-fidelity conditioning before production claims.
