# Validation Plan

- Source: exact MPtrj sign/unit test and JARVIS Voigt tensor expansion.
- Model: affine finite differences for stress, piezoelectric, and
  target-minus-reference magnetoelastic stress; rotation, symmetry, component
  closure, time reversal, and higher-derivative backward tests.
- Dataset: finite values under masks, required shapes, full periodicity,
  positive volume, tensor symmetry, method/convention metadata, co-located
  labels, response-family split isolation, source counts, and element coverage.
- DFT pilot: electronic/ionic convergence, cutoff, k points, constraint lambda,
  penalty below 1e-3 eV/atom, same geometry/method, and stress stability.
- Model challenge: source-stratified energy/force/stress MAE, EOS, elastic
  constants, BEC, piezoelectric tensor, magnetic strain differences, and
  multi-seed uncertainty.
