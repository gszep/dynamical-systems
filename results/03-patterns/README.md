# Results — experiment 03 (1D periodic Swift–Hohenberg patterns)

This directory holds the reproducible artifacts written by
`experiments/03-patterns/scripts/main.jl`. It is owned by experiment 03;
nothing else writes here.

**Status: not yet populated.** The experiment code was written without
execution (concurrent package setup forbade running Julia). The first smoke
run of the parent/orchestrator writes the files below; until then this
directory contains only this README.

## Files and schemas (written by the smoke run)

* `discovery.csv` — one row per seed attempt (both discovery parameters):
  `r, seed_id, family, kind, tag, converged, accepted, is_new, reason,
  newton_iters, n_deflated, residual, rms, mean_u, kbar, pr, f0, f1, f2, f3,
  f4, nz50, lambda_max_sub, n_pos_sub, lambda_max_full, n_pos_full`
* `branch_primary.csv` — one row per accepted continuation point (both
  directions): `point_id, direction, s, r, rms, linf, kbar, lambda_max_sub,
  n_pos_sub, lambda_max_full, n_pos_full, newton_iters, ds`
* `mesh_convergence.csv` — N = 32/64/128 stationary solutions at fixed
  r = -0.1 (large- and small-amplitude seeds):
  `N, r_fix, seed_tag, converged, newton_iters, residual, rms, kbar,
  lambda_max_sub, n_pos_sub, err_u_vs_finest, err_c16_vs_finest`
  (`err_*` are NA on the finest grid, the reference row).
* `fold_convergence.csv` — fold estimate per mesh from independent
  continuations: `N, seed_tag, r_start, r_fold, s_fold, n_points, termination`
* `failures.csv` — every failed nonlinear solve, corrector attempt and guard
  rejection: `stage, tag, direction, r, N, step, attempt, ds, residual, reason`
* `patterns_summary.toml` — reproducibility metadata and headline numbers:
  Julia version, machine, BLAS=1 note, timings (queue/setup/discovery/
  continuation/mesh/io/total), seeds, tolerances, thresholds, per-family
  discovery tallies, branch fold values vs the analytic Galerkin estimate
  -9b²/40, mesh errors, morphology-vs-topology evidence, and an explicit
  scope statement (no exhaustive atlas; single 1D configuration; no snaking
  claims; spectra restricted as documented — no full-space
  asymptotic-stability claims).

## Interpretation rules

* `*_sub` spectra are the physical rates of the reflection-symmetric (even)
  restriction, mass-normalized against G = S'S: eigenvalues of
  G^-1/2 (S' J S) G^-1/2. The translation phase mode is absent from them by
  construction (the restriction is a phase condition, not a translation
  quotient; it also excludes states with no even translate and does not
  identify general translations). `*_full` spectra
  include the ≈ 0 phase eigenvalue. Unstable even-subspace eigenvalues imply
  full-space instability; the converse is never claimed.
* Discovery rows describe what the fixed seed table actually found;
  absence of a family's solutions is recorded absence, not nonexistence.
* Numbers here come from the recorded run that wrote them; the summary TOML
  carries the runtime version, seeds and timings needed to reproduce them.
