# Results — experiment 03 (1D periodic Swift–Hohenberg patterns)

This directory holds the reproducible artifacts written by
`experiments/03-patterns/scripts/main.jl`. It is owned by experiment 03;
nothing else writes here.

**Status: populated by executed smoke runs (2026-10-20 and the gated rerun
2026-10-21).**
The experiment code was initially written without execution (concurrent
package setup); after the setup hold was released, the test suite and smoke
run were executed with the pinned runtime (Julia 1.10.11,
x86_64-apple-darwin24.0.0, one process, one thread, BLAS pinned to 1).

## Executed runs — provenance

* **2026-10-20 — first executed run** (after the setup hold): tests 208/208
  (exit 0), smoke exit 0; logs `runs/20261020-exp03-*.txt`.
* **2026-10-21 — final-review rerun** (acceptance gates added, N=256
  exception restricted to that resolution alone, timing relabeled as
  partially instrumented): tests **224/224** (exit 0), smoke **exit 0** with
  acceptance gates PASS; logs `runs/20261021-exp03-*.txt`. Artifacts below
  are from this rerun.

Commands (from the repository root, single process each; RUNTIME is the
pinned interpreter, e.g.
`/Users/gszep/Documents/repos/dynamical-systems-research-20261019/.tools/julia-1.10.11/bin/julia`):

```sh
RUNTIME=<pinned julia 1.10.11>
JULIA_NUM_THREADS=1 $RUNTIME \
    --project=experiments/03-patterns experiments/03-patterns/scripts/tests.jl
# -> 224 passed, 0 failed, 0 errored (exit 0); log: runs/20261021-exp03-tests.txt

JULIA_NUM_THREADS=1 $RUNTIME \
    --project=experiments/03-patterns experiments/03-patterns/scripts/main.jl
# -> exit 0, acceptance gates PASS; log: runs/20261021-exp03-smoke.txt
```

Headline measured numbers (details in the artifacts):

* Fold of the primary pattern branch: **r_fold = −0.2250098** (N=64 and
  N=128, agreeing to <1e-10; N=32: −0.225018), vs the independent single-mode
  Galerkin estimate −9b²/40 = **−0.225** (agreement to ~1e-5).
* Two coexisting stationary states at r = −0.1: rms 0.72436 (stable,
  n_pos = 0) and 0.27633 (unstable, n_pos = 1); at r = −0.2: 0.63248 and
  0.44712 — each matching the Galerkin amplitudes A = 1.023/0.39 at −0.1
  and 0.894/0.632 at −0.2 to ~1%.
* Mesh convergence (large seed, errors vs N=128): 1.84e-3 (N=32) →
  1.80e-8 (N=64); small seed: 3.73e-5 → 8.37e-12.
* Phase-mode defect ‖J U′‖/‖U′‖ (see below): 0.543 (N=32) → 6.24e-4 (N=64)
  → 1.80e-9 (N=128) → 1.78e-7 (N=256, stall-limited, cause unconfirmed).
* Discovery: 2 new patterns at each r ∈ {−0.1, −0.2} (both from cosk seeds);
  41 failure rows recorded (38 `no_descent`, 3 `known_root_excluded`);
  localized seeds discovered nothing at these parameters (recorded absence,
  not nonexistence).
* Timing (warm-cache rerun): setup 0.481 s + internal 0.982 s = total
  1.464 s reported by the script — a partially instrumented measure (see the
  scope note in the summary); the supervisor-captured process wall time
  (independently measured at ~18–30 s depending on cache state, dominated by
  module load/JIT outside the stage windows) is authoritative.

## Files and schemas

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
  Julia version, machine, BLAS=1 note, timings (queue/setup/stages/internal/
  total — partially instrumented, scope note included), seeds, tolerances,
  thresholds, per-family discovery tallies, branch fold values vs the analytic
  Galerkin estimate −9b²/40, mesh errors, morphology-vs-topology evidence,
  the `[phase_mode]` defect table (see below), the explicit `[acceptance]`
  gates (positive discoveries per sweep, usable continuation with fold,
  required 32/64/128 mesh solves; a failing gate is recorded in
  `failures.csv` and the smoke script exits nonzero), and an explicit scope
  statement (no exhaustive atlas; single 1D configuration; no snaking claims;
  spectra restricted as documented — no full-space asymptotic-stability
  claims).
* `runs/` — captured stdout of the executed test and smoke runs (committed
  for in-repo provenance; the supervisor's central logs remain authoritative).

## Interpretation rules

* `*_sub` rates are the physical rates of the reflection-symmetric (even)
  restriction, mass-normalized against G = S'S: eigenvalues of
  G^-1/2 (S' J S) G^-1/2. The translation phase mode is absent from them by
  construction (the restriction is a phase condition, not a translation
  quotient; it also excludes states with no even translate and does not
  identify general translations). `*_full` spectra
  include the ≈ 0 phase eigenvalue. Unstable even-subspace eigenvalues imply
  full-space instability; the converse is never claimed.
* `[phase_mode] JUprime_over_Uprime` is the relative residual of the
  translation phase direction in the full-space collocation linearization.
  U' is an exact null vector of the *continuous* problem; the undealiased
  collocation nonlinearity preserves only grid shifts, so the defect is a
  resolution-dependent aliasing quantity (dominated by the first nonlinear
  harmonic past Nyquist: k=5 at N=32, k=9 at N=64), NOT an implementation
  error — the tests verify the Jacobian and derivative assemblies against
  independent operator applications. The N=256 entry uses a
  residual-qualified stalled iterate: the solve stalls at ‖R‖ = 1.297e-8
  (`no_descent`) under the default 1e-9 tolerance; **the cause of the stall
  is unconfirmed** (a dimensionless eps·cond bound ≈ 9.3e-10 does not
  establish an absolute residual floor and no independent error analysis
  was performed), so that measurement point alone is accepted at
  ‖R‖ ≤ 1e-6 and the entry is treated as stall-limited rather than
  aliasing-limited.
* Discovery rows describe what the fixed seed table actually found;
  absence of a family's solutions is recorded absence, not nonexistence.
* Numbers here come from the recorded run that wrote them; the summary TOML
  carries the runtime version, seeds and timings needed to reproduce them.
