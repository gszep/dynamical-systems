# Experiment 03 — 1D periodic Swift–Hohenberg stationary patterns

Nonlinear stationary states of the subcritical 1D Swift–Hohenberg equation with
Newton solvers, pseudo-arclength continuation through folds, multi-seed branch
discovery with deflation, symmetry-aware morphology descriptors, and
mesh-convergence evidence on dense small grids (N = 32/64/128).

**Status: code written, NOT executed.** This experiment was implemented while
package setup ran concurrently; Julia execution, builds and simulations were
forbidden in the writing session. The parent/orchestrator runs
`scripts/tests.jl` and `scripts/main.jl` with the pinned runtime. Until that
first run, every numeric expectation below is a prediction, not evidence.
`EXPERIMENTS.md` (a repository root file, outside this experiment's ownership)
is updated by the parent with the achieved evidence.

## Mathematical formulation

State equation on the periodic interval x ∈ [0, L), L = 8π:

```
u_t = r u - (1 + ∂xx)^2 u + b u^3 - u^5 ,      b = 1  (subcritical),  r ∈ ℝ
```

Stationary problem solved here:

```
F(u; r) = r u - (1 + ∂xx)^2 u + b u^3 - u^5 = 0
```

The linear operator r I - (1 + ∂xx)² has exact modal eigenvalues

```
λ(k) = r - (1 - k²)² ,      k = 2π n / L ,
```

so u = 0 loses stability to the k = 1 modes at r = 0. With b > 0 the cubic is
destabilizing and the quintic saturates: the pattern branch leaves the trivial
state subcritically (states for r < 0), folds back, and continues to larger
amplitude — the classic S-curve with an unstable segment. Independent
single-mode Galerkin predictions (u = A cos x, projections over one period):

```
amplitude equation:  0 = r + (3/4) b A² - (5/8) A⁴
fold:                 r* = -9 b²/40 = -0.225 ,   A* = sqrt(3/5) ≈ 0.775
r = -0.1 states:      A⁋ ≈ 0.39 (unstable),  A₊ ≈ 1.02 (stable in the pinned subspace)
```

These analytic values are used as *independent expected values* in the tests
(they are not fitted to the code's own output).

## Discretization

* Fourier pseudospectral collocation on x_j = (j-1)L/N, N even, N ≥ 8, with the
  DFT realized as **dense exact matrices** E[j,n] = exp(-2πi(j-1)(n-1)/N),
  Ein = E'/N (no FFT package; N ≤ 128 keeps everything tiny). The quadratic
  operator (1 + ∂xx)² is pre-assembled as the dense value-space matrix
  `Mmat = real(Ein · Diagonal((1-k²)²) · E)`; the tests verify its spectrum
  against λ(k) to machine precision.
* **No dealiasing** (documented choice): u³ and u⁵ are evaluated pointwise on
  the grid, so coarse grids carry O(aliasing) error — this is precisely what
  the N = 32/64/128 convergence study measures.
* Independent cross-check operator: second-order periodic finite differences,
  D2 (circulant, eigenvalues exactly -4 sin²(πm/N)/h²). The tests verify the
  exact FD circulant spectrum, the O(h²) approach of (I + D2) eigenvalues to
  (1 - k²)² with ratio ≈ 4 per refinement, and that the spectral solution has
  a small but genuinely nonzero FD collocation residual that shrinks with h.

### Phase pinning: the reflection-invariant (even) restriction

The equation commutes with translations, so every nontrivial periodic
stationary state U comes with a one-parameter orbit U(· - x₀) and the
full-space linearization J has the phase direction U' in its kernel
(‖J U'‖ = 0 — verified in the tests). Newton on the full space is therefore
singular at solutions. We search exclusively in the subspace invariant under
the grid reflection x → -x (cosine modes only, dimension N/2 + 1):

* this is an **explicit phase pinning (reflection gauge)**: translations leave
  the subspace, and seeds are projected onto it before Newton
  (`to_coeffs` symmetrizes), so the translation orbit is quotiented out;
* the operator and nonlinearity preserve the subspace exactly, so the
  restriction is the collocation system restricted to even u;
* all Newton/continuation spectra reported as `*_sub` are spectra **of the even
  subspace**: the phase mode is removed *by construction* (tested), and the
  sine-sector (phase/sideband) part of the spectrum is **not** claimed stable.
  An unstable even-subspace eigenvalue does imply full-space instability; the
  converse is never claimed. Full-space spectra are also computed and reported
  (`*_full`, including the ≈ 0 phase eigenvalue) — no full-space
  asymptotic-stability claim is made anywhere in this experiment.

## Algorithms

* **Damped Newton** (dense, N/2+1 unknowns) on the even residual
  R(c; r) = S' F(S c; r): Armijo-style backtracking with sufficient decrease;
  convergence judged on ‖R‖₂ ≤ 1e-9 (true residual, never a deflated one).
  Failures return diagnostic reasons (`linfail`, `nonfinite_direction`,
  `no_descent`, `maxiter`) and are recorded.
* **Classical root-removing deflation** (discovery only):
  R̃(c) = M(c) R(c) with M(c) = Πᵢ 1/‖c - cᵢ‖² (scale-free), which has a pole
  at each known root cᵢ — known solutions cannot be re-found; all other roots
  persist. The Newton direction solves the equivalent unscaled system
  (J + R w') δ = -R with w = -Σᵢ 2(c - cᵢ)/‖c - cᵢ‖² (distance floored at
  1e-30); convergence is always judged on the true residual.
* **Pseudo-arclength continuation** in z = (c, r): predictor z + ds·t,
  corrector Newton on the bordered system
  `[∂R/∂c  ∂R/∂r; t'] (dc, dr) = -(R, t'(z - z_prev) - ds)`,
  which is regular at simple folds — the branch is continued *through* the
  fold, retaining the unstable segment. Tangents solve the same bordered
  matrix against the last unit vector, orientation-fixed. Adaptive step:
  grow ×1.25 after easy steps, shrink ×0.6 after hard ones, halve on corrector
  failure (each failed attempt logged with the current ds).
* **Fold localization**: three-point parabola vertex on r(s) among accepted
  points — an estimate that is step-size limited (documented; not a
  bisection-refined value).
* **Guards** (zero-solution contamination): solutions with rms < 1e-6 are
  classified `trivial` and rejected as discoveries; continuation never accepts
  states with rms < 0.02 (terminating with `trivial_approach` near the
  pitchfork) and never continues *through* the trivial branch; duplicates are
  detected up to the sign symmetry u → -u (the equation is odd, so -U is a
  solution whenever U is); NaN/Inf results are rejected and recorded.

## Discovery protocol (not an atlas)

A fixed, fully recorded seed table (26 seeds per parameter value) is run at
r ∈ {-0.10, -0.20} on N = 64:

| family | seeds | purpose |
|---|---|---|
| `cosk` | A ∈ {0.35, 0.7, 1.2} × k ∈ {1, 2} | periodic-pattern basins, both branch segments |
| `localized` | A ∈ {0.8, 1.3} × w ∈ {2, 4} × center ∈ {0, L/2} | localized (sech²-bump) seeds |
| `random` | 10 deterministic band-limited fields (TinyRNG seeds recorded) | generic even basins |
| `trivial_probe` | 1 tiny-noise seed | exercises the trivial-rejection guard |
| `highfreq` | 1 cos(3x) seed | strongly damped mode; expected collapse |

Every attempt is recorded with outcome (`new`, `duplicate`, `trivial`,
`not_converged:<reason>`), Newton iterations, residual, descriptors and
spectra; failed nonlinear solves go to `failures.csv`. The comparison the
acceptance gate asks for — **localized-seed vs periodic-seed discoveries** —
is exactly what `discovery.csv` records, family by family. Whether localized
states exist at these r values is an empirical question this experiment
answers by recording actual outcomes; **no snaking or completeness claim is
made**.

## Morphology descriptors vs branch topology

Morphology descriptors (all invariant under translations — exactly, via
discrete Parseval for band-limited grid functions — and under reflection;
deliberately *not* complete invariants, which the tests check):

```
rms, mean, kbar (power-weighted wavenumber), pr (participation ratio),
f0..f4 (power fractions in k-bands [0,0.5), [0.5,1.5), [1.5,2.5), [2.5,4.5), [4.5,∞)),
nz50 (number of modes with power > 1e-10 · max)
```

`linf` is recorded for branch points but is *excluded* from the invariant set
(sampled extrema move with the grid). **Morphology vs topology is kept
distinct**: the same wavenumber class (kbar ≈ 1) occurs on both sides of the
fold, where the branch segments differ in norm and in even-subspace
stability; `mesh_convergence.csv` and the summary's
`mesh.morphology_vs_topology` block record that contrast, and the tests
assert it. Descriptors classify shapes; they do not encode branch position,
folds or stability.

## Outputs (written by `scripts/main.jl` into `results/03-patterns/`)

* `discovery.csv` — one row per seed attempt (all r values): outcome, guards,
  iterations, residual, rms, descriptors, even/full spectra, deflation state.
* `branch_primary.csv` — one row per accepted continuation point, both
  directions: s, r, rms, linf, kbar, λmax and unstable counts (even + full),
  Newton iterations, ds.
* `mesh_convergence.csv` — N = 32/64/128 solutions at fixed r = -0.1 for a
  large- and a small-amplitude seed: rms, kbar, spectra, relative errors vs
  the finest grid (values and first 16 cosine coefficients).
* `fold_convergence.csv` — fold estimates per mesh from independent
  continuations.
* `failures.csv` — every failed nonlinear solve / corrector attempt / guard
  rejection, with context (stage, tag, step, attempt, ds, residual, reason).
* `patterns_summary.toml` — reproducibility metadata: Julia version, machine,
  timings (queue/setup/discovery/continuation/mesh/io/total), all seeds,
  tolerances, thresholds, counts, fold numbers vs the Galerkin estimate,
  per-family discovery tallies, morphology-vs-topology evidence, and an
  explicit scope statement (no atlas, no 2D, no snaking claims, no full-space
  stability claims).

See `results/03-patterns/README.md` for the exact column schemas.

## Commands (pinned runtime)

```sh
JULIA=/Users/gszep/Documents/repos/dynamical-systems-research-20261019/.tools/julia-1.10.11/bin/julia
cd <repository-root>

# tests (Test stdlib; nonzero exit on any failure)
JULIA_NUM_THREADS=1 $JULIA --project=experiments/03-patterns \
    experiments/03-patterns/scripts/tests.jl

# smoke run (< 5 min single-thread budget; writes results/03-patterns/)
JULIA_NUM_THREADS=1 $JULIA --project=experiments/03-patterns \
    experiments/03-patterns/scripts/main.jl
```

Scripts `include` the module directly, so `--project` is optional (it only
declares stdlib dependencies). Both scripts pin `BLAS.set_num_threads(1)`.
Budget: single process, one Julia thread, one BLAS thread; all matrices are
≤ 128 × 128 dense; expected wall time is seconds of compute plus JIT
compilation.

## Test catalogue (scripts/tests.jl)

| testset | checks |
|---|---|
| `transforms` | DFT roundtrip exactness; reflection involution; non-vacuous (u is not even); coeffs↔values roundtrip; exact fractional shift of band-limited data |
| `spectral_operator_expected_spectrum` | assembled dense operator spectrum vs analytic r-(1-k²)² (machine precision); cosine columns are exact eigenvectors; even-subspace spectrum at u=0; critical mode sits at r (independent expected value) |
| `fd_operator_expected_spectrum` | exact FD circulant spectrum; (I+D2) eigenvalues differ from spectral values (negative case) and converge at ratio ≈ 4 |
| `newton_stationary_and_independent_expected` | residual < tol; rms matches single-mode Galerkin prediction within 8%; zero mean by symmetry; two coexisting states at r = -0.1 with n_pos = 0 vs 1; FD cross-check residual decreases with h and is genuinely nonzero |
| `phase_mode_and_removal` | ‖J U'‖/‖U'‖ ≈ 0; full spectrum has exactly one ≈0 eigenvalue; even spectrum has none; spectrum invariant under grid-permutation translation (non-vacuous); descriptors translation-invariant (permutation + fractional) |
| `continuation_fold` | ≥ 20 accepted points; fold detected inside (−0.35, −0.12); fold amplitude window; arclength constraint exact at every point; no near-trivial accepted point; up-direction monotone to r_stop, fold-free |
| `discovery_guards_and_deflation` | trivial probe rejected with reason "trivial"; ≥ 1 new cosk pattern; accepted set rms > 0.05; dedupe detects the undeflated repeat; deflated Newton cannot re-find the known state |
| `descriptor_symmetry_and_distinctness` | reflection + fractional-translation invariance (non-vacuous); cos k=1 vs cos k=2 vs sech² bump distinct in kbar/pr/bands |
| `morphology_vs_topology` | two branch points at the same r: kbar agrees (morphology) while norm ratio > 1.4 and even-sector n_pos differs (topology) |
| `mesh_convergence` | N=32/64/128 Newtons converge; strict error decrease (values and coefficients); spectral abscissa converges; fold location converges (|r_f(128)−r_f(64)| < 5e-3) |
| `trivial_solution_contamination` | zero seed converges in 0 iterations and is classified trivial; tiny-noise seed collapses and is rejected |
| `io_roundtrip` | TOML write/parse roundtrip incl. nested tables, bools, arrays; fallback TOML writer exercised directly; CSV formatting incl. NA for NaN |
| `rng_and_seeds_determinism` | TinyRNG determinism/difference/range; seed table deterministic, 26 seeds with the recorded family counts |

## Known risks (untested until the parent's first run)

1. **First execution**: nothing here has been run; a syntax/API slip is
   possible despite the conservative stdlib-only style. The tests are the
   first discriminator.
2. **Deflation changes discovery basins**: with root-removing deflation, some
   seeds that would find a known root now fail (recorded). Discovery counts
   may be smaller than the undeflated run would give; failures.csv shows why.
3. **Continuation near the pitchfork**: the bordered system degenerates at the
   pitchfork; the trivial-approach guard terminates before that, but step
   halvings near the end are expected and logged.
4. **Fold estimate accuracy** is step-size limited (three-point parabola);
   the mesh table shows its convergence but it is not a refined value.
5. **Tolerance windows** in tests (fold window, Galerkin rms within 8%,
   FD-residual bounds, ratio windows [3.4, 4.6]) were chosen with margins
   from hand analysis; if a window is violated by the first run, that is a
   finding to investigate, not to silence.
6. **Localized states may not exist** at r ∈ {-0.10, -0.20} for this
   equation/parameter set — the honest outcome is whatever discovery records.
