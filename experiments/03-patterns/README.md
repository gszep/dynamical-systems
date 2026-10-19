# Experiment 03 — 1D periodic Swift–Hohenberg stationary patterns

Nonlinear stationary states of the subcritical 1D Swift–Hohenberg equation with
Newton solvers, pseudo-arclength continuation through folds, multi-seed branch
discovery with deflation, symmetry-aware morphology descriptors, and
mesh-convergence evidence on dense small grids (N = 32/64/128).

**Status: implemented and first executed (2026-10-20).** The code was
originally written under a concurrent-package-setup hold (no execution); after
release, the pinned runtime executed `scripts/tests.jl`
(**208/208 passed, exit 0**) and `scripts/main.jl` (**exit 0**), writing the
committed artifacts under `results/03-patterns/` (see that README for the
provenance and headline numbers: fold −0.2250098 vs Galerkin −0.225, Galerkin
amplitudes reproduced at both discovery parameters, mesh errors
1.84e-3 → 1.80e-8, phase-mode defect sweep, 41 recorded discovery failures,
no localized states found). Two static-review rounds were fixed before
execution (inner constructor, mass-normalized restricted rates, explicit
known-root exclusion, reflection-restriction scope) and their regressions are
part of the passing suite. `EXPERIMENTS.md` (a repository root file, outside
this experiment's ownership) is updated by the parent.

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

### Reflection-symmetric branch restriction (phase condition, not a translation quotient)

The equation commutes with translations, so every nontrivial periodic
stationary state U comes with a one-parameter orbit U(· - x₀) and the
full-space linearization J has the phase direction U' in its kernel
(‖J U'‖ = 0 — verified in the tests). Newton on the full space is therefore
singular at solutions. We search exclusively in the subspace spanned by the
cosine modes (invariant under the grid reflection x → -x, dimension N/2 + 1).
Its precise scope, demonstrated by regression tests:

* **Invariant restriction, not a quotient.** The operator and nonlinearity map
  even vectors to even vectors, so the restriction is exactly the collocation
  system restricted to reflection-symmetric u: a self-consistent family of
  sub-branches.
* **Phase condition.** Within the subspace the Newton Jacobian is nonsingular
  at solutions whose translation orbit intersects the subspace: the phase
  direction U' is odd, so it leaves the subspace (tested: the restricted
  spectrum has no near-zero rate, min |rate| > 0.05 at the working
  resolution). In the full space the discrete phase mode remains a single
  near-zero eigenvalue (measured ≈ −5.4e-9 at N=64); ‖J U'‖/‖U'‖ is a
  resolution-dependent defect, quantified below — not an exact null.
* **It excludes asymmetric states.** Solutions with no even translate never
  enter the search scope. Example (tested): u = cos(x) + sin(2x) — each
  component alone is a translate of an even function, but no single
  translation makes both even, so this state is invisible to the restriction.
* **It does not identify all translates.** Patterns with discrete translation
  symmetries can appear in the subspace at several even phases (tested:
  cos²x and sin²x are distinct even points and translates of each other);
  dedupe identifies only the sign symmetry u → -u, not general translations.
* **Projection changes Fourier power.** Projecting an asymmetric vector onto
  the subspace removes its odd coefficients (tested), so descriptors are
  computed on the projected (even) solutions by construction.
* **Restricted rates are mass-normalized.** The cosine basis S is not
  orthonormal (S'S = G, diagonal, entries N or N/2), so `*_sub` values are the
  physical rates: eigenvalues of G⁻¹ᐟ²(S' J S)G⁻¹ᐟ² (the generalized problem
  (S'JS)v = λ(S'S)v). The c-coordinate matrix alone would rescale modes by
  diag(G) — at the zero state the critical mode would read (N/2)·r instead of
  r; that defect is measured as a negative case on N = 32/64/128, while the
  mass-normalized zero-state rates equal r − (1−k²)² with the critical mode
  exactly at r.

**Measured phase-mode defect.** The translation phase direction U' is an
exact null vector of the *continuous* linearization, but the undealiased
collocation nonlinearity preserves only grid shifts (continuous translations
are broken by the discretization), so the relative residual
`phase_mode_defect` = ‖J U'‖/‖U'‖ is a resolution-dependent aliasing quantity
— approximately 2·k_Nyquist times the amplitude of the first nonlinear
harmonic past Nyquist. Measured (exposed in `patterns_summary.toml` under
`[phase_mode]`): **0.543** at N=32 (the quintic's k=5 output aliases),
**6.24e-4** at N=64 (its k=9 output aliases), **1.80e-9** at N=128 and
**1.78e-7** at N=256 (both once all significant harmonics fit; the N=256
entry is limited by the measured Newton roundoff floor at that resolution —
see limitations — not by aliasing). The implementation was verified before
interpreting these: the tests check the spectral derivative against
band-limited analytic data and against an independent FD derivative at its
derived truncation order, and check the assembled J·U' against an independent
pointwise/DFT application of r I − (1+∂xx)² + diag(g). The earlier claim that
‖J U'‖ < 1e-9 at the working resolution was a false exact-null assertion and
has been replaced by this measured convergence-based characterization.

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
  1e-30); convergence is always judged on the true residual. Because that
  undeflated merit is zero at known roots, exact and near known roots
  (sign-aware relative distance below `dedup_tol` = 0.05) are excluded
  explicitly: converging there returns `converged = false` with
  `reason = "known_root_excluded"`, never a false success.
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
`not_converged:<reason>` — including `known_root_excluded` for exact/near
re-finds of already-known roots under deflation), Newton iterations, residual,
descriptors and
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
  timings (queue/setup/stages/internal/total with an explicit scope note; the
  supervisor-captured process wall time is authoritative), all seeds,
  tolerances, thresholds, counts, fold numbers vs the Galerkin estimate,
  per-family discovery tallies, morphology-vs-topology evidence, the measured
  `[phase_mode]` defect table, and an
  explicit scope statement (no atlas, no 2D, no snaking claims, no full-space
  stability claims).
* `runs/` — captured stdout of the executed test and smoke runs, committed
  for in-repo provenance.

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
| `transforms` | DFT roundtrip exactness; minimal default-grid construction regression (inner-constructor path; Gdiag weights N, N/2, N; mass-normalized basis orthonormal; Real-convenience path); reflection involution; non-vacuous (u is not even); coeffs↔values roundtrip; exact fractional shift of band-limited data |
| `spectral_operator_expected_spectrum` | assembled dense operator spectrum vs analytic r-(1-k²)² (machine precision); cosine columns are exact eigenvectors; zero-state restricted rates on N=32/64/128: c-coordinate matrix rescales by diag(S'S) (critical mode reads (N/2)r — negative case), mass-normalized rates equal r-(1-k²)² with the critical mode exactly at r (independent expected values, cross-mesh) |
| `fd_operator_expected_spectrum` | exact FD circulant spectrum; (I+D2) eigenvalues differ from spectral values (negative case) and converge at ratio ≈ 4 |
| `newton_stationary_and_independent_expected` | residual < tol; rms matches single-mode Galerkin prediction within 8%; zero mean by symmetry; two coexisting states at r = -0.1 with n_pos = 0 vs 1 (mass-normalized rates); FD cross-check residual decreases with h and is genuinely nonzero |
| `phase_mode_and_removal` | derivative exact on band-limited data; agrees with an independent FD derivative at its derived O((kh)²) truncation (measured 0.0255 = 1−sin(π/8)/(π/8)); J·U' verified against an independent pointwise/DFT assembly; defect ‖J U'‖/‖U'‖ < 0.05 at N=64 (measured 6.24e-4 — an aliasing quantity, not an exact null); full spectrum has exactly one ≈0 eigenvalue (the discrete phase mode); mass-normalized restricted spectrum has none (min rate > 0.05); spectrum invariant under grid-permutation translation (non-vacuous); descriptors translation-invariant |
| `phase_mode_resolution_defect` | defect swept over N=32/64/128/256: strictly decreasing 32→64→128 (measured 0.543 → 6.24e-4 → 1.80e-9) and < 1e-6 for N ≥ 128 once all significant nonlinear harmonics fit; N=256 measured at the Newton roundoff floor (1.78e-7, accepted at ‖R‖ ≤ 1e-6 there because the default 1e-9 is below eps·cond(J_even) ≈ 1.3e-8 at that resolution) |
| `continuation_fold` | ≥ 20 accepted points; fold detected inside (−0.35, −0.12); fold amplitude window; arclength constraint exact at every point; no near-trivial accepted point; up-direction monotone to r_stop, fold-free |
| `discovery_guards_and_deflation` | trivial probe rejected with reason "trivial"; ≥ 1 new cosk pattern; accepted set rms > 0.05; dedupe detects the undeflated repeat; exact known seed → `known_root_excluded` (no false success at a zero of the undeflated merit); near-known seed excluded or lands on a genuinely new root; any deflated success has true residual at tolerance and is far from every known root |
| `descriptor_symmetry_and_distinctness` | reflection + fractional-translation invariance (non-vacuous); cos k=1 vs cos k=2 vs sech² bump distinct in kbar/pr/bands |
| `subspace_scope_not_translation_quotient` | the restriction is not a translation quotient: cos(x) has even translates; cos(x)+sin(2x) has none (every grid translation leaves a >10% odd part) so such states are out of scope; cos²x vs sin²x are retained translated representatives not caught by sign-dedupe; projecting an asymmetric vector changes its Fourier power (kbar) |
| `morphology_vs_topology` | two branch points at the same r: kbar agrees (morphology) while norm ratio > 1.4 and even-sector n_pos differs (topology) |
| `mesh_convergence` | N=32/64/128 Newtons converge; strict error decrease (values and coefficients); spectral abscissa converges; fold location converges (|r_f(128)−r_f(64)| < 5e-3) |
| `trivial_solution_contamination` | zero seed converges in 0 iterations and is classified trivial; tiny-noise seed collapses and is rejected |
| `io_roundtrip` | TOML write/parse roundtrip incl. nested tables, bools, arrays; fallback TOML writer exercised directly; CSV formatting incl. NA for NaN |
| `rng_and_seeds_determinism` | TinyRNG determinism/difference/range; seed table deterministic, 26 seeds with the recorded family counts |

## Known limitations (measured where noted)

1. **Executed and validated (2026-10-20)**: tests 208/208 passed, smoke
   exit 0, artifacts committed under `results/03-patterns/` with logs and
   provenance. Remaining numbers below are the recorded facts and limits of
   the executed configuration.
2. **N=256 Newton roundoff floor (measured)**: the absolute tolerance
   ‖R‖ ≤ 1e-9 is below the achievable floor eps·cond(J_even) ≈ 1.3e-8 at
   N=256 (cond ≈ 4.2e6 from the critical-mode vs Nyquist eigenvalue spread;
   the solve stalls at 1.297e-8 with `no_descent` while the iterate is an
   excellent solution, pointwise residual ~1e-10). The phase-defect sweep
   therefore accepts ‖R‖ ≤ 1e-6 for that measurement point only; all
   resolutions used elsewhere (N ≤ 128) converge under the default
   tolerance. The N=256 defect entry (1.78e-7) is floor-limited, not
   aliasing-limited.
3. **Deflation changes discovery basins (measured)**: with root-removing
   deflation, seeds that would re-find known roots fail instead — the first
   run recorded 38 `no_descent` and 3 `known_root_excluded` rows. At r = −0.1
   the A=1.2 cosk seed failed after P_small was found (basin deflected), and
   the A=0.7 seed found P_large; at r = −0.2, A=0.35 found P_small and A=1.2
   found P_large. Counts are deflated-run counts; failures.csv shows why.
4. **Continuation near the pitchfork**: the bordered system degenerates at
   the pitchfork; the trivial-approach guard terminates before that (both
   fold-study directions terminated with `trivial_approach` in the run),
   with step halvings expected and logged near the end.
5. **Fold estimate accuracy** is step-size limited (three-point parabola):
   measured agreement across meshes is excellent for this branch
   (−0.225018 / −0.2250098 / −0.2250098 at N=32/64/128, and −0.224978
   from the small-seed starts), but the estimate is not a bisection-refined
   value.
6. **Tolerance windows** in tests were all validated by the executed suite;
   any future violation is a finding to investigate, not to silence.
7. **Localized states**: none were discovered at r ∈ {−0.10, −0.20} in the
   executed run (localized seeds: 0/8 accepted at −0.1; 2/8 converged to the
   trivial state at −0.2 and were rejected). This is recorded absence under
   the fixed seed table, not evidence of nonexistence.
8. **Restriction scope**: the reflection-symmetric restriction excludes
   solutions with no even translate (e.g. asymmetric superpositions) and does
   not identify general translations; absence of such states in the outputs
   is a scope limitation of the search, never evidence of nonexistence.
