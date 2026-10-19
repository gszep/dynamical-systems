# Experiment 02 — Cycles: phase-fixed periodic-orbit shooting, Floquet stability and synthetic inference

Baselines for **gate 2** of `EXPERIMENTS.md`: an actual phase-fixed periodic-orbit
solve with unknown period, parameter inference from cycle observations, Floquet
stability from numerically integrated variational equations, and an independent
analytic Hopf/Stuart–Landau reference. Julia **stdlib only** (`LinearAlgebra`,
`Printf`, `Random`, `Test`, `TOML`) — no packages, no builds.

**Validation status: EXECUTED 2026-10-20** on this host (single process, one
Julia/BLAS/OMP thread, Julia 1.10.11 from the pinned repo-local runtime).
Measured evidence: test suite **223/223 pass** (~35 s wall); smoke run
**10/10 scenarios within their declared acceptance sets, exit 0** (~28 s wall;
per-scenario timings in `results/02-cycles/summary.toml`). Artifacts are
committed under `results/02-cycles/`. These are measurements for *these*
scenarios on *this* host with the pinned runtime — evidence for the pipeline's
design contract, not a universal or exhaustive scientific claim, and no CI
configuration exists (per AGENTS.md, none is claimed).

## What is and is not claimed

Claimed (once the run passes): the pipeline solves genuinely nonlinear
phase-fixed shooting problems with unknown period, computes Floquet multipliers
via variational equations, detects model mismatch in cycle observables, and
reproduces the independent Stuart–Landau reference within discretization-honest
tolerances.

Not claimed:

- The Stuart–Landau analytic special case does **not** "solve" general periodic
  orbit inference. It is the *test oracle* only. The pipeline itself is
  model-generic and is exercised without any analytic shortcut on the Van der
  Pol oscillator, whose cycle has no closed form; there the checks are
  *structural* (residual, one neutral multiplier, cross-grid consistency,
  self-convergence, phase-independence of the period) rather than comparisons
  to known values.
- "No cycle found" for `mu <= 0` is a local search outcome, **not** a proof of
  nonexistence.
- Guard rejection of the equilibrium spurious root certifies only that the
  returned root is not a listed equilibrium; it does not certify absence of
  other spurious roots.
- Hopf detection is an equilibrium computation and is deliberately independent
  of the cycle solve. Finding the Hopf point is **not** completion: at the Hopf
  point the "cycle" degenerates into the equilibrium.
- This is 2D only; higher-dimensional sections, multi-shooting and collocation
  are future work.

## Mathematical formulation

### Models

**Stuart–Landau / supercritical Hopf normal form** with parameters
`theta = (mu, omega)`:

```
x1' =  mu*x1 - omega*x2 - (x1^2 + x2^2) x1
x2' =  omega*x1 + mu*x2 - (x1^2 + x2^2) x2
```

Equilibrium at the origin with eigenvalues `mu +- i*omega`; Hopf at `mu = 0`
with transversality `d(Re lambda)/dmu = 1`.

**Van der Pol** (Liénard coordinates), parameter `theta = (mu,)`:

```
x1' = x2
x2' = mu*(1 - x1^2)*x2 - x1
```

For `mu = 1` it has one attracting limit cycle with no closed-form orbit,
period or multipliers.

### Phase-fixed shooting with unknown period

Given an anchor point `xa` with nonzero flow `va = f(xa)`, the phase-fixing
hyperplane is `{x : <x - xa, va> = 0}` with unit in-section direction
`dhat = perp(va)/|va|`. The reduced shooting unknowns are `(s, T)` with

```
x0(s) = xa + s*dhat,
G(s, T) = x(T; x0(s)) - x0(s) = 0   in R^2.
```

This is equivalent to the textbook 3-unknown system
`[x(T;x0) - x0 ; <x0 - xa, va>] = 0` with the phase equation built into the
parametrization; it fixes the phase (the residual has a whole orbit's worth of
solutions otherwise). The period `T` is a genuine unknown — initial guesses are
produced by a generic heuristic (integrate a transient, average the spacing of
successive maxima of `x1`), and in robustness tests the guess is deliberately
30% wrong. The Jacobian used by Newton is

```
dG = [ Psi*dhat - dhat  |  P(T) ]
```

with `Psi = d x(T)/d x0` the monodromy and `P = d x(T)/d T` the period
sensitivity of the *same fixed-N discretization* — both are exact derivatives
of the discrete residual. `P` needs care: with `N` fixed the step size
`h = T/N` changes with `T`, so the continuous-flow identity
`d x(T)/d T = f(x(T))` is wrong by `O(h^4)` for the discrete map and is never
used in a Jacobian. `P` is computed by writing the `N`-step map as RK4 on the
time-rescaled system `dx/dtau = T*f(x)`, `tau in [0,1]`, with `N` steps of
`1/N`: the iterates are identical (RK4 stages are homogeneous in the step
size), and `T` becomes an ordinary parameter of the rescaled rhs, whose
sensitivity equation translated back to physical time is

```
P' = J(x) P + f(x)/T,   P(0) = 0.
```

A coarse-mesh test verifies `P` against a fixed-N finite difference in `T`
(and confirms the naive `f(x(T))` column is visibly `O(h^4)` off there). Damped
Newton with backtracking (Armijo decrease of the residual norm) solves the
system; the discrete residual is driven to ~1e-12 relative to the state scale.

### Spurious equilibrium roots: exclusion, guards, deflation

The equilibrium family `(x_eq, T-any)` solves any shooting residual. With the
section formulation it is a root only if the section hyperplane contains
`x_eq`, i.e. `<x_eq - xa, f(xa)> = 0`. This fails generically but happens
*exactly* when the anchor sits on a circular cycle (Stuart–Landau: the section
is radial and passes through the origin), so the tests exercise it
deliberately with hand-built fixtures:

1. **Nonzero-orbit guards** (always on): a converged root is rejected unless
   `|x0 - eq_j| >= guard_rtol * max(|xa - eq_j|, 1)` for every listed
   equilibrium and `|f(x0)| >= guard_ftol * max(|va|, 1)` (floor 1 documents
   the assumed state scale). Guard rejections carry the measured distances in
   their message — failures are diagnosable, never silent.
2. **Deflation** (opt-in, Farrell-style): solve `sigma(s,T) * G = 0` with
   `sigma = prod_j |x0 - eq_j|^(-alpha)`, `alpha = 2`. Deflation preserves all
   other roots, multiplies the Jacobian by the smooth factor
   `J -> J + G * grad(log sigma)^T`, and pushes Newton away from the deflated
   equilibria (near them the deflated merit diverges). It is singular exactly
   at the deflated points, which is reported as `DeflationSingular`.
3. **Period bounds** `T in (T_min, T_max)`: the T-direction is degenerate at an
   equilibrium root (`dG/dT = P = 0` there, since the trajectory, `f` and hence
   the period sensitivity all vanish along it) and `T -> 0` collapses onto it;
   bounds plus out-of-bounds reporting keep the solve honest.

The test pair is deterministic: started deep in the linear regime near the
equilibrium with a period-aligned guess, plain Newton provably converges to
the spurious root (the linearized reduced system has the equilibrium as its
only root and Newton lands on it in one step) and the guard rejects it, while
deflated Newton doubles its radius away from the equilibrium each step
(`|G|/|x0|^2` decreases monotonically along the doubling) and converges to the
genuine cycle.

### Floquet multipliers from the discrete variational equations

The state is integrated with classical RK4 jointly with the variational
equations `Psi' = J(x(t)) Psi`, the period sensitivity
`P' = J(x(t)) P + f(x(t))/T` (see the shooting section for why `P` is needed)
and, for gradients, the parameter sensitivities
`S' = J(x(t)) S + df/dtheta(x(t))`. Because RK4 stages are smooth in the initial
data, the period and the parameters, differentiation commutes with the method:
`Psi` is *exactly* `d x(T)/d x0`, `P` exactly `d x(T)/d T` (with `N` fixed) and
`S` exactly `d x(T)/d theta` of the discrete map. This is verified against
central finite differences of the discrete map in `x0` and in `T` at `N` fixed
(agreement to FD truncation error only, and central FD beats forward FD by
>10x; on a coarse stable mesh the naive continuous column `f(x(T))` is visibly
`O(h^4)` away from the true fixed-N `T`-derivative).

Multiplier semantics, checked explicitly:

- the **phase direction is neutral** and this neutral multiplier is *not* a
  stability statement; the discrete structure is now measured and understood
  (see the next subsection);
- **stability** is decided by the remaining multipliers
  (`exp(-2*mu*T)` for Stuart–Landau);
- the classification tolerance `tol_neutral = 5e-4` defines an honest
  resolution limit: a Stuart–Landau cycle with `2*mu*T < 5e-4` (i.e.
  `mu <~ 5e-5` at `omega = 1.3`) has a radial multiplier numerically
  indistinguishable from neutral and is flagged `:degenerate_neutral`
  — a classification limit, while the *measured* multiplier gap
  `|lambda_2 - 1|` remains accurate and is reported alongside.

### The neutral multiplier of the discrete map (measured + proven)

A review-driven diagnosis corrected our initial expectation that
`|lambda_1 - 1| = O(h^4)`. That is false, and the measured structure is
sharper:

- **Stuart–Landau (proven exact)**: the SL field is rotation-equivariant,
  `f(R_phi x) = R_phi f(x)`, and RK4 preserves this exactly (each stage is
  `f` at a rotated point, and rotations commute with the affine stage
  combinations). Hence the N-step period-T map obeys
  `Phi(R_phi x) = R_phi Phi(x)`, a converged fixed point `x*` comes with a
  whole *exact circle* of fixed points `R_phi x*`, and differentiating that
  identity gives `Psi (R' x*) = R' x*`: the phase multiplier is **exactly 1**
  for the discrete map at every N. Measured deviation: `6.7e-16 .. 1.0e-11`
  over N = 32..256 — non-monotone, driven by the accepted Newton residual
  and roundoff, with no order signal. Order/monotonicity assertions on it
  are invalid; the tests assert the symmetry-predicted roundoff level
  (`< 1e-9`) instead.
- **Van der Pol (measured, no symmetry proof)**: vdp has no rotational
  symmetry, yet the measured `|lambda_1 - 1|` likewise sits at the
  variational roundoff-accumulation floor (`<= 1.4e-13` over N = 64..512),
  does not scale with `h` and is insensitive to `tol_newton` below ~1e-13
  (verified by a tolerance-scaling experiment). This is consistent with RK
  backward error analysis — the map agrees with the exact flow of a
  modified field to all algebraic orders, and exact flows have exact unit
  phase multipliers — but we state it as a measured regularity, not a
  theorem.
- **The genuine O(h^4) phase-structure error** is carried by the *invariance
  residual* `|Psi f(x0) - f(x0)|/|f(x0)|`: at a discrete fixed point,
  `f(x0)` acquires an O(h^4) component off the exact eigenvector. Measured:
  SL `5.2e-6 -> 3.2e-7 -> 1.9e-8 -> 1.2e-9` (N = 32..256, order 4.0
  throughout) and vdp `1.02e-4 -> 5.50e-6 -> 3.20e-7 -> 1.93e-8`
  (N = 64..512, orders 4.22/4.10/4.05). The convergence tests assert
  order ~4 on THIS quantity (and on period, radius, radial multiplier), not
  on the eigenvalue deviation.

### Hopf detection (independent of the cycle solve)

Bisection on `g(p) = max Re lambda(J(x_eq(p)))` with the equilibrium re-found
by Newton at each evaluation (finite-difference Jacobian by default, so the
detector shares no analytic code with the cycle solver). Transversality is a
central finite difference across the bracketed point. For Stuart–Landau the
independent expected values are `mu_H = 0` and transversality `1`.

### Synthetic cycle parameter inference

Forward map: `theta -> (solve phase-fixed shooting on a FROZEN section,
warm-started) -> observables`. Observables are `:radius` =
`|x0* - eq_ref|` (for Stuart–Landau this *is* the cycle radius; no circularity
is assumed anywhere in the solve), `:period` = `T*`, and optionally
`:logmult` = `log|lambda_2|`. Scales `(2.0, 2*pi, 4.0)` normalize the
least-squares residual.

Inverse step: Gauss-Newton / least squares (`dtheta = -(J_scaled \ r_scaled)`,
so a square system is Newton and a tall one is QR least squares). Gradients:

- **implicit**: at the converged shooting solution,
  `J_red * d(s,T)/d theta_k = -S(T)[:,k]`, then chain to observables, with
  `J_red = [Psi*dhat - dhat, P]` the *exact* Jacobian of the fixed-N discrete
  residual (both columns are exact discrete derivatives, including the period
  column `P`). This is the cycle analogue of the implicit layer used for
  equilibrium bifurcations in the NeurIPS 2021 work — the phase-fixed system
  is exactly what makes `J_red` invertible (the neutral direction is removed
  by the section) — and it is the exact derivative of the converged discrete
  solve, which is verified against central finite differences of the
  re-solved pipeline. Requires observables `(:radius, :period)` and an
  analytic `param_rhs`.
- **finite differences** through the full re-solve (used automatically when
  `:logmult` is among the observables; second-order sensitivity machinery for
  an implicit multiplier gradient is future work).

The two paths are cross-checked against each other and, for Stuart–Landau,
against the independent closed-form gradient
`d radius/d mu = 1/(2 sqrt(mu))`, `d period/d omega = -2*pi/omega^2`.
Returned fields are self-consistent: if the Gauss-Newton loop exits without
recovering (max iterations, stall, gradient failure), the forward map is
re-evaluated at the returned `theta_hat`, so `residual_scaled`, `final_obs`
and the mismatch flag always describe `theta_hat` (a `max_iter=1` test
enforces this, including the mismatch flag).

Negative/honesty cases: invalid targets (radius/period must be positive)
throw; an *inconsistent* observable triple (multiplier squared — off the
Stuart–Landau surface `log lambda = -2 mu T` by `|log lambda|`) produces an
irreducible residual that is flagged as mismatch, never silently fitted; a
**cross-model** case fits the Stuart–Landau family to Van der Pol
observables: `(radius, period)` fits exactly (the map is a bijection — an
honest demonstration that two observables cannot identify the model class),
while adding the nontrivial multiplier exposes the mismatch. Seeded noise
(`sigma = 1%`, fixed MersenneTwister seeds) is checked against linearized
error propagation, and same-N vs cross-N recovery distinguishes exact
identification from discretization-limited identification.

### Independent analytic Stuart–Landau reference (tests/reports only)

In polar coordinates `z = x1 + i*x2 = r e^{i phi}` the normal form becomes

```
r' = mu*r - r^3 ,   phi' = omega
```

so for `mu > 0`: cycle radius `r* = sqrt(mu)` (supercritical amplitude law),
period `T = 2*pi/omega`, radial linearization `d(r')/dr = mu - 3r*^2 = -2 mu`
giving the nontrivial multiplier `exp(-2*mu*T)` next to the neutral (phase)
multiplier `1`. Equilibrium eigenvalues `mu +- i*omega` give Hopf at `mu = 0`.
At `theta = (0.5, 1.3)` these are `r* = 0.70711`, `T = 4.83322`,
`lambda_radial = exp(-2*pi/1.3) ~= 7.96e-3`. **No solver consumes these
values**; they appear only in assertions and reports.

### Tolerance honesty policy

- The Newton residual is a *discrete* residual: ~1e-12 says nothing about the
  O(h^4) distance to the continuous orbit. Accuracy claims therefore come
  from convergence studies (observed order ~4 for RK4 across N in
  {32,...,512} for period, radius, radial multiplier and the neutral
  deviation), not from magic absolute constants.
- Where a fixed number is asserted (e.g. `|T - 2*pi/omega| < 1e-6` at N=256) it
  is sized generously above the expected `C*h^4` discretization error and the
  order test, not the constant, carries the precision claim.
- Near-degeneracy is *reported* (multiplier gap, classification flag,
  `near_degenerate` diagnostic) rather than hidden.

## Files

- `src/CycleBaselines.jl` — the module (models, RK4+variational+sensitivity
  integration, section, guards, deflation, Floquet classification, Hopf
  detection, settle heuristic, inference layer, CSV/TOML writers).
- `main.jl` — scenario driver writing artifacts (default
  `<repo>/results/02-cycles`, override with `--out-dir`).
- `tests.jl` — `Test`-stdlib suite (independent expected values, negative
  cases, convergence, guard/deflation determinism, inference, artifacts).
- `../../results/02-cycles/` — artifact directory (manifest in its README).

## Runnable commands

Runtime (pinned by the parent):
`/Users/gszep/Documents/repos/dynamical-systems-research-20261019/.tools/julia-1.10.11/bin/julia`

```sh
JULIA=/Users/gszep/Documents/repos/dynamical-systems-research-20261019/.tools/julia-1.10.11/bin/julia

$JULIA -t 1 experiments/02-cycles/tests.jl        # test suite
$JULIA -t 1 experiments/02-cycles/main.jl          # full artifact run
$JULIA -t 1 experiments/02-cycles/main.jl --quick  # smoke run (reduced N sweep)
$JULIA -t 1 experiments/02-cycles/main.jl --out-dir /tmp/exp02-smoke
$JULIA -t 1 experiments/02-cycles/main.jl --only sl_reference
```

All paths are anchored to the scripts, so the working directory does not
matter. Single process, one Julia thread, `N <= 512` everywhere (the
cross-grid recheck uses `N = 512` on a `N = 256` solution). Expected wall
time: the full run in seconds to ~a minute on the 12-core macOS host; the
three-process parent budget is respected trivially by this single process.

## Measured outcomes (run of 2026-10-20, this host, pinned Julia 1.10.11)

Test suite: **223/223 pass** (~35 s wall, single thread). Smoke run:
**10/10 scenarios within their acceptance sets, exit 0**, total 27.5 s
(setup 0.34 s; per-scenario timings in `results/02-cycles/summary.toml`).
These numbers are per-run measurements on this host, not universal claims.

| Scenario | Measured outcome |
|---|---|
| `sl_reference` | `SolveSuccess`; `T = 4.8332194817` (err 1.5e-8), radius `= 0.7071067823` (err 1.1e-9), radial multiplier err 1.8e-10, phase multiplier deviation 1.7e-15 (exactly 1 by equivariance, see above), invariance residual 1.19e-9, `:stable_cycle`, one neutral |
| `sl_convergence` | period/radius/radial-multiplier errors decay with observed orders 3.98/4.03/4.03, 3.99/4.03/4.28, 4.00/4.02/4.16 (N = 32..512); neutral deviation flat at 6.7e-16..1.0e-11 (no order signal, as proven above) |
| `hopf_amplitude` | `mu_H = 2.9e-11` (independent expected value 0), transversality 1.0; amplitude law `r/sqrt(mu) = 1.0000000, 0.9999996, 0.9999642` at mu = 1e-2, 1e-4, 1e-6; `:degenerate_neutral` correctly flagged at mu = 1e-6 with gap 9.67e-6 |
| `negative_mu` | honest failures at mu = 0 and mu = -0.5: both `NewtonNotConverged` (resnorm 6.5e-9 / 4.9e-10) — accepted as the expected negative outcome; no fake cycle |
| `inference` | same-N recovery error 8.4e-11; cross-N (512->256) error 3.9e-9 (discretization-limited as designed); seeded-noise recovery error 1.8e-2 with propagated observable error 0.037 at the injected 1% noise level; implicit-vs-FD gradient 2.7e-10; implicit-vs-analytic 1.1e-8 |
| `inconsistent_observable` | consistent triple recovered (residual 2.2e-9); multiplier-squared triple NOT recovered, residual 0.139, mismatch flagged |
| `vdp_reference` | `SolveSuccess`; period `T = 6.6632872562`, phase point `(1.98178, -0.27635)`, multipliers `{1 + 8.9e-16, 8.597e-4}` (attracting, non-neutral), cross-grid gap at N=512 ~1e-10-scale, invariance residual 3.2e-7 |
| `vdp_convergence` | period differences shrink with Richardson ratios 15.0/15.6 (asserted in [4,64]); invariance residual decays with orders 4.22/4.10/4.05 (the genuine O(h^4) phase-structure quantity); neutral deviation <= 1.4e-13 with no h-scaling |
| `vdp_inference_crossmodel` | self-inference error 7.8e-13; SL `(radius, period)` fit to vdp recovers (residual 8.3e-10 — model class underdetermined by two observables, as designed); adding `logmult` flags mismatch: residual 0.622, `:stalled`, not recovered |
| `spurious_roots` | SL: plain Newton guard-rejected at the equilibrium, deflated solve matches the analytic reference (radius err 1.1e-9, period err 1.5e-8); vdp: plain rejected, deflated solve's period agrees with the independent settle-based section solve to 2.6e-14 |

### Smoke success semantics (acceptance audit)

Every scenario declares an acceptance set for its outcome (`accepted` flag in
`summary.toml`, `[audit]` block with `all_accepted`, `failed_acceptance`,
`crashed`). Positive scenarios must actually solve/recover; explicitly negative
scenarios (`negative_mu`, the inconsistent-observable and cross-model mismatch
fits, the spurious-root guard cases) are accepted only when the expected
honest failure or mismatch flag occurs. The run exits 1 on any scenario crash
OR any outcome outside its acceptance set — a required solve that silently
fails can no longer return exit 0, and an expected negative can no longer fail
the run.

Failure of any assertion is a real, reportable finding — the assertions encode
the pipeline's design contract, not wishful thinking. The two vdp
deflation/robustness cases were the highest-uncertainty predictions (line-search
dependent convergence); both now pass, and even there the tests are written so
that the dishonest outcome (silent fake success or landing on the equilibrium)
is what fails.

## Pinned upstream references

The following URLs were pinned and verified during the repository's audited
research pass (2026-10-19, see `reports/` and `research_notes/`); they were
**not re-fetched** while writing this code, and no claim below relies on
unverified web content. All formulas used are re-derived above.

- NeurIPS 2021 paper *Parameter Inference with Bifurcation Diagrams*:
  https://arxiv.org/abs/2106.04243 (implicit-differentiation architecture this
  experiment extends from equilibria to cycles)
- BifurcationInference.jl source: https://github.com/bifurcationkit/BifurcationInference.jl
- BifurcationKit periodic-orbit shooting documentation:
  https://docs.sciml.ai/BifurcationKit/stable/periodicOrbitShooting/
- Farrell, Bünner, Davison, Deflation for finding distinct roots:
  https://arxiv.org/abs/1603.00809 (deflation technique used here)
- AUTO-07p (periodic-orbit continuation heritage):
  https://github.com/auto-07p/auto-07p
- Thesis future-work chapter (designing limit cycles):
  https://github.com/gszep/thesis/blob/master/chapters/6.tex

## Ownership and integration

This experiment owns `experiments/02-cycles/` and `results/02-cycles/` only.
`EXPERIMENTS.md` and other root files are updated by the parent integrator;
evidence from the first executed run should flow there together with the
recorded timings, tolerances and failure counts written by `main.jl`.
