# Experiment 01 — Equilibrium bifurcation inference: original-package reproduction and paper/code objective probes

Branch `experiments/equilibria`, worktree `dynamical-systems-exp01`.
Authority: user authorized experiments 1–3 autonomous overnight (2026-10-19);
this experiment owns `experiments/01-equilibria/` and `results/01-equilibria/`
only. Runtime pinned and SHA-verified (see `reference/PROVENANCE.md`).

## Objective (EXPERIMENTS.md gate 1)

Pin the original inference source; **distinguish original-package reproduction
from independent mathematical probes**; test normalised vs unnormalised
measures, distance vs squared-distance objectives, implicit gradients vs
finite differences, and singular/empty/exact-target cases.

## What was pinned and reproduced

* **Paper**: arXiv:2106.04243 **v3 LaTeX source** (read directly; exact
  equations listed in `src/EquilibriaProbes.jl`).
* **Released code**: BifurcationInference.jl **v0.1.4 at commit
  `534ee7c23af16d542a248f181f5e6be7056e30b6`** — snapshot under
  `reference/`, read in full before any test was written. Registry tree-hash
  match verified: the registered v0.1.4 is byte-identical to the commit.
* **Original-package reproduction: ACHIEVED (as a tiny run)** — the package
  loads and its full pipeline runs on the pinned Julia 1.10.11:
  `deflationContinuation` found the saddle-node fold pair (bifurcations=2),
  `train!` performed 8 genuine Flux-`Descent(0.01)` iterations on the paper's
  minimal saddle-node model (Loss 2.1667 → 2.1506, ~0.7 s/iteration).
  This is a *tiny functional run*, not a reproduction of paper-scale results
  (800-restart toggle-switch runs are out of scope here).
  Required one dependency pin **within the package's own declared compat**:
  KrylovKit 0.7.1 instead of the drifted 0.8.3 (exact blocker analysis in
  `results/01-equilibria/setup_record.toml`). No source modifications, no
  compat overrides, no substitution of my own implementation for the package.

## Three objectives, kept separate (not "normalised variants" of each other)

Measured on identical inputs (probe p5; artifacts `p5_*.csv/toml`):

1. **Main-text Eq (5) determinant measure** φ = (1+|det|/|d det/ds|)⁻¹,
   arclength-normalised in Eq (7): Ψ = ∫φ ds / ∫ds ∈ [0,1].
2. **Supplemental E (E.1) eigen measure**
   Σ_λ (|d log Re λ/ds|⁻¹+1)⁻¹ — per-eigenvalue rational form, Hopf-capable.
3. **Released-code measure** Σ_λ 2/(2+|sinh(2λ)/dλ|) × erf window, summed
   over branch arclength **without** Eq (7)'s normalisation.

These differ in *detector* and in *normalisation*, not merely by a scale:
* On a FitzHugh–Nagumo equilibrium curve with a Hopf at u₁ = ±√(1−b/τ), the
  determinant stays strictly positive everywhere (min det = 0.008 > 0), so the
  main-text measure is **structurally blind** at the Hopf (φ_eq5 ≈ 0.445 there,
  never 1), while E.1 and the released sinh form hit their maxima (≈ 2 = both
  conjugate-pair members) exactly at the crossing.
* The E.1 rational form and the released sinh form are **different functions**:
  at λ = 2, dλ/ds = 1, E.1 gives 1/3, the released form gives 0.068 — the
  released form decays exponentially away from crossings (fast), whereas Eq (5)
  and E.1 decay algebraically (slow, by design: "non-zero gradients far from
  bifurcations").
* Normalisation: with identical φ, the unnormalised sum equals Ψ·L exactly
  (verified to quadrature error 1e-5 across independent quadratures);
  L = total windowed arclength (≈ 7.6 here) — see below for the behavioural
  consequence.

## Results (probe by probe; full data in results/01-equilibria/)

* **p1 analytic references** — augmented-Newton solves reproduce the
  closed-form fold locations of the paper's minimal models exactly:
  SN θ=(5/2,−1): p* = ∓(5/3)√(5/6) = ∓1.5214515486…; PF θ=(1/2,−1):
  p* = 3(1/4)^{2/3} = 1.1905507890…. Independent expected values, all < 1e-10.
* **p2 implicit gradients (Eq 10)** — the adjoint-solve bifurcation velocity
  matches hand-derived closed forms (SN: ∂p*/∂θ = (−u*, −u*³); PF:
  (−1/u*, −u*²), each derived two independent ways) to **< 1e-9**, and central
  finite differences to < 1e-6. The released package's `velocity` matches my
  implementation and the analytic reference **bit-exactly** (0.0 error).
* **p2c C.8 deformation** — the paper's Moore–Penrose normal deformation
  equals the analytic normal component of the true curve displacement to
  < 1e-12 and is purely normal (⟂ tangent), as Supp C claims.
* **p3 normalisation** — Ψ is discretisation-invariant (< 1e-8 under 8×
  refinement, after splitting the |·|-kink points); the unnormalised branch
  sum equals Ψ·L. **Measured sign consequence**: the count-term contribution
  (|P|−|D|)·log(measure) has *opposite signs* between the paper (log Ψ ≤ 0
  always) and the released objective — measured both for the
  normalisation-only surrogate (indicator window, code φ) and for the
  **actual released functional** (erf window × φ_code continuum value
  M_actual: 1.337 for the SN case, 1.470 for the PF case — both > 1, so
  log M_actual > 0 while log Ψ < 0) — for |P|>|D| (−1.06 vs +0.29),
  |P|<|D| (+0.96 vs −0.385) and the empty-target loss (−2.12 vs +0.58).
  The sign flip follows whenever the summed measure exceeds 1
  (equivalently Ψ·L > 1); that condition held in all measured cases.
  No claim is made that it holds universally.
* **p4 distance vs squared distance** — per-target identity gm2 = gm² exact;
  but the arithmetic mean over targets upweights far targets (E_code = 1.25 ≠
  E_paper² = 1.00 for preds {1.5}, targets {1,3}); the paper objective has a
  **flat minimiser interval** between two targets (gradient exactly 0 inside),
  the code objective a unique midpoint minimiser (gradient −1 at 1.5); p-unit
  rescaling multiplies E_paper by 10 but E_code by 100 (the measure term is
  not p-scale-invariant either — the erf window width and the sinh argument
  carry p-units — so unit choices shift the error-vs-measure balance in both
  objectives; no scale-invariance claim is made for either term).
* **p5 measure separation** — see above; plus the B.6 spurious-maximum family
  F = p + u + (K/3)u³ (det = 1+Ku² ≥ 1, **no crossing exists**): all four
  measure forms approach 1 (Eq 5: 0.998 at K = 10⁶) near u ≈ K^(−1/2) where
  d(det)/ds is steep — B.6 confirmed, and **the D.1 tanh fix does not
  suppress it in this family** (measured values, not gradients).
* **p6 empty/exact/multiprediction targets** — empty prediction set: released `errors` returns
  **1 for any targets** (empty-product init; blind to target location; error-term
  gradient exactly zero — only the measure term can move θ); paper Eq (4) is
  undefined there (NaN recorded; convention gap). **Corrected classification of
  the exact-match cases** (an earlier version conflated them): (C1) the code
  objective with a single matched prediction, E = (p−t)², is *smooth* at the
  match with true gradient 0 — the released **factored** gradient form
  gm2·2v/(p−t) returns NaN (0·Inf), a **removable implementation defect** of
  the factored expression, not a mathematical singularity (near the match the
  same form tracks the true derivative 2dv; the independent unfactored
  derivative 2(p−t)·v is exactly 0). (C2) the paper objective |p−t| has a
  **genuine kink** at the match (one-sided gradients ±v). (C3) multiprediction
  with one matched and one far prediction: the paper E ~ (|d|·A)^{1/2} has an
  **unbounded** gradient (∝ |d|^{−1/2}, verified: 840 vs 2.8 across d = 1e-6..0.1
  with |g|·√d constant to 5%), while the code E = |d|·A is genuinely
  nonsmooth but **bounded** (subgradient → ±A·v, identity verified to 1e-9) —
  the structure behind the p8 fixed-step behaviours.
* **p7 degenerate crossings** — SN cusp (θ₁→0): cond(∂G/∂z) grows, the fold
  velocity vanishes like √θ₁, and for θ₁<0 the folds disappear entirely
  (|P| jumps 0→2 across θ₁=0; central FD across the jump is impossible).
  At θ₁=0 the augmented Jacobian [[0,1],[0,0]] is singular: Eq (10) does not
  exist — as the paper states. Perfect pitchfork (θ₁=0): the rectangular
  ∂F/∂z is the **zero matrix** (tangent undefined) and the augmented Jacobian
  is singular, although the determinant crossing restricted to the u=0 branch
  is non-degenerate — the augmented (branch-agnostic) formulation cannot see
  branch-restricted non-degeneracy. PF velocity diverges like |θ₁|^(−1/3).
* **p8 synthetic inference** — the geometric-equivalence manifold
  θ₂ = −4θ₁³/(27p*²) (derived analytically) keeps both folds at the targets:
  E_paper < 1e-5 along the whole family — the paper's central §1.1 claim,
  verified. **Measured Eq (4) pathology**: the geometric mean over predictions
  gives a √δ tail near exact match (E ~ √(δ·far-distance)) with *unbounded*
  gradient. With the paper's own fixed-step descent (Flux Descent(0.01)) on
  the SN 2-fold/2-target case, E fell from 1.47 to a transient minimum below
  the paper's 0.02 convergence threshold (~0.009), then the growing
  near-match gradients produced oscillations of increasing amplitude that
  **flipped the sign of θ₂ and ejected the iterate from the 2-fold regime
  (|P|: 2→0)** — the paper's piecewise-smoothness boundary. (The released
  `train!` stopping rule Loss<0.02 with |P|=|D| would have halted at the
  transient dip.) The code's squared objective for |P|=|D|=2 is |δ|·far — a
  bounded but *non-vanishing* kinked gradient — so fixed-step descent merely
  stalls at E_code ≈ 0.16 (transient sub-0.02 dips from iteration ≈ 103);
  for |P|=|D|=1 it is δ² (smooth) and fixed-step descent converges to machine
  precision (PF) while the paper objective stalls at |δ| ≈ 0.018. Backtracking
  line search converges everywhere; 8/8 seeded restarts (Mersenne
  Twister, seed 20261019) reach the manifold with target error < 1e-11.
* **p10 gradient validation (the discriminating check)** —
  *Probe side (my normalised main-text Ψ gradient, moving boundaries
  accounted)*: an exact analytic Leibniz adjoint (differentiate under the
  integral at fixed curve parameter u + explicit moving-window-endpoint terms
  u_edge'(θ) = −(∂p/∂θ)/(∂p/∂u)) matches central FD of Ψ to **7.6e-10**
  (both windows, both components; FD-step and quadrature sweeps separate FD
  truncation from quadrature error). Dropping the window-endpoint terms
  changes the gradient by **0.29 and flips both component signs**
  (with: (−0.0747, −0.0205); without: (+0.0413, +0.2481)) — moving
  branch/window boundaries are not a small correction.
  *Released package side*: `∇measure` (called **directly** — the |P|=|D| loss
  path never exercises it) runs and returns finite values
  G = (0.01382, 0.50752) at θ=(1,−1). Against central FD of the continuum
  functional it differentiates (M_full = ∫ erf-window·φ_code·ds, exact erf,
  kink-split Simpson): FD = (0.01491, 0.50676). **Derivative agreement is
  INCONCLUSIVE; the previous correctness and O(ds) claims are retracted.**
  The small-component error increases from 0.00109 at ds=0.02 to 0.00200
  at ds=0.01. These data establish neither a convergence order nor its cause.
  FD of the **actually returned adaptive finite sum** also needs its own
  componentwise plateau: at ds=0.02, its first component changes from
  0.01998 (h=0.001) to 0.02724 (h=0.0001). The old absolute gate floors
  0.05/0.10 exceeded the entire 0.01491 reference component and could admit
  a wrong sign. Historical raw artifacts are preserved, but their `pass`
  flags are superseded by this audit, not evidence of validated derivatives.
  Likewise, full-minus-truncated FD combines tail-integral and endpoint
  contributions; its small value is **not an endpoint-omission bound**.
  Explicit endpoint terms and separate convergence tests are required.
  **Audit v2 (results/01-equilibria/gradient-audit-v2, run 2026-10-20/21)**:
  fixed component tolerance 0.1%·|ref| + 1e-8 (declared, not fitted);
  h ∈ {1e-2..1e-6}, ds ∈ {0.02, 0.01, 0.005, 0.0025}. Continuum reference
  resolved (h, n, radius variations ≤10% of budget): (0.014910, 0.506757).
  Released ∇measure by ds — component 1: 0.01382, 0.01691, 0.01393, 0.01418;
  component 2: 0.50752, 0.50941, 0.50593, 0.50607 — **no resolution plateau;
  continuum gate INCONCLUSIVE for both components.** FD of the returned
  adaptive finite sum has h-plateaux per ds for component 1 at
  0.02724, 0.02154, −0.000168, 0.01348 (sign change at ds=0.005) and is
  not ds-stable; component 2 gives the same-discrete-object gate **FAILED**
  at ds=0.02/0.01/0.005 (|G−FD| 0.0097/0.0021/0.0083 > 5.1e-4) and
  INCONCLUSIVE at 0.0025. State counts and bifurcation flags are identical at
  ± perturbations for h ≤ 1e-4, so discrete-point count jumps do not explain
  the instability; its mechanism is unidentified and no nondifferentiability
  is claimed. Explicit fixed-u Leibniz endpoint terms (1.7e-7, 4.0e-7) and
  normal-deformation boundary flux (4.8e-9, 1.1e-8) are computed separately
  from the tail (identity residual ≤ 3e-11). The assessment exits nonzero.
  Failure paths verified by real call-site injection (exit 1): training
  exception, final-loss Inf, FD exception, FD NaN.
  *Package robustness observation*: at θ=(1.2,−0.9) — regular, folds well
  inside the window — the package's `findRoots` seeded Newton diverges from
  u≈0.02 at p=−2 and its `@assert` fires ("No roots found"); θ=(1,−1) works.
  The seeded root search is basin-fragile independently of any degeneracy.
* **p0/p9 package cross-validation + tiny run** — released `velocity`,
  `errors`, `mean` (empty semantics), `window_function` match my
  reimplementations bit-exactly; `measure` matches to 1.25e-11 (their
  Dual-exact dλ vs my FD dλ); `tangent_field` matches (|dot| = 1). The
  package's own continuation+∇loss pipeline agrees with the probes to
  continuation-step granularity (predictions 6.3e-5 ≪ ds/2; gradient 3.6e-5;
  loss 2.16669 vs my exactly-evaluated 2.16667). Tiny `train!` run completed.

## Paper equation anomaly recorded

Published Eq (9) contains a stray, **undefined `λ` factor** in the
measure-gradient term ((|P|−|D|)·λ·∂Ψ/∂θ·Ψ⁻¹)). The released code has no such
multiplier (its gradient is ∇E − λ_count·∇Φ/Φ with λ_count = |D|−|P|), and no
definition of the stray λ appears in the paper. My probes implement the
code-consistent reading.

## Falsified assumptions / corrections during this experiment

1. "A fresh resolution of the package's declared environment installs and
   loads" — **falsified**: KrylovKit 0.8.3 (inside declared compat) breaks the
   load; needed the 0.7.1 pin (within range).
2. "The load failure is a Julia-1.10-specific method-signature rejection" —
   **falsified** by a minimal reproduction; true cause was RecursiveVec API
   drift in KrylovKit 0.8.
3. "Pkg.add by URL pins the revision" — **blocked** by a LibGit2 credential
   prompt in this environment; registered-version add with tree-hash
   verification used instead.
4. "The D.1 tanh fix suppresses spurious measure maxima" — **not in the
   steep-slope/no-crossing family** (all four forms ≈ 1 there); the fix may
   address gradient dynamics, but value-level suppression is absent.
5. "Fixed-step gradient descent converges on the paper objective" —
   **falsified** for |P|=|D|≥2: the √δ tail + unbounded gradient first stalls,
   then (measured) ejects the iterate across the |P|-change boundary after a
   transient sub-threshold dip.
6. "The released code's Φ is just an unnormalised Ψ" — **incomplete**: it is
   unnormalised AND uses a different per-point detector (sinh vs rational);
   both axes measured separately.
7. "The manifold check would show E ~ 1e-12" — the √δ tail amplifies
   Newton-level fold residuals (~1e-12) into E ~ 1e-6; tolerance set from the
   measured mechanism, not from the naive expectation.
8. **(2026-10-20 review round)** "the exact-match gradient NaN is a
   mathematical singularity" — **corrected**: for the squared objective with
   a single matched prediction it is a *removable implementation defect* of
   the released factored expression (the objective is smooth there, true
   gradient 0); genuine nonsmoothness lives in the unsquared |p−t| kink and in
   the multiprediction geometric-mean tails.
9. **(2026-10-20 review round)** my p3 "code measure term" was a
   **normalisation-only surrogate** (indicator window, code φ): the actual
   released per-point measure multiplies by the erf window. Relabelled; the
   actual functional (M_actual = 1.34/1.47 > 1) confirms the sign flips.
10. **(2026-10-20 review round)** an invented "signed vs absolute determinant
    discrepancy" between paper and code was **removed**: the paper's \Det bars
    are determinant notation (Supp B: Det = Π λₙ, the signed determinant),
    consistent with the released `det(∂Fu)`.
11. **(2026-10-20 review round)** a latent **column-order bug in my FHN ∂F/∂θ
    reshape** (∂F/∂a was [0,0] instead of [0,1/τ]) was fixed; no committed
    artifact was affected (FHN Fθ was never exercised by the earlier probes),
    and every jacobian column of every model is now FD-tested at equilibrium
    and off-equilibrium points.
12. **(2026-10-20 review round)** my own Ψ adjoint initially contained a
    sign-flipped quotient rule in the integrand partial — **caught by the
    new integrand-level FD unit test** (mismatch 0.08–0.38 vs FD) and fixed;
    the final adjoint agrees with FD to 7.6e-10. The unit test is retained.
13. "The released package's ∇measure is exercised by the pipeline" — it is
    **not**: with |P|=|D| the loss path switches the measure term off; the
    gradient had never run in any committed check before p10 called it
    directly.

## Remaining blockers / open items (honest limits)

* The package tiny run covers N=1 and 8 iterations only; paper-scale runs
  (toggle switch N=2/M=5 with 800 restarts, E. coli N=4/M=21) were NOT
  attempted (out of overnight scope; each is hours of continuation).
* The released-∇measure derivative gates remain INCONCLUSIVE/FAILED after
  audit v2 (above). Open question: why the returned finite sum's derivative is
  ds-unstable at fixed state count. Re-discretisation sensitivity is a
  candidate, but no cause is established.
* `findRoots` seed fragility (assert fires at regular θ where the seeded
  Newton diverges, e.g. (1.2,−0.9)) is recorded but not characterised.
* My probes replace deflated continuation with warm-started augmented Newton
  on known branch structure; they are **independent mathematical probes**, not
  a package substitute — branch discovery robustness (deflation, maxRoots=3)
  is untested here.
* No CI exists for this repository; all evidence above is from local runs on
  the pinned runtime (hardware/timing record in `runtime.toml` and
  `setup_record.toml`; invocation timings are labelled per actual run).

## Reproduce

```sh
JULIA=.tools/julia-1.10.11/bin/julia   # read-only shared binary
export JULIA_DEPOT_PATH=$PWD/.julia-depot JULIA_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd experiments/01-equilibria
$JULIA tests.jl                          # 184 expected-value tests, exit 0
$JULIA main.jl                            # probes p1–p8, p10 (probe side)
                                         # -> results/01-equilibria/
JULIA_NUM_PRECOMPILE_TASKS=2 $JULIA --project=pkgenv package_checks.jl   # p0/p9
JULIA_NUM_PRECOMPILE_TASKS=2 python3 run_bounded.py --julia $JULIA --record R.json -- gradient_checks.jl --mode {reference|sweep --ds D|assess}  # audit v2; assess exits 1 (INCONCLUSIVE/FAILED)
# all three entry points accept --out-dir DIR so reruns do not overwrite the
# committed artifacts; package_checks.jl also accepts --inject-failure, which
# must exit 1 (regression proof of the failure-propagation path).
```

`pkgenv/{Project,Manifest}.toml` (committed) record the exact resolution,
including the KrylovKit 0.7.1 pin. The upstream snapshot under `reference/`
includes the original MIT LICENSE (Copyright (c) 2021 Gregory Szep), preserved
verbatim; no external licensing claim is made beyond that.
