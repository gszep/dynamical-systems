# Experiment results — inverse bifurcation revival: equilibria, cycles, patterns

*Synthesis of executed local evidence, 2026-10-19 → 2026-10-21. No new numerical runs, searches or fetches were performed for this report; every number is quoted from committed artifacts. Links are relative to this repository; the program context is in the [research synthesis](Reviving%20dynamical%20landscape%20research.md) and the integrator checkpoint [EXPERIMENTS.md](../EXPERIMENTS.md).*

The staged revival program — extending the 2021 equilibrium inverse-bifurcation method to limit cycles and nonlinear PDE patterns — has been executed once end-to-end on the pinned local runtime (Julia 1.10.11, SHA-verified, macOS x86_64; one thread per process, at most three concurrent processes; no CI exists, so all evidence is local). The three outcomes are deliberately asymmetric: **cycles and PDE baselines pass independent reruns**, while the **equilibrium released-gradient gate is retained as FAILED/INCONCLUSIVE** — a scientific result, not an infrastructure blocker.

## 1. Equilibria: reproduced package, unvalidated ∇measure

The original package (BifurcationInference.jl v0.1.4, upstream `534ee7c`, tree-hash verified, MIT notice preserved) reproduces as a **tiny functional run**: `deflationContinuation` found the saddle-node fold pair and `train!` completed 8 genuine iterations (loss 2.1667 → 2.1506), after one dependency pin *within* declared compat (KrylovKit 0.7.1; the drifted 0.8 breaks the load). This is not paper-scale reproduction (800-restart toggle-switch runs were out of scope). Parent reruns passed **184/184 stdlib probe tests** (31.59 s process wall; earlier round 101/101, 31.39 s) — independent mathematical probes: closed-form folds to <1e-10, adjoint velocities to <1e-9, and our own arclength-normalized measure gradient to 7.6e-10 against finite differences *including moving window-endpoint terms* (dropping them shifts the gradient by 0.29 and flips both component signs).

The released `∇measure` — never exercised by the package pipeline when |P|=|D| (the loss switches the measure term off) — was called directly in [audit v2](../results/01-equilibria/gradient-audit-v2/assessment.toml), summarized in the [validation file](../results/validation/equilibrium-audit-v2.toml): 21/21 audit-unit contract passes (3.55 s), assessment exit 1 (19.48 s), gate **INCONCLUSIVE_OR_FAILED**. The continuum gate is inconclusive for both components (reference 0.014910/0.506757; released values drift across ds 0.02→0.0025 with no plateau), and the same-discrete-object gate **fails** for component 2 at ds 0.02/0.01/0.005 (|G−FD| up to 9.7e-3 against a 5.1e-4 budget); FD of the returned adaptive sum is ds-unstable with a component-1 sign change. State counts are identical under ± perturbations, so discrete count jumps do not explain the instability; **no cause is identified**. Scope: scalar saddle-node, θ=(1,−1). The gate is **not validated** — and neither is any stronger claim: no O(ds) order and no incorrect-formula assertion is made. Archived raw sweeps, superseded historical pass flags and injection-verified failure paths (exit 1) are retained as failure receipts. (The audit was briefly interrupted by a provider usage limit and resumed by the same session on another model; completed sweeps were preserved and setup was not repeated.)

## 2. Cycles: accepted phase-fixed shooting baselines

The stdlib-only cycle experiment passed an independent parent rerun: **223/223 tests, exit 0, 10/10 smoke scenarios within declared acceptance sets** ([validation](../results/validation/accepted-cycles.toml)) — queue 0.003 s, supervisor setup 0.10 s, tests 37.74 s, smoke 31.61 s, total 69.45 s (supervisor process walls include Julia startup/JIT and are authoritative; the internal 27.5 s scenario runtime is partial and not the headline). Phase-fixed shooting on a transverse section makes the period a genuine unknown and the reduced Jacobian invertible; the period-sensitivity column is the **exact derivative of the fixed-step-count discrete map** (the continuum identity dx(T)/dT = f(x(T)) is wrong by O(h⁴) and never used). Stuart–Landau: period error 1.5e-8, radius 1.1e-9, radial multiplier 1.8e-10; the phase multiplier is **exactly 1 by rotational equivariance** (proven for the discrete map). Van der Pol (no closed form) is checked structurally: invariance residual order ≈ 4. Inference: recovery error 8.4e-11 (same mesh), 3.9e-9 cross-mesh; implicit gradient vs FD 2.7e-10; an inconsistent observable triple is flagged (residual 0.139), and a two-observable cross-model fit is degenerate until the multiplier exposes it (0.622). Expected negatives (honest non-convergence at μ≤0, guard rejections) are accepted only when they occur. Limits: 2D only; "no cycle found" is a local outcome; no general cycle-inference claim.

## 3. Patterns: nonlinear branch evidence, no localized states

The 1D subcritical Swift–Hohenberg experiment passed **208/208 tests** at the integrated commit (parent rerun 30.40 s tests + 18.41 s smoke process wall per EXPERIMENTS.md; first-run logs committed under [results/03-patterns/runs](../results/03-patterns/)). Measured: **fold at r = −0.2250098** (Galerkin −0.225); **two coexisting periodic states at each** r ∈ {−0.1, −0.2}; mesh errors 1.84e-3 → 1.80e-8 (N 32→128); the phase-mode defect is quantified as an **aliasing quantity** of the undealiased collocation (0.543 → 6.24e-4 → 1.80e-9), replacing a false exact-null assertion; 41 failed discovery attempts are recorded; **no localized states were found** under the fixed 26-seed table — recorded absence, not nonexistence. Final smoke-acceptance and claim fixes (unsupported N256 roundoff-floor attribution, over-broad diagnostic tolerance) are still pending; the parent will update.

## Core insight

Both extensions hinge on the same two facts. First, the neutral phase direction must be made explicit before differentiation: the section (cycles) and the reflection restriction (PDE — a phase condition, **not** a translation quotient, excluding asymmetric states) restore invertibility, and what is differentiated must be the discrete object actually solved — a finite-step/adaptive-sum gradient is not automatically the continuum derivative, as the ds-unstable equilibrium audit demonstrates. Second, inference needs **nonlinear branch data beyond linear onset**: measure gradients live on curved branches with moving window endpoints, and morphology descriptors do not encode branch topology.

## Not established

No general cycle or PDE inference, no complete branch atlas, no 2D spots/stripes, no snaking, no CUDA, no Lean proof, no browser benchmark and no language comparison — none were attempted, and passing standalone suites do not imply any of them; a green stdlib suite is **not** the released-gradient gate.

## Reproduction

- Standalone: `python3 scripts/run_experiments.py --stages 02-cycles --jobs 1 --timeout 300` (default `--julia .tools/julia-1.10.11/bin/julia`; isolated depot; separate queue/setup/process/total timings).
- Cycle/PDE direct commands: [02-cycles README](../experiments/02-cycles/README.md), [03-patterns README](../experiments/03-patterns/README.md).
- Upstream package/audit commands (pkgenv `package_checks.jl`; `run_bounded.py --julia … gradient_checks.jl --mode assess`, which **exits 1 by design**): [01-equilibria README](../experiments/01-equilibria/README.md).

## Next discriminating work

1. **Equilibria:** fixed-quadrature / mesh-transport experiment to isolate the cause of the returned sum's ds-instability (re-discretization is a candidate, not established).
2. **Cycles:** cross-check the pipeline against an independent continuation library, beyond self-consistency.
3. **PDE:** targeted localized-branch search plus dealiasing; confirm the N256 stall cause.
4. **Future stack:** portable viewer and Lean certificate scope remain separate proposals, untouched by these results.
