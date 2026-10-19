# Experiment checkpoint

## Authority and objective

On 2026-10-19 the user authorized experiments 1–3 and autonomous overnight work.
The canonical orchestrator is session `ses_eb033f163ffevGQoqZJbp7OB85`.
The research is in commit `5218aef`; implementation starts on
`experiments/overnight-baselines` in the isolated research clone.

The hybrid architecture is an assistant proposal, not a user-imposed language
constraint. We will test a pinned Julia baseline first. No CUDA, browser stack,
Lean theorem or exhaustive landscape coverage is claimed by these experiments.

## Acceptance gates

1. **Equilibria:** pin the original inference source; distinguish original-package
   reproduction from independent mathematical probes. Test normalized versus
   unnormalized measures, distance versus squared-distance objectives, implicit
   gradients versus finite differences, and singular/empty/exact-target cases.
2. **Cycles:** implement an actual phase-fixed periodic-orbit solve with unknown
   period, parameter inference from cycle observations, Floquet stability and an
   analytic reference. Check mesh/time-step convergence and neutral multiplier;
   Hopf detection alone is not completion.
3. **Patterns:** solve and continue nonlinear Swift–Hohenberg stationary states,
   retain unstable branches where possible, distinguish discovery from completeness,
   compare spatial resolutions and symmetry-aware morphology descriptors. A small
   1D atlas is a baseline, not completion of 2D spots/stripes or PDE inference.

## Current state

- Research notes and report were committed and pushed; source-reading limitations
  are disclosed in the report.
- Local host: macOS x86_64, 12 logical CPUs, 64 GiB RAM. Julia is not on PATH.
- Runtime setup completed: official Julia 1.10.11 macOS x86_64 archive verified
  against published SHA-256
  `bdc6f52ee4fdcf13f0d0e2480733742b7eefb7fbfc66c518daca6e19e804f34b`.
  Download/extraction took 18.63 seconds. Binary (ignored, not committed):
  `.tools/julia-1.10.11/bin/julia`; `--version` succeeded.
- Three linked Lunaroute builders were launched. Experiment 1 exclusively owns
  package setup and may run tests; experiments 2–3 are code-only until setup ends.
  Each must commit in its own worktree; the orchestrator integrates and tests.
- An hourly heartbeat was created successfully through the runtime tool:
  `17a3bc89`, eight runs maximum, cron `17 * * * *`, timezone `Asia/Tokyo`.
  First run: 2026-10-20 00:17 JST; expiry: 2026-10-20 08:22 JST.
  Tool target agent: `92faf50a-40a6-45d1-af0b-20718405c76f`.
  This is a wake-up prompt, not evidence that numerical experiments completed.
- Next check: on child completion, inspect focused commits, integrate them into
  `experiments/overnight-baselines`, then run independent suites within the budget.
- `artemis` has not been accessed. No machine or account configuration changes are
  needed for the initial CPU baselines.

## Linked work (do not duplicate)

| Experiment | Canonical child session | Worktree / branch | Initial task |
|---|---|---|---|
| 1 | `ses_eab758964ffeJCzR58tBi155La` | `../dynamical-systems-exp01` / `experiments/equilibria` | Pin upstream, attempt package reproduction, implement objective/gradient probes; setup owner |
| 2 | `ses_eab75894bffeWRbWOtuYsp6WQG` | `../dynamical-systems-exp02` / `experiments/cycles` | Stdlib phase-fixed shooting, cycle inference and numerical Floquet; code only initially |
| 3 | `ses_eab758926ffe1HvejhOsS6e6uO` | `../dynamical-systems-exp03` / `experiments/patterns` | Stdlib nonlinear 1D Swift–Hohenberg continuation and morphology; code only initially |

The code-only hold expires when experiment 1 reports package setup finished or
stopped; it is an orchestration resource constraint, not a request for user approval.
Children notify automatically: do not poll them or duplicate their work. A blocked
upstream package does not block independent analytic cycle/PDE baselines, but it
does block claiming end-to-end reproduction of the original implementation.

## 2026-10-20 early overnight checkpoint

- Draft PR #1 tracks this branch against `research/landscape-discovery`.
- Cycle implementation `fb6f32c` and PDE implementation `8f9f3e0` were received
  and integrated as `125a086` and `48916f5`. Both are **unexecuted**; code volume
  is not completion evidence. Their readmes' expected values remain predictions.
- `scripts/run_experiments.py` provides at most three one-thread processes,
  per-process deadlines, process-group cleanup, invocation-isolated outputs and
  separate queue/setup/process/total timings. Only Python parsing and `--help`
  have been checked; the Julia suites have not yet run.
- Astra static review `ses_eab4d7f9affesKGLN22Kk6LPor` found six concrete defects
  or scope errors. Both existing builders have focused correction tasks, still
  code-only pending the setup-owner completion notification.
- Falsified implementation assumptions: `f(x(T))` is not the exact period
  derivative of fixed-step-count RK4; unnormalized cosine-basis eigenvalues are
  not physical rates; an even/reflection restriction is not a translation
  quotient; an undeflated zero residual alone does not exclude known roots.
- Other pending corrections: invalid Julia outer-constructor `new(...)` call;
  cycle inference final parameters may not match its returned residual after an
  exhausted iteration budget. Regression tests were requested for every finding.
- Next discriminating checks: corrected source commits, then the minimal PDE
  constructor/zero-state spectrum and coarse-mesh cycle sensitivity tests before
  interpreting any larger run. Experiment 1 has not yet returned a completion
  notification; do not infer its setup/reproduction outcome.

## Setup released and first numerical evidence

The preceding setup hold is **superseded by the experiment-1 completion
notification** (`ses_eab758964ffeJCzR58tBi155La`): setup finished in 759 seconds.
The three-process/one-thread-per-process budget now applies to numerical runs.

- Equilibrium commit `4b6fe24` integrated as `024838a`. Original package v0.1.4,
  source `534ee7c`, loaded and executed a tiny 8-iteration `train!` run, according
  to the builder's committed artifacts. This is not reproduction of the paper's
  large-scale experiments. KrylovKit 0.7.1 was pinned within declared compatibility
  to resolve the 0.8 `RecursiveVec` API change; retrying the drifted resolution is
  not useful. No source patch or credential change was needed.
- Parent independently reran standard-library probe tests: **101/101 passed**,
  31.39 seconds process wall time (23.1 seconds inside the testset), exit 0.
  Evidence: `results/validation/initial-equilibria.txt`.
- Measure-normalization and distance-power differences now have numerical probes;
  scientific interpretation is undergoing independent Astra review. Exact-match
  NaN gradients and empty-prediction behavior are explicitly recorded negatives.
- The equilibrium builder is now testing direct finite-difference consistency of
  the released measure gradient and the normalized main-text measure gradient,
  and adding isolated output-directory support. This gate is not yet passed.
- Cycle and PDE builders are still correcting the static findings. Their suites
  may run as soon as those commits return; no additional user approval is needed.

## Known uncertainties / do not repeat as established facts

- Paper and released code appear to use different measure normalizations and
  target-distance powers; their behavioral consequences have not been tested.
- Language performance and agent productivity rankings are unmeasured.
- Finding no integrated successor in the research pass is not evidence of novelty.
- Persistent homology alone neither identifies a unique branch nor guarantees
  symmetry invariance; descriptors must be tested for their stated invariances.
