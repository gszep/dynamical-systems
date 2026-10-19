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

## Initial launch state (superseded by checkpoints below)

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

## Equilibrium review qualifications (pending corrections)

The 101 passing tests are real but insufficient to establish all written claims.
The independent review identified the following; the existing equilibrium builder
has been given these findings without restarting setup:

- FHN parameter-Jacobian entries were reshaped in the wrong order; current
  value-only Hopf probes do not cover this derivative. Add all-column FD checks.
- Initial normalization artifacts compare an **unwindowed surrogate**, not the
  exact released erf-windowed objective. Their normalization-only comparison is
  retained, but actual-package count-term sign claims must be qualified/retested.
- The singleton squared-distance objective has derivative zero at exact match;
  its NaN factored gradient is a removable implementation defect, not an objective
  singularity. The unsquared cusp and multi-prediction cases are distinct.
- Paper determinant notation is signed (product of eigenvalues); the proposed
  signed-versus-absolute paper/code discrepancy was false and must be removed.
- Package checks must propagate training/final-loss/dimension failures, not print
  `ALL OK` after catching them. The recorded successful tiny run is still evidence,
  but the command's previous exit status alone is not a reliable regression gate.
- The vendored snapshot must retain its upstream MIT license notice.

Next gradient comparison must include the same windowed integrand and moving
physical-window endpoints, refine quadrature and FD steps, and separately test
the quotient derivative for normalization. A count-matched loss disables the
measure term and cannot establish its derivative correctness.

## Latest reconciliation: first overnight wake-up

Heartbeat `17a3bc89` successfully delivered into the canonical conversation.
Existing children own the three correction/gradient tasks; no duplicate builders
or repeated setup were launched. No additional completion notification or corrected
commit has arrived yet. The next action remains integration and execution of the
returned fixes, not running known-defective earlier implementations.

The setup hold is released, and the only remaining waits are for the existing
workers' focused corrections. No user-only approval blocker exists. Keep the
bounded overnight heartbeat active.

## First cycle/PDE execution round

- Corrected PDE commit `6646621` integrated as `e0c9cec`; first parent suite
  returned **194 pass, 1 fail, 1 error**, exit 1. Smoke returned exit 0, but this
  does not override failing validation. Evidence: `results/validation/initial-patterns.toml`.
- The full-space translation derivative residual was `6.24e-4`, not `<1e-9`;
  determine whether undealiased collocation breaks continuous translation symmetry
  at this resolution or the implementation is wrong. Measure convergence rather
  than blindly weakening the assertion. The other error is a test argument type.
- The PDE smoke found two periodic patterns at each sampled parameter, one fold
  near `r=-0.22501`, and zero new localized patterns; 41 failed seeds are recorded.
  This is neither a localized-state atlas nor exhaustive discovery.
- Existing PDE builder is diagnosing these runtime findings and may now execute
  one-thread numerical checks. No package installation is needed.
- Corrected cycle commit `ed692d5` integrated as `03c5d59`; parent run completed:
  **199/200 tests pass**, one unconditional monotonic-error assertion fails.
  The period-sensitivity, implicit-gradient and final-iterate-consistency
  regressions pass. Smoke completed ten scenarios, exit 0; see
  `results/validation/initial-cycles.toml` for measured errors and limitations.
- Cycle baseline recovered synthetic parameters (error `8.4e-11`, cross-resolution
  error `3.9e-9`); the failing convergence assertion remains a gate. Existing builder
  now diagnoses the floating-point/rotational-symmetry floor, retains meaningful
  order checks, and makes smoke status enforce required positive outcomes.
- The equilibrium builder continues the reviewed windowed-gradient investigation.
  Aggregate permitted numerical processes remain three, one per active lane.

## Equilibrium gradient gate escalated

Revision `1038d09` integrated as `9269d91`: six earlier review corrections,
184-test builder result, and direct package-gradient experiments. Parent rerun of
the stdlib tests/probes is active under `sh_154d2d82e001EFiWQvy9IRHdFp`; do not
duplicate it. These independent probes do not themselves certify package gradients.

The reviewer rejected the package-gradient "pass" interpretation: absolute
thresholds 0.05/0.10 exceed the smaller component (~0.01491); its refinement errors
are nonmonotone and finite differences lack a demonstrated plateau. O(ds) and
"correct continuum derivative" claims are **not established**. Full-minus-truncated
derivatives also conflate tail and endpoint terms. Required FD exceptions and
nonfinite results need stricter failure propagation.

The same equilibrium child session is now escalated to `openai/gpt-6-astra` for
this narrow task, rather than another blind retry. It is initially code-only while
the parent rerun occupies its numerical slot. Release that restriction on the
parent job's completion; retain at most three aggregate numerical processes.
Its instructions require component-wise FD plateaus, finer ds checks, explicit
boundary terms, real injected failure paths, and an honest inconclusive/failure
status when the derivative evidence does not support acceptance.

## Known uncertainties / do not repeat as established facts

- Paper and released code use different measure normalizations and target-distance
  powers. Controlled surrogate effects have been measured, but actual windowed
  measure-gradient consistency and reviewed interpretation are still pending.
- Language performance and agent productivity rankings are unmeasured.
- Finding no integrated successor in the research pass is not evidence of novelty.
- Persistent homology alone neither identifies a unique branch nor guarantees
  symmetry invariance; descriptors must be tested for their stated invariances.
