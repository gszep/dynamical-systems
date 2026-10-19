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

## Known uncertainties / do not repeat as established facts

- Paper and released code appear to use different measure normalizations and
  target-distance powers; their behavioral consequences have not been tested.
- Language performance and agent productivity rankings are unmeasured.
- Finding no integrated successor in the research pass is not evidence of novelty.
- Persistent homology alone neither identifies a unique branch nor guarantees
  symmetry invariance; descriptors must be tested for their stated invariances.
