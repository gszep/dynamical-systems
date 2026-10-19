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
- Next check: install a checksum-verified, repository-local Julia runtime, then
  implement independent baselines in three isolated worktrees.
- No jobs or recurring continuation mechanism have yet been established.
- `artemis` has not been accessed. No machine or account configuration changes are
  needed for the initial CPU baselines.

## Known uncertainties / do not repeat as established facts

- Paper and released code appear to use different measure normalizations and
  target-distance powers; their behavioral consequences have not been tested.
- Language performance and agent productivity rankings are unmeasured.
- Finding no integrated successor in the research pass is not evidence of novelty.
- Persistent homology alone neither identifies a unique branch nor guarantees
  symmetry invariance; descriptors must be tested for their stated invariances.
