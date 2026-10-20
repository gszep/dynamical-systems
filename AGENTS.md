# Dynamical landscape research

This repository separates numerical exploration, validated numerics, and formal
proof. Never describe discovered branches as exhaustive or numerical residuals
as proofs. Keep paper-faithful and released-code objectives explicitly named.

## Experiments

- Pin runtime, package and upstream source revisions. Record commands, seeds,
  tolerances, failure counts, hardware and numerical output with each result.
- Keep each experiment under `experiments/01-equilibria`, `02-cycles`, or
  `03-patterns`; write small reproducible result artifacts under `results/`.
- Tests must exercise independent expected values, negative cases and convergence,
  not merely check that the implementation agrees with itself.
- Use isolated worktrees for concurrent builders. Do not change another builder's
  environment or result directory. Integrate reviewed commits, not shared edits.
- Initial local budget: at most three simultaneous numerical processes, each with
  one Julia thread and one BLAS/OpenMP thread; aggregate memory target <= 12 GiB.
  Package installation/precompilation has one owner and no concurrent experiment
  runs until setup finishes. Cap Julia precompile tasks at two.
- Record queue/setup/execution/total timings separately. Scientific stages may be
  implemented independently, but do not infer end-to-end success across an unmet gate.
- Long jobs require tool-supervised completion and bounded timeouts. No detached
  shells, blind retry loops, or changes to remote machines without recording scope.

## Validation

Run changed experiment tests and its reproducible smoke run. Run independent
experiment suites concurrently within the budget above. Documentation-only edits
require local-link/whitespace checks, not numerical suites. No CI configuration or
published benchmark currently exists; do not claim CI validation.

## Delivery

Use feature branches; do not push directly to the default branch. Update
`EXPERIMENTS.md` with achieved evidence, falsified assumptions, remaining blockers
and the next discriminating check. Never overwrite unrelated work.
