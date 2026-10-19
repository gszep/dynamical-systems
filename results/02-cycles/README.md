# results/02-cycles — artifact manifest

Artifacts for experiment 02, written by `experiments/02-cycles/main.jl`.
The committed set below was generated 2026-10-20 on this host (single
process, one Julia/BLAS/OMP thread, pinned Julia 1.10.11) and corresponds to
the measured evidence recorded in `experiments/02-cycles/README.md`:
10/10 scenarios within their declared acceptance sets, exit 0, total 27.5 s
(setup 0.34 s). Per-scenario timings, tolerances, seeds, host and git head
are recorded in `summary.toml`.

| File | Contents |
|---|---|
| `summary.toml` | global record: git head, Julia version, host, threads, seed, args, queue/setup/exec/total timings, tolerances, `[audit]` block (all_accepted, failed_acceptance, crashed) and one sub-dict per scenario with statuses, `accepted` flags and key numbers |
| `sl_orbit_N256.csv` | solved Stuart–Landau orbit sample (`t, x1, x2`) |
| `sl_convergence.csv` | N-sweep with errors vs the analytic reference and observed orders |
| `sl_amplitude_law.csv` | Hopf-continuation amplitude law, multiplier gap, classification, degeneracy flags |
| `sl_negative_mu.csv` | negative-case outcomes for `mu <= 0` (honest failures) |
| `sl_inference_trace.csv` | Gauss–Newton iteration trace (mu, omega, residual) |
| `vdp_orbit_N256.csv` | solved Van der Pol orbit sample |
| `vdp_convergence.csv` | Van der Pol N-sweep (period, phase point, multiplier, Richardson inputs) |

Regenerate with:

```sh
JULIA=/Users/gszep/Documents/repos/dynamical-systems-research-20261019/.tools/julia-1.10.11/bin/julia
$JULIA -t 1 experiments/02-cycles/main.jl            # writes this directory
$JULIA -t 1 experiments/02-cycles/main.jl --out-dir <dir>   # or any directory
```

Exit code is 1 on any scenario crash or any scenario outcome outside its
declared acceptance set (see the `[audit]` block); expected negative outcomes
(guard rejections, honest non-convergence, model-mismatch flags) are accepted.
Re-running regenerates byte-equivalent numeric content for the deterministic
scenarios (fixed seeds; see `summary.toml`).
