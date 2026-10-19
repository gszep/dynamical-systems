# results/02-cycles — artifact manifest

Artifacts for experiment 02 are written here by `experiments/02-cycles/main.jl`.
Nothing has been generated yet: the code was authored while package setup was
running concurrently and has not been executed. After the first run this
directory should contain:

| File | Contents |
|---|---|
| `summary.toml` | global record: git head, Julia version, host, threads, seed, args, queue/setup/exec/total timings, tolerances, and one sub-dict per scenario with statuses and key numbers |
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

The commit policy for generated artifacts (whether `summary.toml` and the CSVs
are committed after runs) belongs to the parent integrator; per `AGENTS.md`,
small reproducible result artifacts are expected to be recorded with their
commands, seeds, tolerances, failure counts and hardware, all of which
`main.jl` captures in `summary.toml`.
