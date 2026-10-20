# Smoke run for experiment 03: 1D periodic Swift-Hohenberg stationary patterns.
#
# Usage (pinned runtime; run from the repository root after package setup):
#   JULIA_NUM_THREADS=1 \
#     /Users/gszep/Documents/repos/dynamical-systems-research-20261019/.tools/julia-1.10.11/bin/julia \
#     --project=experiments/03-patterns experiments/03-patterns/scripts/main.jl
#
# Single process, one Julia thread; the script pins BLAS to one thread.
# Writes artifacts to <repo>/results/03-patterns/. Budget: dense grids
# N <= 128, all dense linear algebra is tiny; expected wall time well under
# five minutes including module load and JIT compilation.

const PKG_DIR = dirname(@__DIR__)
const REPO_ROOT = dirname(dirname(PKG_DIR))
function output_directory(args)
    isempty(args) && return joinpath(REPO_ROOT, "results", "03-patterns")
    length(args) == 2 && args[1] == "--out-dir" && return abspath(args[2])
    error("usage: main.jl [--out-dir DIRECTORY]")
end
const RESULTS_DIR = output_directory(ARGS)

t_setup0 = time_ns()
include(joinpath(PKG_DIR, "src", "SwiftHohenbergPatterns.jl"))
using .SwiftHohenbergPatterns
using LinearAlgebra
LinearAlgebra.BLAS.set_num_threads(1)
const SETUP_SECONDS = (time_ns() - t_setup0) / 1.0e9

const CFG = SHConfig()
mkpath(RESULTS_DIR)
summary = run_smoke(CFG; results_dir=RESULTS_DIR, setup_seconds=SETUP_SECONDS)

println("experiment 03 smoke run complete")
println("results directory: ", RESULTS_DIR)
println("total wall seconds: ", summary["timing"]["total_seconds"])
println("setup / discovery / continuation / mesh / io seconds: ",
        summary["timing"]["setup_seconds"], " / ",
        summary["timing"]["discovery_seconds"], " / ",
        summary["timing"]["continuation_seconds"], " / ",
        summary["timing"]["mesh_seconds"], " / ",
        summary["timing"]["io_seconds"])
println("logged failure rows: ", summary["failures"]["n_logged"])
println("distinct discoveries: r_-0p1 new = ", summary["discovery"]["r_-0p1"]["n_new"],
        ", r_-0p2 new = ", summary["discovery"]["r_-0p2"]["n_new"])

if summary["acceptance"]["ok"]
    println("acceptance gates: PASS (positive discoveries per sweep, usable continuation with fold, required mesh solves)")
else
    println("acceptance gates: FAIL (see [acceptance] in patterns_summary.toml and failures.csv)")
    exit(1)
end
