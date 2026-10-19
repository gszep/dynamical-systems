# Experiment 03: 1D periodic Swift-Hohenberg stationary patterns.
#
# Self-contained module: Julia stdlib only (LinearAlgebra, Printf, TOML).
# Scripts include this file directly (include + `using .SwiftHohenbergPatterns`),
# so no package installation or precompilation of the project is required.

module SwiftHohenbergPatterns

using LinearAlgebra
using Printf
using TOML

export SHConfig, Discretization, TinyRNG,
       refidx, reflect_values, project_even, to_coeffs, to_values,
       shift_values, shift_grid_values, derivative_values, evaluate_coeffs_on,
       rms_of_values, rms_of_c,
       residual_full, residual_even, residual_full_fd, residual_r_even,
       jacobian_full, jacobian_even,
       fd_laplacian, fd_quadratic_op, fd_laplacian_eigvals,
       newton_even, continue_branch, BranchPoint, locate_folds, parabola_vertex,
       spectrum_even, spectrum_full,
       descriptors, descriptor_vector, fourier_powers,
       seed_table, seed_values, SeedSpec, run_discovery,
       classify_solution, is_duplicate, dedup_distance, rel_dist,
       FailLog, record!, write_csv, dump_toml, toml_safe, fmt_val,
       run_smoke, pick_continuation_start

include("config.jl")
include("rng.jl")
include("transforms.jl")
include("operators.jl")
include("newton.jl")
include("continuation.jl")
include("descriptors.jl")
include("spectrum.jl")
include("io.jl")
include("discovery.jl")
include("experiment.jl")

end # module
