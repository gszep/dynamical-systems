# Physical and numerical parameters for the 1D subcritical Swift-Hohenberg experiment.
#
# Equation:  u_t = r*u - (1 + d_xx)^2 u + b*u^3 - u^5,   x in [0, L), periodic.
# With b > 0 the cubic is destabilizing and the quintic saturates: the pattern
# branch bifurcates subcritically at r = 0, has a fold at r* < 0, and coexists
# with the trivial state in (r*, 0). Single-mode Galerkin predictions used as
# independent expected values in the tests (b = 1):
#   amplitude equation:  0 = r + (3/4) b A^2 - (5/8) A^4
#   fold at r* = -9 b^2 / 40 = -0.225, fold amplitude A* = sqrt(3/5) ~ 0.775.

Base.@kwdef struct SHConfig
    # Physical parameters
    L::Float64 = 8.0 * pi                # periodic domain length (k = 1 pattern => 4 wavelengths)
    b::Float64 = 1.0                     # cubic coefficient; b > 0 gives the subcritical case

    # Study parameters
    r_fix::Float64 = -0.10               # fixed physical parameter for the mesh study
    r_disc::Vector{Float64} = [-0.10, -0.20]  # discovery sweep values (bistable window)

    # Newton (even-subspace residual R(c; r) = S' F(S c; r), convergence judged on ||R||_2)
    tol_residual::Float64 = 1.0e-9
    max_newton::Int = 30
    max_backtrack::Int = 8

    # Pseudo-arclength continuation
    ds0::Float64 = 0.03
    ds_min::Float64 = 1.0e-4
    ds_max::Float64 = 0.05
    steps_max::Int = 250
    max_cont_newton::Int = 12
    r_stop_hi::Float64 = 0.10
    r_stop_lo::Float64 = -0.45
    rms_trivial::Float64 = 0.02          # continuation stops when approaching the trivial branch
    rms_cap::Float64 = 3.0               # amplitude safety cap

    # Discovery
    rms_trivial_disc::Float64 = 1.0e-6   # solutions below this rms are the trivial state
    dedup_tol::Float64 = 0.05            # relative distance for duplicate detection (up to sign)

    # Spectrum reporting
    eig_pos_threshold::Float64 = 1.0e-6  # eigenvalues above this count as unstable modes

    # Fixed seed table (recorded in every run's outputs)
    seedA_cos::Vector{Float64} = [0.35, 0.70, 1.20]
    seedK_cos::Vector{Float64} = [1.0, 2.0]
    seedA_loc::Vector{Float64} = [0.8, 1.3]
    seedW_loc::Vector{Float64} = [2.0, 4.0]
    seedC_loc::Vector{Float64} = [0.0, 4.0 * pi]   # bump centers: 0 and L/2 (both even)
    random_seed_ids::Vector{UInt64} = UInt64[101, 203, 307, 401, 503, 601, 701, 803, 907, 1009]
    random_band_modes::Int = 6           # cos(k_n x), k_n = 2*pi*n/L, n = 1..6
    random_targets::Vector{Float64} = [0.4, 0.8, 1.2]  # target rms, round-robin over seeds
end
