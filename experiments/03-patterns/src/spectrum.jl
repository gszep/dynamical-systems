# Linearized spectra about stationary states, with an explicit statement of
# which modes are present or removed.
#
# The linearization about a stationary state U is
#   J = r I - (1 + d_xx)^2 + diag(3 b U^2 - 5 U^4).
# spectrum_full  assembles J on the whole N-dimensional grid space.
# spectrum_even restricts J to the reflection-invariant (cosine) subspace.
#
# Scope of claims (also enforced by the tests):
# * The full-space spectrum of any nontrivial periodic stationary state
#   contains a ~zero eigenvalue whose eigenvector is the translation phase
#   direction U' (relative-equilibrium neutral mode). The even-subspace
#   spectrum removes it BY CONSTRUCTION: the tests check that
#   ||J U'|| ~ 0 while the even-restricted J has no near-zero eigenvalue.
# * An unstable eigenvalue in the even subspace implies full-space instability
#   (the even subspace is invariant). The converse is NOT claimed: stability
#   inside the even subspace says nothing about sine-sector (phase/sideband)
#   perturbations. No full-space asymptotic-stability claim is made anywhere.
# * Counts of "unstable" modes use the threshold cfg.eig_pos_threshold and are
#   borderline near folds; the raw extreme eigenvalues are recorded too.

function spectrum_even(disc::Discretization, cfg::SHConfig, c::Vector{Float64}, r::Float64)
    J = jacobian_even(disc, cfg, c, r)
    vals = eigvals(Symmetric(J))
    return (lambda_max=vals[end], lambda_min=vals[1],
            n_pos=count(v -> v > cfg.eig_pos_threshold, vals),
            min_abs=minimum(abs, vals))
end

function spectrum_full(disc::Discretization, cfg::SHConfig, u::Vector{Float64}, r::Float64)
    J = jacobian_full(disc, cfg, u, r)
    vals = eigvals(Symmetric(J))
    return (lambda_max=vals[end], lambda_min=vals[1],
            n_pos=count(v -> v > cfg.eig_pos_threshold, vals),
            min_abs=minimum(abs, vals))
end

function empty_spectrum_summary()
    nan = Float64(NaN)
    return (lambda_max=nan, lambda_min=nan, n_pos=nan, min_abs=nan)
end
