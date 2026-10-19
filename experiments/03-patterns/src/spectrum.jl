# Linearized spectra about stationary states, with an explicit statement of
# which modes are present or removed and of the normalization.
#
# The linearization about a stationary state U is
#   J = r I - (1 + d_xx)^2 + diag(3 b U^2 - 5 U^4).
# spectrum_full  assembles J on the whole N-dimensional grid space.
# spectrum_even reports the PHYSICAL rates of the reflection-symmetric
# restriction: the cosine basis S is not orthonormal (S' S = G, diagonal, with
# entries N or N/2), so the c-coordinate matrix S' J S must be mass-normalized.
# We compute the eigenvalues of  G^{-1/2} (S' J S) G^{-1/2}  (equivalently the
# generalized problem (S' J S) v = lambda (S' S) v, i.e. the Rayleigh-Ritz
# rates of the restriction). The c-coordinate matrix alone would rescale modes
# by diag(G): at the zero state the critical cosine mode would read
# (N/2) r instead of r - tested as the negative case on N = 32/64/128, while
# the mass-normalized rates equal the analytic r - (1-k^2)^2 with the critical
# mode exactly at r.
#
# Scope of claims (also enforced by the tests):
# * The full-space spectrum of any nontrivial periodic stationary state
#   contains a ~zero eigenvalue whose eigenvector is the translation phase
#   direction U' (relative-equilibrium neutral mode). The restricted spectrum
#   does not contain it: U' is odd, so it leaves the reflection-symmetric
#   subspace (the restriction is a phase condition, not a translation
#   quotient). The tests check ||J U'|| ~ 0 while the restricted spectrum has
#   no near-zero rate.
# * An unstable restricted rate implies full-space instability (the
#   reflection-symmetric subspace is invariant). The converse is NOT claimed:
#   stability inside the subspace says nothing about sine-sector
#   (phase/sideband) perturbations, and states with no even translate are
#   outside this experiment's scope entirely. No full-space
#   asymptotic-stability claim is made anywhere.
# * Counts of "unstable" modes use the threshold cfg.eig_pos_threshold and are
#   borderline near folds; the raw extreme rates are recorded too.

function spectrum_even(disc::Discretization, cfg::SHConfig, c::Vector{Float64}, r::Float64)
    Jc = jacobian_even(disc, cfg, c, r)   # S' J S in c-coordinates
    W = Diagonal(disc.Ghalfinv)           # G^{-1/2}, G = S' S (diagonal)
    A = W * Jc * W                        # mass-normalized restriction
    vals = eigvals(Symmetric(A))
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
