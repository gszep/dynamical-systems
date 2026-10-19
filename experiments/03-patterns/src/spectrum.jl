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
#   contains a ~zero eigenvalue associated with the translation phase
#   direction U' (relative-equilibrium neutral mode). The restricted spectrum
#   does not contain it: U' is odd, so it leaves the reflection-symmetric
#   subspace (the restriction is a phase condition, not a translation
#   quotient). The tests verify that the full-space spectrum has exactly one
#   near-zero eigenvalue while the restricted spectrum has none.
# * U' is an APPROXIMATE null vector of the discrete collocation Jacobian,
#   not an exact one: the undealiased collocation nonlinearity preserves only
#   GRID shifts (continuous translations are broken by discretization), so the
#   relative residual ||J U'||/||U'|| is a resolution-dependent aliasing
#   quantity - dominated by the first nonlinear harmonic past the Nyquist
#   limit (measured ~6e-4 at N=64, O(1) at N=32, roundoff-scale at N >= 128).
#   phase_mode_defect exposes it; the tests verify the Jacobian/derivative
#   assembly against independent operator applications and check the
#   resolution behavior instead of asserting a false exact null.
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

"""
    phase_mode_defect(disc, cfg, c, r) -> Float64

Relative residual of the translation phase direction in the full-space
collocation linearization: ||J U'|| / ||U'||, with U' the spectral derivative
of the stationary state U = S c and J = r I - (1+d_xx)^2 + diag(3bU^2 - 5U^4).

For the CONTINUOUS problem U' is an exact null vector. The undealiased
collocation nonlinearity preserves only grid shifts, so this defect is a
resolution-dependent aliasing quantity - roughly 2*(Nyquist wavenumber) times
the amplitude of the first nonlinear harmonic past Nyquist - not an
implementation error. Measured behavior (see the tests and
patterns_summary.toml): O(1) at N=32 (the quintic's k=5 output aliases),
~6e-4 at N=64 (its k=9 output aliases), roundoff-scale for N >= 128 (all
significant harmonics fit). The full-space spectrum still contains exactly
one near-zero eigenvalue; U' is its approximate eigenvector with the defect
reported here.
"""
function phase_mode_defect(disc::Discretization, cfg::SHConfig, c::Vector{Float64},
                          r::Float64)
    u = disc.S * c
    up = derivative_values(disc, u)
    nup = norm(up)
    if !(nup > 0.0) || !all(isfinite, up)
        return NaN
    end
    J = jacobian_full(disc, cfg, u, r)
    return norm(J * up) / nup
end
