# Damped Newton for R(c; r) = 0 in the reflection-symmetric (even) subspace,
# with optional classical (root-removing) deflation away from known solutions.
#
# Deflation (discovery only): with known solutions c_i, the deflated residual is
#   R~(c) = M(c) R(c),   M(c) = prod_i 1 / ||c - c_i||^2,
# (defined up to a positive scalar, which is irrelevant to Newton), so R~ has
# a POLE at each c_i: known roots are removed from the deflated system and
# Newton cannot re-converge to them, while all other roots are preserved (M is
# finite and nonzero away from the c_i). With
#   w(c) = grad log M(c) = -sum_i 2 (c - c_i) / ||c - c_i||^2,
# the Newton direction solves the equivalent unscaled system
#   (J + R w') delta = -R,
# because M > 0 is a scalar factor. The distance ||c - c_i||^2 is floored at
# 1e-30 to keep the linear algebra finite.
#
# Known-root exclusion: the TRUE (undeflated) residual is always used for the
# descent test and for convergence, so its zeros at known roots could
# otherwise masquerade as successes (an exact known seed would converge
# instantly). With `known` nonempty, exact and near known roots are therefore
# excluded explicitly: if the iteration reaches a point whose sign-aware
# relative distance to some c_i is below cfg.dedup_tol, the solver returns
# `converged = false` with `reason = "known_root_excluded"` instead of a
# false success. Any reported success is then a genuinely new root.

function _deflation_w(c::Vector{Float64}, known::Vector{Vector{Float64}})
    if isempty(known)
        return zeros(length(c))
    end
    w = zeros(length(c))
    for i in 1:length(known)
        d = c .- known[i]
        dd = dot(d, d)
        if dd < 1.0e-30
            dd = 1.0e-30
        end
        w .-= (2.0 ./ dd) .* d
    end
    return w
end

# Sign-aware near-known test (dedup_distance is defined in discovery.jl).
function _near_known(c::Vector{Float64}, known::Vector{Vector{Float64}}, tol::Float64)
    return any(ck -> dedup_distance(c, ck) < tol, known)
end

"""
    newton_even(disc, cfg, c0, r; known = [])

Damped Newton on the even-subspace residual R(c; r) = S' F(S c; r), in
c-coordinates (Newton iterates in value space are coordinate-invariant). If
`known` is nonempty, classical deflation removes those roots, and exact or
near known roots (sign-aware relative distance below `cfg.dedup_tol`) are
excluded explicitly: converging to such a point returns
`(converged = false, reason = "known_root_excluded")`, never a false success.
Returns a NamedTuple `(converged, iters, c, residual, reason)`; `residual` is
the true (never deflated) ||R(c)||_2. Failures return with
`converged = false` and a diagnostic `reason` string, which callers record.
"""
function newton_even(disc::Discretization, cfg::SHConfig, c0::Vector{Float64}, r::Float64;
                     known::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    c = copy(c0)
    R = residual_even(disc, cfg, c, r)
    nrm = norm(R)
    if nrm <= cfg.tol_residual
        if !isempty(known) && _near_known(c, known, cfg.dedup_tol)
            return (converged=false, iters=0, c=c, residual=nrm,
                    reason="known_root_excluded")
        end
        return (converged=true, iters=0, c=c, residual=nrm, reason="converged")
    end
    for it in 1:cfg.max_newton
        J = jacobian_even(disc, cfg, c, r)
        if !isempty(known)
            w = _deflation_w(c, known)
            J = J .+ (R * w')
        end
        delta = try
            J \ (-R)
        catch err
            return (converged=false, iters=it, c=c, residual=nrm,
                    reason="linfail: " * sprint(showerror, err))
        end
        if !all(isfinite, delta)
            return (converged=false, iters=it, c=c, residual=nrm, reason="nonfinite_direction")
        end
        ok_step = false
        alpha = 1.0
        c_new = c
        R_new = R
        nrm_new = nrm
        for _ in 1:cfg.max_backtrack
            c_try = c .+ (alpha .* delta)
            if all(isfinite, c_try)
                R_try = residual_even(disc, cfg, c_try, r)
                if all(isfinite, R_try)
                    nrm_try = norm(R_try)
                    # sufficient decrease, or residual already at tolerance
                    if nrm_try <= (1.0 - 1.0e-4 * alpha) * nrm || nrm_try <= cfg.tol_residual
                        ok_step = true
                        c_new = c_try
                        R_new = R_try
                        nrm_new = nrm_try
                        break
                    end
                end
            end
            alpha *= 0.5
        end
        if !ok_step
            return (converged=false, iters=it, c=c, residual=nrm, reason="no_descent")
        end
        c = c_new
        R = R_new
        nrm = nrm_new
        if nrm <= cfg.tol_residual
            if !isempty(known) && _near_known(c, known, cfg.dedup_tol)
                return (converged=false, iters=it, c=c, residual=nrm,
                        reason="known_root_excluded")
            end
            return (converged=true, iters=it, c=c, residual=nrm, reason="converged")
        end
    end
    return (converged=false, iters=cfg.max_newton, c=c, residual=nrm, reason="maxiter")
end
