# Pseudo-arclength continuation in z = (c, r) for the even-subspace problem.
#
# Corrector system at predicted point z, with t the unit tangent at the
# previously accepted point z_prev and ds the step:
#   [ dR/dc    dR/dr ] [dc]      [ R(c, r)              ]
#   [   t'          ] [dr] = -  [ t'(z - z_prev) - ds  ]
# The border row makes the system regular at simple folds (saddle-nodes), so
# branches are continued THROUGH folds, including unstable segments.
# Tangents: solve the same bordered matrix against the last unit vector,
# orientation-fixed against the previous tangent (or the requested r-direction
# for the first step). Folds are localized post hoc with a three-point parabola
# fit; the estimate is step-size limited (documented limitation).

struct BranchPoint
    s::Float64        # accumulated arclength
    r::Float64
    c::Vector{Float64}
    t::Vector{Float64}   # unit tangent at this point (in (c, r))
    rms::Float64
    newton_iters::Int
    ds::Float64         # step used to reach this point (0.0 for the start point)
end

function bordered_matrix(disc::Discretization, cfg::SHConfig, c::Vector{Float64},
                         r::Float64, trow::Vector{Float64})
    Nc = disc.Nc
    A = Matrix{Float64}(undef, Nc + 1, Nc + 1)
    A[1:Nc, 1:Nc] = jacobian_even(disc, cfg, c, r)
    A[1:Nc, Nc + 1] = residual_r_even(disc, c)
    A[Nc + 1, 1:Nc + 1] = trow
    return A
end

# Unit tangent at (c, r). `tprev === nothing` seeds the border with e_r (dr = 1)
# and `sign_r` (if nonzero orients the tangent so dr has that sign); otherwise
# the border is the previous tangent and orientation follows tprev.
function branch_tangent(disc::Discretization, cfg::SHConfig, c::Vector{Float64},
                        r::Float64, tprev, sign_r::Float64)
    Nc = disc.Nc
    if tprev === nothing
        trow = zeros(Nc + 1)
        trow[Nc + 1] = 1.0
    else
        trow = tprev
    end
    A = bordered_matrix(disc, cfg, c, r, trow)
    rhs = zeros(Nc + 1)
    rhs[Nc + 1] = 1.0
    t = try
        A \ rhs
    catch
        return nothing
    end
    (all(isfinite, t) && norm(t) > 0.0) || return nothing
    if tprev !== nothing && dot(t, tprev) < 0.0
        t = -t
    end
    if tprev === nothing && sign_r != 0.0 && t[Nc + 1] * sign_r < 0.0
        t = -t
    end
    return t ./ norm(t)
end

function correct_step(disc::Discretization, cfg::SHConfig, z_prev::Vector{Float64},
                      t::Vector{Float64}, ds::Float64)
    Nc = disc.Nc
    z = z_prev .+ (ds .* t)
    c = z[1:Nc]
    r = z[Nc + 1]
    for it in 1:cfg.max_cont_newton
        R = residual_even(disc, cfg, c, r)
        arc = dot(t, z .- z_prev) - ds
        nrm = sqrt(norm(R)^2 + arc^2)
        if nrm <= cfg.tol_residual
            return (ok=true, z=z, iters=it, residual=nrm)
        end
        A = bordered_matrix(disc, cfg, c, r, t)
        rhs = vcat(-R, [-arc])
        dz = try
            A \ rhs
        catch
            return (ok=false, z=z, iters=it, residual=nrm)
        end
        if !all(isfinite, dz)
            return (ok=false, z=z, iters=it, residual=nrm)
        end
        alpha = 1.0
        ok_step = false
        for _ in 1:3
            z_try = z .+ (alpha .* dz)
            R_try = residual_even(disc, cfg, z_try[1:Nc], z_try[Nc + 1])
            if all(isfinite, R_try)
                arc_try = dot(t, z_try .- z_prev) - ds
                nrm_try = sqrt(norm(R_try)^2 + arc_try^2)
                if nrm_try < nrm
                    z = z_try
                    c = z[1:Nc]
                    r = z[Nc + 1]
                    ok_step = true
                    break
                end
            end
            alpha *= 0.5
        end
        if !ok_step
            return (ok=false, z=z, iters=it, residual=nrm)
        end
    end
    R = residual_even(disc, cfg, z[1:Nc], z[Nc + 1])
    arc = dot(t, z .- z_prev) - ds
    return (ok=false, z=z, iters=cfg.max_cont_newton, residual=sqrt(norm(R)^2 + arc^2))
end

"""
    continue_branch(disc, cfg, c0, r0, dir_sign; fl, tag, ...)

Pseudo-arclength continuation. `dir_sign = +1` starts toward larger r, `-1`
toward smaller r. Returns `(points, term, folds, dir_sign)`. Every failed
corrector attempt is recorded into the FailLog `fl` (if given) with the current
step and halved ds, and hard termination reasons are returned in `term`:
"r_stop_hi", "r_stop_lo", "trivial_approach" (guard: never accepts states near
the trivial branch), "amplitude_cap_or_nonfinite", "tangent_fail", "step_min",
"step_budget".
"""
function continue_branch(disc::Discretization, cfg::SHConfig, c0::Vector{Float64},
                         r0::Float64, dir_sign::Int;
                         fl=nothing, tag::String="",
                         ds0::Float64=cfg.ds0, ds_max::Float64=cfg.ds_max,
                         steps_max::Int=cfg.steps_max)
    Nc = disc.Nc
    z0 = vcat(copy(c0), [r0])
    t = branch_tangent(disc, cfg, c0, r0, nothing, float(dir_sign))
    if t === nothing
        return (points=BranchPoint[], term="tangent_fail",
                folds=locate_folds(BranchPoint[]), dir_sign=dir_sign)
    end
    pts = BranchPoint[]
    push!(pts, BranchPoint(0.0, r0, copy(c0), t, rms_of_c(disc, c0), 0, 0.0))
    z_prev = z0
    s = 0.0
    ds = ds0
    term = "step_budget"
    for step in 1:steps_max
        accepted = false
        z_acc = Vector{Float64}()
        iters_acc = 0
        for attempt in 1:12
            corr = correct_step(disc, cfg, z_prev, t, ds)
            if corr.ok
                accepted = true
                z_acc = corr.z
                iters_acc = corr.iters
                break
            end
            if fl !== nothing
                record!(fl, Any["continuation", tag, string(dir_sign), z_prev[Nc + 1],
                                disc.N, step, attempt, ds, corr.residual, "corrector_fail"])
            end
            ds = ds / 2.0
            if ds < cfg.ds_min
                break
            end
        end
        if !accepted
            term = "step_min"
            break
        end
        c_new = z_acc[1:Nc]
        r_new = z_acc[Nc + 1]
        rms_new = rms_of_c(disc, c_new)
        if !all(isfinite, z_acc) || rms_new > cfg.rms_cap
            term = "amplitude_cap_or_nonfinite"
            break
        end
        if rms_new < cfg.rms_trivial
            # never accept or continue through the trivial branch
            term = "trivial_approach"
            break
        end
        s += ds
        t_new = branch_tangent(disc, cfg, c_new, r_new, t, 0.0)
        if t_new === nothing
            push!(pts, BranchPoint(s, r_new, copy(c_new), t, rms_new, iters_acc, ds))
            term = "tangent_fail"
            break
        end
        push!(pts, BranchPoint(s, r_new, copy(c_new), t_new, rms_new, iters_acc, ds))
        # step-size adaptation
        if iters_acc <= 4
            ds = min(ds * 1.25, ds_max)
        elseif iters_acc >= 10
            ds = max(ds * 0.6, cfg.ds_min)
        end
        t = t_new
        z_prev = z_acc
        # stopping criteria (tangent sign prevents premature stops at folds)
        if r_new >= cfg.r_stop_hi && t[Nc + 1] > 0.0
            term = "r_stop_hi"
            break
        end
        if r_new <= cfg.r_stop_lo && t[Nc + 1] < 0.0
            term = "r_stop_lo"
            break
        end
    end
    folds = locate_folds(pts)
    return (points=pts, term=term, folds=folds, dir_sign=dir_sign)
end

# Vertex (s*, r*) of the parabola through three points (divided differences).
function parabola_vertex(s1::Float64, r1::Float64, s2::Float64, r2::Float64,
                         s3::Float64, r3::Float64)
    f12 = (r2 - r1) / (s2 - s1)
    f23 = (r3 - r2) / (s3 - s2)
    f123 = (f23 - f12) / (s3 - s1)
    if f123 == 0.0
        return (s2, r2)
    end
    sv = 0.5 * (s1 + s2) - f12 / (2.0 * f123)
    rv = r1 + f12 * (sv - s1) + f123 * (sv - s1) * (sv - s2)
    return (sv, rv)
end

# Local extrema of r(s) along accepted branch points (fold candidates).
# Returns (s*, r*, "min"|"max") tuples; estimates are step-size limited.
function locate_folds(pts::Vector{BranchPoint})
    out = Vector{Tuple{Float64,Float64,String}}()
    n = length(pts)
    if n < 3
        return out
    end
    for i in 2:(n - 1)
        p1 = pts[i - 1]
        p2 = pts[i]
        p3 = pts[i + 1]
        if p2.s <= p1.s || p3.s <= p2.s
            continue
        end
        if p2.r < p1.r && p2.r < p3.r
            sv, rv = parabola_vertex(p1.s, p1.r, p2.s, p2.r, p3.s, p3.r)
            push!(out, (sv, rv, "min"))
        elseif p2.r > p1.r && p2.r > p3.r
            sv, rv = parabola_vertex(p1.s, p1.r, p2.s, p2.r, p3.s, p3.r)
            push!(out, (sv, rv, "max"))
        end
    end
    return out
end
