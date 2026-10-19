"""
    CycleBaselines

Self-contained (Julia stdlib only) baselines for experiment 02 (cycles):

* phase-fixed periodic-orbit shooting with an unknown period on a
  transversal section (Stuart-Landau and Van der Pol models),
* monodromy / Floquet multipliers from the discrete variational equations,
  which are the exact Jacobian of the RK4 discrete flow map,
* parameter sensitivities and implicit differentiation of cycle
  observables through the phase-fixed shooting solve,
* Hopf detection at equilibria, nonzero-orbit guards against spurious
  equilibrium roots, Farrell-style equilibrium deflation, and synthetic
  cycle parameter inference with model-mismatch detection.

Nothing in this module substitutes an analytic orbit into a solver. The
closed-form Stuart-Landau values (radius sqrt(mu), period 2*pi/omega,
multipliers {1, exp(-2*mu*T)}) are exposed only so that tests and reports
can compare against an independent reference.

Validation status at authoring time: this file was written while package
setup was running concurrently and has NOT been executed. See
experiments/02-cycles/README.md.
"""
module CycleBaselines

using LinearAlgebra
using Printf
import TOML

export StuartLandau, VanDerPol, rhs, jacobian, param_rhs, nparams, fd_jacobian,
       integrate_flow, FlowResult, Section, point_on_section, project_to_section,
       SolveStatus, SolveSuccess, GuardRejectedEquilibrium, NewtonNotConverged,
       StepOutOfBounds, DeflationSingular, CycleSolution, solve_cycle,
       find_equilibrium, detect_hopf, HopfInfo, FloquetData, classify_multipliers,
       analytic_stuart_landau, settle_guess, setup_cycle_scenario,
       InferenceConfig, ForwardResult, forward_solve, observables_from_solution,
       observables_jacobian_implicit, observables_jacobian_fd,
       InferenceResult, infer_parameters,
       write_csv, write_toml, solution_summary_dict

# -----------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------

abstract type AbstractCycleModel end

"""
    StuartLandau(mu, omega)

Supercritical Hopf / Stuart-Landau normal form on R^2:

    x1' =  mu*x1 - omega*x2 - (x1^2 + x2^2) * x1
    x2' =  omega*x1 + mu*x2 - (x1^2 + x2^2) * x2

Equilibrium at the origin with eigenvalues mu +- i*omega (Hopf at mu = 0,
transversality 1). For mu > 0 there is a unique (up to phase) attracting
periodic orbit of radius sqrt(mu) and period 2*pi/omega with Floquet
multipliers {1 (phase), exp(-2*mu*T) (radial)}. These closed forms are
derived in the README and used only as independent test references.
"""
struct StuartLandau <: AbstractCycleModel
    mu::Float64
    omega::Float64
end

"""
    VanDerPol(mu)

Van der Pol oscillator in Liénard coordinates:

    x1' = x2
    x2' = mu*(1 - x1^2)*x2 - x1

For mu > 0 the origin is an unstable focus surrounded by one attracting
limit cycle. No closed-form orbit, period or multiplier is known at
mu = O(1); this model exercises the pipeline without analytic shortcuts.
"""
struct VanDerPol <: AbstractCycleModel
    mu::Float64
end

dim(::Union{StuartLandau, VanDerPol}) = 2
nparams(::StuartLandau) = 2   # theta = (mu, omega)
nparams(::VanDerPol) = 1      # theta = (mu,)

function rhs(m::StuartLandau, x)
    r2 = x[1]*x[1] + x[2]*x[2]
    return [m.mu*x[1] - m.omega*x[2] - r2*x[1],
            m.omega*x[1] + m.mu*x[2] - r2*x[2]]
end

function jacobian(m::StuartLandau, x)
    r2 = x[1]*x[1] + x[2]*x[2]
    return [(m.mu - r2 - 2*x[1]*x[1])   (-m.omega - 2*x[1]*x[2]);
            (m.omega - 2*x[1]*x[2])     (m.mu - r2 - 2*x[2]*x[2])]
end

function param_rhs(m::StuartLandau, x)
    # columns: d f / d mu, d f / d omega
    return hcat([x[1], x[2]], [-x[2], x[1]])
end

function rhs(m::VanDerPol, x)
    return [x[2], m.mu*(1 - x[1]*x[1])*x[2] - x[1]]
end

function jacobian(m::VanDerPol, x)
    return [0.0                          1.0;
            (-2*m.mu*x[1]*x[2] - 1.0)    (m.mu*(1 - x[1]*x[1]))]
end

function param_rhs(m::VanDerPol, x)
    # column: d f / d mu
    return hcat([0.0, (1 - x[1]*x[1])*x[2]])
end

# -----------------------------------------------------------------------
# Finite differences (generic fallbacks; analytic paths are the defaults)
# -----------------------------------------------------------------------

function fd_jacobian(m::AbstractCycleModel, x; h::Real=1e-6)
    n = length(x)
    J = Matrix{Float64}(undef, n, n)
    for j in 1:n
        xp = Vector{Float64}(x); xp[j] += h
        xm = Vector{Float64}(x); xm[j] -= h
        J[:, j] = (rhs(m, xp) .- rhs(m, xm)) ./ (2*h)
    end
    return J
end

# -----------------------------------------------------------------------
# RK4 flow, variational equations, parameter sensitivities
# -----------------------------------------------------------------------

struct FlowResult
    xT::Vector{Float64}
    Psi::Matrix{Float64}                 # monodromy; exact Jacobian of the discrete RK4 map
    S::Union{Matrix{Float64}, Nothing}   # d x(T) / d theta; exact for the discrete map
    ts::Union{Vector{Float64}, Nothing}
    xs::Union{Matrix{Float64}, Nothing}
    N::Int
    h::Float64
end

# One RK4 step of the joint (state, variational, parameter-sensitivity)
# system. Because every RK4 stage of the state is a smooth function of x0
# and theta, differentiating the discrete map commutes with this
# integration: Psi is exactly d x_T / d x0 and S is exactly d x_T / d theta
# of the discrete flow (in exact arithmetic).
function rk4_joint_step(m, x, Psi, S, h, jacfn)
    J1 = jacfn(m, x)
    k1 = rhs(m, x)
    K1 = J1 * Psi
    L1 = (S === nothing) ? nothing : J1 * S + param_rhs(m, x)

    x2 = x .+ (h/2) .* k1
    P2 = Psi .+ (h/2) .* K1
    S2 = (S === nothing) ? nothing : S .+ (h/2) .* L1
    J2 = jacfn(m, x2)
    k2 = rhs(m, x2)
    K2 = J2 * P2
    L2 = (S === nothing) ? nothing : J2 * S2 + param_rhs(m, x2)

    x3 = x .+ (h/2) .* k2
    P3 = Psi .+ (h/2) .* K2
    S3 = (S === nothing) ? nothing : S .+ (h/2) .* L2
    J3 = jacfn(m, x3)
    k3 = rhs(m, x3)
    K3 = J3 * P3
    L3 = (S === nothing) ? nothing : J3 * S3 + param_rhs(m, x3)

    x4 = x .+ h .* k3
    P4 = Psi .+ h .* K3
    S4 = (S === nothing) ? nothing : S .+ h .* L3
    J4 = jacfn(m, x4)
    k4 = rhs(m, x4)
    K4 = J4 * P4
    L4 = (S === nothing) ? nothing : J4 * S4 + param_rhs(m, x4)

    xn = x .+ (h/6) .* (k1 .+ (2 .* k2) .+ (2 .* k3) .+ k4)
    Pn = Psi .+ (h/6) .* (K1 .+ (2 .* K2) .+ (2 .* K3) .+ K4)
    Sn = (S === nothing) ? nothing :
         S .+ (h/6) .* (L1 .+ (2 .* L2) .+ (2 .* L3) .+ L4)
    return xn, Pn, Sn
end

"""
    integrate_flow(m, x0, T, N; sensitivities=false, record=false, jac=:analytic)

RK4 integration of the flow of `m` from `x0` over time `T` with `N` uniform
steps, jointly with the variational equations (monodromy Psi) and, if
requested, the parameter sensitivities S. `jac` can be `:analytic` or
`:fd` (finite-difference Jacobian of the rhs, for models without an
analytic Jacobian).
"""
function integrate_flow(m::AbstractCycleModel, x0, T::Real, N::Integer;
                        sensitivities::Bool=false, record::Bool=false,
                        jac::Symbol=:analytic, h_fd::Real=1e-6)
    N >= 1 || throw(ArgumentError("N must be >= 1"))
    T > 0 || throw(ArgumentError("T must be positive"))
    n = length(x0)
    h = T / N
    jacfn = (jac === :fd) ? ((mm, xx) -> fd_jacobian(mm, xx; h=h_fd)) : jacobian
    x = Vector{Float64}(x0)
    Psi = Matrix{Float64}(I, n, n)
    S = sensitivities ? zeros(n, nparams(m)) : nothing
    ts = record ? Vector{Float64}(undef, N + 1) : nothing
    xs = record ? Matrix{Float64}(undef, n, N + 1) : nothing
    if record
        ts[1] = 0.0
        xs[:, 1] = x
    end
    for i in 1:N
        x, Psi, S = rk4_joint_step(m, x, Psi, S, h, jacfn)
        if record
            ts[i + 1] = i * h
            xs[:, i + 1] = x
        end
    end
    return FlowResult(x, Psi, S, ts, xs, N, h)
end

# -----------------------------------------------------------------------
# Phase-fixing section (hyperplane phase condition)
# -----------------------------------------------------------------------

struct Section
    xa::Vector{Float64}
    va::Vector{Float64}
    dhat::Vector{Float64}
    label::String
end

"""
    Section(xa, va; label="section", min_norm=1e-12)

Phase-fixing hyperplane `{x : <x - xa, va> = 0}`, where `va` is typically
the flow `f(xa)` at an anchor point `xa`. The reduced shooting unknowns
are (s, T) with x0(s) = xa + s*dhat, dhat the unit vector perpendicular to
va. The section formulation is equivalent to the textbook 3-unknown system
[x(T;x0) - x0; <x0 - xa, va>] = 0 with the phase equation built in.

Equilibria on the section (i.e. <x_eq - xa, va> = 0) remain spurious roots
of the reduced system; guards and optional deflation handle them. For the
Stuart-Landau family a section anchored on the cycle is radial and passes
through the origin, so this case is exercised deliberately in tests.
"""
function Section(xa, va; label::AbstractString="section", min_norm::Real=1e-12)
    length(xa) == 2 || throw(ArgumentError("Section currently supports 2D states"))
    length(va) == 2 || throw(ArgumentError("Section currently supports 2D states"))
    nv = norm(va)
    nv > min_norm ||
        throw(ArgumentError("section vector va is (near) zero: the phase condition would be degenerate"))
    d = [-va[2], va[1]] ./ nv
    return Section(Vector{Float64}(xa), Vector{Float64}(va), d, String(label))
end

point_on_section(sec::Section, s::Real) = sec.xa .+ s .* sec.dhat
project_to_section(sec::Section, x) = dot(x .- sec.xa, sec.dhat)

# -----------------------------------------------------------------------
# Floquet data
# -----------------------------------------------------------------------

struct FloquetData
    multipliers::Vector{ComplexF64}  # sorted by |lambda - 1|
    neutral_count::Int
    classification::Symbol
    gap_to_unit::Float64             # |lambda_2 - 1| for the second-closest multiplier
    invariance_residual::Float64     # ||Psi*f(x0) - f(x0)|| / ||f(x0)||, NaN if f0 unknown
end

"""
    classify_multipliers(Psi; f0=nothing, tol_neutral=5e-4)

Classify the eigenvalues of a monodromy matrix. The multiplier within
`tol_neutral` of 1 is the phase-neutral one; stability is decided by the
remaining multipliers. Two multipliers within the tolerance raise the
`:degenerate_neutral` flag (honest resolution limit: at tol_neutral = 5e-4
a Stuart-Landau cycle with 2*mu*T < 5e-4 cannot be distinguished from a
double-neutral degeneracy by this criterion alone).
"""
function classify_multipliers(Psi::Matrix; f0=nothing, tol_neutral::Real=5e-4)
    ev_all = eigvals(Psi)
    dists = [abs(e - 1) for e in ev_all]
    ev = ev_all[sortperm(dists)]
    gap = length(ev) >= 2 ? abs(ev[2] - 1) : Inf
    neutral = count(e -> abs(e - 1) <= tol_neutral, ev)
    if neutral == 0
        cls = :no_neutral_multiplier
    elseif neutral > 1
        cls = :degenerate_neutral
    else
        rest = ev[2:end]
        if all(e -> abs(e) < 1, rest)
            cls = :stable_cycle
        elseif all(e -> abs(e) > 1, rest)
            cls = :unstable_cycle
        else
            cls = :saddle_cycle
        end
    end
    inv_res = if f0 === nothing
        NaN
    else
        norm(Psi * f0 .- f0) / max(norm(f0), 1e-300)
    end
    return FloquetData(ev, neutral, cls, gap, inv_res)
end

# -----------------------------------------------------------------------
# Phase-fixed shooting with guards and optional deflation
# -----------------------------------------------------------------------

@enum SolveStatus begin
    SolveSuccess
    GuardRejectedEquilibrium
    NewtonNotConverged
    StepOutOfBounds
    DeflationSingular
end

struct CycleSolution
    status::SolveStatus
    message::String
    model_label::String
    s::Float64
    T::Float64
    x0::Vector{Float64}
    resnorm::Float64
    iters::Int
    Psi::Union{Matrix{Float64}, Nothing}
    floquet::Union{FloquetData, Nothing}
    ts::Union{Vector{Float64}, Nothing}
    xs::Union{Matrix{Float64}, Nothing}
    diagnostics::Dict{String,Any}
    note::String
end

function _failed_solution(status::SolveStatus, msg::AbstractString,
                          model_label::AbstractString, s::Real, T::Real,
                          x0, note::AbstractString)
    return CycleSolution(status, String(msg), String(model_label), Float64(s),
                         Float64(T), Vector{Float64}(x0), NaN, 0, nothing, nothing,
                         nothing, nothing, Dict{String,Any}(), String(note))
end

"""
    solve_cycle(m, section, s0, T0, N; kwargs...)

Solve the phase-fixed periodic-orbit shooting problem with unknown period
on the transversal `section`:

    G(s, T) = x(T; x0(s)) - x0(s) = 0 in R^2,
    x0(s) = xa + s*dhat,

by damped Newton with backtracking on the residual. The Jacobian is
    dG = [Psi*dhat - dhat  |  f(x(T))]
with Psi the monodromy of the same RK4 discretization (so the Newton system
is the exact derivative of the discrete residual).

Keyword arguments:
- `eqs`: known equilibria (Vector of points). Used by the nonzero-orbit
  guards and, if `deflate=true`, by Farrell-style deflation.
- `deflate` (default false): solve sigma(s,T)*G = 0 with
  sigma = prod_j ||x0 - eq_j||^(-alpha). Deflation preserves all other
  roots, scales the Jacobian by a smooth factor and pushes Newton away
  from the deflated equilibria; it is singular exactly at them.
- `tol_newton` (1e-11): convergence threshold on ||G|| *scaled by*
  max(1, ||x0||). This is the DISCRETE residual: driving it to ~1e-12 says
  nothing about the O(h^4) distance to the continuous orbit; convergence
  studies compare across N.
- `guard_rtol`/`guard_ftol` (1e-6): the root is rejected unless
  ||x0 - eq|| >= guard_rtol * max(||xa - eq||, 1) for every listed
  equilibrium and ||f(x0)|| >= guard_ftol * max(||va||, 1). The floor of 1
  documents the assumed problem scale (states O(1e-3 .. 1e3)).
- `T_min`/`T_max`: period bounds (the equilibrium family is a root for
  every T, and T -> 0 collapses onto it; bounds keep the solve honest).
- `jac`: `:analytic` (default) or `:fd`.

Guards are safeguards, not proofs: a guard rejection certifies that the
returned root is not a listed equilibrium; it does not certify that no
other spurious root exists nearby.
"""
function solve_cycle(m::AbstractCycleModel, section::Section, s0::Real, T0::Real, N::Integer;
                     eqs::Vector{Vector{Float64}}=Vector{Vector{Float64}}(),
                     deflate::Bool=false,
                     alpha::Real=2.0,
                     tol_newton::Real=1e-11,
                     max_iter::Integer=40,
                     T_min::Real=1e-2, T_max::Real=1e3,
                     guard_rtol::Real=1e-6, guard_ftol::Real=1e-6,
                     u_defl_min::Real=1e-8,
                     ls_shrink::Real=0.5, ls_max_halvings::Integer=24, ls_c::Real=1e-4,
                     tol_neutral::Real=5e-4,
                     jac::Symbol=:analytic, h_fd::Real=1e-6,
                     model_label::AbstractString="model",
                     note::AbstractString="")

    (T_min < T0 < T_max) || return _failed_solution(
        StepOutOfBounds,
        "initial period guess T0=$T0 outside (T_min, T_max) = ($T_min, $T_max)",
        model_label, s0, T0, point_on_section(section, s0), note)

    x0_init = point_on_section(section, s0)
    if deflate && !isempty(eqs) &&
       any(norm(x0_init .- eq) < u_defl_min for eq in eqs)
        return _failed_solution(
            DeflationSingular,
            "initial guess is within u_defl_min=$u_defl_min of a deflated equilibrium",
            model_label, s0, T0, x0_init, note)
    end

    function residual_at(sv, Tv)
        x0v = point_on_section(section, sv)
        fl = integrate_flow(m, x0v, Tv, N; record=false, jac=jac, h_fd=h_fd)
        return fl.xT .- x0v, fl
    end

    function sigma_defl(x0v)
        (!deflate || isempty(eqs)) && return 1.0
        w = 1.0
        for eq in eqs
            d = norm(x0v .- eq)
            if d < u_defl_min
                return Inf
            end
            w *= d^(-alpha)
        end
        return w
    end

    function merit_of(Gv, sv)
        return norm(Gv) * sigma_defl(point_on_section(section, sv))
    end

    s = Float64(s0)
    T = Float64(T0)
    G, flow = residual_at(s, T)
    iters = 0
    converged = false
    stalled = false
    stall = 0
    while true
        iters += 1
        x0c = point_on_section(section, s)
        resnorm = norm(G)
        if resnorm <= tol_newton * max(1.0, norm(x0c))
            converged = true
            break
        end
        if iters >= max_iter
            break
        end
        J = hcat(flow.Psi * section.dhat .- section.dhat, rhs(m, flow.xT))
        if deflate && !isempty(eqs)
            gs = 0.0
            for eq in eqs
                dvec = x0c .- eq
                gs += -alpha * dot(dvec, section.dhat) / dot(dvec, dvec)
            end
            J = J .+ G .* [gs, 0.0]'
        end
        delta = try
            J \ (-G)
        catch
            nothing
        end
        if delta === nothing || !all(isfinite, delta)
            stall += 1
            stalled = true
            if stall >= 2
                break
            else
                continue
            end
        end
        m0 = merit_of(G, s)
        a = 1.0
        accepted = false
        for _ in 1:ls_max_halvings
            s_try = s + a * delta[1]
            T_try = T + a * delta[2]
            if !(T_min < T_try < T_max)
                a *= ls_shrink
                continue
            end
            G_try, fl_try = residual_at(s_try, T_try)
            m_try = merit_of(G_try, s_try)
            if m_try <= (1.0 - ls_c * a) * m0
                s, T = s_try, T_try
                G, flow = G_try, fl_try
                accepted = true
                break
            end
            a *= ls_shrink
        end
        if !accepted
            stall += 1
            stalled = true
            if stall >= 2
                break
            end
        else
            stall = 0
            stalled = false
        end
    end

    x0s = point_on_section(section, s)
    if !converged
        why = stalled ? "line search stalled" : "max_iter=$max_iter reached"
        return CycleSolution(NewtonNotConverged,
            "Newton on the section did not converge ($why); resnorm=$(norm(G))",
            String(model_label), s, T, x0s, norm(G), iters, flow.Psi, nothing,
            nothing, nothing, Dict{String,Any}(), String(note))
    end

    # Converged: re-integrate with recording for orbit diagnostics + final monodromy.
    flowf = integrate_flow(m, x0s, T, N; record=true, jac=jac, h_fd=h_fd)
    resnorm = norm(flowf.xT .- x0s)
    f0 = rhs(m, x0s)
    nf0 = norm(f0)

    # --- nonzero-orbit guards against spurious equilibrium roots ---
    guard_msgs = String[]
    rmin = Inf
    for eq in eqs
        dist0 = norm(x0s .- eq)
        eta_j = guard_rtol * max(norm(section.xa .- eq), 1.0)
        if dist0 < eta_j
            push!(guard_msgs, "endpoint distance to equilibrium $(round(dist0; sigdigits=3)) < eta $(round(eta_j; sigdigits=3))")
        end
        if flowf.xs !== nothing
            rmin = min(rmin, minimum(norm(flowf.xs[:, j] .- eq) for j in 1:size(flowf.xs, 2)))
        end
    end
    eta_f = guard_ftol * max(norm(section.va), 1.0)
    if nf0 < eta_f
        push!(guard_msgs, "endpoint flow speed $(round(nf0; sigdigits=3)) < eta_f $(round(eta_f; sigdigits=3))")
    end
    if !isempty(guard_msgs)
        msg = "nonzero-orbit guard rejected the root: " * join(guard_msgs, "; ")
        return CycleSolution(GuardRejectedEquilibrium, msg, String(model_label),
            s, T, x0s, resnorm, iters, flowf.Psi, nothing, flowf.ts, flowf.xs,
            Dict{String,Any}("orbit_min_dist_to_eq" => rmin), String(note))
    end

    # --- success: Floquet data + phase-transversality report ---
    flodata = classify_multipliers(flowf.Psi; f0=f0, tol_neutral=tol_neutral)
    transv = abs(dot(section.va, f0)) / max(norm(section.va) * nf0, 1e-300)
    diagd = Dict{String,Any}(
        "orbit_min_dist_to_eq" => rmin,
        "transversality" => transv,
        "h" => flowf.h,
        "N" => N,
        "deflated" => (deflate && !isempty(eqs)),
        "flow_speed_norm" => nf0,
        "near_degenerate" => (flodata.gap_to_unit <= 1e-4),
    )
    msg = "phase-fixed cycle found"
    if transv < 1e-3
        msg *= "; WARNING phase condition nearly tangent (transversality=$(round(transv; sigdigits=3)))"
    end
    return CycleSolution(SolveSuccess, msg, String(model_label), s, T, x0s,
        resnorm, iters, flowf.Psi, flodata, flowf.ts, flowf.xs, diagd, String(note))
end

# -----------------------------------------------------------------------
# Equilibria and Hopf detection
# -----------------------------------------------------------------------

"""
    find_equilibrium(m, x_guess; tol=1e-12, max_iter=50, jac=:analytic)

Plain Newton on f(x) = 0 with finite-difference or analytic Jacobian.
Returns (x, converged::Bool).
"""
function find_equilibrium(m::AbstractCycleModel, x_guess;
                          tol::Real=1e-12, max_iter::Integer=50,
                          jac::Symbol=:analytic, h_fd::Real=1e-6)
    x = Vector{Float64}(x_guess)
    jacfn = (jac === :fd) ? ((mm, xx) -> fd_jacobian(mm, xx; h=h_fd)) : jacobian
    ok = false
    for _ in 1:max_iter
        F = rhs(m, x)
        if norm(F) <= tol * max(1.0, norm(x))
            ok = true
            break
        end
        J = jacfn(m, x)
        if !all(isfinite, J)
            break
        end
        dx = try
            J \ (-F)
        catch
            break
        end
        if !all(isfinite, dx)
            break
        end
        x = x .+ dx
    end
    return x, ok
end

struct HopfInfo
    p_h::Float64
    ok::Bool
    message::String
    bracket::Tuple{Float64,Float64}
    eigenvalues::Vector{ComplexF64}
    max_real_at_h::Float64
    transversality::Float64
    x_eq::Vector{Float64}
    iters::Int
end

"""
    detect_hopf(make_model, p_lo, p_hi; x_guess, p_tol, jac, ...)

Bisection on g(p) = max Re lambda(J(x_eq(p))), where the equilibrium of
`make_model(p)` is re-found by Newton (warm-started) at every evaluation.
This is equilibrium Hopf detection: it is deliberately independent of any
periodic-orbit solve, and a detected Hopf point does NOT imply that a
cycle solver can or should find a finite-amplitude orbit there.
"""
function detect_hopf(make_model::Function, p_lo::Real, p_hi::Real;
                     x_guess=[0.3, -0.1], p_tol::Real=1e-10,
                     max_bisect::Integer=200, jac::Symbol=:fd, h_fd::Real=1e-6,
                     transv_step::Real=1e-3)

    function max_real_at(p, xg)
        m = make_model(p)
        xeq, ok = find_equilibrium(m, xg; jac=jac, h_fd=h_fd)
        ok || return NaN, ComplexF64[], xeq, false
        J = (jac === :fd) ? fd_jacobian(m, xeq; h=h_fd) : jacobian(m, xeq)
        ev = eigvals(J)
        return maximum(real.(ev)), ev, xeq, true
    end

    glo, _, xeq_lo, ok_lo = max_real_at(p_lo, x_guess)
    ghi, _, xeq_hi, ok_hi = max_real_at(p_hi, xeq_lo)
    if !ok_lo || !ok_hi
        return HopfInfo(NaN, false, "equilibrium find failed at a bracket end",
                        (Float64(p_lo), Float64(p_hi)), ComplexF64[], NaN, NaN,
                        xeq_lo, 0)
    end
    if !(glo < 0 && ghi > 0)
        return HopfInfo(NaN, false,
            "no sign change of max Re lambda on [$p_lo, $p_hi] (g_lo=$glo, g_hi=$ghi)",
            (Float64(p_lo), Float64(p_hi)), ComplexF64[], NaN, NaN, xeq_lo, 0)
    end
    a, b = Float64(p_lo), Float64(p_hi)
    iters = 0
    xeqw = copy(xeq_lo)
    while (b - a) > p_tol && iters < max_bisect
        iters += 1
        mid = (a + b) / 2
        g, _, xeqm, okm = max_real_at(mid, xeqw)
        if !okm
            return HopfInfo(NaN, false, "equilibrium find failed during bisection at p=$mid",
                            (a, b), ComplexF64[], NaN, NaN, xeqw, iters)
        end
        xeqw = xeqm
        if g < 0
            a = mid
        else
            b = mid
        end
    end
    p_h = (a + b) / 2
    g_h, ev_h, xeq_h, _ = max_real_at(p_h, xeqw)
    gp, _, _, okp = max_real_at(p_h + transv_step, xeq_h)
    gm, _, _, okm2 = max_real_at(p_h - transv_step, xeq_h)
    transv = (okp && okm2) ? (gp - gm) / (2 * transv_step) : NaN
    return HopfInfo(p_h, true, "Hopf bracketed", (a, b), ev_h, g_h, transv, xeq_h, iters)
end

# -----------------------------------------------------------------------
# Independent analytic Stuart-Landau reference (tests/reports only)
# -----------------------------------------------------------------------

"""
    analytic_stuart_landau(mu, omega)

Closed-form reference for the Stuart-Landau normal form (see README for
the polar-coordinate derivation): equilibrium eigenvalues mu +- i*omega,
Hopf at mu = 0 with transversality 1, and for mu > 0 the cycle
radius = sqrt(mu), period = 2*pi/omega, multipliers {1, exp(-2*mu*T)}.
Used ONLY as an independent comparison in tests and reports; no solver in
this module consumes these values.
"""
function analytic_stuart_landau(mu::Real, omega::Real)
    T = 2 * pi / omega
    return Dict{String,Any}(
        "radius" => sqrt(max(mu, 0.0)),
        "period" => T,
        "eigenvalues" => [complex(mu, omega), complex(mu, -omega)],
        "hopf_mu" => 0.0,
        "transversality" => 1.0,
        "neutral_multiplier" => 1.0,
        "radial_multiplier" => exp(-2 * mu * T),
    )
end

# -----------------------------------------------------------------------
# Generic initial-guess pipeline (no analytic knowledge of any orbit)
# -----------------------------------------------------------------------

"""
    settle_guess(m, x_ic, T_settle, N_settle)

Integrate from `x_ic` for `T_settle` and return (xa, T0, ok, message):
`xa` is the endpoint (an anchor near the attractor, if the model has one)
and `T0` is the mean spacing of successive local maxima of the first
coordinate over the last 60% of the record. Heuristic guess construction
only.
"""
function settle_guess(m::AbstractCycleModel, x_ic, T_settle::Real, N_settle::Integer)
    T_settle > 0 || throw(ArgumentError("T_settle must be positive"))
    N_settle >= 8 || throw(ArgumentError("N_settle too small"))
    fl = integrate_flow(m, x_ic, T_settle, N_settle; record=true)
    xa = Vector{Float64}(fl.xs[:, end])
    xs1 = fl.xs[1, :]
    n0 = max(2, floor(Int, 0.4 * (N_settle + 1)))
    peaks = Float64[]
    for j in (n0 + 1):(length(xs1) - 1)
        if xs1[j] > xs1[j - 1] && xs1[j] >= xs1[j + 1]
            push!(peaks, fl.ts[j])
        end
    end
    if length(peaks) < 2
        return xa, NaN, false,
            "fewer than two local maxima in the settle record; no period guess"
    end
    spacings = diff(peaks)
    T0 = sum(spacings) / length(spacings)
    return xa, T0, true, "period guess from $(length(peaks)) maxima"
end

"""
    setup_cycle_scenario(m, x_ic, T_settle, N_settle; jac=:analytic, label)

Generic initial-guess pipeline: find the equilibrium by Newton, settle
from `x_ic`, anchor the phase-fixing section at the settle endpoint with
va = f(xa). Returns a NamedTuple. `ok=false` means no valid section could
be built (e.g. the model has no attracting cycle from this IC, or the
settling collapsed onto an equilibrium).
"""
function setup_cycle_scenario(m::AbstractCycleModel, x_ic, T_settle::Real,
                              N_settle::Integer; jac::Symbol=:analytic,
                              label::AbstractString="scenario")
    x_eq, eq_ok = find_equilibrium(m, x_ic; jac=jac)
    xa, T0, s_ok, msg = settle_guess(m, x_ic, T_settle, N_settle)
    if !s_ok
        return (ok=false, message=msg, x_eq=x_eq, eq_ok=eq_ok, section=nothing,
                s0=NaN, T0=NaN, label=String(label))
    end
    va = rhs(m, xa)
    if norm(va) < 1e-12
        return (ok=false,
                message="settle endpoint is (near) an equilibrium: no phase-fixing section",
                x_eq=x_eq, eq_ok=eq_ok, section=nothing, s0=NaN, T0=NaN,
                label=String(label))
    end
    sec = Section(xa, va; label=String(label) * "-section")
    return (ok=true, message=msg, x_eq=x_eq, eq_ok=eq_ok, section=sec,
            s0=0.0, T0=T0, label=String(label))
end

# -----------------------------------------------------------------------
# Synthetic cycle parameter inference
# -----------------------------------------------------------------------

struct InferenceConfig
    make_model::Function
    section::Section
    eq_ref::Vector{Float64}       # reference equilibrium for the radius observable
    eqs::Vector{Vector{Float64}}  # guard (and, optionally, deflation) list
    N::Int
    obs::Vector{Symbol}           # subset of [:radius, :period, :logmult]
    scales::Vector{Float64}        # one normalization per observable
    theta_lo::Vector{Float64}
    theta_hi::Vector{Float64}
    tol_newton::Float64
    tol_neutral::Float64
    jac::Symbol
end

function InferenceConfig(make_model::Function, section::Section, eq_ref, eqs,
                         N::Integer, obs, scales;
                         theta_lo=[1e-4, 0.05], theta_hi=[10.0, 20.0],
                         tol_newton::Real=1e-11, tol_neutral::Real=5e-4,
                         jac::Symbol=:analytic)
    for o in obs
        o in (:radius, :period, :logmult) ||
            throw(ArgumentError("unknown observable $o (use :radius, :period, :logmult)"))
    end
    length(scales) == length(obs) ||
        throw(ArgumentError("scales must have one entry per observable"))
    all(s -> s > 0, scales) || throw(ArgumentError("scales must be positive"))
    return InferenceConfig(make_model, section, Vector{Float64}(eq_ref),
                           Vector{Vector{Float64}}(eqs), Int(N),
                           Symbol.(collect(obs)), Vector{Float64}(scales),
                           Vector{Float64}(theta_lo), Vector{Float64}(theta_hi),
                           Float64(tol_newton), Float64(tol_neutral), Symbol(jac))
end

struct ForwardResult
    ok::Bool
    message::String
    obs::Vector{Float64}
    s::Float64
    T::Float64
    sol::Union{CycleSolution, Nothing}
end

function observables_from_solution(cfg::InferenceConfig, sol::CycleSolution)
    vals = Float64[]
    for o in cfg.obs
        if o === :radius
            push!(vals, norm(sol.x0 .- cfg.eq_ref))
        elseif o === :period
            push!(vals, sol.T)
        elseif o === :logmult
            fl = sol.floquet
            if fl === nothing || length(fl.multipliers) < 2
                push!(vals, NaN)
            else
                push!(vals, log(max(abs(fl.multipliers[2]), 1e-300)))
            end
        end
    end
    return vals
end

"""
    forward_solve(theta, cfg, s_warm, T_warm)

One evaluation of the inference forward map: build the model at `theta`,
solve the phase-fixed shooting problem on the FROZEN section (warm-started
at (s_warm, T_warm)) and read off the observables. The forward map is the
full numerical solve; no closed-form Stuart-Landau value enters it.
"""
function forward_solve(theta::Vector{Float64}, cfg::InferenceConfig,
                        s_warm::Float64, T_warm::Float64)
    length(theta) == length(cfg.theta_lo) ||
        throw(ArgumentError("theta dimension does not match the inference config"))
    for k in eachindex(theta)
        if theta[k] < cfg.theta_lo[k] || theta[k] > cfg.theta_hi[k]
            return ForwardResult(false, "theta outside bounds", Float64[], s_warm, T_warm, nothing)
        end
    end
    m = cfg.make_model(theta)
    sol = solve_cycle(m, cfg.section, s_warm, T_warm, cfg.N;
                      eqs=cfg.eqs, deflate=false, tol_newton=cfg.tol_newton,
                      tol_neutral=cfg.tol_neutral, jac=cfg.jac,
                      model_label="forward")
    if sol.status != SolveSuccess
        return ForwardResult(false, "forward shooting failed: " * sol.message,
                             Float64[], sol.s, sol.T, sol)
    end
    obs = observables_from_solution(cfg, sol)
    all(isfinite, obs) ||
        return ForwardResult(false, "non-finite observables", obs, sol.s, sol.T, sol)
    return ForwardResult(true, "", obs, sol.s, sol.T, sol)
end

"""
    observables_jacobian_implicit(theta, cfg, s_star, T_star)

Jacobian d observables / d theta by implicit differentiation through the
converged phase-fixed shooting system:

    J_red * d(s, T)/d theta_k = -S(T)[:, k],

with J_red = [Psi*dhat - dhat, f(x(T))] and S the parameter-sensitivity
solution of the discrete variational equations. This is the cycle analogue
of the implicit layer used for equilibrium bifurcations in the NeurIPS
2021 work. Requires observables subset of (:radius, :period) and an
analytic `param_rhs` (no :logmult gradient: that would need second-order
sensitivity machinery, listed as future work in the README).
"""
function observables_jacobian_implicit(theta::Vector{Float64}, cfg::InferenceConfig,
                                       s_star::Float64, T_star::Float64)
    all(o -> o in (:radius, :period), cfg.obs) ||
        throw(ArgumentError("implicit gradient supports only :radius and :period"))
    m = cfg.make_model(theta)
    x0 = point_on_section(cfg.section, s_star)
    fl = integrate_flow(m, x0, T_star, cfg.N; sensitivities=true, record=false, jac=cfg.jac)
    Jred = hcat(fl.Psi * cfg.section.dhat .- cfg.section.dhat, rhs(m, fl.xT))
    p = nparams(m)
    Jout = Matrix{Float64}(undef, length(cfg.obs), p)
    rr = norm(x0 .- cfg.eq_ref)
    for k in 1:p
        du = Jred \ (-fl.S[:, k])
        for (i, o) in enumerate(cfg.obs)
            if o === :radius
                Jout[i, k] = (rr > 0) ? dot(x0 .- cfg.eq_ref, cfg.section.dhat) / rr * du[1] : NaN
            elseif o === :period
                Jout[i, k] = du[2]
            end
        end
    end
    return Jout
end

function observables_jacobian_fd(theta::Vector{Float64}, cfg::InferenceConfig,
                                 s_warm::Float64, T_warm::Float64;
                                 h_fd::Real=1e-5)
    p = length(theta)
    Jout = Matrix{Float64}(undef, length(cfg.obs), p)
    for k in 1:p
        tp = copy(theta); tp[k] += h_fd
        tm = copy(theta); tm[k] -= h_fd
        fp = forward_solve(tp, cfg, s_warm, T_warm)
        fm = forward_solve(tm, cfg, s_warm, T_warm)
        if !fp.ok || !fm.ok
            return nothing
        end
        Jout[:, k] = (fp.obs .- fm.obs) ./ (2 * h_fd)
    end
    return Jout
end

struct InferenceResult
    status::Symbol   # :recovered, :max_iter, :stalled, :forward_failure, :gradient_failure
    mismatch::Bool   # final scaled residual exceeded tol_mismatch
    message::String
    theta_hat::Vector{Float64}
    residual_scaled::Float64
    iters::Int
    trace_thetas::Vector{Vector{Float64}}
    trace_residuals::Vector{Float64}
    final_obs::Vector{Float64}
end

"""
    infer_parameters(theta0, target_obs, cfg; s_warm, T_warm, ...)

Gauss-Newton / least-squares inference of theta from synthetic cycle
observables, through the numerical forward map `forward_solve`. `scales`
normalize the residual (default meaning documented in the README).
`tols`: `tol_fit` declares recovery (scaled residual <= tol_fit);
`tol_mismatch` is the model-mismatch threshold: set it to your noise floor
for noisy fits, or to the mismatch scale you care about for consistency
checks. Statuses are honest: overdetermined inconsistent targets stall at
a nonzero residual and are flagged, never silently "recovered".
"""
function infer_parameters(theta0::Vector{Float64}, target_obs::Vector{Float64},
                           cfg::InferenceConfig; s_warm::Float64, T_warm::Float64,
                           max_iter::Integer=50, tol_fit::Real=1e-8,
                           tol_mismatch::Real=1e-2, tol_step::Real=1e-10,
                           grad::Symbol=:auto, h_fd_theta::Real=1e-5)

    length(target_obs) == length(cfg.obs) ||
        throw(ArgumentError("target length must match obs length"))
    length(theta0) == length(cfg.theta_lo) ||
        throw(ArgumentError("theta0 dimension does not match the inference config"))
    for (i, o) in enumerate(cfg.obs)
        if (o === :radius || o === :period) && !(target_obs[i] > 0)
            throw(ArgumentError("target $o must be positive, got $(target_obs[i])"))
        end
        if o === :logmult && !isfinite(target_obs[i])
            throw(ArgumentError("target :logmult must be finite"))
        end
    end
    theta = clamp.(Vector{Float64}(theta0), cfg.theta_lo, cfg.theta_hi)
    sw, Tw = Float64(s_warm), Float64(T_warm)
    trace_thetas = [copy(theta)]
    trace_res = Float64[]
    obs_final = Float64[]
    res = Inf
    iters = 0
    status = :max_iter
    msg = ""
    use_implicit = (grad === :implicit) ||
                   (grad === :auto && all(o -> o in (:radius, :period), cfg.obs))
    stall = 0
    for it in 1:max_iter
        iters = it
        fwd = forward_solve(theta, cfg, sw, Tw)
        if !fwd.ok
            status = :forward_failure
            msg = fwd.message
            break
        end
        sw, Tw = fwd.s, fwd.T
        obs_final = fwd.obs
        r = (fwd.obs .- target_obs) ./ cfg.scales
        res = norm(r)
        push!(trace_res, res)
        if res <= tol_fit
            status = :recovered
            break
        end
        J = use_implicit ? observables_jacobian_implicit(theta, cfg, sw, Tw) : nothing
        if J === nothing
            J = observables_jacobian_fd(theta, cfg, sw, Tw; h_fd=h_fd_theta)
        end
        if J === nothing
            status = :gradient_failure
            msg = "finite-difference gradient evaluation failed"
            break
        end
        J_scaled = J ./ cfg.scales
        dtheta = try
            J_scaled \ (-r)
        catch
            nothing
        end
        if dtheta === nothing || !all(isfinite, dtheta)
            status = :gradient_failure
            msg = "non-finite or singular Gauss-Newton step"
            break
        end
        theta_new = clamp.(theta .+ dtheta, cfg.theta_lo, cfg.theta_hi)
        step = norm(theta_new .- theta)
        push!(trace_thetas, copy(theta_new))
        theta = theta_new
        if step <= tol_step
            stall += 1
            if stall >= 2
                status = :stalled
                break
            end
        else
            stall = 0
        end
    end
    mismatch = isfinite(res) && (res > tol_mismatch)
    return InferenceResult(status, mismatch, msg, theta, res, iters,
                           trace_thetas, trace_res, obs_final)
end

# -----------------------------------------------------------------------
# Artifact writers (stdlib only)
# -----------------------------------------------------------------------

function write_csv(path::AbstractString, header::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            cells = String[]
            for v in r
                if v isa AbstractFloat
                    push!(cells, isfinite(v) ? (@sprintf("%.17g", v)) : string(v))
                else
                    push!(cells, string(v))
                end
            end
            println(io, join(cells, ","))
        end
    end
    return path
end

function write_toml(path::AbstractString, data::Dict{String,Any})
    open(path, "w") do io
        TOML.print(io, data)
    end
    return path
end

"""
    solution_summary_dict(sol)

Flatten a CycleSolution into a TOML-friendly Dict (complex multipliers as
separate real/imaginary vectors, no nested arrays of arrays).
"""
function solution_summary_dict(sol::CycleSolution)
    d = Dict{String,Any}(
        "status" => string(sol.status),
        "message" => sol.message,
        "s" => sol.s,
        "T" => sol.T,
        "x0" => collect(sol.x0),
        "resnorm" => sol.resnorm,
        "iters" => sol.iters,
    )
    if sol.Psi !== nothing
        d["monodromy_det"] = det(sol.Psi)
        d["monodromy_trace"] = tr(sol.Psi)
    end
    if sol.floquet !== nothing
        d["multipliers_real"] = [real(z) for z in sol.floquet.multipliers]
        d["multipliers_imag"] = [imag(z) for z in sol.floquet.multipliers]
        d["neutral_count"] = sol.floquet.neutral_count
        d["classification"] = String(sol.floquet.classification)
        d["gap_to_unit"] = sol.floquet.gap_to_unit
        d["invariance_residual"] = sol.floquet.invariance_residual
    end
    for (k, v) in sol.diagnostics
        d["diag_" * string(k)] = v
    end
    return d
end

end # module
