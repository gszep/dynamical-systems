"""
    EquilibriaProbes

Dependency-light (Julia stdlib only) mathematical probes for experiment 01
(equilibria): paper-faithful vs released-code objective components for
"Parameter Inference with Bifurcation Diagrams" (Szep, Dalchau, Csikasz-Nagy,
NeurIPS 2021; arXiv:2106.04243v3), tested against analytic references and
finite differences.

Conventions
-----------
* A point `z` is `Vector{Float64}` of length N+1: `[u...; p]`.
* `θ` is `Vector{Float64}` of length M.
* Paper equations implemented (arXiv v3 LaTeX source, read 2026-10-19):
  - Eq (2)  prediction set P(θ): F=0, det(∂F/∂u)=0, d/ds det ≠ 0.
    The paper's Det notation (bars around ∂F_θ/∂u) is the DETERMINANT
    (bars as determinant notation, not absolute value); Supp B writes
    Det = Πₙ λₙ (product of eigenvalues = signed determinant).
  - Eq (4)  error E = (1/|D|) Σ_{p'∈D} Π_{p∈P} |p-p'|^{1/|P|}   (unsquared)
  - Eq (5)  measure φ = (1 + |Det/(d Det/ds)|)^{-1}; Det = det(∂F/∂u)
    (signed); the outer bars are the absolute value of the ratio
  - Eq (7)  Ψ = (∫_{F=0} φ ds) / (∫_{F=0} ds)  (arclength-normalised)
  - Eq (8)  L = (|P|-|D|) log Ψ + E
  - Eq (9)  ∇L (note: the published Eq (9) contains a stray, undefined `λ`
             factor in the measure-gradient term; not implemented, matching
             the released code which has no such multiplier)
  - Eq (10) ∂p/∂θ = -p̂ (∂G/∂(u,p))^{-1} ∂G/∂θ, G = [F; Det]  (signed det,
    consistent with the released code's det(∂Fu))
  - Supp B (B.1) eigenvalue conditions; (B.6) φ→1 when dλ/ds→±∞ w/o crossing
  - Supp C (C.8) normal deformation dz/dθ = -∂F/∂z' (∂F/∂z ∂F/∂z')^{-1} ∂F/∂θ
  - Supp D (D.1) tanh-flattened measure
  - Supp E (E.1) Hopf measure Σ_λ (|d/ds log Re λ|^{-1} + 1)^{-1}
    (implemented as the log-derivative magnitude |Re λ/(d Re λ/ds)|; the
    literal formula is ill-defined for Re λ < 0)
* Released-code semantics (BifurcationInference.jl v0.1.4, commit 534ee7c,
  read 2026-10-19; snapshot under reference/):
  - errors: arithmetic mean over targets of geometric mean over predictions
    of SQUARED distance (z.p - p')², with empty-product init 1
  - measure: Σ_eigenvalues 2/(2 + |sinh(2λ)/dλ|) over REAL parts, times an
    erf window; branch level: Σ_points measure·ds (NO arclength normalisation)
  - velocity: Eq (10) with G = [F; det(∂Fu)] (raw determinant, not |det|)
"""
module EquilibriaProbes

using LinearAlgebra
using Random

export SNModel, PFModel, FHNModel,
    F, Fz, Fθ, det_state, det_z, det_θ, statejac, curve_points, curve_speed,
    tangent_unit, eigen_ds, φ_eq5, φ_tanh, φ_e1, φ_code, erf_approx,
    window_erf, solve_augmented, find_all_solutions, velocity, velocity_fd,
    sn_fold, sn_velocity_analytic, pf_fold, pf_velocity_analytic,
    branch_domain, simpson_Ψ, trapz_Φ, arclength,
    E_paper, E_code, ∇E_paper, ∇E_code, code_mean,
    deformation_c8, cusp_diagnostics, hopf_location,
    sn_det_curve, sn_ν, sn_edge_roots, sn_edge_du_dθ, sn_φ_eq5_u, sn_dΦint,
    sn_dν_k, ∇Ψ_sn, M_windowed_code

# ---------------------------------------------------------------- models ----

abstract type AbstractModel end
"Minimal saddle-node model F = p + θ₁u + θ₂u³ (N=1, M=2; paper Fig 1A)."
struct SNModel <: AbstractModel end
"Minimal pitchfork model F = θ₁ + pu + θ₂u³ (N=1, M=2; paper Fig 1B)."
struct PFModel <: AbstractModel end
"FitzHugh–Nagumo-style 2-state model with a Hopf on its equilibrium curve
(N=2, M=3): F₁ = u₁ - u₁³/3 - u₂ + p, F₂ = (u₁ + a - b u₂)/τ, θ=(a,b,τ).
Equilibrium curve parametrised by u₁: p(u₁) = (u₁+a)/b - u₁ + u₁³/3."
struct FHNModel <: AbstractModel end

statedim(::Union{SNModel,PFModel}) = 1
statedim(::FHNModel) = 2

F(::SNModel, z, θ) = [z[2] + θ[1]*z[1] + θ[2]*z[1]^3]
Fz(::SNModel, z, θ) = reshape([θ[1] + 3θ[2]*z[1]^2, 1.0], 1, 2)
Fθ(::SNModel, z, θ) = reshape([z[1], z[1]^3], 1, 2)
det_state(::SNModel, z, θ) = θ[1] + 3θ[2]*z[1]^2
det_z(::SNModel, z, θ) = reshape([6θ[2]*z[1], 0.0], 1, 2)
det_θ(::SNModel, z, θ) = reshape([1.0, 3z[1]^2], 1, 2)
"Curve point at parameter u (SN: u is the state)."
curve_points(::SNModel, u, θ) = [u, -θ[1]*u - θ[2]*u^3]
curve_speed(::SNModel, u, θ) = sqrt(1 + (θ[1] + 3θ[2]*u^2)^2)

F(::PFModel, z, θ) = [θ[1] + z[2]*z[1] + θ[2]*z[1]^3]
Fz(::PFModel, z, θ) = reshape([z[2] + 3θ[2]*z[1]^2, z[1]], 1, 2)
Fθ(::PFModel, z, θ) = reshape([1.0, z[1]^3], 1, 2)
det_state(::PFModel, z, θ) = z[2] + 3θ[2]*z[1]^2
det_z(::PFModel, z, θ) = reshape([6θ[2]*z[1], 1.0], 1, 2)
det_θ(::PFModel, z, θ) = reshape([0.0, 3z[1]^2], 1, 2)
curve_points(::PFModel, u, θ) = [u, -(θ[1] + θ[2]*u^3)/u]
curve_speed(::PFModel, u, θ) = sqrt(1 + (θ[1]/u^2 - 2θ[2]*u)^2)

F(::FHNModel, z, θ) = [z[1] - z[1]^3/3 - z[2] + z[3],
                      (z[1] + θ[1] - θ[2]*z[2])/θ[3]]
function Fz(::FHNModel, z, θ)
    a, b, τ = θ
    return [1 - z[1]^2  -1.0   1.0;
            1/τ         -b/τ   0.0]
end
function Fθ(::FHNModel, z, θ)
    a, b, τ = θ
    # column-major layout: column k is ∂F/∂θ_k
    # ∂F/∂a = [0, 1/τ], ∂F/∂b = [0, -u₂/τ], ∂F/∂τ = [0, -(u₁+a-b u₂)/τ²]
    return reshape([0.0, 1/τ,
                    0.0, -z[2]/τ,
                    0.0, -(z[1] + a - b*z[2])/τ^2], 2, 3)
end
det_state(m::FHNModel, z, θ) = (1 - θ[2]*(1 - z[1]^2))/θ[3]
det_z(m::FHNModel, z, θ) = reshape([2θ[2]*z[1]/θ[3], 0.0, 0.0], 1, 3)
det_θ(m::FHNModel, z, θ) = reshape([0.0, -(1 - z[1]^2)/θ[3],
                                    -(1 - θ[2]*(1 - z[1]^2))/θ[3]^2], 1, 3)
curve_points(::FHNModel, u, θ) = [u, (u + θ[1])/θ[2],
    (u + θ[1])/θ[2] - u + u^3/3]
curve_speed(::FHNModel, u, θ) = sqrt(1 + (1/θ[2])^2 + (1/θ[2] - 1 + u^2)^2)

statejac(m, z, θ) = Fz(m, z, θ)[:, 1:statedim(m)]

# ------------------------------------------------------ roots of p(u) = e ----
"All real roots of the polynomial c[1] + c[2]u + c[3]u² + c[4]u³
(companion matrix for the cubic; closed forms below)."
function cubic_real_roots(c::Vector{Float64})
    a = collect(c)
    while length(a) > 1 && a[end] == 0
        pop!(a)
    end
    length(a) ≤ 1 && return Float64[]
    if length(a) == 2
        return [-a[1]/a[2]]
    elseif length(a) == 3
        A, B, C = a[3], a[2], a[1]
        disc = B^2 - 4A*C
        disc < 0 && return Float64[]
        r = sqrt(disc)
        return sort([(-B + r)/(2A), (-B - r)/(2A)])
    end
    # general cubic: companion matrix
    n = length(a) - 1
    A = zeros(n, n)
    for i in 1:n-1
        A[i, i+1] = 1.0
    end
    A[n, :] .= .-(a[1:n]) ./ a[end]
    ev = eigen(A).values
    return sort(real.(ev[abs.(imag.(ev)) .< 1e-9]))
end

"Window u-domain: union of intervals where curve p(u) ∈ [pmin,pmax].
Breakpoints are exact real roots of p(u)-edge; mask tested at midpoints."
function branch_domain(m::AbstractModel, θ, pmin, pmax)
    edge_coeffs(edge) = begin
        if m isa FHNModel
            a, b, τ = θ
            # p(u) = (u+a)/b - u + u³/3  =>  u³/3 + (1/b - 1)u + (a/b - edge) = 0
            [a/b - edge, 1/b - 1, 0.0, 1/3]
        elseif m isa PFModel
            # p(u) = -θ₁/u - θ₂u²  =>  θ₂u³ + edge·u + θ₁ = 0
            [θ[1], edge, 0.0, θ[2]]
        else
            # p(u) = -θ₁u - θ₂u³  =>  θ₂u³ + θ₁u + edge = 0
            [edge, θ[1], 0.0, θ[2]]
        end
    end
    bps = sort(vcat(cubic_real_roots(edge_coeffs(pmin)),
                    cubic_real_roots(edge_coeffs(pmax))))
    m isa PFModel && (bps = sort(vcat(bps, 0.0)))   # pole at u = 0
    lo = isempty(bps) ? -50.0 : bps[1] - 1
    hi = isempty(bps) ? 50.0 : bps[end] + 1
    pushfirst!(bps, lo); push!(bps, hi)
    ivals = Tuple{Float64,Float64}[]
    for i in 1:length(bps)-1
        A, B = bps[i], bps[i+1]
        B - A < 1e-12 && continue
        mid = (A + B)/2
        pm = curve_points(m, mid, θ)[end]
        if pmin ≤ pm ≤ pmax && isfinite(pm)
            push!(ivals, (A, B))
        end
    end
    # merge adjacent intervals split by spurious roots
    merged = Tuple{Float64,Float64}[]
    for iv in ivals
        if !isempty(merged) && abs(merged[end][2] - iv[1]) < 1e-10
            merged[end] = (merged[end][1], iv[2])
        else
            push!(merged, iv)
        end
    end
    return merged
end

"Composite Simpson of f over (A,B) with n subintervals (n even)."
function simpson(f, A, B, n::Int)
    n % 2 == 1 && (n += 1)
    h = (B - A)/n
    s = f(A) + f(B)
    for i in 1:n-1
        s += (i % 2 == 0 ? 2 : 4)*f(A + i*h)
    end
    return s*h/3
end

"""
    simpson_Ψ(m, θ, φfun, pmin, pmax; n=1000)

Paper Eq (7): Ψ = ∫φ ds / ∫ds over the curve restricted to p ∈ [pmin,pmax],
using the analytic speed ν(u) = |dz/du| (Simpson quadrature). The Eq (5)
integrand has |·| kinks where det = 0 and where d det/ds = 0; the domain is
split at those points (sign-change bisection) so Simpson keeps O(h⁴).
"""
function simpson_Ψ(m, θ, φfun, pmin, pmax; n=1000)
    dom = branch_domain(m, θ, pmin, pmax)
    isempty(dom) && return (Ψ=NaN, L=0.0, num=0.0)
    num = den = 0.0
    for (A, B) in dom
        for (a, b) in kink_split(m, θ, A, B)
            num += simpson(u -> φfun(m, curve_points(m, u, θ), θ)*curve_speed(m, u, θ), a, b, n)
            den += simpson(u -> curve_speed(m, u, θ), a, b, n)
        end
    end
    return (Ψ = num/den, L = den, num = num)
end

"Bisection root of f in [a,b] (f(a)f(b) < 0), 60 iterations."
function bisect_root(f, a, b)
    fa = f(a)
    for _ in 1:80
        c = (a + b)/2
        fc = f(c)
        abs(fc) < 1e-15 && return c
        if fa*fc < 0
            b = c
        else
            a = c; fa = fc
        end
    end
    return (a + b)/2
end

"Split (A,B) at kink points of the determinant measure: sign changes of
det(u) and of its u-derivative along the curve (the two |·| kinks of Eq 5)."
function kink_split(m, θ, A, B; n=512)
    us = collect(range(A, B; length=n))
    detof(u) = det_state(m, curve_points(m, u, θ), θ)
    ddetof(u) = (detof(u + 1e-6) - detof(u - 1e-6))/2e-6
    ks = Float64[]
    for i in 1:n-1
        d1, d2 = detof(us[i]), detof(us[i+1])
        (isfinite(d1) && isfinite(d2) && d1*d2 < 0) &&
            push!(ks, bisect_root(detof, us[i], us[i+1]))
    end
    for i in 2:n-1
        d1, d2 = ddetof(us[i]), ddetof(us[i+1])
        (isfinite(d1) && isfinite(d2) && d1*d2 < 0) &&
            push!(ks, bisect_root(ddetof, us[i], us[i+1]))
    end
    pts = sort(vcat([A], ks, [B]))
    return [(pts[i], pts[i+1]) for i in 1:length(pts)-1 if pts[i+1] - pts[i] > 1e-12]
end

"""
    trapz_Φ(m, θ, φfun, pmin, pmax; n=1000)

Released-code branch measure semantics: Φ = Σ_points φ(z)·Δs over a uniform
sample of the window-restricted curve (trapezoid with chord arclengths).
This is the code's `measure(F, branch, θ, targets) = sum(s -> measure(s.z)*s.ds, branch)`.
"""
function trapz_Φ(m, θ, φfun, pmin, pmax; n=1000)
    dom = branch_domain(m, θ, pmin, pmax)
    isempty(dom) && return (Φ=NaN, L=0.0)
    Φ = 0.0
    L = 0.0
    for (A, B) in dom
        us = collect(range(A, B; length=n))
        φs = [φfun(m, curve_points(m, u, θ), θ) for u in us]
        for i in 1:n-1
            ds = norm(curve_points(m, us[i+1], θ) - curve_points(m, us[i], θ))
            Φ += (φs[i] + φs[i+1])/2*ds
            L += ds
        end
    end
    return (Φ = Φ, L = L)
end

# ----------------------------------------------------------- point measures --
"Unit tangent: right-singular vector of Fz with the smallest singular value
(full SVD). Sign is arbitrary; measures use |d/ds| so it does not matter.
The released code uses `qr(A').Q` instead (kernel()); equivalent up to sign
for full-rank Fz. Returns (t, σmin, σ2min) so degeneracy is detectable."
function tangent_unit(m, z, θ)
    A = Fz(m, z, θ)
    s = svd(A; full=true)
    t = s.V[:, end]
    σ = s.S
    σmin = isempty(σ) ? 0.0 : σ[end]
    σ2 = length(σ) ≥ 2 ? σ[end-1] : Inf
    return t/norm(t), σmin, σ2
end

"Eigenvalues and directional derivatives dλ/ds along the unit tangent
(central finite differences, h default 1e-6, complex)."
function eigen_ds(m, z, θ, t; h=1e-6)
    λof(zz) = eigen(statejac(m, zz, θ)).values
    λ0 = λof(z)
    λp = λof(z .+ h .* t)
    λm = λof(z .- h .* t)
    d = similar(λ0)
    for i in eachindex(λ0)
        j = argmin(abs.(λp .- λ0[i]))
        k = argmin(abs.(λm .- λ0[i]))
        d[i] = (λp[j] - λm[k])/(2h)
    end
    return λ0, d
end

_ratio(a, b) = b == 0 ? (a == 0 ? NaN : Inf*sign(a)) : a/b

"Paper Eq (5): φ = (1 + |Det|/|d Det/ds|)^{-1} with Det = det(∂F/∂u)."
function φ_eq5(m, z, θ)
    t, _, _ = tangent_unit(m, z, θ)
    D = det_state(m, z, θ)
    dD = dot(vec(det_z(m, z, θ)), t)
    return 1/(1 + abs(_ratio(D, dD)))
end

"Supp D (D.1): tanh-flattened determinant measure."
function φ_tanh(m, z, θ)
    t, _, _ = tangent_unit(m, z, θ)
    D = det_state(m, z, θ)
    dD = dot(vec(det_z(m, z, θ)), t)
    return 1/(1 + abs(_ratio(tanh(D), (1 - tanh(D)^2)*dD)))
end

"Supp E (E.1): Σ_λ (|d/ds log Re λ|^{-1} + 1)^{-1} over all eigenvalues
(log-derivative magnitude reading: |Re λ/(d Re λ/ds)|)."
function φ_e1(m, z, θ)
    t, _, _ = tangent_unit(m, z, θ)
    λ, d = eigen_ds(m, z, θ, t)
    s = 0.0
    for i in eachindex(λ)
        s += 1/(1 + abs(_ratio(real(λ[i]), real(d[i]))))
    end
    return s
end

"Released code per-point measure: Σ_λ 2/(2 + |sinh(2 Re λ)/d Re λ/ds|)
(Objectives.jl mapreduce), real parts of all eigenvalues (conjugate pairs
counted twice, as `realeigvals` returns both)."
function φ_code(m, z, θ)
    t, _, _ = tangent_unit(m, z, θ)
    λ, d = eigen_ds(m, z, θ, t)
    s = 0.0
    for i in eachindex(λ)
        s += 2/(2 + abs(_ratio(sinh(2*real(λ[i])), real(d[i]))))
    end
    return s
end

# --------------------------------------------------- augmented system (Eq 10) --
"""
    solve_augmented(m, z0, θ; iters, tol)

Newton on G(z) = [F(z,θ); det(∂F/∂u)] = 0 (the paper's augmented system;
the released code's `distance`), with analytic Jacobian. Returns (z, ok).
"""
function solve_augmented(m, z0, θ; iters=100, tol=1e-12)
    z = copy(z0)
    for _ in 1:iters
        g = vcat(F(m, z, θ), det_state(m, z, θ))
        norm(g, Inf) < tol && return z, true
        J = vcat(Fz(m, z, θ), det_z(m, z, θ))
        detJ = det(J)
        abs(detJ) < 1e-300 && return z, false
        Δ = J \ (.-g)
        z = z .+ Δ
        all(isfinite, z) || return z, false
    end
    g = vcat(F(m, z, θ), det_state(m, z, θ))
    return z, norm(g, Inf) < 1e-9
end

"Multistart search for all solutions of G=0 in a box (u∈ubox, p∈pbox)."
function find_all_solutions(m, θ, ubox, pbox; n=25, iters=100)
    sols = Vector{Float64}[]
    for u in range(ubox[1], ubox[2]; length=n), p in range(pbox[1], pbox[2]; length=n)
        z, ok = solve_augmented(m, [u, p], θ; iters=iters)
        ok || continue
        if all(isfinite, z) && ubox[1]-0.5 ≤ z[1] ≤ ubox[2]+0.5 &&
           pbox[1]-0.5 ≤ z[2] ≤ pbox[2]+0.5 &&
           !any(s -> norm(s - z) < 1e-7, sols)
            push!(sols, z)
        end
    end
    return sols
end

"""
    velocity(m, z, θ)

Paper Eq (10) / released code `velocity`: ∂p/∂θ = -p̂ (∂G/∂(u,p))^{-1} ∂G/∂θ
evaluated by an adjoint solve Jᵀ w = e_{N+1}. Analytic Jacobians.
"""
function velocity(m, z, θ)
    J = vcat(Fz(m, z, θ), det_z(m, z, θ))
    e = [zeros(statedim(m)); 1.0]
    w = J' \ e
    Jθ = vcat(Fθ(m, z, θ), det_θ(m, z, θ))
    return .-(Jθ' * w)
end

"Finite-difference reference: central FD of p*(θ) with warm-started re-solves."
function velocity_fd(m, zstar, θ; h=1e-5)
    v = zeros(length(θ))
    for k in eachindex(θ)
        θp = copy(θ); θp[k] += h
        θm = copy(θ); θm[k] -= h
        zp, okp = solve_augmented(m, zstar, θp)
        zm, okm = solve_augmented(m, zstar, θm)
        okp && okm || return v, false
        v[k] = (zp[end] - zm[end])/(2h)
    end
    return v, true
end

# --------------------------------------------------------- analytic references --
"SN fold at branch σ ∈ {+1,-1}: (u*, p*) with u* = σ√(-θ₁/(3θ₂))."
function sn_fold(θ, σ)
    u = σ*sqrt(-θ[1]/(3θ[2]))
    return u, -(2/3)*θ[1]*u
end

"SN analytic fold velocity dp*/dθ = (-u*, -u*³) (derived by hand; verified
against finite differences in tests.jl)."
function sn_velocity_analytic(θ, σ)
    u, _ = sn_fold(θ, σ)
    return [-u, -u^3]
end

"PF bifurcation: u* = cbrt(θ₁/(2θ₂)) (unique real), p* = -3θ₂u*²."
function pf_fold(θ)
    u = cbrt(θ[1]/(2θ[2]))
    return u, -3θ[2]*u^2
end

"PF analytic velocity dp*/dθ = (-1/u*, -u*²)."
function pf_velocity_analytic(θ)
    u, _ = pf_fold(θ)
    return [-1/u, -u^2]
end

# ------------------------------------------------------------- error terms --
"Released code mean (Utils.jl): geometric = product of f(x)^{1/n} with
init 1 (empty → 1); arithmetic = sum/n with init 0 (empty → 0)."
function code_mean(f, x; type::Symbol)
    if type == :geometric
        return mapreduce(y -> f(y)^(1/length(x)), *, x; init=1.0)
    else
        return mapreduce(y -> f(y)/length(x), +, x; init=0.0)
    end
end

"Paper Eq (4): arithmetic mean over targets of geometric mean over
predictions of |p - p'| (unsquared). |P| = 0 → NaN (paper leaves the
convention open; the released code adopts 1 via empty-product init)."
function E_paper(preds, targets)
    n = length(preds)
    return code_mean(t -> (iszero(n) ? NaN : prod(abs(p - t) for p in preds)^(1/n)),
                     targets; type=:arithmetic)
end

"Released code errors: arithmetic mean over targets of geometric mean over
predictions of (p - p')² (Objectives.jl). |P| = 0 → 1 per target (init 1)."
function E_code(preds, targets)
    return code_mean(t -> code_mean(p -> (p - t)^2, preds; type=:geometric),
                     targets; type=:arithmetic)
end

"Gradient of paper Eq (4) via Eq (9) error term:
(1/(|D||P|)) Σ_{p'} Π|p-p'|^{1/|P|} Σ_p v_p (p - p')^{-1}.
The |P|=0 case returns zeros (dimension m; the released code's error-term
gradient is zero when no predictions exist)."
function ∇E_paper(preds, vels, targets; m::Int = isempty(vels) ? 2 : length(first(vels)))
    n = length(preds)
    n == 0 && return zeros(m)
    g = zeros(m)
    for t in targets
        gm = prod(abs(p - t) for p in preds)^(1/n)
        for (i, p) in enumerate(preds)
            g .+= gm .* vels[i] ./ ((p - t)*n*length(targets))
        end
    end
    return g
end

"Gradient of code errors (Gradients.jl):
(1/|D|) Σ_{p'} gm2_{p'} · (1/|P|) Σ_p 2·v_p/(p - p')."
function ∇E_code(preds, vels, targets; m::Int = isempty(vels) ? 2 : length(first(vels)))
    n = length(preds)
    n == 0 && return zeros(m)
    g = zeros(m)
    for t in targets
        gm2 = code_mean(p -> (p - t)^2, preds; type=:geometric)
        for (i, p) in enumerate(preds)
            g .+= gm2 .* (2 .* vels[i] ./ (p - t)) ./ (n*length(targets))
        end
    end
    return g
end

# --------------------------------------------------- C.8 deformation field --
"""
    deformation_c8(m, z, θ)

Supp C (C.8) normal deformation: dz/dθ = -∂F/∂z' (∂F/∂z ∂F/∂z')^{-1} ∂F/∂θ
(Moore–Penrose inverse of the rectangular Jacobian). Returns an (N+1)×M
matrix whose k-th column is dz/dθ_k.
"""
function deformation_c8(m, z, θ)
    J = Fz(m, z, θ)                    # N × (N+1)
    Jθ = Fθ(m, z, θ)                   # N × M
    return .-(J' * ((J*J') \ Jθ))      # (N+1) × M
end

# --------------------------------------------------- erf approx --
"Abramowitz–Stegun 7.1.26 rational erf approximation, |err| ≤ 1.5e-7.
Used only for stdlib-only probes; package cross-checks use SpecialFunctions."
function erf_approx(x::Real)
    s = x < 0 ? -1.0 : 1.0
    ax = abs(x)
    t = 1/(1 + 0.5*ax)
    y = t*exp(-ax^2 - 1.26551223 + t*(1.00002368 + t*(0.37409196 +
        t*(0.09678418 + t*(-0.18628806 + t*(0.27886807 + t*(-1.13520398 +
        t*(1.48851587 + t*(-0.82215223 + t*0.17087277)))))))))
    return s*(1 - y)
end

"Released-code erf window (Objectives.jl window_function, β = 10):
(1 + erf((β(p-pmin)-3)/√2))/2 · (1 - (1 + erf((β(p-pmax)+3)/√2))/2."
function window_erf(p, pmin, pmax; β=10.0, erf=erf_approx)
    return (1 + erf((β*(p - pmin) - 3)/sqrt(2)))/2 *
           (1 - (1 + erf((β*(p - pmax) + 3)/sqrt(2)))/2)
end

# ================= analytic adjoint machinery for the SN model ================
# The SN equilibrium curve admits an explicit parametrisation
# z(u;θ) = (u, p(u;θ)), p(u;θ) = -θ₁u - θ₂u³, so the continuum θ-derivative of
# any line integral over it can be written exactly by differentiating under the
# integral sign at FIXED u (Leibniz) plus explicit moving-endpoint terms
# u_edge'(θ_k) = -(∂p/∂θ_k)/(∂p/∂u). This is the continuum derivative of the
# geometric integral; the paper's C.8 normal-deformation adjoint computes the
# same quantity without an explicit parametrisation (tangential freedom, C.9).

"det(∂F/∂u) along the SN curve at curve parameter u."
sn_det_curve(u, θ) = θ[1] + 3θ[2]*u^2
"Analytic speed ds/du = √(1 + (dp/du)²), dp/du = -det."
sn_ν(u, θ) = sqrt(1 + sn_det_curve(u, θ)^2)
"Window-intersection u roots: p(u;θ) = edge ⇔ θ₂u³ + θ₁u + edge = 0."
sn_edge_roots(θ, edge) = cubic_real_roots([edge, θ[1], 0.0, θ[2]])
"Moving endpoint derivative: u_edge'(θ_k) = -(∂p/∂θ_k)/(∂p/∂u) = -u^{2k-1}/det."
sn_edge_du_dθ(u, θ, k) = k == 1 ? -u/sn_det_curve(u, θ) : -u^3/sn_det_curve(u, θ)

"Eq (5) measure at fixed curve parameter u: φ = |B|/(|B|+|A|), identical to
(1+|A/B|)^{-1} wherever B ≠ 0 (and by continuity at A=0)."
function sn_φ_eq5_u(u, θ)
    A = sn_det_curve(u, θ)
    B = 6θ[2]*u/sn_ν(u, θ)
    return abs(B)/(abs(B) + abs(A))
end

"∂(φ·ν)/∂θ_k at fixed u (a.e. derivative; piecewise signs of A and B are
constant within kink-split quadrature pieces)."
function sn_dΦint(u, θ, k)
    A = sn_det_curve(u, θ)
    ν = sn_ν(u, θ)
    B = 6θ[2]*u/ν
    dA = k == 1 ? 1.0 : 3u^2
    dν = A*dA/ν
    dB = k == 1 ? -6θ[2]*u*dν/ν^2 : 6u/ν - 6θ[2]*u*dν/ν^2
    sA = A ≥ 0 ? 1.0 : -1.0
    sB = B ≥ 0 ? 1.0 : -1.0
    P, Q = sB*B, sA*A
    dP, dQ = sB*dB, sA*dA
    # quotient rule for φ = P/(P+Q): dφ = (Q·dP − P·dQ)/(P+Q)²
    dφ = (Q*dP - P*dQ)/(P + Q)^2
    φ = P/(P + Q)
    return dφ*ν + φ*dν
end

"∂ν/∂θ_k at fixed u."
sn_dν_k(u, θ, k) = sn_det_curve(u, θ)*(k == 1 ? 1.0 : 3u^2)/sn_ν(u, θ)

"""
    ∇Ψ_sn(θ, pmin, pmax; n=2000, boundaries=true)

Analytic Leibniz adjoint of the paper's normalised main-text measure
Ψ = (∫φ ds)/(∫ds) (Eq 5 + Eq 7) over the SN curve restricted to
p ∈ [pmin,pmax] (indicator window), with moving window-intersection
endpoints accounted for: ∇Ψ = (∇N - Ψ∇L)/L with
∇N_k = Σ[Φ·u_edge']_boundaries + ∫ ∂(φν)/∂θ_k du (fixed u),
∇L_k = Σ[ν·u_edge']_boundaries + ∫ ∂ν/∂θ_k du.
`boundaries=false` drops the endpoint terms (used to demonstrate that they
are required). Returns (∇, Ψ, L, boundary, interior, dN, dL).
"""
function ∇Ψ_sn(θ, pmin, pmax; n=2000, boundaries=true)
    dom = branch_domain(SNModel(), θ, pmin, pmax)
    isempty(dom) && error("empty window domain")
    for (a, b) in dom
        (abs(sn_det_curve(a, θ)) < 1e-8 || abs(sn_det_curve(b, θ)) < 1e-8) &&
            error("window edge coincides with a fold (det=0): non-generic setting")
    end
    dN = zeros(2); dL = zeros(2); bN = zeros(2); bL = zeros(2)
    for (a, b) in dom
        Φa = sn_φ_eq5_u(a, θ)*sn_ν(a, θ)
        Φb = sn_φ_eq5_u(b, θ)*sn_ν(b, θ)
        for k in 1:2
            bN[k] += Φb*sn_edge_du_dθ(b, θ, k) - Φa*sn_edge_du_dθ(a, θ, k)
            bL[k] += sn_ν(b, θ)*sn_edge_du_dθ(b, θ, k) -
                     sn_ν(a, θ)*sn_edge_du_dθ(a, θ, k)
        end
        for (x, y) in kink_split(SNModel(), θ, a, b)
            # the a.e. θ-derivative integrand is kinked exactly at the split
            # points (sign flips of A and B); shift pieces inward by ε so
            # Simpson nodes never evaluate the ambiguous one-sided value
            x₊, y₋ = x + 1e-9, y - 1e-9
            y₋ > x₊ || continue
            for k in 1:2
                dN[k] += simpson(u -> sn_dΦint(u, θ, k), x₊, y₋, n)
                dL[k] += simpson(u -> sn_dν_k(u, θ, k), x₊, y₋, n)
            end
        end
    end
    ΨL = simpson_Ψ(SNModel(), θ, φ_eq5, pmin, pmax; n=n)
    Ψ, L = ΨL.Ψ, ΨL.L
    totN = boundaries ? dN .+ bN : dN
    totL = boundaries ? dL .+ bL : dL
    ∇ = (totN .- Ψ.*totL)./L
    return (∇=∇, Ψ=Ψ, L=L, boundary=(bN, bL), interior=(dN, dL),
            dN=totN, dL=totL)
end

"""
    M_windowed_code(θ, pmin, pmax; n=2000, trunc=false, erf=erf_approx, urange=4.0)

Continuum counterpart of the released branch measure
(Objectives.jl `measure(F, branches, θ, targets)`): M = ∫ w_erf(p)·φ_code·ds
over the SN curve, kink-split Simpson. The released code's per-point measure
multiplies the sinh eigen-measure by the erf window; in the continuum the
window is part of the integrand (no hard domain cut).
`trunc=true` restricts the domain to the physical window intersections
u ∈ [u_left(θ), u_right(θ)] recomputed per θ (single spanning interval;
errors for non-generic windows whose edge lies inside the multivalued
region) — the honest continuum counterpart of the branch-truncated discrete
sum, whose FD derivative then includes endpoint motion that the released
adjoint (which relies on the erf decay) does not carry.
"""
function M_windowed_code(θ, pmin, pmax; n=2000, trunc=false, erf=erf_approx,
                         urange=4.0)
    if trunc
        rs = sort(vcat(sn_edge_roots(θ, pmin), sn_edge_roots(θ, pmax)))
        rs = filter(u -> abs(u) ≤ urange, rs)
        length(rs) != 2 &&
            error("non-generic window: expected exactly two intersection roots, got $(length(rs))")
        dom = [(rs[1], rs[2])]
    else
        dom = [(-urange, urange)]
    end
    fint = u -> begin
        z = curve_points(SNModel(), u, θ)
        return window_erf(z[2], pmin, pmax; erf=erf)*φ_code(SNModel(), z, θ)*
               sn_ν(u, θ)
    end
    M = 0.0
    for (a, b) in dom
        for (x, y) in kink_split(SNModel(), θ, a, b)
            M += simpson(fint, x, y, n)
        end
    end
    return M
end

# ------------------------------------------------------------ misc references --
"""
    cusp_diagnostics(m, θ)

Diagnostics near the SN cusp (θ₁ → 0, θ₂ < 0): fold (u*, p*), cond(∂G/∂z),
|velocity|. At θ₁ = 0 exactly the augmented Jacobian is singular
([[0,1],[0,0]]) and Eq (10) does not exist — expected degenerate case.
"""
function cusp_diagnostics(m, θ, σ)
    u, p = sn_fold(θ, σ)
    z = [u, p]
    J = vcat(Fz(m, z, θ), det_z(m, z, θ))
    condJ = try
        cond(J)
    catch
        Inf
    end
    v = try
        velocity(m, z, θ)
    catch
        [NaN, NaN]
    end
    any(!isfinite, v) && (v = [NaN, NaN])
    return (u=u, p=p, condJ=condJ, velocity=v, absdetJ=abs(det(J)))
end

"FHN Hopf location on the curve: trace(∂F/∂u) = 0 → u₁ = ±√(1 - b/τ)."
function hopf_location(θ)
    a, b, τ = θ
    return sqrt(1 - b/τ)
end

end # module
