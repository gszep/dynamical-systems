# tests.jl — experiment 01 expected-value tests (stdlib only)
#
# Every assertion compares against an INDEPENDENT expected value (closed-form
# analytic reference, hand-computed arithmetic, or a documented code-reading
# semantic) — not implementation self-agreement. Scientific negatives (cusp
# singularity, NaN gradients at exact match, sign flips) are asserted as
# EXPECTED cases. Unexpected failures exit nonzero.
#
# Run: <pinned julia> tests.jl   (exit code 0 = all passed)
using Test
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "src", "EquilibriaProbes.jl"))
using .EquilibriaProbes

# local helper (independent of main.jl, which runs the full probe suite)
function sn_folds_local(θ, warm)
    out = Vector{Float64}[]
    for z0 in warm
        z, ok = solve_augmented(SNModel(), z0, θ)
        ok && push!(out, z)
    end
    return out
end

@testset "experiment 01 equilibria" begin

    @testset "p1 analytic references (independent closed forms)" begin
        # SN θ=(5/2,-1) (paper Figure 1A parameters):
        # u* = σ√(θ₁/(3|θ₂|)) = σ√(5/6), p* = -(2/3)θ₁u* = ∓(5/3)√(5/6)
        u_star = sqrt(5/6)
        p_star = (5/3)*sqrt(5/6)                      # 1.521451548625462...
        u, p = sn_fold([2.5, -1.0], +1)
        @test u ≈ u_star atol=1e-14
        @test p ≈ -p_star atol=1e-14
        u, p = sn_fold([2.5, -1.0], -1)
        @test u ≈ -u_star atol=1e-14
        @test p ≈ p_star atol=1e-14
        # independent arithmetic: (5/3)√(5/6) by hand = 1.5214515486254...
        @test abs(p_star - 1.5214515486254619) < 1e-12
        # augmented Newton multistart finds exactly the 2 folds, at p*
        sols = find_all_solutions(SNModel(), [2.5, -1.0], (-3.0, 3.0), (-3.0, 3.0))
        @test length(sols) == 2
        @test sort([s[2] for s in sols]) ≈ [-p_star, p_star] atol=1e-10
        # PF θ=(1/2,-1) (paper Figure 1B parameters): u* = cbrt(-1/4),
        # p* = -3θ₂u*² = 3(1/4)^{2/3} = 1.1905507889761497...
        u, p = pf_fold([0.5, -1.0])
        @test u ≈ -0.6299605249474366 atol=1e-14
        @test p ≈ 3*(0.25)^(2/3) atol=1e-14
        @test abs(p - 1.1905507889761497) < 1e-12
        sols = find_all_solutions(PFModel(), [0.5, -1.0], (-3.0, 3.0), (-3.0, 3.0))
        @test length(sols) == 1
        @test sols[1][2] ≈ 1.1905507889761497 atol=1e-10
    end

    @testset "p2 Eq (10) implicit gradients vs analytic and FD" begin
        # hand-derived closed forms (two independent symbolic routes):
        # SN: dp*/dθ = (-u*, -u*³); PF: dp*/dθ = (-1/u*, -u*²)
        θ = [2.5, -1.0]
        for σ in (+1, -1)
            u, p = sn_fold(θ, σ)
            va = sn_velocity_analytic(θ, σ)
            @test va ≈ [-u, -u^3] atol=1e-15          # internal consistency of derivation
            vi = velocity(SNModel(), [u, p], θ)
            @test vi ≈ va atol=1e-9                    # Eq (10) matches analytic
            vf, ok = velocity_fd(SNModel(), [u, p], θ; h=1e-5)
            @test ok
            @test vf ≈ va atol=1e-6                    # FD matches analytic
        end
        θ = [0.5, -1.0]
        u, p = pf_fold(θ)
        va = pf_velocity_analytic(θ)
        @test va ≈ [-1/u, -u^2] atol=1e-15
        vi = velocity(PFModel(), [u, p], θ)
        @test vi ≈ va atol=1e-9
        vf, ok = velocity_fd(PFModel(), [u, p], θ; h=1e-5)
        @test ok
        @test vf ≈ va atol=1e-6
        # independent spot value: SN σ=+1, dp*/dθ₁ = -√(5/6) = -0.9128709...
        @test abs(velocity(SNModel(), [sn_fold([2.5, -1.0], +1)...], [2.5, -1.0])[1] -
                  (-0.9128709291752769)) < 1e-9
    end

    @testset "p2c C.8 deformation = analytic normal component" begin
        # SN curve z(u) = (u, -θ₁u-θ₂u³): at fixed u the analytic displacement
        # under θ is (0,-u) and (0,-u³); its normal component must equal C.8
        θ = [2.5, -1.0]
        for u0 in (0.25, 0.5, 0.75, 1.3)
            z0 = curve_points(SNModel(), u0, θ)
            D = deformation_c8(SNModel(), z0, θ)
            a = θ[1] + 3θ[2]*u0^2
            T = [1.0, -a]/sqrt(1 + a^2)
            for (k, dz) in enumerate([(0.0, -u0), (0.0, -u0^3)])
                normal = collect(dz) - dot(collect(dz), T)*T
                @test D[:, k] ≈ normal atol=1e-12
                @test abs(dot(D[:, k], T)) < 1e-12     # C.8 is purely normal
            end
        end
    end

    @testset "p3 normalisation: Ψ invariant, Φ = Ψ·L" begin
        θ = [2.5, -1.0]
        Ψ1 = simpson_Ψ(SNModel(), θ, φ_eq5, -2.0, 2.0; n=4000)
        Ψ2 = simpson_Ψ(SNModel(), θ, φ_eq5, -2.0, 2.0; n=32000)
        @test abs(Ψ2.Ψ - Ψ1.Ψ) < 1e-8                  # refinement invariant
        @test 0 < Ψ1.Ψ < 1                             # Eq (7) normalised
        Φ = trapz_Φ(SNModel(), θ, φ_eq5, -2.0, 2.0; n=2000)
        @test Φ.Φ > 1                                   # unnormalised sum exceeds 1
        @test abs(Φ.Φ/Ψ1.Ψ - Ψ1.L)/Ψ1.L < 1e-5         # Φ = Ψ·L (identity)
        # sign consequence: count-term contributions have opposite signs
        @test log(Ψ1.Ψ) < 0
        @test log(Φ.Φ) > 0                             # Φ > 1 here
        @test sign(log(Ψ1.Ψ)) != sign(log(Φ.Φ))
    end

    @testset "p4 distance vs squared distance" begin
        # hand-computed: preds {1.5}, targets {1,3}
        @test E_paper([1.5], [1.0, 3.0]) ≈ 1.0 atol=1e-14
        @test E_code([1.5], [1.0, 3.0]) ≈ 1.25 atol=1e-14
        @test E_code([1.5], [1.0, 3.0]) != (E_paper([1.5], [1.0, 3.0]))^2
        # per-target square identity: gm2_t == gm_t²
        @test code_mean(p -> (p - 1.0)^2, [1.5, 2.0]; type=:geometric) ≈
              (code_mean(p -> abs(p - 1.0), [1.5, 2.0]; type=:geometric))^2 atol=1e-14
        # minimiser sets: paper flat on [1,3], code unique at 2
        @test E_paper([1.0], [1.0, 3.0]) ≈ E_paper([2.0], [1.0, 3.0]) atol=1e-12
        @test E_paper([2.0], [1.0, 3.0]) ≈ E_paper([3.0], [1.0, 3.0]) atol=1e-12
        @test E_code([2.0], [1.0, 3.0]) < E_code([1.9], [1.0, 3.0])
        @test E_code([2.0], [1.0, 3.0]) < E_code([2.1], [1.0, 3.0])
        # gradient at 1.5: paper zero (flat region), code = 2p-4 = -1
        v = [[1.0]]
        @test ∇E_paper([1.5], v, [1.0, 3.0])[1] == 0.0
        @test ∇E_code([1.5], v, [1.0, 3.0])[1] ≈ -1.0 atol=1e-14
        # scale: ×10 in p units scales paper E by 10, code E by 100
        @test E_paper([15.0], [10.0, 30.0]) ≈ 10*E_paper([1.5], [1.0, 3.0]) atol=1e-12
        @test E_code([15.0], [10.0, 30.0]) ≈ 100*E_code([1.5], [1.0, 3.0]) atol=1e-12
    end

    @testset "p5 measure separation" begin
        # FHN Hopf curve: det > 0 everywhere; eigen measures peak at the Hopf
        θf = [0.7, 0.8, 25.0]
        uh = hopf_location(θf)                          # √(1 - b/τ) = 0.983870...
        @test uh ≈ sqrt(1 - 0.8/25.0) atol=1e-15
        detmin = minimum(det_state(FHNModel(), curve_points(FHNModel(), u, θf), θf)
                         for u in range(-2, 2; length=161))
        @test detmin > 0                                 # determinant blind at Hopf
        z_h = curve_points(FHNModel(), uh, θf)
        @test φ_e1(FHNModel(), z_h, θf) > 0.99           # E.1 sees the crossing
        @test φ_code(FHNModel(), z_h, θf) > 1.9          # sinh form sees it (pair ×2)
        @test φ_eq5(FHNModel(), z_h, θf) < 0.6           # determinant form cannot
        # B.6 spurious family: F = p + u + (K/3)u³ has det = 1+Ku² ≥ 1
        θK = [1.0, 1e6/3]
        maxφ = maximum(φ_eq5(SNModel(), curve_points(SNModel(), u, θK), θK)
                       for u in range(-0.02, 0.02; length=401))
        @test maxφ > 0.99                                # φ≈1 with NO crossing
        @test minimum(det_state(SNModel(), curve_points(SNModel(), u, θK), θK)
                      for u in range(-0.02, 0.02; length=401)) ≥ 1.0
        # SN real fold: all measures peak; code form decays faster off-fold
        θ = [2.5, -1.0]
        u, p = sn_fold(θ, +1)
        @test φ_eq5(SNModel(), [u, p], θ) > 0.99
        # at u with det = 1 (u² = (1-θ₁)/(3θ₂)... pick u=1: det = θ₁+3θ₂ = -0.5)
        z1 = curve_points(SNModel(), 1.0, θ)             # det = 2.5-3 = -0.5
        @test abs(det_state(SNModel(), z1, θ) + 0.5) < 1e-14
        @test 0 < φ_code(SNModel(), z1, θ) < φ_e1(SNModel(), z1, θ)
    end

    @testset "jacobian contracts vs FD (every column, eq + off-eq)" begin
        # Catches reshape/row-order bugs (e.g. the FHN ∂F/∂θ column-major bug
        # fixed 2026-10-20: ∂F/∂a must be [0, 1/τ]).
        fdcol(f, x, k; h=1e-6) =
            (f(x .+ h*[j == k for j in eachindex(x)]) .-
             f(x .- h*[j == k for j in eachindex(x)])) ./ (2h)
        θf = [0.7, 0.8, 25.0]
        cases = [
            (SNModel(), [2.5, -1.0], [sn_fold([2.5, -1.0], +1)...], [0.7, 1.3]),
            (PFModel(), [0.5, -1.0], [pf_fold([0.5, -1.0])...], [0.6, 0.2]),
            (FHNModel(), θf, curve_points(FHNModel(), 0.5, θf),
             [0.3, -0.4, 1.1]),
        ]
        for (m, θ, z_eq, z_off) in cases
            for z in (z_eq, z_off)
                for j in eachindex(z)
                    @test fdcol(zz -> F(m, zz, θ), z, j) ≈ Fz(m, z, θ)[:, j] atol=1e-6
                end
                for k in eachindex(θ)
                    @test fdcol(tt -> F(m, z, tt), θ, k) ≈ Fθ(m, z, θ)[:, k] atol=1e-6
                end
                for j in eachindex(z)
                    @test fdcol(zz -> det_state(m, zz, θ), z, j) ≈
                          det_z(m, z, θ)[1, j] atol=1e-6
                end
                for k in eachindex(θ)
                    @test fdcol(tt -> det_state(m, z, tt), θ, k) ≈
                          det_θ(m, z, θ)[1, k] atol=1e-6
                end
            end
        end
    end

    @testset "p10 integrand adjoint units: sn_dΦint vs numeric θ-partial" begin
        θ = [1.0, -1.0]
        h = 1e-6
        Φ(u, θv) = sn_φ_eq5_u(u, θv)*sn_ν(u, θv)
        for u in (-1.2, -0.5, 0.3, 1.0, 1.4)
            for k in 1:2
                θp = copy(θ); θp[k] += h
                θm = copy(θ); θm[k] -= h
                @test abs((Φ(u, θp) - Φ(u, θm))/(2h) - sn_dΦint(u, θ, k)) < 1e-8
            end
        end
    end

    @testset "p10 probe adjoint: ∇Ψ_sn vs FD of Ψ (moving boundaries)" begin
        for (pmin, pmax) in ((-2.0, 2.0), (-0.6, 1.6))
            θ = [1.0, -1.0]
            adj = ∇Ψ_sn(θ, pmin, pmax; n=4000)
            noB = ∇Ψ_sn(θ, pmin, pmax; n=4000, boundaries=false)
            for k in 1:2
                h = 1e-4
                θp = copy(θ); θp[k] += h
                θm = copy(θ); θm[k] -= h
                fd = (simpson_Ψ(SNModel(), θp, φ_eq5, pmin, pmax; n=4000).Ψ -
                      simpson_Ψ(SNModel(), θm, φ_eq5, pmin, pmax; n=4000).Ψ)/(2h)
                @test abs(fd - adj.∇[k]) < 1e-6          # adjoint == FD
                @test abs(fd - noB.∇[k]) > 0.05          # boundary terms required
            end
            # the boundary (window-endpoint) terms are structurally sizeable at
            # the numerator/denominator level, not a small correction
            @test maximum(abs.(adj.boundary[1])) > 0.5
        end
    end

    @testset "p6 empty/exact/multiprediction target semantics" begin
        # empty predictions: released code errors() returns 1 for ANY targets
        # (empty-product init) and its error-term gradient is exactly zero;
        # the paper Eq (4) is undefined there (NaN recorded).
        @test E_code(Float64[], [1.0, 3.0]) == 1.0
        @test E_code(Float64[], [-50.0, 50.0]) == 1.0
        @test ∇E_code(Float64[], Vector{Vector{Float64}}(), [1.0, 3.0]) == [0.0, 0.0]
        @test E_code([1.5], Float64[]) == 0.0
        @test isnan(E_paper(Float64[], [1.0, 3.0]))
        v = velocity(SNModel(), [sn_fold([2.5, -1.0], +1)...], [2.5, -1.0])
        # C1: code objective, single prediction, exact match. E = (p−t)² is
        # SMOOTH there with true gradient 0; the released factored gradient
        # form gm2·2v/(p−t) produces NaN (0·Inf) — a REMOVABLE IMPLEMENTATION
        # DEFECT of the factored expression, not a mathematical singularity.
        @test E_code([1.5], [1.5]) == 0.0
        @test any(isnan, ∇E_code([1.5], [v], [1.5]))
        @test 2*(1.5 - 1.5)*v == zeros(2)                      # independent unfactored derivative
        @test norm(∇E_code([1.5 + 1e-6], [v], [1.5]) - 2e-6*v) < 1e-9  # factored → true limit 2dv
        @test abs(E_code([1.5 + 2e-3], [1.5]) - 4*E_code([1.5 + 1e-3], [1.5])) < 1e-12
        # C2: paper objective, single prediction, exact match: |p−t| has a
        # GENUINE kink — one-sided gradients ±v, undefined at the match
        @test E_paper([1.5], [1.5]) == 0.0
        @test norm(∇E_paper([1.5 + 1e-6], [v], [1.5]) - v) < 1e-6
        @test norm(∇E_paper([1.5 - 1e-6], [v], [1.5]) + v) < 1e-6
        @test any(isnan, ∇E_paper([1.5], [v], [1.5]))
        # C3: multiprediction, one prediction matched, one at distance A:
        # paper E ~ (|d|·A)^{1/2} has an UNBOUNDED gradient (∝ |d|^{-1/2});
        # code E = |d|·A has a bounded kink (subgradient ±A·v)
        A = 2.0; t = 1.5
        g3 = norm(∇E_paper([t + 1e-3, t + A], [v, v], [t]))
        g6 = norm(∇E_paper([t + 1e-6, t + A], [v, v], [t]))
        @test g3/g6 ≈ sqrt(1e-6/1e-3) rtol=0.03
        @test norm(∇E_code([t + 1e-9, t + A], [v, v], [t]) - A*v) < 1e-6
        # near-singleton approach behaviour (recorded in p6_exact_approach.csv)
        @test norm(∇E_paper([1.5 + 1e-9], [v], [1.5])) ≈ norm(v) rtol=1e-6
        @test norm(∇E_code([1.5 + 1e-9], [v], [1.5]))/(1e-9) ≈ 2*norm(v) rtol=1e-4
    end

    @testset "p7 degenerate crossings (expected failures)" begin
        # cusp: at θ₁ = 0 the augmented Jacobian [[0,1],[0,0]] is singular
        θ0 = [0.0, -1.0]
        J0 = vcat(Fz(SNModel(), [0.0, 0.0], θ0), det_z(SNModel(), [0.0, 0.0], θ0))
        @test abs(det(J0)) < 1e-15
        # approaching the cusp: cond(J) grows, velocity vanishes like √θ₁
        conds = Float64[]; vels = Float64[]
        for θ₁ in (1e-1, 1e-2, 1e-4, 1e-6)
            d = cusp_diagnostics(SNModel(), [θ₁, -1.0], +1)
            push!(conds, d.condJ); push!(vels, abs(d.velocity[1]))
            @test d.velocity[1] ≈ -sqrt(θ₁/3) atol=1e-6   # = -u* (SN velocity law)
        end
        @test conds[end] > conds[1]                       # conditioning degrades
        @test vels[end] < vels[1]                         # velocity → 0 at cusp
        # θ₁ < 0: no folds at all (|P| jumps 0→2 at θ₁=0) — expected
        @test isempty(find_all_solutions(SNModel(), [-0.1, -1.0],
                                        (-3.0, 3.0), (-3.0, 3.0)))
        # perfect pitchfork: rank-0 rectangular Jacobian, singular augmented J,
        # Eq (10) nonexistent — expected degenerate case
        θp = [0.0, -1.0]
        @test norm(Fz(PFModel(), [0.0, 0.0], θp)) == 0.0
        Jp = vcat(Fz(PFModel(), [0.0, 0.0], θp), det_z(PFModel(), [0.0, 0.0], θp))
        @test cond(Jp) > 1e10
        vel_succeeded_finite = try
            vv = velocity(PFModel(), [0.0, 0.0], θp)
            all(isfinite, vv)
        catch
            false
        end
        @test vel_succeeded_finite == false                # Eq (10) must fail here
        # PF velocity divergence |dp*/dθ₁| = (2/|θ₁|)^{1/3} as θ₁→0
        for θ₁ in (1e-1, 1e-6)
            u, p = pf_fold([θ₁, -1.0])
            @test abs(velocity(PFModel(), [u, p], [θ₁, -1.0])[1]) ≈
                  (2/θ₁)^(1/3) atol=1e-6
        end
    end

    @testset "p8 inference reproducibility smoke (independent)" begin
        # geometric-equivalence manifold: θ₂ = -4θ₁³/(27p*²) keeps folds at ±p*
        # (derived analytically from p*² = -4θ₁³/(27θ₂)); E_paper must
        # nearly vanish along it — the paper's central §1.1 claim
        t = (5/3)*sqrt(5/6)
        targets = [-t, t]
        for θ₁ in (1.5, 2.5, 3.5)
            θm = [θ₁, -4θ₁^3/(27*t^2)]
            wm = [[sn_fold(θm, +1)...], [sn_fold(θm, -1)...]]
            zs = sn_folds_local(θm, wm)
            @test length(zs) == 2
            @test E_paper([z[end] for z in zs], targets) < 1e-5
        end
        # and OFF the manifold E is strictly positive
        θoff = [2.5, -1.3]
        wo = [[sn_fold(θoff, +1)...], [sn_fold(θoff, -1)...]]
        zs = sn_folds_local(θoff, wo)
        @test E_paper([z[end] for z in zs], targets) > 0.01
    end
end

println("\nall experiment-01 tests passed")
