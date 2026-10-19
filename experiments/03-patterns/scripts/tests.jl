# Test suite for experiment 03 (1D periodic Swift-Hohenberg stationary patterns).
#
# Usage (pinned runtime; run from the repository root):
#   JULIA_NUM_THREADS=1 \
#     /Users/gszep/Documents/repos/dynamical-systems-research-20261019/.tools/julia-1.10.11/bin/julia \
#     --project=experiments/03-patterns experiments/03-patterns/scripts/tests.jl
#
# Coverage map (see experiments/03-patterns/README.md for the expectations):
#   * independent expected values: analytic spectral/FD operator spectra,
#     single-mode Galerkin amplitude and fold predictions, exact FD circulant
#     eigenvalues, phase-mode kernel direction;
#   * negative cases: FD is NOT the spectral operator, trivial-state probes
#     are rejected, distinct patterns get distinct descriptors, invariance
#     tests are non-vacuous (the transformed vectors really differ), the
#     even-restricted spectrum has no phase eigenvalue;
#   * convergence: FD O(h^2) eigenvalue error ratios, mesh N = 32/64/128
#     solution and fold-location convergence;
#   * symmetry: translation (grid-permutation and fractional spectral shift)
#     and reflection invariance of descriptors and of the full spectrum.

using Test
using LinearAlgebra
using TOML

const PKG_DIR = dirname(@__DIR__)
include(joinpath(PKG_DIR, "src", "SwiftHohenbergPatterns.jl"))
using .SwiftHohenbergPatterns

LinearAlgebra.BLAS.set_num_threads(1)

const L_TEST = 8.0 * pi
const B_TEST = 1.0
const R_FIX = -0.10
const D64 = Discretization(L_TEST, 64)
const CFG = SHConfig(L=L_TEST, b=B_TEST)

# ---------- shared fixtures (plain top-level setup, no hidden test state) ----
const C_SEED_LARGE = to_coeffs(D64, [1.2 * cos(D64.x[j]) for j in 1:D64.N])
const C_SEED_SMALL = to_coeffs(D64, [0.35 * cos(D64.x[j]) for j in 1:D64.N])
const NEWTON_LARGE = newton_even(D64, CFG, C_SEED_LARGE, R_FIX)
const NEWTON_SMALL = newton_even(D64, CFG, C_SEED_SMALL, R_FIX)
const BRANCH_DOWN = continue_branch(D64, CFG, NEWTON_LARGE.c, R_FIX, -1; tag="test_down")
const BRANCH_UP = continue_branch(D64, CFG, NEWTON_LARGE.c, R_FIX, 1; tag="test_up")

@testset "experiment03_patterns" begin

    @testset "transforms" begin
        rng = TinyRNG(UInt64(7))
        u = [2.0 * rand(rng) - 1.0 for _ in 1:D64.N]
        uh = D64.E * u
        ub = real.(D64.Ein * uh)
        @test norm(ub .- u) < 1.0e-11
        @test maximum(abs, imag.(D64.Ein * uh)) < 1.0e-11
        # grid reflection basics
        @test norm(reflect_values(D64, reflect_values(D64, u)) .- u) < 1.0e-14
        @test norm(reflect_values(D64, u) .- u) > 1.0e-3   # non-vacuous: u is not even
        ue = project_even(D64, u)
        @test norm(project_even(D64, ue) .- ue) < 1.0e-12  # idempotent
        # coefficient <-> value roundtrip on even vectors
        c = to_coeffs(D64, ue)
        @test norm(to_values(D64, c) .- ue) < 1.0e-10
        # fractional spectral shift of a band-limited cosine is exact
        v = [0.6 * cos(1.0 * D64.x[j]) for j in 1:D64.N]
        vs = shift_values(D64, v, 0.37)
        @test norm(vs .- [0.6 * cos(1.0 * D64.x[j] - 0.37) for j in 1:D64.N]) < 1.0e-12
    end

    @testset "spectral_operator_expected_spectrum" begin
        r = R_FIX
        # assembled full-space linearization at u = 0 must equal the analytic
        # grid spectrum r - (1 - k^2)^2 (independent expected values)
        J0 = jacobian_full(D64, CFG, zeros(D64.N), r)
        lam = sort(eigvals(Symmetric(J0)))
        analytic = sort([r - D64.mfull[n] for n in 1:D64.N])
        @test length(lam) == D64.N
        @test maximum(abs, lam .- analytic) < 1.0e-8
        # cosine columns are exact eigenvectors of the spectral operator
        for n in (2, 5, 9)   # k = 0.25, 1.0, 2.0
            col = D64.S[:, n]
            @test norm(D64.Mmat * col .- D64.mcos[n] .* col) < 1.0e-8
        end
        # even-subspace spectrum at u = 0: r - (1 - k^2)^2 on cosine modes
        JE0 = jacobian_even(D64, CFG, zeros(D64.Nc), r)
        lame = sort(eigvals(Symmetric(JE0)))
        anale = sort([r - D64.mcos[n] for n in 1:D64.Nc])
        @test maximum(abs, lame .- anale) < 1.0e-8
        # independent expected value: the critical cosine mode (k = 1) sits at r
        @test abs(minimum(abs, lame) - 0.1) < 1.0e-8
    end

    @testset "fd_operator_expected_spectrum" begin
        # exact circulant eigenvalues of the periodic FD Laplacian
        for N in (32, 64, 128)
            h = L_TEST / N
            D2 = fd_laplacian(N, h)
            exact = sort(fd_laplacian_eigvals(N, h))
            got = sort(eigvals(Symmetric(D2)))
            @test maximum(abs, got .- exact) < 1.0e-9 * (1.0 + maximum(abs, exact))
        end
        # (I + D2) eigenvalues converge to the spectral analytic values at O(h^2)
        for k in (0.5, 1.0, 2.0)
            target = 1.0 - k * k
            errs = Float64[]
            for N in (32, 64, 128)
                h = L_TEST / N
                m = round(Int, k * L_TEST / (2.0 * pi))
                fdv = 1.0 - 4.0 * sin(pi * m / N)^2 / (h * h)
                @test abs(fdv - target) > 1.0e-6   # negative case: FD is not spectral
                push!(errs, abs(fdv - target))
            end
            @test errs[1] / errs[2] > 3.4 && errs[1] / errs[2] < 4.6
            @test errs[2] / errs[3] > 3.4 && errs[2] / errs[3] < 4.6
            @test errs[1] < 1.5                    # sanity window
        end
    end

    @testset "newton_stationary_and_independent_expected" begin
        @test NEWTON_LARGE.converged
        u = D64.S * NEWTON_LARGE.c
        # collocation residual via the full-space residual functional
        F = residual_full(D64, CFG, u, R_FIX)
        @test maximum(abs, F) < 1.0e-8
        # independent expected value: single-mode Galerkin amplitude
        # 0 = r + (3/4) b A^2 - (5/8) A^4  =>  A^2 = [(3/4)b + sqrt((9/16)b^2 + (5/2) r)]/(5/4)
        yplus = (0.75 * B_TEST + sqrt(0.5625 * B_TEST^2 + 2.5 * R_FIX)) / 1.25
        rms_pred = sqrt(yplus) / sqrt(2.0)
        @test abs(norm(u) / sqrt(D64.N) - rms_pred) < 0.08
        # the odd-symmetric nonlinearity keeps the mean exactly zero
        @test abs(sum(u) / D64.N) < 1.0e-8
        # two coexisting states at the same r (subcritical bistability)
        @test NEWTON_SMALL.converged
        us = D64.S * NEWTON_SMALL.c
        @test norm(us) < 0.6 * norm(u)
        sp_large = spectrum_even(D64, CFG, NEWTON_LARGE.c, R_FIX)
        sp_small = spectrum_even(D64, CFG, NEWTON_SMALL.c, R_FIX)
        @test sp_large.n_pos == 0     # post-fold segment: stable in even subspace
        @test sp_small.n_pos == 1     # pitchfork-to-fold segment: one unstable mode
        # independent FD cross-check: the spectral solution solves an FD
        # discretization of the same PDE up to O(h^2) truncation, and the
        # mismatch shrinks under refinement
        rels = Float64[]
        for N in (64, 128)
            dfn = Discretization(L_TEST, N)
            u_on = evaluate_coeffs_on(dfn, NEWTON_LARGE.c)
            Afd = fd_quadratic_op(N, L_TEST / N)
            Ffd = residual_full_fd(Afd, CFG, u_on, R_FIX)
            push!(rels, norm(Ffd) / norm(u_on))
        end
        @test rels[2] < rels[1]
        @test rels[2] < 0.05
        @test rels[1] > 1.0e-4   # negative case: FD residual is genuinely nonzero
    end

    @testset "phase_mode_and_removal" begin
        @test NEWTON_LARGE.converged
        c = NEWTON_LARGE.c
        u = D64.S * c
        J = jacobian_full(D64, CFG, u, R_FIX)
        up = derivative_values(D64, u)
        # the translation phase direction U' is in the kernel of the full-space
        # linearization (relative-equilibrium neutral mode)
        @test norm(J * up) / norm(up) < 1.0e-9
        lamF = eigvals(Symmetric(J))
        @test count(v -> abs(v) < 1.0e-8, lamF) == 1
        # the even restriction removes exactly that mode
        JE = jacobian_even(D64, CFG, c, R_FIX)
        lamE = eigvals(Symmetric(JE))
        @test count(v -> abs(v) < 1.0e-8, lamE) == 0
        @test minimum(abs, lamE) > 0.05
        # integer-grid translation is an exact permutation: the full spectrum is
        # invariant and really corresponds to a different sample vector
        us = shift_grid_values(u, 4)
        @test norm(us .- u) > 0.1
        Js = jacobian_full(D64, CFG, us, R_FIX)
        @test maximum(abs, sort(eigvals(Symmetric(Js))) .- sort(eigvals(Symmetric(J)))) < 1.0e-8
        # descriptors are invariant under the same permutation
        d1 = descriptors(D64, u)
        d2 = descriptors(D64, us)
        @test maximum(abs, descriptor_vector(d1) .- descriptor_vector(d2)) < 1.0e-10
        # and under a fractional spectral shift (which changes the values)
        uf = shift_values(D64, u, 0.37)
        @test norm(uf .- u) > 0.1
        d3 = descriptors(D64, uf)
        @test maximum(abs, descriptor_vector(d1) .- descriptor_vector(d3)) < 1.0e-10
    end

    @testset "continuation_fold" begin
        pts = BRANCH_DOWN.points
        @test length(pts) >= 20
        @test BRANCH_DOWN.term in ("trivial_approach", "step_min", "step_budget")
        # the branch dips below the start r toward the fold
        mind = minimum([pt.r for pt in pts])
        @test mind < -0.12
        # a fold (local min of r along arclength) is present, in the window of
        # the single-mode Galerkin estimate r* = -9 b^2 / 40 = -0.225
        mins = [f for f in BRANCH_DOWN.folds if f[3] == "min"]
        @test length(mins) >= 1
        rf = minimum([f[2] for f in mins])
        @test rf > -0.35 && rf < -0.12
        # fold amplitude near the Galerkin value A* = sqrt(3/5) ~ 0.775
        ir = 1
        for i in 2:length(pts)
            if pts[i].r < pts[ir].r
                ir = i
            end
        end
        @test pts[ir].rms > 0.3 && pts[ir].rms < 1.1
        # arclength constraint holds at every accepted point
        for i in 2:length(pts)
            zprev = vcat(pts[i - 1].c, [pts[i - 1].r])
            znew = vcat(pts[i].c, [pts[i].r])
            @test abs(dot(pts[i - 1].t, znew .- zprev) - pts[i].ds) < 1.0e-8
        end
        # zero-solution contamination guard: no accepted point is near trivial
        @test all(pt -> pt.rms > 0.0199, pts)
        # up direction runs to the r_stop boundary monotonically, no folds
        @test BRANCH_UP.term == "r_stop_hi"
        @test length(BRANCH_UP.points) >= 3
        @test BRANCH_UP.points[end].r >= 0.099
        @test length(BRANCH_UP.folds) == 0
        upr = [pt.r for pt in BRANCH_UP.points]
        @test all(i -> upr[i + 1] > upr[i], 1:(length(upr) - 1))
        @test BRANCH_UP.points[end].rms > 0.95 * pts[1].rms
    end

    @testset "discovery_guards_and_deflation" begin
        dres = run_discovery(D64, CFG, R_FIX)
        recs = dres.records
        @test length(recs) == 26
        # trivial probe: Newton converges numerically (to the trivial state);
        # the guard rejects it as a discovery
        triv = [rec for rec in recs if rec.spec.kind === :trivial_probe]
        @test length(triv) == 1
        @test triv[1].converged
        @test !triv[1].accepted
        @test triv[1].reason == "trivial"
        # at least one genuine periodic pattern from the cosk family
        @test any(rec -> rec.spec.family == "cosk" && rec.is_new, recs)
        # zero-solution contamination guard over the accepted set
        @test all(rec -> !rec.accepted || rec.rms > 0.05, recs)
        # dedupe without deflation: the same large seed finds a known root
        res2 = newton_even(D64, CFG, C_SEED_LARGE, R_FIX)
        @test res2.converged
        @test is_duplicate(res2.c, dres.known, CFG.dedup_tol)
        # classical deflation removes the known roots: the same seed can no
        # longer converge to the known large state (different root or failure)
        resd = newton_even(D64, CFG, C_SEED_LARGE, R_FIX; known=dres.known)
        @test !resd.converged || rel_dist(resd.c, NEWTON_LARGE.c) > 0.02
    end

    @testset "descriptor_symmetry_and_distinctness" begin
        # synthetic non-even pattern: descriptors invariant under reflection
        ut = [1.0 * cos(1.0 * D64.x[j]) + 0.5 * sin(1.0 * D64.x[j]) +
              0.2 * cos(2.0 * D64.x[j]) for j in 1:D64.N]
        d0 = descriptors(D64, ut)
        ur = reflect_values(D64, ut)
        @test norm(ur .- ut) > 1.0e-3            # non-vacuous: values differ
        d1 = descriptors(D64, ur)
        @test maximum(abs, descriptor_vector(d0) .- descriptor_vector(d1)) < 1.0e-10
        # ... and under fractional translation
        us = shift_values(D64, ut, 0.61)
        @test norm(us .- ut) > 0.1
        d2 = descriptors(D64, us)
        @test maximum(abs, descriptor_vector(d0) .- descriptor_vector(d2)) < 1.0e-10
        # distinct patterns get distinct descriptors (negative case)
        dc1 = descriptors(D64, [cos(1.0 * D64.x[j]) for j in 1:D64.N])
        dc2 = descriptors(D64, [cos(2.0 * D64.x[j]) for j in 1:D64.N])
        dloc = descriptors(D64, [1.3 / cosh(D64.x[j] / 4.0)^2 for j in 1:D64.N])
        @test abs(dc1.kbar - dc2.kbar) > 0.5
        @test abs(dc1.kbar - dloc.kbar) > 0.3
        @test abs(dc1.pr - dloc.pr) > 0.15
        @test dc1.f1 > 0.9        # cos(k=1): power in the fundamental band
        @test dc2.f2 > 0.9        # cos(k=2): power in the second band
        @test dloc.f1 < 0.5       # a wide bump spreads power over low bands
    end

    @testset "morphology_vs_topology" begin
        pts = BRANCH_DOWN.points
        # two branch points at (essentially) the same parameter r_fix, one on
        # each side of the fold: same morphology class, different topology
        near = [pt for pt in pts if abs(pt.r - R_FIX) <= 0.10]
        @test length(near) >= 2
        imx = 1
        imn = 1
        for i in 2:length(near)
            if near[i].rms > near[imx].rms
                imx = i
            end
            if near[i].rms < near[imn].rms
                imn = i
            end
        end
        pl = near[imx]
        ps = near[imn]
        @test pl.rms / ps.rms > 1.4
        dl = descriptors(D64, D64.S * pl.c)
        dsm = descriptors(D64, D64.S * ps.c)
        @test abs(dl.kbar - dsm.kbar) < 0.1       # same morphology class
        @test abs(dl.f1 - dsm.f1) < 0.2
        sl = spectrum_even(D64, CFG, pl.c, pl.r)
        ss = spectrum_even(D64, CFG, ps.c, ps.r)
        @test ss.n_pos >= 1                       # pre-fold segment is unstable
        @test sl.n_pos == 0                       # post-fold segment is stable
        @test sl.n_pos != ss.n_pos                # topology differs
    end

    @testset "mesh_convergence" begin
        Ns = [32, 64, 128]
        sols = Vector{Any}(undef, 3)
        for (i, N) in enumerate(Ns)
            d = Discretization(L_TEST, N)
            u0 = [1.2 * cos(1.0 * d.x[j]) for j in 1:N]
            res = newton_even(d, CFG, to_coeffs(d, u0), R_FIX)
            @test res.converged
            sols[i] = (N=N, d=d, res=res)
        end
        dfin = sols[3].d
        ufin = to_values(dfin, sols[3].res.c)
        nfin = norm(ufin)
        errs_u = Float64[]
        errs_c = Float64[]
        for i in 1:2
            ucf = evaluate_coeffs_on(dfin, sols[i].res.c)
            push!(errs_u, norm(ucf .- ufin) / nfin)
            push!(errs_c, norm(sols[i].res.c[1:16] .- sols[3].res.c[1:16]) /
                          norm(sols[3].res.c[1:16]))
        end
        @test errs_u[1] > errs_u[2]      # strict decrease under refinement
        @test errs_u[1] > 1.0e-7         # non-vacuous: N=32 has real aliasing error
        @test errs_u[2] < 1.0e-5         # fine-grid agreement
        @test errs_c[1] > errs_c[2]
        @test errs_c[2] < 1.0e-5
        # even-subspace spectral abscissa converges
        lam = [spectrum_even(sols[i].d, CFG, sols[i].res.c, R_FIX).lambda_max for i in 1:3]
        @test abs(lam[2] - lam[3]) < abs(lam[1] - lam[3])
        # fold location convergence via independent continuations per mesh
        rfold = Float64[]
        for i in 1:3
            br = continue_branch(sols[i].d, CFG, sols[i].res.c, R_FIX, -1;
                                 tag="meshfold_$(Ns[i])")
            mins = [f for f in br.folds if f[3] == "min"]
            @test length(mins) >= 1
            rf = minimum([f[2] for f in mins])
            @test rf > -0.35 && rf < -0.12
            push!(rfold, rf)
        end
        @test abs(rfold[3] - rfold[2]) < abs(rfold[2] - rfold[1])
        @test abs(rfold[3] - rfold[2]) < 5.0e-3
        @test abs(rfold[2] - rfold[1]) < 5.0e-2
    end

    @testset "trivial_solution_contamination" begin
        # Newton from exactly zero converges in zero iterations (trivial state)
        res0 = newton_even(D64, CFG, zeros(D64.Nc), R_FIX)
        @test res0.converged
        @test res0.iters == 0
        @test norm(D64.S * res0.c) < 1.0e-12
        # the guard classifies it as trivial, never as a discovery
        accepted, is_new, reason = classify_solution(D64, CFG, res0, Vector{Vector{Float64}}())
        @test !accepted
        @test !is_new
        @test reason == "trivial"
        # tiny-amplitude noise at a stable r also collapses to the trivial state
        rng = TinyRNG(UInt64(99))
        un = [1.0e-6 * (rand(rng) - 0.5) for _ in 1:D64.N]
        res1 = newton_even(D64, CFG, to_coeffs(D64, un), R_FIX)
        @test res1.converged
        accepted1, _, reason1 = classify_solution(D64, CFG, res1, Vector{Vector{Float64}}())
        @test !accepted1
        @test reason1 == "trivial"
    end

    @testset "io_roundtrip" begin
        data = Dict{String,Any}(
            "meta" => Dict{String,Any}("version" => "1.10.11", "count" => 3,
                                       "ok" => true, "values" => [1.5, -2.25e-9],
                                       "label" => "exp03"),
            "nested" => Dict{String,Any}("inner" => Dict{String,Any}("x" => 0.25)),
        )
        io = IOBuffer()
        dump_toml(io, data)
        parsed = TOML.parse(String(take!(io)))
        @test parsed["meta"]["count"] == 3
        @test parsed["meta"]["ok"] == true
        @test parsed["meta"]["version"] == "1.10.11"
        @test parsed["meta"]["label"] == "exp03"
        @test isapprox(parsed["meta"]["values"][1], 1.5)
        @test isapprox(parsed["meta"]["values"][2], -2.25e-9)
        @test isapprox(parsed["nested"]["inner"]["x"], 0.25)
        # fallback writer exercised directly (not untested dead code)
        io2 = IOBuffer()
        SwiftHohenbergPatterns._toml_fallback_print(io2, data)
        parsed2 = TOML.parse(String(take!(io2)))
        @test parsed2["meta"]["count"] == 3
        @test parsed2["meta"]["ok"] == true
        @test isapprox(parsed2["nested"]["inner"]["x"], 0.25)
        # CSV round trip
        hdr = ["a", "b", "c"]
        rows = [Any[1, 2.5, "x"], Any[3, NaN, true]]
        io3 = IOBuffer()
        write_csv(io3, hdr, rows)
        lines = split(String(take!(io3)), '\n')
        @test lines[1] == "a,b,c"
        @test lines[2] == "1,2.5,x"
        @test lines[3] == "3,NA,true"
    end

    @testset "rng_and_seeds_determinism" begin
        r1 = TinyRNG(UInt64(42))
        r2 = TinyRNG(UInt64(42))
        r3 = TinyRNG(UInt64(43))
        a = [rand(r1) for _ in 1:8]
        b = [rand(r2) for _ in 1:8]
        cvec = [rand(r3) for _ in 1:8]
        @test a == b
        @test a != cvec
        @test all(x -> 0.0 <= x < 1.0, a)
        # seed table is deterministic and completely specified
        seeds = seed_table(CFG, R_FIX)
        @test length(seeds) == 26
        tags1 = [s.tag for s in seed_table(CFG, R_FIX)]
        tags2 = [s.tag for s in seed_table(CFG, R_FIX)]
        @test tags1 == tags2
        @test count(s -> s.family == "cosk", seeds) == 6
        @test count(s -> s.family == "localized", seeds) == 8
        @test count(s -> s.family == "random", seeds) == 10
        @test count(s -> s.family == "trivial_probe", seeds) == 1
        @test count(s -> s.family == "highfreq", seeds) == 1
    end
end

println("experiment 03 test suite complete")
