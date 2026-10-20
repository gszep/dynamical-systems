#!/usr/bin/env julia
# experiments/02-cycles/tests.jl
#
# Test suite for experiment 02 (cycles), Julia stdlib only (Test,
# LinearAlgebra, Random, TOML). Run:
#   julia -t 1 experiments/02-cycles/tests.jl
#
# Design rules (per AGENTS.md):
#   * independent expected values: the analytic Stuart-Landau reference
#     (radius sqrt(mu), period 2*pi/omega, multipliers {1, exp(-2*mu*T)})
#     is used ONLY here as ground truth, never inside the solvers;
#   * negative cases: mu <= 0, inconsistent observables, cross-model
#     mismatch, spurious equilibrium roots, invalid inputs;
#   * convergence: RK4 time-step order checks and cross-grid consistency;
#   * discrete-derivative exactness: the fixed-N period sensitivity and the
#     implicit gradient are checked against finite differences of the
#     discrete map / of the converged discrete solve (the continuous-flow
#     identity f(x(T)) is O(h^4) off for the fixed-N map and is never used
#     in a Jacobian);
#   * result self-consistency: inference return fields always describe
#     theta_hat (max_iter=1 regression test, including the mismatch flag);
#   * neutral-multiplier expectations follow the measured + proven
#     structure: the phase multiplier is exactly 1 for the discrete map
#     (SL: proven by discrete rotational equivariance; vdp: measured at
#     the roundoff floor, no h-scaling), so only a roundoff-level bound is
#     asserted on |lambda_1 - 1|, while the genuine O(h^4) phase-structure
#     quantity -- the invariance residual ||Psi f - f||/||f|| -- carries
#     the order check.
#
# Validation status: executed 2026-10-20 (single process, one thread);
# see experiments/02-cycles/README.md for measured evidence.

include(joinpath(@__DIR__, "src", "CycleBaselines.jl"))
using .CycleBaselines
using Test
using LinearAlgebra
using Random
import TOML

@testset "experiment 02: cycles" begin

    # ------------------------------------------------------------------
    # Shared fixtures (recomputed once; plain globals inside the testset)
    # ------------------------------------------------------------------
    SL_MU, SL_OMEGA = 0.5, 1.3
    sl_m = StuartLandau(SL_MU, SL_OMEGA)
    sl_setup_fx = setup_cycle_scenario(sl_m, [1.7, 0.4], 40.0, 4000; label="sl")
    @test sl_setup_fx.ok
    sl_ref_sol = solve_cycle(sl_m, sl_setup_fx.section, sl_setup_fx.s0,
                             sl_setup_fx.T0, 256;
                             eqs=[sl_setup_fx.x_eq], model_label="sl-ref")
    @test sl_ref_sol.status == SolveSuccess
    sl_an = analytic_stuart_landau(SL_MU, SL_OMEGA)

    vdp_m = VanDerPol(1.0)
    vdp_setup_fx = setup_cycle_scenario(vdp_m, [1.7, 0.4], 80.0, 8000; label="vdp")
    @test vdp_setup_fx.ok
    vdp_ref_sol = solve_cycle(vdp_m, vdp_setup_fx.section, vdp_setup_fx.s0,
                              vdp_setup_fx.T0, 256;
                              eqs=[vdp_setup_fx.x_eq], model_label="vdp-ref")
    @test vdp_ref_sol.status == SolveSuccess

    # ------------------------------------------------------------------
    @testset "analytic model references" begin
        x = [0.3, -0.2]
        @test maximum(abs.(jacobian(sl_m, x) .- fd_jacobian(sl_m, x; h=1e-6))) < 1e-8
        @test maximum(abs.(jacobian(vdp_m, x) .- fd_jacobian(vdp_m, x; h=1e-6))) < 1e-8
        # param_rhs vs FD over the model parameters (independent construction)
        e = 1e-6
        Bfd_mu = (rhs(StuartLandau(SL_MU + e, SL_OMEGA), x) .-
                  rhs(StuartLandau(SL_MU - e, SL_OMEGA), x)) ./ (2 * e)
        Bfd_om = (rhs(StuartLandau(SL_MU, SL_OMEGA + e), x) .-
                  rhs(StuartLandau(SL_MU, SL_OMEGA - e), x)) ./ (2 * e)
        @test maximum(abs.(param_rhs(sl_m, x)[:, 1] .- Bfd_mu)) < 1e-8
        @test maximum(abs.(param_rhs(sl_m, x)[:, 2] .- Bfd_om)) < 1e-8
        Bfd_vdp = (rhs(VanDerPol(1.0 + e), x) .- rhs(VanDerPol(1.0 - e), x)) ./ (2 * e)
        @test maximum(abs.(param_rhs(vdp_m, x)[:, 1] .- Bfd_vdp)) < 1e-8
        # equilibrium: unique at the origin for both models (expected value 0)
        xeq, ok = find_equilibrium(sl_m, [0.3, -0.1])
        @test ok && norm(xeq) < 1e-10
        xeqv, okv = find_equilibrium(vdp_m, [0.3, 0.3])
        @test okv && norm(xeqv) < 1e-10
        # equilibrium spectrum: mu +- i*omega (independent expected values)
        ev = eigvals(jacobian(sl_m, xeq))
        @test abs(maximum(real.(ev)) - SL_MU) < 1e-12
        @test abs(abs(imag(ev[1])) - SL_OMEGA) < 1e-12
    end

    # ------------------------------------------------------------------
    @testset "discrete variational = derivative of discrete flow" begin
        for (m, x0, T, N) in ((sl_m, [0.7071, 0.0123], 4.7, 128),
                              (vdp_m, [1.3, -0.7], 6.5, 128))
            fl = integrate_flow(m, x0, T, N)
            hfd = 1e-5
            err_central = 0.0
            err_forward = 0.0
            for j in 1:2
                ej = j == 1 ? [1.0, 0.0] : [0.0, 1.0]
                xp = x0 .+ hfd .* ej
                xm = x0 .- hfd .* ej
                flp = integrate_flow(m, xp, T, N)
                flm = integrate_flow(m, xm, T, N)
                central = (flp.xT .- flm.xT) ./ (2 * hfd)
                fwd = (flp.xT .- fl.xT) ./ hfd
                err_central = max(err_central, maximum(abs.(central .- fl.Psi[:, j])))
                err_forward = max(err_forward, maximum(abs.(fwd .- fl.Psi[:, j])))
                @test maximum(abs.(central .- fl.Psi[:, j])) < 1e-6
            end
            # central differences must beat forward differences (O(h^2) vs O(h))
            @test err_central < err_forward / 10
        end
        # parameter sensitivities are the exact derivative too (vdp)
        e = 1e-6
        T, N = 6.5, 128
        x0v = [1.3, -0.7]
        fls = integrate_flow(vdp_m, x0v, T, N; sensitivities=true)
        fdS = (integrate_flow(VanDerPol(1.0 + e), x0v, T, N).xT .-
               integrate_flow(VanDerPol(1.0 - e), x0v, T, N).xT) ./ (2 * e)
        @test maximum(abs.(fdS .- fls.S[:, 1])) < 1e-6
    end

    # ------------------------------------------------------------------
    @testset "fixed-N discrete period sensitivity (coarse mesh)" begin
        # With N fixed, h = T/N changes with T, so d x(T)/d T of the DISCRETE
        # map is not f(x(T)) (continuous-flow identity, O(h^4) off). The
        # period-sensitivity column P must match a fixed-N central finite
        # difference in T to FD accuracy, and the naive column must be
        # visibly worse on this coarse (but stable) mesh.
        for (m, x0, T, N) in ((sl_m, [0.7071, 0.0123], 4.7, 64),
                              (vdp_m, [1.3, -0.7], 6.5, 64))
            fl = integrate_flow(m, x0, T, N; period_sensitivity=true)
            @test fl.P !== nothing
            delta = 1e-5
            flp = integrate_flow(m, x0, T + delta, N)
            flm = integrate_flow(m, x0, T - delta, N)
            fdT = (flp.xT .- flm.xT) ./ (2 * delta)
            errP = maximum(abs.(fl.P .- fdT))
            errF = maximum(abs.(rhs(m, fl.xT) .- fdT))
            @test errP < 1e-6                       # P is the exact discrete derivative
            @test errF > 10 * errP                   # naive f(x(T)) is O(h^4) off here
        end
    end

    # ------------------------------------------------------------------
    @testset "SL cycle solve vs analytic reference" begin
        @test abs(sl_ref_sol.T - sl_an["period"]) < 1e-6
        r = norm(sl_ref_sol.x0 .- sl_setup_fx.x_eq)
        @test abs(r - sl_an["radius"]) < 1e-6
        fl = sl_ref_sol.floquet
        @test fl.neutral_count == 1
        @test fl.classification == :stable_cycle
        # The phase multiplier is EXACTLY 1 for the discrete map: RK4 stages
        # commute with rotations and the SL field is rotation-equivariant, so
        # a converged fixed point comes with a whole circle of exact fixed
        # points and Psi fixes the rotation tangent. The measured deviation
        # is solve tolerance + roundoff (1.7e-15 here; worst observed
        # 1.0e-11 at the default Newton tolerance, residual-path driven).
        @test abs(fl.multipliers[1] - 1) < 1e-9
        @test abs(fl.multipliers[2] - sl_an["radial_multiplier"]) < 1e-4  # exp(-2 mu T)
        # f(x0) is NOT the exact eigenvector: at the discrete fixed point it
        # carries an O(h^4) radial component. This invariance residual is the
        # quantity that genuinely converges at O(h^4) (measured 1.2e-9 at
        # N=256, decreasing like h^4 across N).
        @test fl.invariance_residual < 1e-7
        @test sl_ref_sol.diagnostics["transversality"] > 1e-3
        @test sl_ref_sol.resnorm <= 1e-10 * max(1, norm(sl_ref_sol.x0))
        @test !sl_ref_sol.diagnostics["near_degenerate"]
    end

    # ------------------------------------------------------------------
    @testset "SL RK4 time-step convergence (observed order ~ 4)" begin
        Ns = [32, 64, 128, 256]
        s_w, T_w = sl_setup_fx.s0, sl_setup_fx.T0
        errs = Dict{Int,Vector{Float64}}()
        for N in Ns
            sol = solve_cycle(sl_m, sl_setup_fx.section, s_w, T_w, N;
                              eqs=[sl_setup_fx.x_eq])
            @test sol.status == SolveSuccess
            s_w, T_w = sol.s, sol.T
            rr = norm(sol.x0 .- sl_setup_fx.x_eq)
            # columns: err period, err radius, err radial multiplier,
            # |lambda_1 - 1| (neutral deviation), invariance residual
            errs[N] = [abs(sol.T - sl_an["period"]),
                       abs(rr - sl_an["radius"]),
                       abs(sol.floquet.multipliers[2] - sl_an["radial_multiplier"]),
                       abs(sol.floquet.multipliers[1] - 1.0),
                       sol.floquet.invariance_residual]
            @test all(isfinite, errs[N])
        end
        # Period, radius, radial multiplier and the invariance residual are
        # genuine O(h^4) quantities (measured orders ~4.0/4.0/4.1/4.0):
        # strict monotone refinement gain plus floor-guarded order.
        for k in 1:(length(Ns) - 1)
            @test all(errs[Ns[k + 1]][[1, 2, 3, 5]] .< errs[Ns[k]][[1, 2, 3, 5]])
            for c in (1, 2, 3, 5)
                # only assert an order where both errors are above the
                # roundoff floor (honest: no order claims from floor noise)
                if errs[Ns[k]][c] > 1e-9 && errs[Ns[k + 1]][c] > 1e-12
                    p = log(errs[Ns[k]][c] / errs[Ns[k + 1]][c]) / log(2)
                    @test isfinite(p)
                    @test 3.0 <= p <= 50.0
                end
            end
        end
        # The neutral deviation |lambda_1 - 1| carries NO order signal and
        # must not be asserted monotone: RK4 preserves the rotational
        # equivariance of the SL field exactly, so the converged fixed point
        # has a whole exact circle of fixed points and the phase multiplier
        # is EXACTLY 1 for the discrete map at every N. The measured
        # deviation (6.7e-16 .. 1.0e-11 over N=32..256) is driven by the
        # accepted Newton residual and roundoff, is non-monotone, and the
        # 1.0e-11 excursion at N=128 is a residual-path artifact (it does
        # not reappear at tighter tol_newton). Assert the symmetry-
        # predicted roundoff level instead of an invalid order claim.
        for N in Ns
            @test errs[N][4] < 1e-9
        end
        # cross-grid consistency: the finer map nearly closes at the solution
        fl512 = integrate_flow(sl_m, sl_ref_sol.x0, sl_ref_sol.T, 512)
        @test norm(fl512.xT .- sl_ref_sol.x0) < 1e-5
    end

    # ------------------------------------------------------------------
    @testset "SL robustness: off-cycle anchor and wrong period guess" begin
        # anchor deliberately off the cycle; period guess 30% too large
        xa = [1.0, 0.0]
        sec = Section(xa, rhs(sl_m, xa); label="off-cycle")
        sol = solve_cycle(sl_m, sec, 0.0, 2 * pi, 256; eqs=[[0.0, 0.0]])
        @test sol.status == SolveSuccess
        @test abs(sol.T - 2 * pi / SL_OMEGA) < 1e-6
        @test abs(norm(sol.x0) - sqrt(SL_MU)) < 1e-6
        @test sol.floquet.neutral_count == 1
    end

    # ------------------------------------------------------------------
    @testset "spurious equilibrium root: guard + deflation (SL)" begin
        eq0 = [0.0, 0.0]
        # fixture: anchor exactly on the analytic cycle makes the section
        # radial, hence through the equilibrium (this is a test fixture
        # construction; solvers never see the analytic orbit)
        xa = [sqrt(SL_MU), 0.0]
        sec = Section(xa, rhs(sl_m, xa); label="sl-through-origin")
        @test abs(dot(eq0 .- xa, sec.va)) < 1e-15        # fixture premise: eq on section
        s0 = project_to_section(sec, [0.01, 0.0])       # start near the equilibrium
        T0 = 2 * pi / SL_OMEGA                          # period-aligned guess
        plain = solve_cycle(sl_m, sec, s0, T0, 256; eqs=[eq0], deflate=false)
        @test plain.status == GuardRejectedEquilibrium  # never a fake success
        @test occursin("guard", lowercase(plain.message))
        @test isfinite(plain.resnorm)                   # diagnostics present
        defl = solve_cycle(sl_m, sec, s0, T0, 256; eqs=[eq0], deflate=true)
        @test defl.status == SolveSuccess
        @test abs(norm(defl.x0) - sqrt(SL_MU)) < 1e-6
        @test abs(defl.T - 2 * pi / SL_OMEGA) < 1e-6
        @test defl.floquet.neutral_count == 1
        @test abs(defl.floquet.multipliers[2] - exp(-2 * SL_MU * (2 * pi / SL_OMEGA))) < 1e-4
    end

    # ------------------------------------------------------------------
    @testset "vdp structural solve (no analytic reference exists)" begin
        sol = vdp_ref_sol
        @test sol.resnorm <= 1e-10 * max(1, norm(sol.x0))
        fl = sol.floquet
        @test fl.neutral_count == 1
        @test fl.classification == :stable_cycle
        # measured: the phase multiplier sits at the variational roundoff-
        # accumulation floor (8.9e-16 here), not at an O(h^4) distance --
        # see the convergence testset for the full discussion
        @test abs(fl.multipliers[1] - 1) < 1e-9
        @test 0 < abs(fl.multipliers[2]) < 0.99         # attracting, non-neutral
        # f(x0) is off the exact eigenvector by an O(h^4) component
        # (measured 3.2e-7 at N=256)
        @test fl.invariance_residual < 1e-5
        @test sol.diagnostics["transversality"] > 1e-3
        # cross-grid: the N=512 map must nearly close at the N=256 solution
        fl512 = integrate_flow(vdp_m, sol.x0, sol.T, 512)
        @test norm(fl512.xT .- sol.x0) < 1e-4
    end

    # ------------------------------------------------------------------
    @testset "vdp Richardson self-convergence" begin
        Ns = [64, 128, 256, 512]
        s_w, T_w = vdp_setup_fx.s0, vdp_setup_fx.T0
        Ts = Float64[]
        invars = Float64[]
        devs = Float64[]
        for N in Ns
            sol = solve_cycle(vdp_m, vdp_setup_fx.section, s_w, T_w, N;
                              eqs=[vdp_setup_fx.x_eq])
            @test sol.status == SolveSuccess
            s_w, T_w = sol.s, sol.T
            push!(Ts, sol.T)
            push!(invars, sol.floquet.invariance_residual)
            push!(devs, abs(sol.floquet.multipliers[1] - 1))
        end
        d1 = abs(Ts[2] - Ts[1])
        d2 = abs(Ts[3] - Ts[2])
        d3 = abs(Ts[4] - Ts[3])
        @test d2 < d1 && d3 < d2
        @test d1 > 1e-9                                  # above the roundoff floor
        ratio = d2 / d3
        @test 4 <= ratio <= 64                           # observed order in [2, 6]
        # vdp has no rotational symmetry, so the neutral eigenvalue's exact
        # unit value is NOT protected by an equivariance proof the way it is
        # for Stuart-Landau; it is MEASURED at the variational roundoff-
        # accumulation floor with no h-scaling (<= 1.4e-13 over N=64..512,
        # non-monotone, insensitive to tol_newton below ~1e-13), consistent
        # with RK backward error analysis (the map matches the exact flow
        # of a modified field to all algebraic orders). No order claim is
        # made for the eigenvalue itself.
        for dev in devs
            @test dev < 1e-9
        end
        # The genuine O(h^4) phase-structure error for this non-symmetric
        # system is the invariance residual ||Psi f - f||/||f||: at the
        # discrete fixed point f(x0) acquires an O(h^4) component off the
        # exact eigenvector. Measured: 1.02e-4 -> 5.50e-6 -> 3.20e-7 ->
        # 1.93e-8 over N=64..512 (orders 4.22, 4.10, 4.05).
        for k in 1:(length(Ns) - 1)
            @test invars[k + 1] < invars[k]
            if invars[k] > 1e-9 && invars[k + 1] > 1e-12
                p = log(invars[k] / invars[k + 1]) / log(2)
                @test isfinite(p)
                @test 3.0 <= p <= 50.0
            end
        end
    end

    # ------------------------------------------------------------------
    @testset "vdp with FD Jacobian matches analytic-Jacobian path" begin
        solA = solve_cycle(vdp_m, vdp_setup_fx.section, vdp_setup_fx.s0,
                           vdp_setup_fx.T0, 256; eqs=[vdp_setup_fx.x_eq], jac=:analytic)
        solF = solve_cycle(vdp_m, vdp_setup_fx.section, vdp_setup_fx.s0,
                           vdp_setup_fx.T0, 256; eqs=[vdp_setup_fx.x_eq], jac=:fd)
        @test solA.status == SolveSuccess && solF.status == SolveSuccess
        @test abs(solF.T - solA.T) < 1e-6
        @test norm(solF.x0 .- solA.x0) < 1e-5
        @test abs(solF.floquet.multipliers[2] - solA.floquet.multipliers[2]) < 1e-5
    end

    # ------------------------------------------------------------------
    @testset "spurious equilibrium root: guard + deflation (vdp)" begin
        eq0 = [0.0, 0.0]
        xa = [2.0, 0.0]
        sec = Section(xa, rhs(vdp_m, xa); label="vdp-through-origin")
        @test abs(dot(eq0 .- xa, sec.va)) < 1e-12       # fixture premise
        s0 = project_to_section(sec, [0.005, 0.0])     # deep in the linear regime
        T0 = 4 * pi / sqrt(3.0)                        # linear-spiral period-aligned
        plain = solve_cycle(vdp_m, sec, s0, T0, 256; eqs=[eq0], deflate=false)
        @test plain.status in (GuardRejectedEquilibrium, NewtonNotConverged)
        defl = solve_cycle(vdp_m, sec, s0, T0, 256; eqs=[eq0], deflate=true, max_iter=80)
        # deflation must never land on the deflated equilibrium
        @test defl.status != GuardRejectedEquilibrium
        if defl.status == SolveSuccess
            @test defl.resnorm <= 1e-10 * max(1, norm(defl.x0))
            @test defl.floquet.neutral_count == 1
            @test 0 < abs(defl.floquet.multipliers[2]) < 0.99
            fl512 = integrate_flow(vdp_m, defl.x0, defl.T, 512)
            @test norm(fl512.xT .- defl.x0) < 1e-4
            # period is phase-independent: two different sections agree
            @test abs(defl.T - vdp_ref_sol.T) < 1e-5
            @test abs(abs(defl.floquet.multipliers[2]) -
                      abs(vdp_ref_sol.floquet.multipliers[2])) < 1e-5
        end
    end

    # ------------------------------------------------------------------
    @testset "Hopf detection, amplitude law, near-degeneracy, mu<=0" begin
        h = detect_hopf(p -> StuartLandau(p, SL_OMEGA), -0.5, 0.5; jac=:fd)
        @test h.ok
        @test abs(h.p_h) < 1e-6                         # independent expected value 0
        @test 0.5 < h.transversality < 1.5              # expected value 1
        h2 = detect_hopf(p -> StuartLandau(p, SL_OMEGA), -0.5, 0.5; jac=:analytic)
        @test h2.ok && abs(h2.p_h - h.p_h) < 1e-9

        # supercritical amplitude law r(mu) ~ sqrt(mu) with continuation
        m1 = StuartLandau(1e-2, SL_OMEGA)
        st1 = setup_cycle_scenario(m1, [1.7, 0.4], 600.0, 12000; label="amp")
        @test st1.ok
        sol1 = solve_cycle(m1, st1.section, st1.s0, st1.T0, 256; eqs=[st1.x_eq])
        @test sol1.status == SolveSuccess
        @test abs(norm(sol1.x0 .- st1.x_eq) - sqrt(1e-2)) <= 0.05 * sqrt(1e-2)
        section, s_w, T_w = st1.section, sol1.s, sol1.T
        for (mu, cls_expected) in ((1e-4, :stable_cycle), (1e-6, :degenerate_neutral))
            mm = StuartLandau(mu, SL_OMEGA)
            solc = solve_cycle(mm, section, s_w, T_w, 256; eqs=[[0.0, 0.0]])
            @test solc.status == SolveSuccess
            r = norm(solc.x0)
            @test abs(r - sqrt(mu)) <= 0.05 * sqrt(mu)
            @test abs(solc.T - 2 * pi / SL_OMEGA) < 1e-6
            s_w, T_w = solc.s, solc.T
            fl = solc.floquet
            @test fl.classification == cls_expected
            gap = fl.gap_to_unit
            gap_analytic = abs(exp(-2 * mu * (2 * pi / SL_OMEGA)) - 1)
            @test abs(gap - gap_analytic) <= 0.6 * gap_analytic
        end

        # at the Hopf point itself there is no nonzero cycle: honest failure.
        # mu = 0: the radial decay -r^3 is slow, so the settle still leaves a
        # nonzero anchor, a valid section and a period guess -> the solve
        # itself must fail honestly (guard or non-convergence).
        st0 = setup_cycle_scenario(StuartLandau(0.0, SL_OMEGA), [1.7, 0.4],
                                   40.0, 4000; label="neg-mu0")
        @test st0.ok
        sol0 = solve_cycle(StuartLandau(0.0, SL_OMEGA), st0.section, st0.s0,
                           st0.T0, 256; eqs=[st0.x_eq])
        @test sol0.status != SolveSuccess
        @test sol0.status in (GuardRejectedEquilibrium, NewtonNotConverged,
                              StepOutOfBounds)
        # mu = -0.5: the transient collapses toward the stable focus; either
        # no section can be built or the solve fails honestly. Both are
        # recorded outcomes, never a fake cycle.
        stn = setup_cycle_scenario(StuartLandau(-0.5, SL_OMEGA), [1.7, 0.4],
                                   40.0, 4000; label="neg-mu-half")
        if stn.ok
            soln = solve_cycle(StuartLandau(-0.5, SL_OMEGA), stn.section, stn.s0,
                               stn.T0, 256; eqs=[stn.x_eq])
            @test soln.status != SolveSuccess
            @test soln.status in (GuardRejectedEquilibrium, NewtonNotConverged,
                                  StepOutOfBounds)
        end
    end

    # ------------------------------------------------------------------
    @testset "SL synthetic inference (noiseless)" begin
        mk = theta -> StuartLandau(theta[1], theta[2])
        cfg = InferenceConfig(mk, sl_setup_fx.section, sl_setup_fx.x_eq,
                              [sl_setup_fx.x_eq], 256, [:radius, :period],
                              [2.0, 2 * pi])
        target = observables_from_solution(cfg, sl_ref_sol)
        @test abs(target[1] - sqrt(SL_MU)) < 1e-6
        @test abs(target[2] - 2 * pi / SL_OMEGA) < 1e-6
        theta_true = [SL_MU, SL_OMEGA]
        res = infer_parameters([0.35, 1.0], target, cfg;
                               s_warm=sl_ref_sol.s, T_warm=sl_ref_sol.T)
        @test res.status == :recovered
        # tol_fit=1e-8 on the scaled residual maps to a parameter error of
        # roughly res/|d obs/d theta| ~ 1e-8..1e-7; 1e-6 keeps the
        # identification claim honest and non-brittle
        @test norm(res.theta_hat .- theta_true) < 1e-6
        @test res.iters <= 20
        # cross-N: target at N=512, fit at N=256 -> discretization-limited
        cfg512 = InferenceConfig(mk, sl_setup_fx.section, sl_setup_fx.x_eq,
                                 [sl_setup_fx.x_eq], 512, [:radius, :period],
                                 [2.0, 2 * pi])
        fwd512 = forward_solve(theta_true, cfg512, sl_ref_sol.s, sl_ref_sol.T)
        @test fwd512.ok
        res2 = infer_parameters([0.35, 1.0], fwd512.obs, cfg;
                                s_warm=sl_ref_sol.s, T_warm=sl_ref_sol.T)
        @test res2.status == :recovered
        @test norm(res2.theta_hat .- theta_true) < 1e-3   # honest: O(h^4)-limited
    end

    # ------------------------------------------------------------------
    @testset "inference result self-consistency after max_iter exit" begin
        # Regression (post-review): the loop used to update theta and exit at
        # max_iter/stall without re-evaluating, so residual_scaled/final_obs/
        # mismatch described the PREVIOUS iterate. Every reported field must
        # describe theta_hat.
        mk = theta -> StuartLandau(theta[1], theta[2])
        cfg = InferenceConfig(mk, sl_setup_fx.section, sl_setup_fx.x_eq,
                              [sl_setup_fx.x_eq], 256, [:radius, :period],
                              [2.0, 2 * pi])
        target = observables_from_solution(cfg, sl_ref_sol)
        theta0 = [0.35, 1.0]
        tol_mis = 0.1
        res1 = infer_parameters(theta0, target, cfg; s_warm=sl_ref_sol.s,
                                T_warm=sl_ref_sol.T, max_iter=1,
                                tol_mismatch=tol_mis)
        @test res1.status == :max_iter
        @test norm(res1.theta_hat .- theta0) > 0        # a nonzero GN update was applied
        @test length(res1.trace_thetas) == 2            # theta0 plus one update
        @test length(res1.trace_residuals) == 1         # raw in-loop history only
        # self-consistency: recompute the fitted state at theta_hat, using
        # the same warm start the re-evaluation inside infer_parameters saw
        # (the (s, T) of the iteration-1 solve), so the recompute is
        # bit-identical and the equality checks below are exact
        fwd0 = forward_solve(theta0, cfg, sl_ref_sol.s, sl_ref_sol.T)
        @test fwd0.ok
        fwd1 = forward_solve(res1.theta_hat, cfg, fwd0.s, fwd0.T)
        @test fwd1.ok
        @test norm(res1.final_obs .- fwd1.obs) < 1e-12
        res_check = norm((fwd1.obs .- target) ./ cfg.scales)
        @test abs(res1.residual_scaled - res_check) < 1e-12
        @test res1.mismatch == (res_check > tol_mis)
    end

    # ------------------------------------------------------------------
    @testset "inference gradients: implicit vs FD vs analytic" begin
        mk = theta -> StuartLandau(theta[1], theta[2])
        cfg = InferenceConfig(mk, sl_setup_fx.section, sl_setup_fx.x_eq,
                              [sl_setup_fx.x_eq], 256, [:radius, :period],
                              [2.0, 2 * pi])
        theta_true = [SL_MU, SL_OMEGA]
        Jim = observables_jacobian_implicit(theta_true, cfg,
                                            sl_ref_sol.s, sl_ref_sol.T)
        # independent analytic reference for the normal form
        @test abs(Jim[1, 1] - 1 / (2 * sqrt(SL_MU))) < 1e-5
        @test abs(Jim[1, 2]) < 1e-5
        @test abs(Jim[2, 1]) < 1e-5
        @test abs(Jim[2, 2] + 2 * pi / SL_OMEGA^2) < 1e-4
        Jfd = observables_jacobian_fd(theta_true, cfg, sl_ref_sol.s, sl_ref_sol.T;
                                      h_fd=1e-5)
        @test Jfd !== nothing
        @test maximum(abs.(Jfd .- Jim)) < 1e-4
    end

    # ------------------------------------------------------------------
    @testset "implicit gradient = derivative of the converged discrete solve" begin
        # Hand-rolled central FD of the full pipeline: every evaluation is a
        # converged phase-fixed shooting solve at the same N, section and
        # warm start, so the FD differentiates the converged discrete solve
        # that the implicit path claims to differentiate exactly.
        mk = theta -> StuartLandau(theta[1], theta[2])
        cfg = InferenceConfig(mk, sl_setup_fx.section, sl_setup_fx.x_eq,
                              [sl_setup_fx.x_eq], 256, [:radius, :period],
                              [2.0, 2 * pi])
        s_star, T_star = sl_ref_sol.s, sl_ref_sol.T
        Jim = observables_jacobian_implicit([SL_MU, SL_OMEGA], cfg, s_star, T_star)
        hfd = 1e-5
        Jfd = Matrix{Float64}(undef, 2, 2)
        for k in 1:2
            tp = [SL_MU, SL_OMEGA]; tp[k] += hfd
            tm = [SL_MU, SL_OMEGA]; tm[k] -= hfd
            fp = forward_solve(tp, cfg, s_star, T_star)
            fm = forward_solve(tm, cfg, s_star, T_star)
            @test fp.ok && fm.ok          # both sides are converged discrete solves
            Jfd[:, k] = (fp.obs .- fm.obs) ./ (2 * hfd)
        end
        @test maximum(abs.(Jim .- Jfd)) < 1e-4
    end

    # ------------------------------------------------------------------
    @testset "SL synthetic inference with seeded noise" begin
        mk = theta -> StuartLandau(theta[1], theta[2])
        cfg = InferenceConfig(mk, sl_setup_fx.section, sl_setup_fx.x_eq,
                              [sl_setup_fx.x_eq], 256, [:radius, :period],
                              [2.0, 2 * pi])
        target = observables_from_solution(cfg, sl_ref_sol)
        rng = MersenneTwister(20261019)
        sigma = 0.01
        z = randn(rng, 2)
        target_noisy = target .* (1 .+ sigma .* z)
        res = infer_parameters([0.35, 1.0], target_noisy, cfg;
                               s_warm=sl_ref_sol.s, T_warm=sl_ref_sol.T,
                               tol_mismatch=5 * sigma)
        @test res.status == :recovered          # 2x2 system interpolates exactly
        @test !res.mismatch                    # residual is at the noise level
        # linearized error propagation bounds (expected ~ 2*sigma*mu, |dT/dw|*sigma)
        @test abs(res.theta_hat[1] - SL_MU) < 0.06
        @test abs(res.theta_hat[2] - SL_OMEGA) < 0.8
        @test norm(res.final_obs .- target) <= 1.5 * sigma * norm(target)
    end

    # ------------------------------------------------------------------
    @testset "inconsistent observables are flagged, not silently fitted" begin
        mk = theta -> StuartLandau(theta[1], theta[2])
        cfg3 = InferenceConfig(mk, sl_setup_fx.section, sl_setup_fx.x_eq,
                               [sl_setup_fx.x_eq], 256,
                               [:radius, :period, :logmult], [2.0, 2 * pi, 4.0])
        target3 = observables_from_solution(cfg3, sl_ref_sol)
        @test isfinite(target3[3])
        # consistent overdetermined target lies on the model manifold
        res_ok = infer_parameters([0.4, 1.2], target3, cfg3;
                                 s_warm=sl_ref_sol.s, T_warm=sl_ref_sol.T)
        @test res_ok.status == :recovered
        @test !res_ok.mismatch
        @test norm(res_ok.theta_hat .- [SL_MU, SL_OMEGA]) < 1e-6
        # inconsistent target: multiplier squared (log-mult shifted off the
        # SL surface by |log lambda|): irreducible scaled residual
        target_bad = copy(target3)
        target_bad[3] = 2 * target3[3]
        res_bad = infer_parameters([0.4, 1.2], target_bad, cfg3;
                                   s_warm=sl_ref_sol.s, T_warm=sl_ref_sol.T)
        @test res_bad.status != :recovered
        @test res_bad.mismatch
        @test res_bad.residual_scaled >= 0.02
    end

    # ------------------------------------------------------------------
    @testset "vdp self-inference (1 parameter)" begin
        mk_v = theta -> VanDerPol(theta[1])
        cfg_v = InferenceConfig(mk_v, vdp_setup_fx.section, vdp_setup_fx.x_eq,
                                [vdp_setup_fx.x_eq], 256, [:radius, :period],
                                [2.0, 2 * pi]; theta_lo=[0.05], theta_hi=[8.0])
        target_v = observables_from_solution(cfg_v, vdp_ref_sol)
        resv = infer_parameters([0.7], target_v, cfg_v;
                               s_warm=vdp_ref_sol.s, T_warm=vdp_ref_sol.T)
        @test resv.status == :recovered
        @test abs(resv.theta_hat[1] - 1.0) < 1e-6
        Jim = observables_jacobian_implicit([1.0], cfg_v, vdp_ref_sol.s, vdp_ref_sol.T)
        @test !all(iszero, Jim)
        Jfd = observables_jacobian_fd([1.0], cfg_v, vdp_ref_sol.s, vdp_ref_sol.T;
                                      h_fd=1e-5)
        @test Jfd !== nothing
        @test maximum(abs.(Jim .- Jfd)) < 1e-4
        # noisy: overdetermined (2 obs, 1 param): residual stays at noise level
        rng = MersenneTwister(20261020)
        z = randn(rng, 2)
        target_noisy = target_v .* (1 .+ 0.01 .* z)
        resn = infer_parameters([0.7], target_noisy, cfg_v;
                                s_warm=vdp_ref_sol.s, T_warm=vdp_ref_sol.T,
                                tol_mismatch=0.05)
        @test resn.status in (:stalled, :recovered)
        @test !resn.mismatch
        @test abs(resn.theta_hat[1] - 1.0) < 0.15
    end

    # ------------------------------------------------------------------
    @testset "cross-model mismatch: vdp observables vs SL family" begin
        mk = theta -> StuartLandau(theta[1], theta[2])
        cfg2 = InferenceConfig(mk, sl_setup_fx.section, sl_setup_fx.x_eq,
                               [sl_setup_fx.x_eq], 256, [:radius, :period],
                               [2.0, 2 * pi])
        cfg3 = InferenceConfig(mk, sl_setup_fx.section, sl_setup_fx.x_eq,
                               [sl_setup_fx.x_eq], 256,
                               [:radius, :period, :logmult], [2.0, 2 * pi, 4.0])
        r_v = norm(vdp_ref_sol.x0 .- vdp_setup_fx.x_eq)
        T_v = vdp_ref_sol.T
        loglam_v = log(max(abs(vdp_ref_sol.floquet.multipliers[2]), 1e-300))
        # (radius, period) fit: SL family matches these exactly (bijection):
        # honest demonstration that (r, T) alone cannot identify the model class
        res2 = infer_parameters([1.0, 1.0], [r_v, T_v], cfg2;
                                s_warm=sl_ref_sol.s, T_warm=sl_ref_sol.T)
        @test res2.status == :recovered
        @test !res2.mismatch
        @test norm(res2.final_obs .- [r_v, T_v]) < 1e-6
        # adding the nontrivial multiplier exposes the model mismatch
        res3 = infer_parameters([1.0, 1.0], [r_v, T_v, loglam_v], cfg3;
                                s_warm=sl_ref_sol.s, T_warm=sl_ref_sol.T)
        @test res3.status != :recovered
        @test res3.mismatch
        @test res3.residual_scaled >= 0.02
    end

    # ------------------------------------------------------------------
    @testset "multiplier classification logic" begin
        f1 = classify_multipliers([1.0 0.0; 0.0 0.9])
        @test f1.neutral_count == 1 && f1.classification == :stable_cycle
        f2 = classify_multipliers([1.0 0.0; 0.0 1.4])
        @test f2.neutral_count == 1 && f2.classification == :unstable_cycle
        f3 = classify_multipliers([1.0 0.0; 0.0 1.0001])
        @test f3.neutral_count == 2 && f3.classification == :degenerate_neutral
        f4 = classify_multipliers([1.3 0.0; 0.0 0.5])
        @test f4.neutral_count == 0 && f4.classification == :no_neutral_multiplier
        # invariance residual: aligned vs misaligned f0
        fa = classify_multipliers([1.0 0.0; 0.0 0.3]; f0=[1.0, 0.0])
        @test fa.invariance_residual < 1e-12
        fb = classify_multipliers([1.0 0.0; 0.0 0.3]; f0=[0.0, 1.0])
        @test fb.invariance_residual > 0.5
    end

    # ------------------------------------------------------------------
    @testset "input validation" begin
        @test_throws ArgumentError Section([1.0, 0.0], [0.0, 0.0])
        @test_throws ArgumentError integrate_flow(sl_m, [1.0, 0.0], -1.0, 10)
        @test_throws ArgumentError integrate_flow(sl_m, [1.0, 0.0], 1.0, 0)
        mk = theta -> StuartLandau(theta[1], theta[2])
        @test_throws ArgumentError InferenceConfig(mk, sl_setup_fx.section,
                                                   sl_setup_fx.x_eq,
                                                   [sl_setup_fx.x_eq], 256,
                                                   [:bogus], [1.0])
        cfg = InferenceConfig(mk, sl_setup_fx.section, sl_setup_fx.x_eq,
                              [sl_setup_fx.x_eq], 256, [:radius, :period], [2.0, 2 * pi])
        @test_throws ArgumentError infer_parameters([0.5, 1.3], [0.0, 4.0], cfg;
                                                    s_warm=0.0, T_warm=4.0)
        @test_throws ArgumentError infer_parameters([0.5, 1.3], [0.7, -1.0], cfg;
                                                    s_warm=0.0, T_warm=4.0)
        @test_throws ArgumentError infer_parameters([0.5], [0.7, 4.0], cfg;
                                                    s_warm=0.0, T_warm=4.0)
        # T0 out of bounds is an honest explicit failure
        solb = solve_cycle(sl_m, sl_setup_fx.section, 0.0, 1e5, 256;
                          eqs=[sl_setup_fx.x_eq])
        @test solb.status == StepOutOfBounds
    end

    # ------------------------------------------------------------------
    @testset "artifact writers round-trip" begin
        d = mktempdir()
        try
            p = joinpath(d, "t.csv")
            write_csv(p, ["a", "b"], [[1.0, 2.5], [3.0, NaN]])
            lines = readlines(p)
            @test lines[1] == "a,b"
            @test length(lines) == 3
            @test occursin("NaN", lines[3])
            pt = joinpath(d, "t.toml")
            dd = Dict{String,Any}("x" => 1.5, "y" => [1, 2], "s" => "hi",
                                  "sub" => Dict{String,Any}("k" => true))
            write_toml(pt, dd)
            back = TOML.parsefile(pt)
            @test back["x"] == 1.5 && back["s"] == "hi" && back["sub"]["k"] == true
            @test back["y"] == [1, 2]
        finally
            rm(d; recursive=true, force=true)
        end
    end

end
println("\nexperiment 02 test suite finished")
