#!/usr/bin/env julia
# experiments/02-cycles/main.jl
#
# Experiment 02 (cycles) driver: phase-fixed periodic-orbit baselines,
# Floquet stability, Hopf detection and synthetic cycle parameter inference.
#
# Usage (from any directory; all paths are anchored to this file):
#   julia -t 1 experiments/02-cycles/main.jl                 # full run
#   julia -t 1 experiments/02-cycles/main.jl --quick          # smoke run
#   julia -t 1 experiments/02-cycles/main.jl --out-dir DIR    # custom artifact dir
#   julia -t 1 experiments/02-cycles/main.jl --only sl_reference
#
# Writes CSV/TOML artifacts into the output directory (default:
# <repo>/results/02-cycles) and prints a report. Each scenario declares an
# acceptance set for its outcome via d["accepted"]: positive scenarios must
# actually solve/recover, while explicitly negative scenarios are accepted
# only when the expected honest failure (guard rejection, non-convergence)
# or model-mismatch flag occurs. Exit code is 1 if a scenario crashes OR
# lands outside its acceptance set; expected negative outcomes are accepted,
# silent misses are not.
#
# Validation status: executed 2026-10-20 (single process, one Julia/BLAS/OMP
# thread); measured evidence in results/02-cycles/ and the README.

const _T_MAIN_START = time_ns()
include(joinpath(@__DIR__, "src", "CycleBaselines.jl"))
using .CycleBaselines
using LinearAlgebra
using Printf
using Random
import TOML

const _T_INCLUDE_END = time_ns()

# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

struct CliOptions
    out_dir::String
    quick::Bool
    seed::Int
    only::Vector{String}
end

function usage()
    return """
    usage: julia -t 1 main.jl [--out-dir DIR] [--quick] [--seed N]
                               [--only SCENARIO]

      --out-dir DIR   artifact directory (default: <repo>/results/02-cycles)
      --quick         reduced N sweep for smoke runs
      --seed N        RNG seed for the noisy-inference scenario (default 20261019)
      --only NAME     run a single scenario (repeatable)
    """
end

function parse_cli(args)
    out_dir = abspath(joinpath(@__DIR__, "..", "..", "results", "02-cycles"))
    quick = false
    seed = 20261019
    only = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--out-dir" && i < length(args)
            i += 1
            out_dir = abspath(args[i])
        elseif startswith(a, "--out-dir=")
            out_dir = abspath(a[(length("--out-dir=") + 1):end])
        elseif a == "--quick"
            quick = true
        elseif a == "--seed" && i < length(args)
            i += 1
            seed = parse(Int, args[i])
        elseif startswith(a, "--seed=")
            seed = parse(Int, a[(length("--seed=") + 1):end])
        elseif a == "--only" && i < length(args)
            i += 1
            push!(only, args[i])
        elseif startswith(a, "--only=")
            push!(only, a[(length("--only=") + 1):end])
        elseif a in ("-h", "--help")
            print(usage())
            exit(0)
        else
            error("unknown argument: " * a * "\n\n" * usage())
        end
        i += 1
    end
    return CliOptions(out_dir, quick, seed, only)
end

function git_head()
    try
        return readchomp(`git -C $(joinpath(@__DIR__, "..", "..")) rev-parse HEAD`)
    catch
        return "unknown"
    end
end

# -----------------------------------------------------------------------
# Shared scenario ingredients
# -----------------------------------------------------------------------

const SL_MU = 0.5
const SL_OMEGA = 1.3
const SL_IC = [1.7, 0.4]
const VDP_MU = 1.0
const VDP_IC = [1.7, 0.4]

sl_model() = StuartLandau(SL_MU, SL_OMEGA)
vdp_model() = VanDerPol(VDP_MU)

function sl_setup()
    return setup_cycle_scenario(sl_model(), SL_IC, 40.0, 4000; label="sl")
end

function vdp_setup()
    return setup_cycle_scenario(vdp_model(), VDP_IC, 80.0, 8000; label="vdp")
end

function record_solution!(d::Dict{String,Any}, prefix::AbstractString, sol::CycleSolution)
    sub = solution_summary_dict(sol)
    for (k, v) in sub
        d[prefix * "_" * k] = v
    end
    return d
end

# -----------------------------------------------------------------------
# Scenario 1: SL reference solve vs independent analytic reference
# -----------------------------------------------------------------------

function scenario_sl_reference(opts::CliOptions)
    N = 256
    d = Dict{String,Any}("N" => N, "mu" => SL_MU, "omega" => SL_OMEGA)
    setup = sl_setup()
    d["setup_ok"] = setup.ok
    d["setup_message"] = setup.message
    if !setup.ok
        d["status"] = "setup_failed"
        d["accepted"] = false
        return d
    end
    sol = solve_cycle(sl_model(), setup.section, setup.s0, setup.T0, N;
                      eqs=[setup.x_eq], model_label="stuart-landau",
                      note="reference solve from settle-based guess")
    record_solution!(d, "sol", sol)
    d["accepted"] = (sol.status == SolveSuccess)
    ref = analytic_stuart_landau(SL_MU, SL_OMEGA)
    d["ref_period"] = ref["period"]
    d["ref_radius"] = ref["radius"]
    d["ref_radial_multiplier"] = ref["radial_multiplier"]
    if sol.status == SolveSuccess
        r = norm(sol.x0 .- setup.x_eq)
        d["radius"] = r
        d["err_period"] = abs(sol.T - ref["period"])
        d["err_radius"] = abs(r - ref["radius"])
        if sol.floquet !== nothing
            d["err_radial_multiplier"] = abs(sol.floquet.multipliers[2] - ref["radial_multiplier"])
            d["err_neutral_multiplier"] = abs(sol.floquet.multipliers[1] - 1.0)
        end
        write_csv(joinpath(opts.out_dir, "sl_orbit_N$(N).csv"),
                  ["t", "x1", "x2"],
                  Any[[sol.ts[j], sol.xs[1, j], sol.xs[2, j]] for j in 1:length(sol.ts)])
    end
    d["status"] = string(sol.status)
    return d
end

# -----------------------------------------------------------------------
# Scenario 2: SL time-step convergence (observed RK4 order)
# -----------------------------------------------------------------------

function scenario_sl_convergence(opts::CliOptions)
    Ns = opts.quick ? [64, 128, 256] : [32, 64, 128, 256, 512]
    d = Dict{String,Any}("Ns" => Ns, "mu" => SL_MU, "omega" => SL_OMEGA)
    setup = sl_setup()
    d["setup_ok"] = setup.ok
    if !setup.ok
        d["status"] = "setup_failed"
        d["accepted"] = false
        return d
    end
    ref = analytic_stuart_landau(SL_MU, SL_OMEGA)
    s_warm, T_warm = setup.s0, setup.T0
    rows = Any[]
    prev = nothing
    all_solved = true
    for N in Ns
        sol = solve_cycle(sl_model(), setup.section, s_warm, T_warm, N;
                          eqs=[setup.x_eq], model_label="sl-conv")
        all_solved &= (sol.status == SolveSuccess)
        row = Any[N]
        if sol.status == SolveSuccess
            s_warm, T_warm = sol.s, sol.T
            r = norm(sol.x0 .- setup.x_eq)
            errT = abs(sol.T - ref["period"])
            errR = abs(r - ref["radius"])
            errL = abs(sol.floquet.multipliers[2] - ref["radial_multiplier"])
            devN = abs(sol.floquet.multipliers[1] - 1.0)
            push!(row, sol.T, r, errT, errR, errL, devN)
            if prev !== nothing
                for (a, b) in zip(prev[4:7], (errT, errR, errL, devN))
                    push!(row, (a > 0 && b > 0) ? log(a / b) / log(2) : NaN)
                end
            else
                append!(row, [NaN, NaN, NaN, NaN])
            end
            prev = Any[N, sol.T, r, errT, errR, errL, devN]
        else
            append!(row, [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN])
        end
        push!(rows, row)
    end
    header = ["N", "T", "radius", "err_T", "err_radius", "err_radial_mult",
              "dev_neutral", "p_T", "p_radius", "p_radial_mult", "p_dev_neutral"]
    write_csv(joinpath(opts.out_dir, "sl_convergence.csv"), header, rows)
    d["status"] = "written"
    d["accepted"] = all_solved
    return d
end

# -----------------------------------------------------------------------
# Scenario 3: Hopf detection + supercritical amplitude law + degeneracy
# -----------------------------------------------------------------------

function scenario_hopf_amplitude(opts::CliOptions)
    d = Dict{String,Any}()
    h = detect_hopf(p -> StuartLandau(p, SL_OMEGA), -0.5, 0.5; jac=:fd)
    d["hopf_ok"] = h.ok
    d["hopf_mu"] = h.p_h
    d["hopf_transversality"] = h.transversality
    d["hopf_iters"] = h.iters
    d["hopf_message"] = h.message
    d["hopf_max_real_at_h"] = h.max_real_at_h
    d["hopf_eigs_real"] = [real(z) for z in h.eigenvalues]
    d["hopf_eigs_imag"] = [imag(z) for z in h.eigenvalues]

    # amplitude law r(mu) ~ sqrt(mu), approached by settling at mu=1e-2 and
    # continuation for smaller mu (the settle timescale 1/mu becomes
    # impractical; continuation reuses the radial section, which crosses
    # every SL cycle). Warm start from the previous solution.
    mus = opts.quick ? [1e-2, 1e-4] : [1e-2, 1e-4, 1e-6]
    rows = Any[]
    section = nothing
    s_warm = T_warm = NaN
    amp_ok = true
    for mu in mus
        local sol
        if mu == mus[1]
            m = StuartLandau(mu, SL_OMEGA)
            st = setup_cycle_scenario(m, SL_IC, 600.0, 12000; label="amp-law")
            sol = st.ok ? solve_cycle(m, st.section, st.s0, st.T0, 256;
                                      eqs=[st.x_eq], model_label="sl-amp") :
                          nothing
            section = st.ok ? st.section : nothing
            if sol !== nothing && sol.status == SolveSuccess
                s_warm, T_warm = sol.s, sol.T
            end
        else
            m = StuartLandau(mu, SL_OMEGA)
            sol = (section !== nothing) ?
                  solve_cycle(m, section, s_warm, T_warm, 256; eqs=[[0.0, 0.0]],
                              model_label="sl-amp-cont") : nothing
            if sol !== nothing && sol.status == SolveSuccess
                s_warm, T_warm = sol.s, sol.T
            end
        end
        amp_ok &= (sol !== nothing && sol.status == SolveSuccess)
        r_ref = sqrt(mu)
        if sol !== nothing && sol.status == SolveSuccess
            r = norm(sol.x0 .- [0.0, 0.0])
            push!(rows, Any[mu, r, r_ref, r / r_ref,
                            sol.floquet.gap_to_unit,
                            String(sol.floquet.classification),
                            sol.diagnostics["near_degenerate"],
                            string(sol.status), sol.iters])
        else
            push!(rows, Any[mu, NaN, r_ref, NaN, NaN, "n/a", "n/a",
                            sol === nothing ? "no_setup" : string(sol.status), 0])
        end
    end
    write_csv(joinpath(opts.out_dir, "sl_amplitude_law.csv"),
              ["mu", "radius", "radius_analytic", "ratio", "gap_to_unit",
               "classification", "near_degenerate", "status", "iters"], rows)
    d["status"] = "written"
    # positive scenario: the Hopf bracket must succeed AND every amplitude-
    # law continuation solve must find its cycle
    d["accepted"] = h.ok && amp_ok
    return d
end

# -----------------------------------------------------------------------
# Scenario 4: negative cases mu <= 0 (honest failure, not fake cycles)
# -----------------------------------------------------------------------

function scenario_negative_mu(opts::CliOptions)
    rows = Any[]
    d = Dict{String,Any}()
    # expected honest-failure statuses for mu <= 0: guard rejection, Newton
    # non-convergence, or an out-of-bounds period. A SolveSuccess here would
    # be a fake cycle and must FAIL the audit.
    expected_fail = (string(GuardRejectedEquilibrium), string(NewtonNotConverged),
                     string(StepOutOfBounds))
    neg_ok = true
    for mu in (0.0, -0.5)
        m = StuartLandau(mu, SL_OMEGA)
        st = setup_cycle_scenario(m, SL_IC, 40.0, 4000; label="neg-mu=$mu")
        if !st.ok
            # settling collapsed onto the equilibrium: no section is an
            # honest negative outcome
            push!(rows, Any[mu, false, st.message, "no_setup", NaN, 0])
        else
            sol = solve_cycle(m, st.section, st.s0, st.T0, 256;
                              eqs=[st.x_eq], model_label="sl-neg")
            push!(rows, Any[mu, true, "settle ok", string(sol.status),
                            sol.resnorm, sol.iters])
            neg_ok &= (string(sol.status) in expected_fail)
        end
    end
    write_csv(joinpath(opts.out_dir, "sl_negative_mu.csv"),
              ["mu", "setup_ok", "setup_message", "solve_status", "resnorm", "iters"],
              rows)
    d["note"] = "absence of a found cycle is a local search outcome, NOT a proof of nonexistence"
    d["status"] = "written"
    # explicitly expected negative case: accepted iff every row failed honestly
    d["accepted"] = neg_ok
    return d
end

# -----------------------------------------------------------------------
# Scenario 5: SL synthetic inference (noiseless, cross-N, noisy) + gradients
# -----------------------------------------------------------------------

function sl_inference_cfg(N::Int, obs, scales)
    setup = sl_setup()
    mk = theta -> StuartLandau(theta[1], theta[2])
    cfg = InferenceConfig(mk, setup.section, setup.x_eq, [setup.x_eq], N,
                          obs, scales; theta_lo=[1e-4, 0.05], theta_hi=[10.0, 20.0])
    return setup, cfg
end

function scenario_inference(opts::CliOptions)
    d = Dict{String,Any}()
    N_fit = 256
    setup, cfg = sl_inference_cfg(N_fit, [:radius, :period], [2.0, 2 * pi])
    ref_sol = solve_cycle(sl_model(), setup.section, setup.s0, setup.T0, N_fit;
                          eqs=[setup.x_eq], model_label="sl-ref-for-inference")
    d["reference_status"] = string(ref_sol.status)
    if ref_sol.status != SolveSuccess
        d["status"] = "reference_failed"
        d["accepted"] = false
        return d
    end
    s_star, T_star = ref_sol.s, ref_sol.T
    target = observables_from_solution(cfg, ref_sol)
    d["target_radius"] = target[1]
    d["target_period"] = target[2]
    theta_true = [SL_MU, SL_OMEGA]
    theta0 = [0.35, 1.0]

    # noiseless, same N: exact recovery expected
    res = infer_parameters(theta0, target, cfg; s_warm=s_star, T_warm=T_star)
    d["noiseless_status"] = string(res.status)
    d["noiseless_theta_hat"] = collect(res.theta_hat)
    d["noiseless_err"] = norm(res.theta_hat .- theta_true)
    d["noiseless_iters"] = res.iters
    d["noiseless_residual"] = res.residual_scaled

    # cross-N: target computed at N=512, fit at N=256 (discretization-limited)
    setup512, cfg512 = sl_inference_cfg(512, [:radius, :period], [2.0, 2 * pi])
    fwd512 = forward_solve(theta_true, cfg512, s_star, T_star)
    cross_ok = false
    if fwd512.ok
        res2 = infer_parameters(theta0, fwd512.obs, cfg; s_warm=s_star, T_warm=T_star)
        d["crossN_status"] = string(res2.status)
        d["crossN_theta_hat"] = collect(res2.theta_hat)
        d["crossN_err"] = norm(res2.theta_hat .- theta_true)
        cross_ok = (res2.status == :recovered)
    else
        d["crossN_status"] = "forward_failed: " * fwd512.message
    end

    # noisy: seeded multiplicative noise on the observables
    rng = MersenneTwister(opts.seed)
    sigma = 0.01
    z = randn(rng, 2)
    target_noisy = target .* (1 .+ sigma .* z)
    d["noise_sigma"] = sigma
    d["noise_z"] = collect(z)
    resn = infer_parameters(theta0, target_noisy, cfg; s_warm=s_star, T_warm=T_star,
                            tol_mismatch=5 * sigma)
    d["noisy_status"] = string(resn.status)
    d["noisy_theta_hat"] = collect(resn.theta_hat)
    d["noisy_err"] = norm(resn.theta_hat .- theta_true)
    d["noisy_propagated_obs_err"] = norm(resn.final_obs .- target)
    d["noisy_mismatch"] = resn.mismatch

    # gradient cross-checks at the reference solution
    Jim = observables_jacobian_implicit(theta_true, cfg, s_star, T_star)
    Jfd = observables_jacobian_fd(theta_true, cfg, s_star, T_star; h_fd=1e-5)
    d["grad_implicit"] = vec(Jim)   # column-major: [dr/dmu, dT/dmu, dr/domega, dT/domega]
    d["grad_implicit_vs_fd_maxabs"] = Jfd === nothing ? NaN :
                                      maximum(abs.(Jim .- Jfd))
    d["grad_analytic"] = [1 / (2 * sqrt(SL_MU)), 0.0, 0.0, -2 * pi / SL_OMEGA^2]
    d["grad_implicit_vs_analytic_maxabs"] = maximum(abs.(vec(Jim) .- d["grad_analytic"]))

    # trace artifact
    write_csv(joinpath(opts.out_dir, "sl_inference_trace.csv"),
              ["iter", "mu", "omega", "residual_scaled"],
              Any[[k - 1, res.trace_thetas[k][1], res.trace_thetas[k][2],
                   k <= length(res.trace_residuals) ? res.trace_residuals[k] : NaN]
                  for k in 1:length(res.trace_thetas)])
    d["status"] = "written"
    # positive scenario: reference solve and all three fits must recover,
    # the noisy fit must not be flagged as mismatched, and the implicit-vs-FD
    # gradient cross-check must have been computable
    d["accepted"] = (res.status == :recovered) && cross_ok &&
                    (resn.status == :recovered) && !resn.mismatch &&
                    (Jfd !== nothing)
    return d
end

# -----------------------------------------------------------------------
# Scenario 6: inconsistent observable (impossible Floquet multiplier)
# -----------------------------------------------------------------------

function scenario_inconsistent(opts::CliOptions)
    d = Dict{String,Any}()
    N = 256
    setup, cfg3 = sl_inference_cfg(N, [:radius, :period, :logmult], [2.0, 2 * pi, 4.0])
    ref_sol = solve_cycle(sl_model(), setup.section, setup.s0, setup.T0, N;
                          eqs=[setup.x_eq], model_label="sl-ref-3obs")
    if ref_sol.status != SolveSuccess
        d["status"] = "reference_failed"
        d["accepted"] = false
        return d
    end
    s_star, T_star = ref_sol.s, ref_sol.T
    target3 = observables_from_solution(cfg3, ref_sol)
    d["consistent_target"] = collect(target3)

    # consistent target: overdetermined but on the model manifold -> exact fit
    res_ok = infer_parameters([0.4, 1.2], target3, cfg3; s_warm=s_star, T_warm=T_star)
    d["consistent_status"] = string(res_ok.status)
    d["consistent_theta_hat"] = collect(res_ok.theta_hat)
    d["consistent_residual"] = res_ok.residual_scaled

    # inconsistent target: square the multiplier (log-mult off by |log lambda|),
    # which no Stuart-Landau (mu, omega) can reproduce together with (r, T)
    target_bad = copy(target3)
    target_bad[3] = 2 * target3[3]
    d["inconsistent_target_logmult"] = target_bad[3]
    res_bad = infer_parameters([0.4, 1.2], target_bad, cfg3; s_warm=s_star, T_warm=T_star)
    d["inconsistent_status"] = string(res_bad.status)
    d["inconsistent_theta_hat"] = collect(res_bad.theta_hat)
    d["inconsistent_residual"] = res_bad.residual_scaled
    d["inconsistent_mismatch"] = res_bad.mismatch
    d["status"] = "written"
    # expected-negative case: the consistent triple must be recovered
    # exactly, while the inconsistent triple must NOT be silently recovered
    # and must be flagged as a mismatch
    d["accepted"] = (res_ok.status == :recovered) &&
                    (res_bad.status != :recovered) && res_bad.mismatch
    return d
end

# -----------------------------------------------------------------------
# Scenario 7: vdp structural solve (no analytic reference exists)
# -----------------------------------------------------------------------

function scenario_vdp_reference(opts::CliOptions)
    N = 256
    d = Dict{String,Any}("N" => N, "mu_vdp" => VDP_MU)
    setup = vdp_setup()
    d["setup_ok"] = setup.ok
    d["setup_message"] = setup.message
    if !setup.ok
        d["status"] = "setup_failed"
        d["accepted"] = false
        return d
    end
    sol = solve_cycle(vdp_model(), setup.section, setup.s0, setup.T0, N;
                      eqs=[setup.x_eq], model_label="van-der-pol",
                      note="structural checks only; no closed form exists")
    record_solution!(d, "sol", sol)
    d["accepted"] = (sol.status == SolveSuccess)
    if sol.status == SolveSuccess
        # cross-grid recheck: the finer map must nearly close at the found point
        fl = integrate_flow(vdp_model(), sol.x0, sol.T, 512)
        d["crossgrid_gap_512"] = norm(fl.xT .- sol.x0)
        write_csv(joinpath(opts.out_dir, "vdp_orbit_N$(N).csv"),
                  ["t", "x1", "x2"],
                  Any[[sol.ts[j], sol.xs[1, j], sol.xs[2, j]] for j in 1:length(sol.ts)])
    end
    d["status"] = string(sol.status)
    return d
end

# -----------------------------------------------------------------------
# Scenario 8: vdp Richardson self-convergence (no analytic truth available)
# -----------------------------------------------------------------------

function scenario_vdp_convergence(opts::CliOptions)
    Ns = opts.quick ? [64, 128, 256] : [64, 128, 256, 512]
    d = Dict{String,Any}("Ns" => Ns, "mu_vdp" => VDP_MU)
    setup = vdp_setup()
    d["setup_ok"] = setup.ok
    if !setup.ok
        d["status"] = "setup_failed"
        d["accepted"] = false
        return d
    end
    s_warm, T_warm = setup.s0, setup.T0
    rows = Any[]
    Ts = Float64[]
    all_solved = true
    for N in Ns
        sol = solve_cycle(vdp_model(), setup.section, s_warm, T_warm, N;
                          eqs=[setup.x_eq], model_label="vdp-conv")
        all_solved &= (sol.status == SolveSuccess)
        if sol.status == SolveSuccess
            s_warm, T_warm = sol.s, sol.T
            push!(Ts, sol.T)
            push!(rows, Any[N, sol.T, sol.x0[1], sol.x0[2],
                            sol.floquet.multipliers[2].re,
                            sol.floquet.multipliers[2].im,
                            sol.floquet.gap_to_unit,
                            abs(sol.floquet.multipliers[1] - 1.0), string(sol.status)])
        else
            push!(rows, Any[N, NaN, NaN, NaN, NaN, NaN, NaN, NaN, string(sol.status)])
        end
    end
    # Richardson: successive differences should shrink by ~2^4 for RK4
    for k in 1:(length(Ts) - 2)
        d1 = abs(Ts[k + 1] - Ts[k])
        d2 = abs(Ts[k + 2] - Ts[k + 1])
        d["richardson_ratio_N$(Ns[k + 1])"] = d2 > 0 ? d1 / d2 : NaN
    end
    write_csv(joinpath(opts.out_dir, "vdp_convergence.csv"),
              ["N", "T", "x0_1", "x0_2", "mult2_re", "mult2_im", "gap_to_unit",
               "dev_neutral", "status"], rows)
    d["status"] = "written"
    # positive scenario: every grid must solve and enough periods must be
    # available for the Richardson ratios
    d["accepted"] = all_solved && (length(Ts) >= length(Ns) - 1)
    return d
end

# -----------------------------------------------------------------------
# Scenario 9: vdp self-inference + cross-model mismatch (vdp -> SL family)
# -----------------------------------------------------------------------

function scenario_vdp_inference_crossmodel(opts::CliOptions)
    d = Dict{String,Any}()
    N = 256
    setup = vdp_setup()
    d["setup_ok"] = setup.ok
    if !setup.ok
        d["status"] = "setup_failed"
        d["accepted"] = false
        return d
    end
    sol = solve_cycle(vdp_model(), setup.section, setup.s0, setup.T0, N;
                      eqs=[setup.x_eq], model_label="vdp-ref")
    if sol.status != SolveSuccess
        d["status"] = "vdp_reference_failed: " * sol.message
        d["accepted"] = false
        return d
    end
    mk_vdp = theta -> VanDerPol(theta[1])
    cfg_v = InferenceConfig(mk_vdp, setup.section, setup.x_eq, [setup.x_eq], N,
                            [:radius, :period], [2.0, 2 * pi];
                            theta_lo=[0.05], theta_hi=[8.0])
    s_v, T_v = sol.s, sol.T
    target_v = observables_from_solution(cfg_v, sol)
    d["vdp_target"] = collect(target_v)

    # noiseless self-inference: exact recovery expected
    resv = infer_parameters([0.7], target_v, cfg_v; s_warm=s_v, T_warm=T_v)
    d["vdp_self_status"] = string(resv.status)
    d["vdp_self_theta_hat"] = collect(resv.theta_hat)
    d["vdp_self_err"] = abs(resv.theta_hat[1] - VDP_MU)

    # implicit vs FD gradient on the vdp family
    Jim_v = observables_jacobian_implicit([VDP_MU], cfg_v, s_v, T_v)
    Jfd_v = observables_jacobian_fd([VDP_MU], cfg_v, s_v, T_v; h_fd=1e-5)
    d["vdp_grad_implicit"] = vec(Jim_v)
    d["vdp_grad_vs_fd_maxabs"] = Jfd_v === nothing ? NaN : maximum(abs.(Jim_v .- Jfd_v))

    # noisy self-inference (seeded)
    rng = MersenneTwister(opts.seed + 1)
    z = randn(rng, 2)
    target_v_noisy = target_v .* (1 .+ 0.01 .* z)
    resvn = infer_parameters([0.7], target_v_noisy, cfg_v; s_warm=s_v, T_warm=T_v,
                             tol_mismatch=0.05)
    d["vdp_noisy_status"] = string(resvn.status)
    d["vdp_noisy_theta_hat"] = collect(resvn.theta_hat)
    d["vdp_noisy_err"] = abs(resvn.theta_hat[1] - VDP_MU)
    d["vdp_noisy_mismatch"] = resvn.mismatch

    # cross-model: SL family fitted to vdp observables
    sl_setup2, cfg2 = sl_inference_cfg(N, [:radius, :period], [2.0, 2 * pi])
    _, cfg3 = sl_inference_cfg(N, [:radius, :period, :logmult], [2.0, 2 * pi, 4.0])
    lam2 = sol.floquet.multipliers[2]
    loglam_v = log(max(abs(lam2), 1e-300))
    d["vdp_logmult"] = loglam_v
    # warm start from the SL reference solve at the frozen SL section
    ref_sl = solve_cycle(sl_model(), sl_setup2.section, sl_setup2.s0, sl_setup2.T0, N;
                         eqs=[sl_setup2.x_eq], model_label="sl-warm-for-crossmodel")
    if ref_sl.status != SolveSuccess
        d["status"] = "sl_warm_failed"
        d["accepted"] = false
        return d
    end
    sw, Tw = ref_sl.s, ref_sl.T
    # (radius, period) fit: the 2-parameter SL family can match these (bijection)
    res2 = infer_parameters([1.0, 1.0], target_v, cfg2; s_warm=sw, T_warm=Tw)
    d["crossmodel_2obs_status"] = string(res2.status)
    d["crossmodel_2obs_theta_hat"] = collect(res2.theta_hat)
    d["crossmodel_2obs_residual"] = res2.residual_scaled
    d["crossmodel_2obs_mismatch"] = res2.mismatch
    # (radius, period, logmult) fit: vdp's multiplier is not on the SL surface
    res3 = infer_parameters([1.0, 1.0], [target_v[1], target_v[2], loglam_v], cfg3;
                            s_warm=sw, T_warm=Tw)
    d["crossmodel_3obs_status"] = string(res3.status)
    d["crossmodel_3obs_theta_hat"] = collect(res3.theta_hat)
    d["crossmodel_3obs_residual"] = res3.residual_scaled
    d["crossmodel_3obs_mismatch"] = res3.mismatch
    d["status"] = "written"
    # mixed positive/expected-negative scenario: the vdp self-inference and
    # the underdetermined (radius, period) cross-model fit must recover; the
    # overdetermined 3-observable cross-model fit must NOT be silently
    # recovered and must be flagged as a model mismatch
    d["accepted"] = (resv.status == :recovered) &&
                    (resvn.status in (:recovered, :stalled)) && !resvn.mismatch &&
                    (res2.status == :recovered) && !res2.mismatch &&
                    (res3.status != :recovered) && res3.mismatch
    return d
end

# -----------------------------------------------------------------------
# Scenario 10: spurious equilibrium roots: guard + deflation fixtures
# -----------------------------------------------------------------------

function scenario_spurious_roots(opts::CliOptions)
    d = Dict{String,Any}()
    N = 256
    eq0 = [0.0, 0.0]

    # --- SL fixture: anchor exactly on the analytic cycle => the radial
    # section contains the origin, so the equilibrium IS a spurious root.
    m = StuartLandau(SL_MU, SL_OMEGA)
    xa = [sqrt(SL_MU), 0.0]
    sec = Section(xa, rhs(m, xa); label="sl-through-origin")
    d["sl_fixture_eq_on_section"] = abs(dot(eq0 .- xa, sec.va))
    s0 = project_to_section(sec, [0.01, 0.0])
    T0 = 2 * pi / SL_OMEGA
    plain = solve_cycle(m, sec, s0, T0, N; eqs=[eq0], deflate=false,
                        model_label="sl-plain-seduction")
    defl = solve_cycle(m, sec, s0, T0, N; eqs=[eq0], deflate=true,
                       model_label="sl-deflated")
    record_solution!(d, "sl_plain", plain)
    record_solution!(d, "sl_defl", defl)
    if defl.status == SolveSuccess
        d["sl_defl_err_radius"] = abs(norm(defl.x0) - sqrt(SL_MU))
        d["sl_defl_err_period"] = abs(defl.T - 2 * pi / SL_OMEGA)
    end

    # --- vdp fixture: anchor on the x-axis => section is the x-axis through
    # the origin. Start deep in the linear regime, period-aligned, so plain
    # Newton is attracted to the equilibrium root.
    mv = VanDerPol(VDP_MU)
    xav = [2.0, 0.0]
    secv = Section(xav, rhs(mv, xav); label="vdp-through-origin")
    d["vdp_fixture_eq_on_section"] = abs(dot(eq0 .- xav, secv.va))
    s0v = project_to_section(secv, [0.005, 0.0])
    T0v = 4 * pi / sqrt(3.0)
    plainv = solve_cycle(mv, secv, s0v, T0v, N; eqs=[eq0], deflate=false,
                         model_label="vdp-plain-seduction")
    deflv = solve_cycle(mv, secv, s0v, T0v, N; eqs=[eq0], deflate=true,
                        max_iter=80, model_label="vdp-deflated")
    record_solution!(d, "vdp_plain", plainv)
    record_solution!(d, "vdp_defl", deflv)
    # period must be phase-independent: compare with the settle-based section
    setup = vdp_setup()
    if setup.ok
        sol_ref = solve_cycle(mv, setup.section, setup.s0, setup.T0, N;
                              eqs=[setup.x_eq], model_label="vdp-ref-section")
        record_solution!(d, "vdp_ref_section", sol_ref)
        if sol_ref.status == SolveSuccess && deflv.status == SolveSuccess
            d["vdp_period_diff_sections"] = abs(deflv.T - sol_ref.T)
            d["vdp_mult2_diff_sections"] = abs(abs(deflv.floquet.multipliers[2]) -
                                               abs(sol_ref.floquet.multipliers[2]))
        end
    end
    d["status"] = "written"
    # mixed positive/expected-negative: plain Newton must be seduced to the
    # equilibrium and guard-rejected (SL: deterministically; vdp: honest
    # failure statuses), deflated Newton must find the genuine SL cycle and
    # must never land on the deflated equilibrium for vdp
    d["accepted"] = (plain.status == GuardRejectedEquilibrium) &&
                    (defl.status == SolveSuccess) &&
                    (plainv.status in (GuardRejectedEquilibrium, NewtonNotConverged)) &&
                    (deflv.status != GuardRejectedEquilibrium)
    return d
end

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

function main()
    opts = parse_cli(ARGS)
    mkpath(opts.out_dir)
    t_main0 = time_ns()
    scenarios = [
        ("sl_reference", scenario_sl_reference),
        ("sl_convergence", scenario_sl_convergence),
        ("hopf_amplitude", scenario_hopf_amplitude),
        ("negative_mu", scenario_negative_mu),
        ("inference", scenario_inference),
        ("inconsistent_observable", scenario_inconsistent),
        ("vdp_reference", scenario_vdp_reference),
        ("vdp_convergence", scenario_vdp_convergence),
        ("vdp_inference_crossmodel", scenario_vdp_inference_crossmodel),
        ("spurious_roots", scenario_spurious_roots),
    ]
    results = Dict{String,Any}()
    exec_times = Dict{String,Float64}()
    crashed = String[]
    for (name, fn) in scenarios
        if !isempty(opts.only) && !(name in opts.only)
            continue
        end
        t0 = time_ns()
        local d
        try
            d = fn(opts)
        catch err
            d = Dict{String,Any}("status" => "error",
                                 "error" => sprint(showerror, err),
                                 "accepted" => false)
            push!(crashed, name)
        end
        exec_times[name] = (time_ns() - t0) / 1e9
        d["exec_seconds"] = exec_times[name]
        results[name] = d
        @printf("  %-28s %-10s %-8s %8.3fs\n", name,
                string(get(d, "status", "?")),
                get(d, "accepted", false) ? "accept" : "REJECT", exec_times[name])
    end

    # Scenario acceptance audit: every scenario declares, via d["accepted"],
    # whether its outcome is within its expected set (positive scenarios must
    # actually solve/recover; explicitly negative scenarios are accepted only
    # when the expected honest failure or mismatch flag occurred). A scenario
    # that silently misses its expected outcome fails the run, not only a
    # crash does.
    failed_acceptance = String[]
    for name in sort(collect(keys(results)))
        if get(results[name], "accepted", false) !== true
            push!(failed_acceptance, name)
        end
    end

    total_s = (time_ns() - t_main0) / 1e9
    setup_s = (_T_INCLUDE_END - _T_MAIN_START) / 1e9
    summary = Dict{String,Any}(
        "experiment" => "02-cycles",
        "generated_by" => "experiments/02-cycles/main.jl",
        "git_head" => git_head(),
        "julia_version" => string(VERSION),
        "arch" => string(Sys.ARCH),
        "kernel" => string(Sys.KERNEL),
        "threads" => Threads.nthreads(),
        "seed" => opts.seed,
        "quick" => opts.quick,
        "out_dir" => opts.out_dir,
        "args" => collect(ARGS),
        "timings" => Dict{String,Any}(
            "queue_seconds" => 0.0,
            "queue_note" => "single-process run; no queueing",
            "setup_seconds" => setup_s,
            "scenario_exec_seconds" => exec_times,
            "total_seconds" => total_s,
        ),
        "tolerances" => Dict{String,Any}(
            "tol_newton" => 1e-11,
            "tol_neutral" => 5e-4,
            "guard_rtol" => 1e-6,
            "guard_ftol" => 1e-6,
            "note" => "discrete residual is driven to ~1e-12; accuracy vs the continuous problem is O(h^4), reported by the convergence studies",
        ),
        "audit" => Dict{String,Any}(
            "all_accepted" => (isempty(failed_acceptance) && isempty(crashed)),
            "failed_acceptance" => failed_acceptance,
            "crashed" => crashed,
            "note" => "exit code is 1 on any scenario crash OR any scenario outcome outside its declared acceptance set; expected negative outcomes (guard rejections, honest non-convergence, model-mismatch flags) are accepted",
        ),
        "scenarios" => results,
    )
    write_toml(joinpath(opts.out_dir, "summary.toml"), summary)

    println("\nExperiment 02 (cycles) report")
    println("  artifacts: ", opts.out_dir)
    println("  total: ", round(total_s; digits=3), "s (setup ",
            round(setup_s; digits=3), "s)")
    println("  audit: ", isempty(failed_acceptance) ?
            "all scenarios within acceptance set" :
            "REJECTED scenarios: " * join(failed_acceptance, ", "))
    if !isempty(crashed)
        println("  CRASHED scenarios: ", join(crashed, ", "))
    end
    println()
    if !isempty(crashed) || !isempty(failed_acceptance)
        exit(1)
    end
    return nothing
end

main()
