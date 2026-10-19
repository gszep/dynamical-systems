# Orchestrates the experiment-03 smoke run and writes all result artifacts.
#
# Stages: (1) multi-seed discovery at each r in cfg.r_disc; (2) primary-branch
# pseudo-arclength continuation (both directions) from the strongest accepted
# cosk k=1 discovery at r_fix; (3) mesh convergence (N = 32/64/128) at the
# fixed physical parameter r_fix for a large-amplitude and a small-amplitude
# seed, plus independent fold continuations per mesh; (4) artifact writing.
# Timings for queue/setup/execution stages are recorded separately.

const DISCOVERY_HEADER = ["r", "seed_id", "family", "kind", "tag", "converged", "accepted",
                          "is_new", "reason", "newton_iters", "n_deflated", "residual",
                          "rms", "mean_u", "kbar", "pr", "f0", "f1", "f2", "f3", "f4",
                          "nz50", "lambda_max_sub", "n_pos_sub", "lambda_max_full",
                          "n_pos_full"]

const BRANCH_HEADER = ["point_id", "direction", "s", "r", "rms", "linf", "kbar",
                       "lambda_max_sub", "n_pos_sub", "lambda_max_full", "n_pos_full",
                       "newton_iters", "ds"]

const MESH_HEADER = ["N", "r_fix", "seed_tag", "converged", "newton_iters", "residual",
                     "rms", "kbar", "lambda_max_sub", "n_pos_sub",
                     "err_u_vs_finest", "err_c16_vs_finest"]

const FOLD_HEADER = ["N", "seed_tag", "r_start", "r_fold", "s_fold", "n_points", "termination"]

const FAILURES_HEADER = ["stage", "tag", "direction", "r", "N", "step", "attempt", "ds",
                         "residual", "reason"]

# Continuation start preference: (1) cosk seed with k = 1 at r_fix, then (2)
# any accepted discovery at r_fix, then (3) any accepted discovery; always the
# largest rms (stays away from the trivial branch and the pitchfork).
function pick_continuation_start(discovery_all, cfg::SHConfig)
    best = nothing
    best_rms = -1.0
    for tier in 1:3
        for dres in discovery_all
            for rec in dres.records
                rec.accepted || continue
                tier_ok = false
                if tier == 1
                    tier_ok = (rec.spec.family == "cosk" && rec.spec.k == 1.0 &&
                               abs(rec.spec.r - cfg.r_fix) < 1.0e-12)
                elseif tier == 2
                    tier_ok = abs(rec.spec.r - cfg.r_fix) < 1.0e-12
                else
                    tier_ok = true
                end
                if tier_ok && rec.rms > best_rms
                    best_rms = rec.rms
                    best = rec
                end
            end
        end
        if best !== nothing
            break
        end
    end
    return best
end

"""
    run_smoke(cfg; results_dir, setup_seconds = 0.0, N_main = 64, Ns_mesh = [32,64,128])

Runs the full smoke experiment and writes discovery.csv, branch_primary.csv,
mesh_convergence.csv, fold_convergence.csv, failures.csv and
patterns_summary.toml into `results_dir`. Returns the summary dictionary.
"""
function run_smoke(cfg::SHConfig; results_dir::String, setup_seconds::Float64=0.0,
                   N_main::Int=64, Ns_mesh::Vector{Int}=Int[32, 64, 128])
    t_start = time_ns()
    fl = FailLog()
    mkpath(results_dir)
    L = cfg.L

    # ---- stage 1: multi-seed discovery ----
    t0 = time_ns()
    disc = Discretization(L, N_main)
    discovery_all = []
    for r in cfg.r_disc
        dres = run_discovery(disc, cfg, r; fl=fl)
        push!(discovery_all, dres)
    end
    discovery_seconds = (time_ns() - t0) / 1.0e9

    # ---- stage 2: primary-branch continuation ----
    t0 = time_ns()
    start = pick_continuation_start(discovery_all, cfg)
    branch_rows = Vector{Vector{Any}}()
    branch_info = Dict{String,Any}("status" => "skipped_no_start",
                                    "start_tag" => "-", "start_r" => "-", "start_rms" => "-")
    folds_all = Tuple{Float64,Float64,String,Int}[]
    if start !== nothing
        branch_info["status"] = "ok"
        branch_info["start_tag"] = start.spec.tag
        branch_info["start_r"] = toml_safe(start.spec.r)
        branch_info["start_rms"] = toml_safe(start.rms)
        pid = 0
        for dir_sign in (1, -1)
            br = continue_branch(disc, cfg, start.c, start.spec.r, dir_sign;
                                 fl=fl, tag="primary")
            for pt in br.points
                pid += 1
                u = disc.S * pt.c
                de = descriptors(disc, u)
                se = spectrum_even(disc, cfg, pt.c, pt.r)
                sf = spectrum_full(disc, cfg, u, pt.r)
                push!(branch_rows, Any[pid, dir_sign, pt.s, pt.r, pt.rms, maximum(abs, u),
                                       de.kbar, se.lambda_max, se.n_pos,
                                       sf.lambda_max, sf.n_pos, pt.newton_iters, pt.ds])
            end
            for f in br.folds
                push!(folds_all, (f[1], f[2], f[3], dir_sign))
            end
            branch_info["term_dir$(dir_sign)"] = br.term
            branch_info["npts_dir$(dir_sign)"] = length(br.points)
        end
        mins_down = [f for f in folds_all if f[3] == "min" && f[4] == -1]
        branch_info["folds_min_down"] =
            toml_safe(isempty(mins_down) ? NaN : minimum([f[2] for f in mins_down]))
        branch_info["n_folds_total"] = length(folds_all)
        branch_info["analytic_galerkin_fold_estimate"] = -9.0 * cfg.b^2 / 40.0
    else
        record!(fl, Any["continuation", "primary", "-", NaN, N_main, "-", "-", "-",
                        NaN, "no_start_solution"])
    end
    continuation_seconds = (time_ns() - t0) / 1.0e9

    # ---- stage 3: mesh convergence and fold convergence at r_fix ----
    t0 = time_ns()
    mesh_rows = Vector{Vector{Any}}()
    fold_rows = Vector{Vector{Any}}()
    mesh_stats = Dict{String,Any}()
    fin_sols = Dict{String,Any}()
    mesh_seeds = [("large", maximum(cfg.seedA_cos)), ("small", minimum(cfg.seedA_cos))]
    for (label, A) in mesh_seeds
        sols = Vector{Any}(undef, length(Ns_mesh))
        okall = true
        for (i, N) in enumerate(Ns_mesh)
            d = Discretization(L, N)
            u0 = [A * cos(1.0 * d.x[j]) for j in 1:N]
            res = newton_even(d, cfg, to_coeffs(d, u0), cfg.r_fix)
            sols[i] = (N=N, d=d, res=res)
            if !res.converged
                okall = false
                record!(fl, Any["mesh", "seed_$(label)_A$(A)_N$(N)", "-", cfg.r_fix, N,
                                "-", "-", "-", res.residual, "newton_" * res.reason])
            end
        end
        fin = sols[end]
        fin_sols[label] = fin
        ufin = to_values(fin.d, fin.res.c)
        nfin = norm(ufin)
        cfin = fin.res.c
        ncf = 16
        if minimum([s.d.Nc for s in sols]) < ncf
            ncf = minimum([s.d.Nc for s in sols])
        end
        errs_u = Vector{Float64}(undef, length(sols))
        errs_c = Vector{Float64}(undef, length(sols))
        for i in 1:length(sols)
            if i < length(sols)
                errs_u[i] = norm(evaluate_coeffs_on(fin.d, sols[i].res.c) .- ufin) / nfin
                errs_c[i] = norm(sols[i].res.c[1:ncf] .- cfin[1:ncf]) / norm(cfin[1:ncf])
            else
                errs_u[i] = NaN
                errs_c[i] = NaN
            end
        end
        for (i, s) in enumerate(sols)
            d = s.d
            u = to_values(d, s.res.c)
            de = descriptors(d, u)
            se = spectrum_even(d, cfg, s.res.c, cfg.r_fix)
            push!(mesh_rows, Any[s.N, cfg.r_fix, "seed_$(label)_A$(A)", s.res.converged,
                                 s.res.iters, s.res.residual, rms_of_c(d, s.res.c),
                                 de.kbar, se.lambda_max, se.n_pos, errs_u[i], errs_c[i]])
        end
        if okall
            for s in sols
                br = continue_branch(s.d, cfg, s.res.c, cfg.r_fix, -1;
                                     fl=fl, tag="foldmesh_$(label)_N$(s.N)")
                mins = [f for f in br.folds if f[3] == "min"]
                rf = isempty(mins) ? NaN : minimum([f[2] for f in mins])
                sfold = isempty(mins) ? NaN : minimum([f[1] for f in mins])
                push!(fold_rows, Any[s.N, "seed_$(label)_A$(A)", cfg.r_fix, rf, sfold,
                                     length(br.points), br.term])
            end
        end
        mesh_stats["seed_$(label)"] = Dict{String,Any}(
            "seed_amplitude" => A,
            "all_newton_converged" => okall,
            "newton_converged_by_N" => [s.res.converged for s in sols],
            "rms_by_N" => [fmt_val(rms_of_c(s.d, s.res.c)) for s in sols],
            "err_u_vs_finest_by_N" => [fmt_val(e) for e in errs_u],
            "err_c16_vs_finest_by_N" => [fmt_val(e) for e in errs_c],
        )
    end
    # morphology vs branch topology evidence at r_fix on the finest mesh
    mvt = Dict{String,Any}("available" => false)
    if haskey(fin_sols, "large") && haskey(fin_sols, "small")
        flg = fin_sols["large"]
        fsm = fin_sols["small"]
        if flg.res.converged && fsm.res.converged
            dl = descriptors(flg.d, to_values(flg.d, flg.res.c))
            dsm = descriptors(fsm.d, to_values(fsm.d, fsm.res.c))
            sl = spectrum_even(flg.d, cfg, flg.res.c, cfg.r_fix)
            ss = spectrum_even(fsm.d, cfg, fsm.res.c, cfg.r_fix)
            mvt = Dict{String,Any}(
                "available" => true,
                "kbar_large" => toml_safe(dl.kbar),
                "kbar_small" => toml_safe(dsm.kbar),
                "rms_large" => toml_safe(rms_of_c(flg.d, flg.res.c)),
                "rms_small" => toml_safe(rms_of_c(fsm.d, fsm.res.c)),
                "n_pos_subspace_large" => sl.n_pos,
                "n_pos_subspace_small" => ss.n_pos,
                "note" => "same fundamental-wavenumber morphology class, different branch segments (norm and even-sector stability); morphology descriptors do not encode branch topology",
            )
        end
    end
    mesh_stats["morphology_vs_topology"] = mvt
    mesh_seconds = (time_ns() - t0) / 1.0e9

    # ---- stage 4: artifacts ----
    t0 = time_ns()
    discovery_rows = Vector{Vector{Any}}()
    for dres in discovery_all
        append!(discovery_rows, dres.rows)
    end
    write_csv(joinpath(results_dir, "discovery.csv"), DISCOVERY_HEADER, discovery_rows)
    write_csv(joinpath(results_dir, "branch_primary.csv"), BRANCH_HEADER, branch_rows)
    write_csv(joinpath(results_dir, "mesh_convergence.csv"), MESH_HEADER, mesh_rows)
    write_csv(joinpath(results_dir, "fold_convergence.csv"), FOLD_HEADER, fold_rows)
    write_csv(joinpath(results_dir, "failures.csv"), FAILURES_HEADER, fl.rows)
    io_seconds = (time_ns() - t0) / 1.0e9

    total_seconds = (time_ns() - t_start) / 1.0e9

    # ---- summary ----
    summary = Dict{String,Any}()
    summary["meta"] = Dict{String,Any}(
        "experiment" => "03-patterns",
        "equation" => "u_t = r*u - (1 + d_xx)^2 u + b*u^3 - u^5 (b > 0 subcritical), periodic x in [0, L)",
        "julia_version" => string(VERSION),
        "machine" => string(Sys.MACHINE),
        "started_unix" => time(),
        "blas_threads" => 1,
        "threads_note" => "scripts pin BLAS to one thread; no Julia threads are spawned",
        "provenance_note" => "this code was written without execution (concurrent package setup); the numbers in this file are the first execution evidence from the run that wrote it",
    )
    summary["problem"] = Dict{String,Any}(
        "L" => L,
        "b" => cfg.b,
        "r_fix" => cfg.r_fix,
        "r_discovery" => cfg.r_disc,
        "N_main" => N_main,
        "Ns_mesh" => Ns_mesh,
        "discretization" => "Fourier pseudospectral collocation, dense exact DFT matrices, no dealiasing",
        "subspace" => "reflection-invariant (even/cosine) restriction; translation phase pinned by reflection gauge",
        "linear_spectrum_formula" => "eigenvalues of r*I - (1+d_xx)^2 at wavenumber k: r - (1 - k^2)^2",
    )
    summary["params"] = Dict{String,Any}(
        "tol_residual" => cfg.tol_residual,
        "max_newton" => cfg.max_newton,
        "max_backtrack" => cfg.max_backtrack,
        "ds0" => cfg.ds0,
        "ds_min" => cfg.ds_min,
        "ds_max" => cfg.ds_max,
        "steps_max" => cfg.steps_max,
        "r_stop_hi" => cfg.r_stop_hi,
        "r_stop_lo" => cfg.r_stop_lo,
        "rms_trivial" => cfg.rms_trivial,
        "rms_trivial_disc" => cfg.rms_trivial_disc,
        "rms_cap" => cfg.rms_cap,
        "dedup_tol" => cfg.dedup_tol,
        "eig_pos_threshold" => cfg.eig_pos_threshold,
        "seedA_cos" => cfg.seedA_cos,
        "seedK_cos" => cfg.seedK_cos,
        "seedA_loc" => cfg.seedA_loc,
        "seedW_loc" => cfg.seedW_loc,
        "seedC_loc" => cfg.seedC_loc,
        "random_seed_ids" => [Int(x) for x in cfg.random_seed_ids],
        "random_band_modes" => cfg.random_band_modes,
        "random_targets" => cfg.random_targets,
        "deflation" => "classical root-removing deflation M = prod 1/||c-c_i||^2 (scale-free), discovery only; convergence judged on the true residual",
    )
    dsum = Dict{String,Any}()
    for dres in discovery_all
        key = "r_" * replace(string(dres.r), "." => "p")
        fam = Dict{String,Any}()
        for f in ("cosk", "localized", "random", "trivial_probe", "highfreq")
            recs_f = [rec for rec in dres.records if rec.spec.family == f]
            fam[f] = Dict{String,Any}(
                "n_seeds" => length(recs_f),
                "n_converged" => count(rec -> rec.converged, recs_f),
                "n_accepted" => count(rec -> rec.accepted, recs_f),
                "n_new" => count(rec -> rec.is_new, recs_f),
            )
        end
        dsum[key] = Dict{String,Any}(
            "n_seeds" => length(dres.records),
            "n_converged" => count(rec -> rec.converged, dres.records),
            "n_accepted" => count(rec -> rec.accepted, dres.records),
            "n_new" => count(rec -> rec.is_new, dres.records),
            "n_not_converged" => count(rec -> !rec.converged, dres.records),
            "accepted_new_tags" => [rec.spec.tag for rec in dres.records if rec.is_new],
            "by_family" => fam,
        )
    end
    summary["discovery"] = dsum
    summary["branch_primary"] = branch_info
    summary["mesh"] = mesh_stats
    summary["timing"] = Dict{String,Any}(
        "queue_seconds" => 0.0,
        "queue_note" => "single local process; no queueing",
        "setup_seconds" => setup_seconds,
        "discovery_seconds" => discovery_seconds,
        "continuation_seconds" => continuation_seconds,
        "mesh_seconds" => mesh_seconds,
        "io_seconds" => io_seconds,
        "total_seconds" => total_seconds,
    )
    summary["failures"] = Dict{String,Any}("n_logged" => length(fl.rows))
    summary["scope"] = Dict{String,Any}(
        "completeness" => "no exhaustive atlas is claimed: fixed seed table, deflation-assisted Newton, failures recorded",
        "locality" => "single 1D configuration (L = 8*pi, b = 1); not 2D spots/stripes; no PDE inference",
        "snaking" => "no snaking claims: localized-seed outcomes are recorded as discovered, not asserted",
        "stability_claims" => "eigenvalues are reported for the reflection-invariant (even) subspace and, separately, the full space; the translation phase mode is absent from the even spectrum by construction; no full-space asymptotic-stability claim is made",
    )
    dump_toml(joinpath(results_dir, "patterns_summary.toml"), summary)
    return summary
end
