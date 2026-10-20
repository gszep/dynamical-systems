# main.jl — experiment 01 (equilibria) probe driver (stdlib only)
#
# Runs the dependency-light mathematical probes and writes small TOML/CSV
# artifacts to results/01-equilibria/. Run with the pinned runtime:
#
#     JULIA_DEPOT_PATH=<worktree>/.julia-depot JULIA_NUM_THREADS=1 \
#     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 <pinned julia> main.jl
#
# Paper: arXiv:2106.04243v3 (LaTeX source read; exact equations in
# src/EquilibriaProbes.jl docstring). Code: BifurcationInference.jl v0.1.4
# @ 534ee7c (snapshot under reference/).
#
# Probes
#   p1  analytic saddle-node/pitchfork references vs augmented Newton solves
#   p2  Eq (10) implicit bifurcation gradients vs analytic vs finite differences
#   p2c Supp C (C.8) normal deformation field vs analytic curve displacement
#   p3  Eq (7) arclength normalisation vs released-code unnormalised sum
#   p4  Eq (4) distance vs released-code squared distance
#   p5  measure-formula separation (Eq 5 / D.1 / E.1 / released sinh) on a
#       real fold, a 2D Hopf curve (determinant blind), a steep B.6 family
#   p6  edge cases: empty predictions, empty targets, exact match
#   p7  degenerate crossings: SN cusp (θ₁→0) and perfect pitchfork (θ₁=0)
#   p8  small synthetic parameter inference (own engine; Eq 10 gradients;
#       fixed-η paper-faithful descent AND backtracking descent)
using LinearAlgebra
using Random
using Printf
using Dates
using TOML

include(joinpath(@__DIR__, "src", "EquilibriaProbes.jl"))
using .EquilibriaProbes

# --out-dir DIR: write artifacts to DIR (default: the committed results
# directory). Parent reruns should pass a separate --out-dir so committed
# artifacts are not overwritten.
function parse_outdir(default)
    for i in 1:length(ARGS)-1
        ARGS[i] == "--out-dir" && return ARGS[i+1]
    end
    return default
end
const RESULTS = parse_outdir(joinpath(@__DIR__, "..", "..", "results", "01-equilibria"))
mkpath(RESULTS)

function write_toml(name, dict)
    open(joinpath(RESULTS, name), "w") do io
        TOML.print(io, dict; sorted=true)
    end
end

function write_csv(name, header, rows)
    open(joinpath(RESULTS, name), "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join((x -> isa(x, AbstractFloat) ? string(x) : string(x)).(r), ","))
        end
    end
end

const T0 = time()
const TIMINGS = Vector{Pair{String,Float64}}()
function timed(name, f)
    t = time()
    out = f()
    push!(TIMINGS, name => round(time() - t; digits=3))
    return out
end

const SEED = 20261019

# =============================================================== p1: analytic
function probe_p1()
    sn = SNModel(); pf = PFModel()
    rows = []
    θsn = [2.5, -1.0]
    sols = find_all_solutions(sn, θsn, (-3.0, 3.0), (-3.0, 3.0))
    for σ in (+1, -1)
        u, p = sn_fold(θsn, σ)
        num = sort(sols, by = s -> abs(s[1] - u))[1]
        push!(rows, ["SN", "fold_p_sigma$(σ)", p, num[2], abs(num[2] - p)])
    end
    θpf = [0.5, -1.0]
    u, p = pf_fold(θpf)
    sols_pf = find_all_solutions(pf, θpf, (-3.0, 3.0), (-3.0, 3.0))
    num = sort(sols_pf, by = s -> abs(s[1] - u))[1]
    push!(rows, ["PF", "bif_p", p, num[2], abs(num[2] - p)])
    ok = all(r[5] < 1e-10 for r in rows)
    write_csv("p1_analytic.csv", ["model", "quantity", "analytic", "newton", "abs_err"], rows)
    return (rows=rows, nsol_sn=length(sols), nsol_pf=length(sols_pf), ok=ok)
end

# ======================================================= p2: implicit gradients
function probe_p2()
    sn = SNModel(); pf = PFModel()
    rows = []
    function addcase(name, m, θ, zstar, analytic)
        v_imp = velocity(m, zstar, θ)
        v_fd, fdok = velocity_fd(m, zstar, θ; h=1e-5)
        for k in 1:2
            push!(rows, [name, "theta$k", analytic[k], v_imp[k], v_fd[k],
                         abs(v_imp[k] - analytic[k]),
                         fdok ? abs(v_fd[k] - analytic[k]) : NaN])
        end
    end
    θsn = [2.5, -1.0]
    for σ in (+1, -1)
        u, p = sn_fold(θsn, σ)
        addcase("SN_sigma$(σ)", sn, θsn, [u, p], sn_velocity_analytic(θsn, σ))
    end
    θpf = [0.5, -1.0]
    u, p = pf_fold(θpf)
    addcase("PF", pf, θpf, [u, p], pf_velocity_analytic(θpf))
    write_csv("p2_gradients.csv",
        ["case", "param", "analytic", "implicit_eq10", "fd_central",
         "err_implicit", "err_fd"], rows)
    worst_imp = maximum(r[6] for r in rows)
    worst_fd = maximum(r[7] for r in rows)
    return (rows=rows, worst_implicit=worst_imp, worst_fd=worst_fd,
            ok = worst_imp < 1e-9 && worst_fd < 1e-6)
end

# ============================================== p2c: C.8 deformation vs analytic
function probe_p2c()
    sn = SNModel(); θ = [2.5, -1.0]
    rows = []
    for u0 in (0.25, 0.5, 0.75)
        z0 = curve_points(sn, u0, θ)
        D = deformation_c8(sn, z0, θ)
        a = θ[1] + 3θ[2]*u0^2
        T = [1.0, -a]/sqrt(1 + a^2)
        for (k, dz) in enumerate([(0.0, -u0), (0.0, -u0^3)])
            dzv = collect(dz)
            normal = dzv - dot(dzv, T)*T
            err = norm(D[:, k] - normal)
            push!(rows, [u0, k, D[1, k], D[2, k], normal[1], normal[2], err])
        end
    end
    write_csv("p2c_deformation.csv",
        ["u", "theta_idx", "c8_du", "c8_dp", "analytic_normal_du",
         "analytic_normal_dp", "err"], rows)
    worst = maximum(r[7] for r in rows)
    return (rows=rows, worst=worst, ok = worst < 1e-12)
end

# ================================================== p3: normalisation behaviour
function probe_p3()
    sn = SNModel(); pf = PFModel()
    θsn = [2.5, -1.0]; θpf = [0.5, -1.0]
    pmin, pmax = -2.0, 2.0
    Ψ4000 = simpson_Ψ(sn, θsn, φ_eq5, pmin, pmax; n=4000)
    Ψ32000 = simpson_Ψ(sn, θsn, φ_eq5, pmin, pmax; n=32000)
    Φeq5 = trapz_Φ(sn, θsn, φ_eq5, pmin, pmax; n=2000)
    L_s = Ψ4000.L
    id_rel = abs(Φeq5.Φ/Ψ4000.Ψ - L_s)/L_s
    Ψcodeφ = simpson_Ψ(sn, θsn, φ_code, pmin, pmax; n=1000)
    # NOTE (corrected labelling): trapz_Φ with bare φ_code under the indicator
    # window is a NORMALIZATION-ONLY SURROGATE — it isolates the Φ = Ψ·L axis
    # while keeping the code's per-point detector, but it is NOT the released
    # objective, whose per-point measure multiplies by the erf window. The
    # actual released functional is computed separately below (erf-windowed,
    # via the same integrand w(p)·φ_code; stdlib erf_approx, |err| ≤ 1.5e-7;
    # the exact-erf package-side cross-check lives in gradient_checks.jl).
    Φsurrogate = trapz_Φ(sn, θsn, φ_code, pmin, pmax; n=2000)
    M_actual_sn = M_windowed_code(θsn, pmin, pmax; n=2000, trunc=false)
    M_actual_pf = M_windowed_code(θpf, pmin, pmax; n=2000, trunc=false)
    up, pp = sn_fold(θsn, +1); um, pm = sn_fold(θsn, -1)
    preds = [pp, pm]                      # |P| = 2
    targets_more = [1.0]                  # |D| = 1
    Ep = E_paper(preds, targets_more); Ec = E_code(preds, targets_more)
    mt_paper_PgtD = (2 - 1)*log(Ψ4000.Ψ)
    mt_normfix_PgtD = (2 - 1)*log(Ψcodeφ.Ψ)   # code φ, paper normalisation (surrogate)
    mt_surr_PgtD = (2 - 1)*log(Φsurrogate.Φ) # code φ, unnormalised (surrogate)
    mt_actual_PgtD = (2 - 1)*log(M_actual_sn) # released functional (erf window)
    upf, ppf = pf_fold(θpf)
    Ψpf5 = simpson_Ψ(pf, θpf, φ_eq5, pmin, pmax; n=1000)
    Ψpfc = simpson_Ψ(pf, θpf, φ_code, pmin, pmax; n=1000)
    Φpfc = trapz_Φ(pf, θpf, φ_code, pmin, pmax; n=2000)
    mt_paper_PltD = (1 - 2)*log(Ψpf5.Ψ)
    mt_normfix_PltD = (1 - 2)*log(Ψpfc.Ψ)
    mt_surr_PltD = (1 - 2)*log(Φpfc.Φ)
    mt_actual_PltD = (1 - 2)*log(M_actual_pf)
    L_paper_notargets = 2*log(Ψ4000.Ψ)
    L_surr_notargets = 2*log(Φsurrogate.Φ)
    L_actual_notargets = 2*log(M_actual_sn)
    rows = [
        ["Psi_eq5_sn_n4000", Ψ4000.Ψ], ["Psi_eq5_sn_n32000", Ψ32000.Ψ],
        ["refinement_absdiff", abs(Ψ32000.Ψ - Ψ4000.Ψ)],
        ["Phi_eq5_trapz", Φeq5.Φ], ["L_simpson", L_s],
        ["Phi_over_Psi", Φeq5.Φ/Ψ4000.Ψ], ["identity_relerr", id_rel],
        ["Phi_surrogate_indicator_codephi", Φsurrogate.Φ],
        ["M_actual_released_functional_sn_erf", M_actual_sn],
        ["M_actual_released_functional_pf_erf", M_actual_pf],
        ["E_paper_PgtD", Ep], ["E_code_PgtD", Ec],
        ["measure_term_paper_PgtD", mt_paper_PgtD],
        ["measure_term_normfix_surrogate_PgtD", mt_normfix_PgtD],
        ["measure_term_unnormalised_surrogate_PgtD", mt_surr_PgtD],
        ["measure_term_actual_released_PgtD", mt_actual_PgtD],
        ["measure_term_paper_PltD", mt_paper_PltD],
        ["measure_term_normfix_surrogate_PltD", mt_normfix_PltD],
        ["measure_term_unnormalised_surrogate_PltD", mt_surr_PltD],
        ["measure_term_actual_released_PltD", mt_actual_PltD],
        ["loss_paper_notargets", L_paper_notargets],
        ["loss_unnormalised_surrogate_notargets", L_surr_notargets],
        ["loss_actual_released_notargets", L_actual_notargets],
    ]
    write_csv("p3_normalisation.csv", ["quantity", "value"], rows)
    sigflip_surrogate_PgtD = sign(mt_paper_PgtD) != sign(mt_surr_PgtD)
    sigflip_actual_PgtD = sign(mt_paper_PgtD) != sign(mt_actual_PgtD)
    sigflip_surrogate_PltD = sign(mt_paper_PltD) != sign(mt_surr_PltD)
    sigflip_actual_PltD = sign(mt_paper_PltD) != sign(mt_actual_PltD)
    write_toml("p3_normalisation.toml", Dict(
        "window" => [pmin, pmax],
        "note" => "normalisation axis isolated: with the SAME per-point measure the unnormalised branch sum equals the paper's Ψ times total arclength L (Φ = Ψ·L, verified to quadrature error). Since log enters the loss multiplied by (|P|-|D|), the sign of the count-term contribution depends on whether the summed measure exceeds 1: the paper's Ψ is normalised into [0,1] (log Ψ ≤ 0 always), the released sum is not. Sign flips below are reported for the normalization-only surrogate AND for the actual released functional (erf window times φ_code; stdlib erf approximation |err| ≤ 1.5e-7, full level set in window — the released value additionally depends on which branches deflated continuation discovers).",
        "labelling_correction" => "the earlier version of this artifact labelled the unnormalised indicator-window φ_code sum as 'the code measure term'; it is a normalization-only surrogate. The actual released per-point measure multiplies by the erf window; its continuum value is M_actual_released_*.",
        "psi_refinement_absdiff" => abs(Ψ32000.Ψ - Ψ4000.Ψ),
        "phi_equals_psi_times_L_relerr" => id_rel,
        "total_arclength_sn_window" => L_s,
        "M_actual_sn" => M_actual_sn,
        "M_actual_pf" => M_actual_pf,
        "P_gt_D" => Dict(
            "paper_term" => mt_paper_PgtD,
            "paper_normalisation_code_phi_surrogate" => mt_normfix_PgtD,
            "unnormalised_surrogate" => mt_surr_PgtD,
            "actual_released_term" => mt_actual_PgtD,
            "sign_flip_surrogate_vs_paper" => sigflip_surrogate_PgtD,
            "sign_flip_actual_vs_paper" => sigflip_actual_PgtD,
        ),
        "P_lt_D" => Dict(
            "paper_term" => mt_paper_PltD,
            "paper_normalisation_code_phi_surrogate" => mt_normfix_PltD,
            "unnormalised_surrogate" => mt_surr_PltD,
            "actual_released_term" => mt_actual_PltD,
            "sign_flip_surrogate_vs_paper" => sigflip_surrogate_PltD,
            "sign_flip_actual_vs_paper" => sigflip_actual_PltD,
        ),
        "empty_targets" => Dict(
            "paper_loss" => L_paper_notargets,
            "unnormalised_surrogate_loss" => L_surr_notargets,
            "actual_released_loss" => L_actual_notargets,
            "sign_flip_surrogate_vs_paper" => (sign(L_paper_notargets) != sign(L_surr_notargets)),
            "sign_flip_actual_vs_paper" => (sign(L_paper_notargets) != sign(L_actual_notargets)),
        ),
        "ok" => (abs(Ψ32000.Ψ - Ψ4000.Ψ) < 1e-8 && id_rel < 1e-5 &&
                 sigflip_surrogate_PgtD && sigflip_actual_PgtD &&
                 sigflip_surrogate_PltD),
    ))
    return (Ψ=Ψ4000.Ψ, L=L_s, id_rel=id_rel,
            M_actual_sn=M_actual_sn,
            sigflip_actual_PgtD=sigflip_actual_PgtD,
            sigflip_actual_PltD=sigflip_actual_PltD,
            ok=(abs(Ψ32000.Ψ - Ψ4000.Ψ) < 1e-8 && id_rel < 1e-5 &&
                sigflip_surrogate_PgtD && sigflip_actual_PgtD &&
                sigflip_surrogate_PltD))
end

# ============================================== p4: distance vs squared distance
function probe_p4()
    rows = []
    preds = [1.5]; targets = [1.0, 3.0]
    Ep = E_paper(preds, targets); Ec = E_code(preds, targets)
    push!(rows, ["A_two_targets", "E_paper", Ep])
    push!(rows, ["A_two_targets", "E_code", Ec])
    push!(rows, ["A_two_targets", "E_paper_squared", Ep^2])
    push!(rows, ["A_two_targets", "E_code_minus_Epaper2", Ec - Ep^2])
    for p1 in (1.25, 1.5, 2.0, 2.75)
        push!(rows, ["B_minimiser_p1=$(p1)", "E_paper", E_paper([p1], targets)])
        push!(rows, ["B_minimiser_p1=$(p1)", "E_code", E_code([p1], targets)])
    end
    v = [[1.0]]                            # synthetic velocity dp1/dθ = 1
    gp = ∇E_paper([1.5], v, targets)[1]
    gc = ∇E_code([1.5], v, targets)[1]
    push!(rows, ["B_gradient_at_1p5", "grad_E_paper", gp])
    push!(rows, ["B_gradient_at_1p5", "grad_E_code", gc])
    preds10 = [15.0]; targets10 = [10.0, 30.0]
    push!(rows, ["C_scale_x10", "E_paper", E_paper(preds10, targets10)])
    push!(rows, ["C_scale_x10", "E_code", E_code(preds10, targets10)])
    write_csv("p4_distances.csv", ["case", "quantity", "value"], rows)
    ok = (gp == 0.0 && abs(gc - (-1.0)) < 1e-12 &&
          abs(Ec - 1.25) < 1e-12 && abs(Ep - 1.0) < 1e-12 &&
          abs(Ec - Ep^2) > 1e-6)
    write_toml("p4_distances.toml", Dict(
        "per_target_identity" => "code per-target gm2 == (paper per-target gm)^2 exactly",
        "multi_target_weighting" => "E_code = mean_t gm_t^2 != (mean_t gm_t)^2 = E_paper^2: far targets are upweighted by the square",
        "E_paper_caseA" => Ep, "E_code_caseA" => Ec,
        "E_paper_squared_caseA" => Ep^2,
        "minimiser_structure" => "1 prediction, 2 targets: paper E is flat-minimal on the whole interval between targets; code E has the unique midpoint minimiser",
        "grad_E_paper_at_1p5" => gp, "grad_E_code_at_1p5" => gc,
        "paper_gradient_zero_in_flat_region" => (gp == 0.0),
        "scale_x10_E_paper" => E_paper(preds10, targets10),
        "scale_x10_E_code" => E_code(preds10, targets10),
        "scale_note" => "rescaling p by 10 scales E_paper by 10 but E_code by 100. The measure term is not p-scale-invariant either (the erf window width and the sinh argument carry p-units), so unit choices shift the error-vs-measure balance in both objectives; no scale-invariance claim is made for either term.",
        "ok" => ok,
    ))
    return (gp=gp, gc=gc, ok=ok)
end

# ================================================== p5: measure formula separation
function probe_p5()
    sn = SNModel(); fhn = FHNModel()
    θsn = [2.5, -1.0]
    pmin, pmax = -2.0, 2.0
    dom = branch_domain(sn, θsn, pmin, pmax)
    us = Float64[]
    for (A, B) in dom
        append!(us, range(A, B; length=40))
    end
    rows = []
    for u in us
        z = curve_points(sn, u, θsn)
        push!(rows, [u, z[2], det_state(sn, z, θsn),
                     φ_eq5(sn, z, θsn), φ_tanh(sn, z, θsn),
                     φ_e1(sn, z, θsn), φ_code(sn, z, θsn)])
    end
    write_csv("p5_measures_sn.csv",
        ["u", "p", "det", "phi_eq5", "phi_tanhD1", "phi_e1", "phi_code"], rows)

    θf = [0.7, 0.8, 25.0]
    uh = hopf_location(θf)
    rows2 = []
    detmin = Inf; maxφe1 = -Inf; maxφcode = -Inf; φeq5_at_hopf = -Inf
    for u in range(-2.0, 2.0; length=161)
        z = curve_points(fhn, u, θf)
        D = det_state(fhn, z, θf)
        detmin = min(detmin, D)
        ev = eigen(statejac(fhn, z, θf)).values
        e1 = φ_e1(fhn, z, θf); co = φ_code(fhn, z, θf)
        maxφe1 = max(maxφe1, e1); maxφcode = max(maxφcode, co)
        if abs(abs(u) - uh) < 0.025
            φeq5_at_hopf = max(φeq5_at_hopf, φ_eq5(fhn, z, θf))
        end
        push!(rows2, [u, z[3], D, minimum(real.(ev)), maximum(real.(ev)),
                      φ_eq5(fhn, z, θf), φ_tanh(fhn, z, θf), e1, co])
    end
    write_csv("p5_measures_fhn.csv",
        ["u1", "p", "det", "re_eig_min", "re_eig_max", "phi_eq5",
         "phi_tanhD1", "phi_e1", "phi_code"], rows2)

    rows3 = []
    summary = Dict{String,Float64}()
    for K in (10.0, 1e3, 1e6)
        θK = [1.0, K/3]
        max5 = maxT = maxE = maxC = -Inf
        for u in range(-0.05, 0.05; length=801)
            z = curve_points(sn, u, θK)
            max5 = max(max5, φ_eq5(sn, z, θK))
            maxT = max(maxT, φ_tanh(sn, z, θK))
            maxE = max(maxE, φ_e1(sn, z, θK))
            maxC = max(maxC, φ_code(sn, z, θK))
        end
        summary["K=$(K)_max_phi_eq5"] = max5
        summary["K=$(K)_max_phi_tanh"] = maxT
        summary["K=$(K)_max_phi_e1"] = maxE
        summary["K=$(K)_max_phi_code"] = maxC
        push!(rows3, [K, 1.0, max5, maxT, maxE, maxC])
    end
    write_csv("p5_spurious_b6.csv",
        ["K", "min_det_on_curve", "max_phi_eq5", "max_phi_tanhD1",
         "max_phi_e1", "max_phi_code"], rows3)

    ok_fhn = (detmin > 0 && maxφe1 > 0.99 && maxφcode > 1.9 && φeq5_at_hopf < 0.6)
    ok_b6 = (summary["K=1.0e6_max_phi_eq5"] > 0.99 &&
             summary["K=1.0e6_max_phi_tanh"] > 0.95)
    write_toml("p5_measures.toml", Dict(
        "fhn" => Dict(
            "theta_a_b_tau" => θf,
            "hopf_u1_analytic" => uh,
            "min_det_on_curve" => detmin,
            "max_phi_e1_on_curve" => maxφe1,
            "max_phi_code_on_curve" => maxφcode,
            "phi_eq5_at_hopf" => φeq5_at_hopf,
            "note" => "det(∂F/∂u) > 0 everywhere on the FHN equilibrium curve: the main-text Eq (5) determinant measure cannot reach 1 at the Hopf (phi_eq5 ≈ 0.45 there) while the eigen-real-part measures (E.1, released sinh form) hit their maxima exactly at the crossing. The main-text determinant formulation and the supplemental/released eigen measures are different detectors, not normalization variants.",
        ),
        "b6_spurious_family" => Dict(
            "model" => "F = p + u + (K/3)u^3, det = 1 + K u^2 >= 1: no crossing exists",
            "note" => "all four measure forms approach 1 near u ~ K^{-1/2} where d(det)/ds is steep but det never crosses zero — Supp B (B.6) spurious maxima. The D.1 tanh fix does not suppress them in this family (measure values, not gradients, are compared).",
            "maxima" => summary,
        ),
        "ok" => (ok_fhn && ok_b6),
    ))
    return (detmin=detmin, maxφe1=maxφe1, maxφcode=maxφcode,
            φeq5_hopf=φeq5_at_hopf, b6=summary, ok=(ok_fhn && ok_b6))
end

# ============================================================ p6: edge cases
function probe_p6()
    sn = SNModel(); θ = [2.5, -1.0]
    u, p = sn_fold(θ, +1)
    v = velocity(sn, [u, p], θ)          # nonzero fold velocity (Eq 10)
    out = Dict{String,Any}()
    # --- empty-set semantics (released code reading) ---
    out["empty_predictions_E_paper_is_NaN"] = isnan(E_paper(Float64[], [1.0, 3.0]))
    out["empty_predictions_E_code_equals_1"] = E_code(Float64[], [1.0, 3.0])
    out["empty_predictions_E_code_blind_to_targets"] =
        (E_code(Float64[], [1.0, 3.0]) == E_code(Float64[], [-50.0, 50.0]))
    out["empty_predictions_grad_E_code_zero"] =
        (∇E_code(Float64[], Vector{Vector{Float64}}(), [1.0, 3.0]) == [0.0, 0.0])
    out["empty_targets_E_code_zero"] = E_code([1.5], Float64[])
    # --- C1: code objective, single prediction, exact match ---
    # E_code = (p−t)² is SMOOTH at the match with true gradient 0. The
    # released factored form gm2·2v/(p−t) returns NaN (0·Inf): a REMOVABLE
    # IMPLEMENTATION DEFECT of the factored expression, not a mathematical
    # singularity — the independent unfactored derivative 2(p−t)·v is 0.
    out["C1_E_code_zero_at_match"] = E_code([1.5], [1.5]) == 0.0
    gcode = ∇E_code([1.5], [v], [1.5])
    out["C1_released_factored_grad_NaN_removable_defect"] = any(isnan, gcode)
    out["C1_true_unfactored_grad_zero"] = (2*(1.5 - 1.5)*v == [0.0, 0.0])
    out["C1_factored_form_tracks_true_derivative_near_match"] =
        norm(∇E_code([1.5 + 1e-6], [v], [1.5]) - 2e-6*v) < 1e-9
    # --- C2: paper objective, single prediction, exact match ---
    # E_paper = |p−t| has a GENUINE kink at the match: one-sided gradients
    # ±v exist, the derivative does not (subgradient).
    out["C2_E_paper_zero_at_match"] = E_paper([1.5], [1.5]) == 0.0
    out["C2_one_sided_plus"] = ∇E_paper([1.5 + 1e-6], [v], [1.5])
    out["C2_one_sided_minus"] = ∇E_paper([1.5 - 1e-6], [v], [1.5])
    out["C2_kink_genuine_one_sided_flip"] =
        (norm(∇E_paper([1.5 + 1e-6], [v], [1.5]) - v) < 1e-6 &&
         norm(∇E_paper([1.5 - 1e-6], [v], [1.5]) + v) < 1e-6)
    # --- C3: multiprediction, one matched, one at distance A ---
    # paper E ~ (|d|·A)^{1/2}: UNBOUNDED gradient ∝ |d|^{-1/2} (genuine);
    # code E = |d|·A: bounded kink, subgradient → ±A·v (genuine nonsmooth).
    A = 2.0; t = 1.5
    c3_rows = []
    for d in (1e-1, 1e-2, 1e-3, 1e-6)
        gp3 = norm(∇E_paper([t + d, t + A], [v, v], [t]))
        gc3 = ∇E_code([t + d, t + A], [v, v], [t])
        push!(c3_rows, [d, gp3, gp3*sqrt(d), norm(gc3), norm(gc3 - A*v)])
    end
    write_csv("p6_multiprediction.csv",
        ["d", "abs_grad_E_paper", "grad_paper_times_sqrt_d",
         "abs_grad_E_code", "abs_grad_E_code_minus_Av"], c3_rows)
    out["C3_paper_gradient_unbounded_sqrt_tail"] =
        (abs(c3_rows[1][3] - c3_rows[4][3]) < 0.05*maximum(r[3] for r in c3_rows))
    # exact identity: ∇E_code = (A + d)·v at preds [t+d, t+A] (bounded kink,
    # subgradient → ±A·v); assert it tightly at the smallest d
    out["C3_code_gradient_bounded_kink"] =
        (norm(∇E_code([t + 1e-6, t + A], [v, v], [t]) - (A + 1e-6)*v) < 1e-9)
    # --- near-singleton approach table (contrast released vs unfactored) ---
    rows = []
    for d in (1e-1, 1e-2, 1e-3, 1e-6, 1e-9)
        gd = ∇E_code([1.5 + d], [v], [1.5])
        gp = ∇E_paper([1.5 + d], [v], [1.5])
        push!(rows, [d, norm(gp), norm(gd), norm(gd - 2*d*v), norm(gd)/d])
    end
    write_csv("p6_exact_approach.csv",
        ["d", "abs_grad_E_paper", "abs_grad_E_code",
         "abs_grad_E_code_minus_2dv", "abs_grad_E_code_over_d"], rows)
    out["paper_gradient_tends_to_abs_v"] =
        abs(rows[5][2] - norm(v)) < 1e-6
    out["code_gradient_vanishes_linearly_2vd"] =
        abs(rows[5][5] - 2*norm(v)) < 1e-4*norm(v) + 1e-15
    write_toml("p6_edgecases.toml", Dict(
        "empty_predictions_note" => "released code errors() with empty predictions returns 1 (empty-product init) for ANY targets — the error term is blind to target location when no prediction exists; the paper Eq (4) is undefined there (NaN recorded). The code gradient of the error term is exactly zero: only the measure term can move theta until a first bifurcation appears.",
        "empty_predictions_E_paper_is_NaN" => out["empty_predictions_E_paper_is_NaN"],
        "empty_predictions_E_code_equals_1" => out["empty_predictions_E_code_equals_1"],
        "empty_predictions_E_code_blind_to_targets" => out["empty_predictions_E_code_blind_to_targets"],
        "empty_predictions_grad_E_code_zero" => out["empty_predictions_grad_E_code_zero"],
        "empty_targets_E_code_zero" => out["empty_targets_E_code_zero"],
        "C1_note" => "code objective, single prediction, exact match: E=(p−t)² is smooth with true gradient 0 (independent unfactored derivative 2(p−t)·v = 0). The released factored gradient form gm2·2v/(p−t) returns NaN (0·Inf) — a REMOVABLE IMPLEMENTATION DEFECT of the factored expression, not a mathematical singularity; near the match the same factored form tracks the true derivative 2dv.",
        "C1_released_factored_grad_NaN_removable_defect" => out["C1_released_factored_grad_NaN_removable_defect"],
        "C1_true_unfactored_grad_zero" => out["C1_true_unfactored_grad_zero"],
        "C1_factored_form_tracks_true_derivative_near_match" => out["C1_factored_form_tracks_true_derivative_near_match"],
        "C2_note" => "paper objective, single prediction, exact match: |p−t| has a genuine kink — one-sided gradients +v/−v; genuinely nondifferentiable (unlike C1)",
        "C2_kink_genuine_one_sided_flip" => out["C2_kink_genuine_one_sided_flip"],
        "C2_one_sided_plus" => out["C2_one_sided_plus"],
        "C2_one_sided_minus" => out["C2_one_sided_minus"],
        "C3_note" => "multiprediction (one matched, one at distance A): paper E ~ (|d|A)^(1/2) — genuinely UNBOUNDED gradient ∝ |d|^{-1/2} (the sqrt(δ) tail that destabilises fixed-step descent in p8); code E = |d|·A — genuinely nonsmooth but BOUNDED (subgradient → ±A·v), the structure behind the p8 fixed-step stall of the squared objective at |P|=|D|=2",
        "C3_paper_gradient_unbounded_sqrt_tail" => out["C3_paper_gradient_unbounded_sqrt_tail"],
        "C3_code_gradient_bounded_kink" => out["C3_code_gradient_bounded_kink"],
        "near_match_note" => "approaching an exact singleton match the paper |d| gradient tends to |v| (persistent crossing force), the code d² gradient vanishes like 2|v|d",
        "paper_gradient_tends_to_abs_v" => out["paper_gradient_tends_to_abs_v"],
        "code_gradient_vanishes_linearly_2vd" => out["code_gradient_vanishes_linearly_2vd"],
        "fold_velocity_norm" => norm(v),
        "ok" => (out["empty_predictions_E_code_equals_1"] == 1.0 &&
                 out["empty_predictions_E_code_blind_to_targets"] === true &&
                 out["empty_predictions_grad_E_code_zero"] === true &&
                 out["empty_targets_E_code_zero"] == 0.0 &&
                 out["C1_released_factored_grad_NaN_removable_defect"] === true &&
                 out["C1_true_unfactored_grad_zero"] === true &&
                 out["C1_factored_form_tracks_true_derivative_near_match"] === true &&
                 out["C2_kink_genuine_one_sided_flip"] === true &&
                 out["C3_paper_gradient_unbounded_sqrt_tail"] === true &&
                 out["C3_code_gradient_bounded_kink"] === true &&
                 out["paper_gradient_tends_to_abs_v"] === true &&
                 out["code_gradient_vanishes_linearly_2vd"] === true),
    ))
    ok = (out["empty_predictions_E_code_equals_1"] == 1.0 &&
          out["empty_predictions_E_code_blind_to_targets"] === true &&
          out["empty_predictions_grad_E_code_zero"] === true &&
          out["empty_targets_E_code_zero"] == 0.0 &&
          out["C1_released_factored_grad_NaN_removable_defect"] === true &&
          out["C1_true_unfactored_grad_zero"] === true &&
          out["C1_factored_form_tracks_true_derivative_near_match"] === true &&
          out["C2_kink_genuine_one_sided_flip"] === true &&
          out["C3_paper_gradient_unbounded_sqrt_tail"] === true &&
          out["C3_code_gradient_bounded_kink"] === true &&
          out["paper_gradient_tends_to_abs_v"] === true &&
          out["code_gradient_vanishes_linearly_2vd"] === true)
    return (out=out, ok=ok)
end

# ===================================================== p7: degenerate crossings
function probe_p7()
    sn = SNModel(); pf = PFModel()
    rows = []
    for θ₁ in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6)
        θ = [θ₁, -1.0]
        d = cusp_diagnostics(sn, θ, +1)
        h = min(1e-5, θ₁/10)
        v_fd, fdok = velocity_fd(sn, [d.u, d.p], θ; h=h)
        relerr = (fdok && d.velocity[1] != 0) ?
                 abs(v_fd[1] - d.velocity[1])/abs(d.velocity[1]) : NaN
        push!(rows, [θ₁, d.u, d.p, d.condJ, d.velocity[1],
                     fdok ? v_fd[1] : NaN, relerr])
    end
    θneg = [-0.1, -1.0]
    sols_neg = find_all_solutions(sn, θneg, (-3.0, 3.0), (-3.0, 3.0))
    θpf0 = [0.0, -1.0]
    sols0 = find_all_solutions(pf, θpf0, (-2.0, 2.0), (-2.0, 2.0))
    J0 = vcat(Fz(pf, [0.0, 0.0], θpf0), det_z(pf, [0.0, 0.0], θpf0))
    condJ0 = cond(J0)
    v0 = try
        velocity(pf, [0.0, 0.0], θpf0)
    catch
        [NaN, NaN]
    end
    v0 = all(isfinite, v0) ? v0 : [NaN, NaN]
    div_rows = []
    for θ₁ in (1e-1, 1e-2, 1e-4, 1e-6)
        θ = [θ₁, -1.0]
        u, p = pf_fold(θ)
        push!(div_rows, [θ₁, abs(pf_velocity_analytic(θ)[1]),
                         abs(velocity(pf, [u, p], θ)[1])])
    end
    write_csv("p7_cusp.csv",
        ["theta1", "u_fold", "p_fold", "condJ", "velocity_theta1",
         "velocity_fd", "fd_relerr"], rows)
    write_csv("p7_pitchfork_divergence.csv",
        ["theta1", "abs_analytic_dp_dtheta1", "abs_implicit_dp_dtheta1"], div_rows)
    ok = (length(sols_neg) == 0 && length(sols0) >= 1 && condJ0 > 1e10 &&
          !all(isfinite, v0) && div_rows[end][2] > 10*div_rows[1][2] &&
          all(isnan(r[7]) || r[7] < 1e-2 for r in rows))
    write_toml("p7_degenerate.toml", Dict(
        "cusp_note" => "SN cusp at theta1=0: augmented Jacobian [[0,1],[0,0]] singular; Eq (10) does not exist there. cond(J) grows ~1/sqrt(theta1) approaching the cusp and the fold velocity vanishes like sqrt(theta1). For theta1<0 there are no folds at all: |P| jumps 0->2 across theta1=0 (the paper's piecewise-smoothness boundary); central FD across the jump is impossible.",
        "negative_side_solutions_found" => length(sols_neg),
        "perfect_pitchfork_solutions_found" => length(sols0),
        "perfect_pitchfork_condJ" => condJ0,
        "perfect_pitchfork_velocity" => v0,
        "perfect_pitchfork_note" => "at the perfect pitchfork (theta1=0) the rectangular Jacobian dF/dz is the zero matrix: tangent field undefined (rank 0) and the augmented Jacobian [[0,0],[0,1]] singular, although the determinant crossing restricted to the u=0 branch is non-degenerate — the augmented (branch-agnostic) formulation cannot see branch-restricted non-degeneracy. PF fold velocity dp*/dtheta1 = -1/u* ~ |theta1|^{-1/3} diverges as theta1→0.",
        "pf_velocity_diverges" => (div_rows[end][2] > 10*div_rows[1][2]),
        "ok" => ok,
    ))
    return (rows=rows, sols_neg=length(sols_neg), condJ0=condJ0, v0=v0,
            div=div_rows, ok=ok)
end

# ================================================= p8: synthetic inference runs
"SN folds by warm-started Newton from both branches (σ=±1); PF fold likewise."
function warm_folds(m, θ, warm)
    if m isa SNModel
        return sn_folds(θ, warm)
    end
    z, ok = solve_augmented(m, warm[1], θ)
    return ok ? [z] : Vector{Float64}[]
end

function sn_folds(θ, warm)
    out = Vector{Float64}[]
    for z0 in warm
        z, ok = solve_augmented(SNModel(), z0, θ)
        ok && push!(out, z)
    end
    return out
end

"E and gradient with the requested objective at current folds."
function lossgrad(m, θ, zs, targets, paper_obj)
    preds = [z[end] for z in zs]
    vels = [velocity(m, z, θ) for z in zs]
    E = paper_obj ? E_paper(preds, targets) : E_code(preds, targets)
    g = paper_obj ? ∇E_paper(preds, vels, targets) : ∇E_code(preds, vels, targets)
    return E, g, preds
end

"""
Gradient descent with optional backtracking line search.
paper_obj: true = Eq (4) unsquared error; false = released-code squared error.
Fixed η is paper-faithful (Flux Descent(0.01)); backtracking is our adaptive
variant. On warm-started fold loss the run stops and reports
(ejected=true, last valid state) — |P| changed, the paper's piecewise-
smoothness boundary. Returns (θ, traj, E, it_conv, ejected, minE, ejectθ).
"""
function run_descent(m, θ₀, targets, warm0; paper_obj=true, η=0.01,
                     backtrack=false, maxit=4000, tol=1e-30)
    θ = copy(θ₀); warm = [copy(w) for w in warm0]
    ηk = η
    traj = []
    it_conv = -1
    E_prev = Inf
    minE = Inf
    for it in 1:maxit
        zs = warm_folds(m, θ, warm)
        if length(zs) != length(warm)
            return (θ=θ, traj=traj, E=NaN, it_conv=it_conv, ejected=true,
                    minE=minE, ejectθ=copy(θ))
        end
        warm = zs
        E, g, preds = lossgrad(m, θ, zs, targets, paper_obj)
        all(isfinite, g) || return (θ=θ, traj=traj, E=E, it_conv=-3, ejected=true,
                                    minE=min(minE, E), ejectθ=copy(θ))
        minE = min(minE, E)
        (it == 1 || it % 25 == 0) && push!(traj, [it, E, θ..., preds...])
        E < 0.02 && it_conv < 0 && (it_conv = it)
        E < tol && return (θ=θ, traj=traj, E=E, it_conv=it_conv, ejected=false,
                           minE=minE, ejectθ=copy(θ))
        if backtrack
            ok = false
            for _ in 1:60
                θnew = θ .- ηk .* g
                zs2 = warm_folds(m, θnew, warm)
                if length(zs2) == length(warm)
                    E2 = paper_obj ? E_paper([z[end] for z in zs2], targets) :
                                      E_code([z[end] for z in zs2], targets)
                    if E2 < E
                        θ = θnew; warm = zs2; ηk = min(ηk*1.3, 0.05)
                        ok = true; break
                    end
                end
                ηk *= 0.5
                ηk < 1e-18 && break
            end
            ok || return (θ=θ, traj=traj, E=E, it_conv=it_conv, ejected=false,
                          minE=minE, ejectθ=copy(θ))
            if abs(E - E_prev) < 1e-15 && E < 1e-3
                return (θ=θ, traj=traj, E=E, it_conv=it_conv, ejected=false,
                        minE=minE, ejectθ=copy(θ))
            end
            E_prev = E
        else
            θ = θ .- η .* g
        end
    end
    zs = warm_folds(m, θ, warm)
    E, _, preds = lossgrad(m, θ, zs, targets, paper_obj)
    return (θ=θ, traj=traj, E=E, it_conv=it_conv, ejected=false,
            minE=min(minE, E), ejectθ=copy(θ))
end

function probe_p8()
    rng = MersenneTwister(SEED)
    sn = SNModel(); pf = PFModel()
    θsn_true = [2.5, -1.0]; θpf_true = [0.5, -1.0]
    _, t1 = sn_fold(θsn_true, +1)
    _, tp_ = pf_fold(θpf_true)
    sn_targets = sort([t1, -t1]); pf_target = [tp_]
    θ₀ = [1.0, -1.0]
    warm = [[sn_fold(θ₀, +1)...], [sn_fold(θ₀, -1)...]]
    # fixed-η (paper-faithful Flux Descent(0.01)) — expected to STALL near the
    # geometric-equivalence manifold because Eq (4) has a sqrt(delta) tail
    # near exact match (unbounded gradient); floor ~ (η)^{2/3}-controlled
    rA = run_descent(sn, θ₀, sn_targets, warm; paper_obj=true, maxit=600)
    rB = run_descent(sn, θ₀, sn_targets, warm; paper_obj=false, maxit=600)
    # backtracking: converges to the manifold
    rA2 = run_descent(sn, θ₀, sn_targets, warm; paper_obj=true, backtrack=true, maxit=4000)
    rB2 = run_descent(sn, θ₀, sn_targets, warm; paper_obj=false, backtrack=true, maxit=4000)
    θ₀pf = [0.7, -0.7]
    warmpf = [[pf_fold(θ₀pf)...]]
    rC = run_descent(pf, θ₀pf, pf_target, warmpf; paper_obj=true, maxit=600)
    rD = run_descent(pf, θ₀pf, pf_target, warmpf; paper_obj=false, maxit=600)
    rC2 = run_descent(pf, θ₀pf, pf_target, warmpf; paper_obj=true, backtrack=true, maxit=2000)
    # |P| = 1 < |D| = 2 (pitchfork, two targets): measure term ON, its
    # gradient by finite differences (adjoint ∇measure out of scope here)
    θ = copy(θ₀pf)
    traj_less = []
    function paper_loss_less(θv)
        zs2 = warm_folds(pf, θv, warmpf)
        isempty(zs2) && return NaN
        Ψv = simpson_Ψ(pf, θv, φ_eq5, -2.0, 2.0; n=300).Ψ
        (1 - 2)*log(Ψv) + E_paper([zs2[1][end]], [tp_, 3.0])
    end
    for it in 1:150
        L = paper_loss_less(θ)
        isnan(L) && break
        h = 1e-6; gfd = zeros(2)
        for k in 1:2
            θp = copy(θ); θp[k] += h
            θm = copy(θ); θm[k] -= h
            lp, lm = paper_loss_less(θp), paper_loss_less(θm)
            (isnan(lp) || isnan(lm)) && (gfd[k] = NaN; break)
            gfd[k] = (lp - lm)/(2h)
        end
        all(isfinite, gfd) || break
        (it % 15 == 1) && push!(traj_less, [it, L, θ...])
        θ = θ .- 0.01 .* gfd
    end
    # geometric-equivalence manifold (SN): θ₂ = -4θ₁³/(27 p*²) keeps both
    # folds at ±|p*| exactly; E_paper evaluated along it
    manifold_rows = []
    for θ₁ in (1.5, 2.0, 2.5, 3.0, 3.5)
        θ₂ = -4θ₁^3/(27*abs(t1)^2)
        θm = [θ₁, θ₂]
        wm = [[sn_fold(θm, +1)...], [sn_fold(θm, -1)...]]
        zs = sn_folds(θm, wm)
        Em = E_paper([z[end] for z in zs], sn_targets)
        push!(manifold_rows, [θ₁, θ₂, Em])
    end
    # random restarts within the 2-fold basin (backtracking descent)
    restarts = []
    for i in 1:8
        θ₁ = 2.5*exp(0.4*randn(rng)); θ₂ = -exp(0.4*randn(rng))
        θr = [θ₁, θ₂]
        wr = [[sn_fold(θr, +1)...], [sn_fold(θr, -1)...]]
        rr = run_descent(sn, θr, sn_targets, wr; paper_obj=true,
                         backtrack=true, maxit=4000)
        zs = sn_folds(rr.θ, wr)
        err = length(zs) == 2 ? maximum(abs.(sort([z[end] for z in zs]) - sn_targets)) : NaN
        push!(restarts, [i, θ₁, θ₂, rr.E, rr.it_conv, err])
    end
    write_csv("p8_descent_sn_paper_fixed.csv",
        ["iter", "E", "theta1", "theta2", "p_fold_sigma_plus", "p_fold_sigma_minus"], rA.traj)
    write_csv("p8_descent_sn_code_fixed.csv",
        ["iter", "E", "theta1", "theta2", "p_fold_sigma_plus", "p_fold_sigma_minus"], rB.traj)
    write_csv("p8_descent_sn_paper_backtrack.csv",
        ["iter", "E", "theta1", "theta2", "p_fold_sigma_plus", "p_fold_sigma_minus"], rA2.traj)
    write_csv("p8_descent_sn_code_backtrack.csv",
        ["iter", "E", "theta1", "theta2", "p_fold_sigma_plus", "p_fold_sigma_minus"], rB2.traj)
    write_csv("p8_descent_pf_paper_fixed.csv", ["iter", "E", "theta1", "theta2", "p_star"], rC.traj)
    write_csv("p8_descent_pf_code_fixed.csv", ["iter", "E", "theta1", "theta2", "p_star"], rD.traj)
    write_csv("p8_descent_pf_paper_backtrack.csv", ["iter", "E", "theta1", "theta2", "p_star"], rC2.traj)
    write_csv("p8_pf_less_targets.csv", ["iter", "L", "theta1", "theta2"], traj_less)
    write_csv("p8_manifold.csv", ["theta1", "theta2_analytic", "E_paper"], manifold_rows)
    write_csv("p8_restarts.csv",
        ["restart", "theta1_0", "theta2_0", "E_final", "iters_to_0.02", "max_target_err"],
        restarts)
    nconv = count(r -> r[4] < 1e-4 && r[6] < 0.02, restarts)
    # fixed-η stall/ejection bands; manifold tolerance 1e-5: the sqrt(delta)
    # tail amplifies Newton-level fold errors (~1e-12) into E ~ 1.7e-6
    ok = (rA.ejected && 0.001 < rA.minE < 0.02 && rA.ejectθ[2] > 0 &&
          # ^ paper-obj fixed-η: transient dip below 0.02 then sign-flip of θ₂
          #   ejected the iterate from the 2-fold regime (|P|: 2 → 0)
          0.002 < rB.E < 0.5 &&         # SN code fixed-η: stalls at a lower floor
          rA2.E < 1e-4 &&                # SN paper backtracking: converged
          rB2.E < 1e-10 &&               # SN code backtracking: converged
          1e-3 < rC.E < 0.05 &&         # PF paper fixed-η: stalls near 0.02
          rD.E < 1e-10 &&                # PF code fixed-η: converged (smooth)
          rC2.E < 1e-8 &&                # PF paper backtracking: converged
          all(r[3] < 1e-5 for r in manifold_rows) &&
          nconv == 8)
    write_toml("p8_inference.toml", Dict(
        "seed" => SEED,
        "note" => "Eq (4)'s geometric mean over predictions gives a sqrt(delta) tail near exact match when a target is matched but other predictions remain at nonzero distance: E ~ sqrt(delta*far_distance) with UNBOUNDED gradient. Measured consequence for the SN 2-fold/2-target case with the paper's own fixed-step descent (Flux Descent(0.01)): E decreased 1.47 to a transient minimum below the paper's 0.02 convergence threshold (~0.009), but the descent does not stop there (the released train! stopping rule Loss<0.02 with |P|=|D| would have halted at that dip); continuing, the growing near-match gradients produced oscillations of increasing amplitude that flipped the sign of theta2 and ejected the iterate from the 2-fold regime entirely (|P| changed 2->0, the paper's piecewise-smoothness boundary). The code's squared objective for |P|=|D|=2 is |delta|*far-distance — a bounded but NON-vanishing kinked gradient — so fixed-step descent stalls at E_code ~ 0.16 with transient sub-0.02 dips from iteration ~103; for |P|=|D|=1 it is delta^2 (smooth, vanishing gradient) and fixed-step descent converges to machine precision (PF) while the paper objective |delta| stalls at |delta| ~ 0.018. Backtracking line search converges in all cases.",
        "sn_targets" => sn_targets, "sn_start" => θ₀,
        "sn_paper_fixed_eta" => Dict(
            "E_final" => rA.E, "min_E" => rA.minE,
            "iters_to_first_sub_0.02" => rA.it_conv,
            "ejected_from_basin" => rA.ejected,
            "eject_theta" => rA.ejectθ,
            "ejected_by_leaving_fold_regime" => (rA.ejectθ[1]*rA.ejectθ[2] ≥ 0)),
        "sn_code_fixed_eta" => Dict("E_final" => rB.E, "iters_to_0.02" => rB.it_conv),
        "sn_paper_backtracking" => Dict("E_final" => rA2.E, "iters_to_0.02" => rA2.it_conv),
        "sn_code_backtracking" => Dict("E_final" => rB2.E, "iters_to_0.02" => rB2.it_conv),
        "pf_target" => pf_target, "pf_start" => θ₀pf,
        "pf_paper_fixed_eta" => Dict("E_final" => rC.E, "iters_to_0.02" => rC.it_conv),
        "pf_code_fixed_eta" => Dict("E_final" => rD.E, "iters_to_0.02" => rD.it_conv),
        "pf_paper_backtracking" => Dict("E_final" => rC2.E, "iters_to_0.02" => rC2.it_conv),
        "pf_less_targets_note" => "|P|=1 < |D|=2: measure term active; gradient by FD; the pitchfork model has at most one bifurcation so E cannot vanish; the trajectory records the annealing term's actual effect",
        "manifold_note" => "theta2 = -4*theta1^3/(27 p*^2) keeps both folds at ±|p*| (derived analytically): E_paper < 1e-5 along the whole family (Newton-level fold residuals ~1e-12 are amplified to E ~ sqrt(3e-12) ~ 1.7e-6 by the geometric-mean sqrt tail) — the paper's manifold of geometrically equivalent models, verified numerically",
        "restarts_converged_out_of_8" => nconv,
        "ok" => ok,
    ))
    return (rA=rA, rB=rB, rA2=rA2, rB2=rB2, rC=rC, rD=rD, rC2=rC2,
            manifold=manifold_rows, restarts=restarts, nconv=nconv, ok=ok)
end

# ============================================ p10: probe adjoint (∇Ψ) vs FD
function probe_p10()
    # Analytic Leibniz adjoint of the paper's normalised main-text measure
    # (Eq 5 + Eq 7, indicator window) vs central FD of Ψ itself, with moving
    # window-intersection boundaries recomputed per perturbed θ; FD-step and
    # quadrature sweeps separate FD error from quadrature error (a
    # discretisation-derivative vs continuum-derivative distinction at
    # probe level), and the boundaries=false variant demonstrates that the
    # endpoint terms are required.
    θ = [1.0, -1.0]
    rows = []
    adj_all = Dict{String,Any}()
    for (pmin, pmax) in ((-2.0, 2.0), (-0.6, 1.6))
        adj = ∇Ψ_sn(θ, pmin, pmax; n=4000)
        noB = ∇Ψ_sn(θ, pmin, pmax; n=4000, boundaries=false)
        for k in 1:2
            for h in (1e-2, 1e-3, 1e-4, 1e-5)
                for n in (1000, 4000, 16000)
                    θp = copy(θ); θp[k] += h
                    θm = copy(θ); θm[k] -= h
                    fd = (simpson_Ψ(SNModel(), θp, φ_eq5, pmin, pmax; n=n).Ψ -
                          simpson_Ψ(SNModel(), θm, φ_eq5, pmin, pmax; n=n).Ψ)/(2h)
                    push!(rows, [pmin, pmax, k, h, n, fd, adj.∇[k],
                                 abs(fd - adj.∇[k]), abs(fd - noB.∇[k])])
                end
            end
        end
        adj_all["win_$(pmin)_$(pmax)"] = Dict(
            "Psi" => adj.Ψ, "grad_adjoint" => adj.∇,
            "grad_no_boundaries" => noB.∇,
            "boundary_N" => adj.boundary[1], "boundary_L" => adj.boundary[2],
            "interior_N" => adj.interior[1], "interior_L" => adj.interior[2],
            "max_absdiff_vs_fd" => maximum(r[8] for r in rows if
                r[1] == pmin && r[2] == pmax && r[5] == 4000 && r[4] == 1e-4),
            "min_absdiff_vs_fd_no_boundaries" => minimum(r[9] for r in rows if
                r[1] == pmin && r[2] == pmax && r[5] == 4000 && r[4] == 1e-4),
        )
    end
    write_csv("p10_probe_adjoint.csv",
        ["pmin", "pmax", "theta_k", "fd_h", "quadrature_n", "fd_grad",
         "adjoint_grad", "absdiff", "absdiff_no_boundaries"], rows)
    worst = maximum(r[8] for r in rows if r[4] == 1e-4 && r[5] == 4000)
    worst_nob = minimum(r[9] for r in rows if r[4] == 1e-4 && r[5] == 4000)
    ok = worst < 1e-6 && worst_nob > 0.05
    write_toml("p10_probe_adjoint.toml", Dict(
        "theta" => θ,
        "note" => "analytic Leibniz adjoint (differentiation under the integral at fixed curve parameter u + explicit moving-window-endpoint terms u_edge'(θ) = -(∂p/∂θ)/(∂p/∂u)) vs central FD of Ψ with window intersections recomputed per perturbed θ. The no-boundaries variant differs from FD by 0.1-0.3 (and flips gradient signs in the measured cases): window-endpoint motion is not a small correction. FD-step and quadrature sweeps separate FD truncation (shrinks with h) from quadrature error (shrinks with n).",
        "windows" => adj_all,
        "worst_absdiff_adjoint_vs_fd_h1e-4_n4000" => worst,
        "min_absdiff_no_boundaries_vs_fd_h1e-4_n4000" => worst_nob,
        "ok" => ok,
    ))
    return (rows=rows, worst=worst, worst_nob=worst_nob, ok=ok)
end

# ==================================================================== driver
function main()
    println("== experiment 01 equilibria probes (stdlib only) ==")
    results = Dict{String,Any}()
    results["p1"] = timed("p1_analytic", probe_p1)
    results["p2"] = timed("p2_gradients", probe_p2)
    results["p2c"] = timed("p2c_deformation", probe_p2c)
    results["p3"] = timed("p3_normalisation", probe_p3)
    results["p4"] = timed("p4_distances", probe_p4)
    results["p5"] = timed("p5_measures", probe_p5)
    results["p6"] = timed("p6_edgecases", probe_p6)
    results["p7"] = timed("p7_degenerate", probe_p7)
    results["p8"] = timed("p8_inference", probe_p8)
    results["p10"] = timed("p10_probe_adjoint", probe_p10)
    total = time() - T0
    cpu = Sys.cpu_info()
    write_toml("runtime.toml", Dict(
        "julia_version" => string(VERSION),
        "machine" => Sys.MACHINE,
        "arch" => string(Sys.ARCH),
        "cpu_model" => cpu[1].model,
        "logical_cpus" => length(cpu),
        "blas_config" => string(LinearAlgebra.BLAS.get_config()),
        "threads" => Dict("JULIA_NUM_THREADS" => 1, "OPENBLAS_NUM_THREADS" => 1,
                          "OMP_NUM_THREADS" => 1),
        "seed" => SEED,
        "total_seconds" => round(total; digits=3),
        "probe_timings_seconds" => Dict(TIMINGS),
        "date_utc" => string(Dates.now(Dates.UTC)),
    ))
    summary = Dict{String,Any}()
    for (k, v) in results
        summary[k] = get(v, :ok, missing)
    end
    summary["total_seconds"] = round(total; digits=3)
    write_toml("summary.toml", summary)
    for (k, v) in results
        @printf("  %-4s ok=%s\n", k, get(v, :ok, missing))
    end
    @printf("total %.1fs\n", total)
    if !all(get(v, :ok, false) for v in values(results))
        println("PROBE FAILURE PRESENT")
        exit(1)
    end
end

main()
