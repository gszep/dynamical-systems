# Branch seed/discovery protocol.
#
# This is a discovery mechanism, NOT an exhaustive atlas: a fixed, fully
# recorded seed table is run through damped Newton with deflation; every
# attempt (converged or not) is recorded with its outcome, and the summary
# states counts only. Guards reject the trivial state and NaN results, and
# duplicates are detected up to the sign symmetry u -> -u (the equation is
# odd: -U is a solution whenever U is).
#
# All seeds are projected onto the reflection-symmetric (even) subspace before
# Newton (to_coeffs symmetrizes). That restriction is a PHASE CONDITION, not a
# quotient of the translation group: it keeps the Newton system nonsingular
# for solutions whose translation orbit meets the subspace, it EXCLUDES
# solutions with no even translate from the search scope (see transforms.jl),
# and distinct even translates of one pattern remain distinct points -
# dedupe identifies only the sign symmetry u -> -u, not general translations.
# The seed families are built from cosines, even bumps at x = 0 or L/2, and
# cosine-only random fields, so the projection leaves them unchanged.

struct SeedSpec
    family::String
    tag::String
    kind::Symbol            # :cosk, :localized, :random, :trivial_probe, :highfreq
    A::Float64              # amplitude (cos/localized/highfreq)
    k::Float64              # wavenumber (cosk/highfreq)
    w::Float64              # bump width (localized)
    center::Float64         # bump center (localized)
    rng_seed::UInt64        # for :random and :trivial_probe
    r::Float64              # discovery parameter for this attempt
    rms_target::Float64     # for :random
end

function seed_table(cfg::SHConfig, r::Float64)
    seeds = SeedSpec[]
    for A in cfg.seedA_cos
        for k in cfg.seedK_cos
            push!(seeds, SeedSpec("cosk", @sprintf("cosk_k%.2g_A%.3g", k, A), :cosk,
                                  A, k, 0.0, 0.0, UInt64(0), r, 0.0))
        end
    end
    for A in cfg.seedA_loc
        for w in cfg.seedW_loc
            for c0 in cfg.seedC_loc
                push!(seeds, SeedSpec("localized", @sprintf("loc_A%.3g_w%.3g_c%.3g", A, w, c0),
                                      :localized, A, 0.0, w, c0, UInt64(0), r, 0.0))
            end
        end
    end
    for (i, sid) in enumerate(cfg.random_seed_ids)
        target = cfg.random_targets[mod1(i, length(cfg.random_targets))]
        push!(seeds, SeedSpec("random", @sprintf("rand_%d", Int(sid)), :random,
                              0.0, 0.0, 0.0, 0.0, UInt64(sid), r, target))
    end
    push!(seeds, SeedSpec("trivial_probe", "trivial_probe", :trivial_probe,
                          0.0, 0.0, 0.0, 0.0, UInt64(12345), r, 0.0))
    push!(seeds, SeedSpec("highfreq", "highfreq_k3_A0.5", :highfreq,
                          0.5, 3.0, 0.0, 0.0, UInt64(0), r, 0.0))
    return seeds
end

function seed_values(disc::Discretization, cfg::SHConfig, seed::SeedSpec)
    N = disc.N
    if seed.kind === :cosk
        return [seed.A * cos(seed.k * disc.x[j]) for j in 1:N]
    elseif seed.kind === :localized
        return [seed.A / cosh((disc.x[j] - seed.center) / seed.w)^2 for j in 1:N]
    elseif seed.kind === :random
        rng = TinyRNG(seed.rng_seed)
        u = zeros(N)
        for n in 1:cfg.random_band_modes
            a = (rand(rng) - 0.5) * 2.0 / (n^1.5)
            k = 2.0 * pi * n / disc.L
            for j in 1:N
                u[j] += a * cos(k * disc.x[j])
            end
        end
        cur = norm(u) / sqrt(N)
        if cur > 1.0e-12
            u .*= (seed.rms_target / cur)
        end
        return u
    elseif seed.kind === :trivial_probe
        rng = TinyRNG(seed.rng_seed)
        return [1.0e-6 * (rand(rng) - 0.5) for _ in 1:N]
    elseif seed.kind === :highfreq
        return [seed.A * cos(seed.k * disc.x[j]) for j in 1:N]
    end
    error("unknown seed kind: $(seed.kind)")
end

# Distance for duplicate detection: relative L2 distance, minimized over the
# sign symmetry u -> -u (which maps solutions to solutions).
function dedup_distance(c::Vector{Float64}, ck::Vector{Float64})
    return min(norm(c .- ck), norm(c .+ ck)) / max(norm(c), norm(ck), 1.0e-12)
end

function is_duplicate(c::Vector{Float64}, known::Vector{Vector{Float64}}, tol::Float64)
    return any(ck -> dedup_distance(c, ck) < tol, known)
end

rel_dist(a::Vector{Float64}, b::Vector{Float64}) =
    norm(a .- b) / max(norm(a), norm(b), 1.0e-12)

# Guard/classification of a converged Newton result.
# Returns (accepted, is_new, reason):
#   "nan"       -> non-finite solution, rejected
#   "trivial"   -> converged to the trivial state, rejected (contamination guard)
#   "duplicate" -> same (up to sign) as a known solution, accepted but not new
#   "new"       -> a new distinct solution
function classify_solution(disc::Discretization, cfg::SHConfig, res,
                           known::Vector{Vector{Float64}})
    if !all(isfinite, res.c)
        return (false, false, "nan")
    end
    rms = rms_of_c(disc, res.c)
    if rms < cfg.rms_trivial_disc
        return (false, false, "trivial")
    end
    if is_duplicate(res.c, known, cfg.dedup_tol)
        return (true, false, "duplicate")
    end
    return (true, true, "new")
end

"""
    run_discovery(disc, cfg, r; fl = nothing)

Runs the fixed seed table at parameter r. Every attempt is recorded. Classical
root-removing deflation (see newton.jl) is enabled so later seeds cannot
re-converge to already-found solutions; convergence is always judged on the
true residual. Returns a NamedTuple with `records`, `rows` (CSV rows), `known`
(new solution coefficient vectors), `known_rms`, `seeds`, `r`, and `n_failures`.
"""
function run_discovery(disc::Discretization, cfg::SHConfig, r::Float64;
                       fl::Union{Nothing,FailLog}=nothing)
    seeds = seed_table(cfg, r)
    records = Vector{Any}()
    rows = Vector{Vector{Any}}()
    known = Vector{Vector{Float64}}()
    known_rms = Float64[]
    n_failures = 0
    for (idx, sd) in enumerate(seeds)
        n_defl_before = length(known)
        u_seed = seed_values(disc, cfg, sd)
        c0 = to_coeffs(disc, u_seed)
        res = newton_even(disc, cfg, c0, r; known=known)
        accepted = false
        is_new = false
        reason = ""
        rms = NaN
        cvec = Float64[]
        desc = empty_descriptors()
        specsub = empty_spectrum_summary()
        specfull = empty_spectrum_summary()
        if !res.converged
            reason = "not_converged:" * res.reason
            n_failures += 1
            if fl !== nothing
                record!(fl, Any["discovery", sd.tag, "-", r, disc.N, idx, res.iters, "-",
                                res.residual, res.reason])
            end
        else
            cvec = res.c
            u = disc.S * cvec
            rms = norm(u) / sqrt(disc.N)
            accepted, is_new, reason = classify_solution(disc, cfg, res, known)
            if accepted
                desc = descriptors(disc, u)
                specsub = spectrum_even(disc, cfg, cvec, r)
                specfull = spectrum_full(disc, cfg, u, r)
                if is_new
                    push!(known, copy(cvec))
                    push!(known_rms, rms)
                end
            end
        end
        rec = (spec=sd, idx=idx, converged=res.converged, accepted=accepted, is_new=is_new,
               reason=reason, iters=res.iters, ndefl=n_defl_before, residual=res.residual,
               c=cvec, rms=rms, desc=desc, specsub=specsub, specfull=specfull)
        push!(records, rec)
        push!(rows, Any[rec.spec.r, rec.idx, rec.spec.family, string(rec.spec.kind),
                        rec.spec.tag, rec.converged, rec.accepted, rec.is_new, rec.reason,
                        rec.iters, rec.ndefl, rec.residual, rec.rms, rec.desc.mean,
                        rec.desc.kbar, rec.desc.pr, rec.desc.f0, rec.desc.f1, rec.desc.f2,
                        rec.desc.f3, rec.desc.f4, rec.desc.nz50, rec.specsub.lambda_max,
                        rec.specsub.n_pos, rec.specfull.lambda_max, rec.specfull.n_pos])
    end
    return (records=records, rows=rows, known=known, known_rms=known_rms,
            seeds=seeds, r=r, n_failures=n_failures)
end
