# Gradient audit v2. Old raw p10 artifacts remain untouched and superseded.
# Run separate bounded processes (one at a time): --mode reference,
# --mode sweep --ds {0.02,0.01,0.005,0.0025}, then --mode assess.
# Default mode assess NEVER runs missing jobs or treats missing data as green.
# No package setup. No source patches to the released package.
using LinearAlgebra, StaticArrays, TOML, Dates, Random, SHA
using SpecialFunctions: erf
import BifurcationInference as BI
include(joinpath(@__DIR__, "src", "EquilibriaProbes.jl"))
using .EquilibriaProbes
include(joinpath(@__DIR__, "src", "AuditChecks.jl"))
using .AuditChecks

const OUT = mkpath(option(ARGS, "--out-dir",
    joinpath(@__DIR__, "../../results/01-equilibria/gradient-audit-v2")))
const MODE = option(ARGS, "--mode", "assess")
const INJECTION = option(ARGS, "--inject", "none")
const THETA = [1.0, -1.0]
const LO, HI = -2.0, 2.0
const HS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
const DSS = [0.02, 0.01, 0.005, 0.0025]
const SEED = 20261019
const START = time()
const CODE_HASH = bytes2hex(sha256(read(@__FILE__)))

function save(name, d)
    d = Dict{String,Any}(d)
    d["code_sha256"] = CODE_HASH
    d["julia"] = string(VERSION)
    d["utc"] = string(now(UTC))
    d["execution_seconds_since_imports"] = time() - START
    open(joinpath(OUT, name), "w") do io
        TOML.print(io, d; sorted=true)
    end
end

# Exact scalar version of the released integrand. Unlike the earlier
# eigen_ds reference, it introduces no nested FD noise into outer θ-FDs.
# dλ/ds = 6θ₂u/ν (orientation irrelevant inside abs); ν = ds/du.
function integrand(u, θ)
    D = θ[1] + 3θ[2]*u^2
    ν = sqrt(1 + D^2)
    slope = 6θ[2]*u/ν
    den = 2abs(slope) + abs(sinh(2D))
    den > 0 || error("degenerate scalar measure")
    φ = 2abs(slope)/den
    p = -θ[1]*u - θ[2]*u^3
    return window_erf(p, LO, HI; erf=erf)*φ*ν
end

function edges(θ)
    rs = sort(vcat(sn_edge_roots(θ, LO), sn_edge_roots(θ, HI)))
    length(rs) == 2 || error("physical window must have two transverse intersections")
    all(abs(sn_det_curve(u, θ)) > 1e-8 for u in rs) || error("tangent window edge")
    return rs
end

function integrate_between(θ, a, b, n)
    # Exact kink locations for this scalar SN model, no moving-root FD.
    folds = sqrt(-θ[1]/(3θ[2]))
    knots = sort(vcat([a,b], filter(u -> a < u < b, [-folds, 0.0, folds])))
    return sum(EquilibriaProbes.simpson(u -> integrand(u, θ), x, y, n)
        for (x,y) in zip(knots[1:end-1], knots[2:end]))
end

function continuum(θ, n; trunc=true, radius=4.0)
    a,b = trunc ? edges(θ) : (-radius,radius)
    return finite!(integrate_between(θ, a, b, n), "continuum integral")
end

function fd(f, h)
    return [begin
        a,b = copy(THETA),copy(THETA); a[k]+=h; b[k]-=h
        finite!((f(a)-f(b))/(2h), "finite difference component $k")
    end for k in 1:2]
end

function reference()
    rows = Dict{String,Any}[]
    for n in (512,2048,8192), h in HS, trunc in (false,true)
        g = fd(θ -> continuum(θ,n; trunc=trunc), h)
        push!(rows, Dict("n"=>n,"h"=>h,"truncated"=>trunc,"gradient"=>g))
    end
    a,b = edges(THETA)
    # Fixed-u Leibniz endpoint terms: f(b)b' - f(a)a'.
    explicit = [integrand(b,THETA)*sn_edge_du_dθ(b,THETA,k) -
                integrand(a,THETA)*sn_edge_du_dθ(a,THETA,k) for k in 1:2]
    # Normal-deformation form uses relative tangent flux at the boundary.
    # v_normal·t=0; actual endpoint velocity=(u_edge',0), t=(1,-D)/ν.
    # Hence flux is [φ_window*u_edge'/ν], not the fixed-u term above.
    normal_flux = [integrand(b,THETA)*sn_edge_du_dθ(b,THETA,k)/sn_ν(b,THETA)^2 -
                   integrand(a,THETA)*sn_edge_du_dθ(a,THETA,k)/sn_ν(a,THETA)^2
                   for k in 1:2]
    moving = fd(θ -> continuum(θ,8192), 1e-4)
    fixed = fd(θ -> integrate_between(θ,a,b,8192), 1e-4)
    boundary_residual = moving-fixed-explicit
    # Normalized main-text Ψ is a DIFFERENT functional; preserve its gate.
    ψadj = ∇Ψ_sn(THETA,LO,HI; n=4000)
    ψfd = fd(θ -> simpson_Ψ(SNModel(),θ,φ_eq5,LO,HI;n=4000).Ψ,1e-4)
    ψerror = finite!(norm(ψfd-ψadj.∇),"normalized Psi derivative error")
    ψerror < 1e-6 || error("normalized Psi derivative failed")
    maximum(abs.(boundary_residual)) < 1e-7 || error("explicit boundary identity failed")
    # Check our analytic scalar integrand against the actual package value.
    X = BI.StateSpace(1, LO:0.02:HI, [-0.5,0.5])
    point_errors = [begin
        z = curve_points(SNModel(),u,THETA)
        point = BI.BorderedArray(SizedVector{1}(u),z[2])
        abs(BI.measure(F_sn,point,THETA,X)*sn_ν(u,THETA)-integrand(u,THETA))
    end for u in (-1.4,-0.8,-0.3,0.2,0.7,1.3)]
    maximum(point_errors) < 1e-10 || error("reference integrand differs from released measure")
    save("reference.toml",Dict("status"=>"measured", "rows"=>rows,
        "theta"=>THETA,"window"=>[LO,HI],"point_errors"=>point_errors,
        "explicit_fixed_u_boundary"=>explicit,"normal_deformation_boundary_flux"=>normal_flux,
        "boundary_identity_residual"=>boundary_residual,
        "fixed_endpoints_fd"=>fixed,"moving_endpoints_fd"=>moving,
        "psi_gradient"=>ψadj.∇,"psi_fd"=>ψfd,"psi_error"=>ψerror,
        "radius4_vs5_gradient_delta"=>fd(θ->continuum(θ,8192;trunc=false,radius=5.0),1e-4)-
                                       fd(θ->continuum(θ,8192;trunc=false),1e-4),
        "boundary_note"=>"Tail-difference derivatives are not endpoint omission bounds. Fixed-u Leibniz boundary and normal-deformation relative flux are explicitly computed separately."))
end

F_sn(z::BI.BorderedArray, θ::AbstractVector) = F_sn(z.u,(θ=θ,p=z.p))
function F_sn(u::AbstractVector, pars::NamedTuple)
    v = pars.p+pars.θ[1]*u[1]+pars.θ[2]*u[1]^3
    out = similar(u,typeof(v)); out[1]=v; out
end

function package_value(θ, ds; with_gradient=false)
    # Fresh roots for EVERY sample: no history/shared mutable seed fixture.
    Random.seed!(SEED)
    X = BI.StateSpace(1,LO:ds:HI,[-0.5,0.5])
    branches = BI.deflationContinuation(F_sn,X.roots,(θ=copy(θ),p=LO),BI.getParameters(X))
    isempty(branches) && error("empty branch set")
    value = finite!(BI.measure(F_sn,branches,θ,X),"returned finite-sum measure")
    meta = [Dict("points"=>length(b),"first_u"=>first(b).z.u[1],
        "last_u"=>last(b).z.u[1],"first_p"=>first(b).z.p,"last_p"=>last(b).z.p,
        "sum_ds"=>sum(s.ds for s in b),"bif_flags"=>count(s->s.bif,b),
        "min_abs_det"=>minimum(abs(sn_det_curve(s.z.u[1],θ)) for s in b),
        "max_residual"=>maximum(norm(F_sn(s.z,θ)) for s in b)) for b in branches]
    if with_gradient
        g = required_call(()->collect(BI.∇measure(F_sn,branches,θ,X)),"gradient";injection=INJECTION)
        finite!(g,"released ds gradient")
        length(g)==2 || error("gradient dimension")
        return value,meta,g
    end
    return value,meta
end

function sweep(ds)
    ds in DSS || error("ds must be one of $DSS")
    rows = Dict{String,Any}[]
    # Injections at FD site run before baseline to exercise the exact catch
    # route cheaply; no synthetic post-hoc failure-list test.
    baseline = nothing
    for h in HS, k in 1:2
        a,b=copy(THETA),copy(THETA); a[k]+=h; b[k]-=h
        try
            plus = required_call(()->package_value(a,ds),"fd";injection=INJECTION)
            minus = required_call(()->package_value(b,ds),"fd";injection=INJECTION)
            plus isa Tuple && minus isa Tuple || error("FD call non-finite or invalid return")
            vp,mp=plus; vm,mm=minus
            finite!(vp,"FD plus value"); finite!(vm,"FD minus value")
            g=finite!((vp-vm)/(2h),"FD quotient")
            # Conservative arithmetic summation/cancellation proxy only,
            # NOT a solver-error bound. Must not be used to enlarge gates.
            n=sum(x["points"] for x in mp)+sum(x["points"] for x in mm)
            roundoff=64eps(Float64)*n*(abs(vp)+abs(vm))/(2h)
            push!(rows,Dict("h"=>h,"component"=>k,"fd"=>g,
                "plus_value"=>vp,"minus_value"=>vm,"plus"=>mp,"minus"=>mm,
                "roundoff_proxy"=>roundoff))
        catch e
            save("sweep-$ds.toml",Dict("status"=>"FAILED","ds"=>ds,"rows"=>rows,
                "failed_site"=>"fd","error"=>sprint(showerror,e)))
            rethrow()
        end
    end
    value,meta,g=package_value(THETA,ds;with_gradient=true)
    save("sweep-$ds.toml",Dict("status"=>"measured","ds"=>ds,"rows"=>rows,
        "baseline_value"=>value,"baseline"=>meta,"released_gradient"=>g,"seed"=>SEED))
end

function assess()
    files=vcat(["reference.toml"],["sweep-$ds.toml" for ds in DSS])
    all(isfile(joinpath(OUT,f)) for f in files) || error("INCONCLUSIVE: missing bounded sweep jobs")
    ref=TOML.parsefile(joinpath(OUT,files[1]))
    sweeps=[TOML.parsefile(joinpath(OUT,f)) for f in files[2:end]]
    all(d["status"]=="measured" for d in vcat([ref],sweeps)) || error("FAILED: required job failed")
    # Preserve completed raw jobs: the evaluator can be corrected without
    # pretending those jobs were rerun. Require all raw jobs to share a source
    # hash and record that hash separately from this evaluator's current hash.
    raw_hash = ref["code_sha256"]
    all(d["code_sha256"]==raw_hash for d in sweeps) || error("mixed raw-job code hashes")
    gates=Dict{String,Any}[]
    for k in 1:2
        lookup(n,h,tr)=only(r["gradient"][k] for r in ref["rows"]
            if r["n"]==n && r["h"]==h && r["truncated"]==tr)
        c=lookup(8192,1e-4,false)
        tol=tolerance(c)
        continuum_h=[lookup(8192,h,false) for h in HS[end-2:end]]
        q=[lookup(n,1e-4,false) for n in (512,2048,8192)]
        # Reference numerical variation must use <=10% of accuracy budget.
        reference_ok=all(isfinite,vcat(continuum_h,q)) &&
            maximum(abs.(continuum_h .- c))<=0.1tol &&
            abs(q[end]-q[end-1])<=0.1tol &&
            abs(q[end]-q[end-1])<=abs(q[end-1]-q[end-2])+1e-10 &&
            abs(ref["radius4_vs5_gradient_delta"][k])<=0.1tol
        fdvalues=Float64[]; gradients=Float64[]; plateau_ok=Bool[]
        per_ds=Dict{String,Any}[]
        for d in sweeps
            rr=sort(filter(r->r["component"]==k,d["rows"]);by=r->-r["h"])
            vals=[r["fd"] for r in rr]
            p=plateau(vals,c)
            roundoff_ok=maximum(r["roundoff_proxy"] for r in rr[end-2:end])<=0.1tol
            p=p && roundoff_ok
            v=vals[end]; g=d["released_gradient"][k]
            finite!([v,g],"assessment derivatives")
            push!(fdvalues,v); push!(gradients,g); push!(plateau_ok,p)
            push!(per_ds,Dict("ds"=>d["ds"],"h_values"=>HS,"fd_values"=>vals,
                "fd_plateau"=>p,"arithmetic_proxy_resolved"=>roundoff_ok,
                "gradient_minus_returned_sum_fd"=>g-v,
                "same_discrete_object_gate"=>p ? (abs(g-v)<=tol ? "PASS" : "FAILED") : "INCONCLUSIVE"))
        end
        stable_fd=all(plateau_ok[end-2:end]) && plateau(fdvalues,c)
        stable_gradient=plateau(gradients,c)
        continuum_gate= !reference_ok || !stable_gradient ? "INCONCLUSIVE" :
            (abs(gradients[end]-c)<=tol ? "PASS" : "FAILED")
        discrete_gate= !stable_fd ? "INCONCLUSIVE" :
            (abs(gradients[end]-fdvalues[end])<=tol ? "PASS" : "FAILED")
        push!(gates,Dict("component"=>k,"continuum_reference"=>c,"tolerance"=>tol,
            "reference_resolved"=>reference_ok,"released_gradient_by_ds"=>gradients,
            "returned_sum_fd_by_ds"=>fdvalues,"gradient_resolution_plateau"=>stable_gradient,
            "fd_resolution_plateau"=>stable_fd,"continuum_gate"=>continuum_gate,
            "discrete_gate"=>discrete_gate,"per_ds"=>per_ds))
    end
    passed=all(g["continuum_gate"]=="PASS" && g["discrete_gate"]=="PASS" for g in gates)
    save("assessment.toml",Dict("status"=>passed ? "PASS" : "INCONCLUSIVE_OR_FAILED",
        "raw_jobs_code_sha256"=>raw_hash,
        "components"=>gates,"required_checks_passed"=>passed,
        "criterion"=>"Component accuracy 0.1% + 1e-8 absolute; last three h and ds values must plateau within that fixed budget. Reference and roundoff-proxy variations <=10% of budget. These are empirical resolution checks, not certified bounds; no observed mismatch inflates tolerances.",
        "diagnosis_limit"=>"Counts/endpoints/weights are recorded for every perturbation. Changes indicate re-discretization; they alone do not prove nondifferentiability or identify a unique mechanism."))
    for g in gates
        println("component ",g["component"],": continuum=",g["continuum_gate"],
                " returned_sum=",g["discrete_gate"])
    end
    passed || error("required derivative discrimination is INCONCLUSIVE or FAILED")
end

try
    MODE=="reference" ? reference() : MODE=="sweep" ? sweep(parse(Float64,option(ARGS,"--ds","0.02"))) :
        MODE=="assess" ? assess() : error("unknown mode $MODE")
catch e
    save("failure-$MODE.toml",Dict("status"=>"FAILED","mode"=>MODE,
        "injection"=>INJECTION,"error"=>sprint(showerror,e)))
    println(stderr,sprint(showerror,e))
    exit(1)
end
