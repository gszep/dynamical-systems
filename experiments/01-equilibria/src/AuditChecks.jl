module AuditChecks
export finite!, required_call, option, plateau, tolerance

function finite!(x, label)
    good = x isa Number ? isfinite(x) : all(isfinite, x)
    good || error("$label: non-finite required output")
    return x
end

# Injection is applied inside the real call site's try/catch, never appended
# to the failure list after the run. Numerical corruptions use the same
# finite-output check as real results.
function required_call(f, site; injection="none")
    injection == "$site-exception" && error("injected $site exception")
    value = f()
    injection == "$site-nan" && return NaN
    injection == "$site-inf" && return Inf
    return value
end

function option(args, flag, default)
    i = findfirst(==(flag), args)
    isnothing(i) && return default
    i < length(args) || error("missing value for $flag")
    return args[i+1]
end

# Accuracy goal, NOT an estimate inferred from the measured mismatch.
# 0.1% componentwise accuracy plus an absolute 1e-8 floor. Convergence
# diagnostics must resolve this goal; they cannot inflate it until green.
tolerance(reference) = 1e-8 + 1e-3*abs(reference)

function plateau(values, reference)
    length(values) >= 3 || return false
    all(isfinite, values) || return false
    tail = values[end-2:end]
    return maximum(tail) - minimum(tail) <= tolerance(reference)
end
end
