# Artifact I/O: CSV writers, a structured failure log, and TOML output.
#
# CSV values are pre-formatted: floats via %.10g (non-finite -> "NA"), bools as
# true/false, everything else via string(). TOML output uses TOML.print from
# the stdlib; a small fallback writer is included (and tested) so that the
# experiment does not depend on TOML.print being available. Non-finite floats
# are sanitized to the string "NA" before they reach TOML, and vector-valued
# diagnostics are emitted as homogeneous string arrays.

mutable struct FailLog
    rows::Vector{Vector{Any}}
end

FailLog() = FailLog(Vector{Vector{Any}}())

function record!(fl::FailLog, vals::Vector{Any})
    push!(fl.rows, vals)
    return fl
end

fmt_val(v::Float64) = isfinite(v) ? @sprintf("%.10g", v) : "NA"
fmt_val(v::Int) = string(v)
fmt_val(v::Bool) = v ? "true" : "false"
fmt_val(v::UInt64) = string(v)
fmt_val(v::Symbol) = string(v)
fmt_val(v::AbstractString) = String(v)
fmt_val(::Nothing) = "-"
fmt_val(v) = string(v)

function write_csv(io::IO, header::Vector{String}, rows::Vector{Vector{Any}})
    println(io, join(header, ","))
    for row in rows
        println(io, join([fmt_val(v) for v in row], ","))
    end
    return nothing
end

function write_csv(path::AbstractString, header::Vector{String}, rows::Vector{Vector{Any}})
    open(path, "w") do io
        write_csv(io, header, rows)
    end
    return nothing
end

# Sanitize scalar floats for TOML (TOML floats cannot carry our NaN sentinels).
toml_safe(v::Float64) = isfinite(v) ? v : "NA"
toml_safe(v) = v

function dump_toml(io::IO, data::Dict{String,Any})
    if isdefined(TOML, :print)
        TOML.print(io, data)
    else
        _toml_fallback_print(io, data)
    end
    return nothing
end

function dump_toml(path::AbstractString, data::Dict{String,Any})
    open(path, "w") do io
        dump_toml(io, data)
    end
    return nothing
end

# ---- fallback TOML writer (supports our summary data shapes) ----

function _toml_escape(s::AbstractString)
    s = replace(s, "\\" => "\\\\")
    s = replace(s, "\"" => "\\\"")
    return s
end

function _toml_scalar_str(v)
    if v isa Bool
        return v ? "true" : "false"
    elseif v isa Integer
        return string(v)
    elseif v isa AbstractFloat
        return isfinite(v) ? string(v) : "nan"
    elseif v isa AbstractString
        return "\"" * _toml_escape(v) * "\""
    else
        error("unsupported scalar in fallback TOML writer: $(typeof(v))")
    end
end

function _toml_kv_line(io::IO, k::String, v)
    if v isa AbstractVector
        println(io, k * " = [" * join([_toml_scalar_str(x) for x in v], ", ") * "]")
    elseif v isa AbstractDict
        error("unexpected table in scalar pass for key $k")
    else
        println(io, k * " = " * _toml_scalar_str(v))
    end
    return nothing
end

function _toml_write_section(io::IO, data::Dict{String,Any}, path::Vector{String})
    for (k, v) in data
        if !(v isa AbstractDict)
            _toml_kv_line(io, k, v)
        end
    end
    for (k, v) in data
        if v isa AbstractDict
            sub = vcat(path, [String(k)])
            println(io)
            println(io, "[" * join(sub, ".") * "]")
            _toml_write_section(io, v, sub)
        end
    end
    return nothing
end

function _toml_fallback_print(io::IO, data::Dict{String,Any})
    _toml_write_section(io, data, String[])
    return nothing
end
