# Julia 1.10 lacks --compiled-modules=existing. Its package loader calls this
# two-positional-argument entry point on a cache miss (base/loading.jl:2058).
# Leave cache reads enabled, but fail rather than precompile missing packages.
# Scoped to this process; does not edit Julia or any installed package source.
@eval Base function compilecache(pkg::PkgId, path::String; loadable_exts=nothing)
    error("cache-only audit: precompilation prohibited for $(pkg.name)")
end
