# Grid, exact dense DFT, reflection symmetry, and the reflection-invariant
# (even/cosine) restriction used for Newton and continuation.
#
# Grid: N even (N >= 8), x_j = (j-1)*h, h = L/N, j = 1..N, periodic.
# DFT:   c = E * u  with  E[j,n] = exp(-2*pi*i*(j-1)(n-1)/N),  u = Ein * c
#        with Ein = E'/N (exact matrix inverse, since E E' = N I).
# Wavenumber of DFT index n: k = 2*pi*(n-1)/L for n-1 <= N/2, else
#        k = 2*pi*(n-1-N)/L (negative frequencies).
#
# Even restriction: solutions are searched in the subspace invariant under the
# grid reflection x -> -x (indices j -> 1 + mod(N-(j-1), N)). That subspace is
# spanned by the Nc = N/2 + 1 cosine vectors S[:, n] = cos(2*pi*(n-1)(j-1)/N).
# Because the equation commutes with translations, every nontrivial periodic
# stationary state has a neutral phase mode in the full space; searching inside
# the reflection-invariant subspace is an explicit phase pinning (reflection
# gauge) that removes the translation orbit. Consequences for the spectrum are
# documented in spectrum.jl and the experiment README; no full-space
# asymptotic-stability claims are made anywhere.

struct Discretization
    L::Float64
    N::Int
    Nc::Int
    h::Float64
    x::Vector{Float64}          # grid points, length N
    kfull::Vector{Float64}       # wavenumbers of DFT indices 1..N (negatives included)
    mfull::Vector{Float64}       # (1 - k^2)^2 for DFT indices
    kcos::Vector{Float64}        # cosine wavenumbers, length Nc
    mcos::Vector{Float64}        # (1 - k^2)^2 for cosine modes
    E::Matrix{ComplexF64}        # forward DFT matrix
    Ein::Matrix{ComplexF64}      # inverse DFT matrix (E'/N)
    Mmat::Matrix{Float64}        # value-space matrix of (1 + d_xx)^2
    S::Matrix{Float64}           # cosine evaluation matrix, N x Nc
    Gdiag::Vector{Float64}       # diag(S' * S)
    Ginv::Vector{Float64}        # 1 ./ Gdiag
end

function Discretization(L::Float64, N::Int)
    if N < 8 || N % 2 != 0
        error("N must be even and >= 8 (got N = $N)")
    end
    h = L / N
    x = [h * (j - 1) for j in 1:N]
    half = N ÷ 2
    kfull = Vector{Float64}(undef, N)
    mfull = Vector{Float64}(undef, N)
    for n in 1:N
        m = n - 1
        k = m <= half ? 2.0 * pi * m / L : 2.0 * pi * (m - N) / L
        kfull[n] = k
        mfull[n] = (1.0 - k * k)^2
    end
    Nc = half + 1
    kcos = [2.0 * pi * (n - 1) / L for n in 1:Nc]
    mcos = [(1.0 - k * k)^2 for k in kcos]
    E = Matrix{ComplexF64}(undef, N, N)
    for j in 1:N
        for n in 1:N
            E[j, n] = cis(-2.0 * pi * (j - 1) * (n - 1) / N)
        end
    end
    Ein = collect(E') / N
    Mmat = real(Ein * (Diagonal(mfull) * E))
    Mmat = (Mmat + transpose(Mmat)) * 0.5   # kill O(eps) roundoff asymmetry
    S = Matrix{Float64}(undef, N, Nc)
    for j in 1:N
        for n in 1:Nc
            S[j, n] = cos(2.0 * pi * (n - 1) * (j - 1) / N)
        end
    end
    Gm = S' * S
    Gdiag = [Gm[n, n] for n in 1:Nc]
    Ginv = [1.0 / Gm[n, n] for n in 1:Nc]
    return new(L, N, Nc, h, x, kfull, mfull, kcos, mcos, E, Ein, Mmat, S, Gdiag, Ginv)
end

Discretization(L::Real, N::Int) = Discretization(Float64(L), N)

# Index of the reflected grid point x -> -x (mod L), 1-based.
refidx(j::Int, N::Int) = 1 + mod(N - (j - 1), N)

function reflect_values(disc::Discretization, u::Vector{Float64})
    return [u[refidx(j, disc.N)] for j in 1:disc.N]
end

function project_even(disc::Discretization, u::Vector{Float64})
    return 0.5 .* (u .+ reflect_values(disc, u))
end

to_values(disc::Discretization, c::Vector{Float64}) = disc.S * c

function to_coeffs(disc::Discretization, u::Vector{Float64})
    pu = project_even(disc, u)
    return disc.Ginv .* (disc.S' * pu)
end

rms_of_values(disc::Discretization, u::Vector{Float64}) = norm(u) / sqrt(disc.N)
rms_of_c(disc::Discretization, c::Vector{Float64}) = rms_of_values(disc, disc.S * c)

# Exact integer-grid translation (a permutation of the samples).
function shift_grid_values(u::Vector{Float64}, m::Int)
    N = length(u)
    return [u[1 + mod(j - 1 - m, N)] for j in 1:N]
end

# Fractional spectral shift u(x - delta). The Nyquist coefficient is zeroed so
# the result stays real; for band-limited inputs this is exact trigonometric
# interpolation.
function shift_values(disc::Discretization, u::Vector{Float64}, delta::Float64)
    N = disc.N
    uh = disc.E * u
    uh[N ÷ 2 + 1] = 0.0im
    for n in 1:N
        uh[n] = uh[n] * cis(-disc.kfull[n] * delta)
    end
    return real.(disc.Ein * uh)
end

# Spectral derivative (Nyquist coefficient zeroed; exact for band-limited u).
function derivative_values(disc::Discretization, u::Vector{Float64})
    N = disc.N
    uh = disc.E * u
    uh[N ÷ 2 + 1] = 0.0im
    for n in 1:N
        uh[n] = uh[n] * (im * disc.kfull[n])
    end
    return real.(disc.Ein * uh)
end

# Evaluate a cosine-coefficient vector (produced on any grid with the same L)
# on the grid of `disc`. Exact trigonometric evaluation; requires
# length(c) <= disc.Nc.
function evaluate_coeffs_on(disc::Discretization, c::Vector{Float64})
    if length(c) > disc.Nc
        error("coefficient vector of length $(length(c)) does not fit target grid (Nc = $(disc.Nc))")
    end
    u = zeros(disc.N)
    for n in 1:length(c)
        k = disc.kcos[n]
        for j in 1:disc.N
            u[j] += c[n] * cos(k * disc.x[j])
        end
    end
    return u
end
