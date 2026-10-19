# Residuals and Jacobians of the stationary problem, plus an independent
# second-order periodic finite-difference operator used for cross-checks.
#
# Stationary problem:   F(u; r) = r*u - (1 + d_xx)^2 u + b*u^3 - u^5 = 0.
# Full space: value vector u in R^N, (1 + d_xx)^2 represented by the dense
# spectral matrix Mmat. Even subspace: u = S c, residual R(c; r) = S' F(S c).
# The even restriction is exactly the collocation system restricted to
# reflection-symmetric u, because F maps even vectors to even vectors.

function residual_full(disc::Discretization, cfg::SHConfig, u::Vector{Float64}, r::Float64)
    return (r .* u) .- (disc.Mmat * u) .+ (cfg.b .* (u .^ 3)) .- (u .^ 5)
end

function residual_even(disc::Discretization, cfg::SHConfig, c::Vector{Float64}, r::Float64)
    return disc.S' * residual_full(disc, cfg, disc.S * c, r)
end

# dR/dr = S' (dF/dr) = S' (S c)  (dF/dr = u).
function residual_r_even(disc::Discretization, c::Vector{Float64})
    return disc.S' * (disc.S * c)
end

function jacobian_full(disc::Discretization, cfg::SHConfig, u::Vector{Float64}, r::Float64)
    N = disc.N
    J = copy(disc.Mmat)
    J .*= -1.0
    for j in 1:N
        J[j, j] += r + 3.0 * cfg.b * u[j]^2 - 5.0 * u[j]^4
    end
    return J
end

function jacobian_even(disc::Discretization, cfg::SHConfig, c::Vector{Float64}, r::Float64)
    return disc.S' * jacobian_full(disc, cfg, disc.S * c, r) * disc.S
end

# Independent FD operators (cross-checks only; the solver is spectral).
# Second-order periodic Laplacian; eigenvalues exactly -4 sin^2(pi m/N)/h^2.
function fd_laplacian(N::Int, h::Float64)
    if N < 8
        error("fd_laplacian expects N >= 8 (got N = $N)")
    end
    D = zeros(N, N)
    for j in 1:N
        D[j, j] = -2.0
        D[j, mod1(j + 1, N)] += 1.0
        D[j, mod1(j - 1, N)] += 1.0
    end
    return D ./ (h * h)
end

# (I + D2)^2 = I + 2 D2 + D2^2 as a matrix.
function fd_quadratic_op(N::Int, h::Float64)
    D2 = fd_laplacian(N, h)
    A = 2.0 .* D2 + (D2 * D2)
    for j in 1:N
        A[j, j] += 1.0
    end
    return A
end

# Exact circulant eigenvalues of the periodic FD Laplacian, m = 0..N-1.
function fd_laplacian_eigvals(N::Int, h::Float64)
    return [-4.0 * sin(pi * m / N)^2 / (h * h) for m in 0:(N - 1)]
end

# FD collocation residual (for cross-checking spectral solutions).
function residual_full_fd(Afd::Matrix{Float64}, cfg::SHConfig, u::Vector{Float64}, r::Float64)
    return (r .* u) .- (Afd * u) .+ (cfg.b .* (u .^ 3)) .- (u .^ 5)
end
