# Symmetry-aware morphology descriptors.
#
# All components below are invariant under grid translations (exact for
# band-limited grid functions, via discrete Parseval) and under the reflection
# x -> -x (exact: reflection only conjugates Fourier coefficients). They are
# NOT complete invariants: two genuinely different sample vectors can share the
# same descriptor (tested explicitly). The maximum |u| is deliberately NOT part
# of the invariant descriptor set (sampled extrema move with the grid).
#
# Components:
#   rms   = sqrt(sum u^2 / N)
#   mean  = sum u / N                     (mode-0 power; sign flips under u -> -u)
#   kbar  = power-weighted mean wavenumber over cosine modes 0..N/2
#   pr    = participation ratio (sum u^2)^2 / (N sum u^4); ~2/3 for a single
#           cosine, small for localized states
#   f0..f4 = power fractions in wavenumber bands [0,0.5), [0.5,1.5), [1.5,2.5),
#            [2.5,4.5), [4.5, inf)  (physical k, independent of L >= 2*pi)
#   nz50  = number of Fourier modes with power > 1e-10 * max power

function fourier_powers(disc::Discretization, u::Vector{Float64})
    N = disc.N
    H = N ÷ 2
    uh = disc.E * u
    P = Vector{Float64}(undef, H + 1)
    P[1] = abs2(uh[1])
    for n in 1:(H - 1)
        P[n + 1] = abs2(uh[n + 1]) + abs2(uh[N - n + 1])
    end
    P[H + 1] = abs2(uh[H + 1])
    return P
end

function descriptors(disc::Discretization, u::Vector{Float64})
    N = disc.N
    H = N ÷ 2
    s2 = dot(u, u)
    s4 = sum(u .^ 4)
    rms = sqrt(s2 / N)
    mean_u = sum(u) / N
    P = fourier_powers(disc, u)
    Ptot = sum(P)
    kbar = 0.0
    bands = zeros(5)
    maxP = 0.0
    for n in 0:H
        p = P[n + 1]
        k = disc.kcos[n + 1]
        kbar += k * p
        if k < 0.5
            bands[1] += p
        elseif k < 1.5
            bands[2] += p
        elseif k < 2.5
            bands[3] += p
        elseif k < 4.5
            bands[4] += p
        else
            bands[5] += p
        end
        if p > maxP
            maxP = p
        end
    end
    if Ptot > 0.0
        kbar = kbar / Ptot
        bands ./= Ptot
    else
        kbar = 0.0
    end
    pr = s4 > 0.0 ? (s2 * s2) / (N * s4) : 0.0
    nz50 = maxP > 0.0 ? Float64(count(p -> p > 1.0e-10 * maxP, P)) : 0.0
    return (rms=rms, mean=mean_u, kbar=kbar, pr=pr,
            f0=bands[1], f1=bands[2], f2=bands[3], f3=bands[4], f4=bands[5], nz50=nz50)
end

function descriptor_vector(d)
    return Float64[d.rms, d.mean, d.kbar, d.pr, d.f0, d.f1, d.f2, d.f3, d.f4, d.nz50]
end

function empty_descriptors()
    nan = Float64(NaN)
    return (rms=nan, mean=nan, kbar=nan, pr=nan,
            f0=nan, f1=nan, f2=nan, f3=nan, f4=nan, nz50=nan)
end
