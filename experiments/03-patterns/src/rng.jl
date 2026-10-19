# Deterministic 64-bit linear congruential generator (Knuth MMIX constants).
#
# Stdlib Random is deliberately not used: the generator is part of the
# reproducibility contract and every seed id is recorded in the outputs.
# Output: uniform in [0, 1) with 53 bits of resolution.

struct TinyRNG
    state::Base.RefValue{UInt64}
end

TinyRNG(seed::UInt64) = TinyRNG(Ref(UInt64(seed)))
TinyRNG(seed::Integer) = TinyRNG(UInt64(seed))

function Base.rand(rng::TinyRNG)
    s = rng.state[]
    s = s * 0x5851f42d4c957f2d + 0x14057b7ef767814f   # wraps modulo 2^64
    rng.state[] = s
    return Float64(s >> 11) * (1.0 / 9007199254740992.0)   # divide by 2^53
end
