###
### Functions for Epd based tests
###

function L(y::Real, μ::Real, σ::Real, p::Real)
    gamma(1 + 1/p) * abs(y - μ) / σ
end

# score functions wrt shape
function S(y::Real, μ::Real, σ::Real, p::Real)
    ℓ = L(y, μ, σ, p)
    S_p = ℓ^p * (1/p * digamma(1 + 1/p) - log(ℓ))
    S_σ = p/σ * ℓ^p - 1/σ
    return S_p, S_σ
end

function S(y::Array{<:Real, 1}, μ::Real, σ::Real, p::Real)
    ℓ = L.(y, μ, σ, p)
    S_p = ℓ.^p .* (1/p * digamma(1 + 1/p) .- log.(ℓ))
    S_σ = p/σ .* ℓ.^p .- 1/σ
    return S_p, S_σ
end

"""
    epdTest(y, μ, σ, p)

Computes the efficient score function wrt to the shape parameter, Eq. 8 of PAPER

```math
S_{n,p} = \\frac{1}{\\sqrt{n}} \\sum_{i=1}^{n} \\bigg[ \\ldots \\bigg]
```

# Arguments
- `y::Array{<:Real, 1}`: data
- `μ::Real`: location parameter, μ ∈ ℝ
- `σ::Real`: scale parameter, σ > 0
- `p::Real`: shape parameter, p > 0
"""
function epdTest(y::Array{<:Real, 1}, μ::Real, σ::Real, p::Real)
    n = length(y)
    S_p, S_σ = S(y, μ, σ, p)
    return sum(S_p .+ σ / 4 .* S_σ) / √(n*((p+1)/p^4 * trigamma((p+1)/p) - 1/p^3))
end
