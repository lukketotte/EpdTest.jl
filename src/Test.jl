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
S_{n,p}^*
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

"""
    BivariateNormalTest(X, μ, Σ)

Computes the bivariate normal test based on the multivariate EPD

# Arguments
- `X::Array{<:Real, 2}`: data, n × p
- `μ::Array{<:Real, 1}`: location parameter, μ ∈ ℝᴾ
- `Σ::Array{<:Real, 2}`: Inverse of covariance matrix
"""
function BivariateNormalTest(X::Array{<:Real, 2}, μ::Array{<:Real, 1}, Σ::Array{<:Real, 2})
    n,p = size(X)
    p == length(μ) || throw(DomainError(μ, "μ and X dimension mismatch"))
    u = zeros(n)
    for i in 1:n
        u[i] = dot((X[i,:] - μ)' * S, (X[i,:] - μ))[1,1]
    end
    return √n * (1 + digamma(1) + log(2) - 0.5 * mean(u.* log.(u))) / √((π^2 - 9)/3)
end

"""
    BivariateNormalTest(X)

Computes the bivariate normal test based on the multivariate EPD

# Arguments
- `X::Array{<:Real, 2}`: data, n × p
"""
function BivariateNormalTest(X::Array{<:Real, 2})
    n,p = size(X)
    u = zeros(n)
    S = cov(X)^(-1)
    μ = reshape(sum(X, dims = 1)./n, 2)
    for i in 1:n
        u[i] = dot((X[i,:] - μ)' * S, (X[i,:] - μ))[1,1]
    end
    return √n * (1 + digamma(1) + log(2) - 0.5 * mean(u.* log.(u))) / √((π^2 - 9)/3)
end
