"""
    Epd(μ, σ, p)

The *Exponential power distribution* with location `μ`, scale parameter `σ` and shape
parameter `p` has probability density function

```math
f(x; \\mu, \\sigma, p) = \\frac{1}{\\sigma} \\frac{1}{2 p^{1/p}\\Gamma(1+1/p)} \\exp\\left\\{-\\frac{1}{p}\\Big| \\frac{x-\\mu}{\\sigma} \\Big| \\right\\}
```

for

```
-\\infty < x < \\infty
```

```julia
Epd(μ, σ, p)      # Epd with shape p, scale σ and location μ.
params(d)       # Get the parameters, i.e. (μ, σ, p)
location(d)     # Get the location parameter, i.e. μ
scale(d)        # Get the scale parameter, i.e. σ
shape(d)        # Get the shape parameter, i.e. p (sometimes called θ)
```

External links
* [Epd](https://www.sciencedirect.com/science/article/pii/S0304407608001668?casa_token=kEUNFlIXYBEAAAAA:aCZkSeMVO4y3H7J9cCyR18j5R6QifeKYa_PSkvgDaCBO-xYQKxbeB1YQLpJbTQowZR_4fIKYcOc)
"""
struct Epd{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    p::T
    Epd{T}(µ::T, σ::T, p::T) where {T} = new{T}(µ, σ, p)
end

function Epd(µ::T, σ::T, p::T; check_args=true) where {T <: Real}
    check_args && @check_args(Epd, σ > zero(σ))
    check_args && @check_args(Epd, p > zero(p))
    return Epd{T}(µ, σ, p)
end

Epd(μ::Real, σ::Real, p::Real) = Epd(promote(μ, σ, p)...)
Epd(μ::Integer, σ::Integer, p::Integer) = Laplace(float(μ), float(σ), float(p))
Epd(μ::T, p::T) where {T <: Real} = Epd(μ, one(T), p)
Epd(μ::T) where {T <: Real} = Epd(μ, one(T), 2.)
Epd() = Laplace(0.0, 1.0, 2., check_args=false)

@distr_support Epd -Inf Inf

location(d::Epd) = d.μ
scale(d::Epd) = d.σ
shape(d::Epd) = d.p
params(d::Epd) = (d.μ, d.σ, d.p)
@inline partype(d::Epd{T}) where {T<:Real} = T

#### Statistics

#### Evaluations
function pdf(d::Epd, x::Real)
    μ, σ, p = params(d)
    K = σ * 2 * p^(1/p) * gamma(1 + 1/p)
    exp(-1/p * abs((x - μ)/σ)^p)/K
end

logpdf(d::Epd, x::Real) = log(pdf(d, x))

quantile(d::Epd, x::Real) = sign(2*x-1) * (d.p * quantile(Gamma(1/d.p, d.σ^d.p), abs(2*x-1)) )^(1/d.p)

#### Sampling
rand(rng::AbstractRNG, d::Epd) = quantile(d, rand(rng)) + d.μ
