###
### Functions for Epd based tests
###
function _L!(r::AbstractArray{T}, y::AbstractArray{T}, μ::T, σ::T, p::T) where {T <: Real}
    for i in 1:length(y)
        @inbounds r[i] = gamma(1 + 1/p) * abs(y[i] - μ) / σ
    end
    r
end

function logOrZero(x::Real)
    x == 0. ? 0 : log(x)
end

function effScore(y::AbstractArray{<:T}, μ::T, σ::T, p::T) where {T <: Real}
    r = Vector{T}(undef, length(y))
    _L!(r, y, μ, σ, p)
    Sₚ = r.^p .* (1/p * digamma(1 + 1/p) .- logOrZero.(r))
    Sₛ = p/σ .* r.^p .- 1/σ
    (Sₚ, Sₛ)
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
function epdTest(y::AbstractArray{<:Real}, μ::Real, σ::Real, p::Real)
    n = length(y)
    S_p, S_σ = effScore(y, μ, σ, p)
    return sum(S_p .+ σ / 4 .* S_σ) / √(n*((p+1)/p^4 * trigamma((p+1)/p) - 1/p^3))
end

function loglikEPD(θ, p, x) where {T <: Real}
    μ, σ = θ
    -log.(pdf.(Epd(μ, exp(σ), p), x)) |> sum
end

function MLE(θ::AbstractVector{T}, p::T, x::Array{T, 1}) where {T <: Real}
    length(θ) === 2 || throw(ArgumentError("θ not of length 2"))
    func = TwiceDifferentiable(vars -> loglikEPD(vars, p, x), ones(2), autodiff =:forward)
    optimum = optimize(func, θ)
    Optim.converged(optimum) || throw(ConvergenceError("Optimizer did not converge"))
    mle = Optim.minimizer(optimum)
    mle[2] = exp(mle[2])
    mle
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

function standard(X::AbstractMatrix{<:Real})
    n,d = size(X)
    S = inv(sqrt((n-1)/n * cov(X)))
    (S*(X .- mean(X, dims = 1))')'
end

"""
    DEHU(X)

Computes the test of multivariate normality based on the characteristic function

# Arguments
- `X::AbstractMatrix{<:Real}`: data, n × p
- `a::Real`: parameter > 0
"""
function DEHU(X::AbstractMatrix{<:Real}, a::Real=1)
    n,d = size(X)
    n > d || throw(ArgumentError("Rows of X must be larger than columns"))
    X = standard(X)
    Djk = X * X'
    Rquad = diag(Djk)
    # not sure how to do this the best way
    Dj = zeros(n,n)
    for i in 1:n
        Dj[:, i] = Rquad
    end
    Hilf = Dj - 2 .*Djk+Dj'

    Y=(Rquad*Rquad' - (Dj+Dj').*(Hilf.+2*d*a*(2*a-1))/(4*a^2)+
        (16*d^2*a^3*(a-1)+4*d*(d+2)*a^2 .+(8*d*a^2-(8+4*d)*a).*Hilf+Hilf.^2)/(16*a^4)).*exp.(-Hilf/(4*a))
    ret = (π/a)^(d/2) * sum(Y) / n
    ret*d^(-2) * (a/π)^(d/2)
end

"""
    JB(X)

Computes the Jarque-Bera test of Mardia
# Arguments
- `X::AbstractMatrix{<:Real}`: data, n × p
"""
function JB(X::AbstractMatrix{<:Real})
    n,p = size(X)
    μ = reshape(sum(X, dims = 1)./n, 2)
    S = cov(X)^(-1)
    u = zeros(n)
    for i in 1:n
        u[i] = dot((X[i,:] - μ)' * S, (X[i,:] - μ))[1,1]
    end
    √(n/64) * (mean(u.^2) - 8)
end
