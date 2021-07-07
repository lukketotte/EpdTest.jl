module EPD

export Epd, MLE

using Distributions, Random, Optim, SpecialFunctions
import Distributions.pdf, Distributions.quantile, Base.rand

struct Epd <: ContinuousUnivariateDistribution
    μ::Real
    σ::Real
    p::Real

    Epd(μ, σ, p) = (σ > 0) ? ((p > 0) ?  new(μ, σ, p) :
        throw(DomainError(p, "p must be positive"))) : throw(DomainError(σ, "σ must be positive"))
end

function pdf(d::Epd, x::Real)
    K = d.σ * 2 * d.p^(1/d.p) * gamma(1 + 1/d.p)
    return exp(-1/d.p * abs((x - d.μ)/d.σ)^d.p)/K
end

function quantile(d::Epd, x::Real)
    G = quantile(Gamma(1/d.p, d.σ^d.p), abs(2*x-1))
    return sign(2*x-1) * (d.p*G)^(1/d.p)
end

function rand(rng::AbstractRNG, d::Epd)
    r = rand(rng)
    return quantile(d, r) + d.μ
end

function loglikEPD(θ, p, x) where {T <: Real}
    μ, σ = θ
    σ = exp(σ)
    -log.(pdf.(Epd(μ, σ, p), x)) |> sum
end

# computes the ML estimator using Optim
function MLE(θ::Array{T, 1}, p::T, x::Array{T, 1}) where {T <: Real}
    length(θ) === 2 || throw(ArgumentError("θ not of length 2"))
    func = TwiceDifferentiable(vars -> loglikEPD(vars, p, x), ones(2), autodiff =:forward)
    optimum = optimize(func, θ)
    Optim.converged(optimum) || throw(ConvergenceError("Optimizer did not converge"))
    mle = Optim.minimizer(optimum)
    mle[2] = exp(mle[2])
    mle
end

end
