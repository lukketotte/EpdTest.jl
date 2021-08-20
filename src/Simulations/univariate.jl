"""
    simSize(d, n, nsim, type; twoSided, α)

Simulates size for the EPD test for special cases p = 1 or 2

# Arguments
- `d<:ContinuousUnivariateDistribution: generates data
- `n::Integer`: sample size
- `nsim::Integer`: number of monte carlo replications
- `type::AbstractString`: specifies whether the null distribution is normal or laplace
- `twoSided::Bool`: true for twosided test, false otherwise. Always twosided for laplace.
- `α::Real`: size of the test.
"""
function simSize(d::D, n::N, nsim::N, type::AbstractString; twoSided::Bool = true, α::Real = 0.05) where
    {D<:ContinuousUnivariateDistribution, N <: Integer}
    sims = zeros(nsim)
    if lowercase(type) === "normal"
        z = twoSided ? quantile(Normal(), 1-α/2) : quantile(Normal(), 1-α)
    elseif lowercase(type) === "laplace"
        z = quantile(Chisq(1), 1-α)
    else
        throw(DomainError(type, "type must be normal or laplace"))
    end
    for i in 1:nsim
        simSizeInner!(sims, rand(d, n), z, i, n, twoSided, type)
    end
    return mean(sims)
end

"""
    simSize(n, nsim, p, type; twoSided, α)

Simulates size for the EPD test for general p.

# Arguments
- `d<:ContinuousUnivariateDistribution: generates data
- `n::Integer`: sample size
- `nsim::Integer`: number of monte carlo replications
- `p::Real`: shape under the null
- `twoSided::Bool`: true for twosided test, false otherwise. Always twosided for laplace.
- `α::Real`: size of the test.
"""
function simSize(d::D, n::N, nsim::N, p::Real; twoSided::Bool = true, α::Real = 0.05, χ::Bool = false) where
    {D<:ContinuousUnivariateDistribution, N <: Integer}
    p > 0 || throw(DomainError(p, "p must be positive"))
    ((α > 0) && (α < 1)) || throw(DomainError(α, "α must be on (0,1)"))
    sims = zeros(nsim)
    if χ
        z = quantile(Chisq(1), 1-α)
    else
        z = twoSided ? quantile(Normal(), 1-α/2) : quantile(Normal(), 1-α)
    end
    for i in 1:nsim
        simSizeInner!(sims, rand(d, n), z, i, n, twoSided, p, χ)
    end
    sims[sims .!== NaN] |> mean
end


function simSizeInner!(sims::Array{<:Real, 1}, y::Array{<:Real, 1}, z::Real, i::Integer, n::Integer,
    twoSided::Bool, type::AbstractString)
    if lowercase(type) === "normal"
        y = (y .- mean(y)) ./ √(var(y))
        K₂ = mean(y.^2 .* log.(abs.(y)))
        t = √n * (K₂ - (2 - log(2) + digamma(1))/2) / √((3*π^2 - 28)/8)
        if (twoSided ? abs(t) : t) > z
            sims[i] = 1
        end
    else
        μ = median(y)
        t = epdTest(y, μ, mean(abs.(y .- μ)), 1.)
        sims[i] = t^2 >= z ? 1 : 0
    end
    nothing
end

function simSizeInner!(sims::Array{<:Real, 1}, y::Array{<:Real, 1}, z::Real, i::Integer, n::Integer,
    twoSided::Bool, p::Real, χ::Bool)
    μ, σ = try
            MLE([0, log(2.)], p, y)
        catch err
            NaN, NaN
        end
    if μ == NaN
        sims[i] = NaN
    else
        t = epdTest(y, μ, p^(1/p) * gamma(1 + 1/p) * σ, p)
        if χ
            sims[i] = t^2 > z ? 1 : 0
        else
            if (twoSided ? abs(t) : t) > z
                sims[i] = 1
            end
        end
    end
    nothing
end
