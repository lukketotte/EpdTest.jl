"""
TODO: document
"""
struct MvEpd{T <: Real, Cov<:AbstractPDMat, Mean<:AbstractVector, shape<:Real} <: ContinuousMultivariateDistribution
    μ::Mean
    Σ::Cov
    p::shape
end

### Construction
function MvEpd(μ::AbstractVector{T}, Σ::AbstractPDMat{T}, p::T) where {T <: Real}
    dim(Σ) == length(μ) || throw(DimensionMismatch("The dimensions of mu and Sigma are inconsistent."))
    p > 0 || throw(DomainError("p must be positive"))
    MvEpd{T, typeof(Σ), typeof(μ), typeof(p)}(μ, Σ, p)
end

function MvEpd(μ::AbstractVector{<:Real}, Σ::AbstractPDMat, p::Real)
    R = Base.promote_eltype(μ, Σ)
    MvEpd(convert(AbstractArray{R}, μ), convert(AbstractArray{R}, Σ), p)
end

# constructor with general covariance matrix
MvEpd(μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}, p::Real) = MvEpd(μ, PDMat(Σ), p)
MvEpd(μ::AbstractVector{<:Real}, Σ::Diagonal{<:Real}, p::Real) = MvEpd(μ, PDiagMat(diag(Σ)), p)
MvEpd(μ::AbstractVector{<:Real}, Σ::UniformScaling{<:Real}, p::Real) =
    MvEpd(μ, ScalMat(length(μ), Σ.λ), p)

# constructor with vector of standard deviations
MvEpd(μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}, p::Real) = MvEpd(μ, PDiagMat(abs2.(σ)), p)

# constructor with scalar standard deviation
MvEpd(μ::AbstractVector{<:Real}, σ::Real, p::Real) = MvEpd(μ, ScalMat(length(μ), abs2(σ)), p)

# constructor without mean vector
MvEpd(Σ::AbstractVecOrMat{<:Real}, p::Real) = MvEpd(Zeros{eltype(Σ)}(size(Σ, 1)), Σ, p)

# special constructor
MvEpd(d::Int, σ::Real, p::Real) = MvEpd(Zeros{typeof(σ)}(d), σ, p)

### Basic statistics
length(d::MvEpd) = length(d.μ)
location(d::MvEpd) = d.μ
params(d::MvEpd) = (d.μ, d.Σ, d.p)
@inline partype(d::MvEpd{T}) where {T<:Real} = T
Base.eltype(::Type{<:MvEpd{T}}) where {T} = T

cov(d::MvEpd) = d.Σ * 2^(1/d.p) *
    gamma((length(d.μ)+2)/(2*d.p)) / (length(d.μ) * gamma(length(d.μ)/(2*d.p))) |> Matrix

invcov(d::MvEpd) = Matrix(inv(cov(d)))

### Evaluation
insupport(d::MvEpd, x::AbstractVector{T}) where {T<:Real} =
    length(d) == length(x) && all(isfinite, x)

sqmahal(d::MvEpd, x::AbstractVector{<:Real}) = invquad(d.Σ, x - d.μ)

function sqmahal!(r::AbstractArray, d::MvEpd, x::AbstractMatrix{<:Real})
    invquad!(r, d.Σ, x .- d.μ)
end

sqmahal(d::MvEpd, x::AbstractMatrix{T}) where {T<:Real} = sqmahal!(Vector{T}(undef, size(x, 2)), d, x)

function mvepd_consts(d::MvEpd)
    n = length(d.μ)
    log(n) + loggamma(n/2) - n/2 * log(π) - loggamma(1+n/(2*d.p)) - (1+n/(2*d.p))*log(2) - 0.5*logdet(d.Σ)
end

_logpdf(d::MvEpd, x::AbstractVector{T}) where {T<:Real} = mvepd_consts(d) - 0.5 * sqmahal(d, x)^d.p

function _logpdf!(r::AbstractArray, d::MvEpd, x::AbstractMatrix)
    sqmahal!(r, d, x)
    println(r)
    k = mvepd_consts(d)
    for i = 1:size(x, 2)
        @inbounds r[i] = k - 0.5 * r[i]^d.p
    end
    r
end

_pdf!(r::AbstractArray, d::MvEpd, x::AbstractVector{T}) where {T<:Real} = exp.(r, _logpdf!(r, d, x))

function runifsphere(rng::AbstractRNG, n::Integer, k::Integer)
    k >= 2 || throw(DomainError(k, "k must be [2, ∞)"))
    X = reshape(rand(rng, Normal(), n*k), (n,k))
    rownorms = .√sum(X.^2, dims=2)
    for i in 1:size(X, 2)
        X[:,i] = X[:,i] ./ rownorms
    end
    X
end

function _rand!(rng::AbstractRNG, d::MvEpd, x::AbstractMatrix{T}) where T<:Real
    μ, Σ, p = params(d)
    k = size(Σ, 1)
    n = size(x, 2)
    val, ev = eigen(Σ)
    Σsqrt = ev *  diagm(.√val) * ev'
    radius = (rand(rng,Gamma(k/(2*p), 1/2), n)).^(1/(2*p))
    un = reshape(rand(rng, Normal(), n*k), (n,k))
    rownorms = .√sum(un.^2, dims=2)
    for i in 1:size(un, 2)
        un[:,i] = un[:,i] ./ rownorms
    end
    x = μ .+ reshape(radius .* un * Σsqrt, (k,n))
    x
end
