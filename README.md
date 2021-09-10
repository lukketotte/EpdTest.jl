# EpdTest

[![Build Status](https://github.com/lukketotte/EpdTest.jl/workflows/CI/badge.svg)](https://github.com/lukketotte/EpdTest.jl/actions)
[![Coverage](https://codecov.io/gh/lukketotte/EpdTest.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/lukketotte/EpdTest.jl)

**EpdTest.jl** is a Julia library created for the paper *Neyman’s C(α) test for the shape parameter of the exponential power class*.

## Installation
Through the `pkg` REPL mode by typing
```
] add "https://github.com/lukketotte/EpdTest.jl"
```

## Recreating results
To recreate the second column of Figure 3
```julia
using Distributed
@everywhere using EpdTest, DataFrames

N, nsim = ([50, 100, 500], 10000);
p  = range(1., 4, length = 20);

simDat = DataFrame(n = repeat(N, inner = length(p)),
                   p = repeat(p, length(N)), value = 0.0)

# row 1
for i ∈ 1:length(N)
    β = pmap(kurt -> simSize(Epd(0.0, 1.0, kurt), N[i], nsim, 1), p)
    simDat[simDat.n .== N[i], :value] = β
end

# row 2
for i ∈ 1:length(N)
    β = pmap(kurt -> simSize(Epd(0.0, 1.0, kurt), N[i], nsim, "Normal"), p)
    simDat[simDat.n .== N[i], :value] = β
end

# row 3
for i ∈ 1:length(N)
    β = pmap(kurt -> simSize(Epd(0.0, 1.0, kurt), N[i], nsim, 3.), p)
    simDat[simDat.n .== N[i], :value] = β
end
```

To recreate Figure 5
```julia
using Distributed
@everywhere using EpdTest, DataFrames, Distributions

# MC adjusted sizes
αLapGel = [0.51, 0.235, 0.135, 0.083]
αLap = [0.0625, 0.057, 0.053, 0.051]

# Sample sizes and DF's of the t-distribution
N = [20, 50, 100, 200]
ν = [1, 2, 3, 4, 5, 6, 7]

# Based on the test outlined in Gel, 2010
simDat = DataFrame(n = repeat(N, inner = length(ν)), df = repeat(ν, length(N)), value = 0.0)
for i ∈ 1:length(N)
    β = pmap(df -> simSize(TDist(df), N[i], 50000, "Laplace",
        quantile(Chisq(1), 1-αLapGel[i])), ν)
    simDat[simDat.n .== N[i], :value] = β
end

# Based on the EPD test
simDat = DataFrame(n = repeat(N, inner = length(ν)), df = repeat(ν, length(N)), value = 0.0)
for i ∈ 1:length(N)
    β = pmap(df -> simSizeLaplace(TDist(df), N[i], 50000, 1.,
        quantile(Chisq(1), 1-αLap[i]), χ=true), ν)
    simDat[simDat.n .== N[i], :value] = β
end
```
The likelihood ratio test based on the empirical likelihood is not part of the package, but the test together with code for Figure 5 is
```julia
function empLik(x::AbstractVector{<:Real}, m::Integer)
    n = length(x)
    m <= √n || throw(DomainError(m, "m must be < √n"))
    sort!(x)
    p = 1
    for i ∈ 1:n
        p *= (2*m) / (n*(x[minimum([i+m, n])] - x[maximum([i-m, 1])]) *
            pdf(Laplace(median(x), mean(abs.(x .- median(x)))), x[i]))
    end
    p
end

function empLikTest(x::AbstractVector{<:Real})
    n = length(x)
    m = Integer(floor(√n))
    alt = zeros(m)
    for i ∈ 1:m
        alt[i] = empLik(x, i)
    end
    log(minimum(alt))
end

function simSizeEmpLap(d::D, n::N, nsim::N, critical::T) where
    {D <: ContinuousUnivariateDistribution, N <: Integer, T<: Real}
    sims = [0. for x in 1:nsim]
    for i in 1:nsim
        t = empLikTest(rand(d, n))
        sims[i] = abs(t) >= critical ? 1 : 0
    end
    mean(sims)
end

simDat = DataFrame(n = repeat(N, inner = length(ν)), df = repeat(ν, length(N)), value = 0.0)
crit = [7.662, 9.213, 10.478, 11.616]

for i in 1:length(N)
    β = pmap(df -> simSizeEmpLap(TDist(df), N[i], 50000, crit[i]), ν)
    simDat[simDat.n .== N[i], :value] = β
end
```

To recreate the applications for the bivariate normal case with 50 observations
subsetted from the weather data. To recreate parts of the results in Dörr et. al. (2021),
the subsample is selected through R using `RCall` in Julia.
```Julia
# From RandomFields package in R
X = load("weather.csv") |> DataFrame
X = Matrix(X)

# Requires the RCall package
# gives the same indeces as Ref
idx = RCall.rcopy(R"""
RNGkind(sample.kind = "Rounding")
set.seed(0721)
idx = sample(1:157, 50)
""")

N = 10000
sim(n) = reshape(rand(MvNormal([0,0], diagm([1., 1.])), n), n, 2)

sims = [BivariateNormalTest(sim(50))^2 for i in 1:N]
mean(sims .> BivariateNormalTest(X[idx,:])^2)

sims = [JB(sim(50))^2 for i in 1:N]
mean(sims .> JB(X[idx,:])^2)

sims = [DEHU(sim(50)) for i in 1:N]
mean(sims .> DEHU(X[idx,:]))
```

# References
* Dörr, P., Ebner, B. and Henze, N. A new test of multivariate normality by
a double estimation in a characterizing pde. *Metrika*, 84(3):401-427, 2021.
