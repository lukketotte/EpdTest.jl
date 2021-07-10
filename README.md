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
