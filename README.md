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

N, nsim = ([20, 50, 100, 200], 1000);
p  = range(1., 4, length = 20);

simDat = DataFrame(n = repeat(N, inner = length(p)),
                   p = repeat(p, length(N)), value = 0.0)

# row 1
for i in 1:length(N)
    β = pmap(kurt -> simSize(Epd(0.0, 1.0, kurt), N[i], nsim, 1), p)
    simDat[simDat.n .== N[i], :value] = β
end

# row 2
for i in 1:length(N)
    β = pmap(kurt -> simSize(Epd(0.0, 1.0, kurt), N[i], nsim, "Normal"), p)
    simDat[simDat.n .== N[i], :value] = β
end

# row 3
for i in 1:length(N)
    β = pmap(kurt -> simSize(Epd(0.0, 1.0, kurt), N[i], nsim, 3.), p)
    simDat[simDat.n .== N[i], :value] = β
end
```
