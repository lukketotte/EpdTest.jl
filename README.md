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
Using parallell computation
```julia
using Optim
rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
result = optimize(rosenbrock, zeros(2), BFGS())
```
