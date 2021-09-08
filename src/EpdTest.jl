module EpdTest
# https://medium.com/coffee-in-a-klein-bottle/developing-your-julia-package-682c1d309507
using Distributions, SpecialFunctions, Random, PDMats, LinearAlgebra, StatsFuns, Statistics, FillArrays

import Distributions: pdf, _logpdf, insupport, invcov, sqmahal, sampler, _rand!,
    logpdf, @check_args, @distr_support, shape, location, scale
import Base: rand, length, eltype
import PDMats: dim, PDMat, invquad
import Statistics: cov, quantile
import StatsBase: params

include("Test.jl")
include("Distributions/multivariate.jl")
include("Distributions/univariate.jl")
include("Simulations/univariate.jl")

export epdTest, Epd, simSize, BivariateNormalTest, MvEpd, JB, DEHU

end
