module EpdTest
# https://medium.com/coffee-in-a-klein-bottle/developing-your-julia-package-682c1d309507
include("Test.jl")
include("Simulations.jl")
include("Distributions.jl")
using .EPD

using SpecialFunctions, LinearAlgebra, Distributions

###########
# EXPORTS #
###########
export epdTest, Epd, simSize
end
