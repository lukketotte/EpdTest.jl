using EpdTest
using Test

@testset "EpdTest.jl" begin
    μ,σ,p = 0., 1., 1.
    y = [0.2, -1., 2.2, 3.1]
    @test EpdTest.epdTest(y, μ, σ, p) === -1.4366494480275978
end
