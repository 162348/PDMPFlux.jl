using Test
using Aqua
using PDMPFlux

@testset "Aqua.jl" begin
    # Run the standard Aqua test suite.
    # We keep `ambiguities=true` to catch method ambiguities early.
    Aqua.test_all(PDMPFlux; ambiguities=true)
end


