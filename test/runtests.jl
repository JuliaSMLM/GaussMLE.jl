using GaussMLE
using Test
using Random
using Statistics
using Distributions
using LinearAlgebra

# Include validation utilities
include("validation_utils.jl")

# Main test suite
@testset "GaussMLE.jl" begin
    include("model_validation_tests.jl")
    include("comprehensive_cpu_tests.jl")
end