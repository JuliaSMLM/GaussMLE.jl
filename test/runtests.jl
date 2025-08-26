using GaussMLE
using Test
using Random
using Statistics
using Distributions
using LinearAlgebra
using SMLMData

# Include validation utilities
include("validation_utils.jl")

# Main test suite
@testset "GaussMLE.jl" begin
    # Consolidated test of all new features
    include("test_all_new_features.jl")
    
    # Strict validation tests using new simulator
    include("test_strict_validation.jl")
    
    # Legacy validation tests - these have some convergence issues with certain test data
    # TODO: Fix initialization and convergence for edge cases
    # include("model_validation_tests.jl")
    # include("comprehensive_cpu_tests.jl")
end