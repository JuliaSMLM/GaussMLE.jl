# All imports must be at the top of runtests.jl
using GaussMLE
using SpecialFunctions
using LinearAlgebra
using Statistics 
using Test
using Random
using Distributions

# Main test structure - no test logic here
@testset "GaussMLE.jl" begin
    # User-facing API tests
    include("test_api.jl")
    
    # Internal function tests
    include("test_gausslib.jl")
    include("test_sim.jl")
    
    # Model validation tests (CPU)
    include("model_validation/cpu_model_tests.jl")
end

# Include GPU tests if requested
if get(ENV, "GAUSSMLE_TEST_GPU", "false") == "true"
    include("gpu_tests.jl")
    include("model_validation/gpu_model_tests.jl")
else
    @info "GPU tests skipped. Set GAUSSMLE_TEST_GPU=true to run them."
end
