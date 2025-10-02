using GaussMLE
using Test
using Random
using Statistics
using Distributions
using LinearAlgebra
using SMLMData
using CUDA

# Include validation utilities
include("validation_utils.jl")

# Test configuration
const GPU_AVAILABLE = CUDA.functional()

# Print test configuration
println("="^70)
println("GaussMLE.jl Test Suite")
println("="^70)
if GPU_AVAILABLE
    println("  GPU detected: ", CUDA.name(CUDA.device()))
else
    println("  No GPU detected - CPU tests only")
end
println("="^70)
println()

# Main test suite
@testset "GaussMLE.jl" begin
    # Core CPU tests
    @testset "CPU Tests" begin
        # Consolidated test of all new features
        include("test_all_new_features.jl")

        # Strict validation tests using new simulator
        include("test_strict_validation.jl")

        # Comprehensive validation tests
        include("model_validation_tests.jl")
        include("comprehensive_cpu_tests.jl")
    end

    # GPU tests (auto-detected)
    if GPU_AVAILABLE
        @testset "GPU Tests" begin
            include("gpu_tests.jl")
        end
    end
end