using GaussMLE
using Test
using Random
using Statistics
using Distributions
using LinearAlgebra
using SMLMData
using CUDA
using Printf

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

# Main test suite - wrapped in try/catch/finally to ensure benchmark runs
global test_exception = nothing
try
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
catch e
    global test_exception = e
finally
    # Local performance benchmark (only runs in local environment, not on CI)
    # Runs in finally block so it executes even if tests fail
    println()
    println("="^70)
    if get(ENV, "CI", "false") == "false"
    println("Local environment detected - running comprehensive performance benchmark")
    println("="^70)
    include("local_performance_benchmark.jl")

    # Run the benchmark
    @testset "Local Performance Benchmark" begin
        results = run_comprehensive_benchmark()
        @test results !== nothing

        # Validate that we got some results
        if results !== nothing && !isempty(results)
            @test all(r -> r.fits_per_second > 0, results)

            # Check that std/CRLB ratios are reasonable (within 20% of optimal)
            # This is a sanity check - allows for some statistical variation
            for r in results
                for (param, stats) in r.param_stats
                    if isfinite(stats.std_crlb_ratio)
                        ratio_ok = 0.8 <= stats.std_crlb_ratio <= 1.2
                        if !ratio_ok
                            @warn "$(r.config.model_name)-$(r.config.camera_symbol)-$(r.config.device_symbol): $param has std/CRLB=$(stats.std_crlb_ratio) (outside [0.8, 1.2])"
                        end
                        @test ratio_ok
                    end
                end
            end
        end
    end
    else
        println("CI environment detected - skipping local performance benchmark")
        println("To run comprehensive benchmarks, execute tests locally:")
        println("  julia> using Pkg; Pkg.test(\"GaussMLE\")")
        println("="^70)
    end
end  # end finally block

# Re-throw test exception if any occurred
if test_exception !== nothing
    rethrow(test_exception)
end