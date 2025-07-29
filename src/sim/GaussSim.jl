"""
    GaussSim

Methods for creating simulated data for testing and benchmarking.

"""
module GaussSim

using Distributions
using GaussMLE.GaussModel

include("generate.jl")
include("gpu_generate.jl")

# Export GPU testing utilities
export GPUTestScenario, generate_gpu_test_scenarios, generate_batch_params
export StreamingBatchGenerator, next_batch!
export generate_scmos_variance_map, validate_gpu_results, benchmark_scenario

end