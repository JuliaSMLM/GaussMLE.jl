"""
GPU-optimized data generation for testing and benchmarking
"""

using Random
using Statistics

# Structure to hold GPU test scenarios
struct GPUTestScenario{T<:Real}
    name::String
    boxsize::Int
    n_rois::Int
    n_photons_range::Tuple{T,T}
    background_range::Tuple{T,T}
    position_jitter::T
    σ_PSF::T
    variance_map::Union{Nothing, AbstractArray{T,3}}
end

"""
    generate_gpu_test_scenarios(T=Float32)

Generate a comprehensive set of test scenarios for GPU testing.
"""
function generate_gpu_test_scenarios(::Type{T}=Float32) where T
    scenarios = GPUTestScenario{T}[]
    
    # Small batch for unit testing
    push!(scenarios, GPUTestScenario{T}(
        "small_batch", 7, 1000,
        (T(300), T(700)), (T(1), T(3)), T(0.1), T(1.3), nothing
    ))
    
    # Medium batch - typical workload
    push!(scenarios, GPUTestScenario{T}(
        "medium_batch", 9, 100_000,
        (T(400), T(600)), (T(2), T(4)), T(0.2), T(1.3), nothing
    ))
    
    # Large batch - stress test
    push!(scenarios, GPUTestScenario{T}(
        "large_batch", 7, 1_000_000,
        (T(450), T(550)), (T(1.5), T(2.5)), T(0.15), T(1.3), nothing
    ))
    
    # Variable PSF width test
    push!(scenarios, GPUTestScenario{T}(
        "variable_psf", 11, 50_000,
        (T(300), T(800)), (T(1), T(5)), T(0.3), T(1.5), nothing
    ))
    
    # Low SNR test
    push!(scenarios, GPUTestScenario{T}(
        "low_snr", 7, 50_000,
        (T(100), T(200)), (T(5), T(10)), T(0.2), T(1.3), nothing
    ))
    
    # High density test (overlapping PSFs)
    push!(scenarios, GPUTestScenario{T}(
        "high_density", 15, 10_000,
        (T(500), T(1000)), (T(2), T(4)), T(0.5), T(1.2), nothing
    ))
    
    return scenarios
end

"""
    generate_batch_params(scenario::GPUTestScenario, batch_size::Int; seed=nothing)

Generate parameters for a batch of ROIs based on the test scenario.
"""
function generate_batch_params(scenario::GPUTestScenario{T}, batch_size::Int; 
                             seed::Union{Nothing,Int}=nothing) where T
    if seed !== nothing
        Random.seed!(seed)
    end
    
    # Generate parameters
    center = T(scenario.boxsize / 2)
    x_positions = center .+ scenario.position_jitter .* randn(T, batch_size)
    y_positions = center .+ scenario.position_jitter .* randn(T, batch_size)
    
    n_photons = rand(T, batch_size) .* (scenario.n_photons_range[2] - scenario.n_photons_range[1]) .+ scenario.n_photons_range[1]
    backgrounds = rand(T, batch_size) .* (scenario.background_range[2] - scenario.background_range[1]) .+ scenario.background_range[1]
    
    return (x=x_positions, y=y_positions, n=n_photons, bg=backgrounds)
end

"""
    generate_streaming_batches(scenario::GPUTestScenario, batch_size::Int)

Generator function for streaming batch generation - yields batches on demand.
"""
mutable struct StreamingBatchGenerator{T}
    scenario::GPUTestScenario{T}
    batch_size::Int
    total_generated::Int
    rng::Random.MersenneTwister
end

function StreamingBatchGenerator(scenario::GPUTestScenario{T}, batch_size::Int; seed=1234) where T
    StreamingBatchGenerator(scenario, batch_size, 0, Random.MersenneTwister(seed))
end

function next_batch!(gen::StreamingBatchGenerator{T}) where T
    remaining = gen.scenario.n_rois - gen.total_generated
    if remaining <= 0
        return nothing
    end
    
    actual_batch_size = min(gen.batch_size, remaining)
    
    # Generate batch data
    stack = zeros(T, gen.scenario.boxsize, gen.scenario.boxsize, actual_batch_size)
    params = generate_batch_params(gen.scenario, actual_batch_size, seed=nothing)
    
    # Generate ROIs
    if gen.scenario.name == "variable_psf"
        # For variable PSF test, each ROI has different PSF width
        σ_PSFs = gen.scenario.σ_PSF .+ T(0.3) .* randn(gen.rng, T, actual_batch_size)
        for i in 1:actual_batch_size
            θ = θ_xynb(params.x[i], params.y[i], params.n[i], params.bg[i])
            args = Args_xynb(σ_PSFs[i])
            genroi!(view(stack, :, :, i), θ, args)
        end
    else
        # Standard case - same PSF for all ROIs
        args = Args_xynb(gen.scenario.σ_PSF)
        for i in 1:actual_batch_size
            θ = θ_xynb(params.x[i], params.y[i], params.n[i], params.bg[i])
            genroi!(view(stack, :, :, i), θ, args)
        end
    end
    
    # Add Poisson noise
    poissonnoise!(stack)
    
    # Add variance map if specified
    variance = if gen.scenario.variance_map !== nothing
        gen.scenario.variance_map[:, :, 1:actual_batch_size]
    else
        nothing
    end
    
    gen.total_generated += actual_batch_size
    
    return (data=stack, params=params, variance=variance)
end

"""
    generate_scmos_variance_map(boxsize::Int, n_frames::Int; 
                               gain_std=0.1, offset_std=2.0, read_noise=1.5)

Generate a realistic sCMOS variance map for testing variance-weighted fitting.
"""
function generate_scmos_variance_map(boxsize::Int, n_frames::Int, ::Type{T}=Float32;
                                   gain_std=T(0.1), offset_std=T(2.0), read_noise=T(1.5)) where T
    # Generate per-pixel gain and offset
    gain = 1 .+ gain_std .* randn(T, boxsize, boxsize)
    offset = offset_std .* randn(T, boxsize, boxsize)
    read_noise_sq = read_noise^2
    
    # Replicate for all frames
    variance_map = zeros(T, boxsize, boxsize, n_frames)
    for i in 1:n_frames
        variance_map[:, :, i] = gain .+ read_noise_sq
    end
    
    return variance_map
end

"""
    validate_gpu_results(cpu_results, gpu_results, scenario::GPUTestScenario; 
                        rtol=1e-5, atol=1e-6)

Validate GPU results against CPU reference implementation.
"""
function validate_gpu_results(cpu_params, cpu_crlb, gpu_params, gpu_crlb, 
                            scenario::GPUTestScenario{T}; 
                            rtol=T(1e-5), atol=T(1e-6)) where T
    # Check parameter agreement
    param_fields = [:x, :y, :n, :bg]
    for field in param_fields
        cpu_vals = getproperty.(cpu_params, field)
        gpu_vals = getproperty.(gpu_params, field)
        
        max_diff = maximum(abs.(cpu_vals .- gpu_vals))
        rel_diff = maximum(abs.((cpu_vals .- gpu_vals) ./ (cpu_vals .+ eps(T))))
        
        if max_diff > atol || rel_diff > rtol
            return false, "Parameter $field mismatch: max_diff=$max_diff, rel_diff=$rel_diff"
        end
    end
    
    # Check CRLB agreement
    crlb_fields = [:σ_x, :σ_y, :σ_n, :σ_bg]
    for field in crlb_fields
        cpu_vals = getproperty.(cpu_crlb, field)
        gpu_vals = getproperty.(gpu_crlb, field)
        
        max_diff = maximum(abs.(cpu_vals .- gpu_vals))
        rel_diff = maximum(abs.((cpu_vals .- gpu_vals) ./ (cpu_vals .+ eps(T))))
        
        if max_diff > atol || rel_diff > rtol
            return false, "CRLB $field mismatch: max_diff=$max_diff, rel_diff=$rel_diff"
        end
    end
    
    return true, "Validation passed"
end

"""
    benchmark_scenario(scenario::GPUTestScenario, backend)

Run a benchmark for a specific scenario and backend, returning timing information.
"""
function benchmark_scenario(scenario::GPUTestScenario{T}, fitfunc, warmup_runs=3, 
                          benchmark_runs=10) where T
    # Generate test data
    gen = StreamingBatchGenerator(scenario, min(100_000, scenario.n_rois))
    batch = next_batch!(gen)
    
    if batch === nothing
        error("Failed to generate test batch")
    end
    
    # Warmup runs
    for _ in 1:warmup_runs
        fitfunc(batch.data)
    end
    
    # Benchmark runs
    times = Float64[]
    for _ in 1:benchmark_runs
        t_start = time()
        fitfunc(batch.data)
        push!(times, time() - t_start)
    end
    
    # Calculate statistics
    mean_time = mean(times)
    std_time = std(times)
    min_time = minimum(times)
    rois_per_second = size(batch.data, 3) / mean_time
    
    return (
        mean_time = mean_time,
        std_time = std_time,
        min_time = min_time,
        rois_per_second = rois_per_second,
        n_rois = size(batch.data, 3)
    )
end