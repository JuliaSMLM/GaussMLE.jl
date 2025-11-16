"""
Local Performance Benchmark for GaussMLE.jl

This benchmark runs comprehensive testing of all 16 combinations:
- 4 PSF models (GaussianXYNB, GaussianXYNBS, GaussianXYNBSXSY, AstigmaticXYZNB)
- 2 camera models (IdealCamera, SCMOSCamera)
- 2 devices (CPU, GPU)

The benchmark validates MLE optimality by comparing empirical std to theoretical CRLB.
A std/CRLB ratio of 1.0 indicates the estimator is statistically optimal.

Only runs in local testing environments (not on GitHub Actions).
"""

using GaussMLE
using SMLMData
using Random
using Statistics
using Printf
using CUDA

# Configuration
const WARMUP_ITERATIONS = 100
const BENCHMARK_ITERATIONS = 10_000
const ROI_SIZE = 11
const SEED = 42

# Benchmark result structure
struct BenchmarkConfig
    psf_model::PSFModel
    camera_symbol::Symbol
    device_symbol::Symbol
    model_name::String
end

struct ParameterStats
    std_crlb_ratio::Float32
    bias::Float32
    empirical_std::Float32
    mean_crlb::Float32
end

struct BenchmarkResult
    config::BenchmarkConfig
    fits_per_second::Float64
    convergence_rate::Float64
    param_stats::Dict{Symbol, ParameterStats}
end

"""
    detect_environment() -> Bool

Returns true if running in local environment, false if on CI (GitHub Actions).
"""
function detect_environment()
    return get(ENV, "CI", "false") == "false"
end

"""
    create_psf_models() -> Vector{PSFModel}

Create all 4 PSF models for testing.
"""
function create_psf_models()
    return [
        GaussMLE.GaussianXYNB(1.3f0),
        GaussMLE.GaussianXYNBS(),
        GaussMLE.GaussianXYNBSXSY(),
        GaussMLE.AstigmaticXYZNB{Float32}(
            1.3f0, 1.3f0,    # σx₀, σy₀
            0.0f0, 0.0f0,    # Ax, Ay
            0.0f0, 0.0f0,    # Bx, By
            250.0f0,         # γ (nm)
            400.0f0          # d (nm)
        )
    ]
end

"""
    get_model_name(psf::PSFModel) -> String

Get short name for PSF model.
"""
function get_model_name(psf::PSFModel)
    type_str = string(typeof(psf))
    return split(type_str, ".")[end] |> x -> replace(x, r"\{.*\}" => "")
end

"""
    get_param_names(psf::PSFModel) -> Vector{Symbol}

Get parameter names for a PSF model.
"""
function get_param_names(psf::PSFModel)
    if psf isa GaussMLE.GaussianXYNB
        return [:x, :y, :N, :bg]
    elseif psf isa GaussMLE.GaussianXYNBS
        return [:x, :y, :N, :bg, :σ]
    elseif psf isa GaussMLE.GaussianXYNBSXSY
        return [:x, :y, :N, :bg, :σx, :σy]
    elseif psf isa GaussMLE.AstigmaticXYZNB
        return [:x, :y, :z, :N, :bg]
    else
        error("Unknown PSF model type: $(typeof(psf))")
    end
end

"""
    create_camera(camera_symbol::Symbol, roi_size::Int)

Create camera model from symbol.
"""
function create_camera(camera_symbol::Symbol, roi_size::Int)
    if camera_symbol == :ideal
        return SMLMData.IdealCamera(512, 512, 0.1f0)
    elseif camera_symbol == :scmos
        # Realistic sCMOS parameters: 5.0 e⁻ rms readnoise
        return SMLMData.SCMOSCamera(
            512, 512,
            0.1f0,      # pixel size (μm)
            5.0f0,      # readnoise (e⁻ rms)
            offset = 100.0f0,
            gain = 0.5f0,
            qe = 0.82f0
        )
    else
        error("Unknown camera type: $camera_symbol")
    end
end

"""
    generate_test_data(psf::PSFModel, camera, n_rois::Int, roi_size::Int; seed=42)

Generate synthetic test data with known ground truth.
"""
function generate_test_data(psf::PSFModel, camera, n_rois::Int, roi_size::Int; seed=42)
    Random.seed!(seed)

    # Base parameters (x, y, N, bg)
    center = (roi_size + 1) / 2
    true_params = Float32[
        center .+ 0.3f0 * randn(Float32, n_rois)';  # x
        center .+ 0.3f0 * randn(Float32, n_rois)';  # y
        1000.0f0 .+ 100.0f0 * randn(Float32, n_rois)';  # photons
        10.0f0 .+ 1.0f0 * randn(Float32, n_rois)'   # background
    ]

    # Add model-specific parameters
    if psf isa GaussMLE.GaussianXYNBS
        σ_vals = 1.3f0 .+ 0.1f0 * randn(Float32, n_rois)'
        true_params = vcat(true_params, σ_vals)
    elseif psf isa GaussMLE.GaussianXYNBSXSY
        σx_vals = 1.3f0 .+ 0.1f0 * randn(Float32, n_rois)'
        σy_vals = 1.3f0 .+ 0.1f0 * randn(Float32, n_rois)'
        true_params = vcat(true_params, σx_vals, σy_vals)
    elseif psf isa GaussMLE.AstigmaticXYZNB
        z_vals = 200.0f0 * randn(Float32, n_rois)'
        # Insert z after y (x, y, z, N, bg)
        true_params = vcat(true_params[1:2, :], z_vals, true_params[3:4, :])
    end

    # Generate ROI batch
    batch = GaussMLE.generate_roi_batch(
        camera, psf;
        n_rois = n_rois,
        roi_size = roi_size,
        true_params = true_params,
        seed = seed
    )

    return batch, true_params
end

"""
    run_single_benchmark(config::BenchmarkConfig, warmup::Int, benchmark::Int)

Run benchmark for a single configuration.
"""
function run_single_benchmark(config::BenchmarkConfig, warmup::Int, benchmark::Int)
    try
        # Create camera
        camera = create_camera(config.camera_symbol, ROI_SIZE)

        # Generate warmup data
        warmup_batch, _ = generate_test_data(
            config.psf_model, camera, warmup, ROI_SIZE; seed=SEED
        )

        # Generate benchmark data
        benchmark_batch, true_params = generate_test_data(
            config.psf_model, camera, benchmark, ROI_SIZE; seed=SEED+1
        )

        # Create fitter
        device_obj = config.device_symbol == :cpu ? GaussMLE.CPU() : GaussMLE.GPU()
        fitter = GaussMLE.GaussMLEFitter(
            psf_model = config.psf_model,
            device = device_obj,
            iterations = 20
        )

        # Warmup run (compile kernels, cache data)
        GaussMLE.fit(fitter, warmup_batch)

        # Benchmark run with timing
        t_start = time()
        results = GaussMLE.fit(fitter, benchmark_batch)
        t_elapsed = time() - t_start

        fits_per_second = benchmark / t_elapsed

        # Extract results
        params = results.parameters
        uncertainties = results.uncertainties

        # Calculate statistics for each parameter
        param_names = get_param_names(config.psf_model)
        param_stats = Dict{Symbol, ParameterStats}()

        for (i, name) in enumerate(param_names)
            fitted = params[i, :]
            uncertainty = uncertainties[i, :]
            true_vals = true_params[i, :]

            # Calculate errors
            errors = fitted .- true_vals
            bias = mean(errors)
            empirical_std = std(errors)
            mean_crlb = mean(uncertainty[isfinite.(uncertainty)])

            # Calculate std/CRLB ratio (1.0 = optimal)
            ratio = empirical_std / mean_crlb

            param_stats[name] = ParameterStats(ratio, bias, empirical_std, mean_crlb)
        end

        # Calculate convergence rate
        n_converged = sum(isfinite.(uncertainties[1, :]))
        convergence_rate = n_converged / benchmark

        return BenchmarkResult(config, fits_per_second, convergence_rate, param_stats)

    catch e
        @warn "Benchmark failed for $(config.model_name)-$(config.camera_symbol)-$(config.device_symbol): $e"
        return nothing
    end
end

"""
    format_ratio(ratio::Float32) -> String

Format std/CRLB ratio with color coding for REPL.
Good ratios (0.95-1.05) are shown normally, others are flagged.
"""
function format_ratio(ratio::Float32)
    if !isfinite(ratio)
        return " --.-- "
    elseif 0.95 <= ratio <= 1.05
        return @sprintf(" %5.2f ", ratio)
    else
        return @sprintf(" %5.2f*", ratio)  # Flag sub-optimal ratios
    end
end

"""
    print_benchmark_table(results::Vector{BenchmarkResult})

Print nicely formatted benchmark results table to REPL.
"""
function print_benchmark_table(results::Vector{BenchmarkResult})
    println()
    println("="^100)
    println("PERFORMANCE BENCHMARK - Kernel Abstract Implementation")
    println("="^100)
    println("Configuration: Warmup=$WARMUP_ITERATIONS, Benchmark=$BENCHMARK_ITERATIONS iterations")
    println("ROI Size: $(ROI_SIZE)×$(ROI_SIZE) pixels")
    println()
    println("Std/CRLB Ratio: 1.00 = optimal (statistically efficient estimator)")
    println("                Values marked with * are outside optimal range [0.95, 1.05]")
    println("="^100)
    println()

    # Main table header
    println(@sprintf("%-18s %-7s %-6s │ %6s %6s %6s %6s %6s %6s │ %10s",
        "Model", "Camera", "Device", "x", "y", "N", "bg", "σ/z", "extra", "fits/s"))
    println(@sprintf("%-18s %-7s %-6s │ %s │ %s",
        "", "", "", "std/CRLB ratios (1.0 = optimal)", ""))
    println("─"^100)

    # Group by model for better readability
    models = unique([r.config.model_name for r in results])

    for model in models
        model_results = filter(r -> r.config.model_name == model, results)

        for r in model_results
            # Get stats for each parameter
            x_ratio = format_ratio(get(r.param_stats, :x, ParameterStats(NaN32, NaN32, NaN32, NaN32)).std_crlb_ratio)
            y_ratio = format_ratio(get(r.param_stats, :y, ParameterStats(NaN32, NaN32, NaN32, NaN32)).std_crlb_ratio)
            n_ratio = format_ratio(get(r.param_stats, :N, ParameterStats(NaN32, NaN32, NaN32, NaN32)).std_crlb_ratio)
            bg_ratio = format_ratio(get(r.param_stats, :bg, ParameterStats(NaN32, NaN32, NaN32, NaN32)).std_crlb_ratio)

            # Handle model-specific parameters
            sigma_z_ratio = if haskey(r.param_stats, :σ)
                format_ratio(r.param_stats[:σ].std_crlb_ratio)
            elseif haskey(r.param_stats, :z)
                format_ratio(r.param_stats[:z].std_crlb_ratio)
            else
                "   -   "
            end

            extra_ratio = if haskey(r.param_stats, :σx)
                format_ratio(r.param_stats[:σx].std_crlb_ratio)
            elseif haskey(r.param_stats, :σy)
                format_ratio(r.param_stats[:σy].std_crlb_ratio)
            else
                "   -   "
            end

            fits_str = @sprintf("%10s", r.fits_per_second >= 1000 ?
                string(round(Int, r.fits_per_second ÷ 1000), "k") :
                string(round(Int, r.fits_per_second)))

            println(@sprintf("%-18s %-7s %-6s │%s%s%s%s%s%s │ %10s",
                r.config.model_name,
                r.config.camera_symbol,
                r.config.device_symbol,
                x_ratio, y_ratio, n_ratio, bg_ratio, sigma_z_ratio, extra_ratio,
                fits_str))
        end

        # Add spacing between models
        if model != models[end]
            println("─"^100)
        end
    end

    println("="^100)
    println()

    # Summary statistics
    println("SUMMARY STATISTICS:")
    println("─"^100)

    # Performance analysis
    if !isempty(results)
        speed_sorted = sort(results, by=r->r.fits_per_second, rev=true)
        println(@sprintf("  Fastest: %s-%s-%s (%.0f fits/s)",
            speed_sorted[1].config.model_name,
            speed_sorted[1].config.camera_symbol,
            speed_sorted[1].config.device_symbol,
            speed_sorted[1].fits_per_second))
        println(@sprintf("  Slowest: %s-%s-%s (%.0f fits/s)",
            speed_sorted[end].config.model_name,
            speed_sorted[end].config.camera_symbol,
            speed_sorted[end].config.device_symbol,
            speed_sorted[end].fits_per_second))

        # Find best CRLB match
        best_crlb_matches = []
        for r in results
            x_stats = get(r.param_stats, :x, nothing)
            if x_stats !== nothing && isfinite(x_stats.std_crlb_ratio)
                push!(best_crlb_matches, (abs(1.0 - x_stats.std_crlb_ratio), r))
            end
        end

        if !isempty(best_crlb_matches)
            sort!(best_crlb_matches)
            best = best_crlb_matches[1][2]
            println(@sprintf("  Best CRLB match (x): %s-%s-%s (ratio=%.3f)",
                best.config.model_name,
                best.config.camera_symbol,
                best.config.device_symbol,
                best.param_stats[:x].std_crlb_ratio))
        end

        # Convergence analysis
        avg_convergence = mean([r.convergence_rate for r in results])
        println(@sprintf("  Average convergence rate: %.1f%%", avg_convergence * 100))
    end

    println("="^100)
    println()
end

"""
    run_comprehensive_benchmark()

Run comprehensive benchmark of all 16 configurations.
"""
function run_comprehensive_benchmark()
    # Check if we're in a local environment
    if !detect_environment()
        @info "Skipping local performance benchmark (running on CI)"
        return nothing
    end

    println()
    println("="^100)
    println("Starting Comprehensive Performance Benchmark")
    println("="^100)
    println("This will test all 16 combinations:")
    println("  - 4 PSF models")
    println("  - 2 camera models (Ideal, sCMOS)")
    println("  - 2 devices (CPU, GPU)")
    println()
    println("Each configuration runs:")
    println("  - Warmup: $WARMUP_ITERATIONS iterations")
    println("  - Benchmark: $BENCHMARK_ITERATIONS iterations")
    println()

    # Check GPU availability
    gpu_available = CUDA.functional()
    if !gpu_available
        @warn "GPU not available - will only test CPU configurations"
    else
        println("GPU detected: ", CUDA.name(CUDA.device()))
    end
    println()

    # Create all configurations
    psf_models = create_psf_models()
    camera_symbols = [:ideal, :scmos]
    device_symbols = gpu_available ? [:cpu, :gpu] : [:cpu]

    configs = BenchmarkConfig[]
    for psf in psf_models
        for camera in camera_symbols
            for device in device_symbols
                model_name = get_model_name(psf)
                push!(configs, BenchmarkConfig(psf, camera, device, model_name))
            end
        end
    end

    total = length(configs)
    println("Total configurations to test: $total")
    println("="^100)
    println()

    # Run benchmarks
    results = BenchmarkResult[]
    for (i, config) in enumerate(configs)
        print("[$i/$total] Testing $(config.model_name)-$(config.camera_symbol)-$(config.device_symbol)... ")
        flush(stdout)

        result = run_single_benchmark(config, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS)

        if result !== nothing
            push!(results, result)
            println("✓ ($(round(Int, result.fits_per_second)) fits/s)")
        else
            println("✗ FAILED")
        end
    end

    # Print results table
    if !isempty(results)
        print_benchmark_table(results)
        println("✓ Comprehensive benchmark complete!")
    else
        @warn "No successful benchmarks"
    end

    return results
end

# Export the main function
export run_comprehensive_benchmark, detect_environment
