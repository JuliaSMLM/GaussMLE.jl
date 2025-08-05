using Pkg
Pkg.activate(".")

using GaussMLE
using CUDA
using Random
using Statistics
using Printf
using Distributions

println("Benchmark Report: 10^4 Samples per Model")
println("=" ^ 80)

# Test parameters
n_rois = 10_000
boxsize = 7
n_photons = 1000
bg_photons = 10
σ_psf = 1.5

# Set random seed for reproducibility
Random.seed!(42)

# Results storage
results = Dict{String, Dict{String, Any}}()

# Helper function to generate synthetic data with known parameters
function generate_test_data_fixed(model_type::Symbol, n_rois::Int, boxsize::Int; 
                                 n_photons=1000, bg_photons=10, σ_psf=1.5)
    data = zeros(Float32, boxsize, boxsize, n_rois)
    true_params = Dict{Symbol, Float32}()
    
    # Fixed true parameters for all ROIs (for better statistics)
    x_true = Float32(4.0)  # Center at 4.0
    y_true = Float32(4.0)  # Center at 4.0
    n_true = Float32(n_photons)
    bg_true = Float32(bg_photons)
    
    if model_type == :xynb
        true_params[:x] = x_true
        true_params[:y] = y_true
        true_params[:n] = n_true
        true_params[:bg] = bg_true
        
        θ = GaussMLE.θ_xynb(x_true, y_true, n_true, bg_true)
        args = GaussMLE.GaussModel.Args_xynb(Float32(σ_psf))
        
    elseif model_type == :xynbs
        σ_true = Float32(σ_psf)
        true_params[:x] = x_true
        true_params[:y] = y_true
        true_params[:n] = n_true
        true_params[:bg] = bg_true
        true_params[:σ] = σ_true
        
        θ = GaussMLE.θ_xynbs(x_true, y_true, n_true, bg_true, σ_true)
        args = GaussMLE.GaussModel.Args_xynbs(Float32(σ_psf))
        
    elseif model_type == :xynbz
        z_true = Float32(0.2)  # 200nm above focal plane
        true_params[:x] = x_true
        true_params[:y] = y_true
        true_params[:z] = z_true
        true_params[:n] = n_true
        true_params[:bg] = bg_true
        
        θ = GaussMLE.θ_xynbz(x_true, y_true, z_true, n_true, bg_true)
        calib = GaussMLE.AstigmaticCalibration{Float32}(
            1.5f0, 1.5f0, 0.4f0, 0.4f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
        args = GaussMLE.GaussModel.Args_xynbz{Float32}(calib)
    else
        error("Unknown model type: $model_type")
    end
    
    # Generate noise-free data for all ROIs
    roi_template = zeros(Float32, boxsize, boxsize)
    for j in 1:boxsize
        for i in 1:boxsize
            roi_template[i, j] = GaussMLE.model(θ, args, i, j)
        end
    end
    
    # Add Poisson noise to each ROI
    for roi_idx in 1:n_rois
        for j in 1:boxsize
            for i in 1:boxsize
                # Add Poisson noise
                data[i, j, roi_idx] = Float32(rand(Poisson(roi_template[i, j])))
            end
        end
    end
    
    return data, true_params, (model_type == :xynbz ? args.calib : nothing)
end

# Helper function to compute statistics
function compute_statistics(fitted_params, true_params, param_names, uncertainties=nothing)
    stats = Dict{Symbol, Dict{String, Float64}}()
    
    for (i, param) in enumerate(param_names)
        fitted = [p[i] for p in fitted_params]
        true_val = true_params[param]
        
        # Remove outliers (3σ from median)
        median_val = median(fitted)
        mad_val = median(abs.(fitted .- median_val))
        σ_robust = 1.4826 * mad_val  # Robust estimate of std
        mask = abs.(fitted .- median_val) .< 3σ_robust
        fitted_clean = fitted[mask]
        
        # Compute statistics
        bias = mean(fitted_clean) - true_val
        std_dev = std(fitted_clean)
        
        # Get reported uncertainties if available
        reported_std = nothing
        if uncertainties !== nothing
            uncert_clean = [u[i] for u in uncertainties[mask]]
            reported_std = mean(uncert_clean)
        end
        
        stats[param] = Dict(
            "bias" => bias,
            "std" => std_dev,
            "reported_std" => reported_std,
            "outliers_removed" => sum(.!mask)
        )
    end
    
    return stats
end

# Test working models only
models = [:xynb, :xynbs, :xynbz]  # Skip xynbsxsy for now
model_names = Dict(
    :xynb => "Basic 4-parameter (x,y,n,bg)",
    :xynbs => "5-parameter with PSF width (x,y,n,bg,σ)",
    :xynbz => "5-parameter z-model (x,y,z,n,bg)"
)

param_names = Dict(
    :xynb => [:x, :y, :n, :bg],
    :xynbs => [:x, :y, :n, :bg, :σ],
    :xynbz => [:x, :y, :z, :n, :bg]
)

println("\nGenerating test data and running benchmarks...")
for model in models
    println("\n" * "-"^80)
    println("Model: $(model_names[model])")
    println("-"^80)
    
    # Generate test data
    print("  Generating $(n_rois) ROIs...")
    data, true_params, calib = generate_test_data_fixed(model, n_rois, boxsize; 
                                                       n_photons=n_photons, 
                                                       bg_photons=bg_photons, 
                                                       σ_psf=σ_psf)
    println(" done")
    
    # Test CPU
    println("\n  CPU Backend:")
    cpu_time = @elapsed begin
        θ_cpu, Σ_cpu = GaussMLE.fitstack(data, model; backend=:cpu, calib=calib)
    end
    cpu_fits_per_sec = n_rois / cpu_time
    println("    Time: $(round(cpu_time, digits=2)) seconds")
    println("    Fits/sec: $(round(Int, cpu_fits_per_sec))")
    
    # Extract fitted values
    cpu_fitted = []
    cpu_uncert = []
    if model == :xynb
        cpu_fitted = [[θ.x, θ.y, θ.n, θ.bg] for θ in θ_cpu]
        cpu_uncert = [[σ.σ_x, σ.σ_y, σ.σ_n, σ.σ_bg] for σ in Σ_cpu]
    elseif model == :xynbs
        cpu_fitted = [[θ.x, θ.y, θ.n, θ.bg, θ.σ_PSF] for θ in θ_cpu]
        cpu_uncert = [[σ.σ_x, σ.σ_y, σ.σ_n, σ.σ_bg, σ.σ_σ_PSF] for σ in Σ_cpu]
    elseif model == :xynbz
        cpu_fitted = [[θ.x, θ.y, θ.z, θ.n, θ.bg] for θ in θ_cpu]
        cpu_uncert = [[σ.σ_x, σ.σ_y, σ.σ_z, σ.σ_n, σ.σ_bg] for σ in Σ_cpu]
    end
    
    cpu_stats = compute_statistics(cpu_fitted, true_params, param_names[model], cpu_uncert)
    
    # Test GPU if available
    if CUDA.functional()
        println("\n  GPU Backend:")
        # First run (warm-up)
        print("    Warm-up run...")
        _ = GaussMLE.fitstack(data[1:1,1:1,1:10], model; backend=:gpu, calib=calib)
        println(" done")
        
        # Second run (timed)
        gpu_time = @elapsed begin
            θ_gpu, Σ_gpu = GaussMLE.fitstack(data, model; backend=:gpu, calib=calib)
        end
        gpu_fits_per_sec = n_rois / gpu_time
        println("    Time: $(round(gpu_time, digits=3)) seconds")
        println("    Fits/sec: $(round(Int, gpu_fits_per_sec))")
        println("    Speedup: $(round(cpu_time/gpu_time, digits=1))x")
        
        # Extract fitted values
        gpu_fitted = []
        gpu_uncert = []
        if model == :xynb
            gpu_fitted = [[θ.x, θ.y, θ.n, θ.bg] for θ in θ_gpu]
            gpu_uncert = [[σ.σ_x, σ.σ_y, σ.σ_n, σ.σ_bg] for σ in Σ_gpu]
        elseif model == :xynbs
            gpu_fitted = [[θ.x, θ.y, θ.n, θ.bg, θ.σ_PSF] for θ in θ_gpu]
            gpu_uncert = [[σ.σ_x, σ.σ_y, σ.σ_n, σ.σ_bg, σ.σ_σ_PSF] for σ in Σ_gpu]
        elseif model == :xynbz
            gpu_fitted = [[θ.x, θ.y, θ.z, θ.n, θ.bg] for θ in θ_gpu]
            gpu_uncert = [[σ.σ_x, σ.σ_y, σ.σ_z, σ.σ_n, σ.σ_bg] for σ in Σ_gpu]
        end
        
        gpu_stats = compute_statistics(gpu_fitted, true_params, param_names[model], gpu_uncert)
        
        # Store results
        results[String(model)] = Dict(
            "cpu_time" => cpu_time,
            "cpu_fits_per_sec" => cpu_fits_per_sec,
            "gpu_time" => gpu_time,
            "gpu_fits_per_sec" => gpu_fits_per_sec,
            "speedup" => cpu_time/gpu_time,
            "cpu_stats" => cpu_stats,
            "gpu_stats" => gpu_stats,
            "true_params" => true_params
        )
    else
        results[String(model)] = Dict(
            "cpu_time" => cpu_time,
            "cpu_fits_per_sec" => cpu_fits_per_sec,
            "cpu_stats" => cpu_stats,
            "true_params" => true_params
        )
    end
end

# Generate detailed report
println("\n" * "="^80)
println("DETAILED STATISTICAL REPORT")
println("="^80)
println("\nTest Configuration:")
println("  Number of ROIs: $(n_rois)")
println("  ROI size: $(boxsize)x$(boxsize)")
println("  Photons: $n_photons")
println("  Background: $bg_photons photons/pixel")
println("  PSF width: $σ_psf pixels")

if CUDA.functional()
    # Performance summary
    println("\n" * "="^80)
    println("PERFORMANCE SUMMARY")
    println("="^80)
    println(@sprintf("\n%-15s %15s %15s %12s", "Model", "CPU (fits/s)", "GPU (fits/s)", "Speedup"))
    println("-"^60)
    for model in models
        model_str = String(model)
        if haskey(results, model_str)
            r = results[model_str]
            println(@sprintf("%-15s %15.0f %15.0f %12.1fx", 
                    model_str, r["cpu_fits_per_sec"], r["gpu_fits_per_sec"], r["speedup"]))
        end
    end
    
    # Detailed statistics for each model
    println("\n" * "="^80)
    println("PARAMETER STATISTICS (Bias | Std Dev | Reported CRLB)")
    println("="^80)
    
    for model in models
        model_str = String(model)
        if haskey(results, model_str)
            println("\n$(model_str) - $(model_names[model]):")
            
            # Print true values
            r = results[model_str]
            true_vals = r["true_params"]
            println("True values:")
            for (param, val) in true_vals
                println("  $param = $val")
            end
            
            params = param_names[model]
            
            # Header
            println(@sprintf("\n%-10s %25s %25s", "", "CPU", "GPU"))
            println(@sprintf("%-10s %25s %25s", "Parameter", "Bias | Std | CRLB", "Bias | Std | CRLB"))
            println("-"^62)
            
            # Parameter rows
            for param in params
                cpu_s = r["cpu_stats"][param]
                gpu_s = r["gpu_stats"][param]
                
                cpu_str = @sprintf("%7.4f | %7.4f | %7.4f", 
                                  cpu_s["bias"], cpu_s["std"], 
                                  cpu_s["reported_std"] === nothing ? NaN : cpu_s["reported_std"])
                
                gpu_str = @sprintf("%7.4f | %7.4f | %7.4f", 
                                  gpu_s["bias"], gpu_s["std"], 
                                  gpu_s["reported_std"] === nothing ? NaN : gpu_s["reported_std"])
                
                println(@sprintf("%-10s %25s %25s", param, cpu_str, gpu_str))
                
                # Report outliers if significant
                if cpu_s["outliers_removed"] > 10 || gpu_s["outliers_removed"] > 10
                    println(@sprintf("           (Outliers removed: CPU=%d, GPU=%d)", 
                                    cpu_s["outliers_removed"], gpu_s["outliers_removed"]))
                end
            end
            
            # Add CRLB accuracy check
            println("\nCRLB Accuracy (Ratio of empirical std to reported CRLB):")
            println(@sprintf("%-10s %12s %12s", "Parameter", "CPU", "GPU"))
            println("-"^35)
            for param in params
                cpu_s = r["cpu_stats"][param]
                gpu_s = r["gpu_stats"][param]
                
                cpu_ratio = cpu_s["reported_std"] === nothing ? NaN : cpu_s["std"] / cpu_s["reported_std"]
                gpu_ratio = gpu_s["reported_std"] === nothing ? NaN : gpu_s["std"] / gpu_s["reported_std"]
                
                println(@sprintf("%-10s %12.2f %12.2f", param, cpu_ratio, gpu_ratio))
            end
        end
    end
else
    # CPU-only results
    println("\n" * "="^80)
    println("PERFORMANCE SUMMARY (CPU Only)")
    println("="^80)
    println(@sprintf("\n%-15s %15s", "Model", "CPU (fits/s)"))
    println("-"^32)
    for model in models
        model_str = String(model)
        if haskey(results, model_str)
            r = results[model_str]
            println(@sprintf("%-15s %15.0f", model_str, r["cpu_fits_per_sec"]))
        end
    end
end

println("\n" * "="^80)
println("Notes:")
println("- Bias: Mean fitted value - true value")
println("- Std: Standard deviation of fitted values")
println("- CRLB: Cramér-Rao Lower Bound (theoretical minimum uncertainty)")
println("- CRLB Accuracy: Ratio should be ~1.0 for optimal fitting")
println("- Outliers: Removed if >3σ from median (robust statistics)")
println("\nBenchmark completed!")