"""
GPU Acceleration Example
Demonstrates GPU batch processing for large datasets
"""

# Run from the repository root: julia --project examples/gpu_acceleration.jl

using GaussMLE
using CUDA
using SMLMData
using Statistics
using Printf

# Enable the AMD extension if AMDGPU is in this environment and CUDA is unavailable.
if !CUDA.functional() && Base.find_package("AMDGPU") !== nothing
    using AMDGPU
end

println("=== GPU Acceleration Example ===\n")

# Generate large dataset
n_rois = 10_000
println("Generating large dataset: $n_rois ROIs...")
data = rand(Float32, 11, 11, n_rois)

# CPU fitting
println("\n--- CPU Fitting ---")
fitter_cpu = GaussMLEConfig(backend = :cpu)
println("Running CPU fit...")
t_cpu = @elapsed begin
    smld_cpu, info_cpu = fit(data, fitter_cpu)
end
rate_cpu = length(smld_cpu.emitters) / t_cpu
@printf("Time: %.3f seconds\n", t_cpu)
@printf("Rate: %.0f ROIs/second\n", rate_cpu)

# GPU fitting (CUDA or AMDGPU)
gpu_available = CUDA.functional() ||
    (isdefined(@__MODULE__, :AMDGPU) && AMDGPU.functional())
if gpu_available
    println("\n--- GPU Fitting ---")
    fitter_gpu = GaussMLEConfig(backend = :gpu, batch_size = 5000)
    println("Running GPU fit (batch size: 5000)...")
    t_gpu = @elapsed begin
        smld_gpu, info_gpu = fit(data, fitter_gpu)
    end
    rate_gpu = length(smld_gpu.emitters) / t_gpu
    @printf("Time: %.3f seconds\n", t_gpu)
    @printf("Rate: %.0f ROIs/second\n", rate_gpu)

    # Compare
    if t_gpu < t_cpu
        speedup = t_cpu / t_gpu
        @printf("\nSpeedup: %.1fx faster on GPU\n", speedup)
    else
        println("\nCPU faster for this batch size")
    end

    # Verify consistency
    x_cpu = [e.x for e in smld_cpu.emitters]
    x_gpu = [e.x for e in smld_gpu.emitters]
    photons_cpu = [e.photons for e in smld_cpu.emitters]
    photons_gpu = [e.photons for e in smld_gpu.emitters]

    mean_diff_x = mean(abs.(x_cpu .- x_gpu))
    mean_diff_photons = mean(abs.(photons_cpu .- photons_gpu))

    println("\n=== Result Consistency ===")
    @printf("Mean |Δx|: %.2e μm\n", mean_diff_x)
    @printf("Mean |Δphotons|: %.2e\n", mean_diff_photons)
else
    @warn "No functional CUDA or AMD GPU found; GPU comparison is unavailable."
end

println("\n✓ GPU acceleration example complete!")
println("Output: BasicSMLD ready for ecosystem")
