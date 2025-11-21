using GaussMLE, SMLMData, CUDA

# Test sCMOS variance map indexing
println("Testing sCMOS variance map indexing...")

# Create a camera with spatially-varying readnoise
readnoise_map = Float32[
    (i + j) / 100.0  # Varies from 0.02 to 5.12 e⁻
    for i in 1:256, j in 1:256
]

scmos = SMLMData.SCMOSCamera(
    256, 256, 0.1f0, readnoise_map,
    offset=100.0f0, gain=0.5f0, qe=1.0f0
)

# Generate a single ROI at a known corner
psf = GaussMLE.GaussianXYNB(0.13f0)
batch = GaussMLE.generate_roi_batch(scmos, psf; n_rois=1, roi_size=11)

println("Corner: ", batch.corners[:, 1])
corner_x, corner_y = batch.corners[1, 1], batch.corners[2, 1]

# Check what variance values should be accessed
println("\nExpected variance values (readnoise²) for this ROI:")
for i in 1:3  # Just check first few pixels
    for j in 1:3
        cam_i = corner_y + i - 1
        cam_j = corner_x + j - 1
        expected_var = readnoise_map[cam_i, cam_j]^2
        println("  Pixel ($i,$j) -> Camera ($cam_i,$cam_j): readnoise=$(readnoise_map[cam_i, cam_j]) e⁻, variance=$(expected_var) e⁻²")
    end
end

# Now fit and check what uncertainties we get
println("\nFitting on CPU...")
fitter_cpu = GaussMLE.GaussMLEFitter(psf_model=psf, device=:cpu, iterations=20)
result_cpu = GaussMLE.fit(fitter_cpu, batch)

println("CPU fit result:")
println("  x = $(result_cpu.emitters[1].x)")
println("  σ_x = $(result_cpu.emitters[1].σ_x)")
println("  photons = $(result_cpu.emitters[1].photons)")
println("  σ_photons = $(result_cpu.emitters[1].σ_photons)")

if CUDA.functional()
    println("\nFitting on GPU...")
    fitter_gpu = GaussMLE.GaussMLEFitter(psf_model=psf, device=:gpu, iterations=20)
    result_gpu = GaussMLE.fit(fitter_gpu, batch)

    println("GPU fit result:")
    println("  x = $(result_gpu.emitters[1].x)")
    println("  σ_x = $(result_gpu.emitters[1].σ_x)")
    println("  photons = $(result_gpu.emitters[1].photons)")
    println("  σ_photons = $(result_gpu.emitters[1].σ_photons)")

    # Compare uncertainties
    println("\nUncertainty ratios (GPU / CPU):")
    println("  σ_x ratio = $(result_gpu.emitters[1].σ_x / result_cpu.emitters[1].σ_x)")
    println("  σ_photons ratio = $(result_gpu.emitters[1].σ_photons / result_cpu.emitters[1].σ_photons)")
end
