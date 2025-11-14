"""
Demo of ROIBatch functionality with SMLMData integration
Shows how to fit ROIs with proper coordinate tracking and sCMOS support
"""

using Pkg
Pkg.activate(".")  # Use main project environment

using GaussMLE
using SMLMData
using Random
using Statistics
using StaticArrays
using Printf

println("=== ROIBatch Demo with SMLMData Integration ===\n")

# Set random seed for reproducibility
Random.seed!(42)

# Parameters
roi_size = 11
n_rois = 10
camera_size = 512
pixel_size_um = 0.1  # 100nm pixels

println("Configuration:")
println("  ROI size: $roi_size × $roi_size pixels")
println("  Number of ROIs: $n_rois")
println("  Camera: $camera_size × $camera_size pixels")
println("  Pixel size: $(pixel_size_um*1000) nm\n")

# 1. Generate synthetic ROI data using new simulator
println("1. Generating synthetic ROI data...")

# Create ideal camera for initial demo
ideal_camera = SMLMData.IdealCamera(camera_size, camera_size, pixel_size_um)

# PSF model
psf_model = GaussMLE.GaussianXYNB(1.3f0)

# Generate ROI batch with simulator
roi_batch = generate_roi_batch(
    ideal_camera, 
    psf_model;
    n_rois = n_rois,
    roi_size = roi_size,
    corner_mode = :random,
    seed = 42
)
println("  Created ROIBatch with $(length(roi_batch)) ROIs")
println("  Frames covered: $(minimum(roi_batch.frame_indices)) to $(maximum(roi_batch.frame_indices))")

# 2. Fit with ideal camera model
println("\n2. Fitting with ideal camera model...")

fitter = GaussMLEFitter(
    psf_model = psf_model,
    device = CPU(),
    iterations = 20
)

results_ideal = fit(fitter, roi_batch)
println("  Fitting complete!")

# Show results for first few ROIs
println("\n  First 3 ROI results (ideal camera):")
println("  ROI | X (cam) | Y (cam) | Photons | Background")
println("  ----|---------|---------|---------|------------")
for i in 1:min(3, n_rois)
    @printf("   %2d | %7.2f | %7.2f | %7.0f | %10.1f\n",
            i, results_ideal.x_camera[i], results_ideal.y_camera[i],
            results_ideal.parameters[3, i], results_ideal.parameters[4, i])
end

# 3. Create sCMOS variance map and fit
println("\n3. Creating sCMOS variance map and fitting...")

# Create sCMOS camera with variance map
# Create spatially varying readnoise map using SMLMData 0.4 API
# Base readnoise: sqrt(10) ≈ 3.16 e⁻ rms, High noise: sqrt(50) ≈ 7.07 e⁻ rms
readnoise_map = ones(Float32, camera_size, camera_size) * sqrt(10.0f0)

# Add some regions with higher noise
for j in 200:300, i in 200:300
    readnoise_map[i, j] = sqrt(50.0f0)  # Higher noise region
end

scmos_camera = SMLMData.SCMOSCamera(
    camera_size, camera_size, Float32(pixel_size_um), readnoise_map,
    offset = 100.0f0,
    gain = 0.5f0,
    qe = 0.82f0
)

println("  sCMOS camera created with spatial variation")
println("  Base readnoise: $(sqrt(10.0f0)) e⁻ rms (variance: 10 e⁻²)")
println("  High-noise region: $(sqrt(50.0f0)) e⁻ rms (variance: 50 e⁻²) at (200:300, 200:300)")

# Generate new data with sCMOS noise
roi_batch_scmos = generate_roi_batch(
    scmos_camera,
    psf_model;
    n_rois = n_rois,
    roi_size = roi_size,
    corners = roi_batch.corners,  # Use same positions as ideal
    frame_indices = roi_batch.frame_indices,
    seed = 43
)

# Fit with sCMOS model
results_scmos = fit(fitter, roi_batch_scmos)
println("  sCMOS fitting complete!")

# Compare uncertainties
println("\n  Uncertainty comparison (first 3 ROIs):")
println("  ROI | σ_x (ideal) | σ_x (sCMOS) | σ_y (ideal) | σ_y (sCMOS)")
println("  ----|-------------|-------------|-------------|-------------")
for i in 1:min(3, n_rois)
    @printf("   %2d |    %.4f   |    %.4f   |    %.4f   |    %.4f\n",
            i, 
            results_ideal.uncertainties[1, i],
            results_scmos.uncertainties[1, i],
            results_ideal.uncertainties[2, i],
            results_scmos.uncertainties[2, i])
end

# 4. Convert to SMLMData format
println("\n4. Converting to SMLMData format...")

# Convert results to SMLD using the camera from roi_batch
smld = to_smld(results_ideal, roi_batch; 
               dataset = 1,
               metadata = Dict{String,Any}("experiment" => "ROIBatch Demo"))

println("  Converted to BasicSMLD:")
println("    - $(length(smld)) emitters")
println("    - $(smld.n_frames) frames")
println("    - Camera: $(camera_size)×$(camera_size) pixels")

# 5. Show coordinate transformations
println("\n5. Coordinate transformations:")
for i in 1:min(3, n_rois)
    emitter = smld.emitters[i]
    roi_x = results_ideal.parameters[1, i]
    roi_y = results_ideal.parameters[2, i]
    cam_x = results_ideal.x_camera[i]
    cam_y = results_ideal.y_camera[i]
    
    println("\n  ROI $i:")
    println("    Corner (pixels): ($(roi_batch.corners[1,i]), $(roi_batch.corners[2,i]))")
    println("    ROI coords: ($(round(roi_x, digits=2)), $(round(roi_y, digits=2))) pixels")
    println("    Camera coords: ($(round(cam_x, digits=2)), $(round(cam_y, digits=2))) pixels")
    println("    Physical coords: ($(round(emitter.x, digits=3)), $(round(emitter.y, digits=3))) μm")
end

# 6. Filter using SMLMData capabilities
println("\n6. Using SMLMData filtering capabilities...")

# Filter by photon count
bright_emitters = SMLMData.@filter(smld, photons > 1400)
println("  Bright emitters (>1400 photons): $(length(bright_emitters))")

# Filter by frame
frame1_emitters = SMLMData.filter_frames(smld, 1)
println("  Frame 1 emitters: $(length(frame1_emitters))")

# Summary statistics
photon_counts = [e.photons for e in smld.emitters]
x_positions = [e.x for e in smld.emitters]
y_positions = [e.y for e in smld.emitters]

println("\n=== Summary Statistics ===")
println("Photons: mean=$(round(mean(photon_counts), digits=1)), " *
        "std=$(round(std(photon_counts), digits=1))")
println("X position (μm): mean=$(round(mean(x_positions), digits=2)), " *
        "range=[$(round(minimum(x_positions), digits=2)), " *
        "$(round(maximum(x_positions), digits=2))]")
println("Y position (μm): mean=$(round(mean(y_positions), digits=2)), " *
        "range=[$(round(minimum(y_positions), digits=2)), " *
        "$(round(maximum(y_positions), digits=2))]")

println("\n✓ Demo complete!")