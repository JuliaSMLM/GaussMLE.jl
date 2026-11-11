"""
Demo of ROIBatch functionality with SMLMData integration
Shows how to fit ROIs with proper coordinate tracking and sCMOS support
"""

using Pkg
Pkg.activate(@__DIR__)

using GaussMLE
using SMLMData
using Random
using Statistics
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
println("  Pixel size: $(pixel_size_um * 1000) nm\n")

# 1. Generate synthetic ROI data using new simulator
println("1. Generating synthetic ROI data...")

# Create ideal camera for initial demo
ideal_camera = SMLMData.IdealCamera(camera_size, camera_size, pixel_size_um)

# PSF model
psf_model = GaussMLE.GaussianXYNB(0.13f0)

# True parameters (rows: x, y, photons, background). x and y are ROI-local pixels;
# photons vary between ROIs so that the photon filter below selects a subset.
true_params = Float32[
    6.0 .+ 0.5f0 * randn(Float32, n_rois)';
    6.0 .+ 0.5f0 * randn(Float32, n_rois)';
    range(600.0f0, 1400.0f0; length = n_rois)';
    fill(10.0f0, 1, n_rois)
]

# Generate ROI batch with simulator
roi_batch = generate_roi_batch(
    ideal_camera,
    psf_model;
    n_rois = n_rois,
    roi_size = roi_size,
    true_params = true_params,
    corner_mode = :random,
    seed = 42
)
println("  Created ROIBatch with $(length(roi_batch)) ROIs")
println(
    "  Frames covered: $(minimum(roi_batch.frame_indices)) to \
        $(maximum(roi_batch.frame_indices))",
)

# 2. Fit with ideal camera model
println("\n2. Fitting with ideal camera model...")

fitter = GaussMLEConfig(
    psf_model = psf_model,
    backend = :cpu
)

smld_ideal, info_ideal = fit(roi_batch, fitter)
println("  Fitting complete: $(info_ideal.n_converged) of $(info_ideal.n_fits) converged")

# Emitters are in the order of the ROIs. Positions are microns on the sensor;
# camera pixel k spans (k-1)*pixel_size to k*pixel_size.
cam_x(e) = e.x / pixel_size_um + 0.5
cam_y(e) = e.y / pixel_size_um + 0.5

# Show results for first few ROIs
println("\n  First 3 ROI results (ideal camera):")
println("  ROI | X (cam) | Y (cam) | Photons (true) | Background")
println("  ----|---------|---------|----------------|------------")
for i in 1:min(3, n_rois)
    e = smld_ideal.emitters[i]
    @printf(
        "   %2d | %7.2f | %7.2f | %7.0f (%5.0f) | %10.1f\n",
        i, cam_x(e), cam_y(e), e.photons, true_params[3, i], e.bg
    )
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
println(
    "  High-noise region: $(sqrt(50.0f0)) e⁻ rms (variance: 50 e⁻²) at (200:300, 200:300)",
)

# Generate new data with sCMOS noise
roi_batch_scmos = generate_roi_batch(
    scmos_camera,
    psf_model;
    n_rois = n_rois,
    roi_size = roi_size,
    true_params = true_params,
    corners = vcat(roi_batch.x_corners', roi_batch.y_corners'),  # Same as ideal
    frame_indices = roi_batch.frame_indices,
    seed = 43
)

# Fit with sCMOS model
smld_scmos, info_scmos = fit(roi_batch_scmos, fitter)
println(
    "  sCMOS fitting complete: $(info_scmos.n_converged) of $(info_scmos.n_fits) converged",
)

# Compare uncertainties
println("\n  Uncertainty comparison (first 3 ROIs):")
println("  ROI | σ_x (ideal) | σ_x (sCMOS) | σ_y (ideal) | σ_y (sCMOS)")
println("  ----|-------------|-------------|-------------|-------------")
for i in 1:min(3, n_rois)
    ei = smld_ideal.emitters[i]
    es = smld_scmos.emitters[i]
    @printf(
        "   %2d |    %.4f   |    %.4f   |    %.4f   |    %.4f\n",
        i, ei.σ_x, es.σ_x, ei.σ_y, es.σ_y
    )
end

# 4. The fit returns SMLMData format directly
println("\n4. The fit result is a BasicSMLD")

smld = smld_ideal
println("  BasicSMLD from the ideal camera fit:")
println("    - $(length(smld)) emitters")
println("    - $(smld.n_frames) frames")
println("    - Camera: $(camera_size)×$(camera_size) pixels")

# 5. Show coordinate transformations
println("\n5. Coordinate transformations:")
for i in 1:min(3, n_rois)
    emitter = smld.emitters[i]
    cx = cam_x(emitter)
    cy = cam_y(emitter)
    cx_corner = roi_batch.x_corners[i]
    cy_corner = roi_batch.y_corners[i]
    # ROI-local pixel position: camera position minus corner offset
    roi_x = cx - cx_corner + 1
    roi_y = cy - cy_corner + 1

    println("\n  ROI $i:")
    println("    Corner (pixels): ($cx_corner, $cy_corner)")
    println(
        "    ROI coords: ($(round(roi_x, digits = 2)), \
            $(round(roi_y, digits = 2))) pixels \
            (true: $(round(true_params[1, i], digits = 2)), \
            $(round(true_params[2, i], digits = 2)))",
    )
    println(
        "    Camera coords: ($(round(cx, digits = 2)), \
            $(round(cy, digits = 2))) pixels",
    )
    println(
        "    Physical coords: ($(round(emitter.x, digits = 3)), \
            $(round(emitter.y, digits = 3))) μm",
    )
end

# 6. Filter using SMLMData capabilities
println("\n6. Using SMLMData filtering capabilities...")

# Filter by photon count
bright_emitters = SMLMData.@filter(smld, photons > 1000)
println("  Bright emitters (>1000 photons): $(length(bright_emitters))")

# Filter by frame
frame1_emitters = SMLMData.filter_frames(smld, 1)
println("  Frame 1 emitters: $(length(frame1_emitters))")

# Summary statistics
photon_counts = [e.photons for e in smld.emitters]
x_positions = [e.x for e in smld.emitters]
y_positions = [e.y for e in smld.emitters]

println("\n=== Summary Statistics ===")
println(
    "Photons: mean=$(round(mean(photon_counts), digits = 1)), " *
        "std=$(round(std(photon_counts), digits = 1))"
)
println(
    "X position (μm): mean=$(round(mean(x_positions), digits = 2)), " *
        "range=[$(round(minimum(x_positions), digits = 2)), " *
        "$(round(maximum(x_positions), digits = 2))]"
)
println(
    "Y position (μm): mean=$(round(mean(y_positions), digits = 2)), " *
        "range=[$(round(minimum(y_positions), digits = 2)), " *
        "$(round(maximum(y_positions), digits = 2))]"
)

println("\n✓ Demo complete!")
