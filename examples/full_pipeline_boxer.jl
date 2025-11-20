"""
Full Pipeline Example: SMLMBoxer → GaussMLE
Demonstrates complete workflow from detection to fitting

Pipeline:
1. Generate synthetic SMLM image stack
2. Detect spots with SMLMBoxer
3. Fit spots with GaussMLE
4. Get BasicSMLD output for ecosystem integration
"""

using Pkg
Pkg.activate(@__DIR__)

using GaussMLE
using SMLMBoxer
using SMLMData
using Statistics
using Random

Random.seed!(42)

println("=== Full Pipeline: Boxer → GaussMLE ===\n")

# Step 1: Generate synthetic SMLM image stack
println("Step 1: Generating synthetic SMLM data")
println("-" ^ 50)

# Create camera (SMLMData 0.5 API)
camera = SMLMData.IdealCamera(0:255, 0:255, 0.1)  # pixel_edges, 0.1μm/pixel
width = length(camera.pixel_edges_x) - 1
height = length(camera.pixel_edges_y) - 1
pixelsize = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
println("  Camera: $(width)×$(height) pixels, $(pixelsize*1000)nm/pixel")

# Generate image stack with sparse spots
n_frames = 10
n_spots_per_frame = 20
imagestack = zeros(Float32, 256, 256, n_frames)

# Add Gaussian spots at random locations
println("  Generating $n_spots_per_frame spots × $n_frames frames...")
σ_psf = 1.3  # pixels
for frame in 1:n_frames
    for spot in 1:n_spots_per_frame
        # Random position (avoid edges)
        x = rand(20:236)
        y = rand(20:236)
        photons = 1000.0f0 + 500.0f0 * randn(Float32)

        # Add Gaussian spot
        for i in max(1, y-10):min(256, y+10)
            for j in max(1, x-10):min(256, x+10)
                r2 = (j - x)^2 + (i - y)^2
                imagestack[i, j, frame] += photons * exp(-r2 / (2 * σ_psf^2)) / (2π * σ_psf^2)
            end
        end
    end

    # Add noise
    imagestack[:, :, frame] .+= 10.0f0  # Background
    imagestack[:, :, frame] .+= randn(Float32, 256, 256) * 2.0f0  # Readnoise
end

println("  ✓ Generated image stack: $(size(imagestack))")
println("  Mean intensity: $(round(mean(imagestack), digits=2))")

# Step 2: Detect spots with SMLMBoxer
println("\nStep 2: Detecting spots with SMLMBoxer")
println("-" ^ 50)

boxer_result = getboxes(
    imagestack,
    camera;
    boxsize = 11,
    overlap = 2.0,
    sigma_small = 1.0,
    sigma_large = 2.0,
    minval = 10.0,
    use_gpu = false
)

# Check what Boxer returned
if boxer_result isa NamedTuple
    n_detected = boxer_result.metadata.ndetections
    println("  ✓ Detected $n_detected spots")
    println("  Boxes shape: $(size(boxer_result.boxes))")
    println("  First detection (pixels): $(boxer_result.coords_pixels[1, :])")
    println("  First detection (microns): $(boxer_result.coords_microns[1, :])")

    # Step 3: Create ROIBatch for GaussMLE
    println("\nStep 3: Creating ROIBatch for GaussMLE")
    println("-" ^ 50)

    # Convert Boxer output to ROIBatch
    batch = SMLMData.ROIBatch(
        boxer_result.boxes,
        Int32.(permutedims(boxer_result.boxcoords[:, 1:2])),  # (N×2) → (2×N)
        Int32.(boxer_result.boxcoords[:, 3]),                 # Frame indices
        camera
    )

    println("  ✓ Created ROIBatch with $n_detected ROIs")
    println("  Type: $(typeof(batch))")
elseif boxer_result isa SMLMData.ROIBatch
    # Boxer already returned ROIBatch!
    batch = boxer_result
    n_detected = length(batch)
    println("  ✓ Boxer returned ROIBatch directly!")
    println("  ROIBatch with $n_detected ROIs")
    println("  Type: $(typeof(batch))")
end

# Step 4: Fit with GaussMLE
println("\nStep 4: Fitting with GaussMLE")
println("-" ^ 50)

fitter = GaussMLEFitter(
    psf_model = GaussianXYNB(1.3f0),
    device = :cpu,
    iterations = 20
)

println("  Fitter configuration:")
println("    ", fitter)

results = fit(fitter, batch)
println("\n  ✓ Fitted $(results.n_fits) localizations")
println("  Result type: $(typeof(results).name.name)")

# Step 5: Convert to SMLD (ecosystem standard)
println("\nStep 5: Convert to BasicSMLD (ecosystem standard)")
println("-" ^ 50)

smld = GaussMLE.to_smld(results, batch)
println("  ✓ Created BasicSMLD")
println("  Type: $(typeof(smld))")
println("  Number of emitters: $(length(smld.emitters))")
println("  Number of frames: $(smld.n_frames)")
println("  Camera: $(typeof(smld.camera).name.name)")

# Step 6: Results analysis from SMLD
println("\nStep 6: Analyze SMLD Results")
println("-" ^ 50)

# Extract positions and photons
x_positions = [e.x for e in smld.emitters]
y_positions = [e.y for e in smld.emitters]
photons = [e.photons for e in smld.emitters]
x_errors = [e.σ_x for e in smld.emitters]

println("  Mean position (microns):")
println("    x: $(round(mean(x_positions), digits=3)) ± $(round(mean(x_errors)*1000, digits=1)) nm")
println("    y: $(round(mean(y_positions), digits=3)) ± $(round(mean(x_errors)*1000, digits=1)) nm")
println("  Mean photons: $(round(mean(photons), digits=1))")
println("  Localization precision: $(round(mean(x_errors)*1000, digits=1)) nm")

# First emitter details
if !isempty(smld.emitters)
    e1 = smld.emitters[1]
    println("\n  First emitter:")
    println("    Position: ($(round(e1.x, digits=3)), $(round(e1.y, digits=3))) μm")
    println("    Photons: $(round(e1.photons, digits=1))")
    println("    σ: ($(round(e1.σ_x*1000, digits=1)), $(round(e1.σ_y*1000, digits=1))) nm")
    println("    Frame: $(e1.frame)")
end

println("\n" * "="^50)
println("✅ FULL PIPELINE COMPLETE!")
println("="^50)
println("\nWorkflow Summary:")
println("  Image Stack → SMLMBoxer → ROIBatch → GaussMLE → BasicSMLD")
println("\nOutput ready for:")
println("  - SMLMData ecosystem tools")
println("  - Drift correction (SMLMDriftCorrection.jl)")
println("  - Rendering (SMLMRender.jl)")
println("  - Analysis (SMLMAnalysis.jl)")
