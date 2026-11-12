"""
Complete SMLM Pipeline: SMLMSim → SMLMBoxer → GaussMLE
Demonstrates end-to-end workflow from simulation to localization

Pipeline:
1. SMLMSim - Generate realistic SMLM image stack
2. SMLMBoxer - Detect and extract ROIs
3. GaussMLE - Fit Gaussian PSFs
4. BasicSMLD - Ecosystem-standard output
"""

using Pkg
Pkg.activate(@__DIR__)

using SMLMSim
using SMLMBoxer
using GaussMLE
using SMLMData
using MicroscopePSFs
using Statistics
using Random

Random.seed!(42)

println("="^60)
println("COMPLETE SMLM PIPELINE")
println("="^60)
println("\nSMLMSim → SMLMBoxer → GaussMLE → BasicSMLD\n")

# Step 1: Generate SMLM data with SMLMSim
println("Step 1: Generate Data (SMLMSim)")
println("-"^50)

# Create camera
camera = SMLMData.IdealCamera(0:255, 0:255, 0.1)  # 256×256, 100nm pixels
println("  Camera: 256×256 pixels, 100nm/pixel")

# Create ground truth emitters (uniform distribution)
println("  Creating ground truth emitters...")
n_emitters = 100
n_frames = 10
emitters_true = SMLMData.Emitter2DFit{Float64}[]

for i in 1:n_emitters
    # Random position (0-25 μm)
    x = rand() * 25.0
    y = rand() * 25.0
    frame = rand(1:n_frames)

    # Positional: x, y, photons, bg, then uncertainties (unknown for ground truth)
    push!(
        emitters_true, SMLMData.Emitter2DFit{Float64}(
            x, y, 1500.0, 10.0, 0.0, 0.0, 0.0, 0.0;
            frame = frame, dataset = 1, track_id = 0, id = i
        )
    )
end

smld_true = SMLMData.BasicSMLD(emitters_true, camera, n_frames, 1, Dict{String, Any}())
println("  ✓ Created ground truth: $n_emitters emitters")

# Generate image stack using SMLMSim
println("  Generating image stack with SMLMSim...")
psf = MicroscopePSFs.GaussianPSF(0.13)  # 130nm PSF
imagestack, _ = SMLMSim.gen_images(smld_true, psf, poisson_noise = true, bg = 10.0)
println("  ✓ Image stack: $(size(imagestack))")
println("  Mean intensity: $(round(mean(imagestack), digits = 2))")

# Step 2: Detect spots with SMLMBoxer
println("\nStep 2: Detect Spots (SMLMBoxer)")
println("-"^50)

batch, boxer_info = getboxes(
    imagestack,
    camera;
    boxsize = 11,
    overlap = 2.0,
    sigma_small = 1.0,
    sigma_large = 2.0,
    minval = 5.0,
    backend = :cpu
)

n_detected = length(batch)
println("  ✓ Detected $n_detected ROIs")
println("  Input emitters: $n_emitters")
println("  Detection efficiency: $(round(n_detected / n_emitters * 100, digits = 1))%")

# Step 3: Fit with GaussMLE
println("\nStep 3: Fit PSFs (GaussMLE)")
println("-"^50)

fitter = GaussMLE.GaussMLEConfig(
    psf_model = GaussMLE.GaussianXYNB(0.13f0),
    backend = :cpu
)

println("  Fitter: GaussianXYNB(σ=0.13 μm = 1.3 pixels), default cap of 100 iterations")

smld_output, fit_info = fit(batch, fitter)
println("  ✓ Fitted $(length(smld_output.emitters)) localizations → BasicSMLD")
println("  $(fit_info.n_converged) of $(fit_info.n_fits) fits converged")

# Step 4: Analyze Results
println("\nStep 4: Analyze Results")
println("-"^50)

# Extract statistics
x_positions = [e.x for e in smld_output.emitters]
photons_fit = [e.photons for e in smld_output.emitters]
precision = [e.σ_x for e in smld_output.emitters]

println("  Position range:")
println("    x: $(round(minimum(x_positions), digits = 2)) - \
    $(round(maximum(x_positions), digits = 2)) μm")
println("  Mean photons: $(round(mean(photons_fit), digits = 1)) (input: 1500)")
println("  Mean precision: $(round(mean(precision) * 1000, digits = 1)) nm")

# Ground truth comparison
println("\nStep 5: Ground Truth Comparison")
println("-"^50)

println("  Input: $n_emitters emitters")
println("  Detected: $n_detected spots")
println("  Fitted: $(length(smld_output.emitters)) localizations")
println(
    "  Recovery: $(round(length(smld_output.emitters) / n_emitters * 100, digits = 1))%",
)

# Match each localization to the nearest true emitter in the same frame (within 0.3 μm)
dx = Float64[]
dy = Float64[]
for e in smld_output.emitters
    best = nothing
    best_d = 0.3
    for t in emitters_true
        t.frame == e.frame || continue
        d = hypot(e.x - t.x, e.y - t.y)
        if d < best_d
            best = t
            best_d = d
        end
    end
    if best !== nothing
        push!(dx, e.x - best.x)
        push!(dy, e.y - best.y)
    end
end
println("  Matched to a true emitter: $(length(dx)) of $(length(smld_output.emitters))")
if !isempty(dx)
    # The median is robust to the occasional pair of emitters that land within one PSF
    println("  Median |position error| (nm): x = \
        $(round(median(abs.(dx)) * 1000, digits = 1)), \
        y = $(round(median(abs.(dy)) * 1000, digits = 1))")
end

println("\n" * "="^60)
println("✅ COMPLETE PIPELINE SUCCESS!")
println("="^60)
println("\nWorkflow:")
println("  1. SMLMSim   → gen_images() creates image stack")
println("  2. SMLMBoxer → Detects spots, returns ROIBatch")
println("  3. GaussMLE  → fit() returns BasicSMLD directly!")
println("\nBasicSMLD output ready for ecosystem tools!")
