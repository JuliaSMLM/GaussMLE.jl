using GaussMLE
using LinearAlgebra
using Statistics
using Printf
using StaticArrays

# Test Fisher Information calculation independently
println("Testing Fisher Information Matrix Calculation")
println("="^60)

# Parameters
roi_size = 9
nominal_x = 5.0f0
nominal_y = 5.0f0
nominal_photons = 1000.0f0
nominal_bg = 10.0f0
σ = 1.3f0

# Create a single test ROI
psf_model = GaussMLE.GaussianXYNB(σ)
camera_model = GaussMLE.IdealCamera()

# Generate one test spot
spots = GaussMLE.generate_random_spots(
    1, 
    roi_size;
    x_mean = nominal_x,
    y_mean = nominal_y,
    x_std = 0.0f0,  # No variation - exact position
    y_std = 0.0f0,
    photons_mean = nominal_photons,
    photons_std = 0.0f0,
    background_mean = nominal_bg,
    background_std = 0.0f0,
    model_type = :xynb,
    seed = 42
)

data, true_positions = GaussMLE.generate_spots_data(
    psf_model,
    spots,
    roi_size;
    camera_model = camera_model,
    seed = 42
)

# Extract single ROI
roi = data[:, :, 1]
true_params = SVector{4,Float32}(
    true_positions[:x][1],
    true_positions[:y][1],
    true_positions[:photons][1],
    true_positions[:background][1]
)

println("True parameters:")
println("  x = $(true_params[1])")
println("  y = $(true_params[2])")
println("  N = $(true_params[3])")
println("  bg = $(true_params[4])")
println()

# Manually calculate Fisher Information Matrix at true parameters
function calculate_fisher_manual(roi, true_params, psf_model, camera_model)
    N = 4  # Number of parameters
    FI = zeros(Float32, N, N)
    box_size = size(roi, 1)
    
    # Use true parameters for Fisher calculation
    θ = true_params
    
    for j in 1:box_size, i in 1:box_size
        # Compute model and derivatives at this pixel
        model, dudt, _ = GaussMLE.compute_pixel_derivatives(i, j, θ, psf_model)
        
        # Fisher Information accumulation (exactly as SMITE does)
        if model > 0.0f0
            for k in 1:N, l in k:N
                F_kl = dudt[k] * dudt[l] / model
                FI[k,l] += F_kl
                k != l && (FI[l,k] += F_kl)  # Symmetric
            end
        end
    end
    
    return FI
end

# Calculate Fisher Information manually
FI_manual = calculate_fisher_manual(roi, true_params, psf_model, camera_model)

println("Manual Fisher Information Matrix:")
for i in 1:4
    println(@sprintf("  [%12.2f %12.2f %12.2f %12.2f]", 
                    FI_manual[i,1], FI_manual[i,2], FI_manual[i,3], FI_manual[i,4]))
end
println()

# Compute CRLB from manual Fisher
FI_inv_manual = inv(FI_manual)
crlb_manual = sqrt.(diag(FI_inv_manual))

println("Manual CRLB (from Fisher inverse):")
println(@sprintf("  x:  %.4f", crlb_manual[1]))
println(@sprintf("  y:  %.4f", crlb_manual[2]))
println(@sprintf("  N:  %.4f", crlb_manual[3]))
println(@sprintf("  bg: %.4f", crlb_manual[4]))
println()

# Now run the actual fitter to compare
fitter = GaussMLE.GaussMLEFitter(
    psf_model = psf_model,
    camera_model = camera_model,
    device = :cpu,
    iterations = 20,
    batch_size = 1
)

results = GaussMLE.fit(fitter, data)

println("Fitter results:")
println(@sprintf("  Fitted x:  %.4f (true: %.4f)", results.parameters[1,1], true_params[1]))
println(@sprintf("  Fitted y:  %.4f (true: %.4f)", results.parameters[2,1], true_params[2]))
println(@sprintf("  Fitted N:  %.4f (true: %.4f)", results.parameters[3,1], true_params[3]))
println(@sprintf("  Fitted bg: %.4f (true: %.4f)", results.parameters[4,1], true_params[4]))
println()

println("Fitter CRLB:")
println(@sprintf("  x:  %.4f", results.uncertainties[1,1]))
println(@sprintf("  y:  %.4f", results.uncertainties[2,1]))
println(@sprintf("  N:  %.4f", results.uncertainties[3,1]))
println(@sprintf("  bg: %.4f", results.uncertainties[4,1]))
println()

# Compare ratios
println("CRLB Ratios (Fitter / Manual):")
println(@sprintf("  x:  %.3f", results.uncertainties[1,1] / crlb_manual[1]))
println(@sprintf("  y:  %.3f", results.uncertainties[2,1] / crlb_manual[2]))
println(@sprintf("  N:  %.3f", results.uncertainties[3,1] / crlb_manual[3]))
println(@sprintf("  bg: %.3f", results.uncertainties[4,1] / crlb_manual[4]))
println()

# Also calculate Fisher at the fitted position to see if that's different
fitted_params = SVector{4,Float32}(
    results.parameters[1,1],
    results.parameters[2,1],
    results.parameters[3,1],
    results.parameters[4,1]
)

FI_at_fitted = calculate_fisher_manual(roi, fitted_params, psf_model, camera_model)
FI_inv_fitted = inv(FI_at_fitted)
crlb_at_fitted = sqrt.(diag(FI_inv_fitted))

println("CRLB at fitted position:")
println(@sprintf("  x:  %.4f", crlb_at_fitted[1]))
println(@sprintf("  y:  %.4f", crlb_at_fitted[2]))
println(@sprintf("  N:  %.4f", crlb_at_fitted[3]))
println(@sprintf("  bg: %.4f", crlb_at_fitted[4]))
println()

# Check theoretical value for N
println("Theoretical values:")
println(@sprintf("  √N = %.4f (pure Poisson, no bg)", sqrt(nominal_photons)))
println(@sprintf("  √(N + total_bg) = %.4f (with background)", sqrt(nominal_photons + roi_size^2 * nominal_bg)))