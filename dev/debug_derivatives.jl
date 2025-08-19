using GaussMLE
using LinearAlgebra
using Statistics
using Printf
using StaticArrays

# Debug derivative calculations
println("Debugging Derivative Calculations")
println("="^60)

# Parameters
roi_size = 9
x = 5.0f0
y = 5.0f0
N = 1000.0f0
bg = 10.0f0
σ = 1.3f0

θ = SVector{4,Float32}(x, y, N, bg)
psf_model = GaussMLE.GaussianXYNB(σ)

# Check derivatives at center pixel and neighbors
test_pixels = [
    (5, 5, "center"),
    (4, 5, "left"),
    (6, 5, "right"),
    (5, 4, "above"),
    (5, 6, "below"),
    (1, 1, "corner"),
    (9, 9, "opposite corner")
]

println("Pixel derivatives:")
println("-"^60)
println(@sprintf("%-15s %10s %10s %10s %10s %10s", 
                "Position", "Model", "dudt_x", "dudt_y", "dudt_N", "dudt_bg"))
println("-"^60)

total_psf = 0.0f0
total_fisher_N = 0.0f0

for (i, j, label) in test_pixels
    model, dudt, _ = GaussMLE.compute_pixel_derivatives(i, j, θ, psf_model)
    psf_value = (model - bg) / N  # Extract PSF value
    
    println(@sprintf("(%d,%d) %-8s %10.4f %10.4f %10.4f %10.4f %10.4f", 
                    i, j, label, model, dudt[1], dudt[2], dudt[3], dudt[4]))
    
    # dudt[3] should equal PSF value for ∂/∂N
    if abs(dudt[3] - psf_value) > 1e-6
        println("  WARNING: dudt_N ($(dudt[3])) != PSF ($(psf_value))")
    end
    
    global total_psf += psf_value
    if model > 0
        global total_fisher_N += dudt[3] * dudt[3] / model
    end
end

println()
println("Sum of PSF values (first 7 pixels): $(total_psf)")
println("Partial Fisher Info for N (first 7 pixels): $(total_fisher_N)")

# Now calculate full Fisher Information for N
println()
println("Full Fisher Information calculation:")
println("-"^40)

FI_N = 0.0f0
sum_psf = 0.0f0
sum_psf_squared = 0.0f0

for j in 1:roi_size, i in 1:roi_size
    model, dudt, _ = GaussMLE.compute_pixel_derivatives(i, j, θ, psf_model)
    psf_value = (model - bg) / N
    
    global sum_psf += psf_value
    global sum_psf_squared += psf_value^2
    
    if model > 0
        global FI_N += dudt[3] * dudt[3] / model
    end
end

println(@sprintf("Sum of PSF over ROI:           %.6f", sum_psf))
println(@sprintf("Sum of PSF² over ROI:          %.6f", sum_psf_squared))
println(@sprintf("Fisher Info for N:             %.6f", FI_N))
println(@sprintf("CRLB for N (1/√FI):            %.4f", 1/sqrt(FI_N)))
println()

# Check what the Fisher should be theoretically
println("Theoretical analysis:")
println("-"^40)

# For ideal Poisson with model = bg + N*PSF:
# FI_N = Σ (∂model/∂N)² / model = Σ PSF² / (bg + N*PSF)

theoretical_FI_N = 0.0f0
for j in 1:roi_size, i in 1:roi_size
    # Use integrated Gaussian
    psf_x = GaussMLE.GaussLib.integral_gaussian_1d(i, x, σ)
    psf_y = GaussMLE.GaussLib.integral_gaussian_1d(j, y, σ)
    psf_value = psf_x * psf_y
    model_value = bg + N * psf_value
    
    if model_value > 0
        global theoretical_FI_N += psf_value^2 / model_value
    end
end

println(@sprintf("Theoretical Fisher for N:      %.6f", theoretical_FI_N))
println(@sprintf("Theoretical CRLB (1/√FI):      %.4f", 1/sqrt(theoretical_FI_N)))
println(@sprintf("Simple √N:                     %.4f", sqrt(N)))

# Debug: Check if dudt[3] is actually PSF or something else
println()
println("Verifying dudt[3] calculation:")
println("-"^40)

# Pick center pixel
i, j = 5, 5
model, dudt, _ = GaussMLE.compute_pixel_derivatives(i, j, θ, psf_model)

# Manually calculate what dudt[3] should be
psf_x = GaussMLE.GaussLib.integral_gaussian_1d(i, x, σ)
psf_y = GaussMLE.GaussLib.integral_gaussian_1d(j, y, σ)
expected_dudt_N = psf_x * psf_y

println(@sprintf("At pixel (5,5):"))
println(@sprintf("  dudt[3] from function:        %.6f", dudt[3]))
println(@sprintf("  Expected (PSF_x * PSF_y):     %.6f", expected_dudt_N))
println(@sprintf("  Ratio:                        %.6f", dudt[3] / expected_dudt_N))

if abs(dudt[3] - expected_dudt_N) > 1e-6
    println("  ERROR: dudt[3] doesn't match expected PSF value!")
end