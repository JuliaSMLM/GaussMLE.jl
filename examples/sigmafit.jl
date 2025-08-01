using Pkg
Pkg.activate("examples")

using GaussMLE
using Statistics 
using Printf
using Dates

# Example parameters (users can adjust these)
backend = :auto    # Backend selection (:auto, :cpu, :gpu)
n_boxes = Int(1e4)  # Number of boxes to simulate and fit
boxsz = 7          # Box size
σ_PSF_init = 1.3   # Initial PSF width for fitting
verbose = true     # Print detailed results

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)  # Create if it doesn't exist

println("=== Example: Gaussian Fitting with PSF Width ===")
println("This example demonstrates fitting with variable PSF width (GaussXyNbS model)")
println("Parameters: n_boxes=$n_boxes, boxsz=$boxsz, σ_PSF_init=$σ_PSF_init, backend=$backend")

println()

# Step 1: Generate synthetic data with variable PSF width
println("Generating synthetic data with variable PSF width...")
T = Float32
roi_stack, θ_true, = GaussMLE.GaussSim.genstack(boxsz, n_boxes, :xynbs; T, poissonnoise=true)
println("Generated $(n_boxes) simulated Gaussian blobs with varying PSF widths")
println()

# Step 2: Prepare for fitting
println("Using initial PSF width σ=$σ_PSF_init for fitting")

# Step 3: Fit the data using the new unified API
println("Fitting data using GaussXyNbS model (includes PSF width as free parameter)...")
t = @elapsed begin
    θ_found, Σ_found = GaussMLE.fitstack(roi_stack, :xynbs; 
                                         σ_PSF=σ_PSF_init,
                                         backend=backend,
                                         verbose=verbose)
end
fits_per_sec = n_boxes / t

# Determine which backend was actually used
backend_used = backend == :auto ? "AUTO" : uppercase(string(backend))

# Report performance prominently
println()
println("═" ^ 50)
println("PERFORMANCE: $(@sprintf("%.0f", fits_per_sec)) fits/second ($backend_used)")
println("═" ^ 50)
println()
println("Fitting complete!")

# Step 4: Analyze results
println("Analyzing fitting results...")
# Calculate statistics for each parameter
μ_x_mc = mean(getproperty.(θ_found, :x))
σ_x_mc = std(getproperty.(θ_found, :x))
σ_x_reported = mean(getproperty.(Σ_found, :σ_x))

μ_y_mc = mean(getproperty.(θ_found, :y))
σ_y_mc = std(getproperty.(θ_found, :y))
σ_y_reported = mean(getproperty.(Σ_found, :σ_y))

μ_n_mc = mean(getproperty.(θ_found, :n))
σ_n_mc = std(getproperty.(θ_found, :n))
σ_n_reported = mean(getproperty.(Σ_found, :σ_n))

μ_bg_mc = mean(getproperty.(θ_found, :bg))
σ_bg_mc = std(getproperty.(θ_found, :bg))
σ_bg_reported = mean(getproperty.(Σ_found, :σ_bg))

μ_σ_PSF_mc = mean(getproperty.(θ_found, :σ_PSF))
σ_σ_PSF_mc = std(getproperty.(θ_found, :σ_PSF))
σ_σ_PSF_reported = mean(getproperty.(Σ_found, :σ_σ_PSF))

# Display results
if verbose
    println("Fitting Results Summary:")
    println("========================================")
    println("Parameter | Mean (MC) | Std Dev (MC) | Reported Std Dev √(CRLB)")
    println("----------------------------------------")
    println("x        | $(@sprintf("%.6f", μ_x_mc)) | $(@sprintf("%.6f", σ_x_mc)) | $(@sprintf("%.6f", σ_x_reported))")
    println("y        | $(@sprintf("%.6f", μ_y_mc)) | $(@sprintf("%.6f", σ_y_mc)) | $(@sprintf("%.6f", σ_y_reported))")
    println("n        | $(@sprintf("%.6f", μ_n_mc)) | $(@sprintf("%.6f", σ_n_mc)) | $(@sprintf("%.6f", σ_n_reported))")
    println("bg       | $(@sprintf("%.6f", μ_bg_mc)) | $(@sprintf("%.6f", σ_bg_mc)) | $(@sprintf("%.6f", σ_bg_reported))")
    println("σ_PSF    | $(@sprintf("%.6f", μ_σ_PSF_mc)) | $(@sprintf("%.6f", σ_σ_PSF_mc)) | $(@sprintf("%.6f", σ_σ_PSF_reported))")
    println("========================================")
    println()
    println("Note: σ_PSF is now a fitted parameter, not fixed")
end

# Save results to file
results_file = joinpath(output_dir, "sigmafit_results.txt")
open(results_file, "w") do io
    println(io, "Gaussian Fitting with PSF Width Example Results")
    println(io, "==============================================")
    println(io, "Generated on: $(Dates.now())")
    println(io, "Number of boxes: $n_boxes")
    println(io, "Box size: $boxsz × $boxsz pixels")
    println(io, "Initial PSF width: $σ_PSF_init")
    println(io, "Backend requested: $backend")
    println(io, "Processing speed: $(@sprintf("%.0f", fits_per_sec)) fits/second")
    println(io, "")
    println(io, "Parameter Statistics:")
    println(io, "Parameter | Mean (MC) | Std Dev (MC) | Reported Std Dev √(CRLB)")
    println(io, "x        | $(@sprintf("%.6f", μ_x_mc)) | $(@sprintf("%.6f", σ_x_mc)) | $(@sprintf("%.6f", σ_x_reported))")
    println(io, "y        | $(@sprintf("%.6f", μ_y_mc)) | $(@sprintf("%.6f", σ_y_mc)) | $(@sprintf("%.6f", σ_y_reported))")
    println(io, "n        | $(@sprintf("%.6f", μ_n_mc)) | $(@sprintf("%.6f", σ_n_mc)) | $(@sprintf("%.6f", σ_n_reported))")
    println(io, "bg       | $(@sprintf("%.6f", μ_bg_mc)) | $(@sprintf("%.6f", σ_bg_mc)) | $(@sprintf("%.6f", σ_bg_reported))")
    println(io, "σ_PSF    | $(@sprintf("%.6f", μ_σ_PSF_mc)) | $(@sprintf("%.6f", σ_σ_PSF_mc)) | $(@sprintf("%.6f", σ_σ_PSF_reported))")
end
println("Results saved to: $results_file")

# Example of accessing individual fit results including PSF width
println("\nExample: First 5 fitted results with PSF width:")
for i in 1:min(5, length(θ_found))
    println("  Blob $i: x=$(@sprintf("%.3f", θ_found[i].x)), y=$(@sprintf("%.3f", θ_found[i].y)), σ_PSF=$(@sprintf("%.3f", θ_found[i].σ_PSF))")
end

# Demonstrate difference between models
println("\nKey difference from basic fitting:")
println("- GaussXyNb:  4 parameters (x, y, intensity, background)")
println("- GaussXyNbS: 5 parameters (x, y, intensity, background, PSF_width)")
println("The extra parameter allows fitting when PSF width varies between spots")