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
σ_PSF = 1.3       # PSF width in pixels
verbose = true     # Print detailed results

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)  # Create if it doesn't exist

println("=== Example: Basic Gaussian Fitting ===")
println("This example demonstrates how to use GaussMLE for basic fitting")
println("Parameters: n_boxes=$n_boxes, boxsz=$boxsz, σ_PSF=$σ_PSF, backend=$backend")
println()

# Step 1: Generate synthetic data
println("Generating synthetic data...")
T = Float32  # Data type for calculations
roi_stack, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, n_boxes, :xynb; T, poissonnoise=true)
println("Generated $(n_boxes) simulated Gaussian blobs with Poisson noise")
println()

# Step 2: Fit the data using the new unified API
println("Fitting data using GaussXyNb model...")
t = @elapsed begin
    θ_found, Σ_found = GaussMLE.fitstack(roi_stack, :xynb; 
                                         σ_PSF=σ_PSF, 
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

# Step 3: Analyze results
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
    println("========================================")
    println()
    println("Note: MC = Monte Carlo (empirical), CRLB = Cramér-Rao Lower Bound (theoretical)")
end

# Save results to file
results_file = joinpath(output_dir, "basicfit_results.txt")
open(results_file, "w") do io
    println(io, "Basic Gaussian Fitting Example Results")
    println(io, "=====================================")
    println(io, "Generated on: $(Dates.now())")
    println(io, "Number of boxes: $n_boxes")
    println(io, "Box size: $boxsz × $boxsz pixels")
    println(io, "PSF width (σ): $σ_PSF pixels")
    println(io, "Backend requested: $backend")
    println(io, "Processing speed: $(@sprintf("%.0f", fits_per_sec)) fits/second")
    println(io, "")
    println(io, "Parameter Statistics:")
    println(io, "Parameter | Mean (MC) | Std Dev (MC) | Reported Std Dev √(CRLB)")
    println(io, "x        | $(@sprintf("%.6f", μ_x_mc)) | $(@sprintf("%.6f", σ_x_mc)) | $(@sprintf("%.6f", σ_x_reported))")
    println(io, "y        | $(@sprintf("%.6f", μ_y_mc)) | $(@sprintf("%.6f", σ_y_mc)) | $(@sprintf("%.6f", σ_y_reported))")
    println(io, "n        | $(@sprintf("%.6f", μ_n_mc)) | $(@sprintf("%.6f", σ_n_mc)) | $(@sprintf("%.6f", σ_n_reported))")
    println(io, "bg       | $(@sprintf("%.6f", μ_bg_mc)) | $(@sprintf("%.6f", σ_bg_mc)) | $(@sprintf("%.6f", σ_bg_reported))")
end
println("Results saved to: $results_file")

# Example of accessing individual fit results
println("\nExample: First 5 fitted positions:")
for i in 1:min(5, length(θ_found))
    println("  Blob $i: x=$(@sprintf("%.3f", θ_found[i].x)), y=$(@sprintf("%.3f", θ_found[i].y))")
end