using GaussMLE
using Statistics 
using Printf

# Simulate a stack of boxes with Poisson noise
T = Float32 # Data type
boxsz = 7 # Box size
nboxes = Int(1e5) # Number of boxes
out, θ_true, args= GaussMLE.GaussSim.genstack(boxsz,nboxes,:xynb; T, poissonnoise=true)

# Fit all boxes in the stack
t = @elapsed begin
    θ_found, Σ_found = GaussMLE.GaussFit.fitstack(out, :xynb, args);
end
fits_per_sec = nboxes / t

# Compare the true and found parameters
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


# Formatted output 
println("Results:")
println("========================================")
println("Parameter | Mean (MC) | Std Dev (MC) | Reported Std Dev √(CRLB)")
println("----------------------------------------")
println("x        | $(@sprintf("%.6f", μ_x_mc)) | $(@sprintf("%.6f", σ_x_mc)) | $(@sprintf("%.6f", σ_x_reported))")
println("y        | $(@sprintf("%.6f", μ_y_mc)) | $(@sprintf("%.6f", σ_y_mc)) | $(@sprintf("%.6f", σ_y_reported))")
println("n        | $(@sprintf("%.6f", μ_n_mc)) | $(@sprintf("%.6f", σ_n_mc)) | $(@sprintf("%.6f", σ_n_reported))")
println("bg       | $(@sprintf("%.6f", μ_bg_mc)) | $(@sprintf("%.6f", σ_bg_mc)) | $(@sprintf("%.6f", σ_bg_reported))")
println("Fits per second: $(@sprintf("%.2f", fits_per_sec))")
println("========================================")
