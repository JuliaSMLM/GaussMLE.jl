using GaussMLE
using Statistics 
using CairoMakie
using Printf


# Simualte a stack of boxes
T = Float32
boxsz = 7
nboxes = Int(1e5)
σ_PSF = 1.3
out, θ_true, = GaussMLE.GaussSim.genstack(boxsz,nboxes,:xynbs; T, poissonnoise=true)

# Fit all boxes in the stack
args = GaussMLE.GaussModel.Args_xynbs(T(1.3))
t = @elapsed begin
    θ_found, Σ_found = GaussMLE.GaussFit.fitstack(out, :xynbs, args);
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

μ_σ_PSF_mc = mean(getproperty.(θ_found, :σ_PSF))
σ_σ_PSF_mc = std(getproperty.(θ_found, :σ_PSF))
σ_σ_PSF_reported = mean(getproperty.(Σ_found, :σ_σ_PSF))

# Formatted output 
println("Results:")
println("========================================")
println("Parameter | Mean (MC) | Std Dev (MC) | Reported Std Dev √(CRLB)")
println("----------------------------------------")
println("x        | $(@sprintf("%.6f", μ_x_mc)) | $(@sprintf("%.6f", σ_x_mc)) | $(@sprintf("%.6f", σ_x_reported))")
println("y        | $(@sprintf("%.6f", μ_y_mc)) | $(@sprintf("%.6f", σ_y_mc)) | $(@sprintf("%.6f", σ_y_reported))")
println("n        | $(@sprintf("%.6f", μ_n_mc)) | $(@sprintf("%.6f", σ_n_mc)) | $(@sprintf("%.6f", σ_n_reported))")
println("bg       | $(@sprintf("%.6f", μ_bg_mc)) | $(@sprintf("%.6f", σ_bg_mc)) | $(@sprintf("%.6f", σ_bg_reported))")
println("σ_PSF    | $(@sprintf("%.6f", μ_σ_PSF_mc)) | $(@sprintf("%.6f", σ_σ_PSF_mc)) | $(@sprintf("%.6f", σ_σ_PSF_reported))")
println("Fits per second: $(@sprintf("%.2f", fits_per_sec))")
println("========================================")
