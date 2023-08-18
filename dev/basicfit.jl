using GaussMLE
using Statistics 
using CairoMakie
using Printf


# Simualte a stack of boxes
boxsz = 7
nboxes = 1000
out, θ_true, = GaussMLE.GaussSim.genstack(boxsz,nboxes,:xynb; T= Float64, poissonnoise=true)

# Fit a single box
θ = GaussMLE.GaussModel.θ_xynb()
Σ = GaussMLE.GaussModel.Σ_xynb()
args = GaussMLE.GaussModel.Args_xynb(1.3)
GaussMLE.GaussFit.fitbox!(θ, Σ, out[:,:,1], args)
display(θ_true[1])
display(θ)

# Fit all boxes in the stack
θ_found, Σ_found = GaussMLE.GaussFit.fitstack(out, :xynb, args)

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
println("Parameter | Mean (MC) | Std Dev (MC) | Reported Std Dev")
println("----------------------------------------")
println("x        | $(@sprintf("%.6f", μ_x_mc)) | $(@sprintf("%.6f", σ_x_mc)) | $(@sprintf("%.6f", σ_x_reported))")
println("y        | $(@sprintf("%.6f", μ_y_mc)) | $(@sprintf("%.6f", σ_y_mc)) | $(@sprintf("%.6f", σ_y_reported))")
println("n        | $(@sprintf("%.6f", μ_n_mc)) | $(@sprintf("%.6f", σ_n_mc)) | $(@sprintf("%.6f", σ_n_reported))")
println("bg       | $(@sprintf("%.6f", μ_bg_mc)) | $(@sprintf("%.6f", σ_bg_mc)) | $(@sprintf("%.6f", σ_bg_reported))")
println("========================================")

# Look for outliers visually 
hist(getproperty.(θ_found, :x))
