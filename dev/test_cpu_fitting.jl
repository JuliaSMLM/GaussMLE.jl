#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using GaussMLE
using Statistics

println("CPU Fitting Test")
println("=" ^ 40)

# Create simple synthetic data similar to the failing test
roi_size = 7
n_rois = 100
true_x, true_y = 4.0f0, 4.0f0
true_n, true_bg = 500.0f0, 2.0f0
sigma = 1.3f0

# Generate data
data = zeros(Float32, roi_size, roi_size, n_rois)
for k in 1:n_rois
    for i in 1:roi_size
        for j in 1:roi_size
            dx = Float32(j) - true_x
            dy = Float32(i) - true_y
            gauss = true_n * exp(-(dx^2 + dy^2) / (2*sigma^2)) / (2π*sigma^2)
            data[i, j, k] = true_bg + gauss
        end
    end
end

# Add noise
data .+= 0.1f0 .* randn(Float32, size(data))

println("Test data created:")
println("  True params: x=$true_x, y=$true_y, n=$true_n, bg=$true_bg")
println("  ROI center value: $(data[4,4,1])")

# Test CPU fitting
println("\nCPU Fitting:")
θ_cpu, Σ_cpu = fitstack(data, :xynb)

# Check results
fitted_x = [p.x for p in θ_cpu]
fitted_y = [p.y for p in θ_cpu]
fitted_n = [p.n for p in θ_cpu]
fitted_bg = [p.bg for p in θ_cpu]

println("  Mean fitted params:")
println("    x: $(mean(fitted_x)) (true: $true_x)")
println("    y: $(mean(fitted_y)) (true: $true_y)")
println("    n: $(mean(fitted_n)) (true: $true_n)")
println("    bg: $(mean(fitted_bg)) (true: $true_bg)")

println("  Errors:")
println("    x: $(abs(mean(fitted_x) - true_x)) pixels")
println("    y: $(abs(mean(fitted_y) - true_y)) pixels")
println("    n: $(abs(mean(fitted_n) - true_n)) photons")
println("    bg: $(abs(mean(fitted_bg) - true_bg)) photons")

println("  Standard deviations:")
println("    x: $(std(fitted_x))")
println("    y: $(std(fitted_y))")
println("    n: $(std(fitted_n))")
println("    bg: $(std(fitted_bg))")

# Force CPU backend
println("\nForced CPU Backend:")
θ_cpu2, Σ_cpu2 = fitstack(data, :xynb, backend=:cpu)

fitted_n2 = [p.n for p in θ_cpu2]
println("  Mean intensity with CPU backend: $(mean(fitted_n2))")
println("  Difference from default: $(abs(mean(fitted_n2) - mean(fitted_n)))")

# Check CRLB estimates
crlb_n = [s.σ_n for s in Σ_cpu]
println("\nCRLB estimates:")
println("  Mean σ_n: $(mean(crlb_n))")
println("  Actual σ_n: $(std(fitted_n))")
println("  Ratio: $(std(fitted_n) / mean(crlb_n))")