#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using GaussMLE

println("Debug fitstack routing")
println("=" ^ 40)

# Create realistic synthetic data (Gaussian spots)
n_rois = 10
data = zeros(Float32, 7, 7, n_rois)
for k in 1:n_rois
    x_true = 2.0f0 + rand(Float32) * 3.0f0  # 2-5 range
    y_true = 2.0f0 + rand(Float32) * 3.0f0
    intensity = 500f0 + rand(Float32) * 1000f0
    bg = 5f0 + rand(Float32) * 10f0
    
    for i in 1:7
        for j in 1:7
            dx = Float32(j) - x_true
            dy = Float32(i) - y_true
            gauss = intensity * exp(-(dx^2 + dy^2) / (2*1.3f0^2)) / (2π*1.3f0^2)
            data[i, j, k] = bg + gauss
        end
    end
end
println("Generated realistic Gaussian data")

println("Testing different backends:")

# Test explicit CPU
println("\n1. Explicit CPU backend:")
@time θ_cpu, Σ_cpu = fitstack(data, :xynb, backend=:cpu, verbose=true)
println("  First CRLB σ_n: $(Σ_cpu[1].σ_n)")

# Test default (should be CPU now)
println("\n2. Default backend:")
@time θ_default, Σ_default = fitstack(data, :xynb, verbose=true)
println("  First CRLB σ_n: $(Σ_default[1].σ_n)")

# Test explicit GPU
println("\n3. Explicit GPU backend:")
@time θ_gpu, Σ_gpu = fitstack(data, :xynb, backend=:gpu, verbose=true)
println("  First CRLB σ_n: $(Σ_gpu[1].σ_n)")

# Test auto
println("\n4. Auto backend:")
@time θ_auto, Σ_auto = fitstack(data, :xynb, backend=:auto, verbose=true)
println("  First CRLB σ_n: $(Σ_auto[1].σ_n)")

# Compare results
println("\nResults comparison:")
println("  CPU vs Default match: $(θ_cpu[1].x == θ_default[1].x)")
println("  CPU vs GPU match: $(θ_cpu[1].x == θ_gpu[1].x)")
println("  Default vs Auto match: $(θ_default[1].x == θ_auto[1].x)")