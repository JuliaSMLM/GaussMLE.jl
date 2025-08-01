#!/usr/bin/env julia

# Start completely fresh
using Pkg
empty!(LOAD_PATH)
Pkg.activate(".")

println("Fresh test - force reload all modules")

# Remove from memory if present
if :GaussMLE in names(Main)
    println("Removing existing GaussMLE from memory")
    # Can't actually remove from memory, but we can try to force recompilation
end

using GaussMLE

# Create simple test data
data = zeros(Float32, 7, 7, 2)
data[4, 4, :] .= 100.0f0  # Simple peak

println("Testing fresh default behavior:")
result = @time GaussMLE.fitstack(data, :xynb, verbose=true)
println("First fit result: x=$(result[1][1].x), y=$(result[1][1].y)")

println("\nTesting explicit CPU:")
result_cpu = @time GaussMLE.fitstack(data, :xynb, backend=:cpu, verbose=true)
println("CPU fit result: x=$(result_cpu[1][1].x), y=$(result_cpu[1][1].y)")

println("\nAre they the same? $(result[1][1].x == result_cpu[1][1].x)")