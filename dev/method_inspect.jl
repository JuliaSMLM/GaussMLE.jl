#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using GaussMLE

# Inspect the fitstack method
println("Inspecting fitstack method signatures:")
for m in methods(fitstack)
    println("Method: $m")
    println("  File: $(m.file)")
    println("  Line: $(m.line)")
    
    # Get the default values by examining the method
    try
        signature = Base.unwrap_unionall(m.sig)
        println("  Signature: $signature")
    catch e
        println("  Could not unwrap signature: $e")
    end
    println()
end

# Test what the actual function defaults are
println("Creating test data...")
data = zeros(Float32, 7, 7, 1)
data[4, 4, 1] = 100.0f0

# Let's use @which to see which method is being called
println("Using @which to identify called method:")
@which fitstack(data, :xynb)