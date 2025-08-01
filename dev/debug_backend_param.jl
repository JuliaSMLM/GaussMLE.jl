#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using GaussMLE

# Test what the default parameter actually is
println("Testing backend parameter:")

# Reflection to check the method signature
methods_info = methods(GaussMLE.fitstack)
println("Available methods:")
for m in methods_info
    println("  $m")
end

# Simple test with explicit parameter check
function test_backend_param(data, model; backend=:cpu)
    println("Backend parameter received: $backend")
    return backend
end

# Test data
data = rand(Float32, 7, 7, 2)

println("\nTesting default parameter:")
result = test_backend_param(data, :xynb)
println("Result: $result")

println("\nTesting explicit backend:")
result2 = test_backend_param(data, :xynb, backend=:gpu)
println("Result: $result2")

# Now test the actual function with a simple print
# Let me see what happens when we call fitstack without precompilation issues