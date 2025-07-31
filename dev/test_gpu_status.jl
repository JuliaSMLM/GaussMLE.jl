#!/usr/bin/env julia
#
# GPU Status Test for GaussMLE.jl
# Tests GPU availability, memory, and basic functionality

println("GPU Status Test for GaussMLE.jl")
println("=" ^ 50)
println()

# Test 1: CUDA Installation
print("Testing CUDA installation... ")
cuda_installed = try
    using CUDA
    println("✓ CUDA.jl is installed")
    true
catch e
    println("✗ CUDA.jl not installed")
    println("  Install with: ] add CUDA")
    false
end

if !cuda_installed
    println("\nCUDA.jl must be installed to use GPU features")
    exit(1)
end

# Test 2: CUDA Functionality
print("\nTesting CUDA functionality... ")
cuda_functional = try
    CUDA.functional()
catch e
    false
end

if cuda_functional
    println("✓ CUDA is functional")
else
    println("✗ CUDA is not functional")
    println("  This could mean:")
    println("  - No NVIDIA GPU present")
    println("  - CUDA drivers not installed")
    println("  - Driver/CUDA version mismatch")
    exit(1)
end

# Test 3: GPU Device Information
println("\nGPU Device Information:")
println("-" ^ 30)
device = CUDA.device()
println("Device ID: ", CUDA.deviceid(device))
println("Device Name: ", CUDA.name(device))
println("Compute Capability: ", CUDA.capability(device))
println("Number of SMs: ", CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
println("Warp Size: ", CUDA.attribute(device, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE))

# Test 4: Memory Information
println("\nGPU Memory Information:")
println("-" ^ 30)
mem_info = CUDA.MemoryInfo()
free_bytes = CUDA.available_memory()
total_bytes = CUDA.total_memory()
used_bytes = total_bytes - free_bytes

free_gb = free_bytes / 1024^3
total_gb = total_bytes / 1024^3
used_gb = used_bytes / 1024^3
percent_used = (used_bytes / total_bytes) * 100

println("Total Memory: $(round(total_gb, digits=2)) GB")
println("Used Memory: $(round(used_gb, digits=2)) GB ($(round(percent_used, digits=1))%)")
println("Free Memory: $(round(free_gb, digits=2)) GB")

# Memory status
if free_gb < 0.5
    println("\n⚠️  WARNING: Less than 0.5 GB free GPU memory!")
    println("   Other processes may be using the GPU")
elseif free_gb < 1.0
    println("\n⚠️  WARNING: Less than 1 GB free GPU memory")
    println("   Large fitting jobs may fail")
else
    println("\n✓ Sufficient GPU memory available")
end

# Test 5: Simple GPU Operation
println("\nTesting GPU computation...")
try
    # Create a small test array
    test_size = 1000
    cpu_array = rand(Float32, test_size)
    
    # Transfer to GPU and perform operation
    gpu_time = @elapsed begin
        gpu_array = CuArray(cpu_array)
        gpu_result = gpu_array .* 2.0f0
        result = Array(gpu_result)
    end
    
    # Verify result
    expected = cpu_array .* 2.0f0
    if isapprox(result, expected, rtol=1e-5)
        println("✓ GPU computation successful")
        println("  Test operation time: $(round(gpu_time * 1000, digits=2)) ms")
    else
        println("✗ GPU computation failed - results don't match")
    end
catch e
    println("✗ GPU computation failed with error:")
    println("  ", e)
end

# Test 6: GaussMLE GPU Backend
println("\nTesting GaussMLE GPU backend...")
try
    using GaussMLE
    using GaussMLE: select_backend, CUDABackend, backend_available, backend_name
    
    # Test backend selection
    backend = select_backend()
    println("Selected backend: ", backend_name(backend))
    
    # Test CUDA backend specifically
    cuda_backend = CUDABackend()
    if backend_available(cuda_backend)
        println("✓ CUDA backend is available")
    else
        println("✗ CUDA backend not available")
    end
    
    # Try a small fitting operation
    println("\nTesting GPU fitting with small dataset...")
    test_rois = 100
    roi_size = 7
    
    # Generate test data
    data, θ_true, args = GaussMLE.GaussSim.genstack(roi_size, test_rois, :xynb; 
                                                    T=Float32, poissonnoise=true)
    
    fit_time = @elapsed begin
        θ_gpu, Σ_gpu = GaussMLE.fitstack_gpu(data, :xynb, cuda_backend; verbose=false)
    end
    
    println("✓ GPU fitting completed")
    println("  Time for $test_rois ROIs: $(round(fit_time * 1000, digits=1)) ms")
    println("  Speed: $(round(test_rois / fit_time, digits=0)) ROIs/second")
    
catch e
    println("✗ GaussMLE GPU backend test failed:")
    println("  ", e)
end

# Summary
println("\n" * ("=" ^ 50))
println("GPU Status Summary:")
if cuda_functional && free_gb > 1.0
    println("✓ GPU is ready for use with GaussMLE")
    println("  - CUDA is functional")
    println("  - Sufficient memory available ($(round(free_gb, digits=1)) GB free)")
    println("  - GPU computation verified")
else
    println("⚠️  GPU may have issues:")
    if !cuda_functional
        println("  - CUDA is not functional")
    end
    if free_gb <= 1.0
        println("  - Low GPU memory ($(round(free_gb, digits=1)) GB free)")
    end
end

# Process monitoring tip
println("\nTip: Monitor GPU usage with:")
println("  nvidia-smi              # One-time check")
println("  watch -n 1 nvidia-smi   # Continuous monitoring")