using Pkg
Pkg.activate("dev")

using GaussMLE
using Statistics
using Printf

println("Testing GPU CRLB Fix")
println("===================")

# Generate test data
T = Float32
boxsz = 7
n_rois = 1000
modeltype = :xynb

println("Generating test data...")
roi_stack, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, n_rois, modeltype; T=T, poissonnoise=true)
println("Generated $n_rois ROIs of size $(boxsz)×$(boxsz)")

# Test CPU vs GPU fitting
println("\nTesting CPU vs GPU CRLB calculations...")

# CPU fitting (reference)
println("CPU fitting...")
t_cpu = @elapsed θ_cpu, Σ_cpu = GaussMLE.fitstack(roi_stack, modeltype; σ_PSF=args.σ_PSF, backend=:cpu, verbose=false)
println("CPU time: $(t_cpu * 1000) ms")

# GPU fitting (should now have proper CRLB)
println("GPU fitting...")
try
    t_gpu = @elapsed θ_gpu, Σ_gpu = GaussMLE.fitstack(roi_stack, modeltype; σ_PSF=args.σ_PSF, backend=:gpu, verbose=false)
    println("GPU time: $(t_gpu * 1000) ms")
    println("Speedup: $(t_cpu/t_gpu)x")
    
    # Compare CRLB values
    println("\nComparing CRLB values (first 10 ROIs):")
    println("Parameter | CPU σ    | GPU σ    | Ratio   | Status")
    println("-" ^ 55)
    
    all_good = true
    for i in 1:min(10, n_rois)
        cpu_x = Σ_cpu[i].σ_x
        gpu_x = Σ_gpu[i].σ_x
        cpu_y = Σ_cpu[i].σ_y  
        gpu_y = Σ_gpu[i].σ_y
        cpu_n = Σ_cpu[i].σ_n
        gpu_n = Σ_gpu[i].σ_n
        cpu_bg = Σ_cpu[i].σ_bg
        gpu_bg = Σ_gpu[i].σ_bg
        
        ratio_x = gpu_x / cpu_x
        ratio_y = gpu_y / cpu_y
        ratio_n = gpu_n / cpu_n
        ratio_bg = gpu_bg / cpu_bg
        
        # Check if ratios are reasonable (within 2x)
        x_ok = 0.5 < ratio_x < 2.0
        y_ok = 0.5 < ratio_y < 2.0  
        n_ok = 0.5 < ratio_n < 2.0
        bg_ok = 0.5 < ratio_bg < 2.0
        
        status = (x_ok && y_ok && n_ok && bg_ok) ? "✓" : "✗"
        if !x_ok || !y_ok || !n_ok || !bg_ok
            all_good = false
        end
        
        println("ROI $i x   | $(@sprintf("%.4f", cpu_x)) | $(@sprintf("%.4f", gpu_x)) | $(@sprintf("%.2f", ratio_x)) | $(x_ok ? "✓" : "✗")")
        println("      y   | $(@sprintf("%.4f", cpu_y)) | $(@sprintf("%.4f", gpu_y)) | $(@sprintf("%.2f", ratio_y)) | $(y_ok ? "✓" : "✗")")
        println("      n   | $(@sprintf("%.4f", cpu_n)) | $(@sprintf("%.4f", gpu_n)) | $(@sprintf("%.2f", ratio_n)) | $(n_ok ? "✓" : "✗")")
        println("      bg  | $(@sprintf("%.4f", cpu_bg)) | $(@sprintf("%.4f", gpu_bg)) | $(@sprintf("%.2f", ratio_bg)) | $(bg_ok ? "✓" : "✗")")
        println()
    end
    
    # Overall statistics
    println("Overall CRLB Statistics (all $n_rois ROIs):")
    println("=" ^ 45)
    
    all_cpu_x = [σ.σ_x for σ in Σ_cpu]
    all_gpu_x = [σ.σ_x for σ in Σ_gpu]
    all_cpu_y = [σ.σ_y for σ in Σ_cpu]
    all_gpu_y = [σ.σ_y for σ in Σ_gpu]
    all_cpu_n = [σ.σ_n for σ in Σ_cpu]
    all_gpu_n = [σ.σ_n for σ in Σ_gpu]
    all_cpu_bg = [σ.σ_bg for σ in Σ_cpu]
    all_gpu_bg = [σ.σ_bg for σ in Σ_gpu]
    
    # Check if we still have placeholder values
    placeholder_x = all(abs.(all_gpu_x .- 0.1) .< 1e-6)
    placeholder_y = all(abs.(all_gpu_y .- 0.1) .< 1e-6)  
    placeholder_n = all(abs.(all_gpu_n .- 10.0) .< 1e-6)
    placeholder_bg = all(abs.(all_gpu_bg .- 0.5) .< 1e-6)
    
    if placeholder_x || placeholder_y || placeholder_n || placeholder_bg
        println("⚠️  WARNING: GPU CRLB values appear to be placeholders!")
        println("   σ_x all ≈ 0.1: $placeholder_x")
        println("   σ_y all ≈ 0.1: $placeholder_y")
        println("   σ_n all ≈ 10:  $placeholder_n")
        println("   σ_bg all ≈ 0.5: $placeholder_bg")
        println("   The CRLB fix may not be working properly.")
    else
        println("✓ GPU CRLB values are NOT placeholders - fix is working!")
        
        println("\nParameter | CPU Mean | GPU Mean | Mean Ratio")
        println("-" ^ 45)
        println("σ_x      | $(@sprintf("%.4f", mean(all_cpu_x))) | $(@sprintf("%.4f", mean(all_gpu_x))) | $(@sprintf("%.2f", mean(all_gpu_x)/mean(all_cpu_x)))")
        println("σ_y      | $(@sprintf("%.4f", mean(all_cpu_y))) | $(@sprintf("%.4f", mean(all_gpu_y))) | $(@sprintf("%.2f", mean(all_gpu_y)/mean(all_cpu_y)))")
        println("σ_n      | $(@sprintf("%.4f", mean(all_cpu_n))) | $(@sprintf("%.4f", mean(all_gpu_n))) | $(@sprintf("%.2f", mean(all_gpu_n)/mean(all_cpu_n)))")
        println("σ_bg     | $(@sprintf("%.4f", mean(all_cpu_bg))) | $(@sprintf("%.4f", mean(all_gpu_bg))) | $(@sprintf("%.2f", mean(all_gpu_bg)/mean(all_cpu_bg)))")
    end
    
    println("\n" * "=" ^ 50)
    if all_good && !placeholder_x && !placeholder_y && !placeholder_n && !placeholder_bg
        println("🎉 SUCCESS: GPU CRLB calculations appear to be working correctly!")
    else
        println("❌ ISSUES DETECTED: GPU CRLB calculations may need further debugging.")
    end
    
catch e
    println("❌ GPU fitting failed with error:")
    println(e)
    println("\nThis could mean:")
    println("1. CUDA is not available")
    println("2. GPU kernel compilation failed")
    println("3. Memory issues")
    println("4. Errors in the new CRLB implementation")
end

println("\nTest completed.")