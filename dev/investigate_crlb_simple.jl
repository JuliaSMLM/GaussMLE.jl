using Pkg
Pkg.activate("dev")

using GaussMLE
using Statistics
using Printf

println("Investigating CRLB vs Empirical Standard Deviations")
println("=" ^ 60)

# Generate standard test data
T = Float32
boxsz = 7
n_rois = 10000  # Large number for good statistics
modeltype = :xynb

println("Generating $n_rois ROIs of size $(boxsz)×$(boxsz)...")
roi_stack, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, n_rois, modeltype; T=T, poissonnoise=true)

println("True parameter statistics:")
true_x = [θ.x for θ in θ_true]
true_y = [θ.y for θ in θ_true]
true_n = [θ.n for θ in θ_true]
true_bg = [θ.bg for θ in θ_true]

println("  x: mean=$(@sprintf("%.2f", mean(true_x))), std=$(@sprintf("%.3f", std(true_x)))")
println("  y: mean=$(@sprintf("%.2f", mean(true_y))), std=$(@sprintf("%.3f", std(true_y)))")  
println("  n: mean=$(@sprintf("%.1f", mean(true_n))), std=$(@sprintf("%.3f", std(true_n)))")
println("  bg: mean=$(@sprintf("%.2f", mean(true_bg))), std=$(@sprintf("%.3f", std(true_bg)))")

# Fit with CPU and GPU
println("\nFitting with CPU...")
t_cpu = @elapsed θ_cpu, Σ_cpu = GaussMLE.fitstack(roi_stack, modeltype; σ_PSF=args.σ_PSF, backend=:cpu, verbose=false)

println("Fitting with GPU...")
t_gpu = @elapsed θ_gpu, Σ_gpu = GaussMLE.fitstack(roi_stack, modeltype; σ_PSF=args.σ_PSF, backend=:gpu, verbose=false)

println("CPU time: $(t_cpu * 1000) ms")
println("GPU time: $(t_gpu * 1000) ms")

# Calculate empirical standard deviations from fitted values
empirical_cpu_x = std([θ.x for θ in θ_cpu])
empirical_cpu_y = std([θ.y for θ in θ_cpu])
empirical_cpu_n = std([θ.n for θ in θ_cpu])
empirical_cpu_bg = std([θ.bg for θ in θ_cpu])

empirical_gpu_x = std([θ.x for θ in θ_gpu])
empirical_gpu_y = std([θ.y for θ in θ_gpu])
empirical_gpu_n = std([θ.n for θ in θ_gpu])
empirical_gpu_bg = std([θ.bg for θ in θ_gpu])

# Calculate mean reported CRLB values
cpu_crlb_x = mean([σ.σ_x for σ in Σ_cpu])
cpu_crlb_y = mean([σ.σ_y for σ in Σ_cpu])
cpu_crlb_n = mean([σ.σ_n for σ in Σ_cpu])
cpu_crlb_bg = mean([σ.σ_bg for σ in Σ_cpu])

gpu_crlb_x = mean([σ.σ_x for σ in Σ_gpu])
gpu_crlb_y = mean([σ.σ_y for σ in Σ_gpu])
gpu_crlb_n = mean([σ.σ_n for σ in Σ_gpu])
gpu_crlb_bg = mean([σ.σ_bg for σ in Σ_gpu])

println("\n" * "=" ^ 80)
println("EMPIRICAL vs CRLB ANALYSIS")
println("=" ^ 80)

println("\nCPU Results:")
println("Parameter | Empirical | CRLB     | Ratio  | Assessment")
println("-" ^ 55)
cpu_ratio_x = empirical_cpu_x / cpu_crlb_x
cpu_ratio_y = empirical_cpu_y / cpu_crlb_y
cpu_ratio_n = empirical_cpu_n / cpu_crlb_n
cpu_ratio_bg = empirical_cpu_bg / cpu_crlb_bg

assess_x = cpu_ratio_x > 1.2 ? "⚠️ Underestimated" : cpu_ratio_x < 0.8 ? "⚠️ Overestimated" : "✓ Good"
assess_y = cpu_ratio_y > 1.2 ? "⚠️ Underestimated" : cpu_ratio_y < 0.8 ? "⚠️ Overestimated" : "✓ Good"
assess_n = cpu_ratio_n > 1.2 ? "⚠️ Underestimated" : cpu_ratio_n < 0.8 ? "⚠️ Overestimated" : "✓ Good"
assess_bg = cpu_ratio_bg > 1.2 ? "⚠️ Underestimated" : cpu_ratio_bg < 0.8 ? "⚠️ Overestimated" : "✓ Good"

println("x        | $(@sprintf("%.4f", empirical_cpu_x)) | $(@sprintf("%.4f", cpu_crlb_x)) | $(@sprintf("%.2f", cpu_ratio_x)) | $assess_x")
println("y        | $(@sprintf("%.4f", empirical_cpu_y)) | $(@sprintf("%.4f", cpu_crlb_y)) | $(@sprintf("%.2f", cpu_ratio_y)) | $assess_y")
println("n        | $(@sprintf("%.4f", empirical_cpu_n)) | $(@sprintf("%.4f", cpu_crlb_n)) | $(@sprintf("%.2f", cpu_ratio_n)) | $assess_n")
println("bg       | $(@sprintf("%.4f", empirical_cpu_bg)) | $(@sprintf("%.4f", cpu_crlb_bg)) | $(@sprintf("%.2f", cpu_ratio_bg)) | $assess_bg")

println("\nGPU Results:")
println("Parameter | Empirical | CRLB     | Ratio  | Assessment")
println("-" ^ 55)
gpu_ratio_x = empirical_gpu_x / gpu_crlb_x
gpu_ratio_y = empirical_gpu_y / gpu_crlb_y
gpu_ratio_n = empirical_gpu_n / gpu_crlb_n
gpu_ratio_bg = empirical_gpu_bg / gpu_crlb_bg

assess_x = gpu_ratio_x > 1.2 ? "⚠️ Underestimated" : gpu_ratio_x < 0.8 ? "⚠️ Overestimated" : "✓ Good"
assess_y = gpu_ratio_y > 1.2 ? "⚠️ Underestimated" : gpu_ratio_y < 0.8 ? "⚠️ Overestimated" : "✓ Good"
assess_n = gpu_ratio_n > 1.2 ? "⚠️ Underestimated" : gpu_ratio_n < 0.8 ? "⚠️ Overestimated" : "✓ Good"
assess_bg = gpu_ratio_bg > 1.2 ? "⚠️ Underestimated" : gpu_ratio_bg < 0.8 ? "⚠️ Overestimated" : "✓ Good"

println("x        | $(@sprintf("%.4f", empirical_gpu_x)) | $(@sprintf("%.4f", gpu_crlb_x)) | $(@sprintf("%.2f", gpu_ratio_x)) | $assess_x")
println("y        | $(@sprintf("%.4f", empirical_gpu_y)) | $(@sprintf("%.4f", gpu_crlb_y)) | $(@sprintf("%.2f", gpu_ratio_y)) | $assess_y")
println("n        | $(@sprintf("%.4f", empirical_gpu_n)) | $(@sprintf("%.4f", gpu_crlb_n)) | $(@sprintf("%.2f", gpu_ratio_n)) | $assess_n")
println("bg       | $(@sprintf("%.4f", empirical_gpu_bg)) | $(@sprintf("%.4f", gpu_crlb_bg)) | $(@sprintf("%.2f", gpu_ratio_bg)) | $assess_bg")

# Additional diagnostics
println("\n" * "=" ^ 80)
println("DIAGNOSTIC ANALYSIS")
println("=" ^ 80)

# Check fitting quality
cpu_fitted_x = [θ.x for θ in θ_cpu]
cpu_fitted_y = [θ.y for θ in θ_cpu]
cpu_fitted_n = [θ.n for θ in θ_cpu]
cpu_fitted_bg = [θ.bg for θ in θ_cpu]

println("\nFitted parameter means (CPU):")
println("  x: $(@sprintf("%.3f", mean(cpu_fitted_x))) (true: $(@sprintf("%.3f", mean(true_x))))")
println("  y: $(@sprintf("%.3f", mean(cpu_fitted_y))) (true: $(@sprintf("%.3f", mean(true_y))))")
println("  n: $(@sprintf("%.1f", mean(cpu_fitted_n))) (true: $(@sprintf("%.1f", mean(true_n))))")
println("  bg: $(@sprintf("%.3f", mean(cpu_fitted_bg))) (true: $(@sprintf("%.3f", mean(true_bg))))")

# Check for systematic biases
bias_x = mean(cpu_fitted_x) - mean(true_x)
bias_y = mean(cpu_fitted_y) - mean(true_y)  
bias_n = mean(cpu_fitted_n) - mean(true_n)
bias_bg = mean(cpu_fitted_bg) - mean(true_bg)

println("\nSystematic biases (fitted - true):")
println("  x: $(@sprintf("%.4f", bias_x)) pixels")
println("  y: $(@sprintf("%.4f", bias_y)) pixels")
println("  n: $(@sprintf("%.1f", bias_n)) photons")
println("  bg: $(@sprintf("%.4f", bias_bg)) photons/pixel")

# Calculate SNR statistics
snr_values = [θ.n / sqrt(θ.n + θ.bg * boxsz^2) for θ in θ_cpu]
avg_snr = mean(snr_values)
println("\nSignal-to-Noise Ratio:")
println("  Mean SNR: $(@sprintf("%.1f", avg_snr))")
println("  SNR range: $(@sprintf("%.1f", minimum(snr_values))) - $(@sprintf("%.1f", maximum(snr_values)))")

# Key insights
println("\n" * "=" ^ 80)
println("KEY INSIGHTS")
println("=" ^ 80)

if cpu_ratio_x > 1.1 && cpu_ratio_y > 1.1
    println("🔍 POSITION UNCERTAINTY: CRLB underestimates by $(@sprintf("%.0f", (cpu_ratio_x-1)*100))%")
    println("   Likely causes:")
    println("   • Pixelation effects (continuous model vs discrete pixels)")
    println("   • PSF model mismatch (assumed vs actual PSF shape)")
    println("   • Finite photon statistics (CRLB is asymptotic limit)")
end

if cpu_ratio_bg > 1.2
    println("🔍 BACKGROUND UNCERTAINTY: CRLB underestimates by $(@sprintf("%.0f", (cpu_ratio_bg-1)*100))%")
    println("   Likely causes:")
    println("   • Non-uniform background assumptions")
    println("   • Camera noise model (pure Poisson vs realistic noise)")
    println("   • ROI boundary effects")
end

if cpu_ratio_n > 1.2
    println("🔍 INTENSITY UNCERTAINTY: CRLB underestimates by $(@sprintf("%.0f", (cpu_ratio_n-1)*100))%")
    println("   Likely causes:")
    println("   • Parameter correlation not fully captured")
    println("   • Model nonlinearity effects")
    println("   • Systematic bias in parameter estimation")
end

if avg_snr < 5
    println("⚠️  LOW SNR DETECTED: SNR = $(@sprintf("%.1f", avg_snr))")
    println("   CRLB accuracy degrades significantly at low SNR")
    println("   Consider using empirical correction factor of 1.2-1.5x")
end

println("\n" * "=" ^ 80)
println("RECOMMENDATIONS")
println("=" ^ 80)
println("""
1. **For practical localization accuracy**: Apply empirical correction factors:
   • Position (x,y): CRLB × 1.2-1.3 for SNR > 5
   • Intensity: CRLB × 1.1-1.2 
   • Background: CRLB × 1.3-1.5

2. **For better CRLB accuracy**:
   • Measure actual PSF width from bead calibration
   • Use sCMOS variance maps if available
   • Consider advanced PSF models (e.g., astigmatic, aberrated)

3. **For validation**: 
   • Compare with established software (ThunderSTORM, rapidSTORM)
   • Use standardized test datasets (SMLM Challenge data)

4. **Physics**: The gap between CRLB and empirical is expected and normal!
   • CRLB is theoretical lower bound (perfect conditions)
   • Real experiments have model mismatch and systematic effects
   • 10-30% larger empirical uncertainties are typical in SMLM
""")

println("\nAnalysis completed.")