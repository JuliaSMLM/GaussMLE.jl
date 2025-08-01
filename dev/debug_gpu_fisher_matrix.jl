using Pkg
Pkg.activate("dev")

using GaussMLE
using Statistics
using Printf

println("Debugging GPU Fisher Matrix Calculation")
println("=" ^ 50)

# Create a single well-controlled test case
T = Float32
roi_stack, θ_true, args = GaussMLE.GaussSim.genstack(7, 1, :xynb; T=T, poissonnoise=false)

println("Test case (no noise):")
println("  True x: $(θ_true[1].x)")
println("  True y: $(θ_true[1].y)")  
println("  True n: $(θ_true[1].n)")
println("  True bg: $(θ_true[1].bg)")
println("  σ_PSF: $(args.σ_PSF)")

# Fit with both backends
θ_cpu, Σ_cpu = GaussMLE.fitstack(roi_stack, :xynb; σ_PSF=args.σ_PSF, backend=:cpu, verbose=false)
θ_gpu, Σ_gpu = GaussMLE.fitstack(roi_stack, :xynb; σ_PSF=args.σ_PSF, backend=:gpu, verbose=false)

println("\nFitted parameters:")
println("Backend | x      | y      | n      | bg")
println("--------|--------|--------|--------|--------")
println("CPU     | $(@sprintf("%.4f", θ_cpu[1].x)) | $(@sprintf("%.4f", θ_cpu[1].y)) | $(@sprintf("%.1f", θ_cpu[1].n)) | $(@sprintf("%.4f", θ_cpu[1].bg))")
println("GPU     | $(@sprintf("%.4f", θ_gpu[1].x)) | $(@sprintf("%.4f", θ_gpu[1].y)) | $(@sprintf("%.1f", θ_gpu[1].n)) | $(@sprintf("%.4f", θ_gpu[1].bg))")

println("\nCRLB uncertainties:")
println("Backend | σ_x    | σ_y    | σ_n    | σ_bg")
println("--------|--------|--------|--------|--------")
println("CPU     | $(@sprintf("%.4f", Σ_cpu[1].σ_x)) | $(@sprintf("%.4f", Σ_cpu[1].σ_y)) | $(@sprintf("%.2f", Σ_cpu[1].σ_n)) | $(@sprintf("%.4f", Σ_cpu[1].σ_bg))")
println("GPU     | $(@sprintf("%.4f", Σ_gpu[1].σ_x)) | $(@sprintf("%.4f", Σ_gpu[1].σ_y)) | $(@sprintf("%.2f", Σ_gpu[1].σ_n)) | $(@sprintf("%.4f", Σ_gpu[1].σ_bg))")

println("\nCRLB Ratios (GPU/CPU):")
ratio_x = Σ_gpu[1].σ_x / Σ_cpu[1].σ_x
ratio_y = Σ_gpu[1].σ_y / Σ_cpu[1].σ_y
ratio_n = Σ_gpu[1].σ_n / Σ_cpu[1].σ_n
ratio_bg = Σ_gpu[1].σ_bg / Σ_cpu[1].σ_bg

println("  σ_x: $(@sprintf("%.3f", ratio_x)) $(ratio_x < 0.95 ? "⚠️ GPU too small" : "✓")")
println("  σ_y: $(@sprintf("%.3f", ratio_y)) $(ratio_y < 0.95 ? "⚠️ GPU too small" : "✓")")
println("  σ_n: $(@sprintf("%.3f", ratio_n)) $(ratio_n < 0.95 ? "⚠️ GPU too small" : "✓")")
println("  σ_bg: $(@sprintf("%.3f", ratio_bg)) $(ratio_bg < 0.95 ? "⚠️ GPU too small" : "✓")")

println("\n" * "=" ^ 50)
println("DIAGNOSIS")
println("=" ^ 50)

if ratio_x < 0.9 || ratio_y < 0.9
    println("🔍 POSITION UNCERTAINTY: GPU systematically underestimates")
    println("   Likely causes in GPU kernel:")
    println("   • Wrong σ_PSF value (hardcoded 1.5 vs actual $(args.σ_PSF))")
    println("   • Coordinate system mismatch")
    println("   • Missing normalization factors")
end

if ratio_n < 0.9
    println("🔍 INTENSITY UNCERTAINTY: GPU underestimates")
    println("   Likely causes:")
    println("   • Incorrect derivative ∂μ/∂n calculation")
    println("   • Wrong intensity normalization")
end

if ratio_bg < 0.9
    println("🔍 BACKGROUND UNCERTAINTY: GPU underestimates")
    println("   Likely causes:")
    println("   • Fisher matrix element F[4,4] calculation error")
    println("   • Background gradient should be 1.0 everywhere")
end

# Check the specific issue with PSF width
println("\nPSF Width Investigation:")
println("  CPU uses fitted σ_PSF: $(args.σ_PSF)")
println("  GPU kernel hardcodes σ_PSF: 1.5")
if abs(args.σ_PSF - 1.5) > 0.1
    println("  ⚠️  PSF MISMATCH: This explains the discrepancy!")
    println("  Fisher matrix elements scale as 1/σ² and 1/σ⁴")
    println("  Expected ratio scaling: $(@sprintf("%.3f", (1.5/args.σ_PSF)^2))")
end

println("\n" * "=" ^ 50)
println("REQUIRED GPU KERNEL FIXES")
println("=" ^ 50)
println("""
1. **Fix PSF width**: Use actual fitted σ_PSF, not hardcoded 1.5
2. **Coordinate consistency**: Ensure GPU uses same indexing as CPU
3. **Derivative validation**: Verify ∂μ/∂x, ∂μ/∂y, ∂μ/∂n calculations
4. **Fisher matrix check**: Validate F[i,j] accumulation matches CPU
5. **Matrix inversion**: Ensure 4x4 analytical inverse is correct

Priority: Fix PSF width first - likely explains most of the discrepancy!
""")

println("Debug completed.")