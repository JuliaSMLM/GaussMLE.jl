# PSF Width Fitting Example

This example demonstrates fitting with variable PSF width using the `GaussXyNbS` model, which includes the PSF width as a fitted parameter.

## Running the Example

The complete example is available in `examples/sigmafit.jl`:

```bash
julia --project=. examples/sigmafit.jl
```

## Step-by-Step Walkthrough

### Setup and Data Generation

```@example sigma
using GaussMLE
using Statistics
using Printf

# Example parameters
n_boxes = Int(1e4)  # Number of boxes to simulate and fit
boxsz = 7          # Box size (7×7 pixels)
σ_PSF_init = 1.3   # Initial guess for PSF width

println("=== PSF Width Fitting Example ===")
println("Simulating $n_boxes Gaussian blobs with variable PSF width...")

# Generate synthetic data with variable PSF width
T = Float32
out, θ_true, = GaussMLE.GaussSim.genstack(boxsz, n_boxes, :xynbs; T, poissonnoise=true)

println("Data shape: $(size(out))")
println("Using initial PSF width guess: $σ_PSF_init pixels")
```

### Model Setup

The key difference from basic fitting is using the `GaussXyNbS` model:

```@example sigma
# Create arguments for variable PSF model
args = GaussMLE.GaussModel.Args_xynbs(T(σ_PSF_init))

println("Model: GaussXyNbS (5 parameters)")
println("Parameters: x, y, intensity, background, PSF_width")
```

### Fitting the Data

```@example sigma
# Perform the fitting
println("\nFitting data...")
@time θ_found, Σ_found = GaussMLE.GaussFit.fitstack(out, :xynbs, args)

# Calculate performance metrics
fits_per_second = n_boxes / @elapsed GaussMLE.GaussFit.fitstack(out, :xynbs, args)

println("Fitting completed!")
println("Processing speed: $(round(Int, fits_per_second)) fits/second")
```

### Analyzing Results

```@example sigma
# Calculate statistics for all parameters (including PSF width)
parameters = [:x, :y, :n, :bg, :σ_PSF]
param_names = ["x position", "y position", "intensity", "background", "PSF width"]

println("\nFitting Results Summary:")
println("=" ^ 70)
println("Parameter     | Mean (MC)  | Std (MC)   | CRLB Std   | Agreement")
println("-" ^ 70)

for (param, name) in zip(parameters, param_names)
    # Get fitted values and uncertainties
    fitted_vals = getproperty.(θ_found, param)
    uncertainties = getproperty.(Σ_found, Symbol("σ_", param))
    
    # Calculate statistics
    mean_fitted = mean(fitted_vals)
    std_fitted = std(fitted_vals)
    mean_crlb = mean(uncertainties)
    
    # Agreement ratio (should be ~1.0 for good fits)
    agreement = std_fitted / mean_crlb
    
    @printf("%-12s  | %8.4f   | %8.4f   | %8.4f   | %6.3f\n", 
            name, mean_fitted, std_fitted, mean_crlb, agreement)
end
```

### PSF Width Analysis

The PSF width parameter provides additional insights:

```@example sigma
# Analyze PSF width distribution
psf_widths = getproperty.(θ_found, :σ_PSF)
psf_uncertainties = getproperty.(Σ_found, :σ_σ_PSF)

println("\nPSF Width Analysis:")
println("=" ^ 40)
println("Mean PSF width: $(round(mean(psf_widths), digits=3)) ± $(round(std(psf_widths), digits=3)) pixels")
println("Median PSF width: $(round(median(psf_widths), digits=3)) pixels")
println("PSF width range: $(round.(extrema(psf_widths), digits=3)) pixels")
println("Mean uncertainty: $(round(mean(psf_uncertainties), digits=3)) pixels")

# Check for reasonable PSF widths
reasonable_psf = sum(0.5 .< psf_widths .< 3.0)
println("Reasonable PSF widths (0.5-3.0 px): $reasonable_psf / $n_boxes")
```

### Comparison with Fixed PSF Model

To understand the benefits of variable PSF fitting:

```@example sigma
# Also fit with fixed PSF model for comparison
args_fixed = GaussMLE.GaussModel.Args_xynb(T(σ_PSF_init))
θ_fixed, Σ_fixed = GaussMLE.GaussFit.fitstack(out, :xynb, args_fixed)

# Compare position estimates
x_variable = getproperty.(θ_found, :x)
y_variable = getproperty.(θ_found, :y)
x_fixed = getproperty.(θ_fixed, :x)
y_fixed = getproperty.(θ_fixed, :y)

# Calculate RMS differences
x_diff_rms = sqrt(mean((x_variable - x_fixed).^2))
y_diff_rms = sqrt(mean((y_variable - y_fixed).^2))

println("\nComparison with Fixed PSF Model:")
println("=" ^ 40)
println("X position RMS difference: $(round(x_diff_rms, digits=4)) pixels")
println("Y position RMS difference: $(round(y_diff_rms, digits=4)) pixels")

# Compare uncertainties
σx_variable = mean(getproperty.(Σ_found, :σ_x))
σy_variable = mean(getproperty.(Σ_found, :σ_y))
σx_fixed = mean(getproperty.(Σ_fixed, :σ_x))
σy_fixed = mean(getproperty.(Σ_fixed, :σ_y))

println("Mean X uncertainty - Variable PSF: $(round(σx_variable, digits=4)) pixels")
println("Mean X uncertainty - Fixed PSF:    $(round(σx_fixed, digits=4)) pixels")
println("Uncertainty ratio: $(round(σx_variable/σx_fixed, digits=2))")
```

## When to Use Variable PSF Fitting

### Advantages of GaussXyNbS

```@example sigma
# Demonstrate PSF variation detection
psf_std = std(psf_widths)
psf_cv = psf_std / mean(psf_widths)  # Coefficient of variation

println("\nPSF Variation Assessment:")
println("=" ^ 30)
println("PSF standard deviation: $(round(psf_std, digits=3)) pixels")
println("PSF coefficient of variation: $(round(psf_cv * 100, digits=1))%")

if psf_cv > 0.1
    println("→ Significant PSF variation detected - Variable PSF model recommended")
elseif psf_cv > 0.05
    println("→ Moderate PSF variation - Consider experimental conditions")
else
    println("→ Low PSF variation - Fixed PSF model may be sufficient")
end
```

### Performance Considerations

```@example sigma
# Compare fitting times
time_variable = @elapsed GaussMLE.GaussFit.fitstack(out, :xynbs, args)
time_fixed = @elapsed GaussMLE.GaussFit.fitstack(out, :xynb, args_fixed)

println("\nPerformance Comparison:")
println("=" ^ 25)
println("Variable PSF (5 params): $(round(time_variable, digits=3))s")
println("Fixed PSF (4 params):    $(round(time_fixed, digits=3))s")
println("Speed penalty: $(round(time_variable/time_fixed, digits=1))x")
```

## Practical Applications

### Astigmatic PSF

For astigmatic PSFs (used in 3D SMLM), the fitted PSF width relates to z-position:

```julia
# In real applications, PSF width varies with z-position
# This relationship can be calibrated and used for 3D localization
z_calibration_curve = fit_z_calibration(psf_widths, known_z_positions)
```

### Quality Control

```@example sigma
# Use PSF width for quality control
median_psf = median(psf_widths)
psf_outliers = sum(abs.(psf_widths .- median_psf) .> 0.5)

println("\nQuality Control:")
println("=" ^ 20)
println("PSF outliers (>0.5px from median): $psf_outliers / $n_boxes")

# Flag potentially problematic fits
problematic = sum((psf_widths .< 0.3) .| (psf_widths .> 4.0))
println("Extreme PSF widths (<0.3 or >4.0): $problematic / $n_boxes")
```

### Uncertainty Relationships

```@example sigma
# Examine relationship between PSF width and position uncertainty
using Statistics

# Bin PSF widths and calculate mean uncertainties
psf_bins = 0.8:0.2:2.0
σx_by_psf = Float64[]

for i in 1:(length(psf_bins)-1)
    mask = (psf_bins[i] .<= psf_widths .< psf_bins[i+1])
    if sum(mask) > 10  # Require at least 10 points
        push!(σx_by_psf, mean(getproperty.(Σ_found[mask], :σ_x)))
    else
        push!(σx_by_psf, NaN)
    end
end

println("\nUncertainty vs PSF Width:")
println("=" ^ 30)
for (i, (psf_bin, σx)) in enumerate(zip(psf_bins[1:end-1], σx_by_psf))
    if !isnan(σx)
        println("PSF $(psf_bin)-$(psf_bins[i+1]): σx = $(round(σx, digits=4)) px")
    end
end
```

## Troubleshooting

### Common Issues

1. **PSF width convergence problems**: Very small or very large fitted PSF widths
2. **Increased uncertainty**: Variable PSF fitting has higher parameter uncertainty
3. **Slower convergence**: More parameters mean more iterations needed

### Diagnostic Checks

```@example sigma
# Check for convergence issues
extreme_psf = sum((psf_widths .< 0.1) .| (psf_widths .> 10.0))
negative_intensity = sum(getproperty.(θ_found, :n) .< 0)

println("\nDiagnostic Summary:")
println("=" ^ 20)
println("Extreme PSF widths: $extreme_psf")
println("Negative intensities: $negative_intensity")

if extreme_psf > n_boxes * 0.01
    println("Warning: High rate of extreme PSF widths - check data quality")
end

if negative_intensity > n_boxes * 0.01
    println("Warning: High rate of negative intensities - check fitting stability")
end
```

## Next Steps

- Learn about [GPU acceleration](@ref "GPU Support") for large variable PSF datasets
- Explore the [API reference](@ref) for advanced fitting options
- Check the [models guide](@ref Models) for choosing between fixed and variable PSF models