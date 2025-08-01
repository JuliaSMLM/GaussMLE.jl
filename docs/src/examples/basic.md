# Basic Fitting Example

This example demonstrates the most common use case: fitting Gaussian blobs with a fixed PSF width using the `GaussXyNb` model.

## Running the Example

The complete example is available in `examples/basicfit.jl`:

```bash
julia --project=. examples/basicfit.jl
```

## Step-by-Step Walkthrough

### Setup and Data Generation

```@example basic
using GaussMLE
using Statistics
using Printf

# Example parameters
n_boxes = Int(1e4)  # Number of boxes to simulate and fit
boxsz = 7          # Box size (7×7 pixels)
psf_width = 1.3    # Fixed PSF width

println("=== Basic Gaussian Fitting Example ===")
println("Simulating $n_boxes Gaussian blobs...")

# Generate synthetic data with Poisson noise
T = Float32  # Use Float32 for better performance
out, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, n_boxes, :xynb; T, poissonnoise=true)

println("Data shape: $(size(out))")
println("True PSF width: $(args.σ_PSF)")
```

### Fitting the Data

```@example basic
# Perform the fitting
@time θ_found, Σ_found = GaussMLE.GaussFit.fitstack(out, :xynb, args)

# Calculate performance metrics
fits_per_second = n_boxes / @elapsed GaussMLE.GaussFit.fitstack(out, :xynb, args)

println("Fitting completed!")
println("Processing speed: $(round(Int, fits_per_second)) fits/second")
```

### Analyzing Results

```@example basic
# Calculate statistics for each parameter
parameters = [:x, :y, :n, :bg]
param_names = ["x position", "y position", "intensity", "background"]

println("\nFitting Results Summary:")
println("=" ^ 60)
println("Parameter     | Mean (MC)  | Std (MC)   | CRLB Std   | Agreement")
println("-" ^ 60)

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

### Understanding the Results

The table above shows:

- **Mean (MC)**: Average of fitted parameter values
- **Std (MC)**: Standard deviation of fitted values (Monte Carlo)
- **CRLB Std**: Average theoretical uncertainty (Cramér-Rao Lower Bound)
- **Agreement**: Ratio of empirical to theoretical uncertainty

**Good fits should show:**
- Agreement ratios close to 1.0
- Reasonable parameter ranges
- Low number of failed fits

```@example basic
# Check for fitting failures
valid_fits = sum(θ -> θ.n > 0 && θ.bg >= 0, θ_found)
failure_rate = (n_boxes - valid_fits) / n_boxes * 100

println("\nFit Quality Assessment:")
println("Valid fits: $valid_fits / $n_boxes")
println("Failure rate: $(round(failure_rate, digits=2))%")

# Check parameter ranges
x_vals = getproperty.(θ_found, :x)
y_vals = getproperty.(θ_found, :y)
println("X position range: $(round.(extrema(x_vals), digits=2))")
println("Y position range: $(round.(extrema(y_vals), digits=2))")
```

## Practical Considerations

### Choosing Box Size

The box size should be large enough to capture the full PSF:

```julia
# Rule of thumb: box size ≥ 4 × PSF_width
recommended_size = ceil(Int, 4 * psf_width)
println("Recommended box size: $recommended_size pixels")
```

### Performance Optimization

```@example basic
# Compare different data types
println("\nPerformance comparison:")

# Float64 timing
data_f64 = Float64.(out)
args_f64 = GaussMLE.GaussModel.Args_xynb(Float64(1.3))
time_f64 = @elapsed GaussMLE.GaussFit.fitstack(data_f64, :xynb, args_f64)

# Float32 timing (already done above)
time_f32 = @elapsed GaussMLE.GaussFit.fitstack(out, :xynb, args)

println("Float64: $(round(time_f64, digits=3))s ($(round(Int, n_boxes/time_f64)) fits/s)")
println("Float32: $(round(time_f32, digits=3))s ($(round(Int, n_boxes/time_f32)) fits/s)")
println("Speedup: $(round(time_f64/time_f32, digits=1))x")
```

### Error Handling

```@example basic
# Check for common issues
n_vals = getproperty.(θ_found, :n)
bg_vals = getproperty.(θ_found, :bg)

# Negative intensities indicate fitting problems
negative_intensity = sum(n_vals .< 0)
negative_background = sum(bg_vals .< 0)

if negative_intensity > 0
    println("Warning: $negative_intensity fits with negative intensity")
end

if negative_background > 0
    println("Warning: $negative_background fits with negative background")
end

# Very high or low values may indicate problems
high_intensity = sum(n_vals .> 10000)
high_background = sum(bg_vals .> 1000)

if high_intensity > 0
    println("Note: $high_intensity fits with very high intensity (>10000)")
end
```

## Next Steps

- Try the [PSF width fitting example](@ref "PSF Width Fitting") for variable PSF scenarios
- Learn about [GPU acceleration](@ref "GPU Support") for larger datasets  
- Explore the [API reference](@ref) for advanced options