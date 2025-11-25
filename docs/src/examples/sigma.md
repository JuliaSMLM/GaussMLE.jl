# PSF Width Fitting Example

This example demonstrates fitting with variable PSF width using the `GaussianXYNBS` model, which includes the PSF width as a fitted parameter.

## When to Use Variable PSF Fitting

Use `GaussianXYNBS` when:
- PSF width varies across your dataset (e.g., z-dependent defocus)
- PSF width is unknown and needs to be measured
- Quality control: filter localizations by fitted PSF width
- Detecting PSF changes during acquisition

## Basic Usage

```julia
using GaussMLE
using SMLMData
using Statistics

# Create variable-sigma model (no fixed parameters)
psf = GaussianXYNBS()

# Create fitter
fitter = GaussMLEFitter(psf_model = psf)

# Fit data
data = rand(Float32, 11, 11, 100)
smld = fit(fitter, data)

# Access fitted PSF width from emitters (Emitter2DFitSigma type)
sigmas = [e.sigma for e in smld.emitters]
sigma_uncertainties = [e.sigma_sigma for e in smld.emitters]

println("Mean PSF width: $(mean(sigmas)) microns")
println("Mean sigma uncertainty: $(mean(sigma_uncertainties)) microns")
```

## Complete Working Example

```julia
using GaussMLE
using SMLMData
using Statistics

println("=== PSF Width Fitting Example ===\n")

# Generate synthetic data
n_rois = 100
data = rand(Float32, 11, 11, n_rois)

# Create variable-sigma fitter
fitter = GaussMLEFitter(
    psf_model = GaussianXYNBS(),
    iterations = 25  # More iterations for 5-parameter fit
)

# Fit the data
println("Fitting $n_rois ROIs with variable PSF width...")
smld = fit(fitter, data)

# The emitters are Emitter2DFitSigma type with sigma field
println("\n=== Results ===")
println("Fitted: $(length(smld.emitters)) localizations")

# Extract all fields
x_positions = [e.x for e in smld.emitters]
y_positions = [e.y for e in smld.emitters]
photons = [e.photons for e in smld.emitters]
backgrounds = [e.bg for e in smld.emitters]
sigmas = [e.sigma for e in smld.emitters]

# Uncertainties
sigma_x = [e.sigma_x for e in smld.emitters]
sigma_sigma = [e.sigma_sigma for e in smld.emitters]

println("\nPosition Statistics:")
println("  Mean x: $(round(mean(x_positions), digits=3)) microns")
println("  Mean y: $(round(mean(y_positions), digits=3)) microns")
println("  Mean precision: $(round(mean(sigma_x)*1000, digits=1)) nm")

println("\nPhotometry Statistics:")
println("  Mean photons: $(round(mean(photons), digits=1))")
println("  Mean background: $(round(mean(backgrounds), digits=1))")

println("\nPSF Width Statistics:")
println("  Mean sigma: $(round(mean(sigmas), digits=4)) microns")
println("  Sigma std: $(round(std(sigmas), digits=4)) microns")
println("  Sigma range: $(round.(extrema(sigmas), digits=4)) microns")
println("  Mean sigma uncertainty: $(round(mean(sigma_sigma), digits=4)) microns")
```

## Understanding Emitter2DFitSigma

The `GaussianXYNBS` model returns `Emitter2DFitSigma` emitters with these fields:

| Field | Description | Units |
|-------|-------------|-------|
| `x`, `y` | Position | microns |
| `photons` | Total photon count | photons |
| `bg` | Background level | photons/pixel |
| `sigma` | **Fitted PSF width** | microns |
| `sigma_x`, `sigma_y` | Position uncertainty | microns |
| `sigma_photons`, `sigma_bg` | Photometry uncertainties | photons |
| `sigma_sigma` | **PSF width uncertainty** | microns |
| `pvalue` | Goodness-of-fit | 0-1 |
| `frame` | Frame number | integer |

## Quality Control with PSF Width

One key use of variable PSF fitting is quality control:

```julia
using GaussMLE
using Statistics

# Fit with variable sigma
fitter = GaussMLEFitter(psf_model = GaussianXYNBS())
smld = fit(fitter, data)

# Extract PSF widths
sigmas = [e.sigma for e in smld.emitters]

# Analyze distribution
println("PSF Width Analysis:")
println("  Mean: $(round(mean(sigmas), digits=4)) microns")
println("  Median: $(round(median(sigmas), digits=4)) microns")
println("  Std: $(round(std(sigmas), digits=4)) microns")
println("  IQR: $(round.(quantile(sigmas, [0.25, 0.75]), digits=4)) microns")

# Filter by PSF width (typical range: 100-200nm)
expected_sigma = 0.13  # 130nm expected
tolerance = 0.05       # 50nm tolerance

valid = filter(e -> abs(e.sigma - expected_sigma) < tolerance, smld.emitters)
println("\nValid localizations (sigma within $(tolerance*1000)nm of expected):")
println("  $(length(valid)) / $(length(smld.emitters)) ($(round(100*length(valid)/length(smld.emitters), digits=1))%)")
```

## Comparing Fixed vs Variable PSF Models

```julia
using GaussMLE
using Statistics

# Same data, two models
data = rand(Float32, 11, 11, 1000)

# Fixed PSF model
fitter_fixed = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))
smld_fixed = fit(fitter_fixed, data)

# Variable PSF model
fitter_var = GaussMLEFitter(psf_model = GaussianXYNBS())
smld_var = fit(fitter_var, data)

# Compare position estimates
x_fixed = [e.x for e in smld_fixed.emitters]
x_var = [e.x for e in smld_var.emitters]

rms_diff = sqrt(mean((x_fixed .- x_var).^2))
println("Position difference (RMS): $(round(rms_diff*1000, digits=2)) nm")

# Compare uncertainties
sigma_x_fixed = mean([e.sigma_x for e in smld_fixed.emitters])
sigma_x_var = mean([e.sigma_x for e in smld_var.emitters])

println("Mean x uncertainty:")
println("  Fixed PSF:    $(round(sigma_x_fixed*1000, digits=2)) nm")
println("  Variable PSF: $(round(sigma_x_var*1000, digits=2)) nm")
println("  Ratio: $(round(sigma_x_var/sigma_x_fixed, digits=2))x")

# Performance comparison
t_fixed = @elapsed fit(fitter_fixed, data)
t_var = @elapsed fit(fitter_var, data)

println("\nPerformance:")
println("  Fixed PSF:    $(round(1000/t_fixed)) fits/second")
println("  Variable PSF: $(round(1000/t_var)) fits/second")
println("  Speed ratio: $(round(t_fixed/t_var, digits=2))x")
```

## Using with ROIBatch

```julia
using GaussMLE
using SMLMData

# Create camera
camera = SMLMData.IdealCamera(0:1023, 0:1023, 0.1)  # 100nm pixels

# Generate test data
batch = generate_roi_batch(
    camera,
    GaussianXYNBS(),  # Variable sigma model
    n_rois = 100,
    roi_size = 11
)

# Fit
fitter = GaussMLEFitter(psf_model = GaussianXYNBS())
smld = fit(fitter, batch)

# Extract results - positions in camera coordinates
sigmas = [e.sigma for e in smld.emitters]
println("Fitted PSF widths: $(round.(extrema(sigmas), digits=4)) microns")
```

## Anisotropic PSF (GaussianXYNBSXSY)

For elliptical PSFs, use `GaussianXYNBSXSY`:

```julia
using GaussMLE
using Statistics

# Anisotropic model - fits sigma_x and sigma_y independently
fitter = GaussMLEFitter(psf_model = GaussianXYNBSXSY())
smld = fit(fitter, data)

# Returns Emitter2DFitSigmaXY with sigma_x and sigma_y fields
sigma_x_psf = [e.sigma_x for e in smld.emitters]  # Note: this is position uncertainty
# For fitted PSF widths, the field names are different - check emitter type

# The emitter type Emitter2DFitSigmaXY has:
# - sigmxa, sigma_y: fitted PSF widths
# - sigma_sigmxa, sigma_sigma_y: uncertainties on PSF widths
```

## Troubleshooting

### PSF Width Convergence Issues

If fitted PSF widths are unreasonable:

```julia
# Check for extreme values
sigmas = [e.sigma for e in smld.emitters]

extreme_low = count(s -> s < 0.05, sigmas)   # < 50nm
extreme_high = count(s -> s > 0.5, sigmas)   # > 500nm

println("Extreme low sigma: $extreme_low")
println("Extreme high sigma: $extreme_high")

if extreme_low + extreme_high > length(sigmas) * 0.1
    println("Warning: >10% of fits have extreme PSF widths")
    println("Consider:")
    println("  - Using fixed PSF model (GaussianXYNB)")
    println("  - Checking data quality")
    println("  - Increasing iterations")
end
```

### Slow Convergence

```julia
# Use more iterations for variable PSF
fitter = GaussMLEFitter(
    psf_model = GaussianXYNBS(),
    iterations = 30  # Default is 20
)
```

## Next Steps

- Learn about [3D astigmatic fitting](@ref Models) with `AstigmaticXYZNB`
- Explore [GPU Support](@ref) for large datasets
- See the [API Reference](@ref) for all options
