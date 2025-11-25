# Basic Fitting Example

This example demonstrates the most common use case: fitting Gaussian blobs with a fixed PSF width using the `GaussianXYNB` model.

## Running the Example

The complete example is available in `examples/basic_fitting.jl`:

```bash
julia --project=. examples/basic_fitting.jl
```

## Step-by-Step Walkthrough

### Setup and Imports

```julia
using GaussMLE
using Statistics
```

### Creating a Fitter

```julia
# Create fitter with default settings
# - GaussianXYNB(0.13f0): fixed PSF width of 130nm
# - Auto device selection (GPU if available)
# - 20 Newton-Raphson iterations
fitter = GaussMLEFitter()

# Or with explicit configuration
fitter = GaussMLEFitter(
    psf_model = GaussianXYNB(0.13f0),  # sigma = 130nm in microns
    device = :cpu,                      # Force CPU
    iterations = 20
)
```

### Preparing Data

GaussMLE expects data as a 3D array with dimensions `(roi_size, roi_size, n_rois)`:

```julia
# Example: 100 ROIs of 11x11 pixels each
data = rand(Float32, 11, 11, 100)
```

For real data, each ROI should contain a single fluorescent emitter centered approximately in the ROI.

### Fitting the Data

```julia
# Perform the fitting
smld = fit(fitter, data)

println("Fitted $(length(smld.emitters)) localizations")
```

### Accessing Results

The `fit()` function returns a `SMLMData.BasicSMLD` containing emitter objects:

```julia
# Extract position arrays
x_positions = [e.x for e in smld.emitters]
y_positions = [e.y for e in smld.emitters]

# Extract photometry
photons = [e.photons for e in smld.emitters]
backgrounds = [e.bg for e in smld.emitters]

# Extract uncertainties (CRLB)
precisions_x = [e.sigma_x for e in smld.emitters]
precisions_y = [e.sigma_y for e in smld.emitters]

# Display statistics
println("Mean position: ($(round(mean(x_positions), digits=2)), $(round(mean(y_positions), digits=2))) microns")
println("Mean photons: $(round(mean(photons), digits=1))")
println("Mean precision: $(round(mean(precisions_x)*1000, digits=1)) nm")
```

## Complete Working Example

```julia
using GaussMLE
using Statistics

# ROIBatch typically comes from SMLMBoxer.jl (extracts ROIs from raw movie frames)
# For testing, use generate_roi_batch() or raw arrays:
println("Generating synthetic data...")
data = rand(Float32, 11, 11, 100)

# Create fitter with PSF model (sigma from PSF calibration)
println("Creating fitter with 130nm PSF width...")
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))

# Fit
println("Fitting $(size(data, 3)) ROIs...")
smld = fit(fitter, data)

# Display results
println("\n=== Results ===")
println("Type: BasicSMLD")
println("Fitted: $(length(smld.emitters)) localizations")

# Extract statistics
x_positions = [e.x for e in smld.emitters]
y_positions = [e.y for e in smld.emitters]
photons = [e.photons for e in smld.emitters]
backgrounds = [e.bg for e in smld.emitters]
precisions_x = [e.sigma_x for e in smld.emitters]

println("Mean position: ($(round(mean(x_positions), digits=2)), $(round(mean(y_positions), digits=2))) microns")
println("Mean photons: $(round(mean(photons), digits=1))")
println("Mean background: $(round(mean(backgrounds), digits=1))")
println("Mean precision: $(round(mean(precisions_x)*1000, digits=1)) nm")
```

## Understanding the Results

### Emitter2DFit Fields

Each emitter in `smld.emitters` is an `Emitter2DFit` containing:

| Field | Description | Units |
|-------|-------------|-------|
| `x`, `y` | Fitted position | microns |
| `photons` | Total photon count | photons |
| `bg` | Background level | photons/pixel |
| `σ_x`, `σ_y` | Position uncertainty (CRLB) | microns |
| `σ_photons`, `σ_bg` | Photometry uncertainties | photons |
| `frame` | Frame number | integer |
| `dataset`, `track_id`, `id` | Metadata fields | integer |

### Quality Filtering with @filter

Use SMLMData's `@filter` macro for quality control:

```julia
using GaussMLE

smld = fit(fitter, data)

# Filter by precision and photon count
good = @filter(smld, σ_x < 0.020 && photons > 500)

# Filter by multiple criteria
precise = @filter(smld, σ_x < 0.015 && σ_y < 0.015 && bg < 50)

println("Kept $(length(good.emitters)) / $(length(smld.emitters)) localizations")
```

## Using ROIBatch for Real Data

In a typical SMLM pipeline, `ROIBatch` comes from SMLMBoxer.jl which detects candidates and extracts ROIs from raw movie frames:

```
Raw Movie → SMLMBoxer.jl → ROIBatch → GaussMLE.fit() → BasicSMLD → Analysis
```

For testing with proper coordinate handling:

```julia
using GaussMLE

# Create camera model (100nm pixels)
camera = IdealCamera(0:1023, 0:1023, 0.1)

# Generate synthetic data with proper camera model
batch = generate_roi_batch(
    camera,
    GaussianXYNB(0.13f0),  # PSF model
    n_rois = 100,
    roi_size = 11
)

# Fit with proper coordinate conversion
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))
smld = fit(fitter, batch)

# Positions are now in camera coordinates (microns)
x_positions = [e.x for e in smld.emitters]
println("X range: $(extrema(x_positions)) microns")
```

## Performance Optimization

### Use Float32

```julia
# Float32 is faster and sufficient precision
data = Float32.(your_data)
```

### Batch Processing for Large Datasets

```julia
# For GPU: configure batch size based on memory
fitter = GaussMLEFitter(
    device = :gpu,
    batch_size = 10_000
)

# Fit large dataset
large_data = rand(Float32, 11, 11, 100_000)
@time smld = fit(fitter, large_data)
```

### Timing Comparison

```julia
using GaussMLE

data = rand(Float32, 11, 11, 10_000)

# CPU timing
fitter_cpu = GaussMLEFitter(device = :cpu)
t_cpu = @elapsed fit(fitter_cpu, data)
println("CPU: $(round(10_000/t_cpu)) fits/second")

# GPU timing (if available)
fitter_gpu = GaussMLEFitter(device = :gpu)
t_gpu = @elapsed fit(fitter_gpu, data)
println("GPU: $(round(10_000/t_gpu)) fits/second")
```

## Practical Considerations

### Choosing ROI Size

The ROI should be large enough to capture the full PSF:

```julia
# Rule of thumb: ROI size >= 4 * PSF_width_in_pixels
# For 130nm PSF with 100nm pixels: PSF = 1.3 pixels
# Minimum ROI: 6 pixels (use 7 or 11 for safety)
```

### Handling Edge Cases

```julia
# Filter out failed fits using @filter
good = @filter(smld, photons > 0)
println("Valid fits: $(length(good.emitters)) / $(length(smld.emitters))")
```

## Next Steps

- Try the [PSF Width Fitting](@ref "PSF Width Fitting Example") example for variable PSF scenarios
- Learn about [GPU Support](@ref) for larger datasets
- See the [API Reference](@ref) for all options
