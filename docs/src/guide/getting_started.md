# Getting Started

## Installation

```julia
using Pkg
Pkg.add("GaussMLE")
```

## Re-exports from SMLMData

GaussMLE re-exports commonly needed types from SMLMData.jl, so you typically only need:

```julia
using GaussMLE  # Includes ROIBatch, camera types, etc.
```

No explicit `using SMLMData` required for basic workflows.

## Basic Workflow

The typical workflow for using GaussMLE.jl consists of three main steps:

1. **Create a fitter**: Configure PSF model, device (CPU/GPU), and fitting parameters
2. **Fit data**: Pass ROI data to the `fit()` function
3. **Access results**: Iterate over emitters in the returned `BasicSMLD`

### Step 1: Create a Fitter

```julia
using GaussMLE

# Create fitter with PSF model (sigma must match your microscope)
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))  # 130nm PSF width

# Or configure explicitly
fitter = GaussMLEFitter(
    psf_model = GaussianXYNB(0.13f0),  # sigma = 130nm in microns
    device = :cpu,                      # :cpu, :gpu, or :auto
    iterations = 20                     # Newton-Raphson iterations
)
```

### Step 2: Fit Data

GaussMLE.jl expects data as a 3D array with dimensions `(roi_size, roi_size, n_rois)`. Each slice along the third dimension represents a small image region (typically 7x7 to 13x13 pixels) containing a single Gaussian blob.

```julia
# Your data: (roi_size, roi_size, n_rois)
data = rand(Float32, 11, 11, 100)

# Fit the data
smld = fit(fitter, data)
```

### Step 3: Access Results

The `fit()` function returns a `SMLMData.BasicSMLD` containing emitter objects:

```julia
# Number of localizations
println("Fitted $(length(smld.emitters)) localizations")

# Iterate over emitters
for emitter in smld.emitters
    println("Position: ($(emitter.x), $(emitter.y)) microns")
    println("Photons: $(emitter.photons)")
    println("Precision: $(emitter.sigma_x) microns")
end

# Extract arrays for analysis
x_positions = [e.x for e in smld.emitters]
y_positions = [e.y for e in smld.emitters]
photons = [e.photons for e in smld.emitters]
precisions = [e.sigma_x for e in smld.emitters]
```

## Complete Example

```julia
using GaussMLE
using Statistics

# ROIBatch typically comes from SMLMBoxer.jl (extracts ROIs from raw movie data)
# For testing, use generate_roi_batch() or raw arrays:
data = rand(Float32, 11, 11, 100)

# Create fitter with PSF model (sigma from PSF calibration)
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))  # 130nm PSF width

# Fit the data
smld = fit(fitter, data)

# Display results
println("Fitted: $(length(smld.emitters)) localizations")

# Extract statistics
x_positions = [e.x for e in smld.emitters]
y_positions = [e.y for e in smld.emitters]
photons = [e.photons for e in smld.emitters]
precisions_x = [e.sigma_x for e in smld.emitters]

println("Mean position: ($(round(mean(x_positions), digits=2)), $(round(mean(y_positions), digits=2))) microns")
println("Mean photons: $(round(mean(photons), digits=1))")
println("Mean precision: $(round(mean(precisions_x)*1000, digits=1)) nm")
```

## Understanding the Output

### BasicSMLD Structure

The `fit()` function returns a `SMLMData.BasicSMLD` containing:

- `emitters`: Vector of emitter objects with fitted parameters
- `camera`: Camera model used for fitting
- `metadata`: Additional information

### Emitter Types

Different PSF models return different emitter types, all subtypes of `SMLMData.AbstractEmitter`:

| PSF Model | Emitter Type | Additional Fields |
|-----------|--------------|-------------------|
| `GaussianXYNB` | `Emitter2DFit` | (base type) |
| `GaussianXYNBS` | `Emitter2DFitSigma` | `σ`, `σ_σ` (fitted PSF width) |
| `GaussianXYNBSXSY` | `Emitter2DFitSigmaXY` | `sigma_x`, `sigma_y` (fitted PSF widths) |
| `AstigmaticXYZNB` | `Emitter3DFit` | `z`, `σ_z` (z-position) |

### Emitter2DFit Fields

All 2D emitter types include these fields:

| Field | Description | Units |
|-------|-------------|-------|
| `x`, `y` | Fitted position | microns |
| `photons` | Total photon count | photons |
| `bg` | Background level | photons/pixel |
| `σ_x`, `σ_y` | Position uncertainty (CRLB) | microns |
| `σ_photons`, `σ_bg` | Photometry uncertainties | photons |
| `frame` | Frame number | integer |
| `dataset`, `track_id`, `id` | Metadata fields | integer |

### Filtering Results

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

## Unit Convention

**All user-facing parameters use physical units (microns)**:

- PSF widths: specified in microns (e.g., `GaussianXYNB(0.13)` for 130nm PSF)
- Output positions: microns
- Output uncertainties: microns
- Internally converted to pixels for computation based on camera pixel size

## Working with ROIBatch

In a typical SMLM pipeline, `ROIBatch` comes from SMLMBoxer.jl which detects candidates and extracts ROIs from raw movie frames:

```
Raw Movie → SMLMBoxer.jl → ROIBatch → GaussMLE.fit() → BasicSMLD → Analysis
```

You can also create ROIBatch manually:

```julia
using GaussMLE

# Create camera model (65nm pixels)
camera = IdealCamera(0:2047, 0:2047, 0.065)

# Create ROIBatch with camera and corner positions
batch = ROIBatch(
    data,           # (roi_size, roi_size, n_rois) Float32 array
    x_corners,      # Vector{Int32} - x position of each ROI on sensor
    y_corners,      # Vector{Int32} - y position of each ROI on sensor
    frame_indices,  # Vector{Int32} - frame number for each ROI
    camera          # Camera model
)

# Fit with proper coordinate conversion
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))
smld = fit(fitter, batch)
```

## Generating Test Data

For testing and development, use `generate_roi_batch()`:

```julia
using GaussMLE

# Create camera
camera = IdealCamera(0:1023, 0:1023, 0.1)  # 100nm pixels

# Generate synthetic data with Poisson noise
batch = generate_roi_batch(
    camera,
    GaussianXYNB(0.13f0),  # PSF model (sigma in microns)
    n_rois = 100,
    roi_size = 11
)

# Fit the generated data
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))
smld = fit(fitter, batch)
```

## Performance Tips

- Use `Float32` data for best performance (native GPU type)
- For large datasets, GPU acceleration provides significant speedups
- Choose the simplest PSF model that adequately describes your data
- Pre-allocate arrays when processing multiple datasets

## Next Steps

- Learn about [different models](@ref Models) and when to use them
- Explore [GPU acceleration](@ref "GPU Support") for large datasets
- Check out the [examples](@ref "Basic Fitting Example") for more detailed use cases
