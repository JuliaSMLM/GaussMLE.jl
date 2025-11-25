# Getting Started

## Installation

```julia
using Pkg
Pkg.add("GaussMLE")
```

GaussMLE re-exports `ROIBatch`, `IdealCamera`, `SCMOSCamera`, and `@filter` from SMLMData.jl.

!!! info "Unit Convention"
    PSF parameters and output positions are in **microns**. The camera's `pixel_size` handles all internal conversions.

## Basic Workflow

### Step 1: Define Camera and PSF

```julia
using GaussMLE

# Camera model (defines pixel size for unit conversion)
camera = IdealCamera(0:1023, 0:1023, 0.1)  # 100nm pixels

# PSF model (sigma in microns, from your PSF calibration)
psf = GaussianXYNB(0.13f0)  # 130nm PSF width
```

### Step 2: Create or Load ROIBatch

In a real workflow, `ROIBatch` comes from SMLMBoxer.jl:

```
Raw Movie → SMLMBoxer.jl → ROIBatch → GaussMLE.fit() → BasicSMLD
```

For testing, use `generate_roi_batch()`:

```julia
batch = generate_roi_batch(camera, psf, n_rois=100, roi_size=11)
```

### Step 3: Fit and Access Results

```julia
fitter = GaussMLEFitter(psf_model = psf)
smld = fit(fitter, batch)

# Results are in microns (camera coordinates)
for e in smld.emitters[1:3]
    println("Position: ($(e.x), $(e.y)) μm")
    println("Precision: $(e.σ_x * 1000) nm")
end
```

## Complete Example

```julia
using GaussMLE
using Statistics

# 1. Camera defines pixel size
camera = IdealCamera(0:1023, 0:1023, 0.1)  # 100nm pixels

# 2. PSF from calibration (in microns)
psf = GaussianXYNB(0.13f0)  # 130nm

# 3. Generate test data (or load from SMLMBoxer)
batch = generate_roi_batch(camera, psf, n_rois=100, roi_size=11)

# 4. Fit
fitter = GaussMLEFitter(psf_model = psf)
smld = fit(fitter, batch)

# 5. Results in microns
println("Fitted: $(length(smld.emitters)) localizations")
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

## Working with ROIBatch

In a typical SMLM pipeline, `ROIBatch` comes from SMLMBoxer.jl which detects candidates and extracts ROIs from raw movie frames:

```
Raw Movie → SMLMBoxer.jl → ROIBatch → GaussMLE.fit() → BasicSMLD → Analysis
```

### Wrapping Raw Data in ROIBatch

If you have raw 3D array data and want proper unit conversion, wrap it in `ROIBatch` with a camera model. This is the recommended approach even for simple cases:

```julia
using GaussMLE

# Your raw data: (roi_size, roi_size, n_rois) Float32 array
data = rand(Float32, 11, 11, 100)  # 100 ROIs of 11×11 pixels
n_rois = size(data, 3)

# Camera provides pixel_size for unit conversion
camera = IdealCamera(0:1023, 0:1023, 0.1)  # 100nm pixels

# Create ROIBatch - all ROIs at camera position (1,1) if corners don't matter
batch = ROIBatch(
    data,
    ones(Int32, n_rois),       # x_corners: all at column 1
    ones(Int32, n_rois),       # y_corners: all at row 1
    collect(Int32, 1:n_rois),  # frame_indices: 1, 2, 3, ...
    camera
)

# Fit with proper unit handling
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))
smld = fit(fitter, batch)

# Results in microns (relative to ROI corner)
println("Position: ($(smld.emitters[1].x), $(smld.emitters[1].y)) μm")
```

!!! note "When corners matter"
    For sCMOS cameras, the corners must be the actual camera positions because the fitter indexes into the pixel-specific variance map. For IdealCamera with uniform noise, corners can all be (1,1) since no variance map lookup is needed.

### Creating ROIBatch from Known Positions

You can also create ROIBatch manually with specific corner positions:

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
