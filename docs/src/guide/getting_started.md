# Getting Started

## Installation

GaussMLE.jl requires SMLMData.jl as a dependency:

```julia
using Pkg

# Install SMLMData.jl dependency first
Pkg.add(url="https://github.com/JuliaSMLM/SMLMData.jl")

# Then install GaussMLE.jl
Pkg.add(url="https://github.com/JuliaSMLM/GaussMLE.jl")
```

Once registered in Julia General:
```julia
using Pkg
Pkg.add("GaussMLE")
```

## Basic Workflow

The typical workflow for using GaussMLE.jl consists of three main steps:

1. **Create a fitter**: Configure PSF model, device (CPU/GPU), and fitting parameters
2. **Fit data**: Pass ROI data to the `fit()` function
3. **Access results**: Iterate over emitters in the returned `BasicSMLD`

### Step 1: Create a Fitter

```julia
using GaussMLE
using SMLMData

# Create fitter with default settings (fixed-sigma Gaussian, auto device selection)
fitter = GaussMLEFitter()

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
using SMLMData
using Statistics

# Generate synthetic test data
data = rand(Float32, 11, 11, 100)

# Create fitter with fixed PSF width (130nm = 0.13 microns)
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))

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
| `GaussianXYNB` | `Emitter2DFitGaussMLE` | `pvalue` (goodness-of-fit) |
| `GaussianXYNBS` | `Emitter2DFitSigma` | `sigma`, `sigma_sigma` (fitted PSF width) |
| `GaussianXYNBSXSY` | `Emitter2DFitSigmaXY` | `sigma_x`, `sigma_y` (fitted PSF widths) |
| `AstigmaticXYZNB` | `Emitter3DFitGaussMLE` | `z`, `sigma_z` (z-position) |

### Common Emitter Fields

All emitter types share these fields:

- `x`, `y`: Position in microns
- `photons`: Total photon count
- `bg`: Background level
- `sigma_x`, `sigma_y`: Position uncertainties (CRLB, microns)
- `sigma_photons`, `sigma_bg`: Photometry uncertainties
- `frame`: Frame number
- `pvalue`: Goodness-of-fit p-value

## Unit Convention

**All user-facing parameters use physical units (microns)**:

- PSF widths: specified in microns (e.g., `GaussianXYNB(0.13)` for 130nm PSF)
- Output positions: microns
- Output uncertainties: microns
- Internally converted to pixels for computation based on camera pixel size

## Working with ROIBatch

For real microscopy data, use `SMLMData.ROIBatch` which includes camera information and ROI positions:

```julia
using GaussMLE
using SMLMData

# Create camera model (65nm pixels)
camera = SMLMData.IdealCamera(0:2047, 0:2047, 0.065)

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
using SMLMData

# Create camera
camera = SMLMData.IdealCamera(0:1023, 0:1023, 0.1)  # 100nm pixels

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
