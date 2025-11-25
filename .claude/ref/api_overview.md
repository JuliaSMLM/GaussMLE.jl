# GaussMLE.jl API Overview (LLM Documentation)

This document provides a comprehensive overview of the GaussMLE.jl API for LLM assistants. For formal Documenter.jl documentation, see `docs/src/api.md`.

## Package Purpose

GaussMLE.jl performs Maximum Likelihood Estimation of Gaussian blob parameters for single-molecule localization microscopy. It supports:
- Multiple PSF models (fixed/variable sigma, astigmatic 3D)
- **Physical units (microns)** for all PSF parameters
- Ideal and sCMOS camera noise models
- Automatic CPU/GPU acceleration via KernelAbstractions.jl
- Integration with JuliaSMLM ecosystem (SMLMData.jl, SMLMBoxer.jl)

## Installation

```julia
using Pkg
Pkg.add("GaussMLE")
```

## Re-export Convention

GaussMLE re-exports commonly needed types from SMLMData.jl, so users typically only need:

```julia
using GaussMLE  # Includes ROIBatch, SingleROI, camera types
```

No explicit `using SMLMData` required for basic workflows.

## Unit Convention

**All user-facing PSF parameters are in physical units (microns)**:
- `GaussianXYNB(σ)`: σ in microns (e.g., 0.13 for 130nm PSF)
- `AstigmaticXYZNB(...)`: All spatial params (σx₀, σy₀, γ, d) in microns
- Output positions and uncertainties in microns
- Internally converted to pixels for computation

## Core Workflow

```julia
using GaussMLE

# 1. Define camera (provides pixel_size for unit conversion)
camera = IdealCamera(0:511, 0:511, 0.1)  # 512×512 sensor, 100nm pixels

# 2. Create ROIBatch (typically from SMLMBoxer.jl, here using test generator)
batch = generate_roi_batch(
    camera,
    GaussianXYNB(0.13f0),  # PSF σ = 130nm (in microns)
    n_rois = 100,
    roi_size = 11
)

# 3. Create fitter with PSF model (sigma must match your microscope)
fitter = GaussMLEFitter(
    psf_model = GaussianXYNB(0.13f0),  # σ = 130nm in microns
    iterations = 20
)

# 4. Fit - returns SMLMData.BasicSMLD
smld = fit(fitter, batch)

# 5. Access results (positions in microns, from camera pixel_size)
for emitter in smld.emitters
    println("Position: ($(emitter.x), $(emitter.y)) μm")
    println("Photons: $(emitter.photons)")
    println("Precision: $(emitter.σ_x * 1000) nm")  # Convert to nm
end
```

## Typical Pipeline

In a real SMLM workflow:

```
Raw Movie → SMLMBoxer.jl → ROIBatch → GaussMLE.fit() → BasicSMLD → Analysis
```

- **SMLMBoxer.jl**: Detects candidates in raw frames, extracts ROIs with corners
- **GaussMLE.jl**: Fits PSF model to each ROI, returns localized positions
- **ROIBatch**: Contains pixel data + camera model + corner positions

## Main Types and Functions

### `GaussMLEFitter`

Configuration type for fitting.

```julia
fitter = GaussMLEFitter(;
    psf_model = GaussianXYNB(0.13f0),  # PSF model with physical params
    device = :auto,                      # :auto, :cpu, or :gpu
    iterations = 20,                     # Newton-Raphson iterations
    batch_size = 10_000                  # GPU batch size
)
```

### `fit(fitter, data)` → `SMLMData.BasicSMLD`

Fit Gaussian PSF to ROI data.

**Signatures:**
- `fit(fitter, data::Array{T,3})` - Fit raw 3D array (roi_size × roi_size × n_rois)
- `fit(fitter, batch::ROIBatch)` - Fit ROIBatch (preferred for real data)

**Returns:** `SMLMData.BasicSMLD` with emitters (type depends on PSF model)

### Output Emitter Types

| PSF Model | Emitter Type | Key Fields |
|-----------|--------------|------------|
| `GaussianXYNB` | `Emitter2DFit` | x, y, photons, bg, σ_x, σ_y |
| `GaussianXYNBS` | `Emitter2DFitSigma` | + σ, σ_σ (fitted PSF width) |
| `GaussianXYNBSXSY` | `Emitter2DFitSigmaXY` | + σ_x_psf, σ_y_psf (fitted PSF widths) |
| `AstigmaticXYZNB` | `Emitter3DFit` | x, y, z, photons, bg, σ_x, σ_y, σ_z |

All emitter types subtype `SMLMData.AbstractEmitter`.

### Emitter2DFit Fields

| Field | Description | Units |
|-------|-------------|-------|
| `x`, `y` | Fitted position | microns |
| `photons` | Total photon count | photons |
| `bg` | Background level | photons/pixel |
| `σ_x`, `σ_y` | Position uncertainty (CRLB) | microns |
| `σ_photons`, `σ_bg` | Photometry uncertainties | photons |
| `frame` | Frame number | integer |
| `dataset`, `track_id`, `id` | Metadata fields | integer |

## PSF Models

### `GaussianXYNB(σ::Float32)` - Fixed Width (Recommended)

**4 parameters:** [x, y, photons, background]

```julia
# σ from PSF calibration (typical: 0.10-0.15 μm for visible light, 100nm pixels)
psf = GaussianXYNB(0.13f0)  # 130nm PSF width
```

**Use when:** PSF width known from calibration (most common case)

### `GaussianXYNBS()` - Variable Width

**5 parameters:** [x, y, photons, background, σ]

```julia
psf = GaussianXYNBS()  # σ fitted per ROI
```

**Use when:** PSF varies (z-drift, aberrations) or for quality filtering

### `GaussianXYNBSXSY()` - Anisotropic

**6 parameters:** [x, y, photons, background, σx, σy]

```julia
psf = GaussianXYNBSXSY()  # Independent σx, σy
```

**Use when:** Elliptical PSF from astigmatism or aberrations

### `AstigmaticXYZNB{Float32}(...)` - 3D Astigmatic

**5 parameters:** [x, y, z, photons, background]

```julia
# All spatial params in microns
psf = AstigmaticXYZNB{Float32}(
    0.13f0, 0.13f0,   # σx₀, σy₀: in-focus widths (μm)
    0.05f0, -0.05f0,  # Ax, Ay: cubic coefficients
    0.01f0, -0.01f0,  # Bx, By: quartic coefficients
    0.2f0,            # γ: focal plane offset (μm)
    0.5f0             # d: depth scale (μm)
)
```

**Use when:** 3D localization with cylindrical lens

**Note:** z is in microns throughout (not pixels). The calibration parameters γ and d stay in microns.

## Camera Models

### `IdealCamera`

Poisson noise only (photon-limited).

```julia
camera = IdealCamera(512, 512, 0.1)  # nx, ny, pixel_size (μm)
```

### `SCMOSCamera`

Poisson + per-pixel readout noise, following [Huang et al. (2013)](https://doi.org/10.1038/nmeth.2488).

```julia
camera = SCMOSCamera(
    512, 512, 0.1,     # nx, ny, pixel_size (μm)
    readnoise_map,     # Per-pixel readnoise σ (electrons)
    offset = 100.0f0,  # Dark level (ADU)
    gain = 0.5f0,      # Conversion (e⁻/ADU)
    qe = 0.82f0        # Quantum efficiency
)
```

**Data flow for sCMOS:**
1. Raw data in ADU
2. `fit()` converts to electrons: `(ADU - offset) × gain`
3. Variance map extracted: `readnoise²`
4. Fitting uses Gaussian likelihood with variance = model + readnoise²

## ROIBatch

Container for ROI data with camera and position metadata.

```julia
# Typically created by SMLMBoxer.jl, but can create manually:
batch = ROIBatch(
    data,           # (roi_size, roi_size, n_rois) array
    x_corners,      # Vector{Int32} - x position of ROI on camera (1-indexed)
    y_corners,      # Vector{Int32} - y position of ROI on camera (1-indexed)
    frame_indices,  # Vector{Int32} - frame number for each ROI
    camera          # IdealCamera or SCMOSCamera
)
```

**Why corners matter:** For sCMOS cameras, the fitter needs to index the variance map at the correct camera position, not ROI-local position.

## Test Data Generation

```julia
# For testing/development (not typical workflow)
batch = generate_roi_batch(
    camera,
    psf_model;
    n_rois = 100,
    roi_size = 11,
    true_params = nothing,  # Auto-generate or provide matrix
    seed = 42               # Reproducibility
)
```

## GPU Acceleration

```julia
# Auto-detect (uses GPU if available)
fitter = GaussMLEFitter(device = :auto)

# Force specific device
fitter = GaussMLEFitter(device = :gpu, batch_size = 10_000)
fitter = GaussMLEFitter(device = :cpu)
```

GPU provides significant speedup for large batches (>1000 ROIs).

## Goodness-of-Fit

Each emitter includes a p-value from χ² test:

```julia
for e in smld.emitters
    if e.pvalue < 0.01
        # Poor fit - possibly overlapping emitters, wrong PSF model, etc.
    end
end
```

**Calculation:**
- IdealCamera: Poisson deviance → χ²(df) by Wilks' theorem
- sCMOS: Weighted residuals → χ²(df) (Gaussian approximation)
- df = n_pixels - n_parameters

## Common Patterns

### Wrapping Raw Data in ROIBatch

If you have raw 3D array data and want proper unit conversion, wrap it in `ROIBatch`:

```julia
using GaussMLE

# Your raw data: (roi_size, roi_size, n_rois) Float32 array
data = rand(Float32, 11, 11, 100)
n_rois = size(data, 3)

# Camera provides pixel_size for unit conversion
camera = IdealCamera(0:1023, 0:1023, 0.1)  # 100nm pixels

# Create ROIBatch - all ROIs at (1,1) if corners don't matter
batch = ROIBatch(
    data,
    ones(Int32, n_rois),       # x_corners: all at column 1
    ones(Int32, n_rois),       # y_corners: all at row 1
    collect(Int32, 1:n_rois),  # frame_indices
    camera
)

# Fit with proper unit handling
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))
smld = fit(fitter, batch)
```

**Note:** For IdealCamera, corners can all be (1,1) since no variance map lookup is needed. For sCMOS cameras, corners must be actual camera positions.

### Basic Fitting with ROIBatch

```julia
using GaussMLE

# Create camera and test data
camera = IdealCamera(0:511, 0:511, 0.1)  # 100nm pixels
batch = generate_roi_batch(camera, GaussianXYNB(0.13f0), n_rois=100, roi_size=11)

# Fit
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))
smld = fit(fitter, batch)

# Extract positions (in microns)
positions = [(e.x, e.y) for e in smld.emitters]
```

### Quality Filtering with @filter

Use SMLMData's `@filter` macro for quality control:

```julia
using GaussMLE

smld = fit(fitter, data)

# Filter by precision and photon count
good = @filter(smld, σ_x < 0.030 && photons > 500)

# Filter by multiple criteria
precise = @filter(smld, σ_x < 0.020 && σ_y < 0.020 && bg < 50)

println("Kept $(length(good.emitters)) / $(length(smld.emitters)) localizations")
```

### sCMOS with ROIBatch

```julia
using GaussMLE

# ROIBatch from SMLMBoxer.jl contains camera info
# batch = SMLMBoxer.extract_rois(movie, camera, detections)

fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))
smld = fit(fitter, batch)  # Automatically handles ADU→electrons
```

### 3D Localization

```julia
using GaussMLE

# Astigmatic PSF from calibration
psf_3d = AstigmaticXYZNB{Float32}(
    0.13f0, 0.13f0,   # σx₀, σy₀
    0.05f0, -0.05f0,  # Ax, Ay
    0.01f0, -0.01f0,  # Bx, By
    0.2f0, 0.5f0      # γ, d
)

fitter = GaussMLEFitter(psf_model = psf_3d, iterations = 30)
smld = fit(fitter, batch)

# Z positions in microns
z_values = [e.z for e in smld.emitters]
```

## Coordinate Conventions

- **ROI coordinates:** 1-indexed, pixel (1,1) is top-left
- **Camera coordinates:** 1-indexed, `x_corner`/`y_corner` are camera pixel indices
- **Output positions:** Microns from camera origin
- **Array indexing:** Julia convention `[row, col]` = `[y, x]`

## Performance Tips

1. **Use Float32** - Native GPU type, sufficient precision
2. **Fixed sigma when possible** - `GaussianXYNB` is ~20% faster than variable models
3. **Appropriate ROI size** - Minimum 4σ + 3 pixels (e.g., 11×11 for σ=1.3 pixels)
4. **GPU for large batches** - Overhead makes CPU faster for <1000 ROIs

## Related Packages

- **SMLMData.jl**: Core types (AbstractEmitter, ROIBatch, cameras)
- **SMLMBoxer.jl**: ROI extraction from raw movies
- **SMLMSim.jl**: Advanced simulation
- **SMLMMetrics.jl**: Performance metrics

## References

**MLE Algorithm:**
> Smith, C.S., Joseph, N., Rieger, B., & Lidke, K.A. (2010). Fast, single-molecule localization that achieves theoretically minimum uncertainty. *Nature Methods*, 7(5), 373-375. DOI: 10.1038/nmeth.1449

**sCMOS Camera Model:**
> Huang, F., Hartwich, T.M.P., Rivera-Molina, F.E., et al. (2013). Video-rate nanoscopy using sCMOS camera-specific single-molecule localization algorithms. *Nature Methods*, 10(7), 653-658. DOI: 10.1038/nmeth.2488

## Troubleshooting

**Poor precision:**
- Check PSF sigma matches actual microscope PSF
- Verify sufficient photons (>500 for good localization)
- Check ROI size adequate for PSF

**Negative photons:**
- Data preprocessing issue (sCMOS offset/gain)
- Background estimation problem

**GPU not detected:**
```julia
using CUDA
CUDA.functional()  # Should return true
```
