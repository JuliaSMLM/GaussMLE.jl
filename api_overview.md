# GaussMLE.jl API Overview (LLM Documentation)

This document provides a comprehensive overview of the GaussMLE.jl API for LLM assistants. For formal Documenter.jl documentation, see `docs/src/api.md`.

## Package Purpose

GaussMLE.jl performs Maximum Likelihood Estimation of Gaussian blob parameters in 2D images. It supports:
- Multiple PSF models (fixed/variable sigma, astigmatic)
- Ideal and sCMOS camera noise models
- Automatic CPU/GPU acceleration via KernelAbstractions.jl
- Integration with SMLMData.jl ecosystem

## Core Workflow

```julia
using GaussMLE
using SMLMData

# 1. Create camera model
camera = SMLMData.IdealCamera(512, 512, 0.1)  # nx, ny, pixel_size (μm)

# 2. Choose PSF model
psf = GaussianXYNB(1.3f0)  # Fixed sigma PSF

# 3. Create fitter
fitter = GaussMLEFitter(
    psf_model = psf,
    camera_model = camera,
    device = :auto,      # :auto, :cpu, or :gpu
    iterations = 20
)

# 4. Fit data
results = fit(fitter, data)  # data: 3D array (roi_size, roi_size, n_rois)

# 5. Access results
results.x          # x positions
results.y          # y positions
results.photons    # photon counts
results.background # background levels
results.x_error    # CRLB uncertainties
```

## Main Types and Functions

### `GaussMLEFitter{D,P,C,PC}`

Main configuration type for fitting. Fields:
- `device::D` - ComputeDevice (CPU or GPU)
- `psf_model::P` - PSF model type
- `camera_model::C` - Camera noise model
- `iterations::Int` - Number of Newton-Raphson iterations (default: 20)
- `constraints::PC` - Parameter bounds and step limits
- `batch_size::Int` - GPU batch size (default: 10,000)

Constructor:
```julia
GaussMLEFitter(;
    psf_model = GaussianXYNB(1.3f0),
    camera_model = IdealCamera(),
    device = nothing,  # auto-detect
    iterations = 20,
    constraints = nothing,  # use defaults
    batch_size = 10_000
)
```

### `fit(fitter, data)` → `GaussMLEResults`

Fit Gaussian blobs to ROI data.

**Arguments:**
- `fitter::GaussMLEFitter` - Configured fitter
- `data::AbstractArray{T,3}` - ROI stack (roi_size × roi_size × n_rois)

**Keyword Arguments:**
- `variance_map=nothing` - Optional sCMOS variance map (legacy)

**Returns:** `GaussMLEResults` with fitted parameters and uncertainties

**Dispatches:**
- `fit(fitter, data::AbstractArray{T,3})` - Fit raw 3D array
- `fit(fitter, roi::AbstractMatrix)` - Fit single ROI (convenience)
- `fit(fitter, batch::ROIBatch)` - Fit ROIBatch from SMLMData

### `GaussMLEResults`

Result container with parameter-specific field access:

**Common to all PSF models:**
- `.x`, `.y` - Positions in pixels (relative to ROI corner)
- `.photons` - Total photon count
- `.background` - Background level
- `.x_error`, `.y_error`, `.photons_error`, `.background_error` - CRLB uncertainties

**Model-specific:**
- `.sigma` (GaussianXYNBS only) - PSF width
- `.sigma_x`, `.sigma_y` (GaussianXYNBSXSY only) - Separate widths
- `.z` (AstigmaticXYZNB only) - Axial position

**Raw access:**
- `.parameters` - Full parameter matrix (n_params × n_fits)
- `.uncertainties` - Full uncertainty matrix
- `.log_likelihoods` - Log-likelihood values

## PSF Models

All PSF models inherit from `PSFModel` abstract type. Each has a fixed parameter order.

### `GaussianXYNB(σ::Float32)`

Fixed-width Gaussian PSF. **4 parameters:** [x, y, photons, background]

```julia
psf = GaussianXYNB(1.3f0)  # σ = 1.3 pixels (typical for ~500nm PSF, 100nm pixels)
```

### `GaussianXYNBS()`

Variable-width Gaussian PSF. **5 parameters:** [x, y, photons, background, σ]

```julia
psf = GaussianXYNBS()  # σ fitted independently for each ROI
```

### `GaussianXYNBSXSY()`

Elliptical Gaussian PSF. **6 parameters:** [x, y, photons, background, σx, σy]

```julia
psf = GaussianXYNBSXSY()  # Independent widths in x and y
```

### `AstigmaticXYZNB(calibration)`

3D astigmatic PSF for z-position estimation. **5 parameters:** [x, y, z, photons, background]

```julia
using SMLMData

# Load calibration from file
cal = SMLMData.load_astig_calibration("calibration.mat")
psf = AstigmaticXYZNB(cal)

# Or create manually
psf = AstigmaticXYZNB(γ, d, Ax, Bx, Ay, By, z_offset)
```

**Calibration parameters:**
- `γ` - Asymmetry ratio (σy/σx at focus)
- `d` - Defocus distance (μm)
- `Ax, Bx, Ay, By` - Polynomial coefficients for σx(z), σy(z)
- `z_offset` - Z-position of focal plane

## Camera Models

### `IdealCamera()`

Poisson noise only (photon-counting statistics). No additional parameters needed.

```julia
camera = IdealCamera()
```

### sCMOS Cameras (SMLMData 0.4)

Use `SMLMData.SCMOSCamera` for realistic per-pixel calibration:

```julia
using SMLMData

camera = SMLMData.SCMOSCamera(
    256, 256,           # nx, ny pixels
    0.1,                # pixel size (μm)
    5.0f0,              # readnoise (e⁻ rms)
    offset = 100.0f0,   # dark level (ADU)
    gain = 0.5f0,       # conversion gain (e⁻/ADU)
    qe = 0.82f0         # quantum efficiency
)
```

**Important:** Data preprocessing happens automatically:
- Raw data in ADU is converted to electrons using gain/offset
- Readout noise variance is extracted
- Fitting operates in electron space
- This happens once before fitting, not per iteration

**Per-pixel calibration:**
All parameters (offset, gain, readnoise, qe) can be scalar or per-pixel matrices.

```julia
# Spatially varying readnoise
readnoise_map = rand(256, 256) .* 2.0f0 .+ 4.0f0  # 4-6 e⁻ rms
camera = SMLMData.SCMOSCamera(256, 256, 0.1, readnoise_map,
                              offset=100.0f0, gain=0.5f0, qe=0.82f0)
```

### `SCMOSCameraInternal{T}` (Internal use)

Lightweight internal type used during fitting (after preprocessing):

```julia
# Only stores variance map - used internally after preprocessing
camera_internal = SCMOSCameraInternal(variance_map)  # variance in e⁻²
```

**Preprocessing helpers:**
```julia
# Convert ADU data to electrons
data_e = to_electrons(data_adu, scmos_camera)

# Extract variance map
var_map = get_variance_map(scmos_camera)  # readnoise² in e⁻²
```

## Data Structures

### `ROIBatch{T,N,A,C}`

Container for batch of ROIs with camera and metadata from SMLMData.jl.

**Fields:**
- `data::Array{T,3}` - ROI stack (roi_size × roi_size × n_rois)
- `corners::Matrix{Int32}` - ROI corners (2 × n_rois) in camera coordinates
- `frame_indices::Vector{Int32}` - Frame index for each ROI
- `camera::C` - Camera object (IdealCamera or SCMOSCamera)

**Creation:**
```julia
batch = ROIBatch(data, corners, frame_indices, camera)
```

**Iteration:**
```julia
for roi in batch
    # roi is a SingleROI with .data, .corner, .frame_index
end
```

### `LocalizationResult`

Result with camera-space coordinates (from fitting ROIBatch).

**Fields:**
- `parameters::Matrix{Float32}` - Fitted parameters (camera coordinates)
- `uncertainties::Matrix{Float32}` - CRLB uncertainties
- `log_likelihoods::Vector{Float32}` - Log-likelihood values
- `roi_corners::Matrix{Int32}` - ROI corners in camera space
- `frame_indices::Vector{Int32}` - Frame indices

**Conversion to SMLMData types:**
```julia
# Convert to SMLMData.Emitter2DFit
emitters = to_emitter2dfit(loc_result, psf_model)

# Convert to SMLD format (legacy)
smld = to_smld(loc_result, psf_model)
```

## Parameter Constraints

### `ParameterConstraints{N}`

Defines bounds and step limits for fitting parameters.

**Fields:**
- `lower_bounds::SVector{N,Float32}` - Minimum allowed values
- `upper_bounds::SVector{N,Float32}` - Maximum allowed values
- `max_step::SVector{N,Float32}` - Maximum update per iteration

**Creation:**
```julia
using StaticArrays

constraints = ParameterConstraints{4}(
    SVector{4,Float32}(0.5, 0.5, 10.0, 0.0),      # lower: x, y, photons, bg
    SVector{4,Float32}(10.5, 10.5, 1e5, 100.0),   # upper
    SVector{4,Float32}(1.0, 1.0, Inf32, Inf32)    # max_step (Inf = unlimited)
)
```

**Default constraints:**
```julia
constraints = default_constraints(psf_model, roi_size)
```

Defaults for 11×11 ROI:
- **x, y:** [0.5, 10.5], max_step = 1.0
- **photons:** [10, 1e5], max_step = Inf
- **background:** [0, 100], max_step = Inf
- **σ:** [0.5, 3.0], max_step = 0.5
- **z:** [-1.0, 1.0], max_step = 0.2

## Simulation

### `generate_roi_batch(camera, psf_model; kwargs...)`

Generate synthetic ROI data with realistic noise.

**Arguments:**
- `camera::AbstractCamera` - Camera model (determines noise characteristics)
- `psf_model::PSFModel` - PSF model (determines parameter structure)

**Keyword Arguments:**
- `n_rois::Int = 100` - Number of ROIs to generate
- `roi_size::Int = 11` - Size of each square ROI
- `true_params::Matrix = nothing` - Use specific parameters or defaults
- `corners::Matrix{Int32} = nothing` - ROI positions or auto-generate
- `frame_indices::Vector{Int32} = nothing` - Frame indices or all frame 1
- `xy_variation::Float32 = 1.0` - Position jitter (±pixels) when using defaults
- `corner_mode::Symbol = :random` - Corner generation (:random, :grid, :clustered)
- `min_spacing::Int = 20` - Minimum pixel spacing between ROIs
- `seed::Int = nothing` - Random seed for reproducibility

**Returns:** `ROIBatch` ready for fitting

**Example:**
```julia
# Simple: generate with defaults
batch = generate_roi_batch(camera, psf, n_rois=100, seed=42)

# Custom parameters
true_params = Float32[
    6.0 .+ randn(100)';   # x positions
    6.0 .+ randn(100)';   # y positions
    1000.0 .+ 200*randn(100)';  # photons
    10.0 .+ 2*randn(100)'   # background
]
batch = generate_roi_batch(camera, psf, true_params=true_params)

# Spatially organized ROIs
batch = generate_roi_batch(camera, psf, n_rois=100,
                          corner_mode=:grid, min_spacing=30)
```

**Noise model:**
- **IdealCamera:** Poisson noise only
- **SCMOSCamera:** Full pipeline simulation
  1. Photons → electrons (apply QE, Poisson noise)
  2. Add readout noise (Gaussian, per-pixel)
  3. Electrons → ADU (apply gain, offset)

## Device Management

### `ComputeDevice` Types

Abstract type with two concrete implementations:
- `CPU()` - CPU execution
- `GPU()` - CUDA GPU execution

### `auto_device()` → `ComputeDevice`

Auto-detect best available device. Returns `GPU()` if CUDA functional, else `CPU()`.

```julia
device = auto_device()
```

### `select_device(device)` → `ComputeDevice`

Validate and normalize device specification.

```julia
device = select_device(:auto)   # Auto-detect
device = select_device(:cpu)    # Force CPU
device = select_device(:gpu)    # Force GPU (warns if unavailable)
device = select_device(GPU())   # Pass device object directly
```

## Common Usage Patterns

### Basic Fitting

```julia
using GaussMLE

# Simple fitting with defaults
data = rand(Float32, 11, 11, 100)  # Your ROI data
fitter = GaussMLEFitter()
results = fit(fitter, data)

# Access results
println("Mean precision: $(mean(results.x_error)) pixels")
```

### sCMOS Camera with Real Calibration

```julia
using GaussMLE
using SMLMData

# Real camera parameters (e.g., Hamamatsu ORCA-Flash4.0)
camera = SMLMData.SCMOSCamera(
    2048, 2048, 0.065,  # 2048×2048, 65nm pixels
    1.6f0,              # 1.6 e⁻ rms readnoise
    offset = 100.0f0,
    gain = 0.46f0,      # From spec sheet
    qe = 0.72f0
)

# Generate test data (returns data in ADU)
batch = generate_roi_batch(camera, GaussianXYNB(1.3f0), n_rois=1000)

# Fit - preprocessing happens automatically
fitter = GaussMLEFitter(camera_model=camera)
results = fit(fitter, batch)  # Returns LocalizationResult
```

### Variable Sigma Fitting

```julia
# When PSF width varies (e.g., z-drift, aberrations)
psf = GaussianXYNBS()
fitter = GaussMLEFitter(psf_model=psf, iterations=30)
results = fit(fitter, data)

# Access fitted widths
println("Mean σ: $(mean(results.sigma)) pixels")
```

### 3D Astigmatic Fitting

```julia
using SMLMData

# Load calibration
cal = load_astig_calibration("astig_calibration.mat")
psf = AstigmaticXYZNB(cal)

# Fit with z-estimation
fitter = GaussMLEFitter(psf_model=psf)
results = fit(fitter, data)

# Z positions in microns
z_positions = results.z
z_precision = results.z_error
```

### GPU Acceleration

```julia
# Force GPU (with fallback)
fitter = GaussMLEFitter(device=:gpu, batch_size=5000)
results = fit(fitter, large_dataset)  # Processes in batches of 5000

# Check if GPU was used
if fitter.device isa GPU
    println("Using GPU acceleration")
end
```

### Custom Constraints

```julia
using StaticArrays

# Tight constraints for crowded field
constraints = ParameterConstraints{4}(
    SVector{4,Float32}(3.0, 3.0, 100.0, 5.0),    # Tighter bounds
    SVector{4,Float32}(8.0, 8.0, 5000.0, 50.0),
    SVector{4,Float32}(0.5, 0.5, Inf32, Inf32)   # Smaller position steps
)

fitter = GaussMLEFitter(constraints=constraints)
```

## Key Implementation Details

### Coordinate Systems

- **ROI coordinates:** Origin at top-left corner of ROI, pixel (1,1) center at (0.5, 0.5)
- **Camera coordinates:** Origin at top-left of sensor
- **Physical units:** Positions in pixels, pixel_size in microns
- **Conversion:** `camera_x = roi_x + corner_x - 1`

### Fitting Algorithm

1. **Initialization:** Center of ROI (automatic z-init for astigmatic)
2. **Iteration:** Newton-Raphson with diagonal Hessian approximation
3. **Convergence:** Fixed iterations (no convergence criterion)
4. **Uncertainties:** Cramér-Rao Lower Bound from full Fisher Information Matrix

### Performance Tips

1. **Use GPU for large batches:** Speedup ~10-50× for >1000 ROIs
2. **Batch size:** Tune `batch_size` for GPU memory (default 10,000 works well)
3. **Type stability:** Keep data as Float32 for best performance
4. **Preprocess once:** sCMOS preprocessing happens automatically before fitting
5. **Fixed sigma when possible:** GaussianXYNB faster than GaussianXYNBS

## Version Compatibility

- **Julia:** 1.6+
- **SMLMData:** 0.4+
- **CUDA (optional):** 4.x or 5.x
- **KernelAbstractions:** 0.9+

## Related Packages

- **SMLMData.jl:** Data structures for single-molecule localization microscopy
- **SMLMSim.jl:** Advanced simulation tools
- **SMLMMetrics.jl:** Performance metrics and benchmarking
- **SMITE (MATLAB):** Reference implementation

## Migration from Legacy API

**Old (GaussLib/GaussFit):**
```julia
using GaussMLE.GaussFit
coords = fitstack(data, model=GaussXyNb(), sigma=1.3)
```

**New (Refactored):**
```julia
using GaussMLE
fitter = GaussMLEFitter(psf_model=GaussianXYNB(1.3f0))
results = fit(fitter, data)
coords = [results.x results.y results.photons results.background]
```

## Troubleshooting

**Negative data warning:**
- Indicates ADU values < 0 (likely preprocessing issue)
- Check camera offset/gain calibration
- For sCMOS: ensure offset > readnoise × 5

**GPU not detected:**
- Install CUDA.jl: `using Pkg; Pkg.add("CUDA")`
- Check: `using CUDA; CUDA.functional()`
- Fallback to CPU is automatic

**Poor fits:**
- Check ROI size matches PSF (use roi_size ≥ 4σ + 3)
- Verify initial sigma estimate reasonable
- Try more iterations (default 20, try 30-50)
- Check constraints aren't too tight

**High uncertainties:**
- Low photon count (increase signal or integration time)
- High background (improve sample prep, reduce illumination)
- sCMOS readnoise (use EMCCD for low-light applications)
