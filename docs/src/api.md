# API Reference

```@index
```

## Main Types and Functions

### Fitter and Fitting

```@docs
GaussMLEFitter
fit
```

### PSF Models

```@docs
GaussMLE.PSFModel
GaussianXYNB
GaussianXYNBS
GaussianXYNBSXSY
AstigmaticXYZNB
```

### Custom Emitter Types

GaussMLE defines custom emitter types that extend `SMLMData.AbstractEmitter` with additional fields for fitted PSF parameters and goodness-of-fit metrics.

```@docs
Emitter2DFitGaussMLE
Emitter2DFitSigma
Emitter2DFitSigmaXY
Emitter3DFitGaussMLE
```

### Data Structures

ROI data structures are provided by SMLMData.jl and re-exported for convenience:

- `ROIBatch` - Batch of ROIs with camera and position information
- `SingleROI` - Single ROI for fitting

### Simulation

```@docs
generate_roi_batch
```

### Device Management

Device selection is controlled via the `device` keyword argument to `GaussMLEFitter`:

- `:auto` or `nothing` - Automatically detect best device (default)
- `:cpu` - Force CPU execution
- `:gpu` - Use GPU if available, fallback to CPU

Example:
```julia
fitter = GaussMLEFitter(device = :gpu)  # Use GPU
fitter = GaussMLEFitter(device = :cpu)  # Force CPU
```

### Constraints

Parameter constraints can be configured via the `constraints` keyword argument to `GaussMLEFitter`. Default constraints are automatically generated based on the PSF model and ROI size.

## Camera Models

Camera models are defined in SMLMData.jl. GaussMLE uses these types:

- `SMLMData.IdealCamera` - Poisson noise only (no readout noise)
- `SMLMData.SCMOSCamera` - Poisson + per-pixel readout noise variance

See [SMLMData.jl documentation](https://github.com/JuliaSMLM/SMLMData.jl) for camera type details.

## Output Format

The `fit()` function returns `SMLMData.BasicSMLD` containing:

- `emitters::Vector{<:AbstractEmitter}` - Fitted emitter objects (type depends on PSF model)
- `camera::AbstractCamera` - Camera model used
- `n_frames::Int` - Number of frames
- `n_datasets::Int` - Number of datasets

### Emitter Type Dispatch

| PSF Model | Emitter Type |
|-----------|--------------|
| `GaussianXYNB` | `Emitter2DFitGaussMLE` |
| `GaussianXYNBS` | `Emitter2DFitSigma` |
| `GaussianXYNBSXSY` | `Emitter2DFitSigmaXY` |
| `AstigmaticXYZNB` | `Emitter3DFitGaussMLE` |
