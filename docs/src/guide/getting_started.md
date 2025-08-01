# Getting Started

## Installation

GaussMLE.jl is available through the Julia package registry:

```julia
using Pkg
Pkg.add("GaussMLE")
```

## Basic Workflow

The typical workflow for using GaussMLE.jl consists of three main steps:

1. **Generate or load data**: Either simulate synthetic data or load real microscopy images
2. **Choose a model**: Select the appropriate Gaussian model for your data
3. **Fit the data**: Use `fitstack` to estimate parameters

### Step 1: Data Preparation

GaussMLE.jl expects data as a 3D array with dimensions `(height, width, n_images)`. Each "slice" along the third dimension represents a small image region (typically 5×5 to 13×13 pixels) containing a single Gaussian blob.

```@example getting_started
using GaussMLE

# Generate synthetic data for demonstration
boxsz = 7  # Size of each image region
nboxes = 1000  # Number of regions to fit
data_type = Float32

# Generate synthetic data with Poisson noise
out, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynb; 
                                               T=data_type, poissonnoise=true)

println("Data shape: $(size(out))")
println("First few true x positions: $(getproperty.(θ_true[1:5], :x))")
```

### Step 2: Model Selection

Choose the appropriate model based on your experimental conditions:

- **GaussXyNb** (`:xynb`): Use when PSF width is known and constant
- **GaussXyNbS** (`:xynbs`): Use when PSF width varies and needs to be fitted

```@example getting_started
# For fixed PSF width (most common case)
model_type = :xynb

# For variable PSF width
# model_type = :xynbs
# args = GaussMLE.GaussModel.Args_xynbs(Float32(1.3))  # Initial PSF width
```

### Step 3: Fitting

Use `fitstack` to perform the fitting:

```@example getting_started
# Perform the fitting
θ_found, Σ_found = GaussMLE.GaussFit.fitstack(out, model_type, args)

# Display results
println("Number of fits: $(length(θ_found))")
println("First fit result:")
println("  x = $(θ_found[1].x) ± $(Σ_found[1].σ_x)")
println("  y = $(θ_found[1].y) ± $(Σ_found[1].σ_y)")
println("  intensity = $(θ_found[1].n) ± $(Σ_found[1].σ_n)")
println("  background = $(θ_found[1].bg) ± $(Σ_found[1].σ_bg)")
```

## Understanding the Results

The fitting returns two arrays:

- **`θ_found`**: Fitted parameter values for each region
- **`Σ_found`**: Uncertainty estimates (standard deviations) based on Cramér-Rao Lower Bound

### Parameter Access

Each fitted result is a structured type with named fields:

```@example getting_started
# Access parameters for the first fit
first_fit = θ_found[1]
first_uncertainty = Σ_found[1]

println("Parameter access:")
println("  Position: ($(first_fit.x), $(first_fit.y))")
println("  Intensity: $(first_fit.n)")
println("  Background: $(first_fit.bg)")

println("Uncertainties:")
println("  σ_x: $(first_uncertainty.σ_x)")
println("  σ_y: $(first_uncertainty.σ_y)")
println("  σ_n: $(first_uncertainty.σ_n)")
println("  σ_bg: $(first_uncertainty.σ_bg)")
```

## Performance Tips

- Use `Float32` instead of `Float64` for better performance when precision allows
- For large datasets, consider using GPU acceleration (see [GPU Support](@ref))
- Pre-allocate arrays when processing multiple datasets
- Choose the simplest model that adequately describes your data

## Next Steps

- Learn about [different models](@ref Models) and when to use them
- Explore [GPU acceleration](@ref "GPU Support") for large datasets
- Check out the [examples](@ref Examples) for more detailed use cases