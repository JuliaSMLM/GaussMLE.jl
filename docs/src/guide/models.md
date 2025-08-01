# Models

GaussMLE.jl supports different Gaussian models to accommodate various experimental conditions. Each model differs in which parameters are fitted versus fixed.

## Available Models

### GaussXyNb (`:xynb`)

The most commonly used model with 4 fitted parameters:

- **x, y**: Blob center position (pixels)
- **n**: Integrated intensity (photons)
- **bg**: Background counts per pixel

The PSF width (σ) is **fixed** and specified through the `args` parameter.

```julia
# PSF width is fixed at 1.3 pixels
args = GaussMLE.GaussModel.Args_xynb(Float32(1.3))
θ_found, Σ_found = GaussMLE.GaussFit.fitstack(data, :xynb, args)
```

**Use this model when:**
- PSF width is known and approximately constant across your dataset
- You want maximum fitting speed and stability
- Working with well-characterized microscopy systems

### GaussXyNbS (`:xynbs`)

Extended model with 5 fitted parameters:

- **x, y**: Blob center position (pixels)
- **n**: Integrated intensity (photons)
- **bg**: Background counts per pixel
- **σ_PSF**: PSF width (pixels)

```julia
# Initial guess for PSF width (will be refined during fitting)
args = GaussMLE.GaussModel.Args_xynbs(Float32(1.3))
θ_found, Σ_found = GaussMLE.GaussFit.fitstack(data, :xynbs, args)

# Access the fitted PSF width
psf_widths = getproperty.(θ_found, :σ_PSF)
```

**Use this model when:**
- PSF width varies significantly across your dataset
- You want to measure PSF variations
- Working with astigmatic PSFs or depth-dependent imaging
- PSF width is unknown

## Model Comparison

| Feature | GaussXyNb | GaussXyNbS |
|---------|-----------|------------|
| Parameters | 4 | 5 |
| Speed | Faster | Slower |
| Stability | More stable | Less stable |
| PSF handling | Fixed | Variable |
| Memory usage | Lower | Higher |
| Use case | Standard SMLM | Variable PSF |

## Mathematical Formulation

Both models use the same underlying Gaussian expectation:

```math
\mu(\theta)_{i,j} = \theta_b + \theta_n \int_{i-0.5}^{i+0.5} \int_{j-0.5}^{j+0.5} 
\frac{1}{2\pi\sigma^2} \exp\left(-\frac{(x-\theta_x)^2 + (y-\theta_y)^2}{2\sigma^2}\right) dx \, dy
```

Where:
- For **GaussXyNb**: σ is fixed (specified in args)
- For **GaussXyNbS**: σ = θ_σ_PSF (fitted parameter)

## Choosing the Right Model

### Decision Tree

1. **Is your PSF width well-characterized and stable?**
   - Yes → Use `GaussXyNb`
   - No → Continue to step 2

2. **Do you need to measure PSF variations?**
   - Yes → Use `GaussXyNbS`
   - No → Consider if stability vs. accuracy trade-off favors `GaussXyNb`

3. **Is fitting stability critical?**
   - Yes → Use `GaussXyNb` with best-estimate fixed PSF
   - No → Use `GaussXyNbS`

### Performance Considerations

**GaussXyNb advantages:**
- ~20-30% faster fitting
- More robust convergence
- Lower memory usage
- Fewer parameters to estimate

**GaussXyNbS advantages:**
- Adapts to PSF variations
- Can detect PSF changes
- More accurate when PSF is variable
- Provides PSF width measurements

## Practical Examples

### Example 1: Standard SMLM with Fixed PSF

```julia
using GaussMLE

# Known PSF width from calibration
psf_width = 1.35  # pixels

# Set up fixed PSF model
args = GaussMLE.GaussModel.Args_xynb(Float32(psf_width))

# Fit data
θ_found, Σ_found = GaussMLE.GaussFit.fitstack(data, :xynb, args)
```

### Example 2: Astigmatic PSF or Unknown PSF

```julia
using GaussMLE

# Initial guess for PSF width
initial_psf = 1.3  # pixels

# Set up variable PSF model
args = GaussMLE.GaussModel.Args_xynbs(Float32(initial_psf))

# Fit data
θ_found, Σ_found = GaussMLE.GaussFit.fitstack(data, :xynbs, args)

# Analyze PSF width distribution
using Statistics
psf_widths = getproperty.(θ_found, :σ_PSF)
println("Mean PSF width: $(mean(psf_widths))")
println("PSF width std: $(std(psf_widths))")
```

## Model Validation

### Checking Fit Quality

```julia
# Check convergence issues
n_converged = sum(θ -> θ.n > 0 && θ.bg >= 0, θ_found)
println("Converged fits: $n_converged / $(length(θ_found))")

# Check parameter ranges
x_positions = getproperty.(θ_found, :x)
y_positions = getproperty.(θ_found, :y)
println("X range: $(extrema(x_positions))")
println("Y range: $(extrema(y_positions))")
```

### Comparing Models

```julia
# Fit same data with both models
θ_fixed, Σ_fixed = GaussMLE.GaussFit.fitstack(data, :xynb, args_fixed)
θ_var, Σ_var = GaussMLE.GaussFit.fitstack(data, :xynbs, args_var)

# Compare position estimates
using Statistics
x_fixed = getproperty.(θ_fixed, :x)
x_var = getproperty.(θ_var, :x)
println("Position difference RMS: $(sqrt(mean((x_fixed - x_var).^2)))")
```