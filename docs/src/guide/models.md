# Models

GaussMLE.jl provides four PSF model types to accommodate different experimental conditions. Each model differs in which parameters are fitted versus fixed.

## Available Models

### GaussianXYNB - Fixed Width 2D Gaussian

The most commonly used model with 4 fitted parameters:

- **x, y**: Blob center position
- **N**: Integrated photon count
- **bg**: Background counts per pixel

The PSF width (sigma) is **fixed** and specified at construction time.

```julia
using GaussMLE

# Create model with fixed sigma = 130nm
psf = GaussianXYNB(0.13f0)  # sigma in microns

# Use in fitter
fitter = GaussMLEFitter(psf_model = psf)
smld = fit(fitter, data)
```

**Use this model when:**
- PSF width is known from calibration
- PSF width is approximately constant across your dataset
- You want maximum fitting speed and stability
- Working with well-characterized microscopy systems

### GaussianXYNBS - Variable Width 2D Gaussian

Extended model with 5 fitted parameters:

- **x, y**: Blob center position
- **N**: Integrated photon count
- **bg**: Background counts per pixel
- **sigma**: PSF width (fitted)

```julia
using GaussMLE

# Create model - no fixed sigma needed
psf = GaussianXYNBS()

# Use in fitter
fitter = GaussMLEFitter(psf_model = psf)
smld = fit(fitter, data)

# Access fitted sigma from emitters
sigmas = [e.sigma for e in smld.emitters]
sigma_uncertainties = [e.sigma_sigma for e in smld.emitters]
```

**Use this model when:**
- PSF width varies across your dataset
- You want to measure PSF variations
- PSF width is unknown
- Quality control (filter by PSF width)

### GaussianXYNBSXSY - Anisotropic 2D Gaussian

Model with 6 fitted parameters for elliptical PSFs:

- **x, y**: Blob center position
- **N**: Integrated photon count
- **bg**: Background counts per pixel
- **sigma_x**: PSF width in x (fitted)
- **sigma_y**: PSF width in y (fitted)

```julia
using GaussMLE

# Create model - no fixed parameters
psf = GaussianXYNBSXSY()

# Use in fitter
fitter = GaussMLEFitter(psf_model = psf)
smld = fit(fitter, data)

# Access fitted sigma_x and sigma_y
sigma_x = [e.sigma_x for e in smld.emitters]
sigma_y = [e.sigma_y for e in smld.emitters]
```

**Use this model when:**
- PSF is elliptical/anisotropic
- Optical aberrations cause asymmetric PSF
- Measuring PSF anisotropy for quality control

### AstigmaticXYZNB - 3D Astigmatic PSF

Model for 3D localization using engineered astigmatism (5 fitted parameters):

- **x, y**: Lateral position
- **z**: Axial position
- **N**: Integrated photon count
- **bg**: Background counts per pixel

The z-position is encoded in the PSF shape through astigmatism calibration parameters.

```julia
using GaussMLE

# Create model with calibration parameters (all spatial params in microns)
psf = AstigmaticXYZNB{Float32}(
    0.13f0, 0.13f0,  # sigma_x0, sigma_y0: in-focus widths (microns)
    0.05f0, 0.05f0,  # Ax, Ay: cubic coefficients (dimensionless)
    0.3f0, 0.3f0,    # Bx, By: quartic coefficients (dimensionless)
    0.05f0,          # gamma: astigmatism offset (microns)
    0.4f0            # d: depth scale (microns)
)

# Use in fitter
fitter = GaussMLEFitter(psf_model = psf, iterations = 30)
smld = fit(fitter, data)

# Access z-position from emitters
z_positions = [e.z for e in smld.emitters]
z_uncertainties = [e.sigma_z for e in smld.emitters]
```

The PSF width varies with z according to:
```math
\sigma_x(z) = \sigma_{x0} \sqrt{1 + \left(\frac{z-\gamma}{d}\right)^2 + A_x\left(\frac{z-\gamma}{d}\right)^3 + B_x\left(\frac{z-\gamma}{d}\right)^4}
```
```math
\sigma_y(z) = \sigma_{y0} \sqrt{1 + \left(\frac{z+\gamma}{d}\right)^2 + A_y\left(\frac{z+\gamma}{d}\right)^3 + B_y\left(\frac{z+\gamma}{d}\right)^4}
```

**Use this model when:**
- You have a cylindrical lens or other astigmatism
- You need z-position information
- You have calibrated the PSF-z relationship

## Model Comparison

| Feature | GaussianXYNB | GaussianXYNBS | GaussianXYNBSXSY | AstigmaticXYZNB |
|---------|--------------|---------------|------------------|-----------------|
| Parameters | 4 | 5 | 6 | 5 |
| PSF sigma | Fixed | Fitted | Fitted (x,y) | Calibrated |
| Dimensions | 2D | 2D | 2D | 3D |
| Speed | Fastest | Medium | Slowest | Medium |
| Stability | Most stable | Less stable | Least stable | Stable |

## Unit Convention

**All PSF model parameters use physical units (microns):**

- `GaussianXYNB(sigma)`: sigma in microns (e.g., `0.13f0` = 130nm)
- `AstigmaticXYZNB`: sigma_x0, sigma_y0, gamma, d all in microns

The package internally converts to pixel units based on the camera pixel size during fitting.

## Choosing the Right Model

### Decision Tree

1. **Do you need 3D localization?**
   - Yes -> Use `AstigmaticXYZNB` (requires calibration)
   - No -> Continue to step 2

2. **Is your PSF well-characterized and stable?**
   - Yes -> Use `GaussianXYNB` (fastest, most stable)
   - No -> Continue to step 3

3. **Is your PSF isotropic (circular)?**
   - Yes -> Use `GaussianXYNBS`
   - No -> Use `GaussianXYNBSXSY`

### Performance Considerations

**GaussianXYNB advantages:**
- ~20-30% faster than variable-sigma models
- More robust convergence
- Lower memory usage
- Fewer parameters to estimate

**Variable-sigma model advantages:**
- Adapts to PSF variations
- Can detect PSF changes
- More accurate when PSF is variable
- Provides PSF width measurements for quality control

## Practical Examples

### Example 1: Standard SMLM with Fixed PSF

```julia
using GaussMLE
using Statistics

# Known PSF width from calibration (130nm)
psf = GaussianXYNB(0.13f0)

# Create fitter
fitter = GaussMLEFitter(psf_model = psf)

# Fit data
smld = fit(fitter, data)

# Extract results
x_positions = [e.x for e in smld.emitters]
precisions = [e.sigma_x for e in smld.emitters]

println("Mean precision: $(mean(precisions) * 1000) nm")
```

### Example 2: Variable PSF for Quality Control

```julia
using GaussMLE
using Statistics

# Variable sigma model
psf = GaussianXYNBS()

fitter = GaussMLEFitter(psf_model = psf)
smld = fit(fitter, data)

# Analyze PSF width distribution
sigmas = [e.sigma for e in smld.emitters]
println("Mean PSF width: $(mean(sigmas)) microns")
println("PSF width std: $(std(sigmas)) microns")

# Filter by PSF width
valid = filter(e -> 0.1 < e.sigma < 0.2, smld.emitters)
println("Valid localizations: $(length(valid)) / $(length(smld.emitters))")
```

### Example 3: 3D Astigmatic Localization

```julia
using GaussMLE
using Statistics

# Astigmatic PSF from calibration
psf = AstigmaticXYZNB{Float32}(
    0.13f0, 0.13f0,  # sigma_x0, sigma_y0
    0.05f0, 0.05f0,  # Ax, Ay
    0.3f0, 0.3f0,    # Bx, By
    0.05f0,          # gamma
    0.10f0           # d
)

fitter = GaussMLEFitter(psf_model = psf, iterations = 30)
smld = fit(fitter, data)

# Extract 3D positions
x = [e.x for e in smld.emitters]
y = [e.y for e in smld.emitters]
z = [e.z for e in smld.emitters]

println("Z range: $(extrema(z)) microns")
println("Mean Z precision: $(mean([e.sigma_z for e in smld.emitters]) * 1000) nm")
```

## Mathematical Formulation

All models use the Gaussian expectation with pixel integration:

```math
\mu_{i,j}(\theta) = \theta_{bg} + \theta_N \int_{i-0.5}^{i+0.5} \int_{j-0.5}^{j+0.5}
\frac{1}{2\pi \sigma_x \sigma_y} \exp\left(-\frac{(x-\theta_x)^2}{2\sigma_x^2} - \frac{(y-\theta_y)^2}{2\sigma_y^2}\right) dx \, dy
```

Where:
- For **GaussianXYNB**: sigma_x = sigma_y = sigma (fixed)
- For **GaussianXYNBS**: sigma_x = sigma_y = sigma (fitted)
- For **GaussianXYNBSXSY**: sigma_x, sigma_y (independently fitted)
- For **AstigmaticXYZNB**: sigma_x(z), sigma_y(z) from calibration curve

## Model Validation

### Checking Fit Quality

```julia
# All emitters have goodness-of-fit p-value
pvalues = [e.pvalue for e in smld.emitters]

# Filter by p-value (reject poor fits)
good_fits = filter(e -> e.pvalue > 0.01, smld.emitters)
println("Good fits: $(length(good_fits)) / $(length(smld.emitters))")
```

### Comparing Models

```julia
using GaussMLE
using Statistics

# Fit with both models
fitter_fixed = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))
fitter_var = GaussMLEFitter(psf_model = GaussianXYNBS())

smld_fixed = fit(fitter_fixed, data)
smld_var = fit(fitter_var, data)

# Compare position estimates
x_fixed = [e.x for e in smld_fixed.emitters]
x_var = [e.x for e in smld_var.emitters]

println("Position RMS difference: $(sqrt(mean((x_fixed .- x_var).^2)) * 1000) nm")
```
