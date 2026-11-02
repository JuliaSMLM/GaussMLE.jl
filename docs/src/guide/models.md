# Models

GaussMLE.jl provides five PSF model types, including `SplinePSFModel` for calibrated
tabulated or learned 3D PSFs. All models run on CPU and GPU. Each differs in which
parameters are fitted versus fixed.

## Available Models

### GaussianXYNB - Fixed Width 2D Gaussian

The most commonly used model with 4 fitted parameters:

- **x, y**: Blob center position
- **N**: Integrated signal (photons for IdealCamera, electrons for sCMOS)
- **bg**: Background counts per pixel

The PSF width (sigma) is **fixed** and specified at construction time.

```julia
using GaussMLE

# Create model with fixed sigma = 130nm
psf = GaussianXYNB(0.13f0)  # sigma in microns

# Use in fitter
fitter = GaussMLEConfig(psf_model = psf)
smld, info = fit(data, fitter)
```

**Use this model when:**
- PSF width is known from calibration
- PSF width is approximately constant across your dataset
- You want maximum fitting speed and stability
- Working with well-characterized microscopy systems

### GaussianXYNBS - Variable Width 2D Gaussian

Extended model with 5 fitted parameters:

- **x, y**: Blob center position
- **N**: Integrated signal (photons for IdealCamera, electrons for sCMOS)
- **bg**: Background counts per pixel
- **sigma**: PSF width (fitted)

```julia
using GaussMLE

# Create model - no fixed sigma needed
psf = GaussianXYNBS()

# Use in fitter
fitter = GaussMLEConfig(psf_model = psf)
smld, info = fit(data, fitter)

# Access fitted sigma from emitters
sigmas = [e.σ for e in smld.emitters]
sigma_uncertainties = [e.σ_σ for e in smld.emitters]
```

**Use this model when:**
- PSF width varies across your dataset
- You want to measure PSF variations
- PSF width is unknown
- Quality control (filter by PSF width)

### GaussianXYNBSXSY - Anisotropic 2D Gaussian

Model with 6 fitted parameters for elliptical PSFs:

- **x, y**: Blob center position
- **N**: Integrated signal (photons for IdealCamera, electrons for sCMOS)
- **bg**: Background counts per pixel
- **sigma_x**: PSF width in x (fitted)
- **sigma_y**: PSF width in y (fitted)

```julia
using GaussMLE

# Create model - no fixed parameters
psf = GaussianXYNBSXSY()

# Use in fitter
fitter = GaussMLEConfig(psf_model = psf)
smld, info = fit(data, fitter)

# Access fitted PSF widths (σx, σy) and position uncertainties (σ_x, σ_y)
psf_widths_x = [e.σx for e in smld.emitters]
psf_widths_y = [e.σy for e in smld.emitters]
```

**Use this model when:**
- PSF is elliptical/anisotropic
- Optical aberrations cause asymmetric PSF
- Measuring PSF anisotropy for quality control

### AstigmaticXYZNB - 3D Astigmatic PSF

Model for 3D localization using engineered astigmatism (5 fitted parameters):

- **x, y**: Lateral position
- **z**: Axial position
- **N**: Integrated signal (photons for IdealCamera, electrons for sCMOS)
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
fitter = GaussMLEConfig(psf_model = psf, iterations = 30)
smld, info = fit(data, fitter)

# Access z-position from emitters
z_positions = [e.z for e in smld.emitters]
z_uncertainties = [e.σ_z for e in smld.emitters]
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

### SplinePSFModel - Tabulated 3D PSF

A 3D model (x, y, z, N, bg) backed by a tabulated PSF: a measured or learned PSF, such as a
MicroscopePSFs `SplinePSF` written by PSFLearning's `save_psf`. The callable supplies the point
PSF; GaussMLE integrates it over each camera pixel. z is in microns throughout, and the model
runs on CPU (POCL), CUDA and AMDGPU from the same pixel-integral table and fitting kernel.

```julia
using GaussMLE, MicroscopePSFs

psf = load_psf("psf.h5")                       # a 3D SplinePSF, in microns
model = SplinePSFModel(psf; pixel_size = 0.1)  # z_range defaults to the spline's own range
smld, info = fit(batch; psf_model = model)     # GPU when available, otherwise CPU
```

Any callable `(x, y, z)` in microns works as well, without MicroscopePSFs:
`SplinePSFModel(f; pixel_size, z_range)`. `pixel_size` accepts a `Real` for square pixels
or an `(x, y)` tuple for rectangular pixels, for example `(0.1313, 0.1258)` in microns.
The model stores both sizes as a Float32 tuple, and `fit(::ROIBatch)` rescales each axis
to the camera's own spacing. This rectangular-pixel support is specific to `SplinePSFModel`.

Construction builds a Float32 table of pixel averages, using composite two-point Gauss-Legendre
quadrature with five sub-intervals per lateral axis. The lateral table spacings are
`pixel_size_x / 5` and `pixel_size_y / 5`, extending to at least
`±(lateral_range + pixel_size_axis)` per axis; axial nodes span the inward-rounded
`z_range` with spacing at most 0.02 µm. A natural cubic B-spline interpolates these averages,
with zero second derivative at each axis endpoint. Fitting uses the table, with zero outside
its lateral range and z clamped to `z_range`. `fit` rebuilds the table and signal scale on
every call whose camera pixel size differs from the model's, about a second for a
PSFLearning-size table, so a chunked pipeline pays it per chunk: build the model at the
camera pixel size. An unchanged `(x, y)` pair reuses the model.

At fit time, Adapt transfers the coefficient table to the selected GPU and removes the
point callable from the kernel model. The table is shared across batches; a retry after
a GPU failure transfers it again to the selected device. POCL uses a device-array view
of the host table. No kernel calls the original point PSF or allocates a new table.
The Float32 coefficients occupy `4 * prod(size(model.pixel_psf.coefficients))` bytes,
typically about 5 to 10 MiB for a PSFLearning export, in addition to the ROI data.

Each pixel evaluation gathers 4 × 4 × 4 coefficients once and obtains the value and
analytic first derivatives from cubic B-spline basis weights. Each lateral derivative uses its own pixel size. Non-positive values
are clipped to zero with zero derivatives; axial derivatives are evaluated at clamped z.
The initial z scan uses a value-only gather from the same table.

The input must describe a point PSF: an already pixel-integrated callable would be integrated
twice. Tabulate slow callables, such as a vector PSF, as a `SplinePSF` before constructing the model.

N is the signal scale in photons for IdealCamera or electrons for sCMOS; background uses
the corresponding units per pixel. For a `MicroscopePSFs.AbstractPSF`, the point PSF is a
normalized photon density in µm^-2: N is total photons (or total electrons for sCMOS), and
the pixel averages are multiplied by the pixel area without renormalization. A stack used
to construct a `SplinePSF` must already have this density normalization. Construction throws
`ArgumentError` if the pixel-integrated sum over the `lateral_range` patch at the reference
plane lies outside `[0.5, 1.05]`, indicating an unnormalized density such as a raw bead image.

A plain callable has arbitrary scale and is normalized on the reference plane, so N is the
photons within the `lateral_range` patch at `z_ref` (electrons for sCMOS). Its energy must be
the same at every z for that signal interpretation to hold across z. The reference plane is
nearest z = 0 within the callable's valid range; for a `SplinePSF`, its own valid range is
used. Pixel-size rescaling preserves this plane and recomputes the appropriate scale.
A non-finite or non-positive reference sum for a plain callable throws `ArgumentError`.
Point samples are clipped at zero before quadrature for both conventions.

Fitting starts at the intensity centroid for x and y, the edge-pixel mean for the
background, the positive in-box sum above it for N, and the z of a Poisson likelihood scan
at those values (grid spacing at most 0.1 µm). The fit is Levenberg-Marquardt on the full
Fisher matrix, climbing the Poisson likelihood (IdealCamera) or the Poisson-shifted
likelihood `(d + v) log(μ + v) - (μ + v)` with `v` the read-noise variance (sCMOS). The
default constraints cap x and y steps at 1 pixel and z steps at 0.05 µm; photon and
background steps are not capped (photons ≥ 1, background ≥ 0.01). A parameter on a bound
whose gradient points out of the box is held there while the others take their step.
`iterations` counts Levenberg-Marquardt pixel passes, where the Gaussian models count
Newton-Raphson iterations: the last pass only evaluates, so the fit takes at most
`iterations - 1` steps and `iterations = 1` returns the start. For sCMOS the reported
log-likelihood and p-value use the Gaussian approximation -(d - μ)²/(2(μ + v)), not the
fitted objective. `fit` on a bare `Array` uses this model's own `pixel_size`, not the 0.1 µm
of the Gaussian models.

**Use this model when:**
- You have a calibrated PSF that a Gaussian does not describe well
- You need z-position information from that PSF

## Custom PSF Models

A subtype of `GaussMLE.PSFModel` defined outside this package must be `isbits` after
`Adapt.adapt_structure` to run on GPUs and on the CPU backend of KernelAbstractions 0.10
(POCL), which a fresh environment resolves: the kernel receives every model through Adapt.
The CPU backend of KernelAbstractions 0.9 does not require it. Hold numbers, `SVector`s and
arrays that Adapt moves to the device, and drop host-only fields (a callable, a heap array)
in `adapt_structure`, as `SplinePSFModel` does with its point PSF. `simple_initialize` must
return an `SVector`, as `SplinePSFModel`'s does, not an `MVector`. Otherwise the fit fails
with a `GPUCompiler.KernelError` ("passing non-bitstype argument").

## Model Comparison

| Feature | GaussianXYNB | GaussianXYNBS | GaussianXYNBSXSY | AstigmaticXYZNB | SplinePSFModel |
|---------|--------------|---------------|------------------|-----------------|----------------|
| Parameters | 4 | 5 | 6 | 5 | 5 |
| PSF sigma | Fixed | Fitted | Fitted (x,y) | Calibrated | Tabulated shape |
| Dimensions | 2D | 2D | 2D | 3D | 3D |
| Speed | Fastest | Medium | Slowest | Medium | Table lookup |
| Stability | Most stable | Less stable | Least stable | Stable | Depends on calibration |
| Backend | CPU/GPU | CPU/GPU | CPU/GPU | CPU/GPU | CPU/GPU |

## Unit Convention

**Spatial PSF calibration parameters use physical units (microns):**

- `GaussianXYNB(sigma)`: sigma in microns (e.g., `0.13f0` = 130nm)
- `AstigmaticXYZNB`: sigma_x0, sigma_y0, gamma, d all in microns
- `AstigmaticXYZNB`: Ax, Ay, Bx and By are dimensionless
- `SplinePSFModel`: callable coordinates, z range, and pixel size are in microns

Internally, x, y and lateral widths use pixels based on the camera pixel size; z for both 3D models, and astigmatic gamma and d, remain in microns.

## Choosing the Right Model

### Decision Tree

1. **Do you need 3D localization?**
   - Yes, with an astigmatism calibration -> Use `AstigmaticXYZNB`
   - Yes, with a tabulated or learned PSF -> Use `SplinePSFModel`
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
fitter = GaussMLEConfig(psf_model = psf)

# Fit data
smld, info = fit(data, fitter)

# Extract results
x_positions = [e.x for e in smld.emitters]
precisions = [e.σ_x for e in smld.emitters]

println("Mean precision: $(mean(precisions) * 1000) nm")
```

### Example 2: Variable PSF for Quality Control

```julia
using GaussMLE
using Statistics

# Variable sigma model
psf = GaussianXYNBS()

fitter = GaussMLEConfig(psf_model = psf)
smld, info = fit(data, fitter)

# Analyze PSF width distribution
sigmas = [e.σ for e in smld.emitters]
println("Mean PSF width: $(mean(sigmas)) microns")
println("PSF width std: $(std(sigmas)) microns")

# Filter by PSF width
valid = filter(e -> 0.1 < e.σ < 0.2, smld.emitters)
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

fitter = GaussMLEConfig(psf_model = psf, iterations = 30)
smld, info = fit(data, fitter)

# Extract 3D positions
x = [e.x for e in smld.emitters]
y = [e.y for e in smld.emitters]
z = [e.z for e in smld.emitters]

println("Z range: $(extrema(z)) microns")
println("Mean Z precision: $(mean([e.σ_z for e in smld.emitters]) * 1000) nm")
```

## Mathematical Formulation

The four Gaussian models use the Gaussian expectation with pixel integration:

```math
\mu_{i,j}(\theta) = \theta_{bg} + \theta_N \int_{i-0.5}^{i+0.5} \int_{j-0.5}^{j+0.5}
\frac{1}{2\pi \sigma_x \sigma_y} \exp\left(-\frac{(x-\theta_x)^2}{2\sigma_x^2} - \frac{(y-\theta_y)^2}{2\sigma_y^2}\right) dx \, dy
```

Where:
- For **GaussianXYNB**: sigma_x = sigma_y = sigma (fixed)
- For **GaussianXYNBS**: sigma_x = sigma_y = sigma (fitted)
- For **GaussianXYNBSXSY**: sigma_x, sigma_y (independently fitted)
- For **AstigmaticXYZNB**: sigma_x(z), sigma_y(z) from calibration curve

For `SplinePSFModel`, let `f` be the point PSF in micron coordinates and pₓ, pᵧ the camera pixel sizes.
The table interpolates the pixel average:

```math
A_{p_x,p_y}(d_x,d_y,z) = \frac{1}{p_x p_y}
\int_{d_y-p_y/2}^{d_y+p_y/2} \int_{d_x-p_x/2}^{d_x+p_x/2}
\max(0, f(u,v,z))\,du\,dv.
```

With `table` the natural cubic interpolant of these averages, the expected count is:

```julia
μ = bg + N * px_scale * max(0, table((j-x)*p_x, (i-y)*p_y, clamp(z, z_range...)))
```

For a `MicroscopePSFs.AbstractPSF` photon density, `px_scale = p_x * p_y`, so the signal is N times
the pixel integral. For a plain callable, `px_scale` is the reciprocal sum of pixel averages
on the reference patch at `z_ref`, so N counts the photons in that patch on the reference
plane. The final clipping removes negative interpolation overshoot.

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
fitter_fixed = GaussMLEConfig(psf_model = GaussianXYNB(0.13f0))
fitter_var = GaussMLEConfig(psf_model = GaussianXYNBS())

smld_fixed, info_fixed = fit(data, fitter_fixed)
smld_var, info_var = fit(data, fitter_var)

# Compare position estimates
x_fixed = [e.x for e in smld_fixed.emitters]
x_var = [e.x for e in smld_var.emitters]

println("Position RMS difference: $(sqrt(mean((x_fixed .- x_var).^2)) * 1000) nm")
```
