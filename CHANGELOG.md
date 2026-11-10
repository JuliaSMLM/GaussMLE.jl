# Changelog

All notable changes to GaussMLE.jl are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Before 1.0, a release that raises
the last number (0.4.Z) is non-breaking; one that raises the middle number (0.Y.0) may break.
Releases before this file are described on the GitHub releases page.

## [Unreleased]

### Changed: what existing settings and outputs mean

The way `fit` is called is unchanged, but these settings and outputs change their meaning or
default, and the fitted values, uncertainties and p-values of every model change.

- **Gaussian models fitted to convergence.** The four Gaussian models use Levenberg-Marquardt
  on the full Fisher matrix (Fisher scoring) in place of 20 diagonal Newton-Raphson steps
  capped at 100 photons and 2 background photons per step, which stopped short of the
  maximum for bright, defocused and low-count emitters. A fit stops once the gain the
  undamped step predicts, g' F^-1 g / 2, is below 1e-4, within 0.014 CRLB of the maximum of
  the quadratic model. The damping follows the ratio of actual to predicted gain (Nielsen's
  rule), a gain too small for the Float32 objective to resolve is taken from the gradients
  (the trapezoid along the step), and the whole step is scaled to its limits.
- `iterations` caps the Levenberg-Marquardt pixel passes of a fit (of each start for
  `AstigmaticXYZNB` and `SplinePSFModel`): `iterations = k` takes at most k - 1 steps and
  `iterations = 1` returns the start, where 0.4.2 took exactly k steps. Its default is 100;
  a fit stops earlier once converged. `iterations = 20`, the 0.4.2 default that scripts
  and packages often pass explicitly, is now a cap that some fits reach before converging:
  on simulated low-count astigmatic ROIs (300 to 500 photons) up to 5 % of the fits need
  more than 20 passes. Drop an explicit `iterations = 20` and take the default.
- `GaussMLEFitInfo.n_converged` counts the fits that met the convergence test (it equaled
  `n_fits` before). A fit that reaches the cap keeps its CRLB. A model fitted from several
  starts counts as converged when the start it keeps did.
- The default step limits no longer cap photon and background steps, and a user `max_step`
  scales the whole step instead of clamping each component.
- p-values come from the Pearson statistic for both cameras; the IdealCamera p-value used
  the Poisson deviance, which is not chi-squared at low counts.
- A fit whose x or y (or z) ends on a bound, or whose Fisher matrix is singular to working
  precision, reports infinite uncertainties and zero covariances.
- The CRLB is the inverse of the Jacobi-scaled Fisher matrix with no absolute regularizer
  (which made the photon CRLB too small for bright emitters).
- `fit(::Array)` reports each emitter's x and y within its own ROI (microns from the ROI's
  top-left edge). It placed ROI k (k - 1) boxes along x before, where Float32 lost position
  precision after tens of thousands of ROIs.
- `AstigmaticXYZNB` fits each ROI a second time from the first fit's end point with z
  negated and keeps the higher likelihood. With weak astigmatism a single start ended in
  the mirror-z maximum, z of the wrong sign, for 1 to 12 % of in-range simulated emitters at
  300 to 1,000 photons. The second start removes those, at about twice the fit time.
- Julia 1.10 is no longer supported; the oldest supported Julia is 1.11.

### Added

- **AMD GPUs.** `backend = :gpu`, and `:auto` when CUDA is not functional, fit on an AMD
  GPU when AMDGPU is loaded (weak dependency, extension `GaussMLEAMDGPUExt`). AMD launches
  use work-groups of 64; CUDA launches are unchanged.
- **KernelAbstractions 0.10.** The compat range is 0.9.38 to 0.10. Under 0.10 the CPU runs
  the same kernel through POCL (in work-groups of 64); under 0.9 natively.
- **SplinePSFModel**: a tabulated 3D PSF, from any callable or from a MicroscopePSFs PSF
  through a weak-dependency extension, fitted on the CPU and GPU from one table of pixel
  integrals, with rectangular pixels and analytic spline derivatives, by the same
  Levenberg-Marquardt fit (Poisson objective; Poisson-shifted for sCMOS). By default it also
  fits from fixed z starts, one per 0.4 um of the central 80 % of `z_range`, and keeps the
  highest likelihood.
- `fit(Array; variance_map)` accepts a box x box or box x box x n map.
- A precompile workload compiles the first CPU fit of `GaussianXYNBS` and `SplinePSFModel`
  for both cameras.

### Fixed

- A parameter fixed by equal bounds is always held out of the step (before, one with no
  information in the data could stop every parameter from moving).
- A fit whose background may be zero no longer gets a NaN objective from pixels with no
  counts.
- A read-noise variance map, or a camera's read noise, must be finite and non-negative;
  `fit` throws an `ArgumentError` where the kernel threw or returned nonsense. So must a
  photon or background lower bound in user constraints be at least 0; a negative one could
  make the fit take the log of a negative number, which threw on the CPU.
- `generate_roi_batch` on an sCMOS camera reads its calibration maps as (y, x), as the
  camera and the fit do; before, it took them at swapped coordinates.
- GPU retries: only resource errors are retried; an out-of-memory error halves the batch;
  any other GPU error rethrows under `:gpu` and falls back to the CPU under `:auto`.

### Removed

- The kernel's log-likelihood output, which no function returned, and the internal
  `GaussMLEResults` type, which nothing constructed.

### Notes

- A custom `PSFModel` must be isbits after Adapt to run on the CPU (POCL).
- CPU results under KernelAbstractions 0.9 and 0.10, and CPU and GPU results, agree to the
  stop tolerance (about 0.02 CRLB), not to rounding.
- The first fit in a session of a model outside the precompile workload compiles its
  kernel (seconds).
