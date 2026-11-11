# Theory

This section provides the mathematical foundation for Gaussian Maximum Likelihood Estimation as implemented in GaussMLE.jl.

## Problem Formulation

Single Molecule Localization Microscopy (SMLM) aims to precisely determine the position and intensity of individual fluorescent molecules from noisy images. Each molecule appears as a diffraction-limited spot approximated by a 2D Gaussian function.

### Image Formation Model

The expected photon count at pixel $(i,j)$ is given by:

```math
\mu(\theta)_{i,j} = \theta_b + \theta_n \int_{i-0.5}^{i+0.5} \int_{j-0.5}^{j+0.5} 
\frac{1}{2\pi\sigma^2} \exp\left(-\frac{(x-\theta_x)^2 + (y-\theta_y)^2}{2\sigma^2}\right) dx \, dy
```

Where:
- $\theta_x, \theta_y$: molecule position (pixels)
- $\theta_n$: integrated intensity (photons)
- $\theta_b$: background level (photons/pixel)
- $\sigma$: PSF width (pixels)

The integral accounts for pixel integration effects, ensuring accurate modeling of the discrete sampling process.

### Noise Model

Photon detection follows Poisson statistics:

```math
P(N_{i,j} = k) = \frac{\mu(\theta)_{i,j}^k e^{-\mu(\theta)_{i,j}}}{k!}
```

Where $N_{i,j}$ is the observed photon count at pixel $(i,j)$.

## Maximum Likelihood Estimation

### Likelihood Function

For an image region with pixels indexed by $(i,j)$, the likelihood function is:

```math
L(\theta) = \prod_{i,j} \frac{\mu(\theta)_{i,j}^{N_{i,j}} e^{-\mu(\theta)_{i,j}}}{N_{i,j}!}
```

### Log-Likelihood

The log-likelihood simplifies to:

```math
\ell(\theta) = \sum_{i,j} \left[ N_{i,j} \log \mu(\theta)_{i,j} - \mu(\theta)_{i,j} - \log(N_{i,j}!) \right]
```

The constant term $\log(N_{i,j}!)$ can be dropped for optimization.

### Levenberg-Marquardt Optimization

The MLE estimate is found by solving:

```math
\frac{\partial \ell(\theta)}{\partial \theta} = 0
```

Every model is fitted by Levenberg-Marquardt on the full Fisher matrix. At the current
point $\theta$, with $\mathrm{Var}_{i,j} = \mu_{i,j}$ (IdealCamera) or
$\mu_{i,j} + \sigma^2_{\text{read},i,j}$ (sCMOS), the gradient and Fisher matrix are

```math
g_a = \sum_{i,j} \frac{N_{i,j} - \mu_{i,j}}{\mathrm{Var}_{i,j}} \frac{\partial \mu_{i,j}}{\partial \theta_a},
\qquad
F_{ab} = \sum_{i,j} \frac{1}{\mathrm{Var}_{i,j}} \frac{\partial \mu_{i,j}}{\partial \theta_a} \frac{\partial \mu_{i,j}}{\partial \theta_b}
```

and the trial step solves

```math
(F + \lambda\,\mathrm{diag}\,F)\,\Delta = g, \qquad \theta' = \theta + \Delta
```

with the step held to its step limits (below) and $\theta'$ clamped to the bounds. Because $F$
is the expected curvature rather than the observed one, this is the Levenberg-Marquardt form of
Fisher scoring. The objective is the
Poisson log-likelihood (IdealCamera) or the Poisson-shifted likelihood
$(N + v)\log(\mu + v) - (\mu + v)$ with $v$ the read-noise variance (sCMOS); $g$ is its exact
gradient for both. Every model, `SplinePSFModel` included, follows the stop and damping
rules below. $\lambda$ starts at $10^{-3}$. $F$ is the expected negative
Hessian (below), positive semidefinite, so a small enough step along the damped solution raises
the likelihood. A parameter that sits on a bound with the gradient pointing out of the box is
held there: its row and column of the damped matrix are replaced by the identity and its
gradient set to zero, so the solve moves only the free parameters.

Every model runs to convergence (each start, for a model with several):

- **Stop.** A fit stops once the gain the undamped Fisher-scoring step predicts,
  $g^\top F^{-1} g / 2$ over the free parameters, is below $10^{-4}$. $F^{-1}$ is the CRLB
  covariance at $\theta$, so $g^\top F^{-1} g$ is the squared distance, in CRLB units, from
  $\theta$ to the maximum of the quadratic model with curvature $F$: the fit stops within
  $\sqrt{2 \times 10^{-4}} \approx 0.014$ CRLB of it. The likelihood's own maximum sits where
  the observed curvature puts it, which differs from $F$ on noisy data. Where the observed
  information is a small fraction of $F$ (z at low counts) or the likelihood has more than
  one maximum (an emitter beyond the z range), a fit can stop farther from it: in
  simulations of every Gaussian model that include such cases (156,000 fits), 103 stopped
  more than 0.1 CRLB from the likelihood's maximum, nearly all at low counts or beyond the
  z range. The Long test group checks this against an independent Float64 maximizer of the
  same objective, started from the fit and from the truth: in 64 conditions (the four
  Gaussian models, both cameras, 500 to 100,000 photons, 2,000 simulated fits each), at
  least 99.95 % of the fits that pass the usual filter end within 0.1 CRLB of the higher
  maximum.
- **Damping.** On a kept trial, $\lambda$ follows the ratio $\rho$ of the actual to the
  predicted gain of the step (Nielsen's rule): it is multiplied by
  $\max(1/3,\ 1 - (2\rho - 1)^3)$. A discarded trial multiplies $\lambda$ by $\nu$, which
  starts at 2 and doubles with each discard in a row. Where the observed curvature exceeds
  $F$, as for z at low counts, the undamped step overshoots; this damps it without
  switching the damping back and forth between passes.
- **Gains below the objective's rounding.** When a trial changes the objective by less than
  10 times the Float32 rounding of its sum, the gain is taken from the gradients instead, as
  the trapezoid $(g_\mathrm{kept} + g)^\top s / 2$ of the step $s$, exact for a quadratic
  along the step, and the trial is kept if that gain is positive. A bright emitter otherwise
  stalls on gains its objective cannot resolve. A trial whose objective is not finite is
  discarded.
- **Step limits.** The whole step is scaled so that no parameter exceeds its step limit,
  which keeps its direction; clamping one component alone breaks the coupled step.
- **Fits that end unconverged.** A fit ends without meeting the stop test when it reaches
  the `iterations` cap, or when its next step rounds to its current point in Float32 (the
  objective's rounding then exceeds the remaining gain, as for very bright or very narrow
  PSFs; once the damping reaches its cap every later pass makes that same step). Its uncertainties stay the CRLB at its end
  point, and `GaussMLEFitInfo.n_converged` does not count it. The stop test measures distance
  with $F$, so such a fit is often at its maximum all the same: in the simulations above
  about half of the capped fits were within 0.1 CRLB of it.

Every model uses these rules. One fit climbs to the maximum of the basin it starts in, so
two models fit from more than one start and keep the one with the highest objective; the
fit counts as converged when that start did. `AstigmaticXYZNB` fits a second time from the
first fit's end point with z negated: its likelihood has a second maximum near the mirrored
z, where the two widths swap, and with weak astigmatism (a focal offset of 0.05 µm against a
depth scale of 0.4 µm) a single start ended there for 1 to 12 % of in-range emitters at 300
to 1,000 photons, with z of the wrong sign; the second start removes those, at 2 to 3.5 times
the fit time. `SplinePSFModel` fits from the scan start and from `starts - 1` fixed z
starts (see the model guide). `iterations` caps the pixel passes of every start (see
Iteration Strategy).

The full Fisher matrix at the final $\theta$ is inverted once for uncertainties. The
p-value uses the Pearson statistic for every model (see Goodness of Fit).

## Derivatives

### Gradient Components

For the GaussianXYNB model ($\theta = [x, y, n, b]$):

```math
\frac{\partial \ell}{\partial \theta_x} = \sum_{i,j} \left( \frac{N_{i,j}}{\mu_{i,j}} - 1 \right) \frac{\partial \mu_{i,j}}{\partial \theta_x}
```

```math
\frac{\partial \ell}{\partial \theta_y} = \sum_{i,j} \left( \frac{N_{i,j}}{\mu_{i,j}} - 1 \right) \frac{\partial \mu_{i,j}}{\partial \theta_y}
```

```math
\frac{\partial \ell}{\partial \theta_n} = \sum_{i,j} \left( \frac{N_{i,j}}{\mu_{i,j}} - 1 \right) \frac{\partial \mu_{i,j}}{\partial \theta_n}
```

```math
\frac{\partial \ell}{\partial \theta_b} = \sum_{i,j} \left( \frac{N_{i,j}}{\mu_{i,j}} - 1 \right)
```

### Hessian Components

The exact Poisson Hessian elements are:

```math
\frac{\partial^2 \ell}{\partial \theta_i \partial \theta_j} = -\sum_{k,l} \left[ \frac{N_{k,l}}{\mu_{k,l}^2} \frac{\partial \mu_{k,l}}{\partial \theta_i} \frac{\partial \mu_{k,l}}{\partial \theta_j} - \left( \frac{N_{k,l}}{\mu_{k,l}} - 1 \right) \frac{\partial^2 \mu_{k,l}}{\partial \theta_i \partial \theta_j} \right]
```

Its expectation over the data ($\mathrm{E}[N_{k,l}] = \mu_{k,l}$) drops the second-derivative
term and is $-F$, the Fisher matrix of the optimizer above. The fit uses $F$ in place of the
Hessian: it needs only first derivatives, it is positive semidefinite at every point (the Hessian
is not, away from the maximum), and at the maximum its inverse is the CRLB.

## Gaussian Integral Computation

### Analytical Integration

The key computational challenge is evaluating:

```math
G_{i,j}(\theta_x, \theta_y, \sigma) = \int_{i-0.5}^{i+0.5} \int_{j-0.5}^{j+0.5} 
\frac{1}{2\pi\sigma^2} \exp\left(-\frac{(x-\theta_x)^2 + (y-\theta_y)^2}{2\sigma^2}\right) dx \, dy
```

For array pixel `[i,j] = [y,x]`, this separates as:

```math
G_{i,j} = G_x(j, \theta_x, \sigma) \cdot G_y(i, \theta_y, \sigma)
```

Where:

```math
G_x(j, \theta_x, \sigma) = \int_{j-0.5}^{j+0.5} \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\theta_x)^2}{2\sigma^2}\right) dx
```

### Error Function Implementation

Using the error function:

```math
G_x(j, \theta_x, \sigma) = \frac{1}{2} \left[ \text{erf}\left(\frac{j+0.5-\theta_x}{\sqrt{2}\sigma}\right) - \text{erf}\left(\frac{j-0.5-\theta_x}{\sqrt{2}\sigma}\right) \right]
```

## Uncertainty Estimation

### Fisher Information Matrix

The Fisher Information Matrix (FIM) provides the theoretical lower bound on parameter uncertainties:

```math
I_{ij}(\theta) = E\left[ -\frac{\partial^2 \ell}{\partial \theta_i \partial \theta_j} \right]
```

For the Poisson model:

```math
I_{ij}(\theta) = \sum_{k,l} \frac{1}{\mu_{k,l}} \frac{\partial \mu_{k,l}}{\partial \theta_i} \frac{\partial \mu_{k,l}}{\partial \theta_j}
```

The Fisher matrix is positive semidefinite; its off-diagonal entries may have either sign.

### Cramér-Rao Lower Bound

The CRLB gives the minimum achievable standard deviation:

```math
\text{CRLB}(\theta_i) = \sqrt{[I^{-1}(\theta)]_{ii}}
```

GaussMLE.jl computes this at the MLE estimate to provide uncertainty estimates.

### Parameter Covariances

The off-diagonal elements of the inverse Fisher matrix provide covariance information between parameters:

```math
\text{Cov}(\theta_i, \theta_j) = [I^{-1}(\theta)]_{ij} \quad (i \neq j)
```

Let `C` be the internal inverse Fisher matrix and `pₓ`, `pᵧ` the camera pixel sizes in microns. GaussMLE.jl converts the spatial covariances to microns²:

- **2D models**: `σ_xy = pₓ * pᵧ * C[1,2]`
- **3D models** (`AstigmaticXYZNB` and `SplinePSFModel`): additionally `σ_xz = pₓ * C[1,3]` and `σ_yz = pᵧ * C[2,3]`, since z already uses microns

These covariances capture correlations between position estimates. For example, a non-zero `σ_xy` indicates that the x and y position estimates are correlated, which can arise from asymmetric pixel sampling or off-center emitter positions within the ROI.

## Model Extensions

### Variable PSF Width (GaussianXYNBS)

For the 5-parameter model with variable PSF width:

```math
\theta = [x, y, n, b, \sigma]
```

Additional derivatives include:

```math
\frac{\partial \mu_{i,j}}{\partial \sigma} = \theta_n \frac{\partial G_{i,j}}{\partial \sigma}
```

Where:

```math
\frac{\partial G_{i,j}}{\partial \sigma} = \frac{\partial G_x}{\partial \sigma} G_y + G_x \frac{\partial G_y}{\partial \sigma}
```

### sCMOS Noise Model

For sCMOS cameras, each pixel has independent readout noise characterized by a per-pixel variance $\sigma^2_{\text{read},i,j}$. The total noise variance at each pixel is the sum of Poisson shot noise and readout noise:

```math
\text{Var}_{i,j} = \mu(\theta)_{i,j} + \sigma^2_{\text{read},i,j}
```

The fit maximizes the Poisson-shifted likelihood $(N + v)\log(\mu + v) - (\mu + v)$, with
$v = \sigma^2_{\text{read}}$: it treats $N + v$ as Poisson with mean $\mu + v$, which has the
right mean and variance. Its gradient is the weighted residual
$(N_{i,j} - \mu_{i,j}) / \text{Var}_{i,j}$ times $\partial\mu_{i,j}/\partial\theta$, and its Fisher
matrix weights each pixel by $1/\text{Var}_{i,j}$ (see Levenberg-Marquardt Optimization).

The reported residual statistic is:

```math
R(\theta) = -\frac{1}{2} \sum_{i,j} \frac{(N_{i,j} - \mu(\theta)_{i,j})^2}{\text{Var}_{i,j}}
```

It is reported, not maximized. Here the data and model use electrons, and the readout variance uses electrons².

Note that unlike the pure Poisson case, the variance depends on both the model value (signal-dependent) and the camera calibration (pixel-dependent). The readout variance map is extracted from the camera calibration as $\sigma^2_{\text{read}} = \text{readnoise}^2$ and indexed at the correct camera coordinates using the ROI corner positions.

## Goodness of Fit

The `pvalue` of each localization is the upper tail of $\chi^2(\mathrm{df})$ at the Pearson statistic

```math
X^2 = \sum_{i,j:\,\mathrm{Var}_{i,j} > 0} \frac{(N_{i,j} - \mu(\hat\theta)_{i,j})^2}{\text{Var}_{i,j}},
\qquad \mathrm{df} = n_{\text{pixels}} - n_{\text{params}}
```

with $\text{Var}_{i,j} = \mu_{i,j} + \sigma^2_{\text{read},i,j}$ (the readout term is zero for `IdealCamera`), the same statistic for both cameras and every PSF model. The Poisson deviance, $-2$ times the log-likelihood ratio of `IdealCamera`, is not $\chi^2$-distributed at low counts: at 1 to 2 photons per pixel a 0.05 cut on its p-value rejected 17 to 32 % of correct fits. The Pearson statistic is calibrated there (0.045 to 0.061 of correct fits below 0.05 in the 24 conditions measured on SMLMSim boxes), and rejects wrong-width and two-emitter ROIs as often as a moment-matched deviance. The p-value is computed with `ccdf`, so a poor fit does not round to exactly zero in Float32.

## Numerical Considerations

### Iteration Strategy

`iterations` caps the Levenberg-Marquardt pixel passes of a fit: by default 100, and most fits
converge in far fewer. Each pass
evaluates the objective, gradient and Fisher matrix at one trial point; the first evaluates the
start and the last only evaluates, so a fit takes at most `iterations - 1` steps and
`iterations = 1` returns the start. A fit stops before the cap once converged (see
Levenberg-Marquardt Optimization); for a model fitted from several starts the cap applies to
each start. Photon and
background steps have no default step limit: the damped step and the likelihood check keep them
in place, where the diagonal Newton-Raphson update of GaussMLE 0.4.2 and earlier limited them to
100 photons and 2 background photons per iteration and fell short of the maximum for bright
emitters.

### Stability Measures

To ensure numerical stability:

- **Parameter constraints**: Restrict parameter updates with step limits and enforce bounds (positive intensity, bounded position, etc.)
- **Fisher matrix scaling**: The Fisher Information Matrix is positive semidefinite and its entries span many orders of magnitude (the position entries grow like $N$, the photon entry shrinks like $1/N$). It is inverted after Jacobi scaling, $G = D I D$ with $D = \mathrm{diag}(I_{kk}^{-1/2})$, so $G$ has a unit diagonal, and $I^{-1} = D G^{-1} D$. Nothing is added to the diagonal, so the photon uncertainty stays at the Cramér-Rao bound for bright emitters
- **Cholesky decomposition**: $G$ is inverted by Cholesky only. A Cholesky pivot (the squared diagonal of the factor) at or below $n_\mathrm{pix}\,\epsilon$, with $n_\mathrm{pix}$ the number of pixels of the box and $\epsilon$ the machine epsilon of the working precision, means $G$ is singular to working precision: the accumulation over the pixels perturbs the entries of $G$ by about that much, so a smaller pivot cannot be told from that of an exactly singular matrix; there is no LU fallback, since $I$ is positive semidefinite by construction and LU there only turns rounding into a number. If a diagonal entry of $I$ is not positive and finite, the Cholesky fails, or a diagonal entry of $I^{-1}$ is not positive and finite, the fitted parameters are retained with infinite uncertainties and zero covariances
- **Fits that did not converge**: A fit that ended without meeting its stop test (see Levenberg-Marquardt Optimization) is left out of `GaussMLEFitInfo.n_converged`; its parameters, uncertainties and p-value are those of its last kept trial. A model fitted from several starts counts as converged when the start it keeps did
- **Fits on a bound**: A fit whose x or y (and z, for `AstigmaticXYZNB` and `SplinePSFModel`) ends exactly on a lower or upper bound of its constraints did not localize an emitter inside the allowed region. After the kernel, on the host, it reports infinite uncertainties and zero covariances as well; its parameters and p-value are unchanged. A parameter the caller holds fixed (equal lower and upper bounds) is not a bound hit

With `constraints=nothing`, defaults are built at fit time from ROI size n, and the config retains `nothing`. Default x and y bounds are `[-1,n+2]` in ROI pixels, extending 1.5 pixels beyond each ROI edge.

### Computational Complexity

For an $N \times N$ pixel region with $p$ parameters:

- **Objective, gradient and Fisher matrix per pass**: $O(N^2p^2)$ operations
- **Final Fisher matrix accumulation**: $O(N^2p^2)$ operations
- **Final Fisher matrix inversion**: $O(p^3)$ operations

The models have at most six parameters ($p \leq 6$, typically $N \sim 5-13$). Passes cost $O(N^2p^2 + p^3)$ each; final uncertainty estimation costs $O(N^2p^2 + p^3)$ once after optimization.

## References

1. Smith, C. S., et al. "Fast, single-molecule localization that achieves theoretically minimum uncertainty." *Nature Methods* 7.5 (2010): 373-375.

2. Mortensen, K. I., et al. "Optimized localization analysis for single-molecule tracking and super-resolution microscopy." *Nature Methods* 7.5 (2010): 377-381.

3. Rieger, B., and S. Stallinga. "The lateral and axial localization uncertainty in super-resolution light microscopy." *ChemPhysChem* 15.4 (2014): 664-670.
