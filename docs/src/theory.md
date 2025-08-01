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

### Newton-Raphson Optimization

The MLE estimate is found by solving:

```math
\frac{\partial \ell(\theta)}{\partial \theta} = 0
```

GaussMLE.jl uses Newton-Raphson iteration:

```math
\theta^{(k+1)} = \theta^{(k)} - H^{-1}(\theta^{(k)}) \nabla \ell(\theta^{(k)})
```

Where:
- $\nabla \ell(\theta)$ is the gradient (first derivatives)
- $H(\theta)$ is the Hessian matrix (second derivatives)

## Derivatives

### Gradient Components

For the GaussXyNb model ($\theta = [x, y, n, b]$):

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

The Hessian elements are:

```math
\frac{\partial^2 \ell}{\partial \theta_i \partial \theta_j} = -\sum_{k,l} \left[ \frac{N_{k,l}}{\mu_{k,l}^2} \frac{\partial \mu_{k,l}}{\partial \theta_i} \frac{\partial \mu_{k,l}}{\partial \theta_j} - \left( \frac{N_{k,l}}{\mu_{k,l}} - 1 \right) \frac{\partial^2 \mu_{k,l}}{\partial \theta_i \partial \theta_j} \right]
```

## Gaussian Integral Computation

### Analytical Integration

The key computational challenge is evaluating:

```math
G_{i,j}(\theta_x, \theta_y, \sigma) = \int_{i-0.5}^{i+0.5} \int_{j-0.5}^{j+0.5} 
\frac{1}{2\pi\sigma^2} \exp\left(-\frac{(x-\theta_x)^2 + (y-\theta_y)^2}{2\sigma^2}\right) dx \, dy
```

This separates as:

```math
G_{i,j} = G_x(i, \theta_x, \sigma) \cdot G_y(j, \theta_y, \sigma)
```

Where:

```math
G_x(i, \theta_x, \sigma) = \int_{i-0.5}^{i+0.5} \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\theta_x)^2}{2\sigma^2}\right) dx
```

### Error Function Implementation

Using the error function:

```math
G_x(i, \theta_x, \sigma) = \frac{1}{2} \left[ \text{erf}\left(\frac{i+0.5-\theta_x}{\sqrt{2}\sigma}\right) - \text{erf}\left(\frac{i-0.5-\theta_x}{\sqrt{2}\sigma}\right) \right]
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

### Cramér-Rao Lower Bound

The CRLB gives the minimum achievable standard deviation:

```math
\text{CRLB}(\theta_i) = \sqrt{[I^{-1}(\theta)]_{ii}}
```

GaussMLE.jl computes this at the MLE estimate to provide uncertainty estimates.

## Model Extensions

### Variable PSF Width (GaussXyNbS)

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

For sCMOS cameras with pixel-dependent variance $\text{Var}_{i,j}$:

```math
N_{i,j} \sim \mathcal{N}(\mu(\theta)_{i,j}, \text{Var}_{i,j})
```

The log-likelihood becomes:

```math
\ell(\theta) = -\frac{1}{2} \sum_{i,j} \left[ \frac{(N_{i,j} - \mu(\theta)_{i,j})^2}{\text{Var}_{i,j}} + \log(2\pi \text{Var}_{i,j}) \right]
```

## Numerical Considerations

### Convergence Criteria

Newton-Raphson iteration continues until:

1. **Gradient norm**: $\|\nabla \ell(\theta)\| < \epsilon_g$
2. **Parameter change**: $\|\Delta \theta\| < \epsilon_p$
3. **Likelihood change**: $|\Delta \ell| < \epsilon_\ell$

### Stability Measures

To ensure numerical stability:

- **Hessian regularization**: Add small diagonal terms if poorly conditioned
- **Step limiting**: Restrict parameter updates to reasonable ranges
- **Boundary conditions**: Enforce physical constraints (positive intensity, etc.)

### Computational Complexity

For an $N \times N$ pixel region:
- **Gradient evaluation**: $O(N^2)$ operations
- **Hessian evaluation**: $O(N^2)$ operations  
- **Matrix inversion**: $O(p^3)$ where $p$ is number of parameters
- **Overall per iteration**: $O(N^2 + p^3)$

Since $p \ll N$ for typical problems ($p \leq 5$, $N \sim 5-13$), the complexity is dominated by the $O(N^2)$ pixel operations.

## References

1. Smith, C. S., et al. "Fast, single-molecule localization that achieves theoretically minimum uncertainty." *Nature Methods* 7.5 (2010): 373-375.

2. Mortensen, K. I., et al. "Optimized localization analysis for single-molecule tracking and super-resolution microscopy." *Nature Methods* 7.5 (2010): 377-381.

3. Rieger, B., and S. Stallinga. "The lateral and axial localization uncertainty in super-resolution light microscopy." *ChemPhysChem* 15.4 (2014): 664-670.