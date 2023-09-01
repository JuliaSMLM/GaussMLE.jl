```@meta
CurrentModule = GaussMLE
```

# GaussMLE

Documentation for [GaussMLE](https://github.com/JuliaSMLM/GaussMLE.jl).

## Overview
*GaussMLE* provides a fast method for performing Maximum Likelihood Estimation (MLE) of Gaussian blob parameters under a Poisson noise model.  The form of the Gaussian expectation models are:

$$\mu(\theta)_{i,j} = \theta_b + \theta_n \int_{i-0.5}^{i+0.5} \int_{j-0.5}^{j+0.5} 
\mathcal{N}(\theta_x - x, \sigma_x^2)
\mathcal{N}(\theta_y - y, \sigma_y^2)dxdy$$


- $ \theta_n $: Integrated intenstity in the blob (does not include background)
- $ {\theta\_x, \theta\_y} $: Location of blob center
- $ \theta_n $: Integrated intenstity in the blob (does not include background)
- $ \theta_b $: Background counts per pixel
- $ \sigma\_x, \sigma\_y $: Width of Gaussian blob   
- $ i,j $: Pixel location (column major)

The models differ by how they treat $\sigma_x$ and $\sigma_y$.  

| Symbol | $\theta^\top$ | $\sigma_x, \sigma_y$ |
|--------|----------|---------------------|
|:xynb   | $\{x,y,n,b\}$ | $\sigma_x = \sigma_y$ = $\sigma$_PSF |
|:xynbs   | $\{x,y,n,b,\sigma\_\mathrm{PSF}\}$ | $\sigma_x = \sigma_y = \theta_{\sigma\_\mathrm{PSF}}$|
|:xynbsxsy | $\{x,y,n,b,\sigma\_\mathrm{x},\sigma\_\mathrm{y}\}$ | $\sigma_x = \theta_{\sigma\_\mathrm{x}}, \sigma_y = \theta_{\sigma\_\mathrm{y}}$|
|:xynbz | $\{x,y,n,b,z\}$ | $\sigma_x = \sigma\_\mathrm{x}(\theta_z) , \sigma_y = \sigma\_\mathrm{y}(\theta_z)$|

## Usage

### Basic Usage

A basic example that demonstrates how to use `GaussFit.fitstack` to fit Gaussian blobs in a stack of 2D image boxes:

```julia
using GaussMLE
using Statistics 

# Simulate a stack of boxes with Poisson noise
T = Float32 # Data type
boxsz = 7 # Box size
nboxes = Int(1e5) # Number of boxes
modeltype = :xynb # Fit model type 
out, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynb; T=T, poissonnoise=true)

# Fit all boxes in the stack
θ_found, Σ_found = GaussMLE.GaussFit.fitstack(out, modeltype, args);

```



