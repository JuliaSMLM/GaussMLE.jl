```@meta
CurrentModule = GaussMLE
```

# GaussMLE

Documentation for [GaussMLE](https://github.com/JuliaSMLM/GaussMLE.jl).

## Overview
*GaussMLE* provides a fast method for performing Maximum Likelihood Estimation (MLE) of Gaussian blob parameters under a Poisson noise model.  The form of the Gaussian expectation models are:

$$\mu(\theta)_{i,j} = b + n \int_{i-0.5}^{i+0.5} \int_{j-0.5}^{j+0.5} 
\mathcal{N}(\theta_x - j, \sigma_x^2)
\mathcal{N}(\theta_y - i, \sigma_y^2)$$

- $ x,y $: Location of blob center
- $ n $: Integrated intenstity in the blob (does not include background)
- $ b $: Background counts per pixel
- $ \sigma $: Width of Gaussian blob   
- $ i,j $: Pixel location (column major)

The models differ by how they treat $\sigma_x$ and $\sigma_y$.  

| Symbol | $\theta^\top$ | $\sigma_x, \sigma_y$ |
|--------|----------|---------------------|
|:xynb   | $\{x,y,n,b\}$ | $\sigma_x = \sigma_y$ = $\sigma$_PSF |
|:xynbs   | $\{x,y,n,b,\sigma\_\mathrm{PSF}\}$ | $\sigma_x = \sigma_y = \theta_{\sigma\_\mathrm{PSF}}$|
|:xynbsxsy | $\{x,y,n,b,\sigma\_\mathrm{x},\sigma\_\mathrm{y}\}$ | $\sigma_x = \theta_{\sigma\_\mathrm{x}}, \sigma_y = \theta_{\sigma\_\mathrm{y}}$|
|:xynbz | $\{x,y,n,b,z\}$ | $\sigma_x = \sigma\_\mathrm{x}(\theta_z) , \sigma_y = \sigma\_\mathrm{y}(\theta_z)$|

## Usage



