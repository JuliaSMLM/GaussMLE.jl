"""
    GaussModel

The `GaussModel` module offers a unified framework for defining, manipulating, 
and fitting Gaussian models to data. It prescribes a set of abstract types, 
concrete model types, and function signatures that are expected for fitting and simulation codes.

### Abstract Types Hierarchy

1. `GaussMLEParams{T}`: An abstract type for the parameters of a Gaussian model.
2. `GaussMLEArgs{T}`: An abstract type for model-specific arguments, such as non-fitted parameters.
3. `GaussMLEΣ{T}`: An abstract type for representing the uncertainty associated with a Gaussian model.

### Required Methods

Each concrete Gaussian model, such as `:xynb`, must implement the following methods:

- `model(θ, args, i, j)`: Calculates the model's predicted intensity at pixel locations `(i, j)` based on the parameters `θ` and additional arguments `args`.
- `gradient!(θ, args, out, i, j)`: Computes and stores the gradient of the model with respect to its parameters at pixel locations `(i, j)` in the `out` array.
- `curvature!(θ, args, out, i, j)`: Computes and stores the diagonal elements of the model's Hessian matrix (curvature) at pixel locations `(i, j)` in the `out` array.
- `initialize_parameters!(θ, data, boxsize, args)`: Initializes the parameters of the model type based on the provided 2D data matrix, box size, and additional arguments `args`.
- `update!(θ, numerator, denominator)`: Updates the model parameters based on provided numerator and denominator arrays, typically sums of gradients and curvatures over data points.
- `genargs(::Type{θ}, args)`: Returns an instance of the model-specific `GaussMLEArgs` type.
- `genθ(::Type{θ}, boxsize)`: Returns an instance of the `θ_xynbs` type, initialized based on the given `boxsize`.
- `genΣ(::Type{θ}): Returns an instance of the `Σ_xynbs` type, initialized with default values.

# Notes
-  All function are Parametric in the type `T`, which is expected to be a subtype of `Real`.

### Concrete Models

- `xynb`: A 2D Gaussian model that includes parameters for the x and y coordinates, sum intensity, and background. 
- `xynbs`: An extended version of `xynb` that also includes an additional term for the PSF width, σ_PSF.

### Internal Constants

- `MODEL_MAP`: A dictionary that associates model symbols with their corresponding concrete types, facilitating internal model selection and mapping.

"""
module GaussModel

using GaussMLE.GaussLib

#
# Each model must define these methods for thier type of model
export model, gradient!, curvature!, genargs, genθ, genΣ, initialize_parameters!, update!

# Each model must define a type inheriting from these abstract types and export
export GaussMLEParams, GaussMLEArgs, GaussMLEΣ

# Exports for the xynb model 
export θ_xynb, Args_xynb, Σ_xynb
export θ_xynbs, Args_xynbs, Σ_xynbs

# Used internally to map a symbol to a model type.
export MODEL_MAP

include("types.jl")
include("gauss_xynb.jl")
include("gauss_xynbs.jl")

# Mapping from a symbol to the new model type. Add new models here
const MODEL_MAP = Dict(
    :xynb => θ_xynb,
    :xynbs => θ_xynbs
)



end
