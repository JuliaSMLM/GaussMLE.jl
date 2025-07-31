"""
   fitstack(stack::AbstractArray{T}, modelsymbol::Symbol, args::GaussMLEArgs{T};
            varimage::Union{T,AbstractArray{T}}=T(0), 
            boxcorners::Union{T,AbstractArray{T}}=T(0)) where T <: Real

Fit Gaussian models to each box in a stack of 2D "boxes" of data.

# Arguments
- `stack::AbstractArray{T}`: 3D array containing the data to be fitted. Each slice along the third dimension is treated as a separate 2D box.
- `modelsymbol::Symbol`: Symbol indicating the type of Gaussian model to fit.
- `args::GaussMLEArgs{T}`: Additional model-specific arguments required for fitting.

# Keyword Arguments
- `varimage::Union{T,AbstractArray{T}}`: Camera variance image, or a single variance for all pixels (default is `T(0)`).
- `boxcorners::Union{T,AbstractArray{T}}`: The top left corners of the boxes in the variance image (default is `T(0)`).

# Returns
- `θ`: Array of fitted parameters for each 2D box.
- `Σ`: Array of uncertainties for each set of fitted parameters.

# Notes
- Data and Variance image must be in units of photons (Poisson distributed).

# Example
```julia
fitstack(stack, :xynb, args; varimage=myvarimage, boxcorners=myboxcorners)
```
"""
function fitstack(stack::AbstractArray{T}, modelsymbol::Symbol, args::GaussMLEArgs{T};
    varimage::Union{T,AbstractArray{T}}=T(0),
    boxcorners::Union{T,AbstractArray{T}}=T(0)) where {T<:Real}

    boxsz = size(stack, 1)
    nboxes = size(stack, 3)
    modeltype = get(MODEL_MAP, modelsymbol, -1)
    if modeltype == -1
        error("Model symbol not found")
    end

    θ = [genθ(modeltype, boxsz; T) for i in 1:nboxes]
    Σ = [genΣ(modeltype; T) for i in 1:nboxes]

    Threads.@threads for idx in eachindex(θ)
        fitbox!(θ[idx], Σ[idx], stack[:, :, idx], args, varimage, boxcorners)
    end

    return θ, Σ
end
 