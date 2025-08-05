

function getvariance(varimage::AbstractArray{T}, boxcorners::AbstractArray{T}, i, j) where {T<:Real}
    return varimage[boxcorners[1] + i + (j-1)*size(varimage,1)]
end

function getvariance(varimage::T, boxcorners::T, i, j) where {T<:Real}
    return varimage
end


"""
    calclogL(θ::GaussMLEParams{T}, box::AbstractArray{T}, args::GaussMLEArgs) where {T<:Real}

Calculate the log-likelihood of a Gaussian MLE fit for a given set of parameters and data.

# Arguments
- `θ::GaussMLEParams{T}`: A `GaussMLEParams` object containing the parameters of the Gaussian MLE fit.
- `box::AbstractArray{T}`: An `AbstractArray` containing the data to fit.
- `args::GaussMLEArgs`: A `GaussMLEArgs` object containing additional arguments for the fit.

# Returns
- logL: The log-likelihood of the fit.

"""
function calclogL(θ::GaussMLEParams{T}, box::AbstractArray{T}, args::GaussMLEArgs) where {T<:Real}
    varimage = T(0)
    boxcorners = T(0)
    return calclogL(θ, box, args, varimage, boxcorners)
end

function calclogL(θ::GaussMLEParams{T}, box::AbstractArray{T}, args::GaussMLEArgs,
    varimage::Union{T,AbstractArray{T}}, boxcorners::Union{T,AbstractArray{T}}) where {T<:Real}

    boxsize = Int(sqrt(length(box)))
    loglikelihood = T(0)
    for i = 1:boxsize, j = 1:boxsize
        data_pixel = box[boxsize*(j-1)+i]
        model_pixel = model(θ, args, i, j) + getvariance(varimage, boxcorners, i, j)    
        # LogLikelihood
        if (model_pixel > 0)
            if (data_pixel > 0)
                loglikelihood += data_pixel * log(model_pixel) - model_pixel - data_pixel * log(data_pixel) + data_pixel
            else
                loglikelihood += -model_pixel
            end
        end
    end
    return loglikelihood
end



"""
    crlb!(Σ::GaussMLEΣ{T}, grad_pixel::AbstractArray{T}, θ::GaussMLEParams{T}, boxsize::Int, args::GaussMLEArgs)

Calculate the Cramer-Rao lower bound (CRLB) for a given set of parameters and data.

# Arguments
- `Σ::GaussMLEΣ{T}`: A `GaussMLEΣ` object to store the CRLB values.
- `grad_pixel::AbstractArray{T}`: An `AbstractArray` to store the gradient values.
- `θ::GaussMLEParams{T}`: A `GaussMLEParams` object containing the parameters of the Gaussian MLE fit.
- `boxsize::Int`: The size of the box to fit.
- `args::GaussMLEArgs`: A `GaussMLEArgs` object containing additional arguments for the fit.

"""
function crlb!(Σ::GaussMLEΣ{T}, grad_pixel::AbstractArray{T}, θ::GaussMLEParams{T}, boxsize::Int, args::GaussMLEArgs) where {T<:Real}
    varimage = T(0)
    boxcorners = T(0)
    crlb!(Σ, grad_pixel, θ, boxsize, args, varimage, boxcorners)
    return nothing
end

function crlb!(Σ::GaussMLEΣ{T}, grad_pixel::AbstractArray{T}, θ::GaussMLEParams{T}, boxsize::Int, args::GaussMLEArgs,
    varimage::Union{T,AbstractArray{T}}, boxcorners::Union{T,AbstractArray{T}}) where {T<:Real}

    fi = zeros(T, θ.nparams,  θ.nparams)
    for i = 1:boxsize, j = 1:boxsize
        model_pixel = model(θ, args, i, j) + getvariance(varimage, boxcorners, i, j)
        if model_pixel < 0
           println("$i, $j, Warning: model_pixel < 0")
        end
        gradient!(grad_pixel, θ, args, i, j)
        # Building the Fisher Information Matrix
        for k in 1:θ.nparams
            for l in k:θ.nparams
                fi[(l-1)*θ.nparams+k] += grad_pixel[l] * grad_pixel[k] / model_pixel
                fi[(k-1)*θ.nparams+l] = fi[(l-1)*θ.nparams+k]
            end
        end
    end

    # Check for NaN/Inf in Fisher Information matrix
    has_nan_inf = false
    for i in 1:size(fi, 1)
        for j in 1:size(fi, 2)
            if !isfinite(fi[i,j])
                has_nan_inf = true
                break
            end
        end
    end
    
    if has_nan_inf
        # Replace NaN/Inf with small positive values
        for i in 1:size(fi, 1)
            for j in 1:size(fi, 2)
                if !isfinite(fi[i,j])
                    fi[i,j] = i == j ? T(1e-3) : T(0)
                end
            end
        end
    end
    
    # Try to invert the Fisher Information matrix
    # Add regularization if needed for singular matrices
    try
        crlb = inv(fi)
        GaussMLE.GaussModel.fill!(Σ, sqrt.(abs.(diag(crlb))))
    catch e
        if isa(e, LinearAlgebra.SingularException) || isa(e, ArgumentError)
            # Matrix is singular - use pseudoinverse or regularization
            # Add small regularization to diagonal
            for i in 1:θ.nparams
                fi[i,i] += T(1e-6)
            end
            try
                crlb = inv(fi)
                GaussMLE.GaussModel.fill!(Σ, sqrt.(abs.(diag(crlb))))
            catch
                # If still singular, use crude approximation
                crude_crlb = zeros(T, θ.nparams)
                for i in 1:θ.nparams
                    crude_crlb[i] = fi[i,i] > 0 ? T(1) / sqrt(fi[i,i]) : T(1)
                end
                GaussMLE.GaussModel.fill!(Σ, crude_crlb)
            end
        else
            rethrow(e)
        end
    end
end
