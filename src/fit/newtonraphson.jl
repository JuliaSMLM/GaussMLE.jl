"""
    fitbox!(θ::GaussMLEParams{T}, Σ::GaussMLEΣ{T}, box::AbstractArray{T}, args::GaussMLEArgs,
            varimage::Union{T,AbstractArray{T}}, boxcorners::Union{T,AbstractArray{T}};
            maxiter::Int = 100) where {T<:Real}

Fits a Gaussian blob model using maximum likelihood estimation and Poisson noise model to a single 2D "box" of data.

# Arguments
- `θ::GaussMLEParams{T}`: Initial parameters for the Gaussian MLE model.
- `Σ::GaussMLEΣ{T}`: Container for the calculated Cramer-Rao Lower Bound (CRLB) and log likelihood.
- `box::AbstractArray{T}`: 2D array containing the data to be fitted.
- `args::GaussMLEArgs`: Additional arguments required for fitting.
- `varimage::Union{T,AbstractArray{T}}`: Camera readnoise variance image, or a single variance for all pixels.
- `boxcorners::Union{T,AbstractArray{T}}`: The corners of the box in the full image. Used when `varimage` is an image.

# Keyword Arguments
- `maxiter::Int`: Maximum number of iterations for the fitting process (default is 100).

# Output
- Updates `θ` and `Σ` in-place with the fitted parameters and their uncertainties.

# Notes
- This function performs iterative fitting and stops either when the fit converges or when it reaches `maxiter` iterations. 
- The input `box` must be a square 2D array.
- Input data `box` is assumed to be background subtracted.

"""
function fitbox!(θ::GaussMLEParams{T}, Σ::GaussMLEΣ{T}, box::AbstractArray{T}, args::GaussMLEArgs,
    varimage::Union{T,AbstractArray{T}}, boxcorners::Union{T,AbstractArray{T}}; maxiter::Int = 100) where {T<:Real}

    grad_pixel = zeros(T, MAXPARAMS)
    curve_pixel = zeros(T, MAXPARAMS)
    
    numerator = zeros(T, MAXPARAMS)
    denominator = zeros(T, MAXPARAMS)

    # initial values
    boxsize = Int(sqrt(length(box)))
    initialize_parameters!(θ, box, boxsize, args)

    smallmodel = T(1e-3)
    bigratio = T(1e4)

    iterations = 0
    tol = false
    while !tol && iterations < maxiter

        iterations += 1
        numerator .= 0.0
        denominator .= 0.0

        for i = 1:boxsize, j = 1:boxsize

            variance_pixel = getvariance(varimage, boxcorners, i, j)

            data_pixel = box[boxsize*(i-1)+j]
            
            # this calculates model, grad, and curvature
            model_pixel = curvature!(grad_pixel, curve_pixel, θ, args, i, j) + variance_pixel
            
            cf = (model_pixel > smallmodel) ? data_pixel / model_pixel - 1 : 0.0
            df = (model_pixel > smallmodel) ? data_pixel / model_pixel^2 : 0.0

            cf = min(cf, bigratio)
            df = min(df, bigratio)

            for ll = 1:θ.nparams
                numerator[ll] += grad_pixel[ll] * cf
                denominator[ll] += curve_pixel[ll] * cf - grad_pixel[ll]^2 * df
            end
        end

        tol = update!(θ, numerator, denominator)
    end
    
    crlb!(Σ, grad_pixel, θ, boxsize, args)
    Σ.logL = calclogL(θ, box, args)

    return nothing
end

function fitbox!(θ::GaussMLEParams{T}, Σ::GaussMLEΣ{T}, box::AbstractArray{T}, args::GaussMLEArgs) where {T<:Real}
    varimage = T(0)
    boxcorners = T(0)
    fitbox!(θ, Σ, box, args, varimage, boxcorners)
    return nothing
end
