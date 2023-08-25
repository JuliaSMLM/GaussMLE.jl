function fitbox!(θ::GaussMLEParams{T}, Σ::GaussMLEΣ{T}, box::AbstractArray{T}, args::GaussMLEArgs,
    varimage::Union{T,AbstractArray{T}}, boxcorners::Union{T,AbstractArray{T}}) where {T<:Real}

    grad_pixel = zeros(T, MAXPARAMS)
    curve_pixel = zeros(T, MAXPARAMS)
    
    numerator = zeros(T, MAXPARAMS)
    denominator = zeros(T, MAXPARAMS)

    # initial values
    boxsize = Int(sqrt(length(box)))
    initialize_parameters!(θ, box, boxsize, args)

    smallmodel = T(1e-3)
    bigratio = T(1e4)

    maxiter = 100
    iterations = 0
    tol = false
    while !tol && iterations < maxiter

        iterations += 1
        numerator .= 0.0
        denominator .= 0.0

        for i = 1:boxsize, j = 1:boxsize

            variance_pixel = getvariance(varimage, boxcorners, i, j)

            data_pixel = box[boxsize*(i-1)+j]
            model_pixel = model(θ, args, i, j) + variance_pixel
            gradient!(grad_pixel, θ, args, i, j)
            curvature!(curve_pixel, θ, args, i, j)

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
