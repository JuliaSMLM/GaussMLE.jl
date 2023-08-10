function fitbox!(θ::GaussMLEParams{T},Σ::GaussMLEΣ{T}, box::AbstractArray{T}, args::GaussMLEArgs) where T <: Real
	
    grad_pixel = zeros(T, MAXPARAMS)
    curve_pixel = zeros(T, MAXPARAMS)
    fi = zeros(T, MAXPARAMS, MAXPARAMS)

    numerator = zeros(T, MAXPARAMS)
    denominator = zeros(T, MAXPARAMS)
        
    # initial values
    boxsize = Int(sqrt(length(box)))
    initialize_parameters!(θ, box, boxsize, args)

    println("initial parameters: ", θ)  

    smallmodel = T(1e-3)
    bigratio = T(1e4)

    maxiter = 50
    iterations = 0
    tol=false
    while !tol && iterations < maxiter

        iterations += 1
        numerator.=0.0
        denominator.=0.0

        for ii = 1:boxsize
            for jj = 1:boxsize
                
                data_pixel = box[boxsize * (jj - 1) + ii]
                model_pixel = model(θ, args, ii, jj)
                gradient!(grad_pixel, θ, args, ii, jj)
                curvature!(curve_pixel, θ, args, ii, jj)

                cf = (model_pixel > smallmodel) ? data_pixel / model_pixel - 1 : 0.0
                df = (model_pixel > smallmodel) ? data_pixel / model_pixel^2 : 0.0

                cf = min(cf, bigratio)
                df = min(df, bigratio)
                
                for ll = 1:θ.nparams
                    numerator[ll] += grad_pixel[ll] * cf
                    denominator[ll] += curve_pixel[ll] * cf - grad_pixel[ll]^2 * df
                end
            end
        end

        tol = update!(θ, numerator, denominator)
        println("tolerance: ", tol)
        println("parameters: ", θ)  
    end
    
    # Calculating the CRLB and LogLikelihood

    loglikelihood = T(0)
    for ii = 1:boxsize
        for jj = 1:boxsize

            data_pixel = box[boxsize * (jj - 1) + ii]
            model_pixel = model(θ, args, ii, jj)
            gradient!(grad_pixel, θ, args, ii, jj)
        
            # Building the Fisher Information Matrix
            for kk in 1:θ.nparams
                for ll in kk:θ.nparams
                    fi[(ll-1)*θ.nparams+kk] += grad_pixel[ll] * grad_pixel[kk] / model_pixel
                    fi[(kk-1)*θ.nparams+ll] = fi[(ll-1)*θ.nparams+kk]
                end
            end
        
            # LogLikelihood
            if (model_pixel > 0)
                if (data_pixel > 0)
                    loglikelihood += data_pixel * log(model_pixel) - model_pixel - data_pixel * log(data_pixel) + data_pixel
                else
                    loglikelihood += -model_pixel
                end
            end
    
        end
    end

    # Inverting the Fisher Information Matrix
    _, minv_diag = matrix_inverse!(fi, boxsize)
    GaussMLE.GaussModel.fill!(Σ, minv_diag)

    return nothing
    
end
