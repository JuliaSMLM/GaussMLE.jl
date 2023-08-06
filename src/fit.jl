

# if args.fittype == :xynb
#     maxjump = [1.0f0, 1.0f0, 100.0f0, 2.0f0]    
# end

function model_pp(θ_est, i, j, fitargs)
    fittype = fitargs.fittype
    if fittype == 1 #"xynb"
        return f_xynb(θ_est, i, j, fitargs), fp_xynb(θ_est, i, j, fitargs), fpp_xynb(θ_est, i, j, fitargs)
    else
        return nothing
    end
end


function model_init!(θ_est, roi, fitargs)
    fittype = fitargs.fittype
    if fittype == 1 #"xynb"
        return f_init_xynb!(θ_est, roi, fitargs)
    end
end




"""
    fitroi!(θ, Σ, logL, roi, args, maxjump, constraints,
        f::Function, fp::Array{<:Function}, fpp::Array{<:Function})

Calculate the MLE for a 2D Region of Interest

#Arguments
-`data_3d`  : The subset of data to fit
-`PSFSigma` : The variance for the Point Spread Function
-`sz`       : The size of the subregion within the subset of data
"""
function fitroi!(θ, Σ, logL, roi, fitargs::ArgsFit)

    sz = Int(sqrt(length(roi)))
    nparams = length(θ)
    dtype = typeof(roi[1])
    #dtype = Float32

    θ_est = zeros(dtype, nparams)
    fi = zeros(dtype, nparams, nparams)

    # initial values
    model_init!(θ_est, roi, fitargs)
    numerator = zeros(dtype, nparams)
    denominator = zeros(dtype, nparams)

    for k in 1:fitargs.niterations
        numerator .= 0.0f0
        denominator .= 0.0f0

        for i = 1:sz, j = 1:sz
            data = roi[sz*(j-1)+i]
            model, fp, fpp = model_pp(θ_est, i, j, fitargs)
  
            cf = 0.0f0
            df = 0.0f0
            if (model > 1e-3)
                cf = data / model - 1
            end
            if (model > 1e-3)
                df = data / model^2
            end
            cf = min(cf, 1e4)
            df = min(df, 1e4)

            numerator .+= fp .* cf
            denominator .+= fpp .* cf .- fp .^ 2 .* df

        end

        denominator .= max.(eps(denominator[1]), denominator)

        # The update
        θ_est .-= min.(max.(numerator ./ denominator, -fitargs.maxjump), fitargs.maxjump)

        # constraints
        θ_est .= min.(max.(θ_est, fitargs.constraints[:,2]), fitargs.constraints[:,1])
        
    end

    # Calculating the LogLikelihood
    loglikelihood = 0.0f0
    for i = 1:sz, j = 1:sz
        data = roi[sz*(j-1)+i]
        model, _, _ = model_pp(θ_est, i, j, fitargs)
        if (model > 0)
            if (data > 0)
                loglikelihood += data * log(model) - model - data * log(data) + data
            else
                loglikelihood += -model
            end
        end
    end

    # Build FI Matrix
    for i = 1:sz, j = 1:sz
        model, fp, _ = model_pp(θ_est, i, j, fitargs)
        fi .+= fp * fp' ./ model
    end

    # Matrix inverse (CRLB=F^-1) and output assigments
    v = matInv(fi,sz)

    # Write values into arrays
    θ .= θ_est
    Σ .= sqrt.(diag(v))
    logL[1] = loglikelihood

    return nothing
end
