

function f_xynb(θ, i, j, fitargs::ArgsFit)
    return θ[4] + θ[3] *
                  intGauss1D(i, θ[1], fitargs.σ_PSF) *
                  intGauss1D(j, θ[2], fitargs.σ_PSF)
end

function fp_xynb(θ, i, j, fitargs::ArgsFit)

    PSFx = intGauss1D(i, θ[1], fitargs.σ_PSF)
    PSFy = intGauss1D(j, θ[2], fitargs.σ_PSF)

    dudt = zeros(typeof(θ[1]), length(θ))

    (dudt[1], _) = derivativeIntGauss1D(i, θ[1], fitargs.σ_PSF, θ[3], PSFy)
    (dudt[2], _) = derivativeIntGauss1D(j, θ[2], fitargs.σ_PSF, θ[3], PSFx)
    dudt[3] = PSFx * PSFy
    dudt[4] = 1.0

    return dudt
end


function fpp_xynb(θ, i, j, fitargs::ArgsFit)

    PSFx = intGauss1D(i, θ[1], fitargs.σ_PSF)
    PSFy = intGauss1D(j, θ[2], fitargs.σ_PSF)

    d2udt2 = zeros(typeof(θ[1]), length(θ))

    (_, d2udt2[1]) = derivativeIntGauss1D(i, θ[1], fitargs.σ_PSF, θ[3], PSFy)
    (_, d2udt2[2]) = derivativeIntGauss1D(j, θ[2], fitargs.σ_PSF, θ[3], PSFx)

    return d2udt2
end

function f_init_xynb!(θ, roi, fitargs::ArgsFit)
    sz = Int(sqrt(length(roi)))
    (θ[1], θ[2]) = centerofMass2D(sz, roi)
    (nmax, θ[4]) = gaussFMaxMin2D(sz, fitargs.σ_PSF, roi)
    θ[3] = max(0, (nmax - θ[4]) * 4 * pi * fitargs.σ_PSF^2)
    return nothing
end

function gen_xynb(sz::Int, args::ArgsGMLE)

    maxjump = [1.0, 1.0, 100.0, 2.0]
    constraints = [
            -3*args.σ_PSF   3*args.σ_PSF
            -3*args.σ_PSF   3*args.σ_PSF
            1e-3    Inf 
            1e-6    Inf
        ]
    display(constraints)
    
    niterations = 10
    
    fitargs = 
        ArgsFit(
            Float32.(maxjump),
            Float32.(constraints),
            niterations,
            1,
            Float32.(args.σ_PSF),
            ZFitParams()
            )

    return fitargs
end
    

