

function f_xynb(θ, i, j, fitargs::ArgsFit)
    return θ[4] + θ[3] *
                  integral_gaussian_1d(i, θ[1], fitargs.σ_PSF) *   # Using the function from baselibrary
                  integral_gaussian_1d(j, θ[2], fitargs.σ_PSF)
end

function fp_xynb(θ, i, j, fitargs::ArgsFit)

    PSFx = integral_gaussian_1d(i, θ[1], fitargs.σ_PSF)
    PSFy = integral_gaussian_1d(j, θ[2], fitargs.σ_PSF)

    dudt = zeros(typeof(θ[1]), length(θ))

    (dudt[1], _) = derivative_integral_gaussian_1d(i, θ[1], fitargs.σ_PSF, θ[3], PSFy)  # Using the function from baselibrary
    (dudt[2], _) = derivative_integral_gaussian_1d(j, θ[2], fitargs.σ_PSF, θ[3], PSFx)
    dudt[3] = PSFx * PSFy
    dudt[4] = 1.0

    return dudt
end

function fpp_xynb(θ, i, j, fitargs::ArgsFit)

    PSFx = integral_gaussian_1d(i, θ[1], fitargs.σ_PSF)
    PSFy = integral_gaussian_1d(j, θ[2], fitargs.σ_PSF)

    d2udt2 = zeros(typeof(θ[1]), length(θ))

    (_, d2udt2[1]) = derivative_integral_gaussian_1d(i, θ[1], fitargs.σ_PSF, θ[3], PSFy)
    (_, d2udt2[2]) = derivative_integral_gaussian_1d(j, θ[2], fitargs.σ_PSF, θ[3], PSFx)

    return d2udt2
end

function f_init_xynb!(θ, roi, fitargs::ArgsFit)
    sz = Int(sqrt(length(roi)))
    (θ[1], θ[2]) = center_of_mass_2d(sz, roi)  # Using the function from baselibrary
    (nmax, θ[4]) = gaussian_max_min_2d(sz, fitargs.σ_PSF, roi)  # Assuming gaussFMaxMin2D is a function you have
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
