
mutable struct θ_xynb{T} <: GaussMLEParams{T}  
    x::T
    y::T
    n::T
    bg::T
end

struct Args_xynb{T} <: GaussMLEArgs{T} 
    σ_PSF::T
end

mutable struct Σ_xynb{T} <: GaussMLEΣ{T} 
    σ_x::T
    σ_y::T
    σ_n::T
    σ_bg::T
end

function model(θ::θ_xynb{T}, i, j, args::Args_xynb{T}) where T <: Real
    return θ.bg + θ.n *
                  integral_gaussian_1d(i, θ.x, args.σ_PSF) *  
                  integral_gaussian_1d(j, θ.y, args.σ_PSF)
end


function gradient(θ::θ_xynb{T}, i, j, args::Args_xynb{T}) where T <: Real
    PSFx = integral_gaussian_1d(i, θ.x, args.σ_PSF)
    PSFy = integral_gaussian_1d(j, θ.y, args.σ_PSF)

    dudt = zeros(T, length(θ))

    (dudt.x, _) = derivative_integral_gaussian_1d(i, θ.x, args.σ_PSF, θ.n, PSFy)  
    (dudt.y, _) = derivative_integral_gaussian_1d(j, θ.y, args.σ_PSF, θ.n, PSFx)
    dudt.n = PSFx * PSFy
    dudt.bg = 1.0

    return dudt
end

function curvature(θ::θ_xynb{T}, i, j, args::Args_xynb{T}) where T <: Real
    PSFx = integral_gaussian_1d(i, θ.x, args.σ_PSF)
    PSFy = integral_gaussian_1d(j, θ.y, args.σ_PSF)

    d2udt2 = zeros(typeof(θ.x), length(θ))

    (_, d2udt2.x) = derivative_integral_gaussian_1d(i, θ.x, args.σ_PSF, θ.n, PSFy)
    (_, d2udt2.y) = derivative_integral_gaussian_1d(j, θ.y, args.σ_PSF, θ.n, PSFx)
    d2udt2.n = PSFx * PSFy
    d2udt2.bg = 0.0

    return d2udt2
end

function initialize_parameters!(θ::θ_xynb{T}, data::Matrix{T}, args::Args_xynb{T}) where T <: Real
    sz = Int(sqrt(length(data)))
    (θ.x, θ.y) = center_of_mass_2d(sz, data)  
    (nmax, θ.bg) = gaussian_max_min_2d(sz, args.σ_PSF, data)  
    θ.n = max(0, (nmax - θ.bg) * 4 * pi * args.σ_PSF^2)
    return nothing
end

function genargs(::Type{θ_xynb}; T::Type{<:Real} = Float32) 
    σ_PSF = T(1.3)
    return Args_xynb(σ_PSF)
end

function genparams(::Type{θ_xynb}, boxsize::Int; T::Type{<:Real} = Float32)
    x = T(boxsize / 2)
    y = T(boxsize / 2)
    n = T(500)
    bg = T(2)
    return θ_xynb(x, y, n, bg)
end



# function gen_xynb(sz::Int, args::ArgsGMLE)

#     maxjump = [1.0, 1.0, 100.0, 2.0]
#     constraints = [
#             -3*args.σ_PSF   3*args.σ_PSF
#             -3*args.σ_PSF   3*args.σ_PSF
#             1e-3    Inf 
#             1e-6    Inf
#         ]
#     display(constraints)
    
#     niterations = 10
    
#     fitargs = 
#         ArgsFit(
#             Float32.(maxjump),
#             Float32.(constraints),
#             niterations,
#             1,
#             Float32.(args.σ_PSF),
#             ZFitParams()
#             )

#     return fitargs
# end
