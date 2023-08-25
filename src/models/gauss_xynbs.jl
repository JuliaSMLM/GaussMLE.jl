
mutable struct θ_xynbs{T} <: GaussMLEParams{T}
    modeltype::Int # For CUDA
    nparams::Int # For CUDA
    x::T
    y::T
    n::T
    bg::T
    σ_PSF::T
    θ_xynbs(x::T, y::T, n::T, bg::T, σ_PSF::T) where {T} = new{T}(2, 5, x, y, n, bg, σ_PSF) 
end

function θ_xynbs()
    return θ_xynbs(0.0, 0.0, 0.0, 0.0, 0.0)
end

struct Args_xynbs{T} <: GaussMLEArgs{T}
    σ_PSF::T
end

mutable struct Σ_xynbs{T} <: GaussMLEΣ{T}
    σ_x::T
    σ_y::T
    σ_n::T
    σ_bg::T
    σ_σ_PSF::T
    logL::T
end

function Σ_xynbs()
    return Σ_xynbs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function fill!(σ::Σ_xynbs{T}, values::AbstractArray{T}, logL::T) where T <: Real
    σ.σ_x = values[1]
    σ.σ_y = values[2]
    σ.σ_n = values[3]
    σ.σ_bg = values[4]
    σ.σ_σ_PSF = values[5]
    σ.logL = logL
    return nothing
end

function fill!(σ::Σ_xynbs{T}, values::AbstractArray{T}) where T <: Real
    σ.σ_x = values[1]
    σ.σ_y = values[2]
    σ.σ_n = values[3]
    σ.σ_bg = values[4]
    σ.σ_σ_PSF = values[5]
    return nothing
end

function model(θ::θ_xynbs{T}, args::Args_xynbs{T}, i, j) where {T<:Real}
    return θ.bg + θ.n *
                  integral_gaussian_1d(i, θ.x, θ.σ_PSF) *
                  integral_gaussian_1d(j, θ.y, θ.σ_PSF)
end

function gradient!(grad::AbstractArray{T}, θ::θ_xynbs{T}, args::Args_xynbs{T}, i, j) where {T<:Real}
    PSFx = integral_gaussian_1d(i, θ.x, θ.σ_PSF)
    PSFy = integral_gaussian_1d(j, θ.y, θ.σ_PSF)

    grad[1], = derivative_integral_gaussian_1d(i, θ.x, θ.σ_PSF, θ.n, PSFy)
    grad[2], = derivative_integral_gaussian_1d(j, θ.y, θ.σ_PSF, θ.n, PSFx)
    grad[3] = PSFx * PSFy
    grad[4] = T(1)
    grad[5], = derivative_integral_gaussian_2d_sigma(i, j, θ.x, θ.y, θ.σ_PSF, θ.n, PSFx, PSFy) 

    return nothing
end

function curvature!(hessdiag::AbstractArray{T}, θ::θ_xynbs{T}, args::Args_xynbs{T}, i, j) where {T<:Real}
    PSFx = integral_gaussian_1d(i, θ.x, θ.σ_PSF)
    PSFy = integral_gaussian_1d(j, θ.y, θ.σ_PSF)

    (_, hessdiag[1]) = derivative_integral_gaussian_1d(i, θ.x, θ.σ_PSF, θ.n, PSFy)
    (_, hessdiag[2]) = derivative_integral_gaussian_1d(j, θ.y, θ.σ_PSF, θ.n, PSFx)
    hessdiag[3] = T(0)
    hessdiag[4] = T(0)
    (_, hessdiag[5]) = derivative_integral_gaussian_2d_sigma(i, j, θ.x, θ.y, θ.σ_PSF, θ.n, PSFx, PSFy)

    # println(hessdiag)
    return nothing
end

function initialize_parameters!(θ::θ_xynbs{T}, data::Matrix{T}, boxsize::Int, args::Args_xynbs{T}) where {T<:Real}
    θ.σ_PSF = args.σ_PSF
    (θ.x, θ.y) = center_of_mass_2d(boxsize, data)
    (nmax, θ.bg) = gaussian_max_min_2d(boxsize, args.σ_PSF, data)
    θ.n = max(0, (nmax - θ.bg) * 4 * pi * args.σ_PSF^2)
    return nothing
end

function update!(θ::θ_xynbs{T}, numerator::AbstractArray{T}, denominator::AbstractArray{T}) where {T<:Real}

    nparams = 5
    θ_tol = [1e-3, 1e-3, 1e-1, 1e-1, 1e-2]
    maxjump = [1.0, 1.0, 100.0, 2.0, 0.1]

    Δθ = numerator[1:nparams] ./ denominator[1:nparams]

    Δθ[1] = -min(max(Δθ[1], -maxjump[1]), maxjump[1])
    Δθ[2] = -min(max(Δθ[2], -maxjump[2]), maxjump[2])
    Δθ[3] = -min(max(Δθ[3], -maxjump[3]), maxjump[3])
    Δθ[4] = -min(max(Δθ[4], -maxjump[4]), maxjump[4])
    Δθ[5] = -min(max(Δθ[5], -maxjump[5]), maxjump[5])

    θ.x = θ.x + Δθ[1]
    θ.y = θ.y + Δθ[2]
    θ.n = θ.n + Δθ[3]
    θ.bg = θ.bg + Δθ[4]
    θ.σ_PSF = θ.σ_PSF + Δθ[5]

    # Other constraints
    θ.n = max(θ.n, 0.1) # Make sure Photons is postitve
    θ.bg = max(θ.bg, 0.01) # Make sure Background is postitve
    θ.σ_PSF = max(θ.σ_PSF, 0.01) # Make sure PSF is postitve

    return all(abs.(Δθ) .<= θ_tol)
end


function genargs(::Type{θ_xynbs}; T::Type{<:Real}=Float32)
    σ_PSF = T(1.3)
    return Args_xynbs(σ_PSF)
end

function genθ(::Type{θ_xynbs}, boxsize::Int; T::Type{<:Real}=Float32)
    x = T(boxsize / 2)
    y = T(boxsize / 2)
    n = T(500)
    bg = T(2)
    σ_PSF = T(1.3)
    return θ_xynbs(x, y, n, bg, σ_PSF)
end

function genΣ(::Type{θ_xynbs}; T::Type{<:Real}=Float32)
    σ_x = T(0)
    σ_y = T(0)
    σ_n = T(0)
    σ_bg = T(0)
    σ_σ_PSF = T(0)
    logL = T(0)
    return Σ_xynbs(σ_x, σ_y, σ_n, σ_bg, σ_σ_PSF, logL)
end

