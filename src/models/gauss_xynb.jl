
mutable struct θ_xynb{T} <: GaussMLEParams{T}
    modeltype::Int # For CUDA
    nparams::Int # For CUDA
    x::T
    y::T
    n::T
    bg::T
    θ_xynb(x::T, y::T, n::T, bg::T) where {T} = new{T}(1, 4, x, y, n, bg) # Default modeltype is 1
end

function θ_xynb()
    return θ_xynb(0.0, 0.0, 0.0, 0.0)
end

struct Args_xynb{T} <: GaussMLEArgs{T}
    σ_PSF::T
end

mutable struct Σ_xynb{T} <: GaussMLEΣ{T}
    σ_x::T
    σ_y::T
    σ_n::T
    σ_bg::T
    logL::T
end

function Σ_xynb()
    return Σ_xynb(0.0, 0.0, 0.0, 0.0, 0.0)
end

function fill!(σ::Σ_xynb{T}, values::AbstractArray{T}, logL::T) where T <: Real
    σ.σ_x = values[1]
    σ.σ_y = values[2]
    σ.σ_n = values[3]
    σ.σ_bg = values[4]
    σ.logL = logL
    return nothing
end

function fill!(σ::Σ_xynb{T}, values::AbstractArray{T}) where T <: Real
    σ.σ_x = values[1]
    σ.σ_y = values[2]
    σ.σ_n = values[3]
    σ.σ_bg = values[4]
    return nothing
end

# Helper function to compute model, gradient and/or curvature
function compute_all!(model_val::Union{Nothing, Ref{T}}, grad::Union{Nothing, AbstractArray{T}}, hessdiag::Union{Nothing, AbstractArray{T}}, θ::θ_xynb{T}, args::Args_xynb{T}, i, j) where {T<:Real}
    PSFx = integral_gaussian_1d(i, θ.x, args.σ_PSF)
    PSFy = integral_gaussian_1d(j, θ.y, args.σ_PSF)
    
    # Compute model if needed
    if model_val !== nothing
        model_val[] = θ.bg + θ.n * PSFx * PSFy
    end
    
    # Compute gradient if needed
    if grad !== nothing
        grad[1], _ = derivative_integral_gaussian_1d(i, θ.x, args.σ_PSF, θ.n, PSFy)
        grad[2], _ = derivative_integral_gaussian_1d(j, θ.y, args.σ_PSF, θ.n, PSFx)
        grad[3] = PSFx * PSFy
        grad[4] = T(1)
    end
    
    # Compute curvature if needed
    if hessdiag !== nothing
        _, hessdiag[1] = derivative_integral_gaussian_1d(i, θ.x, args.σ_PSF, θ.n, PSFy)
        _, hessdiag[2] = derivative_integral_gaussian_1d(j, θ.y, args.σ_PSF, θ.n, PSFx)
        hessdiag[3] = T(0)
        hessdiag[4] = T(0)
    end
    
    return nothing
end


# Functions for individual tasks call the helper function
function model(θ::θ_xynb{T}, args::Args_xynb{T}, i, j) where {T<:Real}
    val = Ref{T}(0)
    compute_all!(val, nothing, nothing, θ, args, i, j)
    return val[]
end

function gradient!(grad::AbstractArray{T}, θ::θ_xynb{T}, args::Args_xynb{T}, i, j) where {T<:Real}
    val = Ref{T}(0)
    compute_all!(val, grad, nothing, θ, args, i, j)
    return val[]
end

function curvature!(grad::AbstractArray{T}, hessdiag::AbstractArray{T}, θ::θ_xynb{T}, args::Args_xynb{T}, i, j) where {T<:Real}
    val = Ref{T}(0)
    compute_all!(val, grad, hessdiag, θ, args, i, j)
    return val[]
end


function initialize_parameters!(θ::θ_xynb{T}, data::Matrix{T}, boxsize::Int, args::Args_xynb{T}) where {T<:Real}
    (θ.x, θ.y) = center_of_mass_2d(boxsize, data)
    (nmax, θ.bg) = gaussian_max_min_2d(boxsize, args.σ_PSF, data)
    θ.n = max(0, (nmax - θ.bg) * 4 * pi * args.σ_PSF^2)
    return nothing
end

function update!(θ::θ_xynb{T}, numerator::AbstractArray{T}, denominator::AbstractArray{T}) where {T<:Real}

    nparams = 4
    θ_tol = [1e-3, 1e-3, 1e-1, 1e-1]
    maxjump = [1.0, 1.0, 100.0, 2.0]

    Δθ = numerator[1:nparams] ./ denominator[1:nparams]

    # display(Δθ)

    Δθ[1] = -min(max(Δθ[1], -maxjump[1]), maxjump[1])
    Δθ[2] = -min(max(Δθ[2], -maxjump[2]), maxjump[2])
    Δθ[3] = -min(max(Δθ[3], -maxjump[3]), maxjump[3])
    Δθ[4] = -min(max(Δθ[4], -maxjump[4]), maxjump[4])

    θ.x = θ.x + Δθ[1]
    θ.y = θ.y + Δθ[2]
    θ.n = θ.n + Δθ[3]
    θ.bg = θ.bg + Δθ[4]

    # Other constraints
    θ.n = max(θ.n, 0.1) # Make sure Photons is postitve
    θ.bg = max(θ.bg, 0.01) # Make sure Background is postitve
    # display(Δθ)
    return all(abs.(Δθ) .<= θ_tol)
end


function genargs(::Type{θ_xynb}; T::Type{<:Real}=Float32)
    σ_PSF = T(1.3)
    return Args_xynb(σ_PSF)
end

function genθ(::Type{θ_xynb}, boxsize::Int; T::Type{<:Real}=Float32)
    x = T(boxsize / 2)
    y = T(boxsize / 2)
    n = T(500)
    bg = T(2)
    return θ_xynb(x, y, n, bg)
end

function genΣ(::Type{θ_xynb}; T::Type{<:Real}=Float32)
    σ_x = T(0)
    σ_y = T(0)
    σ_n = T(0)
    σ_bg = T(0)
    logL = T(0)
    return Σ_xynb(σ_x, σ_y, σ_n, σ_bg, logL)
end

