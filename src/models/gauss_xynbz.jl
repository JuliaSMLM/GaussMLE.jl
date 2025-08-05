"""
5-parameter astigmatic z-model: x, y, z, intensity, bg

This model includes axial position (z) fitting using pre-calibrated astigmatic PSF parameters.
The PSF width varies with z-position according to astigmatic calibration stored in ZFitParams.

Mathematical model:
- σ_x(z) = σ_PSFx * sqrt(α_x(z))
- σ_y(z) = σ_PSFy * sqrt(α_y(z)) 
- α_x(z) = 1 + ((z-γ)/d)² + ax*((z-γ)/d)³ + bx*((z-γ)/d)⁴
- α_y(z) = 1 + ((z+γ)/d)² + ay*((z+γ)/d)³ + by*((z+γ)/d)⁴

Where γ, d, Ax, Ay, Bx, By, σ_x0, σ_y0 are calibration parameters.
"""

mutable struct θ_xynbz{T} <: GaussMLEParams{T}
    modeltype::Int
    nparams::Int
    x::T
    y::T
    z::T        # axial position
    n::T        # intensity 
    bg::T       # background

    θ_xynbz(x::T, y::T, z::T, n::T, bg::T) where {T} = new{T}(4, 5, x, y, z, n, bg)
end

function θ_xynbz()
    return θ_xynbz(0.0, 0.0, 0.0, 0.0, 0.0)
end

function θ_xynbz{T}() where T
    return θ_xynbz(T(0), T(0), T(0), T(0), T(0))
end

"""
Calibration parameters for astigmatic PSF z-fitting.

These parameters are measured during calibration by fitting beads at known z-positions
and characterizing how the PSF width changes with depth.
"""
mutable struct AstigmaticCalibration{T}
    # Base PSF widths (at z=0, α=1)
    σ_x0::T    # Base PSF width in x direction
    σ_y0::T    # Base PSF width in y direction
    
    # Astigmatism parameters
    γ::T       # Astigmatism offset (focal plane difference between x and y)
    d::T       # Depth scaling parameter
    
    # Higher-order correction terms
    Ax::T      # Third-order correction for x direction
    Ay::T      # Third-order correction for y direction  
    Bx::T      # Fourth-order correction for x direction
    By::T      # Fourth-order correction for y direction
end

function AstigmaticCalibration{T}() where T
    return AstigmaticCalibration{T}(
        T(1.3),   # σ_x0
        T(1.3),   # σ_y0
        T(0.0),   # γ
        T(1.0),   # d
        T(0.0),   # Ax
        T(0.0),   # Ay  
        T(0.0),   # Bx
        T(0.0)    # By
    )
end

mutable struct Args_xynbz{T} <: GaussMLEArgs{T}
    calib::AstigmaticCalibration{T}
end

function Args_xynbz{T}() where T
    return Args_xynbz{T}(AstigmaticCalibration{T}())
end

mutable struct Σ_xynbz{T} <: GaussMLEΣ{T}
    σ_x::T      # Uncertainty in x position
    σ_y::T      # Uncertainty in y position  
    σ_z::T      # Uncertainty in z position
    σ_n::T      # Uncertainty in intensity
    σ_bg::T     # Uncertainty in background
    logL::T     # Log-likelihood
end

function Σ_xynbz()
    return Σ_xynbz(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function Σ_xynbz{T}() where T
    return Σ_xynbz(T(0), T(0), T(0), T(0), T(0), T(0))
end

"""
    compute_psf_widths(z::T, calib::AstigmaticCalibration{T}) where T

Compute the PSF widths at axial position z using calibration parameters.
"""
function compute_psf_widths(z::T, calib::AstigmaticCalibration{T}) where T
    # Compute alpha factors using existing library functions
    α_x = GaussLib.compute_alpha(z - calib.γ, calib.Ax, calib.Bx, calib.d)
    α_y = GaussLib.compute_alpha(z + calib.γ, calib.Ay, calib.By, calib.d)
    
    # Compute PSF widths
    σ_x = calib.σ_x0 * sqrt(α_x)
    σ_y = calib.σ_y0 * sqrt(α_y)
    
    return σ_x, σ_y
end

"""
    model(θ::θ_xynbz{T}, args::Args_xynbz{T}, i::Int, j::Int) where T

Compute the model value at pixel (i,j).
"""
function model(θ::θ_xynbz{T}, args::Args_xynbz{T}, i::Int, j::Int) where T
    # Compute PSF widths at current z position
    σ_x, σ_y = compute_psf_widths(θ.z, args.calib)
    
    # Compute 2D Gaussian integral over pixel
    PSF_x = GaussLib.integral_gaussian_1d(i, θ.x, σ_x)
    PSF_y = GaussLib.integral_gaussian_1d(j, θ.y, σ_y)
    
    return θ.bg + θ.n * PSF_x * PSF_y
end

"""
    gradient!(grad::AbstractArray{T}, θ::θ_xynbz{T}, args::Args_xynbz{T}, i::Int, j::Int) where T

Compute gradient with respect to all parameters at pixel (i,j).
"""
function gradient!(grad::AbstractArray{T}, θ::θ_xynbz{T}, args::Args_xynbz{T}, i, j) where T
    # Compute PSF widths and derivatives
    σ_x, σ_y = compute_psf_widths(θ.z, args.calib)
    
    # Compute PSF integrals
    PSF_x = GaussLib.integral_gaussian_1d(i, θ.x, σ_x)
    PSF_y = GaussLib.integral_gaussian_1d(j, θ.y, σ_y)
    
    # Gradients with respect to position
    (dPSF_x_dx, _) = GaussLib.derivative_integral_gaussian_1d(i, θ.x, σ_x, θ.n, PSF_y)
    (dPSF_y_dy, _) = GaussLib.derivative_integral_gaussian_1d(j, θ.y, σ_y, θ.n, PSF_x)
    
    grad[1] = dPSF_x_dx  # ∂/∂x
    grad[2] = dPSF_y_dy  # ∂/∂y
    
    # For z gradient, we need to use the specialized z-derivative function
    dudt_temp = zeros(T, 5)
    d2udt2_temp = zeros(T, 5) 
    theta_array = [θ.x, θ.y, θ.n, θ.bg, θ.z]
    
    GaussLib.derivative_integral_gaussian_2d_z(
        i, j, theta_array, 
        args.calib.σ_x0, args.calib.σ_y0,
        args.calib.Ax, args.calib.Ay, 
        args.calib.Bx, args.calib.By,
        args.calib.γ, args.calib.d,
        dudt_temp, d2udt2_temp
    )
    
    grad[3] = dudt_temp[5]  # ∂/∂z
    grad[4] = PSF_x * PSF_y  # ∂/∂n (intensity)
    grad[5] = T(1)           # ∂/∂bg (background)
end

"""
    curvature!(grad::AbstractArray{T}, hessdiag::AbstractArray{T}, θ::θ_xynbz{T}, args::Args_xynbz{T}, i::Int, j::Int) where T

Compute diagonal Hessian elements at pixel (i,j).
"""
function curvature!(grad::AbstractArray{T}, hessdiag::AbstractArray{T}, θ::θ_xynbz{T}, args::Args_xynbz{T}, i, j) where T
    # Compute PSF widths
    σ_x, σ_y = compute_psf_widths(θ.z, args.calib)
    
    # Compute PSF integrals
    PSF_x = GaussLib.integral_gaussian_1d(i, θ.x, σ_x)
    PSF_y = GaussLib.integral_gaussian_1d(j, θ.y, σ_y)
    
    # Second derivatives with respect to position
    (_, d2PSF_x_dx2) = GaussLib.derivative_integral_gaussian_1d(i, θ.x, σ_x, θ.n, PSF_y)
    (_, d2PSF_y_dy2) = GaussLib.derivative_integral_gaussian_1d(j, θ.y, σ_y, θ.n, PSF_x)
    
    hessdiag[1] = d2PSF_x_dx2  # ∂²/∂x²
    hessdiag[2] = d2PSF_y_dy2  # ∂²/∂y²
    
    # For z curvature, use the specialized function
    dudt_temp = zeros(T, 5)
    d2udt2_temp = zeros(T, 5)
    theta_array = [θ.x, θ.y, θ.n, θ.bg, θ.z]
    
    GaussLib.derivative_integral_gaussian_2d_z(
        i, j, theta_array,
        args.calib.σ_x0, args.calib.σ_y0,
        args.calib.Ax, args.calib.Ay,
        args.calib.Bx, args.calib.By, 
        args.calib.γ, args.calib.d,
        dudt_temp, d2udt2_temp
    )
    
    hessdiag[3] = d2udt2_temp[5]  # ∂²/∂z²
    
    # For intensity and background, use small positive values to avoid singular matrix
    # These represent the curvature of the log-likelihood, not the model itself
    hessdiag[4] = T(1e-6) / max(θ.n, T(1))     # ∂²/∂n² (approximate)
    hessdiag[5] = T(1e-6)                       # ∂²/∂bg² (approximate)
end

"""
    compute_all!(model_val::Ref{T}, grad::AbstractArray{T}, hessdiag::AbstractArray{T}, 
                 θ::θ_xynbz{T}, args::Args_xynbz{T}, i::Int, j::Int) where T

Compute model value, gradient, and Hessian diagonal simultaneously for efficiency.
"""
function compute_all!(model_val::Ref{T}, grad::AbstractArray{T}, hessdiag::AbstractArray{T}, 
                     θ::θ_xynbz{T}, args::Args_xynbz{T}, i::Int, j::Int) where T
    # Compute model value
    model_val[] = model(θ, args, i, j)
    
    # Compute gradient
    gradient!(grad, θ, args, i, j)
    
    # Compute curvature
    curvature!(grad, hessdiag, θ, args, i, j)
end

"""
    initialize_parameters!(θ::θ_xynbz{T}, data::Matrix{T}, boxsize::Int, args::Args_xynbz{T}) where T

Initialize parameters using center of mass and intensity estimates.
"""
function initialize_parameters!(θ::θ_xynbz{T}, data::Matrix{T}, boxsize::Int, args::Args_xynbz{T}) where T
    # Use center of mass for x,y initialization
    θ.x, θ.y = GaussLib.center_of_mass_2d(boxsize, data)
    
    # Initialize background as minimum value
    θ.bg = minimum(data)
    
    # Initialize intensity as integrated signal above background
    θ.n = max(sum(data) - θ.bg * boxsize^2, T(100))
    
    # Initialize z at focal plane (γ offset)
    θ.z = args.calib.γ
end

"""
    update!(θ::θ_xynbz{T}, numerator::Vector{T}, denominator::Vector{T}) where T

Update parameters using Newton-Raphson step.
"""
function update!(θ::θ_xynbz{T}, numerator::Vector{T}, denominator::Vector{T}) where T
    max_step = T(1.0)  # Maximum step size
    converged = true
    
    # Update x
    if abs(denominator[1]) > T(1e-12)
        step = numerator[1] / denominator[1]
        step = clamp(step, -max_step, max_step)
        θ.x += step
        converged = converged && abs(step) < T(1e-6)
    end
    
    # Update y  
    if abs(denominator[2]) > T(1e-12)
        step = numerator[2] / denominator[2]
        step = clamp(step, -max_step, max_step)
        θ.y += step
        converged = converged && abs(step) < T(1e-6)
    end
    
    # Update z with constraints
    if abs(denominator[3]) > T(1e-12)
        step = numerator[3] / denominator[3]
        step = clamp(step, -max_step, max_step)
        θ.z += step
        converged = converged && abs(step) < T(1e-6)
    end
    
    # Update intensity with positivity constraint
    if abs(denominator[4]) > T(1e-12)
        step = numerator[4] / denominator[4]
        step = clamp(step, -max_step, max_step)
        θ.n = max(θ.n + step, T(1))
        converged = converged && abs(step) < T(1e-6)
    end
    
    # Update background with positivity constraint
    if abs(denominator[5]) > T(1e-12)
        step = numerator[5] / denominator[5]
        step = clamp(step, -max_step, max_step)
        θ.bg = max(θ.bg + step, T(0))
        converged = converged && abs(step) < T(1e-6)
    end
    
    return converged
end

"""
    genargs(::Type{θ_xynbz}; T::Type{<:Real}=Float32, calib::Union{Nothing,AstigmaticCalibration}=nothing)

Generate arguments for the astigmatic z-model.
"""
function genargs(::Type{θ_xynbz}; T::Type{<:Real}=Float32, calib::Union{Nothing,AstigmaticCalibration}=nothing)
    if calib === nothing
        calib = AstigmaticCalibration{T}()
    end
    return Args_xynbz{T}(calib)
end

"""
    genθ(::Type{θ_xynbz}, boxsize::Int; T::Type{<:Real}=Float32)

Generate parameter struct for the astigmatic z-model.
"""
function genθ(::Type{θ_xynbz}, boxsize::Int; T::Type{<:Real}=Float32)
    return θ_xynbz{T}()
end

"""
    genΣ(::Type{θ_xynbz}; T::Type{<:Real}=Float32)

Generate uncertainty struct for the astigmatic z-model.
"""
function genΣ(::Type{θ_xynbz}; T::Type{<:Real}=Float32)
    return Σ_xynbz{T}()
end

# Fill methods for CRLB
function fill!(σ::Σ_xynbz{T}, values::AbstractArray{T}, logL::T) where T <: Real
    σ.σ_x = values[1]
    σ.σ_y = values[2]
    σ.σ_z = values[3]
    σ.σ_n = values[4]
    σ.σ_bg = values[5]
    σ.logL = logL
    return nothing
end

function fill!(σ::Σ_xynbz{T}, values::AbstractArray{T}) where T <: Real
    σ.σ_x = values[1]
    σ.σ_y = values[2]
    σ.σ_z = values[3]
    σ.σ_n = values[4]
    σ.σ_bg = values[5]
    return nothing
end