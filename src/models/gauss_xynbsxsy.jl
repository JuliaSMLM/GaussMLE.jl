"""
6-parameter Gaussian model with asymmetric PSF: x, y, intensity, bg, σ_x, σ_y

This model extends the basic 4-parameter Gaussian to include separate PSF widths
in x and y directions, allowing fitting of asymmetric point spread functions.
"""

mutable struct θ_xynbsxsy{T} <: GaussMLEParams{T}
    modeltype::Int
    nparams::Int
    x::T
    y::T
    n::T        # intensity
    bg::T       # background
    σ_x::T      # PSF width in x direction
    σ_y::T      # PSF width in y direction

    θ_xynbsxsy(x::T, y::T, n::T, bg::T, σ_x::T, σ_y::T) where {T} = new{T}(3, 6, x, y, n, bg, σ_x, σ_y) 
end

function θ_xynbsxsy()
    return θ_xynbsxsy(0.0, 0.0, 0.0, 0.0, 1.3, 1.3)
end

mutable struct Args_xynbsxsy{T} <: GaussMLEArgs{T}
    # No additional arguments needed - all parameters are fitted
end

mutable struct Σ_xynbsxsy{T} <: GaussMLEΣ{T}
    σ_x::T      # Uncertainty in x position
    σ_y::T      # Uncertainty in y position
    σ_n::T      # Uncertainty in intensity
    σ_bg::T     # Uncertainty in background
    σ_σx::T     # Uncertainty in σ_x parameter
    σ_σy::T     # Uncertainty in σ_y parameter
    logL::T     # Log-likelihood
    
    Σ_xynbsxsy(σ_x::T, σ_y::T, σ_n::T, σ_bg::T, σ_σx::T, σ_σy::T, logL::T) where {T} = new{T}(σ_x, σ_y, σ_n, σ_bg, σ_σx, σ_σy, logL)
end

# Helper function for θ_xynbsxsy
function compute_all!(model_val::Union{Nothing, Ref{T}}, grad::Union{Nothing, AbstractArray{T}}, 
                     hessdiag::Union{Nothing, AbstractArray{T}}, θ::θ_xynbsxsy{T}, args::Args_xynbsxsy{T}, i, j) where {T<:Real}
    
    # Pixel coordinates (convert to T type)
    xi = T(j)  # x corresponds to column
    yi = T(i)  # y corresponds to row
    
    # Distance from center
    dx = xi - θ.x
    dy = yi - θ.y
    
    # Asymmetric Gaussian calculation
    σx2 = θ.σ_x * θ.σ_x
    σy2 = θ.σ_y * θ.σ_y
    
    # Normalization factor for 2D asymmetric Gaussian
    norm_factor = T(1) / (T(2π) * θ.σ_x * θ.σ_y)
    
    # Exponential term
    exp_term = exp(-(dx*dx)/(T(2)*σx2) - (dy*dy)/(T(2)*σy2))
    
    # Gaussian component
    gauss = θ.n * norm_factor * exp_term
    
    # Total model value
    model = θ.bg + gauss
    
    # Store model value if requested
    if model_val !== nothing
        model_val[] = model
    end
    
    # Compute gradients if requested
    if grad !== nothing
        # ∂μ/∂x
        grad[1] = gauss * dx / σx2
        
        # ∂μ/∂y  
        grad[2] = gauss * dy / σy2
        
        # ∂μ/∂n (intensity)
        grad[3] = gauss / θ.n
        
        # ∂μ/∂bg (background)
        grad[4] = T(1)
        
        # ∂μ/∂σ_x
        grad[5] = gauss * ((dx*dx)/σx2 - T(1)) / θ.σ_x
        
        # ∂μ/∂σ_y
        grad[6] = gauss * ((dy*dy)/σy2 - T(1)) / θ.σ_y
    end
    
    # Compute Hessian diagonal if requested
    if hessdiag !== nothing
        # Following the pattern from xynb model - compute actual second derivatives for position
        # but use zero/small values for intensity and background
        
        # ∂²μ/∂x² - actual second derivative
        hessdiag[1] = gauss * (T(1)/σx2 - (dx*dx)/(σx2*σx2))
        
        # ∂²μ/∂y² - actual second derivative
        hessdiag[2] = gauss * (T(1)/σy2 - (dy*dy)/(σy2*σy2))
        
        # ∂²μ/∂n² - set to 0 like in xynb model
        hessdiag[3] = T(0)
        
        # ∂²μ/∂bg² - set to 0 like in xynb model
        hessdiag[4] = T(0)
        
        # ∂²μ/∂σ_x² - actual second derivative
        dx2_term = (dx*dx)/σx2
        hessdiag[5] = gauss * ((dx2_term*dx2_term - T(3)*dx2_term + T(2)) / (θ.σ_x * θ.σ_x))
        
        # ∂²μ/∂σ_y² - actual second derivative
        dy2_term = (dy*dy)/σy2
        hessdiag[6] = gauss * ((dy2_term*dy2_term - T(3)*dy2_term + T(2)) / (θ.σ_y * θ.σ_y))
    end
end

function model(θ::θ_xynbsxsy{T}, args::Args_xynbsxsy{T}, i, j) where {T<:Real}
    model_val = Ref{T}()
    compute_all!(model_val, nothing, nothing, θ, args, i, j)
    return model_val[]
end

function gradient!(grad::AbstractArray{T}, θ::θ_xynbsxsy{T}, args::Args_xynbsxsy{T}, i, j) where {T<:Real}
    compute_all!(nothing, grad, nothing, θ, args, i, j)
end

function curvature!(grad::AbstractArray{T}, hessdiag::AbstractArray{T}, θ::θ_xynbsxsy{T}, args::Args_xynbsxsy{T}, i, j) where {T<:Real}
    val = Ref{T}(0)
    compute_all!(val, grad, hessdiag, θ, args, i, j)
    return val[]
end

function initialize_parameters!(θ::θ_xynbsxsy{T}, data::Matrix{T}, boxsize::Int, args::Args_xynbsxsy{T}) where {T<:Real}
    # Find minimum value as background estimate
    θ.bg = minimum(data)
    
    # Calculate center of mass for initial position
    total_signal = T(0)
    x_weighted = T(0)
    y_weighted = T(0)
    
    for j in 1:boxsize
        for i in 1:boxsize
            signal = max(data[i, j] - θ.bg, T(0))
            total_signal += signal
            x_weighted += signal * T(j)  # x corresponds to column
            y_weighted += signal * T(i)  # y corresponds to row
        end
    end
    
    if total_signal > T(0)
        θ.x = x_weighted / total_signal
        θ.y = y_weighted / total_signal
        θ.n = total_signal
    else
        θ.x = T(boxsize + 1) / T(2)
        θ.y = T(boxsize + 1) / T(2)
        θ.n = T(100)
    end
    
    # Initialize PSF widths to reasonable defaults
    θ.σ_x = T(1.3)
    θ.σ_y = T(1.3)
end

function update!(θ::θ_xynbsxsy{T}, numerator::AbstractArray{T}, denominator::AbstractArray{T}) where {T<:Real}
    # Newton-Raphson parameter updates with improved stability
    
    # Maximum step size for stability
    max_step = T(1.0)
    
    # Position updates with step limiting and sign correction (like xynb model)
    if abs(denominator[1]) > T(1e-12)
        step_x = -numerator[1] / denominator[1]  # Note negative sign
        step_x = clamp(step_x, -max_step, max_step)
        θ.x = clamp(θ.x + step_x, T(0.5), T(20.0))
    end
    
    if abs(denominator[2]) > T(1e-12)
        step_y = -numerator[2] / denominator[2]  # Note negative sign
        step_y = clamp(step_y, -max_step, max_step)
        θ.y = clamp(θ.y + step_y, T(0.5), T(20.0))
    end
    
    # Intensity update with positivity constraint
    if abs(denominator[3]) > T(1e-12)
        step_n = -numerator[3] / denominator[3]  # Note negative sign
        step_n = clamp(step_n, -θ.n * T(0.5), θ.n * T(2.0))  # Limit to 50% decrease or 200% increase
        θ.n = max(θ.n + step_n, T(1))
    end
    
    # Background update with non-negativity constraint
    if abs(denominator[4]) > T(1e-12)
        step_bg = -numerator[4] / denominator[4]  # Note negative sign
        step_bg = clamp(step_bg, -θ.bg * T(0.5), θ.bg * T(2.0) + T(10))
        θ.bg = max(θ.bg + step_bg, T(0))
    end
    
    # PSF width updates with strict bounds
    if abs(denominator[5]) > T(1e-12)
        step_σx = -numerator[5] / denominator[5]  # Note negative sign
        step_σx = clamp(step_σx, -θ.σ_x * T(0.3), θ.σ_x * T(0.3))  # Limit to 30% change per iteration
        θ.σ_x = clamp(θ.σ_x + step_σx, T(0.5), T(3.0))
    end
    
    if abs(denominator[6]) > T(1e-12)
        step_σy = -numerator[6] / denominator[6]  # Note negative sign
        step_σy = clamp(step_σy, -θ.σ_y * T(0.3), θ.σ_y * T(0.3))  # Limit to 30% change per iteration
        θ.σ_y = clamp(θ.σ_y + step_σy, T(0.5), T(3.0))
    end
    
    # Check convergence based on step sizes
    max_rel_change = T(0)
    if abs(denominator[1]) > T(1e-12)
        max_rel_change = max(max_rel_change, abs(numerator[1] / denominator[1]) / max(abs(θ.x), T(1)))
    end
    if abs(denominator[2]) > T(1e-12)
        max_rel_change = max(max_rel_change, abs(numerator[2] / denominator[2]) / max(abs(θ.y), T(1)))
    end
    if abs(denominator[3]) > T(1e-12)
        max_rel_change = max(max_rel_change, abs(numerator[3] / denominator[3]) / max(abs(θ.n), T(1)))
    end
    if abs(denominator[4]) > T(1e-12)
        max_rel_change = max(max_rel_change, abs(numerator[4] / denominator[4]) / max(abs(θ.bg), T(1)))
    end
    if abs(denominator[5]) > T(1e-12)
        max_rel_change = max(max_rel_change, abs(numerator[5] / denominator[5]) / max(abs(θ.σ_x), T(1)))
    end
    if abs(denominator[6]) > T(1e-12)
        max_rel_change = max(max_rel_change, abs(numerator[6] / denominator[6]) / max(abs(θ.σ_y), T(1)))
    end
    
    return max_rel_change < T(1e-6)
end

function genargs(::Type{θ_xynbsxsy}; T::Type{<:Real}=Float32)
    return Args_xynbsxsy{T}()
end

function genθ(::Type{θ_xynbsxsy}, boxsize::Int; T::Type{<:Real}=Float32)
    x = T(boxsize + 1) / T(2)
    y = T(boxsize + 1) / T(2)
    n = T(500)
    bg = T(2)
    σ_x = T(1.3)
    σ_y = T(1.3)
    return θ_xynbsxsy(x, y, n, bg, σ_x, σ_y)
end

function genΣ(::Type{θ_xynbsxsy}; T::Type{<:Real}=Float32)
    return Σ_xynbsxsy(T(0.1), T(0.1), T(10), T(0.5), T(0.1), T(0.1), T(0))
end

# Fill methods for CRLB
function fill!(σ::Σ_xynbsxsy{T}, values::AbstractArray{T}, logL::T) where T <: Real
    σ.σ_x = values[1]
    σ.σ_y = values[2]
    σ.σ_n = values[3]
    σ.σ_bg = values[4]
    σ.σ_σx = values[5]
    σ.σ_σy = values[6]
    σ.logL = logL
    return nothing
end

function fill!(σ::Σ_xynbsxsy{T}, values::AbstractArray{T}) where T <: Real
    σ.σ_x = values[1]
    σ.σ_y = values[2]
    σ.σ_n = values[3]
    σ.σ_bg = values[4]
    σ.σ_σx = values[5]
    σ.σ_σy = values[6]
    return nothing
end