"""
Legacy compatibility layer for backward compatibility with existing code
"""

# Map old fitstack function to new API
function fitstack(roi_stack::AbstractArray{T,3}, model::Symbol; 
                  σ_PSF::Real=1.3,
                  backend::Symbol=:auto,
                  variance::Union{Nothing,AbstractArray}=nothing,
                  verbose::Bool=false,
                  calib=nothing) where T <: Real
    
    # Map old model symbols to new PSF types
    psf_model = if model == :xynb
        GaussianXYNB(Float32(σ_PSF))
    elseif model == :xynbs
        GaussianXYNBS()
    elseif model == :xynbsxsy
        GaussianXYNBSXSY()
    elseif model == :xynbz
        if !isnothing(calib)
            AstigmaticXYZNB{Float32}(
                calib.σx₀, calib.σy₀, 
                calib.Ax, calib.Ay,
                calib.Bx, calib.By,
                calib.γ, calib.d
            )
        else
            # Default calibration
            AstigmaticXYZNB{Float32}(1.5, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        end
    else
        error("Unknown model: $model")
    end
    
    # Map backend selection
    device = if backend == :cpu
        CPU()
    elseif backend == :gpu
        GPU()
    else
        nothing  # auto-detect
    end
    
    # Create camera model
    camera_model = if !isnothing(variance)
        SCMOSCamera(variance)
    else
        IdealCamera()
    end
    
    # Create fitter
    fitter = GaussMLEFitter(
        psf_model=psf_model,
        camera_model=camera_model,
        device=device
    )
    
    # Perform fitting
    results = fit(fitter, roi_stack)
    
    # Return in old format (parameters, uncertainties)
    return results.parameters, results.uncertainties
end

# Export legacy function
export fitstack