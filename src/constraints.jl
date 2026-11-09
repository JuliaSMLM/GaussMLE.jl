"""
Parameter constraint system for optimization
"""

# Flexible constraint system
struct ParameterConstraints{N}
    lower::SVector{N, Float32}
    upper::SVector{N, Float32}
    max_step::SVector{N, Float32}
end

# Type alias for convenience
const Params{N} = SVector{N, Float32}

# x, y bounds symmetric about the box: pixel centers are 1..box_size, so the box spans
# 0.5..box_size+0.5, and the bounds reach 1.5 px past each edge.
_default_position_bounds(box_size) = (-1.0f0, Float32(box_size + 2))

# Default constraints for each model
function default_constraints(::GaussianXYNB, box_size)
    lower, upper = _default_position_bounds(box_size)
    return ParameterConstraints{4}(
        Params{4}(lower, lower, 1.0f0, 0.01f0), # lower bounds
        Params{4}(upper, upper, Inf32, Inf32),  # upper bounds
        Params{4}(1.0f0, 1.0f0, Inf32, Inf32)   # max step
    )
end

function default_constraints(::GaussianXYNBS, box_size)
    lower, upper = _default_position_bounds(box_size)
    return ParameterConstraints{5}(
        Params{5}(lower, lower, 1.0f0, 0.01f0, 0.3f0), # lower bounds
        Params{5}(upper, upper, Inf32, Inf32, 5.0f0),  # upper bounds
        Params{5}(1.0f0, 1.0f0, Inf32, Inf32, 0.5f0)   # max step
    )
end

function default_constraints(::GaussianXYNBSXSY, box_size)
    lower, upper = _default_position_bounds(box_size)
    return ParameterConstraints{6}(
        Params{6}(lower, lower, 1.0f0, 0.01f0, 0.3f0, 0.3f0), # lower bounds
        Params{6}(upper, upper, Inf32, Inf32, 5.0f0, 5.0f0),  # upper bounds
        Params{6}(1.0f0, 1.0f0, Inf32, Inf32, 0.5f0, 0.5f0)   # max step
    )
end

function default_constraints(::AstigmaticXYZNB, box_size)
    # Note: x,y in pixels, z in microns (physical units for axial position)
    lower, upper = _default_position_bounds(box_size)
    return ParameterConstraints{5}(
        Params{5}(lower, lower, -1.0f0, 1.0f0, 0.01f0), # lower: z = -1μm
        Params{5}(upper, upper, 1.0f0, Inf32, Inf32),   # upper: z = +1μm
        Params{5}(1.0f0, 1.0f0, 0.1f0, Inf32, Inf32)   # max_step: z = 0.1μm
    )
end

# Default cap on the Levenberg-Marquardt pixel passes of a fit. The Gaussian fits stop once
# converged, most within 10 passes, so the cap binds only on a slow fit: Fisher scoring
# converges linearly where the observed curvature differs from F, as for z at low counts.
# A fit capped before it converges keeps its CRLB and is left out of `n_converged`. In
# simulations of every Gaussian model (2,000 to 20,000 ROIs per condition, Poisson noise
# with sCMOS read noise of 0 to 5 e-), up to 5 % of low-count astigmatic fits (300 to 500
# photons) needed more than 20 passes. At 100, up to 0.3 % of emitters inside the z range
# had not stopped (PSFs at 1e6 photons or more, narrow PSFs, low-count astigmatic fits
# with 5 e- read noise), about half of them within 0.1 CRLB of their maximum; near the
# ends of the z range up to 8.5 % had not. SplinePSFModel runs every pass from each start (its own method).
_default_iterations(::PSFModel) = 100

_resolve_constraints(constraints::ParameterConstraints, ::PSFModel, box_size) = constraints

function _resolve_constraints(::Nothing, psf_model::PSFModel, box_size)
    return default_constraints(psf_model, box_size)
end
