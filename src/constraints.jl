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
        Params{4}(1.0f0, 1.0f0, 100.0f0, 2.0f0) # max step
    )
end

function default_constraints(::GaussianXYNBS, box_size)
    lower, upper = _default_position_bounds(box_size)
    return ParameterConstraints{5}(
        Params{5}(lower, lower, 1.0f0, 0.01f0, 0.3f0), # lower bounds
        Params{5}(upper, upper, Inf32, Inf32, 5.0f0),  # upper bounds
        Params{5}(1.0f0, 1.0f0, 100.0f0, 2.0f0, 0.5f0) # max step
    )
end

function default_constraints(::GaussianXYNBSXSY, box_size)
    lower, upper = _default_position_bounds(box_size)
    return ParameterConstraints{6}(
        Params{6}(lower, lower, 1.0f0, 0.01f0, 0.3f0, 0.3f0), # lower bounds
        Params{6}(upper, upper, Inf32, Inf32, 5.0f0, 5.0f0),  # upper bounds
        Params{6}(1.0f0, 1.0f0, 100.0f0, 2.0f0, 0.5f0, 0.5f0) # max step
    )
end

function default_constraints(::AstigmaticXYZNB, box_size)
    # Note: x,y in pixels, z in microns (physical units for axial position)
    lower, upper = _default_position_bounds(box_size)
    return ParameterConstraints{5}(
        Params{5}(lower, lower, -1.0f0, 1.0f0, 0.01f0), # lower: z = -1μm
        Params{5}(upper, upper, 1.0f0, Inf32, Inf32),   # upper: z = +1μm
        Params{5}(1.0f0, 1.0f0, 0.1f0, 100.0f0, 2.0f0)  # max_step: z = 0.1μm
    )
end

_resolve_constraints(constraints::ParameterConstraints, ::PSFModel, box_size) = constraints

function _resolve_constraints(::Nothing, psf_model::PSFModel, box_size)
    return default_constraints(psf_model, box_size)
end

@inline function apply_constraints!(θ::Params{N}, Δθ::Params{N}, 
                                   constraints::ParameterConstraints{N}) where N
    # Apply step size limits and bounds
    θ_new = θ - clamp.(Δθ, -constraints.max_step, constraints.max_step)
    return clamp.(θ_new, constraints.lower, constraints.upper)
end