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

# Default constraints for each model
function default_constraints(::GaussianXYNB, box_size)
    return ParameterConstraints{4}(
        Params{4}(-2.0f0, -2.0f0, 1.0f0, 0.01f0),        # lower bounds
        Params{4}(box_size+2, box_size+2, Inf32, Inf32), # upper bounds
        Params{4}(1.0f0, 1.0f0, 100.0f0, 2.0f0)          # max step
    )
end

function default_constraints(::GaussianXYNBS, box_size)
    return ParameterConstraints{5}(
        Params{5}(-2.0f0, -2.0f0, 1.0f0, 0.01f0, 0.3f0),       # lower bounds
        Params{5}(box_size+2, box_size+2, Inf32, Inf32, 5.0f0), # upper bounds
        Params{5}(1.0f0, 1.0f0, 100.0f0, 2.0f0, 0.5f0)         # max step
    )
end

function default_constraints(::GaussianXYNBSXSY, box_size)
    return ParameterConstraints{6}(
        Params{6}(-2.0f0, -2.0f0, 1.0f0, 0.01f0, 0.3f0, 0.3f0),       # lower bounds
        Params{6}(box_size+2, box_size+2, Inf32, Inf32, 5.0f0, 5.0f0), # upper bounds
        Params{6}(1.0f0, 1.0f0, 100.0f0, 2.0f0, 0.5f0, 0.5f0)         # max step
    )
end

function default_constraints(::AstigmaticXYZNB, box_size)
    return ParameterConstraints{5}(
        Params{5}(-2.0f0, -2.0f0, -1000.0f0, 1.0f0, 0.01f0),    # lower bounds
        Params{5}(box_size+2, box_size+2, 1000.0f0, Inf32, Inf32), # upper bounds
        Params{5}(1.0f0, 1.0f0, 100.0f0, 100.0f0, 2.0f0)        # max step
    )
end

@inline function apply_constraints!(θ::Params{N}, Δθ::Params{N}, 
                                   constraints::ParameterConstraints{N}) where N
    # Apply step size limits and bounds
    θ_new = θ - clamp.(Δθ, -constraints.max_step, constraints.max_step)
    return clamp.(θ_new, constraints.lower, constraints.upper)
end