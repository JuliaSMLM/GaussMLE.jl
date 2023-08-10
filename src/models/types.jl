
abstract type GaussMLEModel end
abstract type GaussMLEParams{T <: Real} end
abstract type GaussMLEArgs{T <: Real} end
abstract type GaussMLEΣ{T <: Real} end

function model(θ::GaussMLEParams, args::GaussMLEArgs)
    @error "model not implemented"
end







