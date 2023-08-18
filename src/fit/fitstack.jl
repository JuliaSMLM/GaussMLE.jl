function fitstack(stack::AbstractArray{T}, modelsymbol::Symbol, args::GaussMLEArgs{T}) where T <: Real

    boxsz = size(stack,1)
    nboxes = size(stack,3)
    modeltype = get(MODEL_MAP, modelsymbol, -1)
    if modeltype == -1
        error("Model symbol not found")
    end
    
    θ = [genθ(modeltype, boxsz; T) for i in 1:nboxes]
    Σ = [genΣ(modeltype; T) for i in 1:nboxes]
    
    boxsz = size(stack,1)
    for idx in eachindex(θ)
        fitbox!(θ[idx], Σ[idx], stack[:,:,idx], args)
    end
    return θ, Σ
end

