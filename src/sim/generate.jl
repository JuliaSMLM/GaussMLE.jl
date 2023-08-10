


# Mapping from a symbol to the new model type
const MODEL_MAP = Dict(
    :xynb => θ_xynb    
)

function genroi!(box::AbstractMatrix{T}, θ::GaussMLEParams{T}, args::GaussMLEArgs{T}) where T <: Real
    for i in 1:size(box,1)
        for j in 1:size(box,2)
            box[i,j] = model(θ, args, i, j)
        end
    end
    return nothing
end


function genstack(boxsize::Int, stackdepth::Int, θ::Vector{<:GaussMLEParams{T}}, args::GaussMLEArgs{T}) where T <: Real
    stack = zeros(T,boxsize, boxsize, stackdepth)
    for i in 1:stackdepth
        genroi!(view(stack, :,:,i), θ[i], args)
    end    
    return stack
end


function genstack(boxsize::Int, stackdepth::Int, modelsymbol::Symbol; T::Type{<:Real} = Float32)  
    modeltype = get(MODEL_MAP, modelsymbol, -1)
    if modeltype == -1
        error("Model symbol not found")
    end     
    args = genargs(modeltype; T)
    θ = [genparams(modeltype, boxsize; T) for i in 1:stackdepth]
    stack = genstack(boxsize, stackdepth, θ, args)
    return stack, θ, args
end




function gennoisystack(boxsize::Int, stackdepth::Int, modelsymbol::Symbol)
    stack, θ, args = genstack(boxsize, stackdepth, modelsymbol)
    poissonnoise!(stack)
    return stack, θ, args
end

function poissonnoise!(stack::Array{T}) where T <: Real
    for idx in eachindex(stack)
        stack[idx] = rand(T,Poisson(stack[idx]))
    end
    return nothing
end



