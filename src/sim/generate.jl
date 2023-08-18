

function genroi!(box::AbstractMatrix{T}, θ::GaussMLEParams{T}, args::GaussMLEArgs{T}) where T <: Real
    for i in 1:size(box,1)
        for j in 1:size(box,2)
            box[i,j] = model(θ, args, i, j)
        end
    end
    return nothing
end


function genstack(boxsize::Int, stackdepth::Int, θ::Vector{<:GaussMLEParams{T}}, args::GaussMLEArgs{T}; poissonnoise::Bool=false) where T <: Real
    stack = zeros(T,boxsize, boxsize, stackdepth)
    for i in 1:stackdepth
        genroi!(view(stack, :,:,i), θ[i], args)
    end    
    if poissonnoise
        poissonnoise!(stack)
    end
    return stack
end


function genstack(boxsize::Int, stackdepth::Int, modelsymbol::Symbol; T::Type{<:Real} = Float32, poissonnoise::Bool = false)  
    modeltype = get(MODEL_MAP, modelsymbol, -1)
    if modeltype == -1
        error("Model symbol not found")
    end     
    args = genargs(modeltype; T)
    θ = [genθ(modeltype, boxsize; T) for i in 1:stackdepth]
    stack = genstack(boxsize, stackdepth, θ, args; poissonnoise=poissonnoise)
    return stack, θ, args
end

function poissonnoise!(stack::Array{T}) where T <: Real
    for idx in eachindex(stack)
        stack[idx] = rand(Poisson(stack[idx]))
    end
    return nothing
end



