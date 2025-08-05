"""
Results structure for Gaussian MLE fitting
"""

# Results structure
struct GaussMLEResults{T, P<:PSFModel}
    parameters::Matrix{T}
    uncertainties::Matrix{T}
    log_likelihoods::Vector{T}
    psf_model::P
    n_fits::Int
end

# Convenient accessors for GaussianXYNB (4 parameters)
Base.getproperty(r::GaussMLEResults{T, <:GaussianXYNB}, s::Symbol) where T = 
    s === :x ? r.parameters[1, :] :
    s === :y ? r.parameters[2, :] :
    s === :photons ? r.parameters[3, :] :
    s === :background ? r.parameters[4, :] :
    s === :x_error ? r.uncertainties[1, :] :
    s === :y_error ? r.uncertainties[2, :] :
    s === :photons_error ? r.uncertainties[3, :] :
    s === :background_error ? r.uncertainties[4, :] :
    getfield(r, s)

# Convenient accessors for GaussianXYNBS (5 parameters)
Base.getproperty(r::GaussMLEResults{T, GaussianXYNBS}, s::Symbol) where T = 
    s === :x ? r.parameters[1, :] :
    s === :y ? r.parameters[2, :] :
    s === :photons ? r.parameters[3, :] :
    s === :background ? r.parameters[4, :] :
    s === :sigma ? r.parameters[5, :] :
    s === :x_error ? r.uncertainties[1, :] :
    s === :y_error ? r.uncertainties[2, :] :
    s === :photons_error ? r.uncertainties[3, :] :
    s === :background_error ? r.uncertainties[4, :] :
    s === :sigma_error ? r.uncertainties[5, :] :
    getfield(r, s)

# Convenient accessors for GaussianXYNBSXSY (6 parameters)
Base.getproperty(r::GaussMLEResults{T, GaussianXYNBSXSY}, s::Symbol) where T = 
    s === :x ? r.parameters[1, :] :
    s === :y ? r.parameters[2, :] :
    s === :photons ? r.parameters[3, :] :
    s === :background ? r.parameters[4, :] :
    s === :sigma_x ? r.parameters[5, :] :
    s === :sigma_y ? r.parameters[6, :] :
    s === :x_error ? r.uncertainties[1, :] :
    s === :y_error ? r.uncertainties[2, :] :
    s === :photons_error ? r.uncertainties[3, :] :
    s === :background_error ? r.uncertainties[4, :] :
    s === :sigma_x_error ? r.uncertainties[5, :] :
    s === :sigma_y_error ? r.uncertainties[6, :] :
    getfield(r, s)

# Convenient accessors for AstigmaticXYZNB (5 parameters)
Base.getproperty(r::GaussMLEResults{T, <:AstigmaticXYZNB}, s::Symbol) where T = 
    s === :x ? r.parameters[1, :] :
    s === :y ? r.parameters[2, :] :
    s === :z ? r.parameters[3, :] :
    s === :photons ? r.parameters[4, :] :
    s === :background ? r.parameters[5, :] :
    s === :x_error ? r.uncertainties[1, :] :
    s === :y_error ? r.uncertainties[2, :] :
    s === :z_error ? r.uncertainties[3, :] :
    s === :photons_error ? r.uncertainties[4, :] :
    s === :background_error ? r.uncertainties[5, :] :
    getfield(r, s)

# Pretty printing
function Base.show(io::IO, r::GaussMLEResults)
    println(io, "GaussMLEResults with $(r.n_fits) fits")
    println(io, "  Model: $(typeof(r.psf_model))")
    println(io, "  Mean parameters:")
    
    params = mean(r.parameters, dims=2)
    errors = mean(r.uncertainties, dims=2)
    
    if r.psf_model isa GaussianXYNB
        println(io, "    x: $(params[1]) ± $(errors[1])")
        println(io, "    y: $(params[2]) ± $(errors[2])")
        println(io, "    photons: $(params[3]) ± $(errors[3])")
        println(io, "    background: $(params[4]) ± $(errors[4])")
    elseif r.psf_model isa GaussianXYNBS
        println(io, "    x: $(params[1]) ± $(errors[1])")
        println(io, "    y: $(params[2]) ± $(errors[2])")
        println(io, "    photons: $(params[3]) ± $(errors[3])")
        println(io, "    background: $(params[4]) ± $(errors[4])")
        println(io, "    sigma: $(params[5]) ± $(errors[5])")
    elseif r.psf_model isa GaussianXYNBSXSY
        println(io, "    x: $(params[1]) ± $(errors[1])")
        println(io, "    y: $(params[2]) ± $(errors[2])")
        println(io, "    photons: $(params[3]) ± $(errors[3])")
        println(io, "    background: $(params[4]) ± $(errors[4])")
        println(io, "    sigma_x: $(params[5]) ± $(errors[5])")
        println(io, "    sigma_y: $(params[6]) ± $(errors[6])")
    elseif r.psf_model isa AstigmaticXYZNB
        println(io, "    x: $(params[1]) ± $(errors[1])")
        println(io, "    y: $(params[2]) ± $(errors[2])")
        println(io, "    z: $(params[3]) ± $(errors[3])")
        println(io, "    photons: $(params[4]) ± $(errors[4])")
        println(io, "    background: $(params[5]) ± $(errors[5])")
    end
    
    println(io, "  Mean log-likelihood: $(mean(r.log_likelihoods))")
end

# Export results type
export GaussMLEResults