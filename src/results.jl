"""
Results structure for Gaussian MLE fitting
"""

"""
    FitInfo

Metadata about a fit operation, returned as the second element of the fit() tuple.

# Fields
- `elapsed_ns::UInt64`: Elapsed time in nanoseconds (from `time_ns()`)
- `backend::Symbol`: Actual execution backend (`:cpu` or `:gpu`, never `:auto`)
- `device_id::Int`: GPU device index (0-based) or -1 for CPU
- `n_fits::Int`: Number of ROIs attempted
- `n_converged::Int`: Number of ROIs that converged (currently equals n_fits since all iterations run)
- `batch_size::Int`: Actual batch size used for processing
- `n_batches::Int`: Number of batches processed (1 if all-at-once)
- `memory_per_batch::Int`: Estimated memory bytes per batch

# Example
```julia
smld, info = fit(batch, fitter)
println("Fit \$(info.n_fits) ROIs in \$(info.elapsed_ns / 1e6) ms on \$(info.backend)")
println("Processed in \$(info.n_batches) batches of \$(info.batch_size)")
```

# See also
[`fit`](@ref), [`GaussMLEFitter`](@ref)
"""
struct FitInfo
    elapsed_ns::UInt64
    backend::Symbol
    device_id::Int
    n_fits::Int
    n_converged::Int
    batch_size::Int
    n_batches::Int
    memory_per_batch::Int
end

function Base.show(io::IO, info::FitInfo)
    elapsed_ms = info.elapsed_ns / 1e6
    device_str = info.backend == :gpu ? "GPU:$(info.device_id)" : "CPU"
    mem_str = info.memory_per_batch > 1024^2 ? "$(round(info.memory_per_batch / 1024^2, digits=1)) MB" : "$(round(info.memory_per_batch / 1024, digits=1)) KB"
    print(io, "FitInfo($(info.n_fits) fits, $(round(elapsed_ms, digits=2)) ms, $device_str, $(info.n_batches) batches × $(info.batch_size), $mem_str/batch)")
end

export FitInfo

"""
    GaussMLEResults{T,P}

Results from Maximum Likelihood Estimation fitting.

# Fields
- `parameters::Matrix{T}`: Fitted parameters (n_params × n_fits)
- `uncertainties::Matrix{T}`: Parameter uncertainties from CRLB (n_params × n_fits)
- `log_likelihoods::Vector{T}`: Log-likelihood value for each fit
- `psf_model::P`: PSF model used for fitting
- `n_fits::Int`: Number of fits performed

# Property Accessors
Convenient property access is provided based on the PSF model:

For `GaussianXYNB` (4 parameters):
- `.x, .y, .photons, .background`: Parameter values
- `.x_error, .y_error, .photons_error, .background_error`: Uncertainties

For `GaussianXYNBS` (5 parameters):
- Adds: `.sigma, .sigma_error`

For `GaussianXYNBSXSY` (6 parameters):
- Adds: `.sigma_x, .sigma_y, .sigma_x_error, .sigma_y_error`

For `AstigmaticXYZNB` (5 parameters):
- `.x, .y, .z, .photons, .background` and corresponding `_error` fields

# Examples
```julia
results = fit(fitter, data)

# Access parameters
mean_x = mean(results.x)
mean_precision = mean(results.x_error)

# Access raw matrices
all_params = results.parameters  # n_params × n_fits
```

# See also
[`fit`](@ref), [`GaussMLEFitter`](@ref)
"""
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
Base.getproperty(r::GaussMLEResults{T, <:GaussianXYNBS}, s::Symbol) where T =
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
Base.getproperty(r::GaussMLEResults{T, <:GaussianXYNBSXSY}, s::Symbol) where T =
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

# Iterator interface - enables `for fit in results`
"""
    iterate(results::GaussMLEResults, state=1)

Iterate over individual fits in the results.

# Example
```julia
results = fit(fitter, data)

for fit in results
    println("Position: (", fit.parameters[1], ", ", fit.parameters[2], ")")
    println("Precision: ", fit.uncertainties[1])
end
```
"""
function Base.iterate(r::GaussMLEResults, state=1)
    if state > r.n_fits
        return nothing
    end
    fit = (
        parameters = r.parameters[:, state],
        uncertainties = r.uncertainties[:, state],
        log_likelihood = r.log_likelihoods[state],
        index = state
    )
    return (fit, state + 1)
end

# Length and indexing support
Base.length(r::GaussMLEResults) = r.n_fits
Base.size(r::GaussMLEResults) = (r.n_fits,)

"""
    getindex(results::GaussMLEResults, i::Int)

Get the i-th fit from the results.

# Returns
A NamedTuple with fields:
- `parameters`: Parameter vector for this fit
- `uncertainties`: Uncertainty vector for this fit
- `log_likelihood`: Log-likelihood value
- `index`: Fit index

# Example
```julia
results = fit(fitter, data)
fit_10 = results[10]
println("Fit 10 position: ", fit_10.parameters[1:2])
```
"""
function Base.getindex(r::GaussMLEResults, i::Int)
    if i < 1 || i > r.n_fits
        throw(BoundsError(r, i))
    end
    return (
        parameters = r.parameters[:, i],
        uncertainties = r.uncertainties[:, i],
        log_likelihood = r.log_likelihoods[i],
        index = i
    )
end

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