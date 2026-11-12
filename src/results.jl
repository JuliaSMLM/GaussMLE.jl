"""
Results structure for Gaussian MLE fitting
"""

"""
    GaussMLEFitInfo

Metadata about a fit operation, returned as the second element of the fit() tuple.

# Fields
- `elapsed_s::Float64`: Elapsed time in seconds
- `backend::Symbol`: Actual execution backend (`:cpu` or `:gpu`, never `:auto`)
- `device_id::Int`: GPU device index (0-based) or -1 for CPU
- `n_fits::Int`: Number of ROIs attempted
- `n_converged::Int`: Number of fits that met their convergence test. A Gaussian fit that
  did not (it reached the `iterations` cap, or its next step rounded to its current point
  in Float32) keeps the CRLB at its end point as its uncertainties; raise `iterations` if
  many reach the cap. A model fitted from several starts (AstigmaticXYZNB, SplinePSFModel)
  counts as converged when the start it keeps did
- `batch_size::Int`: Actual batch size used for processing
- `n_batches::Int`: Number of batches processed (1 if all-at-once)
- `memory_per_batch::Int`: Estimated memory bytes per batch, including a SplinePSFModel's
  table

# Example
```julia
smld, info = fit(batch, fitter)
println("Fit \$(info.n_fits) ROIs in \$(info.elapsed_s * 1000) ms on \$(info.backend)")
println("Processed in \$(info.n_batches) batches of \$(info.batch_size)")
```

# See also
[`fit`](@ref), [`GaussMLEConfig`](@ref)
"""
struct GaussMLEFitInfo <: AbstractSMLMInfo
    elapsed_s::Float64
    backend::Symbol
    device_id::Int
    n_fits::Int
    n_converged::Int
    batch_size::Int
    n_batches::Int
    memory_per_batch::Int
end

function Base.show(io::IO, info::GaussMLEFitInfo)
    elapsed_ms = info.elapsed_s * 1000
    device_str = info.backend == :gpu ? "GPU:$(info.device_id)" : "CPU"
    mem_str = info.memory_per_batch > 1024^2 ?
        "$(round(info.memory_per_batch / 1024^2, digits = 1)) MB" :
        "$(round(info.memory_per_batch / 1024, digits = 1)) KB"
    return print(
        io,
        "GaussMLEFitInfo($(info.n_fits) fits, $(round(elapsed_ms, digits = 2)) ms, \
        $device_str, $(info.n_batches) batches × $(info.batch_size), $mem_str/batch)",
    )
end

export GaussMLEFitInfo
