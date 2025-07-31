"""
Dispatch system for routing fitting calls to appropriate backends
"""

# Main dispatch function for batched fitting
function fit_batched(backend::FittingBackend, data::AbstractArray{T,3}, 
                    modelsymbol::Symbol, config::BatchConfig,
                    variance::Union{Nothing,AbstractArray{T,3}},
                    verbose::Bool) where T
    
    # Get model type
    modeltype = get(MODEL_MAP, modelsymbol, nothing)
    modeltype === nothing && error("Unknown model symbol: $modelsymbol")
    
    # Create batch iterator
    batch_iter = BatchIterator(data, config.max_batch_size)
    n_rois = size(data, 3)
    
    # Initialize results
    results = BatchResults{T}(n_rois)
    
    # Progress tracking
    if verbose
        println("Fitting $n_rois ROIs using $(backend_name(backend)) backend")
        println("Batch size: $(config.max_batch_size), Number of batches: $(length(batch_iter))")
    end
    
    # Process batches
    start_time = time()
    for (batch_idx, (batch_data, start_idx, end_idx)) in enumerate(batch_iter)
        if verbose && batch_idx % 10 == 0
            elapsed = time() - start_time
            rate = (start_idx - 1) / elapsed
            eta = (n_rois - start_idx + 1) / rate
            println("Batch $batch_idx/$(length(batch_iter)): " *
                   "$(round(rate, digits=0)) ROIs/sec, " *
                   "ETA: $(round(eta, digits=1))s")
        end
        
        # Extract variance batch if provided
        batch_variance = variance === nothing ? nothing : 
                        view(variance, :, :, start_idx:end_idx)
        
        # Fit batch
        batch_params, batch_uncert = fit_batch(backend, batch_data, modeltype, 
                                              batch_variance)
        
        # Store results
        add_batch_results!(results, batch_params, batch_uncert, start_idx)
    end
    
    if verbose
        total_time = time() - start_time
        println("Completed in $(round(total_time, digits=2))s " *
               "($(round(n_rois/total_time, digits=0)) ROIs/sec)")
    end
    
    return results.parameters, results.uncertainties
end

# CPU backend implementation
function fit_batch(backend::CPUBackend, data::AbstractArray{T,3}, 
                  modeltype::Type, 
                  variance::Union{Nothing,AbstractArray{T,3}}) where T
    
    # Create model arguments
    args = genargs(modeltype; T=T)
    
    # Prepare for parallel execution
    n_rois = size(data, 3)
    params = Vector{modeltype}(undef, n_rois)
    uncerts = Vector{crlb_struct(modeltype)}(undef, n_rois)
    
    # Use threading for CPU parallelism
    Threads.@threads for i in 1:n_rois
        roi_data = data[:, :, i]  # Copy to ensure it's a Matrix
        roi_variance = variance === nothing ? nothing : variance[:, :, i]
        
        # Fit single ROI (reuse existing CPU code)
        params[i], uncerts[i] = fit_single_roi(roi_data, modeltype, args, roi_variance)
    end
    
    return params, uncerts
end

# CUDA backend implementation (placeholder)
function fit_batch(backend::CUDABackend, data::AbstractArray{T,3}, 
                  modeltype::Type, 
                  variance::Union{Nothing,AbstractArray{T,3}}) where T
    
    # This will be implemented in cuda_kernels.jl
    return cuda_fit_batch(backend, data, modeltype, variance)
end

# Metal backend implementation (placeholder)
function fit_batch(backend::MetalBackend, data::AbstractArray{T,3}, 
                  modeltype::Type, 
                  variance::Union{Nothing,AbstractArray{T,3}}) where T
    
    # This will be implemented in metal_kernels.jl
    error("Metal backend not yet implemented")
end

# Helper function to fit a single ROI (CPU)
function fit_single_roi(data::AbstractMatrix{T}, modeltype::Type,
                       args::GaussMLEArgs, variance::Union{Nothing,AbstractMatrix{T}}) where T
    
    # Initialize parameters
    θ = genθ(modeltype, size(data, 1); T=T)
    initialize_parameters!(θ, data, size(data, 1), args)
    
    # Create workspace
    nparams = θ.nparams
    numerator = zeros(T, nparams)
    denominator = zeros(T, nparams)
    
    # Newton-Raphson optimization
    max_iterations = 50
    converged = false
    
    for iter in 1:max_iterations
        # Reset accumulators
        fill!(numerator, zero(T))
        fill!(denominator, zero(T))
        
        # Accumulate over pixels
        for j in 1:size(data, 2)
            for i in 1:size(data, 1)
                # Get model value and derivatives
                model_val = Ref{T}()
                grad = zeros(T, nparams)
                hessdiag = zeros(T, nparams)
                
                compute_all!(model_val, grad, hessdiag, θ, args, i, j)
                
                # Weight by variance if provided
                weight = variance === nothing ? one(T) : one(T) / variance[i, j]
                
                # Accumulate for Newton-Raphson update
                diff = data[i, j] - model_val[]
                for k in 1:nparams
                    numerator[k] += weight * grad[k] * diff / max(model_val[], one(T))
                    denominator[k] += weight * (grad[k]^2 / max(model_val[], one(T)) + 
                                              hessdiag[k] * diff / max(model_val[], one(T)))
                end
            end
        end
        
        # Update parameters
        converged = update!(θ, numerator, denominator)
        
        if converged
            break
        end
    end
    
    # Compute CRLB
    Σ = genΣ(modeltype; T=T)
    compute_crlb!(Σ, θ, args, size(data, 1), variance)
    
    return θ, Σ
end

# Helper to get the CRLB struct type from model type
function crlb_struct(modeltype::Type)
    # This is a bit of a hack, but works for current models
    if modeltype == θ_xynb
        return Σ_xynb
    elseif modeltype == θ_xynbs
        return Σ_xynbs
    else
        error("Unknown CRLB struct for model type $modeltype")
    end
end

# Helper to compute CRLB for a single ROI
function compute_crlb!(Σ::GaussMLEΣ, θ::GaussMLEParams, args::GaussMLEArgs, 
                      boxsize::Int, variance::Union{Nothing,AbstractMatrix{T}}) where T
    # This is a simplified CRLB computation
    # Full implementation would compute Fisher Information Matrix
    
    # For now, use approximate uncertainties based on photon count
    n_photons = hasproperty(θ, :n) ? θ.n : T(500)
    bg_photons = hasproperty(θ, :bg) ? θ.bg : T(2)
    σ_psf = hasproperty(args, :σ_PSF) ? args.σ_PSF : T(1.3)
    
    # Thompson et al. (2002) approximations
    σ_xy = σ_psf / sqrt(n_photons) * sqrt(1 + 4 * bg_photons / n_photons)
    σ_n = sqrt(n_photons)
    σ_bg = sqrt(bg_photons * boxsize^2)
    
    # Set uncertainties
    if hasproperty(Σ, :σ_x)
        Σ.σ_x = σ_xy
        Σ.σ_y = σ_xy
    end
    if hasproperty(Σ, :σ_n)
        Σ.σ_n = σ_n
    end
    if hasproperty(Σ, :σ_bg)
        Σ.σ_bg = σ_bg
    end
end