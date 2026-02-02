"""
High-level API for Gaussian MLE fitting
"""

using KernelAbstractions
using CUDA
using SMLMData

# Input validation helpers
function validate_fit_input(data::AbstractArray{T,3}, camera=nothing) where T
    # Check for empty data
    if isempty(data)
        throw(ArgumentError("Input data array is empty"))
    end

    # Check for square ROIs
    if size(data, 1) != size(data, 2)
        throw(ArgumentError("ROIs must be square, got size $(size(data, 1))×$(size(data, 2))"))
    end

    # Check for minimum ROI size
    if size(data, 1) < 3
        throw(ArgumentError("ROI size must be at least 3×3, got $(size(data, 1))×$(size(data, 1))"))
    end

    # Check for NaN or Inf values
    if any(!isfinite, data)
        throw(ArgumentError("Input data contains NaN or Inf values"))
    end

    # Check for negative values (only warn for IdealCamera - sCMOS can have negative after preprocessing)
    if any(<(0), data)
        if camera isa IdealCamera || camera isa SMLMData.IdealCamera
            @warn "Input data contains negative values, which may indicate preprocessing issues"
        end
        # For sCMOS: negative values are expected after offset subtraction
    end

    return true
end

# Helper to extract variance map from SMLMData.SCMOSCamera
function extract_variance_map(camera::SMLMData.SCMOSCamera, ::Type{T}) where T
    # SMLMData uses 'readnoise' field (std dev), we need variance (readnoise²)
    # Handle both scalar and matrix readnoise
    variance = camera.readnoise .^ 2
    if variance isa Real
        # Scalar readnoise: expand to full sensor matrix
        ny = length(camera.pixel_edges_y) - 1
        nx = length(camera.pixel_edges_x) - 1
        return fill(T(variance), ny, nx)
    else
        return T.(variance)
    end
end

"""
    GaussMLEFitter{P,PC}

Main type for configuring and performing Maximum Likelihood Estimation of Gaussian blob parameters.

# Fields
- `backend::Symbol`: Compute backend (`:cpu`, `:gpu`, or `:auto`)
- `psf_model::P<:PSFModel`: Point spread function model
- `iterations::Int`: Number of Newton-Raphson iterations
- `constraints::PC<:ParameterConstraints`: Parameter bounds and step limits
- `batch_size::Int`: Batch size for GPU processing
- `auto_timeout::Float64`: Seconds to wait for GPU in auto mode
- `gpu_timeout::Float64`: Seconds to wait for GPU in explicit gpu mode
- `on_wait`: Callback for GPU wait progress feedback

# See also
[`fit`](@ref), [`GaussMLEResults`](@ref), [`PSFModel`](@ref), [`CameraModel`](@ref)
"""
struct GaussMLEFitter{P<:PSFModel, PC<:ParameterConstraints, W}
    backend::Symbol
    psf_model::P
    iterations::Int
    constraints::PC
    batch_size::Int
    auto_timeout::Float64
    gpu_timeout::Float64
    on_wait::W
end

"""
    GaussMLEFitter(; kwargs...)

Create a fitter for Gaussian MLE with sensible defaults.

Camera noise model is determined by the ROIBatch type:
- ROIBatch with SMLMData.IdealCamera → Poisson noise only
- ROIBatch with SMLMData.SCMOSCamera → Poisson + per-pixel readnoise variance

# Keyword Arguments
- `psf_model = GaussianXYNB(0.13f0)`: PSF model to use
- `backend = :auto`: Compute backend (`:cpu`, `:gpu`, or `:auto`)
- `iterations = 20`: Number of Newton-Raphson iterations
- `constraints = nothing`: Parameter constraints (uses defaults if nothing)
- `batch_size = 10_000`: Number of ROIs to process per GPU batch
- `auto_timeout = 30.0`: Seconds to wait for GPU in auto mode before falling back to CPU
- `gpu_timeout = Inf`: Seconds to wait for GPU in explicit gpu mode
- `on_wait = nothing`: Callback `(elapsed, available, required) -> nothing` for GPU wait progress

# Backend Semantics
- `:cpu` - Always use CPU, no waiting
- `:gpu` - Explicit GPU request, wait up to gpu_timeout, error if unavailable
- `:auto` - Try GPU with auto_timeout, fall back to CPU with warning

# Examples
```julia
# Simple usage with auto GPU detection
fitter = GaussMLEFitter()

# Force CPU execution
cpu_fitter = GaussMLEFitter(backend = :cpu)

# Variable sigma model with more iterations
fitter = GaussMLEFitter(
    psf_model = GaussianXYNBS(),
    iterations = 30
)

# Explicit GPU with custom timeout
fitter = GaussMLEFitter(
    backend = :gpu,
    gpu_timeout = 60.0,
    on_wait = (e, a, r) -> @info "Waiting..." elapsed=e
)
```

# See also
[`fit`](@ref), [`GaussianXYNB`](@ref), [`GaussianXYNBS`](@ref)
"""
function GaussMLEFitter(;
    psf_model = GaussianXYNB(0.13f0),
    backend::Symbol = :auto,
    device = nothing,  # deprecated, use backend
    iterations = 20,
    constraints = nothing,
    batch_size = 10_000,
    auto_timeout = 30.0,
    gpu_timeout = Inf,
    on_wait = nothing
)
    # Handle deprecated device kwarg
    if device !== nothing
        @warn "device kwarg is deprecated, use backend instead" maxlog=1
        backend = if device isa Symbol
            device
        elseif device isa CPU
            :cpu
        elseif device isa GPU
            :gpu
        else
            :auto
        end
    end

    backend in (:cpu, :gpu, :auto) || error("backend must be :cpu, :gpu, or :auto")

    # Default constraints based on typical box size
    if isnothing(constraints)
        constraints = default_constraints(psf_model, 11)  # typical 11x11 box
    end

    return GaussMLEFitter(backend, psf_model, iterations, constraints, batch_size,
                          Float64(auto_timeout), Float64(gpu_timeout), on_wait)
end

"""
    estimate_batch_memory(batch_size, box_size, n_params)

Estimate GPU memory needed for one batch of MLE fitting.

Memory components:
- data: batch_size × box_size² × 4 bytes
- results: batch_size × n_params × 4 bytes
- uncertainties: batch_size × n_params × 4 bytes
- variance_map: box_size² × 4 bytes (shared)
- corners: batch_size × 2 × 4 bytes
"""
function estimate_batch_memory(batch_size::Integer, box_size::Integer, n_params::Integer)
    data_bytes = batch_size * box_size^2 * 4
    results_bytes = batch_size * n_params * 4
    uncertainties_bytes = batch_size * n_params * 4
    variance_bytes = box_size^2 * 4
    corners_bytes = batch_size * 2 * 4
    return data_bytes + results_bytes + uncertainties_bytes + variance_bytes + corners_bytes
end

"""
    compute_cpu_batch_size(n_fits, box_size, n_params; memory_multiplier=4)

Compute optimal batch size for CPU processing based on available memory.
Returns n_fits if all data fits in memory, otherwise a smaller batch size.
"""
function compute_cpu_batch_size(n_fits::Integer, box_size::Integer, n_params::Integer;
                                 memory_multiplier::Integer=4)
    free_mem = Sys.free_memory()
    memory_per_roi = (box_size^2 + n_params * 2 + 3 + 1) * 4  # data + results + uncertainties + covariances + llr
    total_memory_needed = n_fits * memory_per_roi * memory_multiplier

    if total_memory_needed <= free_mem
        return n_fits  # Process all at once
    else
        # Compute batch size that fits in 50% of free memory (conservative)
        usable_mem = free_mem ÷ 2
        batch_size = max(1, Int(usable_mem ÷ (memory_per_roi * memory_multiplier)))
        return min(batch_size, n_fits)
    end
end

"""
    _run_mle_kernel!(results, uncertainties, covariances, log_likelihoods,
                     data, psf_pixels, use_scmos, variance_map, x_corners, y_corners,
                     fitter) -> (actual_backend, device_id, actual_batch_size, actual_n_batches)

Internal helper that runs the MLE kernel on CPU or GPU with appropriate batching.
Handles all backend selection and batch management logic.

# Arguments
- Pre-allocated output arrays: results, uncertainties, covariances, log_likelihoods
- Input data and parameters: data, psf_pixels, use_scmos, variance_map, x_corners, y_corners
- fitter: GaussMLEFitter with backend/batching configuration

# Returns
Tuple of (actual_backend::Symbol, device_id::Int, actual_batch_size::Int, actual_n_batches::Int)
for use in FitInfo construction.
"""
function _run_mle_kernel!(
    results::Matrix{Float32},
    uncertainties::Matrix{Float32},
    covariances::Matrix{Float32},
    log_likelihoods::Vector{Float32},
    data::Array{Float32,3},
    psf_pixels,
    use_scmos::Val,
    variance_map::Matrix{Float32},
    x_corners::AbstractVector{Int32},
    y_corners::AbstractVector{Int32},
    fitter::GaussMLEFitter
)
    n_fits = size(data, 3)
    n_params = size(results, 1)
    box_size = size(data, 1)

    # Select backend with GPU memory wait
    actual_batch = min(fitter.batch_size, n_fits)
    required_memory = estimate_batch_memory(actual_batch, box_size, n_params)
    device = select_backend(fitter.backend, required_memory;
                           auto_timeout=fitter.auto_timeout,
                           gpu_timeout=fitter.gpu_timeout,
                           on_wait=fitter.on_wait)

    # Track actual backend and device_id for FitInfo
    actual_backend = device isa CPU ? :cpu : :gpu
    device_id = device isa CPU ? -1 : Int(CUDA.device().handle)

    # Track batch info for FitInfo
    actual_batch_size = 0
    actual_n_batches = 0

    if device isa CPU
        # CPU with memory-aware batching
        ka_backend = KernelAbstractions.CPU()
        cpu_batch_size = compute_cpu_batch_size(n_fits, box_size, n_params)
        actual_batch_size = cpu_batch_size >= n_fits ? n_fits : cpu_batch_size

        if cpu_batch_size >= n_fits
            # Process all at once
            actual_n_batches = 1
            kernel = unified_gaussian_mle_kernel!(ka_backend)
            kernel(results, uncertainties, covariances, log_likelihoods,
                   data, psf_pixels, use_scmos, variance_map, x_corners, y_corners,
                   fitter.constraints, fitter.iterations,
                   ndrange=n_fits)
            KernelAbstractions.synchronize(ka_backend)
        else
            # Batch processing for memory efficiency
            actual_n_batches = cld(n_fits, cpu_batch_size)
            for batch_start in 1:cpu_batch_size:n_fits
                batch_end = min(batch_start + cpu_batch_size - 1, n_fits)
                batch_size_actual = batch_end - batch_start + 1

                batch_data = data[:, :, batch_start:batch_end]
                batch_x_corners = x_corners[batch_start:batch_end]
                batch_y_corners = y_corners[batch_start:batch_end]

                # Views into pre-allocated result arrays
                batch_results = @view results[:, batch_start:batch_end]
                batch_uncertainties = @view uncertainties[:, batch_start:batch_end]
                batch_covariances = @view covariances[:, batch_start:batch_end]
                batch_llr = @view log_likelihoods[batch_start:batch_end]

                kernel = unified_gaussian_mle_kernel!(ka_backend)
                kernel(batch_results, batch_uncertainties, batch_covariances, batch_llr,
                       batch_data, psf_pixels, use_scmos, variance_map, batch_x_corners, batch_y_corners,
                       fitter.constraints, fitter.iterations,
                       ndrange=batch_size_actual)
                KernelAbstractions.synchronize(ka_backend)

                # Help GC between batches
                batch_data = nothing
                GC.gc(false)
            end
        end
    else
        # GPU batch processing
        actual_batch_size = min(fitter.batch_size, n_fits)
        actual_n_batches = cld(n_fits, fitter.batch_size)
        for batch_start in 1:fitter.batch_size:n_fits
            batch_end = min(batch_start + fitter.batch_size - 1, n_fits)
            batch_size_actual = batch_end - batch_start + 1

            batch_data = data[:, :, batch_start:batch_end]
            d_batch_data = KernelAbstractions.allocate(backend(device), Float32, size(batch_data))
            copyto!(d_batch_data, batch_data)

            d_variance_map = KernelAbstractions.allocate(backend(device), Float32, size(variance_map))
            copyto!(d_variance_map, variance_map)

            batch_x_corners = x_corners[batch_start:batch_end]
            batch_y_corners = y_corners[batch_start:batch_end]
            d_x_corners = KernelAbstractions.allocate(backend(device), Int32, length(batch_x_corners))
            d_y_corners = KernelAbstractions.allocate(backend(device), Int32, length(batch_y_corners))
            copyto!(d_x_corners, batch_x_corners)
            copyto!(d_y_corners, batch_y_corners)

            d_results = KernelAbstractions.allocate(backend(device), Float32, (n_params, batch_size_actual))
            d_uncertainties = KernelAbstractions.allocate(backend(device), Float32, (n_params, batch_size_actual))
            d_covariances = KernelAbstractions.allocate(backend(device), Float32, (3, batch_size_actual))
            d_log_likelihoods = KernelAbstractions.allocate(backend(device), Float32, batch_size_actual)

            kernel = unified_gaussian_mle_kernel!(backend(device))
            kernel(d_results, d_uncertainties, d_covariances, d_log_likelihoods,
                   d_batch_data, psf_pixels, use_scmos, d_variance_map, d_x_corners, d_y_corners,
                   fitter.constraints, fitter.iterations,
                   ndrange=batch_size_actual)

            KernelAbstractions.synchronize(backend(device))

            results[:, batch_start:batch_end] = Array(d_results)
            uncertainties[:, batch_start:batch_end] = Array(d_uncertainties)
            covariances[:, batch_start:batch_end] = Array(d_covariances)
            log_likelihoods[batch_start:batch_end] = Array(d_log_likelihoods)
        end
    end

    return (actual_backend, device_id, actual_batch_size, actual_n_batches)
end

"""
    fit(data::AbstractArray{T,3}, fitter::GaussMLEFitter; variance_map=nothing) -> (BasicSMLD, FitInfo)

Fit Gaussian blobs to a stack of ROIs using Maximum Likelihood Estimation.

# Arguments
- `data::AbstractArray{T,3}`: ROI data as (roi_size, roi_size, n_rois) array
- `fitter::GaussMLEFitter`: Configured fitter object

# Keyword Arguments
- `variance_map=nothing`: Optional sCMOS variance map (will override fitter's camera model)

# Returns
- `Tuple{BasicSMLD, FitInfo}`: Fitted localizations and fit metadata

# Examples
```julia
# Fit 1000 ROIs
data = zeros(Float32, 7, 7, 1000)  # Your ROI data here
fitter = GaussMLEFitter(psf_model = GaussianXYNB(0.13f0))
smld, info = fit(data, fitter)

# Access results
println("Fitted \$(info.n_fits) ROIs in \$(info.elapsed_s * 1000) ms")
println("Mean x position: ", mean([e.x for e in smld.emitters]))
```

# See also
[`GaussMLEFitter`](@ref), [`FitInfo`](@ref)
"""
function fit(data::AbstractArray{T,3}, fitter::GaussMLEFitter;
             variance_map=nothing) where T
    # Start timing
    t0 = time_ns()

    # Validate input
    validate_fit_input(data, nothing)

    n_fits = size(data, 3)
    n_params = length(fitter.psf_model)
    box_size = size(data, 1)

    # Convert data to Float32 if needed
    data_f32 = convert(Array{Float32,3}, data)

    # Default pixel size for Array input (no camera attached)
    pixel_size = 0.1f0
    psf_pixels = to_pixel_units(fitter.psf_model, pixel_size)

    # Allocate result arrays
    results = Matrix{Float32}(undef, n_params, n_fits)
    uncertainties = Matrix{Float32}(undef, n_params, n_fits)
    covariances = Matrix{Float32}(undef, 3, n_fits)  # [σ_xy, σ_xz, σ_yz]
    log_likelihoods = Vector{Float32}(undef, n_fits)

    # Determine camera model from variance_map keyword
    use_scmos = isnothing(variance_map) ? Val(false) : Val(true)
    var_map = isnothing(variance_map) ? zeros(Float32, box_size, box_size) : Float32.(variance_map)

    # Create dummy corners for fit(Array): [1, 1+roi_size, 1+2*roi_size, ...]
    x_corners = Int32[1 + (i-1) * box_size for i in 1:n_fits]
    y_corners = fill(Int32(1), n_fits)

    # Run MLE kernel with batching
    actual_backend, device_id, actual_batch_size, actual_n_batches = _run_mle_kernel!(
        results, uncertainties, covariances, log_likelihoods,
        data_f32, psf_pixels, use_scmos, var_map, x_corners, y_corners,
        fitter
    )

    # Compute p-values from log-likelihood ratios
    # χ² = -2×LLR, df = n_pixels - n_params
    pvalues = Vector{Float32}(undef, n_fits)
    df = box_size^2 - n_params
    chi2_dist = Chisq(df)

    for i in 1:n_fits
        llr = log_likelihoods[i]
        χ² = -2.0f0 * llr
        pvalues[i] = 1.0f0 - Float32(cdf(chi2_dist, χ²))
    end

    # Create minimal ROIBatch for SMLD conversion
    roi_size = size(data_f32, 1)
    x_corners_smld = Int32[1 + (i-1) * roi_size for i in 1:n_fits]  # [1, 1+roi_size, ...] (1-indexed for Julia)
    y_corners_smld = fill(Int32(1), n_fits)  # All at y=1
    frame_indices = ones(Int32, n_fits)

    # Create minimal camera for SMLD conversion (fit(Array) has no real camera)
    camera_smld = SMLMData.IdealCamera(0:1023, 0:1023, pixel_size)

    batch = SMLMData.ROIBatch(data_f32, x_corners_smld, y_corners_smld, frame_indices, camera_smld)
    loc_result = create_localization_result(results, uncertainties, covariances, log_likelihoods, pvalues, batch, fitter.psf_model)

    # Calculate elapsed time and create FitInfo
    elapsed_s = (time_ns() - t0) / 1e9
    memory_per_batch = estimate_batch_memory(actual_batch_size, box_size, n_params)
    info = FitInfo(elapsed_s, actual_backend, device_id, n_fits, n_fits, actual_batch_size, actual_n_batches, memory_per_batch)

    # Return tuple (BasicSMLD, FitInfo)
    return (to_smld(loc_result, batch), info)
end

# Convenience function for single ROI fitting
function fit(roi::AbstractMatrix{T}, fitter::GaussMLEFitter) where T
    # Reshape to 3D array with single ROI
    data = reshape(roi, size(roi, 1), size(roi, 2), 1)
    smld, info = fit(data, fitter)

    # Return first emitter and info
    return (smld.emitters[1], info)
end

# Fit method for ROIBatch - returns (BasicSMLD, FitInfo) with real camera coordinates
function fit(roi_batch::ROIBatch{T,N,A,<:SMLMData.IdealCamera}, fitter::GaussMLEFitter) where {T,N,A}
    # Start timing
    t0 = time_ns()

    # Fit the raw data with Poisson-only likelihood
    n_fits = size(roi_batch.data, 3)
    n_params = length(fitter.psf_model)
    box_size = size(roi_batch.data, 1)

    # Get pixel size and convert PSF from microns to pixels
    pixel_size = roi_batch.camera.pixel_edges_x[2] - roi_batch.camera.pixel_edges_x[1]
    psf_pixels = to_pixel_units(fitter.psf_model, pixel_size)

    # Allocate result arrays
    results = Matrix{Float32}(undef, n_params, n_fits)
    uncertainties = Matrix{Float32}(undef, n_params, n_fits)
    covariances = Matrix{Float32}(undef, 3, n_fits)  # [σ_xy, σ_xz, σ_yz]
    log_likelihoods = Vector{Float32}(undef, n_fits)

    # IdealCamera: Poisson noise only (no variance map)
    use_scmos = Val(false)
    variance_map = zeros(Float32, box_size, box_size)  # Unused but needed for kernel signature

    # Use unified kernel on CPU/GPU
    data_f32 = convert(Array{Float32,3}, roi_batch.data)

    # Run MLE kernel with batching
    actual_backend, device_id, actual_batch_size, actual_n_batches = _run_mle_kernel!(
        results, uncertainties, covariances, log_likelihoods,
        data_f32, psf_pixels, use_scmos, variance_map, roi_batch.x_corners, roi_batch.y_corners,
        fitter
    )

    # Compute p-values from log-likelihood ratios
    pvalues = Vector{Float32}(undef, n_fits)
    df = box_size^2 - n_params
    chi2_dist = Chisq(df)

    for i in 1:n_fits
        llr = log_likelihoods[i]
        χ² = -2.0f0 * llr
        pvalues[i] = 1.0f0 - Float32(cdf(chi2_dist, χ²))
    end

    # Use real ROIBatch for coordinate conversion (preserves corners!)
    loc_result = create_localization_result(results, uncertainties, covariances, log_likelihoods, pvalues, roi_batch, fitter.psf_model)

    # Calculate elapsed time and create FitInfo
    elapsed_s = (time_ns() - t0) / 1e9
    memory_per_batch = estimate_batch_memory(actual_batch_size, box_size, n_params)
    info = FitInfo(elapsed_s, actual_backend, device_id, n_fits, n_fits, actual_batch_size, actual_n_batches, memory_per_batch)

    return (to_smld(loc_result, roi_batch), info)
end

# Fit method for ROIBatch with SMLMData.SCMOSCamera
function fit(roi_batch::ROIBatch{T,N,A,<:SMLMData.SCMOSCamera}, fitter::GaussMLEFitter) where {T,N,A}
    # Start timing
    t0 = time_ns()

    # Preprocess: ADU → electrons using per-pixel calibration at ROI positions
    data_electrons = to_electrons(roi_batch.data, roi_batch.camera, roi_batch.x_corners, roi_batch.y_corners)
    variance_map = extract_variance_map(roi_batch.camera, Float32)

    n_fits = size(data_electrons, 3)
    n_params = length(fitter.psf_model)
    box_size = size(data_electrons, 1)

    # Get pixel size and convert PSF from microns to pixels
    pixel_size = roi_batch.camera.pixel_edges_x[2] - roi_batch.camera.pixel_edges_x[1]
    psf_pixels = to_pixel_units(fitter.psf_model, pixel_size)

    # Allocate result arrays
    results = Matrix{Float32}(undef, n_params, n_fits)
    uncertainties = Matrix{Float32}(undef, n_params, n_fits)
    covariances = Matrix{Float32}(undef, 3, n_fits)  # [σ_xy, σ_xz, σ_yz]
    log_likelihoods = Vector{Float32}(undef, n_fits)

    # SCMOSCamera: Poisson + readnoise variance
    use_scmos = Val(true)

    # Use unified kernel on CPU/GPU (data already in electrons)
    data_f32 = convert(Array{Float32,3}, data_electrons)

    # Run MLE kernel with batching
    actual_backend, device_id, actual_batch_size, actual_n_batches = _run_mle_kernel!(
        results, uncertainties, covariances, log_likelihoods,
        data_f32, psf_pixels, use_scmos, variance_map, roi_batch.x_corners, roi_batch.y_corners,
        fitter
    )

    # Compute p-values from log-likelihood ratios
    pvalues = Vector{Float32}(undef, n_fits)
    df = box_size^2 - n_params
    chi2_dist = Chisq(df)

    for i in 1:n_fits
        llr = log_likelihoods[i]
        χ² = -2.0f0 * llr
        pvalues[i] = 1.0f0 - Float32(cdf(chi2_dist, χ²))
    end

    # Use original ROIBatch for coordinate conversion (preserves corners and original camera!)
    loc_result = create_localization_result(results, uncertainties, covariances, log_likelihoods, pvalues, roi_batch, fitter.psf_model)

    # Calculate elapsed time and create FitInfo
    elapsed_s = (time_ns() - t0) / 1e9
    memory_per_batch = estimate_batch_memory(actual_batch_size, box_size, n_params)
    info = FitInfo(elapsed_s, actual_backend, device_id, n_fits, n_fits, actual_batch_size, actual_n_batches, memory_per_batch)

    return (to_smld(loc_result, roi_batch), info)
end

"""
    fit(batch::ROIBatch; model=GaussianXYNB(), max_iterations=20, backend=:auto, ...) -> (BasicSMLD, FitInfo)

Convenience form of fit() that creates a GaussMLEFitter from keyword arguments.

# Arguments
- `batch::ROIBatch`: Input ROI data with camera calibration

# Keyword Arguments
- `model=GaussianXYNB(0.13f0)`: PSF model to use
- `max_iterations=20`: Number of Newton-Raphson iterations
- `backend=:auto`: Compute backend (`:cpu`, `:gpu`, or `:auto`)
- `constraints=nothing`: Parameter constraints (uses defaults if nothing)
- `batch_size=10_000`: Batch size for GPU processing
- `auto_timeout=30.0`: Seconds to wait for GPU in auto mode
- `gpu_timeout=Inf`: Seconds to wait for GPU in explicit gpu mode
- `on_wait=nothing`: Callback for GPU wait progress

# Returns
- `Tuple{BasicSMLD, FitInfo}`: Fitted localizations and fit metadata

# Examples
```julia
# Simple fit with defaults
smld, info = fit(batch)

# Custom model and iterations
smld, info = fit(batch; model=GaussianXYNBS(), max_iterations=30)
```

# See also
[`GaussMLEFitter`](@ref), [`FitInfo`](@ref)
"""
function fit(batch::ROIBatch;
             model = GaussianXYNB(0.13f0),
             max_iterations = 20,
             backend = :auto,
             constraints = nothing,
             batch_size = 10_000,
             auto_timeout = 30.0,
             gpu_timeout = Inf,
             on_wait = nothing)
    fitter = GaussMLEFitter(;
        psf_model = model,
        backend = backend,
        iterations = max_iterations,
        constraints = constraints,
        batch_size = batch_size,
        auto_timeout = auto_timeout,
        gpu_timeout = gpu_timeout,
        on_wait = on_wait
    )
    return fit(batch, fitter)
end


# Export API
export GaussMLEFitter, fit, FitInfo