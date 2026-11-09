"""
High-level API for Gaussian MLE fitting
"""

# Input validation helpers

# Throw ArgumentError for the first ROI of box size `box` at `x_corners`, `y_corners` that
# does not lie on a sensor of `sensor_size = (rows, columns)`: 1 <= corner and
# corner + box - 1 <= sensor size on both axes. A device-side lookup outside the sensor is
# a kernel exception, not a clear error.
function _validate_roi_corners(
        box::Integer, x_corners::AbstractVector, y_corners::AbstractVector,
        sensor_size::Tuple{Int, Int}
    )
    rows, columns = sensor_size
    for k in eachindex(x_corners, y_corners)
        x, y = Int(x_corners[k]), Int(y_corners[k])
        if x < 1 || y < 1 || x + box - 1 > columns || y + box - 1 > rows
            throw(
                ArgumentError(
                    "ROI $k with corner (x = $x, y = $y) and box $box does not lie on the \
                    $columns×$rows sensor (1 <= corner and corner + box - 1 <= sensor \
                    size on both axes)"
                )
            )
        end
    end
    return nothing
end

function validate_fit_input(data::AbstractArray{T, 3}, camera = nothing) where {T}
    # Check for empty data
    if isempty(data)
        throw(ArgumentError("Input data array is empty"))
    end

    # Check for square ROIs
    if size(data, 1) != size(data, 2)
        throw(ArgumentError("ROIs must be square, got size $(size(data, 1))×\
            $(size(data, 2))"))
    end

    # Check for minimum ROI size
    if size(data, 1) < 3
        throw(ArgumentError("ROI size must be at least 3×3, got $(size(data, 1))×\
            $(size(data, 1))"))
    end

    # Check for NaN or Inf values
    if any(!isfinite, data)
        throw(ArgumentError("Input data contains NaN or Inf values"))
    end

    # Check for negative values (only warn for IdealCamera - sCMOS can have negative after
    # preprocessing)
    if any(<(0), data)
        if camera isa IdealCamera || camera isa SMLMData.IdealCamera
            @warn "Input data contains negative values, which may indicate preprocessing \
                issues"
        end
        # For sCMOS: negative values are expected after offset subtraction
    end

    return true
end

# Helper to extract variance map from SMLMData.SCMOSCamera
function extract_variance_map(camera::SMLMData.SCMOSCamera, ::Type{T}) where {T}
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
    GaussMLEConfig{P,PC,W}

Main type for configuring and performing Maximum Likelihood Estimation of Gaussian blob
parameters.

`:gpu` and `:auto` use a CUDA GPU, or an AMD GPU when AMDGPU is loaded and CUDA is not
functional.

# Fields
- `backend::Symbol`: Compute backend (`:cpu`, `:gpu`, or `:auto`)
- `psf_model::P<:PSFModel`: Point spread function model
- `iterations::Int`: Cap on the Levenberg-Marquardt pixel passes of a fit (of each
  start for `SplinePSFModel`); the last pass only evaluates, so a fit takes at most
  `iterations - 1` steps and `iterations = 1` returns the start. The Gaussian models
  stop earlier once converged; one that reaches the cap first reports infinite
  uncertainties
- `constraints::PC<:Union{Nothing, ParameterConstraints}`: Parameter bounds and step limits;
  `nothing` builds defaults from the ROI size at fit time
- `batch_size::Int`: Batch size for GPU processing
- `auto_timeout::Float64`: Seconds to wait for GPU in auto mode
- `gpu_timeout::Float64`: Seconds to wait for GPU in explicit gpu mode
- `on_wait`: Callback for GPU wait progress feedback

# See also
[`fit`](@ref), [`GaussMLEFitInfo`](@ref), [`PSFModel`](@ref GaussMLE.PSFModel)
"""
struct GaussMLEConfig{
        P <: PSFModel, PC <: Union{Nothing, ParameterConstraints}, W,
    } <: AbstractSMLMConfig
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
    GaussMLEConfig(; kwargs...)

Create a fitter for Gaussian MLE with sensible defaults.

Camera noise model is determined by the ROIBatch type:
- ROIBatch with SMLMData.IdealCamera → Poisson noise only
- ROIBatch with SMLMData.SCMOSCamera → Poisson + per-pixel readnoise variance

# Keyword Arguments
- `psf_model = GaussianXYNB(0.13f0)`: PSF model to use
- `backend = :auto`: Compute backend (`:cpu`, `:gpu`, or `:auto`)
- `iterations`: Cap on the Levenberg-Marquardt pixel passes of a fit (of each
  start for `SplinePSFModel`); the last pass only evaluates, so a fit takes at most
  `iterations - 1` steps and `iterations = 1` returns the start. Default 100 for the
  Gaussian models, which stop earlier once converged (one that reaches the cap first
  reports infinite uncertainties), and 20 for `SplinePSFModel`, which runs every pass
- `constraints = nothing`: Parameter constraints; `nothing` builds defaults from the
  ROI size at fit time
- `batch_size = 10_000`: Number of ROIs to process per GPU batch; a GPU out-of-memory
  error halves it for the rest of the fit
- `auto_timeout = 300.0`: Seconds to wait for GPU in auto mode before falling back to CPU
- `gpu_timeout = Inf`: Seconds to wait for GPU in explicit gpu mode
- `on_wait = nothing`: Callback `(elapsed, available, required) -> nothing` for GPU wait
  progress

# Backend Semantics
- `:cpu` - Always use CPU, no waiting
- `:gpu` - Error immediately if no functional GPU is found; otherwise wait for resources
  and retry resource errors (out of memory, context creation) up to `gpu_timeout`, then
  error; any other GPU error rethrows. Waiting happens only while another process holds
  the memory. A request (batch plus PSF table, with a 1.5x margin) that no idle GPU can
  hold, because its free memory or its total memory is below the request, halves the
  batch and polls again at once; for one ROI it errors, naming the request and the largest
  free memory of an idle GPU. An out-of-memory error also halves the batch and retries at
  once. A one-ROI out-of-memory failure that follows a poll which returned at once (the
  device was reported available without waiting) is retried after a 0.5 s pause; the third
  in a row is permanent and rethrows even at `gpu_timeout = Inf`. A poll that had to wait
  resets that count. After any other resource failure whose poll returned at once, the
  loop also pauses 0.5 s before polling again. Each batch size logs one warning.
- `:auto` - Use CPU silently if no functional GPU is found; otherwise wait for resources
  and retry resource errors (out of memory, context creation) up to `auto_timeout`, then
  warn and fall back to CPU; any other GPU error warns and falls back to CPU at once.
  Out-of-memory errors and requests no idle GPU can hold halve the batch as in `:gpu`, and
  the CPU is used at once (with one warning) when one ROI cannot fit any idle GPU or when
  three one-ROI out-of-memory failures in a row follow polls that returned at once

A failure while releasing GPU memory after an error is logged as a warning and never
replaces the original error, which `:gpu` rethrows and `:auto` handles as above.

GPU modes prefer CUDA. When CUDA is not functional, an AMD GPU can be used by loading
AMDGPU (`using AMDGPU`) before `fit`.

# Examples
```julia
# Simple usage with auto GPU detection
fitter = GaussMLEConfig()

# Force CPU execution
cpu_fitter = GaussMLEConfig(backend = :cpu)

# Variable sigma model with a higher cap on pixel passes (default 100)
fitter = GaussMLEConfig(
    psf_model = GaussianXYNBS(),
    iterations = 200
)

# Explicit GPU with custom timeout
fitter = GaussMLEConfig(
    backend = :gpu,
    gpu_timeout = 60.0,
    on_wait = (e, a, r) -> @info "Waiting..." elapsed=e
)
```

# See also
[`fit`](@ref), [`GaussianXYNB`](@ref), [`GaussianXYNBS`](@ref)
"""
function GaussMLEConfig(;
        psf_model = GaussianXYNB(0.13f0),
        backend::Symbol = :auto,
        device = nothing,  # deprecated, use backend
        iterations = _default_iterations(psf_model),
        constraints = nothing,
        batch_size = 10_000,
        auto_timeout = 300.0,
        gpu_timeout = Inf,
        on_wait = nothing
    )
    # Handle deprecated device kwarg
    if device !== nothing
        @warn "device kwarg is deprecated, use backend instead" maxlog = 1
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

    return GaussMLEConfig(
        backend, psf_model, iterations, constraints, batch_size,
        Float64(auto_timeout), Float64(gpu_timeout), on_wait
    )
end

"""
    estimate_batch_memory(batch_size, box_size, n_params; map_bytes = box_size^2 * 4)

Estimate GPU memory needed for one batch of MLE fitting.

Memory components:
- data: batch_size × box_size² × 4 bytes
- results: batch_size × n_params × 4 bytes
- uncertainties: batch_size × n_params × 4 bytes
- covariances: batch_size × 3 × 4 bytes
- log-likelihoods and Pearson statistics: batch_size × 2 × 4 bytes
- corners: batch_size × 2 × 4 bytes
- variance map: `map_bytes` (shared; the kernel receives the whole camera map, so pass
  `4 * length(variance_map)` for a sCMOS fit)
"""
function estimate_batch_memory(
        batch_size::Integer, box_size::Integer, n_params::Integer;
        map_bytes::Integer = box_size^2 * 4
    )
    data_bytes = batch_size * box_size^2 * 4
    results_bytes = batch_size * n_params * 4
    uncertainties_bytes = batch_size * n_params * 4
    covariances_bytes = batch_size * 3 * 4
    statistics_bytes = batch_size * 2 * 4
    corners_bytes = batch_size * 2 * 4
    return data_bytes + results_bytes + uncertainties_bytes + covariances_bytes +
        statistics_bytes + corners_bytes + map_bytes
end

# Goodness-of-fit p-values from the Pearson statistic `chi2` = sum (d - mu)^2 / variance
# over a fit's pixels, against chi-squared with `df` degrees of freedom. `ccdf` keeps the
# upper tail exact where `1 - cdf` rounds to zero in Float32.
function _gof_pvalues(chi2::AbstractVector{Float32}, df::Integer)
    dist = Chisq(df)
    return Float32[Float32(ccdf(dist, Float64(c))) for c in chi2]
end

"""
    compute_cpu_batch_size(n_fits, box_size, n_params; memory_multiplier=4)

Compute optimal batch size for CPU processing based on available memory.
Returns n_fits if all data fits in memory, otherwise a smaller batch size.
"""
function compute_cpu_batch_size(
        n_fits::Integer, box_size::Integer, n_params::Integer;
        memory_multiplier::Integer = 4
    )
    free_mem = Sys.free_memory()
    # data + results + uncertainties + covariances + llr
    memory_per_roi = (box_size^2 + n_params * 2 + 3 + 1) * 4
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

# KA's default can overflow POCL's work-group stack with few Julia threads;
# small groups keep every thread busy.
# Measured on kitt: ~34-42% higher XYNB throughput with 16-32 Julia threads.
const CPU_WORKGROUP_SIZE = 64

# After a failed GPU attempt. A context that could not be created is released, so its
# reservation cannot block other processes. A live context is kept: releasing CUDA's primary
# context under CUDA.jl makes every later call in this process fail with
# CONTEXT_IS_DESTROYED (709), so a retry with a smaller batch could never run. Only the
# failed attempt's arrays are collected and the pool's memory returned.
function _cleanup_after_failure(vendor, dev_idx, context_active::Bool)
    if context_active
        GC.gc(true)
        _reclaim_gpu(vendor)
    else
        _release_gpu_context(vendor, dev_idx)
    end
    return nothing
end

# A fit whose x or y (and z, for a model with an axial position) ended exactly on a bound
# of `constraints` did not localize an emitter inside the allowed region: its Fisher matrix
# says nothing about where the emitter is. Its uncertainties become `Inf` and its
# covariances zero, the convention of a failed inversion; fitted parameters, log-likelihood
# and chi2 stay. The clamp returns the bound itself, so equality is exact. A parameter
# the caller holds fixed (equal bounds) is skipped.
function _mark_pinned_fits!(
        uncertainties::Matrix{Float32}, covariances::Matrix{Float32},
        results::Matrix{Float32}, constraints::ParameterConstraints, psf_model::PSFModel
    )
    n_pinned = _has_axial_position(typeof(psf_model)) ? 3 : 2
    for i in axes(results, 2)
        pinned = false
        for k in 1:n_pinned
            # A parameter the caller holds fixed (lower == upper) is not a bound hit.
            constraints.lower[k] == constraints.upper[k] && continue
            pinned |=
                results[k, i] == constraints.lower[k] ||
                results[k, i] == constraints.upper[k]
        end
        if pinned
            uncertainties[:, i] .= Inf32
            covariances[:, i] .= 0.0f0
        end
    end
    return nothing
end

# Consecutive one-ROI out-of-memory failures on an available device before the failure is
# treated as permanent, and the pause between them in seconds.
const ONE_ROI_OOM_LIMIT = 3
const ONE_ROI_OOM_PAUSE = 0.5

# Only a tabulated model has a finite lateral extent; SplinePSFModel specializes this.
_validate_psf_box(::Any, ::Integer) = nothing

"""
    _run_mle_kernel!(results, uncertainties, covariances, log_likelihoods, chi2,
                     data, psf_pixels, use_scmos, variance_map, x_corners, y_corners,
                     fitter; vendor) -> (actual_backend, device_id, actual_batch_size,
                     actual_n_batches)

Internal helper that runs the MLE kernel on CPU or GPU with appropriate batching.
Handles all backend selection and batch management logic.

# Arguments
- Pre-allocated output arrays: results, uncertainties, covariances, log_likelihoods, chi2
  (the Pearson statistic the p-values come from)
- Input data and parameters: data, psf_pixels, use_scmos, variance_map, x_corners,
  y_corners
- fitter: GaussMLEConfig with backend/batching configuration
- vendor: the GPU vendor (`nothing` for none); defaults to `_gpu_vendor()` unless the
  backend is `:cpu`. A test passes a fake vendor to drive the retry loop.

# Returns
Tuple of (actual_backend::Symbol, device_id::Int, actual_batch_size::Int,
actual_n_batches::Int) for use in GaussMLEFitInfo construction.
"""
function _run_mle_kernel!(
        results::Matrix{Float32},
        uncertainties::Matrix{Float32},
        covariances::Matrix{Float32},
        log_likelihoods::Vector{Float32},
        chi2::Vector{Float32},
        data::Array{Float32, 3},
        psf_pixels,
        use_scmos::Val,
        variance_map::Matrix{Float32},
        x_corners::AbstractVector{Int32},
        y_corners::AbstractVector{Int32},
        fitter::GaussMLEConfig;
        vendor = fitter.backend == :cpu ? nothing : _gpu_vendor()
    )
    n_fits = size(data, 3)
    n_params = size(results, 1)
    box_size = size(data, 1)
    # Every fit method reaches the kernel here, so a ROI the PSF table cannot cover is
    # refused before the CPU path and before any GPU poll.
    _validate_psf_box(psf_pixels, box_size)
    # Every sCMOS entry point reaches the kernel here, and the kernel indexes the variance
    # map by corner, so every ROI must lie on the map's sensor.
    use_scmos isa Val{true} &&
        _validate_roi_corners(box_size, x_corners, y_corners, size(variance_map))
    constraints = _resolve_constraints(fitter.constraints, fitter.psf_model, box_size)

    # Track for GaussMLEFitInfo
    actual_backend = :cpu
    device_id = -1
    actual_batch_size = 0
    actual_n_batches = 0

    # --- CPU fast path ---
    use_cpu = fitter.backend == :cpu || (fitter.backend != :gpu && vendor === nothing)

    if fitter.backend == :gpu && vendor === nothing
        error("No functional GPU found: CUDA is not functional; AMD GPUs require \
            using AMDGPU (ROCm) before fit")
    end

    if !use_cpu
        # --- Unified GPU retry loop ---
        # Handles: (1) no free memory (vendor polls), (2) TOCTOU context creation
        # race, (3) runtime OOM - all in one loop. On an out-of-memory error, halve the
        # batch cap and retry at once; only a failing one-ROI batch (or a device that is
        # unavailable) waits and re-polls. Any other error rethrows (:gpu) or falls to
        # the CPU (:auto) at once. Falls to CPU (:auto) or errors (:gpu) on timeout.
        # The cleanup after a failure runs in its own try: it can throw on a poisoned
        # context and must never replace the error that sent us here.
        timeout = fitter.backend == :auto ? fitter.auto_timeout : fitter.gpu_timeout
        deadline = time() + timeout
        batch_cap = max(1, min(fitter.batch_size, n_fits))
        psf_bytes = _psf_table_bytes(psf_pixels)
        # The kernel receives the whole variance map, so its real size is counted.
        map_bytes = sizeof(variance_map)
        failed_batch = 0  # the batch cap of the last resource failure
        gpu_succeeded = false
        # An out-of-memory error for one ROI while the poll reports the device available
        # may be a transient clash or permanent (a hard memory limit below the table plus
        # one ROI): after ONE_ROI_OOM_LIMIT in a row it is treated as permanent.
        one_roi_ooms = 0
        warned_batches = Set{Int}()  # batch sizes whose resource warning was logged

        # A single ROI that exceeds every device's total memory can never fit.
        one_roi_bytes = estimate_batch_memory(1, box_size, n_params; map_bytes) +
            psf_bytes
        totals = try
            _gpu_total_memory(vendor)
        catch e
            e isa InterruptException && rethrow()
            Int[]  # best effort: a failed query skips the check
        end
        if _exceeds_device_memory(one_roi_bytes, totals)
            msg = "One ROI (box $box_size) plus the PSF table needs about " *
                "$(Base.format_bytes(round(Int, one_roi_bytes * GPU_MEMORY_MARGIN))) " *
                "with margin, more than the largest GPU's " *
                "$(Base.format_bytes(maximum(totals))) total memory"
            fitter.backend == :gpu && error(msg)
            @warn "$msg; using CPU"
            use_cpu = true
        end

        while !gpu_succeeded && !use_cpu && time() < deadline
            required_memory = estimate_batch_memory(
                batch_cap, box_size, n_params; map_bytes
            ) + psf_bytes
            # Poll the vendor for an available GPU
            # Every poll tick that finds no device calls `on_wait` first, so a call
            # tells whether the poll had to wait.
            poll_waited = Ref(false)
            tracked_wait = (elapsed, free, required) -> begin
                poll_waited[] = true
                fitter.on_wait === nothing || fitter.on_wait(elapsed, free, required)
                return nothing
            end
            dev_idx, available, failed, never_fits = _poll_gpu(
                vendor, fitter.backend, required_memory;
                timeout = max(0.0, deadline - time()), on_wait = tracked_wait
            )
            if failed
                use_cpu = true
                break
            end
            if never_fits !== nothing
                # No visible device can hold this request even when idle, so waiting
                # cannot help: shrink the batch, or give up at one ROI.
                if batch_cap > 1
                    batch_cap = cld(batch_cap, 2)
                    continue
                end
                msg = "One ROI (box $box_size) plus the PSF table needs about " *
                    "$(Base.format_bytes(round(Int, required_memory * GPU_MEMORY_MARGIN))) \
                    with margin (batch of $batch_cap ROI), more than the largest free \
                    memory of an idle GPU, $(Base.format_bytes(never_fits))"
                fitter.backend == :gpu && error(msg)
                @warn "$msg; using CPU"
                use_cpu = true
                break
            end
            if !available
                break  # Timeout reached
            end

            context_active = false
            try
                # Acquire GPU context
                _activate_gpu!(vendor, dev_idx)
                context_active = true
                gpu_backend = _ka_gpu_backend(vendor)
                # Transfer a spline's table once for all batches on this device. A retry
                # may replace the device/context, so it must adapt the host model again.
                device_psf = _prepare_psf_model(gpu_backend, psf_pixels)
                # The variance map is the same for every batch: upload it once per attempt.
                d_variance_map = KernelAbstractions.allocate(
                    gpu_backend, Float32, size(variance_map)
                )
                copyto!(d_variance_map, variance_map)

                # Process all batches on GPU
                actual_batch_size = batch_cap
                actual_n_batches = cld(n_fits, batch_cap)
                for batch_start in 1:batch_cap:n_fits
                    batch_end = min(batch_start + batch_cap - 1, n_fits)
                    batch_size_actual = batch_end - batch_start + 1

                    batch_data = data[:, :, batch_start:batch_end]
                    d_batch_data = KernelAbstractions.allocate(
                        gpu_backend, Float32, size(batch_data)
                    )
                    copyto!(d_batch_data, batch_data)

                    batch_x_corners = x_corners[batch_start:batch_end]
                    batch_y_corners = y_corners[batch_start:batch_end]
                    d_x_corners = KernelAbstractions.allocate(
                        gpu_backend, Int32, length(batch_x_corners)
                    )
                    d_y_corners = KernelAbstractions.allocate(
                        gpu_backend, Int32, length(batch_y_corners)
                    )
                    copyto!(d_x_corners, batch_x_corners)
                    copyto!(d_y_corners, batch_y_corners)

                    d_results = KernelAbstractions.allocate(
                        gpu_backend, Float32, (n_params, batch_size_actual)
                    )
                    d_uncertainties = KernelAbstractions.allocate(
                        gpu_backend, Float32, (n_params, batch_size_actual)
                    )
                    d_covariances = KernelAbstractions.allocate(
                        gpu_backend, Float32, (3, batch_size_actual)
                    )
                    d_log_likelihoods = KernelAbstractions.allocate(
                        gpu_backend, Float32, batch_size_actual
                    )
                    d_chi2 = KernelAbstractions.allocate(
                        gpu_backend, Float32, batch_size_actual
                    )

                    kernel = unified_gaussian_mle_kernel!(gpu_backend)
                    kernel(
                        d_results, d_uncertainties, d_covariances, d_log_likelihoods,
                        d_chi2, d_batch_data, device_psf, use_scmos, d_variance_map,
                        d_x_corners, d_y_corners,
                        constraints, fitter.iterations,
                        ndrange = batch_size_actual
                    )

                    KernelAbstractions.synchronize(gpu_backend)

                    results[:, batch_start:batch_end] = Array(d_results)
                    uncertainties[:, batch_start:batch_end] = Array(d_uncertainties)
                    covariances[:, batch_start:batch_end] = Array(d_covariances)
                    log_likelihoods[batch_start:batch_end] = Array(d_log_likelihoods)
                    chi2[batch_start:batch_end] = Array(d_chi2)
                end

                # All batches succeeded - reclaim pool memory so queries show it as free
                # for other processes competing for the same GPU
                # (the results are on the host; a failing reclaim must not discard them)
                try
                    _reclaim_gpu(vendor)
                catch re
                    re isa InterruptException && rethrow()
                    @warn "GPU memory reclaim failed after the fit" exception = (
                        re, catch_backtrace(),
                    )
                end
                gpu_succeeded = true
                actual_backend = :gpu
                device_id = try
                    _active_device_id(vendor)
                catch ie
                    ie isa InterruptException && rethrow()
                    dev_idx
                end
            catch e
                # Only a resource error (out of memory, context creation) can clear by
                # waiting. Anything else is deterministic: release the GPU, then rethrow
                # under :gpu or fall back to the CPU at once under :auto.
                resource = _is_resource_error(vendor, e)
                oom = resource && _is_oom_error(vendor, e)
                if resource && !(batch_cap in warned_batches)
                    push!(warned_batches, batch_cap)
                    @warn "GPU error on device $dev_idx with a batch of $batch_cap \
                        ROIs, releasing resources" exception = (e, catch_backtrace())
                end
                try
                    _cleanup_after_failure(vendor, dev_idx, context_active)
                catch ce
                    ce isa InterruptException && rethrow()
                    @warn "GPU cleanup on device $dev_idx failed; handling the original \
                        error" cleanup_exception = (ce, catch_backtrace())
                end
                # Only a failure after a poll that returned at once counts toward the
                # permanent one-ROI out-of-memory error: a poll that waited means the
                # device was busy, and the failure may clear.
                one_roi_ooms = oom && batch_cap == 1 && !poll_waited[] ?
                    one_roi_ooms + 1 : 0
                if resource
                    failed_batch = batch_cap
                    if oom && batch_cap > 1
                        batch_cap = cld(batch_cap, 2)  # smaller batch, retry at once
                    elseif one_roi_ooms >= ONE_ROI_OOM_LIMIT
                        # Permanent: waiting does not clear it. The poll returned at once
                        # each time, so the loop would otherwise spin until the deadline.
                        fitter.backend == :gpu && rethrow()
                        @warn "One ROI (box $box_size) plus the PSF table ran out of \
                            memory $one_roi_ooms times in a row on an available GPU; \
                            using CPU"
                        use_cpu = true
                        break
                    else
                        fitter.backend == :gpu && time() >= deadline && rethrow()
                        # A transient clash can clear. A poll that returned at once found
                        # the device available, so pause before the next attempt instead
                        # of spinning; a poll that waited already did.
                        poll_waited[] ||
                            sleep(min(ONE_ROI_OOM_PAUSE, max(0.0, deadline - time())))
                    end
                else
                    (fitter.backend == :gpu || e isa InterruptException) && rethrow()
                    @warn "GPU error on device $dev_idx, using CPU" exception = (
                        e, catch_backtrace(),
                    )
                    use_cpu = true
                    break
                end
            end
        end

        if !gpu_succeeded
            if fitter.backend == :gpu
                error(
                    "GPU failed after $(timeout)s timeout" *
                        (failed_batch > 0 ? " (last failure with a batch of \
                        $failed_batch ROIs)" : "")
                )
            end
            # :auto falls through to CPU below
            if !use_cpu
                @warn "GPU unavailable after $(timeout)s, using CPU" * (
                    failed_batch > 0 ? " (last failure with a batch of $failed_batch \
                    ROIs)" : ""
                )
            end
            use_cpu = true
        end
    end

    # --- CPU path ---
    if use_cpu
        actual_backend = :cpu
        device_id = -1
        ka_backend = KernelAbstractions.CPU()
        cpu_psf = _prepare_psf_model(ka_backend, psf_pixels)
        cpu_batch_size = compute_cpu_batch_size(n_fits, box_size, n_params)
        actual_batch_size = cpu_batch_size >= n_fits ? n_fits : cpu_batch_size

        if cpu_batch_size >= n_fits
            # Process all at once
            actual_n_batches = 1
            kernel = unified_gaussian_mle_kernel!(ka_backend, CPU_WORKGROUP_SIZE)
            kernel(
                results, uncertainties, covariances, log_likelihoods, chi2,
                data, cpu_psf, use_scmos, variance_map, x_corners, y_corners,
                constraints, fitter.iterations,
                ndrange = n_fits
            )
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

                batch_results = @view results[:, batch_start:batch_end]
                batch_uncertainties = @view uncertainties[:, batch_start:batch_end]
                batch_covariances = @view covariances[:, batch_start:batch_end]
                batch_llr = @view log_likelihoods[batch_start:batch_end]
                batch_chi2 = @view chi2[batch_start:batch_end]

                kernel = unified_gaussian_mle_kernel!(ka_backend, CPU_WORKGROUP_SIZE)
                kernel(
                    batch_results, batch_uncertainties, batch_covariances, batch_llr,
                    batch_chi2, batch_data, cpu_psf, use_scmos, variance_map,
                    batch_x_corners, batch_y_corners,
                    constraints, fitter.iterations,
                    ndrange = batch_size_actual
                )
                KernelAbstractions.synchronize(ka_backend)

                batch_data = nothing
                GC.gc(false)
            end
        end
    end

    _mark_pinned_fits!(uncertainties, covariances, results, constraints, fitter.psf_model)
    return (actual_backend, device_id, actual_batch_size, actual_n_batches)
end

"""
    fit(data::AbstractArray{T,3}, fitter::GaussMLEConfig; variance_map=nothing) ->
        (BasicSMLD, GaussMLEFitInfo)

Fit Gaussian blobs to a stack of ROIs using Maximum Likelihood Estimation.

# Arguments
- `data::AbstractArray{T,3}`: ROI data as (roi_size, roi_size, n_rois) array
- `fitter::GaussMLEConfig`: Configured fitter object

# Keyword Arguments
- `variance_map=nothing`: Poisson noise when `nothing`; a supplied readnoise-variance map
  (electrons²) selects the sCMOS path for data already calibrated in electrons

Array input assumes 0.1 μm pixels (a `SplinePSFModel` keeps its own `pixel_size`, so
emitter x, y in microns follow the model's pixel) and synthetic horizontally tiled ROI
corners.
For n ROIs of side b, `variance_map` is one of
- a `(b, b)` map, used for every ROI;
- a `(b, b, n)` array, one map per ROI;
- a `(b, b*n)` map tiled horizontally like the ROI corners (ROI k reads columns
  `(k-1)*b+1:k*b`); a larger map that covers rows `1:b` and columns `1:b*n` is also read
  that way.

Any other shape throws an `ArgumentError`.
Use `ROIBatch` for real camera calibration and coordinates.

# Returns
- `Tuple{BasicSMLD, GaussMLEFitInfo}`: Fitted localizations and fit metadata

A fit with a singular Fisher matrix, or whose x, y (or z) ended exactly on a bound of its
constraints, reports `Inf` for every uncertainty and zero covariances; its parameters,
log-likelihood and p-value are as fitted.

# Examples
```julia
using Statistics: mean

# Fit 1000 ROIs
data = zeros(Float32, 7, 7, 1000)  # Your ROI data here
fitter = GaussMLEConfig(psf_model = GaussianXYNB(0.13f0))
smld, info = fit(data, fitter)

# Access results
println("Fitted \$(info.n_fits) ROIs in \$(info.elapsed_s * 1000) ms")
println("Mean x position: ", mean([e.x for e in smld.emitters]))
```

# See also
[`GaussMLEConfig`](@ref), [`GaussMLEFitInfo`](@ref)
"""
function fit(
        data::AbstractArray{T, 3}, fitter::GaussMLEConfig;
        variance_map = nothing
    ) where {T}
    # Start timing
    t0 = time_ns()

    # Validate input
    validate_fit_input(data, nothing)

    n_fits = size(data, 3)
    n_params = length(fitter.psf_model)
    box_size = size(data, 1)

    # Convert data to Float32 if needed
    data_f32 = convert(Array{Float32, 3}, data)

    # Pixel size for Array input (no camera attached): 0.1 µm, or the model's own
    pixel_size = _array_fit_pixel_size(fitter.psf_model)
    psf_pixels = to_pixel_units(fitter.psf_model, pixel_size)

    # Allocate result arrays
    results = Matrix{Float32}(undef, n_params, n_fits)
    uncertainties = Matrix{Float32}(undef, n_params, n_fits)
    covariances = Matrix{Float32}(undef, 3, n_fits)  # [σ_xy, σ_xz, σ_yz]
    log_likelihoods = Vector{Float32}(undef, n_fits)
    chi2 = Vector{Float32}(undef, n_fits)

    # Determine camera model from variance_map keyword
    use_scmos = isnothing(variance_map) ? Val(false) : Val(true)
    # Kernel corners index the map: tiled for a `(box, box * n)` map, 1 for every ROI when
    # one `(box, box)` map serves all. The output coordinates use `x_corners_smld` below.
    x_corners = Int32[1 + (i - 1) * box_size for i in 1:n_fits]
    y_corners = fill(Int32(1), n_fits)
    if isnothing(variance_map)
        var_map = zeros(Float32, box_size, box_size)
    else
        var_map, shared = _kernel_variance_map(Float32.(variance_map), box_size, n_fits)
        shared && fill!(x_corners, Int32(1))
    end

    # Run MLE kernel with batching
    actual_backend, device_id, actual_batch_size, actual_n_batches = _run_mle_kernel!(
        results, uncertainties, covariances, log_likelihoods, chi2,
        data_f32, psf_pixels, use_scmos, var_map, x_corners, y_corners,
        fitter
    )

    # Goodness-of-fit p-values: Pearson statistic against χ²(df), df = n_pixels - n_params
    pvalues = _gof_pvalues(chi2, box_size^2 - n_params)

    # Create minimal ROIBatch for SMLD conversion
    roi_size = size(data_f32, 1)
    # [1, 1+roi_size, ...] (1-indexed for Julia)
    x_corners_smld = Int32[1 + (i - 1) * roi_size for i in 1:n_fits]
    y_corners_smld = fill(Int32(1), n_fits)  # All at y=1
    frame_indices = ones(Int32, n_fits)

    # Create minimal camera for SMLD conversion (fit(Array) has no real camera)
    camera_smld = SMLMData.IdealCamera(0:1023, 0:1023, pixel_size)

    batch = SMLMData.ROIBatch(
        data_f32, x_corners_smld, y_corners_smld, frame_indices, camera_smld
    )
    loc_result = create_localization_result(
        results, uncertainties, covariances, log_likelihoods, pvalues, batch,
        fitter.psf_model
    )

    # Calculate elapsed time and create GaussMLEFitInfo
    elapsed_s = (time_ns() - t0) / 1.0e9
    memory_per_batch = estimate_batch_memory(
        actual_batch_size, box_size, n_params; map_bytes = sizeof(var_map)
    ) + _psf_table_bytes(psf_pixels)
    info = GaussMLEFitInfo(
        elapsed_s, actual_backend, device_id, n_fits, n_fits, actual_batch_size,
        actual_n_batches, memory_per_batch
    )

    # Return tuple (BasicSMLD, GaussMLEFitInfo)
    return (to_smld(loc_result, batch), info)
end

# The map and the tiling of the kernel corners for fit(Array)'s `variance_map` keyword:
# `(map, shared)`. A `(box, box)` map serves every ROI as is (`shared`, corners of 1); a
# `(box, box, n)` map is reshaped to the `(box, box * n)` map that the tiled corners index,
# and so is a map that already covers rows `1:box` and columns `1:box * n`.
function _kernel_variance_map(vm::AbstractArray{Float32}, box::Int, n::Int)
    sz = size(vm)
    if ndims(vm) == 2 && sz[1] >= box && sz[2] >= box * n
        return Matrix(vm), false
    elseif sz == (box, box)
        return Matrix(vm), true
    elseif sz == (box, box, n)
        # ROI k fills columns (k-1)*box+1:k*box
        return reshape(Array(vm), box, box * n), false
    end
    throw(
        ArgumentError(
            "variance_map of size $sz does not fit $n ROIs of $box×$box pixels: use \
            ($box, $box) (one map for every ROI), ($box, $box, $n) (one map per ROI) or \
            ($box, $(box * n)) (the maps tiled horizontally)"
        )
    )
end

# Pixel size (µm) that fit(::Array) uses: 0.1 µm for every model except SplinePSFModel,
# which keeps its own (x, y) pixel size.
_array_fit_pixel_size(::PSFModel) = 0.1f0

# Convenience function for single ROI fitting
function fit(roi::AbstractMatrix{T}, fitter::GaussMLEConfig) where {T}
    # Reshape to 3D array with single ROI
    data = reshape(roi, size(roi, 1), size(roi, 2), 1)
    smld, info = fit(data, fitter)

    # Return first emitter and info
    return (smld.emitters[1], info)
end

# Models can specialize camera conversion without changing the ROIBatch fit methods.
function _model_for_camera(model::PSFModel, camera)
    pixel_size = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    return to_pixel_units(model, pixel_size)
end

# Fit method for ROIBatch - returns (BasicSMLD, GaussMLEFitInfo) with real camera
# coordinates
function fit(
        roi_batch::ROIBatch{T, N, A, <:SMLMData.IdealCamera}, fitter::GaussMLEConfig
    ) where {T, N, A}
    # Start timing
    t0 = time_ns()

    # Fit the raw data with Poisson-only likelihood
    n_fits = size(roi_batch.data, 3)
    n_params = length(fitter.psf_model)
    box_size = size(roi_batch.data, 1)

    # Convert the model using its camera pixel geometry.
    psf_pixels = _model_for_camera(fitter.psf_model, roi_batch.camera)

    # Allocate result arrays
    results = Matrix{Float32}(undef, n_params, n_fits)
    uncertainties = Matrix{Float32}(undef, n_params, n_fits)
    covariances = Matrix{Float32}(undef, 3, n_fits)  # [σ_xy, σ_xz, σ_yz]
    log_likelihoods = Vector{Float32}(undef, n_fits)
    chi2 = Vector{Float32}(undef, n_fits)

    # IdealCamera: Poisson noise only (no variance map)
    use_scmos = Val(false)
    # Unused but needed for kernel signature
    variance_map = zeros(Float32, box_size, box_size)

    # Use unified kernel on CPU/GPU
    data_f32 = convert(Array{Float32, 3}, roi_batch.data)

    # Run MLE kernel with batching
    actual_backend, device_id, actual_batch_size, actual_n_batches = _run_mle_kernel!(
        results, uncertainties, covariances, log_likelihoods, chi2,
        data_f32, psf_pixels, use_scmos, variance_map, roi_batch.x_corners,
        roi_batch.y_corners,
        fitter
    )

    # Goodness-of-fit p-values: Pearson statistic against χ²(df), df = n_pixels - n_params
    pvalues = _gof_pvalues(chi2, box_size^2 - n_params)

    # Use real ROIBatch for coordinate conversion (preserves corners!)
    loc_result = create_localization_result(
        results, uncertainties, covariances, log_likelihoods, pvalues, roi_batch,
        fitter.psf_model
    )

    # Calculate elapsed time and create GaussMLEFitInfo
    elapsed_s = (time_ns() - t0) / 1.0e9
    memory_per_batch = estimate_batch_memory(
        actual_batch_size, box_size, n_params; map_bytes = sizeof(variance_map)
    ) + _psf_table_bytes(psf_pixels)
    info = GaussMLEFitInfo(
        elapsed_s, actual_backend, device_id, n_fits, n_fits, actual_batch_size,
        actual_n_batches, memory_per_batch
    )

    return (to_smld(loc_result, roi_batch), info)
end

# Fit method for ROIBatch with SMLMData.SCMOSCamera
function fit(
        roi_batch::ROIBatch{T, N, A, <:SMLMData.SCMOSCamera}, fitter::GaussMLEConfig
    ) where {T, N, A}
    # Start timing
    t0 = time_ns()

    variance_map = extract_variance_map(roi_batch.camera, Float32)
    # Checked here too, before to_electrons indexes the calibration at the corners.
    _validate_roi_corners(
        size(roi_batch.data, 1), roi_batch.x_corners, roi_batch.y_corners,
        size(variance_map)
    )

    # Preprocess: ADU → electrons using per-pixel calibration at ROI positions
    data_electrons = to_electrons(
        roi_batch.data, roi_batch.camera, roi_batch.x_corners, roi_batch.y_corners
    )

    n_fits = size(data_electrons, 3)
    n_params = length(fitter.psf_model)
    box_size = size(data_electrons, 1)

    # Convert the model using its camera pixel geometry.
    psf_pixels = _model_for_camera(fitter.psf_model, roi_batch.camera)

    # Allocate result arrays
    results = Matrix{Float32}(undef, n_params, n_fits)
    uncertainties = Matrix{Float32}(undef, n_params, n_fits)
    covariances = Matrix{Float32}(undef, 3, n_fits)  # [σ_xy, σ_xz, σ_yz]
    log_likelihoods = Vector{Float32}(undef, n_fits)
    chi2 = Vector{Float32}(undef, n_fits)

    # SCMOSCamera: Poisson + readnoise variance
    use_scmos = Val(true)

    # Use unified kernel on CPU/GPU (data already in electrons)
    data_f32 = convert(Array{Float32, 3}, data_electrons)

    # Run MLE kernel with batching
    actual_backend, device_id, actual_batch_size, actual_n_batches = _run_mle_kernel!(
        results, uncertainties, covariances, log_likelihoods, chi2,
        data_f32, psf_pixels, use_scmos, variance_map, roi_batch.x_corners,
        roi_batch.y_corners,
        fitter
    )

    # Goodness-of-fit p-values: Pearson statistic against χ²(df), df = n_pixels - n_params
    pvalues = _gof_pvalues(chi2, box_size^2 - n_params)

    # Use original ROIBatch for coordinate conversion (preserves corners and original
    # camera!)
    loc_result = create_localization_result(
        results, uncertainties, covariances, log_likelihoods, pvalues, roi_batch,
        fitter.psf_model
    )

    # Calculate elapsed time and create GaussMLEFitInfo
    elapsed_s = (time_ns() - t0) / 1.0e9
    memory_per_batch = estimate_batch_memory(
        actual_batch_size, box_size, n_params; map_bytes = sizeof(variance_map)
    ) + _psf_table_bytes(psf_pixels)
    info = GaussMLEFitInfo(
        elapsed_s, actual_backend, device_id, n_fits, n_fits, actual_batch_size,
        actual_n_batches, memory_per_batch
    )

    return (to_smld(loc_result, roi_batch), info)
end

"""
    fit(batch::ROIBatch; psf_model=GaussianXYNB(0.13f0), iterations,
        backend=:auto, ...) -> (BasicSMLD, GaussMLEFitInfo)

Convenience form of fit() that creates a GaussMLEConfig from keyword arguments.
Kwargs match GaussMLEConfig fields exactly.

# Arguments
- `batch::ROIBatch`: Input ROI data with camera calibration

# Keyword Arguments
- `psf_model=GaussianXYNB(0.13f0)`: PSF model to use
- `iterations`: Cap on the Levenberg-Marquardt pixel passes of a fit (of each
  start for `SplinePSFModel`); the last pass only evaluates, so a fit takes at most
  `iterations - 1` steps and `iterations = 1` returns the start. Default 100 for the
  Gaussian models, which stop earlier once converged (one that reaches the cap first
  reports infinite uncertainties), and 20 for `SplinePSFModel`, which runs every pass
- `backend=:auto`: Compute backend (`:cpu`, `:gpu`, or `:auto`)
- `constraints=nothing`: Parameter constraints; `nothing` builds defaults from the ROI size
  at fit time
- `batch_size=10_000`: Batch size for GPU processing
- `auto_timeout=300.0`: Seconds to wait for GPU in auto mode
- `gpu_timeout=Inf`: Seconds to wait for GPU in explicit gpu mode
- `on_wait=nothing`: Callback for GPU wait progress

# Throws
`ArgumentError` for an `SCMOSCamera` batch with an ROI that does not lie on the sensor
(`1 <= corner` and `corner + box - 1 <= sensor size` on both axes), naming the first such
ROI, before any kernel launch.

# Returns
- `Tuple{BasicSMLD, GaussMLEFitInfo}`: Fitted localizations and fit metadata

A fit with a singular Fisher matrix, or whose x, y (or z) ended exactly on a bound of its
constraints, reports `Inf` for every uncertainty and zero covariances; its parameters,
log-likelihood and p-value are as fitted.

# Examples
Fit 10 simulated ROIs on the CPU:
```jldoctest
julia> batch = generate_roi_batch(IdealCamera(64, 64, 0.1), GaussianXYNB(0.13f0);
           n_rois = 10, seed = 1);

julia> smld, info = fit(batch; backend = :cpu);

julia> length(smld.emitters), info.n_fits
(10, 10)
```

With the default `backend = :auto`, which uses a GPU when one is available:
```julia
# Simple fit with defaults
smld, info = fit(batch)

# Custom model and iterations
smld, info = fit(batch; psf_model=GaussianXYNBS(), iterations=200)
```

# See also
[`GaussMLEConfig`](@ref), [`GaussMLEFitInfo`](@ref)
"""
function fit(
        batch::ROIBatch;
        psf_model = GaussianXYNB(0.13f0),
        iterations = _default_iterations(psf_model),
        backend = :auto,
        constraints = nothing,
        batch_size = 10_000,
        auto_timeout = 300.0,
        gpu_timeout = Inf,
        on_wait = nothing
    )
    fitter = GaussMLEConfig(;
        psf_model = psf_model,
        backend = backend,
        iterations = iterations,
        constraints = constraints,
        batch_size = batch_size,
        auto_timeout = auto_timeout,
        gpu_timeout = gpu_timeout,
        on_wait = on_wait
    )
    return fit(batch, fitter)
end


# Export API
export GaussMLEConfig, fit, GaussMLEFitInfo
