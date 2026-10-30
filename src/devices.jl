"""
Device abstraction with automatic GPU detection
"""

# Device abstraction
abstract type ComputeDevice end
struct CPU <: ComputeDevice end
struct GPU <: ComputeDevice end

# GPU vendor interface; ROCm methods live in GaussMLEAMDGPUExt.
abstract type GPUVendor end
struct CUDAGPU <: GPUVendor end
struct ROCmGPU <: GPUVendor end

_functional(::CUDAGPU) = CUDA.functional()
_activate_gpu!(::CUDAGPU, device_idx) = CUDA.device!(device_idx)
_ka_gpu_backend(::CUDAGPU) = CUDABackend()
_reclaim_gpu(::CUDAGPU) = CUDA.reclaim()
_active_device_id(::CUDAGPU) = Int(CUDA.device().handle)
_release_gpu_context(::CUDAGPU, device_idx) = _release_gpu_context(device_idx)

function _poll_gpu(
        ::CUDAGPU, backend::Symbol, required_bytes::Integer; timeout, on_wait = nothing,
    )
    return _poll_gpu(backend, required_bytes; timeout = timeout, on_wait = on_wait)
end

"""
    _gpu_vendor()

Return the functional GPU vendor, preferring CUDA, or `nothing`. AMD requires AMDGPU loaded.
"""
function _gpu_vendor()
    _functional(CUDAGPU()) && return CUDAGPU()
    if Base.get_extension(@__MODULE__, :GaussMLEAMDGPUExt) !== nothing &&
            _functional(ROCmGPU())
        return ROCmGPU()
    end
    return nothing
end

# Smart device selection
function auto_device()
    if CUDA.functional()
        @info "CUDA GPU detected, using GPU acceleration"
        return GPU()
    else
        @info "No CUDA GPU available, using CPU"
        return CPU()
    end
end

# Backend mapping for KernelAbstractions
backend(::CPU) = KernelAbstractions.CPU()
backend(::GPU) = CUDABackend()

"""
    _nvml_device(device_idx::Integer)

Map a 0-based CUDA device ordinal to its NVML device by UUID. CUDA ordinals honor
`CUDA_VISIBLE_DEVICES`; NVML indices enumerate every GPU and do not, so
`NVML.Device(ordinal)` can read the wrong GPU when it is set. `parent_uuid` names the
physical GPU, mapping a MIG instance to its parent as `CUDA.versioninfo` does.
`CUDA.uuid` would return the MIG UUID, which `NVML.Device(::UUID)` looks up as a GPU
UUID and fails. The UUID query needs no CUDA context.
On a MIG instance, memory and process figures are the parent GPU's.
"""
_nvml_device(device_idx::Integer) =
    CUDA.NVML.Device(CUDA.parent_uuid(CUDA.CuDevice(device_idx)))

"""
    _gpu_free_memory(device_idx::Integer; nvml_free = ...) -> (free::Int, from_nvml::Bool)

Query free bytes through NVML first, without creating a CUDA context. `nvml_free` takes
a 0-based CUDA device ordinal. If NVML reports `NVML_ERROR_NOT_SUPPORTED` (for example, the
DGX Spark GB10 driver lacks `nvmlDeviceGetMemoryInfo`), query `CUDA.free_memory()` on
that device instead. All other errors propagate.
"""
function _gpu_free_memory(
        device_idx::Integer;
        nvml_free = idx -> CUDA.NVML.memory_info(_nvml_device(idx)).free,
    )
    try
        return Int(nvml_free(device_idx)), true
    catch e
        if e isa CUDA.NVML.NVMLError && e.code == CUDA.NVML.NVML_ERROR_NOT_SUPPORTED
            return Int(CUDA.device!(CUDA.free_memory, device_idx)), false
        end
        rethrow()
    end
end

"""
    _gpu_contended(device_idx::Integer; my_pid = getpid(),
        processes = CUDA.NVML.compute_processes,
        utilization = CUDA.NVML.utilization_rates) -> Bool

Return true when another process is present and GPU compute utilization exceeds 90%.
`device_idx` is a 0-based CUDA device ordinal. Both queries use its NVML device,
and `utilization` returns compute utilization as a fraction in [0, 1].
A failed process query means not contended; a failed utilization query uses zero.
NVML device lookup errors propagate.
"""
function _gpu_contended(
        device_idx::Integer; my_pid = getpid(),
        processes = CUDA.NVML.compute_processes,
        utilization = CUDA.NVML.utilization_rates
    )
    nvml_dev = _nvml_device(device_idx)
    procs = try
        processes(nvml_dev)
    catch
        # NVML process query can fail on some drivers; skip contention check
        return false
    end
    any(p -> p.first != my_pid, procs) || return false

    # Other processes present - check if they're actually threatening
    util = try
        utilization(nvml_dev)
    catch
        (; compute = 0.0, memory = 0.0)
    end
    return util.compute > 0.9
end

# Multi-GPU support
"""
    find_best_gpu()

Find the GPU with most free memory and switch to it.
Returns the 0-based CUDA device ordinal, which honors `CUDA_VISIBLE_DEVICES`.

For single-GPU systems, simply returns 0.
"""
function find_best_gpu()
    !CUDA.functional() && return 0

    n_devices = length(CUDA.devices())
    n_devices == 1 && return 0

    # Prefer NVML to avoid creating CUDA contexts under contention.
    best_device = 0
    max_free = 0

    for i in 0:(n_devices - 1)
        free, _ = _gpu_free_memory(i)
        if free > max_free
            max_free = free
            best_device = i
        end
    end

    # Activate the winning device after querying free memory.
    CUDA.device!(best_device)

    if n_devices > 1
        @info "Selected GPU $best_device with $(Base.format_bytes(max_free)) free memory"
    end

    return best_device
end

# Allow explicit device selection with fallback (legacy API)
function select_device(device::Union{ComputeDevice, Nothing} = nothing)
    if isnothing(device)
        return auto_device()
    elseif device isa GPU && !CUDA.functional()
        @warn "GPU requested but not available, falling back to CPU"
        return CPU()
    else
        return device
    end
end

"""
    _release_gpu_context(device_idx)

Release the CUDA primary context on a device to free reserved memory.
Called after a failed CUDA.device!() to avoid zombie context reservations
that deadlock other processes polling for free GPU memory.
"""
function _release_gpu_context(device_idx::Integer)
    # GC first to finalize Julia-side CUDA objects before releasing context
    GC.gc(false)
    return try
        dev = CUDA.CuDevice(device_idx)
        CUDA.CUDACore.cuDevicePrimaryCtxRelease(dev)
    catch e
        # Best-effort: context may not exist if creation failed early
        e isa CUDA.CuError || rethrow()
    end
    # NOTE: Do NOT call CUDA.reclaim() here - it requires an active context
    # and will either hang or re-create the context we just released.
end

# GPU memory wait utilities
"""
    wait_for_gpu_memory(required_bytes; timeout=30.0, poll=0.5, on_wait=nothing)

Wait until GPU has sufficient free memory. Returns true if memory became available,
false if timeout reached.

# Arguments
- `required_bytes`: Minimum bytes needed (will wait for 1.5x this amount as safety margin)
- `timeout`: Maximum seconds to wait (default 30.0)
- `poll`: Polling interval in seconds (default 0.5)
- `on_wait`: Optional callback `(elapsed, available, required) -> nothing` for progress
  feedback
"""
function wait_for_gpu_memory(
        required_bytes::Integer;
        timeout::Float64 = 30.0,
        poll::Float64 = 0.5,
        on_wait = nothing
    )

    !CUDA.functional() && return false

    deadline = time() + timeout
    start = time()

    while time() < deadline
        CUDA.reclaim()  # Free cached memory
        available = CUDA.free_memory()

        # 1.5x buffer for fragmentation safety
        if available > required_bytes * 1.5
            return true
        end

        if on_wait !== nothing
            on_wait(time() - start, available, required_bytes)
        end

        # Poll with jitter to avoid thundering herd
        sleep(poll * (1.0 + 0.2 * rand()))
    end

    return false
end

"""
    wait_for_gpu_nvml(required_bytes; timeout=30.0, poll=0.5, on_wait=nothing)

Wait for a GPU with sufficient free memory, preferring NVML without CUDA context creation.
When NVML memory queries are unsupported, query CUDA memory on that device instead.
Scans all visible GPUs each iteration - first available wins.

Returns `(device_index, true)` if a GPU became available, `(-1, false)` if timeout reached.
The index is a 0-based CUDA device ordinal, which honors `CUDA_VISIBLE_DEVICES`.

Contention detection runs only when memory came from NVML: a GPU is considered contended
when another process is present and compute utilization exceeds 90%. GPUs with too little
free memory are skipped before this check. NVML device lookup and memory-query errors
propagate, except for the documented `NVML_ERROR_NOT_SUPPORTED` fallback. A failed process
or utilization query disables that part of the contention check.

# Arguments
- `required_bytes`: Minimum bytes needed (checks for 1.5x as safety margin)
- `timeout`: Maximum seconds to wait (default 30.0)
- `poll`: Polling interval in seconds (default 0.5)
- `on_wait`: Optional callback `(elapsed, available, required) -> nothing` for progress
"""
function wait_for_gpu_nvml(
        required_bytes::Integer;
        timeout::Float64 = 30.0,
        poll::Float64 = 0.5,
        on_wait = nothing
    )

    n_devices = length(CUDA.devices())
    my_pid = getpid()
    required_with_margin = required_bytes * 1.5
    deadline = time() + timeout
    start = time()

    while true
        # Scan all GPUs - first available wins
        for i in 0:(n_devices - 1)
            free, from_nvml = _gpu_free_memory(i)

            # Check if sufficient memory (with fragmentation margin)
            free < required_with_margin && continue

            # The CUDA fallback cannot rely on NVML contention queries.
            from_nvml || return (i, true)

            _gpu_contended(i; my_pid) && continue

            # GPU i is available: sufficient memory, not contended (or sole user)
            return (i, true)
        end

        # No GPU available this tick - check timeout
        if time() >= deadline
            return (-1, false)
        end

        if on_wait !== nothing
            # Report best available memory across all GPUs
            best_free = maximum(first(_gpu_free_memory(i)) for i in 0:(n_devices - 1))
            on_wait(time() - start, best_free, required_bytes)
        end

        # Jittered backoff to avoid thundering herd
        sleep(poll * (1.0 + 0.2 * rand()))
    end
    return
end

"""
    _poll_gpu(backend, required_bytes; timeout, on_wait = nothing)

Poll GPU availability, returning `(device_idx, available, failed)`. Interrupts and
explicit `:gpu` errors propagate; other failures warn once and request CPU fallback.
"""
function _poll_gpu(backend::Symbol, required_bytes::Integer; timeout, on_wait = nothing)
    try
        device_idx, available = wait_for_gpu_nvml(
            required_bytes; timeout = timeout, on_wait = on_wait
        )
        return device_idx, available, false
    catch e
        (backend == :gpu || e isa InterruptException) && rethrow()
        @warn "GPU availability check failed, using CPU" exception = (e, catch_backtrace())
        return -1, false, true
    end
end

"""
    select_backend(backend::Symbol, required_bytes; kwargs...)

Select compute backend with NVML-based contention detection and GPU memory wait.

Memory checks prefer NVML without CUDA context creation, falling back to CUDA when
NVML memory queries are unsupported. In `:auto` mode, polling errors warn and return
`CPU()`; explicit `:gpu` requests propagate them. A runtime try/catch in
`_run_mle_kernel!` provides defense in depth.

# Arguments
- `backend`: `:cpu`, `:gpu`, or `:auto`
- `required_bytes`: Estimated GPU memory needed for operation

# Keyword Arguments
- `auto_timeout=300.0`: Seconds to wait for GPU in auto mode before falling back to CPU
- `gpu_timeout=Inf`: Seconds to wait for GPU in explicit gpu mode
- `on_wait=nothing`: Callback `(elapsed, available, required) -> nothing` for progress

# Returns
- `CPU()` or `GPU()` device instance

# Semantics
- `:cpu` - Always CPU, no waiting
- `:gpu` - Explicit GPU, NVML poll up to gpu_timeout, error if unavailable
- `:auto` - NVML poll up to auto_timeout, fall back to CPU with warning
"""
function select_backend(
        backend::Symbol, required_bytes::Integer;
        auto_timeout::Float64 = 300.0,
        gpu_timeout::Float64 = Inf,
        on_wait = nothing
    )

    backend in (:cpu, :gpu, :auto) || error("backend must be :cpu, :gpu, or :auto")

    if backend == :cpu
        return CPU()

    elseif backend == :gpu
        if !CUDA.functional()
            error("GPU requested but CUDA not functional")
        end
        # NVML poll: scan all GPUs, wait for first available
        # Retry loop handles TOCTOU race: NVML check passes but another process
        # grabs the GPU between check and CUDA.device!() context creation.
        # On catch, release the CUDA context to avoid holding zombie memory
        # reservations that deadlock other processes polling for free memory.
        deadline = time() + gpu_timeout
        while true
            device_idx, available, _ = _poll_gpu(
                backend, required_bytes;
                timeout = max(0.0, deadline - time()), on_wait = on_wait
            )
            if !available
                error("No GPU with sufficient memory after $(gpu_timeout)s")
            end
            try
                CUDA.device!(device_idx)
                return GPU()
            catch e
                # Release context to free reserved memory before retrying
                _release_gpu_context(device_idx)
                if time() >= deadline
                    rethrow()
                end
                @warn "GPU $device_idx context creation failed (contention race), \
                    retrying" exception = e
            end
        end

    else  # :auto
        if !CUDA.functional()
            return CPU()
        end
        # Same retry pattern as :gpu but falls back to CPU on timeout
        # instead of erroring. On TOCTOU catch, release context and re-poll
        # with remaining timeout - GPU contention spikes are typically short.
        deadline = time() + auto_timeout
        while true
            device_idx, available, failed = _poll_gpu(
                backend, required_bytes;
                timeout = max(0.0, deadline - time()), on_wait = on_wait
            )
            failed && return CPU()
            if !available
                @warn "No GPU available after $(auto_timeout)s, using CPU"
                return CPU()
            end
            try
                CUDA.device!(device_idx)
                return GPU()
            catch e
                _release_gpu_context(device_idx)
                if time() >= deadline
                    @warn "GPU context creation failed and timeout reached, using \
                        CPU" exception = e
                    return CPU()
                end
                @warn "GPU $device_idx context creation failed (contention race), \
                    retrying" exception = e
            end
        end
    end
end

# Default on_wait callback for user feedback
const DEFAULT_ON_WAIT = (elapsed, available, required) ->
@info "Waiting for GPU memory..." elapsed = round(elapsed, digits = 1) available =
    Base.format_bytes(available) required = Base.format_bytes(required)
