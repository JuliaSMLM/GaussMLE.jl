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

# Whether `e` is a device out-of-memory error: a smaller batch can clear it at once.
_is_oom_error(::GPUVendor, e) = false
function _is_oom_error(::CUDAGPU, e)
    e isa CUDA.OutOfGPUMemoryError && return true
    return e isa CUDA.CuError && e.code == CUDA.ERROR_OUT_OF_MEMORY
end

# Whether `e` is a resource error that waiting can clear (out of memory, a context that
# cannot be created); the retry loop rethrows or falls back to the CPU for any other.
function _is_resource_error(v::CUDAGPU, e)
    _is_oom_error(v, e) && return true
    return e isa CUDA.CuError && e.code == CUDA.ERROR_DEVICE_UNAVAILABLE
end

# A vendor without a total-memory query skips the one-ROI check.
_gpu_total_memory(::GPUVendor) = Int[]

# Safety margin on a batch's estimated bytes, for polling and for the total-memory check.
const GPU_MEMORY_MARGIN = 1.5

"""
    _exceeds_device_memory(required_bytes, totals) -> Bool

Whether `required_bytes`, with the polling margin, exceeds the total memory of every
device in `totals`. Such a batch cannot fit however long the fit waits. An empty `totals`
(a failed query) never exceeds.
"""
function _exceeds_device_memory(required_bytes::Real, totals)
    return !isempty(totals) && all(t -> required_bytes * GPU_MEMORY_MARGIN > t, totals)
end

# Total bytes of each visible device, through NVML without a context; devices whose NVML
# query is unsupported (DGX Spark GB10) are asked through CUDA instead.
function _gpu_total_memory(::CUDAGPU)
    return map(0:(length(CUDA.devices()) - 1)) do i
        try
            Int(CUDA.NVML.memory_info(_nvml_device(i)).total)
        catch e
            e isa CUDA.NVML.NVMLError && e.code == CUDA.NVML.NVML_ERROR_NOT_SUPPORTED ||
                rethrow()
            Int(CUDA.device!(CUDA.total_memory, i))
        end
    end
end

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

# CUDA 6 keeps the driver API wrappers in CUDACore; CUDA 5 defines them in CUDA itself.
_cuda_driver_module() = isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA

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
        _cuda_driver_module().cuDevicePrimaryCtxRelease(dev)
    catch e
        # Best-effort: context may not exist if creation failed early
        e isa CUDA.CuError || rethrow()
    end
    # NOTE: Do NOT call CUDA.reclaim() here - it requires an active context
    # and will either hang or re-create the context we just released.
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
    required_with_margin = required_bytes * GPU_MEMORY_MARGIN
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
