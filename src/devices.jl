"""
Device abstraction with automatic GPU detection
"""

# Device abstraction
abstract type ComputeDevice end
struct CPU <: ComputeDevice end
struct GPU <: ComputeDevice end

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

# Multi-GPU support
"""
    find_best_gpu()

Find the GPU with most free memory and switch to it.
Returns the device index (0-based) of the selected GPU.

For single-GPU systems, simply returns 0.
"""
function find_best_gpu()
    !CUDA.functional() && return 0

    n_devices = length(CUDA.devices())
    n_devices == 1 && return 0

    # Query memory via NVML (no CUDA context needed, safe under contention)
    # Avoids cuDevicePrimaryCtxRetain OOM when multiple processes compete
    best_device = 0
    max_free = 0

    for i in 0:(n_devices - 1)
        info = CUDA.NVML.memory_info(CUDA.NVML.Device(i))
        if info.free > max_free
            max_free = info.free
            best_device = i
        end
    end

    # Only create context on the winner
    CUDA.device!(best_device)

    if n_devices > 1
        @info "Selected GPU $best_device with $(Base.format_bytes(max_free)) free memory"
    end

    return best_device
end

# Allow explicit device selection with fallback (legacy API)
function select_device(device::Union{ComputeDevice, Nothing}=nothing)
    if isnothing(device)
        return auto_device()
    elseif device isa GPU && !CUDA.functional()
        @warn "GPU requested but not available, falling back to CPU"
        return CPU()
    else
        return device
    end
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
- `on_wait`: Optional callback `(elapsed, available, required) -> nothing` for progress feedback
"""
function wait_for_gpu_memory(required_bytes::Integer;
        timeout::Float64=30.0,
        poll::Float64=0.5,
        on_wait=nothing)

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

Wait for a GPU with sufficient free memory using NVML queries only (no CUDA context creation).
Scans ALL GPUs each iteration - first available wins.

Returns `(device_index, true)` if a GPU became available, `(-1, false)` if timeout reached.

Contention detection: a GPU is considered contended when other processes are present AND
either free memory is insufficient or compute utilization exceeds 90%.

# Arguments
- `required_bytes`: Minimum bytes needed (checks for 1.5x as safety margin)
- `timeout`: Maximum seconds to wait (default 30.0)
- `poll`: Polling interval in seconds (default 0.5)
- `on_wait`: Optional callback `(elapsed, available, required) -> nothing` for progress
"""
function wait_for_gpu_nvml(required_bytes::Integer;
        timeout::Float64=30.0,
        poll::Float64=0.5,
        on_wait=nothing)

    n_devices = length(CUDA.devices())
    my_pid = getpid()
    required_with_margin = required_bytes * 1.5
    deadline = time() + timeout
    start = time()

    while true
        # Scan all GPUs - first available wins
        for i in 0:(n_devices - 1)
            nvml_dev = CUDA.NVML.Device(i)
            mem = CUDA.NVML.memory_info(nvml_dev)

            # Check if sufficient memory (with fragmentation margin)
            mem.free < required_with_margin && continue

            # Check contention: other processes on this GPU
            procs = try
                CUDA.NVML.compute_processes(nvml_dev)
            catch
                # NVML process query can fail on some drivers; skip contention check
                Dict{UInt32,UInt64}()
            end
            other_procs = count(p -> p.first != my_pid, procs)

            if other_procs > 0
                # Other processes present - check if they're actually threatening
                util = try
                    CUDA.NVML.utilization_rates(nvml_dev)
                catch
                    (; compute=0, memory=0)
                end
                # Contended: other procs AND (low memory OR high compute)
                if mem.free < required_with_margin || util.compute > 90
                    continue
                end
            end

            # GPU i is available: sufficient memory, not contended (or sole user)
            return (i, true)
        end

        # No GPU available this tick - check timeout
        if time() >= deadline
            return (-1, false)
        end

        if on_wait !== nothing
            # Report best available memory across all GPUs
            best_free = maximum(CUDA.NVML.memory_info(CUDA.NVML.Device(i)).free for i in 0:(n_devices-1))
            on_wait(time() - start, best_free, required_bytes)
        end

        # Jittered backoff to avoid thundering herd
        sleep(poll * (1.0 + 0.2 * rand()))
    end
end

"""
    select_backend(backend::Symbol, required_bytes; kwargs...)

Select compute backend with NVML-based contention detection and GPU memory wait.

All pre-flight GPU checks use NVML queries (no CUDA context creation). A CUDA context
is only created on the selected GPU after confirming availability. For `:auto` mode,
a runtime try/catch in `_run_mle_kernel!` provides defense in depth.

# Arguments
- `backend`: `:cpu`, `:gpu`, or `:auto`
- `required_bytes`: Estimated GPU memory needed for operation

# Keyword Arguments
- `auto_timeout=30.0`: Seconds to wait for GPU in auto mode before falling back to CPU
- `gpu_timeout=Inf`: Seconds to wait for GPU in explicit gpu mode
- `on_wait=nothing`: Callback `(elapsed, available, required) -> nothing` for progress

# Returns
- `CPU()` or `GPU()` device instance

# Semantics
- `:cpu` - Always CPU, no waiting
- `:gpu` - Explicit GPU, NVML poll up to gpu_timeout, error if unavailable
- `:auto` - NVML poll up to auto_timeout, fall back to CPU with warning
"""
function select_backend(backend::Symbol, required_bytes::Integer;
        auto_timeout::Float64=30.0,
        gpu_timeout::Float64=Inf,
        on_wait=nothing)

    backend in (:cpu, :gpu, :auto) || error("backend must be :cpu, :gpu, or :auto")

    if backend == :cpu
        return CPU()

    elseif backend == :gpu
        if !CUDA.functional()
            error("GPU requested but CUDA not functional")
        end
        # NVML poll: scan all GPUs, wait for first available
        device_idx, available = wait_for_gpu_nvml(required_bytes;
            timeout=gpu_timeout, on_wait=on_wait)
        if !available
            error("No GPU with sufficient memory after $(gpu_timeout)s")
        end
        # Create CUDA context only on the confirmed winner
        CUDA.device!(device_idx)
        return GPU()

    else  # :auto
        if !CUDA.functional()
            return CPU()
        end
        # NVML poll: scan all GPUs, wait for first available
        device_idx, available = wait_for_gpu_nvml(required_bytes;
            timeout=auto_timeout, on_wait=on_wait)
        if available
            CUDA.device!(device_idx)
            return GPU()
        else
            @warn "No GPU available after $(auto_timeout)s, using CPU"
            return CPU()
        end
    end
end

# Default on_wait callback for user feedback
const DEFAULT_ON_WAIT = (elapsed, available, required) ->
    @info "Waiting for GPU memory..." elapsed=round(elapsed, digits=1) available=Base.format_bytes(available) required=Base.format_bytes(required)