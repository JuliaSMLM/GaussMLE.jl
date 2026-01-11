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

    # Save current device to restore if needed
    original_device = CUDA.device()

    best_device = 0
    max_free = 0

    for i in 0:(n_devices - 1)
        CUDA.device!(i)
        free = CUDA.free_memory()
        if free > max_free
            max_free = free
            best_device = i
        end
    end

    # Switch to best device
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
    select_backend(backend::Symbol, required_bytes; kwargs...)

Select compute backend with GPU memory wait behavior.

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
- `:gpu` - Explicit GPU, wait up to gpu_timeout, error if unavailable
- `:auto` - Try GPU with auto_timeout, fall back to CPU with warning
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
        # Select best GPU (most free memory) for multi-GPU systems
        find_best_gpu()
        if !wait_for_gpu_memory(required_bytes; timeout=gpu_timeout, on_wait=on_wait)
            error("GPU memory not available after $(gpu_timeout)s")
        end
        return GPU()

    else  # :auto
        if !CUDA.functional()
            return CPU()
        end
        # Select best GPU (most free memory) for multi-GPU systems
        find_best_gpu()
        if wait_for_gpu_memory(required_bytes; timeout=auto_timeout, on_wait=on_wait)
            return GPU()
        else
            @warn "GPU memory not available after $(auto_timeout)s, using CPU"
            return CPU()
        end
    end
end

# Default on_wait callback for user feedback
const DEFAULT_ON_WAIT = (elapsed, available, required) ->
    @info "Waiting for GPU memory..." elapsed=round(elapsed, digits=1) available=Base.format_bytes(available) required=Base.format_bytes(required)