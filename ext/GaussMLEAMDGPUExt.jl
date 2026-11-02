"""
AMD GPU support for GaussMLE through AMDGPU and ROCm. Loaded when AMDGPU is loaded.
"""
module GaussMLEAMDGPUExt

using GaussMLE: GaussMLE, ROCmGPU
using AMDGPU: AMDGPU, ROCBackend

GaussMLE._functional(::ROCmGPU) = AMDGPU.functional()
GaussMLE._ka_gpu_backend(::ROCmGPU) = ROCBackend()
GaussMLE._activate_gpu!(::ROCmGPU, i) = AMDGPU.device!(AMDGPU.devices()[i + 1])
GaussMLE._reclaim_gpu(::ROCmGPU) = AMDGPU.HIP.reclaim()

# Zero-based ordinal of the active device, as `_activate_gpu!` takes it (AMDGPU 2.10).
GaussMLE._active_device_id(::ROCmGPU) = Int(AMDGPU.HIP.device_id(AMDGPU.device()))

function GaussMLE._is_oom_error(::ROCmGPU, e)
    return e isa AMDGPU.HIP.HIPError && e.code == AMDGPU.HIP.hipErrorOutOfMemory
end

GaussMLE._is_resource_error(v::ROCmGPU, e) = GaussMLE._is_oom_error(v, e)

# Total bytes of each device (AMDGPU 2.10: `info()` returns (free, total) for the active
# device, so the scoped switch restores the previous one).
function GaussMLE._gpu_total_memory(::ROCmGPU)
    return [Int(AMDGPU.device!(AMDGPU.info, dev)[2]) for dev in AMDGPU.devices()]
end

function GaussMLE._release_gpu_context(::ROCmGPU, i)
    # Finalize the failed attempt's arrays and return pooled memory to HIP free memory.
    GC.gc(true)
    try
        AMDGPU.device!(AMDGPU.devices()[i + 1]) do
            AMDGPU.synchronize()
            AMDGPU.HIP.reclaim()
        end
    catch e
        # Best effort: HIP device activation or cleanup may fail.
        e isa AMDGPU.HIP.HIPError || rethrow()
    end
    return nothing
end

function GaussMLE._poll_gpu(
        ::ROCmGPU, backend::Symbol, required_bytes::Integer; timeout, on_wait = nothing,
    )
    try
        devices = AMDGPU.devices()
        required_with_margin = required_bytes * GaussMLE.GPU_MEMORY_MARGIN
        start = time()
        deadline = start + timeout
        poll = 0.5

        while true
            best_free = 0
            never_fits = true
            for (i, dev) in enumerate(devices)
                # AMDGPU 2.10 exposes HIP free memory as info(), outside Mem.
                # The scoped device switch restores the previous context on errors too.
                free, total = AMDGPU.device!(AMDGPU.info, dev)
                best_free = max(best_free, free)

                # No contention check: ROCm under WSL has no SMI process/utilization data.
                if free > required_with_margin
                    return i - 1, true, false, nothing
                end
                # No process list either, so only a total below the request is final.
                never_fits &= GaussMLE._never_fits(
                    free, total, required_with_margin, nothing
                )
            end

            never_fits && return (-1, false, false, best_free)
            time() >= deadline && return (-1, false, false, nothing)
            if on_wait !== nothing
                on_wait(time() - start, best_free, required_bytes)
            end

            # Jittered backoff to avoid thundering herd.
            sleep(poll * (1.0 + 0.2 * rand()))
        end
    catch e
        (backend == :gpu || e isa InterruptException) && rethrow()
        @warn "GPU availability check failed, using CPU" exception = (e, catch_backtrace())
        return -1, false, true, nothing
    end
end

end # module
