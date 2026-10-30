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

function GaussMLE._active_device_id(::ROCmGPU)
    return Int(findfirst(==(AMDGPU.device()), AMDGPU.devices()) - 1)
end

function GaussMLE._release_gpu_context(::ROCmGPU, i)
    try
        # HIP needs no CUDA-style primary-context release; finalize unreachable arrays.
        GC.gc(false)
    catch
        # Best effort, including when device activation failed.
    end
    return nothing
end

function GaussMLE._poll_gpu(
        ::ROCmGPU, backend::Symbol, required_bytes::Integer; timeout, on_wait = nothing,
    )
    try
        devices = AMDGPU.devices()
        required_with_margin = required_bytes * 1.5
        start = time()
        deadline = start + timeout
        poll = 0.5

        while true
            best_free = 0
            for (i, dev) in enumerate(devices)
                # AMDGPU 2.10 exposes HIP free memory as info(), outside Mem.
                # The scoped device switch restores the previous context on errors too.
                free, _ = AMDGPU.device!(AMDGPU.info, dev)
                best_free = max(best_free, free)

                # No contention check: ROCm under WSL has no SMI process/utilization data.
                if free > required_with_margin
                    return i - 1, true, false
                end
            end

            time() >= deadline && return (-1, false, false)
            if on_wait !== nothing
                on_wait(time() - start, best_free, required_bytes)
            end

            # Jittered backoff to avoid thundering herd.
            sleep(poll * (1.0 + 0.2 * rand()))
        end
    catch e
        (backend == :gpu || e isa InterruptException) && rethrow()
        @warn "GPU availability check failed, using CPU" exception = (e, catch_backtrace())
        return -1, false, true
    end
end

end # module
