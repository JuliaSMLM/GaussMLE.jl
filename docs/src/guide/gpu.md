# GPU Support

GaussMLE.jl accelerates fitting on CUDA or AMD GPUs through KernelAbstractions.jl.
AMD support requires loading AMDGPU. Memory polling and retries support shared GPU use;
CUDA also checks contention through NVML when available.

## Quick Start

```julia
using GaussMLE

# Auto-detect backend (default: uses GPU if available, falls back to CPU)
fitter = GaussMLEConfig()

# Force CPU
fitter = GaussMLEConfig(backend = :cpu)

# Force GPU (errors immediately if no functional GPU is found)
fitter = GaussMLEConfig(backend = :gpu)

# Auto with custom timeout
fitter = GaussMLEConfig(backend = :auto, auto_timeout = 300.0)
```

## Backend Selection

The `backend` parameter controls compute device selection:

| Backend | Behavior | On timeout |
|---------|----------|------------|
| `:auto` (default) | Try GPU, fall back to CPU | Warning + CPU fallback |
| `:cpu` | Always CPU, no GPU interaction | N/A |
| `:gpu` | Require GPU | Error |

GPU modes prefer CUDA whenever `CUDA.functional()`. Otherwise they use a functional
AMD GPU when AMDGPU is loaded.

If neither vendor is functional, `:gpu` errors immediately and `:auto` uses CPU
without a warning. Timeouts govern resource waits and retries only after a functional
vendor is found; a timeout in `:auto` warns before falling back to CPU.

### Timeout Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `auto_timeout` | `300.0` | Seconds for resource waits and retries in `:auto` mode |
| `gpu_timeout` | `Inf` | Seconds for resource waits and retries in `:gpu` mode |
| `on_wait` | `nothing` | Progress callback `(elapsed, available, required) -> nothing` |

```julia
# Auto mode: with a functional GPU, wait/retry up to 5 min, then fall back to CPU
fitter = GaussMLEConfig(backend = :auto, auto_timeout = 300.0)

# Explicit GPU: with a functional GPU, wait/retry up to 60s, then error
fitter = GaussMLEConfig(backend = :gpu, gpu_timeout = 60.0)

# Progress callback for monitoring waits
fitter = GaussMLEConfig(
    backend = :auto,
    auto_timeout = 60.0,
    on_wait = (elapsed, available, required) ->
        @info "Waiting for GPU" elapsed=round(elapsed, digits=1) available=Base.format_bytes(available)
)
```

## AMD GPUs (ROCm)

CUDA support comes with GaussMLE. AMD support is a package extension: add AMDGPU to
your environment once and load it before `fit`:

```julia
using Pkg; Pkg.add("AMDGPU")  # once per environment

using GaussMLE
using AMDGPU

smld, info = fit(batch, GaussMLEConfig(backend = :gpu))
```

Both `:gpu` and `:auto` support AMD. CUDA wins when both vendors are functional.
`info.backend` is `:gpu` on either vendor, and `info.device_id` is a 0-based index.
Tested on a Radeon 780M under WSL2 with ROCm 10.1 and AMDGPU 2.10.

Memory waits query HIP free memory on each visible AMD device. There is no NVML-style
contention check on AMD: ROCm under WSL has no SMI process/utilization data. On success,
the HIP pool's cached memory is reclaimed. Failure cleanup performs full garbage
collection, then attempts synchronization and HIP pool reclamation on the failed
device, restoring the previous device afterward. HIP cleanup errors are ignored;
other errors propagate. No CUDA-style primary context is released.

## GPU Scheduling System

### The Problem

When multiple Julia processes share a GPU server, device selection can cause failures.
CUDA examples include:

- **Context creation OOM**: `CUDA.device!(i)` creates a CUDA context (~300 MB - 6 GB), which can OOM if another process is using the GPU
- **Memory pool retention**: CUDA.jl's memory pool holds freed arrays, making NVML report memory as "used" even when no GPU arrays exist
- **TOCTOU races**: Free memory looks sufficient during the check, but another process grabs it before context creation completes

### Solution: Unified Retry Loop

GaussMLE uses a single retry loop in `_run_mle_kernel!` that handles all GPU failure modes uniformly:

```
while !gpu_succeeded && before deadline:
    1. Poll vendor memory (CUDA: NVML with CUDA fallback; AMD: HIP free memory);
       skip a GPU with free memory < 1.5x required
    2. If memory came from NVML, skip if other_procs > 0 AND compute > 90%
    3. Try:
         activate selected GPU     # Use the selected vendor
         <process all batches>     # Run fitting kernel
         reclaim vendor pool       # Return cached memory
         gpu_succeeded = true
    4. Catch:
         clean up GPU resources    # CUDA: release context; AMD: GC, synchronize, reclaim
         loop back to step 1       # Re-poll memory for next available GPU
```

After the loop: `:auto` falls back to CPU, `:gpu` errors.

### Three Failure Modes

The loop handles three distinct GPU failure scenarios:

1. **No free memory** - Memory polling waits with jittered backoff (0.5s intervals). All GPUs are scanned each tick; whichever frees up first is used.

2. **TOCTOU context creation race** - Memory looks sufficient, but activating the GPU
   fails because another process acquired it. Resources are cleaned up and memory is
   re-polled.

3. **Runtime OOM** - GPU allocation or kernel execution fails mid-batch. Resources are
   cleaned up and all batches restart on an available GPU.

### NVML Contention Detection

On CUDA, memory polling prefers NVIDIA Management Library (NVML) queries without
creating CUDA contexts. If NVML reports `NVML_ERROR_NOT_SUPPORTED`, polling uses
`CUDA.free_memory()` on that device, which may create a context. Other errors from
NVML memory queries or NVML device lookups propagate for `:gpu`; `:auto` warns and
falls back to CPU. A failed process or utilization query counts as not contended.

Contention checks apply only when memory came from NVML. A GPU with free memory below 1.5x required is skipped before the contention check. A GPU is considered **contended** when:

- Other processes are present (`compute_processes > 0` excluding own PID), AND
- Compute utilization > 90%

If other processes are present but memory is sufficient and compute utilization is at most 90%, the GPU is still selected.

### CUDA Context Lifecycle

**On failure:** `_release_gpu_context()` makes a best effort to release this process's reference to the device's primary context by calling `cuDevicePrimaryCtxRelease`. The context and its reserved memory are freed when the last reference is released. `CUDA.reclaim()` is NOT called after context release because it requires an active context and would either hang or silently re-create the released context.

**On success:** `CUDA.reclaim()` is called after all GPU batches complete to return pooled memory to the OS. Without this, NVML reports the memory as used even though no GPU arrays exist, blocking other processes waiting for GPU availability.

### Multi-GPU Support

Memory polling scans all visible devices of the selected vendor each iteration. The first
GPU with sufficient free memory and, when NVML is used, low contention is selected.

CUDA polling sees only the GPUs exposed by `CUDA_VISIBLE_DEVICES` and reads each GPU's
NVML data by UUID, so a restricted process reads the right GPU.

## Batch Processing

For datasets larger than GPU memory, GaussMLE automatically batches the data:

```julia
# Configure batch size (default: 10,000 ROIs per batch)
fitter = GaussMLEConfig(
    backend = :gpu,
    batch_size = 5000  # Process 5000 ROIs at a time
)

# Large dataset - automatically batched
smld, info = fit(large_data, fitter)
println("Processed in $(info.n_batches) batches of $(info.batch_size)")
```

Tune batch size based on:
- Available GPU memory (larger batches = more memory)
- ROI size (larger ROIs need smaller batches)
- Number of parameters (more params = more memory per ROI)

## Performance

### Benchmarking Your Hardware

Run `GROUP=GPU julia --project -e 'using Pkg; Pkg.test()'` locally to benchmark your hardware. The GPU test group includes a comprehensive performance benchmark that tests all model/camera/device combinations and reports fits/second along with MLE optimality (std/CRLB ratio).

GPU throughput scales with batch size and varies significantly by GPU model. Typical speedups range from 100-200x over single-threaded CPU for large batches.

### When to Use GPU

**GPU is beneficial for:**
- Large datasets (>10,000 fits)
- Batch processing of multiple files
- Real-time analysis requirements

**CPU may be better for:**
- Small datasets (<1,000 fits)
- Systems without capable GPUs
- Debugging and development

### Performance Tips

1. **Use Float32** - Native GPU type, sufficient precision for localization
2. **Fixed sigma** - `GaussianXYNB` is ~20% faster than variable-sigma models
3. **Warm-up** - First GPU call includes kernel compilation overhead
4. **Appropriate batch size** - Too small has transfer overhead, too large may OOM

## Result Consistency

GPU and CPU results are numerically consistent (differences at machine precision ~1e-6 microns). The same unified kernel runs on both devices via KernelAbstractions.jl.

## Troubleshooting

### CUDA Not Available

```julia
using CUDA
CUDA.functional()  # Should return true
```

If false: check NVIDIA driver installation, GPU hardware, CUDA.jl configuration.

### AMD Not Available

```julia
using AMDGPU
AMDGPU.functional()  # Should return true
```

Load AMDGPU before `fit` and check the ROCm installation if this returns false.

### Out of Memory

```julia
# Reduce batch size
fitter = GaussMLEConfig(backend = :gpu, batch_size = 2000)

# Or use auto mode for automatic CPU fallback
fitter = GaussMLEConfig(backend = :auto, auto_timeout = 5.0)
```

### Multi-Process Contention

If parallel scripts deadlock waiting for GPU:
- Ensure all scripts use `backend = :auto` (not `:gpu`) for graceful fallback
- Set reasonable `auto_timeout` values (default 5 min is usually sufficient)
- On CUDA, check `nvidia-smi` for zombie processes holding GPU memory

## Architecture

### Unified Kernel Design

GaussMLE uses KernelAbstractions.jl for portable GPU/CPU code:

```julia
@kernel function unified_gaussian_mle_kernel!(...)
    # Same code runs on CPU and GPU
end
```

Since KernelAbstractions 0.10, the CPU backend compiles kernels through POCL, an OpenCL
runtime. The first CPU fit of each PSF model in a session takes a few seconds longer
for compilation; subsequent fits reuse the compiled kernel.
The CPU path uses work-groups of 64 fits.

### Data Flow

1. **Input**: ROI data on CPU (Float32)
2. **Transfer**: Copy batch to GPU memory
3. **Compute**: Run fitting kernel
4. **Transfer**: Copy results back to CPU
5. **Repeat**: Next batch (steps 2-4)
6. **Reclaim**: Return GPU memory pool to OS after all batches succeed
7. **Output**: BasicSMLD with fitted parameters
