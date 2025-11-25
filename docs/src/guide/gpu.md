# GPU Support

GaussMLE.jl includes GPU acceleration support through CUDA.jl and KernelAbstractions.jl for processing large datasets efficiently. The GPU backend provides 20-100x speedup for large batches of fits.

## Current Status

- **CPU Backend**: Fully functional, uses unified kernel with KernelAbstractions
- **CUDA Backend**: Fully functional, native GPU kernel execution
- **Automatic Device Selection**: GPU used when available, CPU fallback

## Quick Start

```julia
using GaussMLE

# Auto-detect device (uses GPU if available)
fitter = GaussMLEFitter()

# Force CPU
fitter_cpu = GaussMLEFitter(device = :cpu)

# Force GPU (falls back to CPU if unavailable)
fitter_gpu = GaussMLEFitter(device = :gpu)
```

## Device Selection

### Automatic Detection

By default, `GaussMLEFitter()` automatically selects the best available device:

```julia
using GaussMLE

# Auto-detect (preferred)
fitter = GaussMLEFitter()

# Explicit auto-detect (same behavior)
fitter = GaussMLEFitter(device = :auto)
```

### Manual Device Selection

Use symbols for convenience:

```julia
# Force CPU
fitter = GaussMLEFitter(device = :cpu)

# Force GPU (warns and falls back to CPU if unavailable)
fitter = GaussMLEFitter(device = :gpu)
```

### Checking GPU Availability

```julia
using CUDA

# Check if CUDA is functional
println("CUDA available: $(CUDA.functional())")

# Check GPU device
if CUDA.functional()
    println("GPU: $(CUDA.name(CUDA.device()))")
    println("Memory: $(CUDA.totalmem(CUDA.device()) / 1e9) GB")
end
```

## Batch Processing

For datasets larger than GPU memory, GaussMLE.jl automatically batches the data:

```julia
using GaussMLE

# Configure batch size (default: 10,000 ROIs per batch)
fitter = GaussMLEFitter(
    device = :gpu,
    batch_size = 5000  # Process 5000 ROIs at a time
)

# Large dataset - automatically batched
large_data = rand(Float32, 11, 11, 100_000)
smld = fit(fitter, large_data)
```

The batch size should be tuned based on:
- Available GPU memory
- ROI size (larger ROIs need smaller batches)
- GPU memory bandwidth

## Performance Benchmarking

```julia
using GaussMLE
using Statistics

# Generate test data
n_rois = 10_000
data = rand(Float32, 11, 11, n_rois)

# CPU benchmark
fitter_cpu = GaussMLEFitter(device = :cpu)
t_cpu = @elapsed smld_cpu = fit(fitter_cpu, data)
rate_cpu = n_rois / t_cpu
println("CPU: $(round(rate_cpu)) ROIs/second")

# GPU benchmark
fitter_gpu = GaussMLEFitter(device = :gpu, batch_size = 5000)
t_gpu = @elapsed smld_gpu = fit(fitter_gpu, data)
rate_gpu = n_rois / t_gpu
println("GPU: $(round(rate_gpu)) ROIs/second")

# Speedup
if t_gpu < t_cpu
    println("Speedup: $(round(t_cpu / t_gpu, digits=1))x")
end
```

### Typical Performance

Performance on modern hardware (11x11 pixel ROIs, GaussianXYNB model):

| Device | Fits/Second | Notes |
|--------|-------------|-------|
| CPU (Ryzen 9 5950X) | ~100K | Multi-threaded |
| GPU (RTX 4090) | ~10M | Batch size 50K |
| GPU (RTX 3080) | ~5M | Batch size 30K |

## When to Use GPU

### GPU acceleration is beneficial for:
- Large datasets (>10,000 fits)
- Repeated processing of similar data
- Real-time analysis requirements
- Batch processing of multiple files

### CPU may be better for:
- Small datasets (<1,000 fits)
- Single-use analysis
- Systems without capable GPUs
- Debugging and development

## Memory Management

### Estimating Memory Requirements

```julia
# Memory per ROI (approximate)
roi_size = 11
n_params = 4  # GaussianXYNB

bytes_per_roi = roi_size^2 * 4  # Float32 data
bytes_per_result = n_params * 4 * 3  # params + uncertainties + temp
total_per_roi = bytes_per_roi + bytes_per_result

# For 50,000 ROIs
n_rois = 50_000
total_memory = n_rois * total_per_roi / 1e6
println("Estimated GPU memory: $(round(total_memory)) MB")
```

### Handling Memory Limits

If you encounter out-of-memory errors:

1. **Reduce batch size**:
```julia
fitter = GaussMLEFitter(device = :gpu, batch_size = 2000)
```

2. **Process in chunks**:
```julia
# Manual chunking for very large datasets
chunk_size = 10_000
results = []
for i in 1:chunk_size:size(data, 3)
    chunk_end = min(i + chunk_size - 1, size(data, 3))
    chunk = data[:, :, i:chunk_end]
    push!(results, fit(fitter, chunk))
end
```

## Result Consistency

GPU and CPU results are numerically consistent:

```julia
using GaussMLE
using Statistics

data = rand(Float32, 11, 11, 1000)

fitter_cpu = GaussMLEFitter(device = :cpu)
fitter_gpu = GaussMLEFitter(device = :gpu)

smld_cpu = fit(fitter_cpu, data)
smld_gpu = fit(fitter_gpu, data)

# Compare results
x_cpu = [e.x for e in smld_cpu.emitters]
x_gpu = [e.x for e in smld_gpu.emitters]

mean_diff = mean(abs.(x_cpu .- x_gpu))
println("Mean position difference: $(mean_diff) microns")
# Should be near machine precision (~1e-6 microns)
```

## Troubleshooting

### CUDA Not Available

```julia
using CUDA

if !CUDA.functional()
    println("CUDA not available")
    println("Possible causes:")
    println("  - No NVIDIA GPU")
    println("  - CUDA driver not installed")
    println("  - CUDA.jl not properly configured")
end
```

### Out of Memory

```julia
# Reduce batch size
fitter = GaussMLEFitter(device = :gpu, batch_size = 1000)

# Or fall back to CPU
fitter = GaussMLEFitter(device = :cpu)
```

### Slow GPU Performance

If GPU is slower than expected:

1. **Check batch size**: Too small batch sizes have overhead
2. **Check data type**: Use Float32, not Float64
3. **Warm-up**: First GPU call includes compilation

```julia
# Warm-up the GPU kernel
small_data = rand(Float32, 11, 11, 10)
_ = fit(fitter, small_data)

# Then benchmark with real data
@time smld = fit(fitter, data)
```

## Architecture Details

### Unified Kernel Design

GaussMLE.jl uses KernelAbstractions.jl for portable GPU/CPU code:

```julia
# Single kernel implementation works on both devices
@kernel function unified_gaussian_mle_kernel!(...)
    # Same code runs on CPU and GPU
end
```

This ensures:
- Consistent results across devices
- Easier maintenance
- Automatic backend selection

### Data Flow

1. **Input**: ROI data on CPU
2. **Transfer**: Copy to GPU memory (batched)
3. **Compute**: Run fitting kernel on GPU
4. **Transfer**: Copy results back to CPU
5. **Output**: BasicSMLD with fitted parameters

For large datasets, steps 2-4 are pipelined across batches.
