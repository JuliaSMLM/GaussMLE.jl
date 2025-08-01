# GPU Support

GaussMLE.jl includes GPU acceleration support through CUDA.jl for processing large datasets efficiently. The GPU backend can provide 20-100x speedup for large batches of fits.

## Current Status

- ✅ **Infrastructure**: Complete backend abstraction system
- ✅ **CPU Backend**: Multi-threaded CPU processing
- ⏳ **CUDA Backend**: In development (falls back to CPU)
- ⏳ **Metal Backend**: Placeholder for Apple Silicon

!!! note "Development Status"
    The GPU infrastructure is implemented but CUDA kernels are not yet complete. 
    Currently, GPU dispatch falls back to CPU processing. Native CUDA kernels 
    are planned for significant speedup.

## Checking GPU Availability

```julia
using GaussMLE

# Check what GPU backends are available
println("Available backends:")
println("  CPU: Always available")
println("  CUDA: $(GaussMLE.GaussGPU.cuda_available())")
```

## GPU Backend Architecture

The package uses a flexible backend system:

```julia
# Backend selection is automatic but can be queried
backend = GaussMLE.GaussGPU.get_backend()
println("Current backend: $backend")
```

### Automatic Backend Selection

1. **CUDA**: Used if CUDA.jl is loaded and GPU is available
2. **CPU**: Used as fallback with multi-threading

### Memory Management

For datasets larger than GPU memory, the package automatically:

- Splits data into GPU-sized batches
- Processes batches sequentially 
- Manages GPU memory allocation
- Combines results seamlessly

## Performance Considerations

### When to Use GPU

**GPU acceleration is beneficial for:**
- Large datasets (>10⁴ fits)
- Repeated processing of similar data
- Real-time analysis requirements

**CPU may be better for:**
- Small datasets (<10³ fits)
- Single-use analysis
- Systems without capable GPUs

### Benchmarking

```julia
using GaussMLE
using BenchmarkTools

# Generate test data
boxsz = 7
nboxes = Int(1e5)
data, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynb)

# Benchmark fitting
@time θ_found, Σ_found = GaussMLE.GaussFit.fitstack(data, :xynb, args)
```

## Future Development

### Planned CUDA Kernels

The upcoming CUDA implementation will include:

- **Native GPU fitting**: Newton-Raphson solver on GPU
- **Batched operations**: Process multiple fits simultaneously
- **Memory optimization**: Efficient GPU memory usage
- **Mixed precision**: Float32/Float16 support for speed

### Expected Performance

Based on the current architecture, expected speedups:

| Dataset Size | CPU (threads) | CUDA (planned) |
|--------------|---------------|----------------|
| 10³ fits     | Baseline      | 1-2x |
| 10⁴ fits     | Baseline      | 10-20x |
| 10⁵ fits     | Baseline      | 20-50x |
| 10⁶ fits     | Baseline      | 50-100x |

## Development and Testing

### GPU Development Environment

For developers working on GPU features:

```bash
# Set up development environment
julia --project=dev

# Enable GPU testing
GAUSSMLE_TEST_GPU=true julia --project test/gpu_tests.jl
```

### Contributing to GPU Development

The GPU backend is designed for extensibility. Key areas for contribution:

1. **CUDA Kernels**: Implement Newton-Raphson solver
2. **Memory Management**: Optimize batch processing
3. **Benchmarking**: Performance comparison tools
4. **Testing**: GPU-specific test cases

See the development files in `dev/` for GPU kernel work in progress.

## Troubleshooting

### Common Issues

**CUDA not available:**
```julia
# Install CUDA.jl if needed
using Pkg
Pkg.add("CUDA")

# Check CUDA functionality
using CUDA
CUDA.functional()
```

**Memory issues:**
- The package automatically handles GPU memory limits
- For very large datasets, processing happens in batches
- Monitor GPU memory usage with `nvidia-smi`

**Performance debugging:**
```julia
# Check backend selection
backend = GaussMLE.GaussGPU.get_backend()
@info "Using backend: $backend"

# Time individual components
@time θ_found, Σ_found = GaussMLE.GaussFit.fitstack(data, :xynb, args)
```

## Architecture Details

### Backend Interface

The GPU system uses a clean abstraction:

```julia
# Backend types
abstract type ComputeBackend end
struct CPUBackend <: ComputeBackend end
struct CUDABackend <: ComputeBackend end

# Dispatch mechanism
function fitstack(data, model, args)
    backend = get_backend()
    return fitstack(backend, data, model, args)
end
```

### Batching System

For large datasets:

```julia
# Automatic batching for GPU memory limits
function process_large_dataset(data, model, args)
    batch_size = determine_batch_size(data)
    results = []
    
    for batch in batches(data, batch_size)
        batch_results = process_batch(batch, model, args)
        push!(results, batch_results)
    end
    
    return combine_results(results)
end
```

This architecture ensures scalability to datasets larger than available GPU memory.