# GPU Architecture Design for GaussMLE.jl

## Overview

This document outlines the GPU acceleration architecture for GaussMLE.jl, supporting CUDA, Metal, and optimized CPU backends.

## Design Principles

1. **Backend Abstraction**: Clean separation between fitting logic and hardware-specific implementations
2. **Batching**: Automatic batching to handle datasets larger than GPU memory (10^7 ROIs)
3. **Zero-Copy Interface**: Minimize data movement between CPU and GPU
4. **Performance**: Achieve 20-100x speedup over single-threaded CPU
5. **Extensibility**: Easy to add new backends without modifying core logic

## Architecture Components

### 1. Backend Interface

```julia
abstract type FittingBackend end

struct CPUBackend <: FittingBackend
    nthreads::Int
end

struct CUDABackend <: FittingBackend
    device::CuDevice
    compute_capability::VersionNumber
end

struct MetalBackend <: FittingBackend
    device::MTLDevice
end
```

### 2. Batch Processing System

```julia
struct BatchConfig
    max_batch_size::Int      # Maximum ROIs per batch
    n_streams::Int           # Number of concurrent streams
    pinned_memory::Bool      # Use pinned memory for transfers
    overlap_compute::Bool    # Overlap computation with transfers
end

struct BatchIterator{T}
    data::AbstractArray{T,3}
    batch_size::Int
    config::BatchConfig
end
```

### 3. GPU Kernel Structure

Each backend implements:
- `allocate_buffers`: Pre-allocate GPU memory
- `transfer_batch`: Move data to device
- `launch_kernel!`: Execute fitting kernel
- `transfer_results`: Move results back

### 4. Memory Layout

Optimized for coalesced access:
- Input: `(pixels_per_roi, n_rois_in_batch)` - transposed for better access
- Parameters: `(n_params, n_rois_in_batch)`
- Shared memory: Per-block workspace for Newton-Raphson

## Implementation Phases

### Phase 1: Infrastructure (This PR)
- Backend abstraction layer
- Batch processing system
- Enhanced simulators for GPU testing
- Comprehensive test suite

### Phase 2: CUDA Implementation
- Native CUDA kernels
- Optimized for different ROI sizes
- Stream-based overlap

### Phase 3: Metal Implementation
- Metal Performance Shaders
- macOS/iOS support

### Phase 4: Optimization
- Auto-tuning for different GPUs
- Mixed precision support
- Advanced scheduling

## Performance Targets

| Backend | Expected Speedup | Target Throughput |
|---------|-----------------|-------------------|
| CUDA    | 50-100x         | 1M ROIs/sec      |
| Metal   | 20-50x          | 500K ROIs/sec    |
| CPU     | 5-10x           | 50K ROIs/sec     |

## Testing Strategy

1. **Unit Tests**: Each backend component
2. **Integration Tests**: Full pipeline with each backend
3. **Performance Tests**: Benchmark against targets
4. **Accuracy Tests**: Ensure numerical equivalence
5. **Stress Tests**: Large datasets, edge cases