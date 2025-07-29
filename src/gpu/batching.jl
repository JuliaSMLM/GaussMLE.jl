"""
Batching system for processing large datasets
"""

# Configuration for batch processing
struct BatchConfig
    max_batch_size::Int
    n_streams::Int
    pinned_memory::Bool
    overlap_compute::Bool
    prefetch_batches::Int
    
    function BatchConfig(; 
        max_batch_size::Int=100_000,
        n_streams::Int=4,
        pinned_memory::Bool=true,
        overlap_compute::Bool=true,
        prefetch_batches::Int=2)
        
        new(max_batch_size, n_streams, pinned_memory, overlap_compute, prefetch_batches)
    end
end

# Get optimal batch configuration for a backend
function optimal_batch_config(backend::FittingBackend, data_size::Tuple)
    n_rois = data_size[3]
    max_batch = min(max_batch_size(backend), n_rois)
    
    # Adjust batch size to be divisible by warp size (32) for GPU efficiency
    if backend isa Union{CUDABackend, MetalBackend}
        max_batch = (max_batch ÷ 32) * 32
        max_batch = max(max_batch, 32)  # At least one warp
    end
    
    # Determine number of streams based on dataset size
    n_batches = cld(n_rois, max_batch)
    n_streams = if backend isa CPUBackend
        1  # No benefit from multiple streams on CPU
    else
        min(4, n_batches)  # Up to 4 streams, but not more than batches
    end
    
    return BatchConfig(
        max_batch_size = max_batch,
        n_streams = n_streams,
        pinned_memory = backend isa CUDABackend,
        overlap_compute = backend isa Union{CUDABackend, MetalBackend},
        prefetch_batches = backend isa CPUBackend ? 0 : 2
    )
end

# Iterator for processing data in batches
struct BatchIterator{T,N}
    data::AbstractArray{T,N}
    batch_size::Int
    n_batches::Int
    remainder::Int
    
    function BatchIterator(data::AbstractArray{T,3}, batch_size::Int) where T
        n_rois = size(data, 3)
        n_batches = cld(n_rois, batch_size)
        remainder = n_rois % batch_size
        new{T,3}(data, batch_size, n_batches, remainder)
    end
end

Base.length(iter::BatchIterator) = iter.n_batches

function Base.iterate(iter::BatchIterator, state=1)
    if state > iter.n_batches
        return nothing
    end
    
    # Calculate batch range
    start_idx = (state - 1) * iter.batch_size + 1
    end_idx = state == iter.n_batches && iter.remainder > 0 ? 
              start_idx + iter.remainder - 1 : 
              start_idx + iter.batch_size - 1
    
    # Extract batch
    batch = view(iter.data, :, :, start_idx:end_idx)
    
    return (batch, start_idx, end_idx), state + 1
end

# Result accumulator for batched processing
mutable struct BatchResults{T}
    parameters::Vector{Any}  # Will hold parameter structs
    uncertainties::Vector{Any}  # Will hold uncertainty structs
    current_index::Int
    total_size::Int
    
    function BatchResults{T}(total_size::Int) where T
        new(Vector{Any}(undef, total_size), 
            Vector{Any}(undef, total_size),
            1, total_size)
    end
end

function add_batch_results!(results::BatchResults, batch_params, batch_uncertainties, 
                          start_idx::Int)
    n_results = length(batch_params)
    end_idx = start_idx + n_results - 1
    
    results.parameters[start_idx:end_idx] = batch_params
    results.uncertainties[start_idx:end_idx] = batch_uncertainties
    results.current_index = end_idx + 1
end

# Pinned memory allocator for CUDA
struct PinnedAllocator{T}
    buffers::Vector{CuArray{T}}
    host_buffers::Vector{Array{T}}
    free_indices::Vector{Int}
    buffer_size::Tuple{Int,Int,Int}
    
    function PinnedAllocator{T}(buffer_size::Tuple{Int,Int,Int}, n_buffers::Int) where T
        buffers = [CUDA.zeros(T, buffer_size...) for _ in 1:n_buffers]
        host_buffers = [CUDA.unsafe_wrap(Array, pointer(buf), size(buf); 
                                        own=false) for buf in buffers]
        free_indices = collect(1:n_buffers)
        new(buffers, host_buffers, free_indices, buffer_size)
    end
end

function allocate_buffer!(alloc::PinnedAllocator{T}) where T
    if isempty(alloc.free_indices)
        error("No free buffers available")
    end
    
    idx = popfirst!(alloc.free_indices)
    return alloc.buffers[idx], alloc.host_buffers[idx], idx
end

function free_buffer!(alloc::PinnedAllocator, idx::Int)
    push!(alloc.free_indices, idx)
end

# Stream pool for overlapped execution
struct StreamPool
    streams::Vector{CuStream}
    current::Ref{Int}
    
    function StreamPool(n_streams::Int)
        streams = [CuStream() for _ in 1:n_streams]
        new(streams, Ref(1))
    end
end

function next_stream!(pool::StreamPool)
    stream = pool.streams[pool.current[]]
    pool.current[] = mod1(pool.current[] + 1, length(pool.streams))
    return stream
end

function synchronize_all(pool::StreamPool)
    for stream in pool.streams
        CUDA.synchronize(stream)
    end
end