"""
Camera models that extend SMLMData.AbstractCamera
"""

using SMLMData

# sCMOS camera extending SMLMData's AbstractCamera
struct SCMOSCamera{T} <: SMLMData.AbstractCamera
    pixel_edges_x::Vector{T}       # From IdealCamera
    pixel_edges_y::Vector{T}       # From IdealCamera  
    readnoise_variance::Matrix{T}  # Pixel-wise readout noise variance (e²)
    
    function SCMOSCamera(pixel_edges_x::Vector{T}, pixel_edges_y::Vector{T}, 
                        readnoise_variance::Matrix{T}) where T
        # Check dimensions match
        n_pixels_x = length(pixel_edges_x) - 1
        n_pixels_y = length(pixel_edges_y) - 1
        @assert size(readnoise_variance) == (n_pixels_x, n_pixels_y) "Variance map size must match pixel grid"
        new{T}(pixel_edges_x, pixel_edges_y, readnoise_variance)
    end
end

# Convenience constructor matching IdealCamera style
function SCMOSCamera(nx::Int, ny::Int, pixel_size::T, readnoise_variance::Matrix{T}) where T
    pixel_edges_x = collect(range(zero(T), nx * pixel_size, length=nx+1))
    pixel_edges_y = collect(range(zero(T), ny * pixel_size, length=ny+1))
    SCMOSCamera(pixel_edges_x, pixel_edges_y, readnoise_variance)
end

# Constructor with uniform pixel size (scalar or tuple)
function SCMOSCamera(nx::Int, ny::Int, pixel_size::Union{T, NTuple{2,T}}, 
                    readnoise_variance::Matrix{T}) where T
    if pixel_size isa Number
        px_size = py_size = pixel_size
    else
        px_size, py_size = pixel_size
    end
    pixel_edges_x = collect(range(zero(T), nx * px_size, length=nx+1))
    pixel_edges_y = collect(range(zero(T), ny * py_size, length=ny+1))
    SCMOSCamera(pixel_edges_x, pixel_edges_y, readnoise_variance)
end

# Export the new camera type
export SCMOSCamera