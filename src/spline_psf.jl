"""
Spline PSF model: 3D fitting (x, y, z, N, bg) with a tabulated PSF, such as a learned PSF
from PSFLearning saved as a MicroscopePSFs `SplinePSF`. Moved from PSFLearning's GaussMLE
bridge (PSFLearning b2a2283, `src/interop/gaussmle_bridge.jl`). CPU only.
"""

"""
    SplinePSFModel{T, S} <: PSFModel{5, T}

A 3D PSF model backed by a tabulated PSF `S`: any callable `(x, y, z)` in microns relative
to the emitter, such as a `MicroscopePSFs.SplinePSF`. It runs on the CPU only (see
[`gpu_compatible`](@ref)).

# Parameters (in order)
1. x: x-position (fitted in ROI pixels, output in microns)
2. y: y-position (fitted in ROI pixels, output in microns)
3. z: axial position (microns throughout)
4. N: total photon count
5. bg: background per pixel

# Constructors
    SplinePSFModel(spline; pixel_size, z_range, lateral_range = 1.5)

`spline` is any callable `(x, y, z)` in microns. `z_range` is the valid axial range in
microns; the fit clamps z into it.

    SplinePSFModel(psf::MicroscopePSFs.SplinePSF; pixel_size,
                   z_range = (psf.z_min, psf.z_max), lateral_range = 1.5)

Available when MicroscopePSFs is loaded (package extension). `psf` must be 3D, and `z_range`
must lie within the spline's own `(z_min, z_max)` (the spline is zero outside it), else
`ArgumentError`.

`pixel_size` is the camera pixel size in microns; `fit` rescales the model to the camera's
pixel size, which must be square. The PSF is sampled at pixel centres and normalized so that
its sum over pixels at z = 0 is 1.

# Fields
- `spline_psf::S`: the tabulated PSF, evaluated as `spline_psf(x, y, z)` in microns
- `pixel_size::T`: camera pixel size in microns
- `z_range::Tuple{T, T}`: valid axial range in microns
- `lateral_range::T`: lateral extent in microns used for the normalization sum
- `px_scale::T`: normalization factor so that the pixel-summed PSF ≈ 1

# Example
```jldoctest
julia> astig(x, y, z) = exp(-x^2 / (2 * (0.13 + 0.1z)^2) - y^2 / (2 * (0.13 - 0.1z)^2));

julia> psf = SplinePSFModel(astig; pixel_size = 0.1, z_range = (-0.5, 0.5))
SplinePSFModel(pixel_size=0.1, z_range=(-0.5f0, 0.5f0))

julia> length(psf)
5
```
"""
struct SplinePSFModel{T, S} <: PSFModel{5, T}
    spline_psf::S
    pixel_size::T
    z_range::Tuple{T, T}
    lateral_range::T
    px_scale::T
end

function SplinePSFModel(spline; pixel_size::Real, z_range, lateral_range::Real = 1.5)
    z_range[1] < z_range[2] ||
        throw(ArgumentError("z_range $z_range must be increasing"))
    # A plain callable is valid on z_range itself
    limits = (z_min = Float64(z_range[1]), z_max = Float64(z_range[2]))
    return _spline_psf_model(spline, pixel_size, z_range, limits, lateral_range)
end

# Shared by the core constructor and the MicroscopePSFs extension. `limits` has the spline's
# own `z_min`/`z_max`.
function _spline_psf_model(spline, pixel_size::Real, z_range, limits, lateral_range::Real)
    T = Float32
    px = T(pixel_size)

    # Compute normalization at z=0: sum of PSF at pixel centers should ≈ 1
    px_scale = _compute_psf_scale(spline, px, T(lateral_range))

    # The model clamps z into z_range, so both Float32 bounds must lie inside the spline's
    # own range: clamp to it, then round inward (a bound rounded outward evaluates to 0).
    zlo, zhi = _inward_bounds(T, z_range, limits)
    return SplinePSFModel{T, typeof(spline)}(
        spline, px, (zlo, zhi),
        T(lateral_range), px_scale
    )
end

gpu_compatible(::SplinePSFModel) = false

# fit passes one pixel size (from pixel_edges_x) to to_pixel_units, so a non-square camera
# would scale y wrongly without an error
function _check_pixel_geometry(model::SplinePSFModel, camera)
    px_x = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    px_y = camera.pixel_edges_y[2] - camera.pixel_edges_y[1]
    isapprox(px_x, px_y; rtol = 1.0e-6) || throw(
        ArgumentError(
            "SplinePSFModel needs square pixels; the camera's pixels are " *
                "$(px_x) x $(px_y) µm"
        )
    )
    return nothing
end

# Float32 z bounds inside [spline.z_min, spline.z_max] (a spline is zero outside its range).
function _inward_bounds(::Type{T}, zr, spline) where {T}
    lo = max(Float64(zr[1]), Float64(spline.z_min))
    hi = min(Float64(zr[2]), Float64(spline.z_max))
    tlo, thi = T(lo), T(hi)
    Float64(tlo) < spline.z_min && (tlo = nextfloat(tlo))
    Float64(thi) > spline.z_max && (thi = prevfloat(thi))
    return tlo, thi
end

"""
    _compute_psf_scale(spline, pixel_size, lateral_range) -> Float32

Compute normalization factor so that summing the PSF over integer pixel positions ≈ 1.
"""
function _compute_psf_scale(spline, pixel_size::T, lateral_range::T) where {T}
    n_px = ceil(Int, lateral_range / pixel_size)
    norm_sum = zero(T)
    for iy in (-n_px):n_px, ix in (-n_px):n_px
        val = T(spline(ix * pixel_size, iy * pixel_size, zero(T)))
        norm_sum += max(zero(T), val)
    end
    return norm_sum > zero(T) ? one(T) / norm_sum : pixel_size^2
end

# --- PSFModel interface ---

@inline function evaluate_psf(
        model::SplinePSFModel{T}, i, j,
        θ::SVector{5, T}
    ) where {T}
    x, y, z, N, bg = θ
    dx = (j - x) * model.pixel_size
    dy = (i - y) * model.pixel_size
    z_c = clamp(z, model.z_range[1], model.z_range[2])
    psf_val = max(zero(T), T(model.spline_psf(dx, dy, z_c)))
    return bg + N * psf_val * model.px_scale
end

@inline function compute_pixel_derivatives(
        i, j, θ::SVector{5, T},
        model::SplinePSFModel{T}
    ) where {T}
    x, y, z, N, bg = θ
    px = model.pixel_size
    scale = model.px_scale
    dx = (j - x) * px
    dy = (i - y) * px
    z_c = clamp(z, model.z_range[1], model.z_range[2])

    # PSF value
    f0 = max(zero(T), T(model.spline_psf(dx, dy, z_c))) * scale

    # Central finite differences for spatial derivatives
    h_lat = T(0.005)   # 5 nm step for lateral
    h_z = T(0.005)   # 5 nm step for axial

    fxp = max(zero(T), T(model.spline_psf(dx + h_lat, dy, z_c))) * scale
    fxm = max(zero(T), T(model.spline_psf(dx - h_lat, dy, z_c))) * scale
    fyp = max(zero(T), T(model.spline_psf(dx, dy + h_lat, z_c))) * scale
    fym = max(zero(T), T(model.spline_psf(dx, dy - h_lat, z_c))) * scale

    z_p = clamp(z_c + h_z, model.z_range[1], model.z_range[2])
    z_m = clamp(z_c - h_z, model.z_range[1], model.z_range[2])
    fzp = max(zero(T), T(model.spline_psf(dx, dy, z_p))) * scale
    fzm = max(zero(T), T(model.spline_psf(dx, dy, z_m))) * scale

    # First derivatives of PSF w.r.t. spatial coords
    df_ddx = (fxp - fxm) / (2 * h_lat)
    df_ddy = (fyp - fym) / (2 * h_lat)
    dz_denom = z_p - z_m
    df_dz = dz_denom > zero(T) ? (fzp - fzm) / dz_denom : zero(T)

    # Second derivatives
    d2f_ddx2 = (fxp - 2 * f0 + fxm) / (h_lat^2)
    d2f_ddy2 = (fyp - 2 * f0 + fym) / (h_lat^2)
    d2f_dz2 = dz_denom > zero(T) ? (fzp - 2 * f0 + fzm) / ((dz_denom / 2)^2) : zero(T)

    # Chain rule: model = bg + N * f0
    # dx = (j-x)*px → ∂dx/∂x = -px, ∂dy/∂y = -px
    model_val = bg + N * f0

    dudt = @SVector [
        -N * px * df_ddx,    # ∂model/∂x
        -N * px * df_ddy,    # ∂model/∂y
        N * df_dz,           # ∂model/∂z
        f0,                  # ∂model/∂N
        one(T),               # ∂model/∂bg
    ]

    d2udt2 = @SVector [
        N * px^2 * d2f_ddx2, # ∂²model/∂x²
        N * px^2 * d2f_ddy2, # ∂²model/∂y²
        N * d2f_dz2,         # ∂²model/∂z²
        zero(T),             # ∂²model/∂N²
        zero(T),              # ∂²model/∂bg²
    ]

    return model_val, dudt, d2udt2
end

@inline function simple_initialize(
        roi, box_size::Int,
        model::SplinePSFModel{T}
    ) where {T}
    # Background from edge pixels
    edge_sum = zero(T)
    edge_count = 0
    @inbounds for j in 1:box_size, i in 1:box_size
        if i == 1 || i == box_size || j == 1 || j == box_size
            edge_sum += T(roi[i, j])
            edge_count += 1
        end
    end
    bg = edge_sum / T(edge_count)

    # Center of mass for x, y
    sum_x = zero(T)
    sum_y = zero(T)
    total = zero(T)
    @inbounds for j in 1:box_size, i in 1:box_size
        val = max(T(roi[i, j]) - bg, zero(T))
        sum_y += T(i) * val
        sum_x += T(j) * val
        total += val
    end
    total = max(total, one(T))
    x, y = sum_x / total, sum_y / total

    # Starting every fit at z = 0 sends ROIs far from focus (|z| 0.4-0.5 µm for the
    # astigmatic test spline) into a wrong minimum on the other side of focus, so start at
    # the best z of a coarse likelihood scan instead.
    z = _spline_z_scan(roi, box_size, model, x, y, total, bg)
    return MVector{5, T}(x, y, z, total, bg)
end

# Grid spacing (µm) of the z start scan. The basin of the true z is several tenths of a µm
# wide; a 0.2 µm grid already picks a neighbouring basin at |z| 0.3-0.4 µm.
const SPLINE_Z_SCAN_STEP = 0.1f0

# z on an even grid over z_range (spacing at most SPLINE_Z_SCAN_STEP) with the highest
# Poisson log-likelihood at the given x, y, N and bg.
@inline function _spline_z_scan(
        roi, box_size::Int, model::SplinePSFModel{T}, x, y, N, bg
    ) where {T}
    zlo, zhi = model.z_range
    n = max(2, ceil(Int, (zhi - zlo) / SPLINE_Z_SCAN_STEP) + 1)
    bg_scan = max(bg, T(0.01))  # the bg lower bound of default_constraints; keeps μ > 0
    best_z, best_ll = zero(T), typemin(T)
    for k in 0:(n - 1)
        z = zlo + (zhi - zlo) * T(k) / T(n - 1)
        θ = SVector{5, T}(x, y, z, N, bg_scan)
        ll = zero(T)
        @inbounds for j in 1:box_size, i in 1:box_size
            μ = evaluate_psf(model, i, j, θ)
            ll += T(roi[i, j]) * log(μ) - μ
        end
        if ll > best_ll
            best_z, best_ll = z, ll
        end
    end
    return best_z
end

function to_pixel_units(model::SplinePSFModel{T}, pixel_size::Real) where {T}
    new_px = T(pixel_size)
    new_scale = _compute_psf_scale(model.spline_psf, new_px, model.lateral_range)
    return SplinePSFModel{T, typeof(model.spline_psf)}(
        model.spline_psf, new_px,
        model.z_range, model.lateral_range,
        new_scale
    )
end

function default_constraints(model::SplinePSFModel, box_size)
    return ParameterConstraints{5}(
        SVector{5, Float32}(-2.0f0, -2.0f0, model.z_range[1], 1.0f0, 0.01f0),
        SVector{5, Float32}(box_size + 2, box_size + 2, model.z_range[2], Inf32, Inf32),
        SVector{5, Float32}(1.0f0, 1.0f0, 0.05f0, 100.0f0, 2.0f0)
    )
end

function initialize_parameters(
        roi::AbstractMatrix{T},
        ::SplinePSFModel
    ) where {T}
    box_size = size(roi, 1)
    bg = minimum(roi)
    signal = roi .- bg
    total = max(sum(signal), one(T))
    y = sum((1:box_size) .* sum(signal, dims = 2)[:]) / total
    x = sum((1:box_size) .* sum(signal, dims = 1)[:]) / total
    return SVector{5, Float32}(Float32(x), Float32(y), 0.0f0, Float32(total), Float32(bg))
end

Base.show(io::IO, m::SplinePSFModel) = print(
    io,
    "SplinePSFModel(pixel_size=$(m.pixel_size), z_range=$(m.z_range))"
)

# --- to_emitter: convert fit result → SMLMData emitter ---

function to_emitter(
        ::SplinePSFModel{T},
        result::LocalizationResult{T},
        idx::Int,
        camera::SMLMData.AbstractCamera;
        dataset::Int = 1,
        track_id::Int = 0,
        id::Int = idx
    ) where {T}
    pixel_size_x = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    pixel_size_y = camera.pixel_edges_y[2] - camera.pixel_edges_y[1]

    # Convert pixel → micron coordinates
    x_microns = camera.pixel_edges_x[1] + (result.x_camera[idx] - 0.5f0) * pixel_size_x
    y_microns = camera.pixel_edges_y[1] + (result.y_camera[idx] - 0.5f0) * pixel_size_y
    z_microns = result.parameters[3, idx]  # already in microns

    photons = result.parameters[4, idx]
    bg = result.parameters[5, idx]

    # Uncertainties
    σ_x = result.uncertainties[1, idx] * pixel_size_x
    σ_y = result.uncertainties[2, idx] * pixel_size_y
    σ_z = result.uncertainties[3, idx]
    σ_xy = result.covariances[1, idx] * pixel_size_x * pixel_size_y
    σ_xz = result.covariances[2, idx] * pixel_size_x
    σ_yz = result.covariances[3, idx] * pixel_size_y
    σ_photons = result.uncertainties[4, idx]
    σ_bg = result.uncertainties[5, idx]

    pvalue = result.pvalues[idx]

    return Emitter3DFitGaussMLE{T}(
        T(x_microns), T(y_microns), T(z_microns),
        photons, bg,
        T(σ_x), T(σ_y), T(σ_z), T(σ_xy), T(σ_xz), T(σ_yz),
        σ_photons, σ_bg,
        pvalue,
        Int(result.frame_indices[idx]),
        dataset, track_id, id
    )
end
