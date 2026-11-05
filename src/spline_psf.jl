"""
Spline PSF model: 3D fitting (x, y, z, N, bg) with a tabulated PSF, such as a learned PSF
from PSFLearning saved as a MicroscopePSFs `SplinePSF`. Moved from PSFLearning's GaussMLE
bridge (PSFLearning b2a2283, `src/interop/gaussmle_bridge.jl`). CPU and GPU.
"""

"""
    SplinePSFModel{T, S, P} <: PSFModel{5, T}

A 3D PSF model backed by a point PSF `S`: any callable `(x, y, z)` in microns relative to
the emitter, such as a `MicroscopePSFs.SplinePSF`. GaussMLE integrates this point PSF over
each camera pixel on the host. CPU, CUDA and AMDGPU fitting use the same table
and kernel; the point callable never reaches a kernel.

# Parameters (in order)
1. x: x-position (fitted in ROI pixels, output in microns)
2. y: y-position (fitted in ROI pixels, output in microns)
3. z: axial position (microns throughout)
4. N: signal scale (photons for IdealCamera, electrons for sCMOS; conventions below)
5. bg: background per pixel (photons for IdealCamera, electrons for sCMOS)

# Constructors
    SplinePSFModel(spline; pixel_size, z_range, lateral_range = 1.5, starts = nothing)

`spline` is a point PSF callable `(x, y, z)` in microns. `z_range` is the valid axial range
in microns; it must be finite, increasing, and at least 0.01 µm wide after inward Float32
rounding. The fit clamps z into it. Custom z constraints are intersected with `z_range`;
an empty intersection throws `ArgumentError`.

`starts` is the number of Levenberg-Marquardt fits per ROI; the fit keeps the one that
reaches the highest likelihood. `starts = nothing` (the default) gives the scan start plus
one fixed z start per 0.4 µm over the central 80 % of `z_range`: 6 for ±1 µm, 5 for ±0.6 µm,
4 for ±0.5 µm. `starts = 1` fits from the scan start alone; `starts < 1` throws
`ArgumentError`.

    SplinePSFModel(psf::MicroscopePSFs.SplinePSF; pixel_size,
                   z_range = (psf.z_min, psf.z_max), lateral_range = 1.5,
                   starts = nothing)

Available when MicroscopePSFs is loaded (package extension). `psf` must be 3D, and `z_range`
must lie within the spline's own `(z_min, z_max)` (the spline is zero outside it), else
`ArgumentError`.

`pixel_size` is the camera pixel size in microns: a `Real` for square pixels or a tuple
`(x, y)` for rectangular pixels. `fit` rescales the model to both camera pixel sizes.
`lateral_range` must cover the half-diagonal of the ROI, `sqrt((box * pixel_size_x / 2)^2 +
(box * pixel_size_y / 2)^2)`: the table ends at `lateral_range` plus one pixel and is zero
beyond it, so `fit` throws `ArgumentError` for a larger ROI instead of fitting a truncated
PSF. Each component and `lateral_range` must be finite and positive before and after Float32
conversion. Construction builds a Float32 table of pixel averages using composite
two-point Gauss-Legendre quadrature on five sub-intervals per axis. Its lateral nodes have
spacing `pixel_size_x / 5` and `pixel_size_y / 5`, extending to at least
`±(lateral_range + pixel_size_axis)` on each axis; its axial nodes span `z_range` with
spacing at most 0.02 µm.
A natural cubic B-spline interpolates the table, returning zero outside its lateral range.
Fitting uses this table. `fit` rebuilds it on every call whose camera pixel size differs
from the model's (about a second for a PSFLearning-size table), so build the model at the
camera pixel size; the same pixel size reuses the model. A callable that is already
pixel-integrated would be integrated twice. Tabulate slow callables, such as a vector PSF,
as a `SplinePSF` first.

At fit time, Adapt moves the coefficient array to the selected backend once and drops
the point callable from the kernel model. All batches share that table; a retry after a
GPU resource error transfers it again. On the CPU, POCL (KernelAbstractions 0.10) uses the
host array through its device-array view.
The table occupies `4 * prod(size(model.pixel_psf.coefficients))` bytes, typically about
5 to 10 MiB for a PSFLearning export, in addition to the ROI and result arrays.

For a `MicroscopePSFs.AbstractPSF`, the point PSF must be a normalized photon density in
µm^-2, including when constructed from a stack. Pixel averages are multiplied by the pixel
area without renormalization, so N is total photons (or total electrons for sCMOS) at every
z. The pixel-integrated sum on the `lateral_range` patch at the reference plane must lie in
`[0.5, 1.05]`; otherwise construction throws `ArgumentError` indicating an unnormalized
density. The reference plane is nearest z = 0 within the valid PSF range (the spline's own
range for a `SplinePSF`).

A plain callable has arbitrary scale. Its pixel averages are normalized on the reference
plane nearest z = 0 within its valid range, so N is the signal within the `lateral_range`
patch on that plane. The scale is reused at every z, so the callable's energy must be the
same at every z. Pixel-size rescaling preserves the reference plane. A non-finite or
non-positive reference sum throws `ArgumentError`. Point samples are clipped at zero before
quadrature for both conventions.

One 4 × 4 × 4 coefficient gather gives the value and analytic first derivatives of the
cubic interpolant, using each axis's pixel size; the fit uses no second derivatives.
Non-positive values are clipped to zero with zero derivatives. z is differentiated at its
clamped value. The scan start is the centroid for x and y, the edge-pixel mean for the
background, the positive in-box sum above it for N, and the z of a Poisson likelihood scan
at those values (grid spacing at most 0.1 µm). A single Levenberg-Marquardt fit reaches the
maximum of the basin it starts in, and the scan start is often in the wrong basin for a
defocused ROI. So the fit also starts from `starts - 1` fixed z values with the scan
start's x, y, N and background, as SMAP's fitter does, and keeps the fit with the highest
likelihood (the earlier start on a tie). With one fixed start it sits at the center of
`z_range`; with `m >= 2` they are `center + 0.8 R (q / (m - 1) - 1/2)` for `q = 0 .. m - 1`,
`R` the width of `z_range`. Each start costs one more Levenberg-Marquardt fit. On 34,341
real cell ROIs, `starts = 1` left 8.7 % (in-situ PSF) and 7.1 % (bead PSF) of fits at a
lower maximum in another basin, against 0.29 % and 0.20 % with the default. Each fit is
Levenberg-Marquardt on the full Fisher matrix, climbing the Poisson likelihood
(IdealCamera) or the Poisson-shifted likelihood `(d + v) log(μ + v) - (μ + v)` with `v`
the read-noise variance (sCMOS). The
default constraints cap x and y steps at 1 pixel and z steps at 0.05 µm; photon and
background steps are not capped (photons ≥ 1, background ≥ 0.01). For sCMOS the
reported log-likelihood is the Gaussian approximation
`-(d - μ)²/(2(μ + v))`, not the fitted objective; the p-value is the Pearson statistic
`(d - μ)²/(μ + v)` as for every model. `iterations` counts
Levenberg-Marquardt pixel passes per start: the last pass only evaluates, so the fit
takes at most `iterations - 1` steps and `iterations = 1` returns the start. A parameter
on a bound whose gradient points out of the box is held there while the others take their
step. Default x/y bounds are `[-1,n+2]` ROI pixels for a box of side n; z bounds use
`z_range`.

# Fields
- `spline_psf::S`: the point PSF callable, retained for rebuilding at another pixel size
- `pixel_psf::P`: the internal table of pixel averages, evaluated in microns
- `pixel_size::Tuple{T, T}`: camera pixel sizes `(x, y)` in microns
- `z_range::Tuple{T, T}`: valid axial range in microns
- `lateral_range::T`: lateral extent in microns used for the reference patch; `fit`
  refuses a ROI whose half-diagonal exceeds it
- `px_scale::T`: pixel area for a photon density, or reference normalization for a callable
- `z_ref::T`: reference plane nearest zero in the inward-rounded callable range
  (microns), preserved on rescaling
- `starts::Int`: Levenberg-Marquardt starts per ROI (the scan start plus `starts - 1`
  fixed z starts)

# Example
```jldoctest
julia> astig(x, y, z) = exp(-x^2 / (2 * (0.13 + 0.1z)^2) - y^2 / (2 * (0.13 - 0.1z)^2));

julia> psf = SplinePSFModel(astig; pixel_size = 0.1, z_range = (-0.5, 0.5))
SplinePSFModel(pixel_size=(0.1f0, 0.1f0), z_range=(-0.5f0, 0.5f0), starts=4)

julia> length(psf)
5
```
"""
struct SplinePSFModel{T, S, P} <: PSFModel{5, T}
    spline_psf::S
    pixel_psf::P
    pixel_size::Tuple{T, T}
    z_range::Tuple{T, T}
    lateral_range::T
    px_scale::T
    z_ref::T
    starts::Int
end

function SplinePSFModel(
        spline; pixel_size::Union{Real, Tuple{Real, Real}}, z_range,
        lateral_range::Real = 1.5, starts::Union{Nothing, Integer} = nothing
    )
    # A plain callable is valid on z_range itself
    limits = (z_min = Float64(z_range[1]), z_max = Float64(z_range[2]))
    return _spline_psf_model(spline, pixel_size, z_range, limits, lateral_range, starts)
end

# Number of starts when `starts = nothing`: the scan start plus one fixed start per 0.4 µm
# over the central 80 % of a z range of width R (0.8 R / 0.4 = 2R intervals, hence
# ceil(2R) + 1 fixed starts; the 1e-3 keeps a width with 2R an integer, such as 1.0 or 2.0,
# from rounding up on a small excess). 6 for ±1 µm, 5 for ±0.6 µm, 4 for ±0.5 µm.
function _default_spline_starts(z_range)
    R = Float64(z_range[2]) - Float64(z_range[1])
    return 1 + ceil(Int, 2R - 1.0e-3) + 1
end

function _spline_starts(starts::Union{Nothing, Integer}, z_range)
    starts === nothing && return _default_spline_starts(z_range)
    starts >= 1 || throw(ArgumentError("starts must be at least 1, got $starts"))
    return Int(starts)
end

# z (µm) of fixed start q = 0 .. m - 1 of m: the center of z_range for m = 1, else m points
# evenly over the central 80 % of the range (SMAP's wrapper places its starts the same way).
@inline function _spline_start_z(model::SplinePSFModel{T}, q::Int, m::Int) where {T}
    zlo, zhi = model.z_range
    R = zhi - zlo
    center = (zlo + zhi) / 2
    m == 1 && return center
    return center + T(0.8) * R * (T(q) / T(m - 1) - T(0.5))
end

function _spline_pixel_size(::Type{T}, pixel_size::Union{Real, Tuple{Real, Real}}) where {T}
    sizes = pixel_size isa Real ? (pixel_size, pixel_size) : pixel_size
    all(p -> isfinite(p) && p > 0, sizes) ||
        throw(ArgumentError("each pixel_size component must be finite and positive"))
    px = T.(sizes)
    all(p -> isfinite(p) && p > zero(T), px) || throw(
        ArgumentError(
            "each pixel_size component must be finite and positive after Float32 conversion"
        )
    )
    return px
end

# Shared by the core constructor and the MicroscopePSFs extension. `limits` has the spline's
# own `z_min`/`z_max`.
function _spline_psf_model(
        spline, pixel_size, z_range, limits, lateral_range::Real,
        starts::Union{Nothing, Integer}
    )
    (isfinite(z_range[1]) && isfinite(z_range[2]) && z_range[1] < z_range[2]) ||
        throw(ArgumentError("z_range $z_range must be finite and increasing"))
    (isfinite(lateral_range) && lateral_range > 0) ||
        throw(ArgumentError("lateral_range must be finite and positive"))
    T = Float32
    px = _spline_pixel_size(T, pixel_size)
    lateral = T(lateral_range)
    (isfinite(lateral) && lateral > zero(T)) || throw(
        ArgumentError("lateral_range must be finite and positive after Float32 conversion")
    )

    # The model clamps z into z_range, so both Float32 bounds must lie inside the spline's
    # own range: clamp to it, then round inward (a bound rounded outward evaluates to 0).
    zlo, zhi = _inward_bounds(T, z_range, limits)
    (isfinite(zlo) && isfinite(zhi) && zlo < zhi) || throw(
        ArgumentError("z_range $z_range must increase after inward Float32 rounding")
    )
    h = T(0.005)
    (zlo + h <= zhi - h && zlo + h > zlo && zhi - h < zhi) || throw(
        ArgumentError("z_range $z_range must span at least 0.01 µm in Float32")
    )

    # Round the callable's full range inward and choose the plane nearest zero. Keep it
    # when fit rescales the model to another camera pixel size.
    ref_lo, ref_hi = _inward_bounds(T, (limits.z_min, limits.z_max), limits)
    z_ref = clamp(zero(T), ref_lo, ref_hi)
    px_scale = _compute_psf_scale(spline, px, lateral, z_ref)
    pixel_psf = _build_pixel_psf(spline, px, lateral, (zlo, zhi))
    return SplinePSFModel{T, typeof(spline), typeof(pixel_psf)}(
        spline, pixel_psf, px, (zlo, zhi),
        lateral, px_scale, z_ref, _spline_starts(starts, (zlo, zhi))
    )
end

_has_axial_position(::Type{<:SplinePSFModel}) = true

function Adapt.adapt_structure(to, model::SplinePSFModel{T}) where {T}
    table = Adapt.adapt(to, model.pixel_psf)
    return SplinePSFModel{T, Nothing, typeof(table)}(
        nothing, table, model.pixel_size, model.z_range, model.lateral_range,
        model.px_scale, model.z_ref, model.starts
    )
end

# Backend adaptation also strips the host-only callable on the CPU path.
_prepare_psf_model(backend, model::SplinePSFModel) = Adapt.adapt(backend, model)
_psf_table_bytes(model::SplinePSFModel{T}) where {T} =
    sizeof(T) * length(model.pixel_psf.coefficients)

_array_fit_pixel_size(model::SplinePSFModel) = model.pixel_size

# The table is zero beyond lateral_range plus one pixel, so a ROI whose half-diagonal
# exceeds lateral_range would be fitted with a truncated PSF.
function _validate_psf_box(model::SplinePSFModel, box_size::Integer)
    shown_x, shown_y = model.pixel_size
    px_x, px_y = Float64(shown_x), Float64(shown_y)
    half_diagonal = hypot(box_size * px_x / 2, box_size * px_y / 2)
    lateral = Float64(model.lateral_range)
    half_diagonal <= lateral && return nothing
    hd = round(half_diagonal; digits = 2)
    throw(
        ArgumentError(
            "ROI box of $box_size px ($(shown_x) x $(shown_y) um pixels) has a half-diagonal " *
                "of $(hd) um, beyond the SplinePSFModel lateral_range of $(model.lateral_range) um; " *
                "the model is zero beyond its lateral range plus one pixel, so the fit " *
                "would use a truncated PSF. Build the model with lateral_range >= " *
                "$(ceil(half_diagonal; digits = 2)) " *
                "(SplinePSFModel(...; lateral_range = ...)) or use a smaller box."
        )
    )
end

function _model_for_camera(model::SplinePSFModel, camera)
    px_x = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    px_y = camera.pixel_edges_y[2] - camera.pixel_edges_y[1]
    return to_pixel_units(model, px_x, px_y)
end

# Float32 z bounds inside [spline.z_min, spline.z_max] (a spline is zero outside its range).
function _inward_bounds(::Type{T}, zr, spline) where {T}
    lo = max(Float64(zr[1]), Float64(spline.z_min))
    hi = min(Float64(zr[2]), Float64(spline.z_max))
    tlo, thi = T(lo), T(hi)
    Float64(tlo) < lo && (tlo = nextfloat(tlo))
    Float64(thi) > hi && (thi = prevfloat(thi))
    return tlo, thi
end

# Cubic coefficients include one ghost node at each end of every axis. The host builds
# an Array; Adapt supplies the backend's array or device view without changing the math.
struct _PixelIntegratedPSF{T, A <: AbstractArray{T, 3}}
    coefficients::A
    lateral_step::Tuple{T, T}
    lateral_radius::Tuple{T, T}
    z_range::Tuple{T, T}
    z_step::T
end

function Adapt.adapt_structure(to, table::_PixelIntegratedPSF)
    return _PixelIntegratedPSF(
        Adapt.adapt(to, table.coefficients), table.lateral_step, table.lateral_radius,
        table.z_range, table.z_step
    )
end

@inline function _spline_cubic_weights(t::T) where {T}
    t2, t3 = t * t, t * t * t
    return (
        (one(T) - t)^3 / T(6),
        (T(4) - T(6) * t2 + T(3) * t3) / T(6),
        (one(T) + T(3) * t + T(3) * t2 - T(3) * t3) / T(6),
        t3 / T(6),
    )
end

@inline function _spline_cubic_derivatives(t::T) where {T}
    t2 = t * t
    return (
        -(one(T) - t)^2 / T(2),
        T(1.5) * t2 - T(2) * t,
        T(0.5) + t - T(1.5) * t2,
        t2 / T(2),
    )
end

@inline function _spline_interval(u::T, n::Int) where {T}
    # clamp leaves NaN unchanged, and a NaN truncated below is poison, so map it to the
    # first node here.
    node = clamp(isnan(u) ? zero(T) : u, zero(T), T(n - 1))
    # Nonnegative and bounded by an existing array: truncation is floor, without a
    # checked conversion's exception path in device code.
    i = min(unsafe_trunc(Int, node), n - 2)
    return i, node - T(i)
end

@inline function _spline_coordinates(table::_PixelIntegratedPSF{T}, dx, dy, z) where {T}
    c = table.coefficients
    ny, nx, nz = size(c, 1) - 2, size(c, 2) - 2, size(c, 3) - 2
    h_x, h_y = table.lateral_step
    ix, tx = _spline_interval(dx / h_x + T((nx - 1) ÷ 2), nx)
    iy, ty = _spline_interval(dy / h_y + T((ny - 1) ÷ 2), ny)
    z_c = clamp(z, table.z_range...)
    iz, tz = _spline_interval((z_c - table.z_range[1]) / table.z_step, nz)
    return ix, iy, iz, tx, ty, tz
end

# Value-only gather for evaluation and the initial z scan.
@inline function (table::_PixelIntegratedPSF{T})(dx::T, dy::T, z::T) where {T}
    radius_x, radius_y = table.lateral_radius
    (abs(dx) > radius_x || abs(dy) > radius_y) && return zero(T)
    ix, iy, iz, tx, ty, tz = _spline_coordinates(table, dx, dy, z)
    c = table.coefficients
    wx, wy, wz = _spline_cubic_weights(tx), _spline_cubic_weights(ty),
        _spline_cubic_weights(tz)
    value = zero(T)
    @inbounds for k in 1:4
        plane = zero(T)
        for j in 1:4
            row = zero(T)
            for i in 1:4
                row += wy[i] * c[iy + i, ix + j, iz + k]
            end
            plane += wx[j] * row
        end
        value += wz[k] * plane
    end
    return value
end

# A single coefficient load feeds all four accumulators. Differentiate the basis in
# grid coordinates, then convert to derivatives per micron along each physical axis.
@inline function _spline_value_derivatives(
        table::_PixelIntegratedPSF{T}, dx::T, dy::T, z::T
    ) where {T}
    zeros4 = (zero(T), zero(T), zero(T), zero(T))
    radius_x, radius_y = table.lateral_radius
    (abs(dx) > radius_x || abs(dy) > radius_y) && return zeros4
    ix, iy, iz, tx, ty, tz = _spline_coordinates(table, dx, dy, z)
    wx, wy, wz = _spline_cubic_weights(tx), _spline_cubic_weights(ty),
        _spline_cubic_weights(tz)
    dwx = _spline_cubic_derivatives(tx)
    dwy = _spline_cubic_derivatives(ty)
    dwz = _spline_cubic_derivatives(tz)
    c = table.coefficients
    value, gx, gy, gz = zeros4
    @inbounds for k in 1:4
        plane, plane_x, plane_y = zero(T), zero(T), zero(T)
        for j in 1:4
            row, row_y = zero(T), zero(T)
            for i in 1:4
                coefficient = c[iy + i, ix + j, iz + k]
                row += wy[i] * coefficient
                row_y += dwy[i] * coefficient
            end
            plane += wx[j] * row
            plane_x += dwx[j] * row
            plane_y += wx[j] * row_y
        end
        value += wz[k] * plane
        gx += wz[k] * plane_x
        gy += wz[k] * plane_y
        gz += dwz[k] * plane
    end
    value <= zero(T) && return zeros4
    h_x, h_y = table.lateral_step
    h_z = table.z_step
    return value, gx / h_x, gy / h_y, gz / h_z
end

# At an interior node, c[i-1] + 4c[i] + c[i+1] = 6f[i]. Natural ends give
# c[1] = f[1], c[n] = f[n]; solve the remaining tridiagonal system in place.
function _spline_prefilter_line!(line::AbstractVector{T}, inverse_pivots) where {T}
    n = length(line)
    for i in 2:(n - 1)
        rhs = T(6) * line[i]
        i == 2 && (rhs -= line[1])
        i == n - 1 && (rhs -= line[n])
        i > 2 && (rhs -= line[i - 1])
        line[i] = rhs * inverse_pivots[i]
    end
    for i in (n - 2):-1:2
        line[i] -= inverse_pivots[i] * line[i + 1]
    end
    return nothing
end

function _spline_prefilter!(values::Array{T, 3}) where {T}
    ny, nx, nz = size(values)
    inverse_pivots = Vector{T}(undef, max(ny, nx, nz))
    for axis in 1:3
        n = size(values, axis)
        inverse_pivots[2] = inv(T(4))
        for i in 3:(n - 1)
            inverse_pivots[i] = inv(T(4) - inverse_pivots[i - 1])
        end
        if axis == 1
            for k in 1:nz, j in 1:nx
                _spline_prefilter_line!(@view(values[:, j, k]), inverse_pivots)
            end
        elseif axis == 2
            for k in 1:nz, i in 1:ny
                _spline_prefilter_line!(@view(values[i, :, k]), inverse_pivots)
            end
        else
            for j in 1:nx, i in 1:ny
                _spline_prefilter_line!(@view(values[i, j, :]), inverse_pivots)
            end
        end
    end
    return values
end

function _spline_pad_coefficients(values::Array{T, 3}) where {T}
    ny, nx, nz = size(values)
    c = Array{T, 3}(undef, ny + 2, nx + 2, nz + 2)
    c[2:(ny + 1), 2:(nx + 1), 2:(nz + 1)] .= values
    # Linear extrapolation gives zero second derivative at each endpoint. Fill y, x,
    # then z so the edges and corners also use initialized coefficients.
    for k in 2:(nz + 1), j in 2:(nx + 1)
        c[1, j, k] = 2 * c[2, j, k] - c[3, j, k]
        c[ny + 2, j, k] = 2 * c[ny + 1, j, k] - c[ny, j, k]
    end
    for k in 2:(nz + 1), i in 1:(ny + 2)
        c[i, 1, k] = 2 * c[i, 2, k] - c[i, 3, k]
        c[i, nx + 2, k] = 2 * c[i, nx + 1, k] - c[i, nx, k]
    end
    for j in 1:(nx + 2), i in 1:(ny + 2)
        c[i, j, 1] = 2 * c[i, j, 2] - c[i, j, 3]
        c[i, j, nz + 2] = 2 * c[i, j, nz + 1] - c[i, j, nz]
    end
    return c
end

# Mean of four clipped point samples in one sub-cell (two Gauss nodes per axis).
@inline function _spline_cell_average(spline, x::T, y::T, z::T, q_x::T, q_y::T) where {T}
    return (
        max(zero(T), T(spline(x - q_x, y - q_y, z))) +
            max(zero(T), T(spline(x + q_x, y - q_y, z))) +
            max(zero(T), T(spline(x - q_x, y + q_y, z))) +
            max(zero(T), T(spline(x + q_x, y + q_y, z)))
    ) / T(4)
end

function _build_pixel_psf(
        spline, pixel_size::NTuple{2, T}, lateral_range::T, z_range
    ) where {T}
    px_x, px_y = pixel_size
    h_x, h_y = px_x / T(5), px_y / T(5)
    q_x, q_y = h_x / (T(2) * sqrt(T(3))), h_y / (T(2) * sqrt(T(3)))
    K_x = ceil(Int, (lateral_range + px_x) / h_x)
    K_y = ceil(Int, (lateral_range + px_y) / h_y)
    nx, ny = 2K_x + 1, 2K_y + 1
    zlo, zhi = z_range
    nz = max(2, ceil(Int, (zhi - zlo) / T(0.02)) + 1)
    dz = (zhi - zlo) / T(nz - 1)
    values = Array{T, 3}(undef, ny, nx, nz)
    cells = Matrix{T}(undef, ny + 4, nx + 4)
    x_average = Matrix{T}(undef, ny + 4, nx)
    for k in 1:nz
        # Use the exact inward-rounded endpoint on the final plane.
        z = k == nz ? zhi : clamp(zlo + T(k - 1) * dz, zlo, zhi)
        for j in 1:(nx + 4), i in 1:(ny + 4)
            x, y = T(j - K_x - 3) * h_x, T(i - K_y - 3) * h_y
            cells[i, j] = _spline_cell_average(spline, x, y, z, q_x, q_y)
        end
        # Separable five-cell sliding sums, first x then y. Sum each short window
        # directly to avoid cancellation in the PSF tails from a rolling subtraction.
        for j in 1:nx, i in 1:(ny + 4)
            total = zero(T)
            for s in 0:4
                total += cells[i, j + s]
            end
            x_average[i, j] = total / T(5)
        end
        for j in 1:nx, i in 1:ny
            total = zero(T)
            for s in 0:4
                total += x_average[i + s, j]
            end
            values[i, j, k] = total / T(5)
        end
    end
    coefficients = _spline_pad_coefficients(_spline_prefilter!(values))
    return _PixelIntegratedPSF(
        coefficients, (h_x, h_y), (T(K_x) * h_x, T(K_y) * h_y), (zlo, zhi), dz
    )
end

# Direct quadrature is also needed at z_ref, which can lie outside the fitted z_range.
function _spline_pixel_average(spline, x::T, y::T, z::T, pixel_size::NTuple{2, T}) where {T}
    px_x, px_y = pixel_size
    h_x, h_y = px_x / T(5), px_y / T(5)
    q_x, q_y = h_x / (T(2) * sqrt(T(3))), h_y / (T(2) * sqrt(T(3)))
    total = zero(T)
    for iy in -2:2, ix in -2:2
        total += _spline_cell_average(
            spline, x + T(ix) * h_x, y + T(iy) * h_y, z, q_x, q_y
        )
    end
    return total / T(25)
end

function _spline_reference_sum(
        spline, pixel_size::NTuple{2, T}, lateral_range::T, z_ref::T
    ) where {T}
    px_x, px_y = pixel_size
    n_px_x, n_px_y = ceil(Int, lateral_range / px_x), ceil(Int, lateral_range / px_y)
    norm_sum = zero(T)
    for iy in (-n_px_y):n_px_y, ix in (-n_px_x):n_px_x
        norm_sum += _spline_pixel_average(
            spline, ix * px_x, iy * px_y, z_ref, pixel_size
        )
    end
    return norm_sum
end

"""
    _compute_psf_scale(spline, pixel_size, lateral_range, z_ref) -> Float32

Normalize a plain callable's pixel averages on the reference patch at `z_ref`, using the
table's quadrature. Throw `ArgumentError` for a non-finite or non-positive reference sum.
The MicroscopePSFs extension instead preserves photon densities using the pixel area.
Construction and pixel-size rescaling both dispatch through this function.
"""
function _compute_psf_scale(
        spline, pixel_size::NTuple{2, T}, lateral_range::T, z_ref::T
    ) where {T}
    norm_sum = _spline_reference_sum(spline, pixel_size, lateral_range, z_ref)
    (isfinite(norm_sum) && norm_sum > zero(T)) || throw(
        ArgumentError("PSF normalization sum at z = $z_ref must be finite and positive")
    )
    return one(T) / norm_sum
end

# --- PSFModel interface ---

@inline function evaluate_psf(
        model::SplinePSFModel{T}, i, j,
        θ::SVector{5, T}
    ) where {T}
    x, y, z, N, bg = θ
    px_x, px_y = model.pixel_size
    dx = (j - x) * px_x
    dy = (i - y) * px_y
    z_c = clamp(z, model.z_range[1], model.z_range[2])
    psf_val = max(zero(T), model.pixel_psf(dx, dy, z_c))
    return bg + N * psf_val * model.px_scale
end

@inline function compute_pixel_derivatives(
        i, j, θ::SVector{5, T},
        model::SplinePSFModel{T}
    ) where {T}
    x, y, z, N, bg = θ
    px_x, px_y = model.pixel_size
    scale = model.px_scale
    dx = (j - x) * px_x
    dy = (i - y) * px_y
    z_c = clamp(z, model.z_range[1], model.z_range[2])

    value, gx, gy, gz = _spline_value_derivatives(
        model.pixel_psf, dx, dy, z_c
    )
    f0 = value * scale

    # Chain rule: model = bg + N * f0
    # dx = (j-x)*px_x, dy = (i-y)*px_y → ∂dx/∂x = -px_x, ∂dy/∂y = -px_y
    model_val = bg + N * f0

    dudt = @SVector [
        -N * px_x * gx * scale, # ∂model/∂x
        -N * px_y * gy * scale, # ∂model/∂y
        N * gz * scale,        # ∂model/∂z
        f0,                   # ∂model/∂N
        one(T),               # ∂model/∂bg
    ]

    # The spline is fitted by Levenberg-Marquardt, which uses no second derivatives.
    return model_val, dudt, zero(dudt)
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
    return _spline_z_scan(roi, box_size, model, x, y, total, bg)
end

# Grid spacing (µm) of the z start scan. The basin of the true z is several tenths of a µm
# wide; a 0.2 µm grid already picks a neighbouring basin at |z| 0.3-0.4 µm.
const SPLINE_Z_SCAN_STEP = 0.1f0

# z on an even grid over z_range (spacing at most SPLINE_Z_SCAN_STEP) with the highest
# Poisson log-likelihood at the given x, y, N and bg; return all starting parameters.
@inline function _spline_z_scan(
        roi, box_size::Int, model::SplinePSFModel{T}, x, y, N, bg
    ) where {T}
    zlo, zhi = model.z_range
    # Host construction already allocated a finer axial grid, so this count fits Int.
    n = max(2, unsafe_trunc(Int, ceil((zhi - zlo) / SPLINE_Z_SCAN_STEP)) + 1)
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
    return SVector{5, T}(x, y, best_z, N, bg)
end

function to_pixel_units(
        model::SplinePSFModel{T}, pixel_size::Union{Real, Tuple{Real, Real}}
    ) where {T}
    new_px = _spline_pixel_size(T, pixel_size)
    new_px == model.pixel_size && return model
    new_scale = _compute_psf_scale(
        model.spline_psf, new_px, model.lateral_range, model.z_ref
    )
    pixel_psf = _build_pixel_psf(
        model.spline_psf, new_px, model.lateral_range, model.z_range
    )
    return SplinePSFModel{T, typeof(model.spline_psf), typeof(pixel_psf)}(
        model.spline_psf, pixel_psf, new_px,
        model.z_range, model.lateral_range,
        new_scale, model.z_ref, model.starts
    )
end

function to_pixel_units(model::SplinePSFModel, pixel_size_x::Real, pixel_size_y::Real)
    return to_pixel_units(model, (pixel_size_x, pixel_size_y))
end

# Levenberg-Marquardt on the full Fisher matrix (the kernel's `_optimize`): the
# per-parameter Newton step stops short on the coupled x, y, z, N and background, and LM
# needs no cap on the photon and background steps. One LM fit climbs to the maximum of the
# basin it starts in, and the scan start θ0 is often in the wrong one for a defocused ROI.
# So the fit also runs from `starts - 1` fixed z starts (x, y, N and bg from θ0), each with
# the full `iterations`, and keeps the θ with the strictly higher objective; on a tie the
# earlier start wins, the scan start first.
@inline function _optimize(
        roi, box_size::Int, psf_model::SplinePSFModel{T}, camera_model, variance_map,
        corner_x, corner_y, constraints, iterations::Int, θ0
    ) where {T}
    θ_best, obj_best = _lm_fit(
        roi, box_size, psf_model, camera_model, variance_map, corner_x, corner_y,
        constraints, iterations, θ0
    )
    m = psf_model.starts - 1
    for q in 0:(m - 1)
        θs = SVector{5, T}(θ0[1], θ0[2], _spline_start_z(psf_model, q, m), θ0[4], θ0[5])
        θ, obj = _lm_fit(
            roi, box_size, psf_model, camera_model, variance_map, corner_x, corner_y,
            constraints, iterations, θs
        )
        if obj > obj_best
            θ_best = θ
            obj_best = obj
        end
    end
    return θ_best
end

function default_constraints(model::SplinePSFModel, box_size)
    lower, upper = _default_position_bounds(box_size)
    return ParameterConstraints{5}(
        SVector{5, Float32}(lower, lower, model.z_range[1], 1.0f0, 0.01f0),
        SVector{5, Float32}(upper, upper, model.z_range[2], Inf32, Inf32),
        SVector{5, Float32}(1.0f0, 1.0f0, 0.05f0, Inf32, Inf32)
    )
end

function _resolve_constraints(
        constraints::ParameterConstraints{5}, model::SplinePSFModel, box_size
    )
    lo = max(constraints.lower[3], model.z_range[1])
    hi = min(constraints.upper[3], model.z_range[2])
    lo <= hi || throw(ArgumentError("custom z constraints do not intersect model.z_range"))
    return ParameterConstraints{5}(
        Base.setindex(constraints.lower, lo, 3), Base.setindex(constraints.upper, hi, 3),
        constraints.max_step
    )
end

Base.show(io::IO, m::SplinePSFModel) = print(
    io,
    "SplinePSFModel(pixel_size=$(m.pixel_size), z_range=$(m.z_range), " *
        "starts=$(m.starts))"
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
    return _to_emitter_xyznb(result, idx, camera; dataset, track_id, id)
end
