"""
Joint spline PSF model: one emitter seen in two images (views) of the same frame, such as
the two polarization channels of a beam splitter, fitted to both at once with a tabulated
PSF per view.
"""

"""
    JointSplinePSFModel{T, M1, M2} <: PSFModel{6, T}

One emitter in two images of the same frame, fitted to both at once: shared x, y, z and
total photons, one background per image. Built from a two-channel
`MicroscopePSFs.MultiChannelPSF` (package extension), whose forward law it fits:

    μ_v(pixel) = N * transmission[v] * PSF_v pixel average at channel_position(mc, v, x, y, z)
                 + bg_v

where `channel_position` maps the reference camera position (microns, absolute pixel-edge
coordinates) through view `v`'s registration `[A t]` and shifts z by its `z_offset`. The
registration, transmission (the photon split between the views, learned with the PSF) and
z offsets are held fixed; the reference view's transform is the identity.

# Parameters (in order)
1. x: x-position in the reference view (fitted in ROI pixels, output in microns)
2. y: y-position in the reference view (fitted in ROI pixels, output in microns)
3. z: axial position in the reference view (microns throughout)
4. N: total photons over both views before transmission (electrons for sCMOS)
5. bg: background per pixel of the reference view
6. bg2: background per pixel of the other view

# Constructor
    JointSplinePSFModel(mc::MicroscopePSFs.MultiChannelPSF; pixel_size,
                        z_range = <the z range both channels cover>,
                        lateral_range = nothing, starts = nothing)

Available when MicroscopePSFs is loaded (package extension). `mc` must have two channels,
each a 3D `SplinePSF` (or a `PlanePSF` of one), and its reference channel is view 1.
`z_range` is in the reference view's z; each channel's table must cover it shifted by the
channel's `z_offset`, else `ArgumentError`. `lateral_range`, `pixel_size` and `starts`
mean what they mean for [`SplinePSFModel`](@ref), and each view gets that model's checks:
its grid must cover `lateral_range` plus one pixel, and `fit` refuses an ROI whose
half-diagonal exceeds `lateral_range`.

# Fitting
`fit((batch1, batch2), fitter)` with two `ROIBatch`es: `batch1` holds the reference view's
ROIs and `batch2` the other view's, one pair per emitter in the same order, with the same
ROI size and frame indices. Each batch's corners place its ROIs on its own camera, and
each batch's camera calibration applies to its own pixels (the two may be the same camera
object or two crops of one sensor). The camera pixel sizes may differ from `pixel_size` and
from each other: each view's table is integrated again on its camera's pixel grid. Both
batches must use the same camera type (`IdealCamera` or `SCMOSCamera`).

The likelihood is that of [`SplinePSFModel`](@ref) summed over both images: Poisson for
`IdealCamera`, Poisson-shifted `(d + v) log(μ + v) - (μ + v)` with `v` the read-noise
variance for `SCMOSCamera`. The fit is the same Levenberg-Marquardt fit on the full 6 × 6
Fisher matrix with the same stop test and the same z starts; the start uses both images
(edge backgrounds, centroids, a joint z scan). Uncertainties are the CRLB from the inverse
Fisher matrix, and the p-value is the Pearson statistic over both images' pixels against
χ² with `2 * box^2 - 6` degrees of freedom. CPU, CUDA and AMDGPU.

The result is a `BasicSMLD` of [`Emitter3DFitJoint`](@ref), positions in the reference
view's camera microns.
"""
struct JointSplinePSFModel{T, M1 <: SplinePSFModel{T}, M2 <: SplinePSFModel{T}} <:
    PSFModel{6, T}
    views::Tuple{M1, M2}
    transmission::Tuple{T, T}
    z_offset::Tuple{T, T}
    # View 2's registration [A t]: reference camera microns to view-2 camera microns
    registration::SMatrix{2, 3, Float64, 6}
    z_range::Tuple{T, T}
    starts::Int
    # The same map in camera pixels (1-based pixel centres), for the fit's cameras: view-2
    # pixel = A_px * reference pixel + t_px. Set by `_model_for_cameras`.
    A_px::SMatrix{2, 2, T, 4}
    t_px::SVector{2, T}
    # A_px's inverse, for the start
    A_px_inv::SMatrix{2, 2, T, 4}
end

_has_axial_position(::Type{<:JointSplinePSFModel}) = true

Base.show(io::IO, m::JointSplinePSFModel) = print(
    io,
    "JointSplinePSFModel(pixel_size=$(m.views[1].pixel_size), z_range=$(m.z_range), " *
        "transmission=$(m.transmission), starts=$(m.starts))"
)

# Built by the extension from the two channels' point PSFs. `views` are SplinePSFModels in
# each channel's own z (z_range shifted by z_offset).
function _joint_spline_model(
        views::Tuple{SplinePSFModel{T}, SplinePSFModel{T}}, transmission, z_offset,
        registration::AbstractMatrix, z_range::Tuple{T, T}, starts::Int
    ) where {T}
    reg = SMatrix{2, 3, Float64}(registration)
    all(isfinite, reg) || throw(ArgumentError("registration must be finite"))
    a = reg[:, SVector(1, 2)]
    d = a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]
    (isfinite(d) && abs(d) > 1.0e-6 * maximum(abs, a)^2) ||
        throw(ArgumentError("the registration's linear part must be invertible"))
    tr = (T(transmission[1]), T(transmission[2]))
    all(t -> isfinite(t) && t >= zero(T), tr) && (tr[1] + tr[2]) > zero(T) || throw(
        ArgumentError("transmission must be finite, non-negative and not both zero")
    )
    I2 = SMatrix{2, 2, T}(1, 0, 0, 1)
    return JointSplinePSFModel{T, typeof(views[1]), typeof(views[2])}(
        views, tr, (T(z_offset[1]), T(z_offset[2])), reg, z_range, starts,
        I2, zero(SVector{2, T}), I2
    )
end

# The model for the two cameras the views were recorded on: each view's table at its
# camera's pixel size, and the registration in camera pixels. Camera microns of pixel
# centre c (1-based) are e + (c - 1/2) p, e the camera's first pixel edge and p its pixel
# size, per axis; with X2 = A X1 + t in microns, c2 = P2^-1 A P1 c1 + P2^-1 (A e1 + t - e2)
# - P2^-1 A P1 / 2 + 1/2, P the diagonal pixel sizes.
function _model_for_cameras(model::JointSplinePSFModel{T}, camera1, camera2) where {T}
    v1 = _model_for_camera(model.views[1], camera1)
    v2 = _model_for_camera(model.views[2], camera2)
    origin(c) = SVector{2, Float64}(c.pixel_edges_x[1], c.pixel_edges_y[1])
    pixel(c) = SVector{2, Float64}(
        c.pixel_edges_x[2] - c.pixel_edges_x[1], c.pixel_edges_y[2] - c.pixel_edges_y[1]
    )
    e1, e2, p1, p2 = origin(camera1), origin(camera2), pixel(camera1), pixel(camera2)
    A = model.registration[:, SVector(1, 2)]
    t = model.registration[:, 3]
    A_px = (A .* p1') ./ p2
    half = SVector(0.5, 0.5)
    t_px = (A * e1 + t - e2) ./ p2 - A_px * half + half
    A_px_inv = inv(A_px)
    return JointSplinePSFModel{T, typeof(v1), typeof(v2)}(
        (v1, v2), model.transmission, model.z_offset, model.registration, model.z_range,
        model.starts, SMatrix{2, 2, T}(A_px), SVector{2, T}(t_px), SMatrix{2, 2, T}(A_px_inv)
    )
end

function Adapt.adapt_structure(to, model::JointSplinePSFModel{T}) where {T}
    v1 = Adapt.adapt(to, model.views[1])
    v2 = Adapt.adapt(to, model.views[2])
    return JointSplinePSFModel{T, typeof(v1), typeof(v2)}(
        (v1, v2), model.transmission, model.z_offset, model.registration, model.z_range,
        model.starts, model.A_px, model.t_px, model.A_px_inv
    )
end

_prepare_psf_model(backend, model::JointSplinePSFModel) = Adapt.adapt(backend, model)
_psf_table_bytes(model::JointSplinePSFModel) =
    _psf_table_bytes(model.views[1]) + _psf_table_bytes(model.views[2])

function _validate_psf_box(model::JointSplinePSFModel, box_size::Integer)
    _validate_psf_box(model.views[1], box_size)
    _validate_psf_box(model.views[2], box_size)
    return nothing
end

_fit_pixels(box_size::Int, ::JointSplinePSFModel) = 2 * box_size * box_size

_photon_background_indices(::JointSplinePSFModel) = (4, 5, 6)

function default_constraints(model::JointSplinePSFModel, box_size)
    lower, upper = _default_position_bounds(box_size)
    return ParameterConstraints{6}(
        SVector{6, Float32}(lower, lower, model.z_range[1], 1.0f0, 0.01f0, 0.01f0),
        SVector{6, Float32}(upper, upper, model.z_range[2], Inf32, Inf32, Inf32),
        SVector{6, Float32}(1.0f0, 1.0f0, 0.05f0, Inf32, Inf32, Inf32)
    )
end

function _resolve_constraints(
        constraints::ParameterConstraints{6}, model::JointSplinePSFModel, box_size
    )
    lo = max(constraints.lower[3], model.z_range[1])
    hi = min(constraints.upper[3], model.z_range[2])
    lo <= hi || throw(ArgumentError("custom z constraints do not intersect model.z_range"))
    return ParameterConstraints{6}(
        Base.setindex(constraints.lower, lo, 3), Base.setindex(constraints.upper, hi, 3),
        constraints.max_step
    )
end

# --- Kernel side ---

# View 2's position in its ROI pixels for the reference ROI position (x, y): A_px (x, y) +
# o, with o = A_px (c1 - 1) + t_px - (c2 - 1) from the two ROIs' corners. o is computed
# once per pass from camera-scale numbers (Float32 error about 1e-4 pixel); the position
# itself then adds only ROI-scale terms.
@inline function _joint_offset(
        model::JointSplinePSFModel{T}, corner_x, corner_y
    ) where {T}
    c1 = SVector{2, T}(corner_x[1] - 1, corner_y[1] - 1)
    c2 = SVector{2, T}(corner_x[2] - 1, corner_y[2] - 1)
    return model.A_px * c1 + model.t_px - c2
end

# Element v (1 or 2) of a pair by static indices, so a kernel compiler can select between
# two values in registers rather than index a tuple at run time.
@inline _pick(p::Tuple, v::Int) = v == 1 ? p[1] : p[2]

# Everything one view's pixel loop needs, chosen once per view and pass: the view's
# SplinePSFModel, ROI, variance map and corner, its position map A (the identity for the
# reference view), its photon share, and its own parameters at the joint θ: (x_v, y_v,
# z + z_offset_v, N * transmission_v, bg_v), (x_v, y_v) = A (x, y) + o_v.
@inline function _view_context(
        model::JointSplinePSFModel{T}, v::Int, θ::SVector{6, T}, o, roi, variance_map,
        corner_x, corner_y
    ) where {T}
    A = v == 1 ? SMatrix{2, 2, T}(1, 0, 0, 1) : model.A_px
    ov = v == 1 ? zero(SVector{2, T}) : o
    tr = _pick(model.transmission, v)
    θv = SVector{5, T}(
        A[1, 1] * θ[1] + A[1, 2] * θ[2] + ov[1], A[2, 1] * θ[1] + A[2, 2] * θ[2] + ov[2],
        θ[3] + _pick(model.z_offset, v), θ[4] * tr, v == 1 ? θ[5] : θ[6]
    )
    return (
        _pick(model.views, v), _pick(roi, v), _pick(variance_map, v), _pick(corner_x, v),
        _pick(corner_y, v), A, tr, θv,
    )
end

# A view's expected count at pixel (i, j) and its derivatives with respect to the joint
# parameters: the chain rule through the view's position map A and photon share tr.
@inline function _joint_view_pixel(
        view_model, v::Int, i, j, θv::SVector{5, T}, A, tr::T
    ) where {T}
    u, d = compute_pixel_derivatives(i, j, θv, view_model)
    dudt = SVector{6, T}(
        d[1] * A[1, 1] + d[2] * A[2, 1], d[1] * A[1, 2] + d[2] * A[2, 2], d[3], d[4] * tr,
        ifelse(v == 1, d[5], zero(T)), ifelse(v == 2, d[5], zero(T))
    )
    return u, dudt
end

# One loop over both views' pixels, so the kernel holds one copy of the pixel code.
@inline function _lm_pass(
        θ, roi, box_size::Int, model::JointSplinePSFModel{T}, camera_model, variance_map,
        corner_x, corner_y
    ) where {T}
    o = _joint_offset(model, corner_x, corner_y)
    acc = (zero(T), zero(T), zero(SVector{6, T}), zero(SMatrix{6, 6, T}))
    for v in 1:2
        acc = _view_lm_pass(
            acc, v, box_size, camera_model,
            _view_context(model, v, θ, o, roi, variance_map, corner_x, corner_y)...
        )
    end
    return acc
end

# One view's share of an LM pass. Not inlined: the POCL CPU kernel ran the pass several
# times slower with it inlined into the loop over views (as `_lm_solve` and `_trial_step`).
@noinline function _view_lm_pass(
        acc, v::Int, box_size::Int, camera_model, vmodel, r, vmap, cx, cy, A, tr, θv
    )
    @inbounds for j in 1:box_size, i in 1:box_size
        u, dudt = _joint_view_pixel(vmodel, v, i, j, θv, A, tr)
        w = _readout_variance(camera_model, u, vmap, cx, cy, i, j)
        acc = _lm_accumulate(acc, r[i, j], u, dudt, w)
    end
    return acc
end

@inline function _final_pass(
        θ, roi, box_size::Int, model::JointSplinePSFModel{T}, camera_model, variance_map,
        corner_x, corner_y
    ) where {T}
    o = _joint_offset(model, corner_x, corner_y)
    acc = (zero(T), zero(SMatrix{6, 6, T}))
    for v in 1:2
        acc = _view_final_pass(
            acc, v, box_size, camera_model,
            _view_context(model, v, θ, o, roi, variance_map, corner_x, corner_y)...
        )
    end
    return acc
end

@noinline function _view_final_pass(
        acc, v::Int, box_size::Int, camera_model, vmodel, r, vmap, cx, cy, A, tr, θv
    )
    @inbounds for j in 1:box_size, i in 1:box_size
        u, dudt = _joint_view_pixel(vmodel, v, i, j, θv, A, tr)
        w = _readout_variance(camera_model, u, vmap, cx, cy, i, j)
        acc = _final_accumulate(acc, r[i, j], u, dudt, w)
    end
    return acc
end

# Edge-pixel mean and the positive in-box sum above it with its centroid, for one ROI.
@inline function _roi_moments(roi, box_size::Int, ::Type{T}) where {T}
    edge_sum = zero(T)
    edge_count = 0
    @inbounds for j in 1:box_size, i in 1:box_size
        if i == 1 || i == box_size || j == 1 || j == box_size
            edge_sum += T(roi[i, j])
            edge_count += 1
        end
    end
    bg = edge_sum / T(edge_count)
    sum_x = zero(T)
    sum_y = zero(T)
    total = zero(T)
    @inbounds for j in 1:box_size, i in 1:box_size
        val = max(T(roi[i, j]) - bg, zero(T))
        sum_y += T(i) * val
        sum_x += T(j) * val
        total += val
    end
    centre = T(box_size + 1) / 2
    x = total > zero(T) ? sum_x / total : centre
    y = total > zero(T) ? sum_y / total : centre
    return bg, total, x, y
end

# Start: each view's edge background; x, y from the two centroids (view 2's mapped back to
# the reference view), weighted by their in-box sums; N from both sums over the summed
# transmission; z from a Poisson likelihood scan over both images.
@inline function _fit_start(
        roi, box_size::Int, model::JointSplinePSFModel{T}, corner_x, corner_y
    ) where {T}
    bg1, s1, x1, y1 = _roi_moments(roi[1], box_size, T)
    bg2, s2, x2, y2 = _roi_moments(roi[2], box_size, T)
    o = _joint_offset(model, corner_x, corner_y)
    p2 = model.A_px_inv * (SVector{2, T}(x2, y2) - o)
    w = s1 + s2
    x = w > zero(T) ? (s1 * x1 + s2 * p2[1]) / w : x1
    y = w > zero(T) ? (s1 * y1 + s2 * p2[2]) / w : y1
    N = max(w, one(T)) / (model.transmission[1] + model.transmission[2])
    return _joint_z_scan(roi, box_size, model, o, x, y, N, bg1, bg2)
end

@inline function _joint_z_scan(
        roi, box_size::Int, model::JointSplinePSFModel{T}, o, x, y, N, bg1, bg2
    ) where {T}
    zlo, zhi = model.z_range
    n = max(2, unsafe_trunc(Int, ceil((zhi - zlo) / SPLINE_Z_SCAN_STEP)) + 1)
    b1 = max(bg1, T(0.01))
    b2 = max(bg2, T(0.01))
    best_z, best_ll = zero(T), typemin(T)
    for k in 0:(n - 1)
        z = zlo + (zhi - zlo) * T(k) / T(n - 1)
        θ = SVector{6, T}(x, y, z, N, b1, b2)
        ll = zero(T)
        for v in 1:2
            vmodel, r, _, _, _, _, _, θv = _view_context(
                model, v, θ, o, roi, roi, (0, 0), (0, 0)
            )
            @inbounds for j in 1:box_size, i in 1:box_size
                μ = evaluate_psf(vmodel, i, j, θv)
                ll += T(r[i, j]) * log(μ) - μ
            end
        end
        if ll > best_ll
            best_z, best_ll = z, ll
        end
    end
    return SVector{6, T}(x, y, best_z, N, bg1, bg2)
end

@inline function _spline_start_z(model::JointSplinePSFModel{T}, q::Int, m::Int) where {T}
    zlo, zhi = model.z_range
    R = zhi - zlo
    center = (zlo + zhi) / 2
    m == 1 && return center
    return center + T(0.8) * R * (T(q) / T(m - 1) - T(0.5))
end

# As SplinePSFModel: the scan start, then `starts - 1` fixed z starts, keeping the fit with
# the strictly higher objective (the earlier start on a tie). One `_lm_fit` call site, so
# the kernel holds one copy of the fit.
@inline function _optimize(
        roi, box_size::Int, psf_model::JointSplinePSFModel{T}, camera_model, variance_map,
        corner_x, corner_y, constraints, iterations::Int, θ0
    ) where {T}
    θ_best = θ0
    obj_best = T(-Inf)
    stopped_best = false
    m = psf_model.starts - 1
    for q in -1:(m - 1)
        θs = q < 0 ? θ0 : Base.setindex(θ0, _spline_start_z(psf_model, q, m), 3)
        θ, obj, _, stopped = _lm_fit(
            roi, box_size, psf_model, camera_model, variance_map, corner_x, corner_y,
            constraints, iterations, θs
        )
        if q < 0 || obj > obj_best
            θ_best = θ
            obj_best = obj
            stopped_best = stopped
        end
    end
    return θ_best, stopped_best
end
