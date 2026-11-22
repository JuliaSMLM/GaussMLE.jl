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
z offsets are held fixed. The reference view's registration must be exactly the identity
(MicroscopePSFs accepts one within rounding of it; that is refused here), because the
fitted position is reported on the reference camera.

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

Available when MicroscopePSFs 0.5.7 or later is loaded (package extension; earlier
versions have no `MultiChannelPSF`). `mc` must have two channels, each a 3D `SplinePSF` (or
a `PlanePSF` of one) that integrates to one at every z (a 4Pi channel, normalized only on
average over its fringe, is not supported). View 1 is `mc`'s reference channel
(`mc.reference`), whichever index it has, and view 2 the other one. `z_range` is in the
reference view's z; each channel's table must cover it shifted by the channel's `z_offset`,
else `ArgumentError`. `lateral_range`, `pixel_size` and `starts` mean what they mean for
[`SplinePSFModel`](@ref), and each view gets that model's checks: its grid must cover
`lateral_range` plus one pixel, and `fit` refuses an ROI whose half-diagonal exceeds
`lateral_range`. The registration's linear part must be invertible and not close to
singular (`|det| > 1e-6` times its largest entry squared).

# Fitting
`fit((batch1, batch2), fitter)` with two `ROIBatch`es: `batch1` holds the reference view's
ROIs (the channel `mc.reference`, not necessarily channel 1) and `batch2` the other
view's, one pair per emitter in the same order, with the same ROI size and frame indices.
Each batch's corners place its ROIs on its own camera, and each batch's camera calibration
applies to its own pixels (the two may be the same camera object or two crops of one
sensor). Both batches must use the same camera type (`IdealCamera` or `SCMOSCamera`), and
each camera's pixel edges must be evenly spaced.

The corners enter the fit: they and the registration say where the emitter of ROI 1 lies
in ROI 2. `fit` therefore refuses a ROI that does not lie on its camera, and a pair whose
ROI 2 is not where the registration puts ROI 1: the centre of ROI 1, mapped to camera 2,
must lie within rounding of the centre of ROI 2 (half a camera-1 pixel mapped to camera 2,
plus one camera-2 pixel, per axis). Swapped batches, corners left at 1 and a crop with the
wrong origin are refused by this check unless the registration is close to the identity.

The camera pixel sizes may differ from `pixel_size` and from each other: each view's table
is integrated again on its camera's pixel grid, on every `fit` call for a view whose camera
pixel size is not `pixel_size` (as `SplinePSFModel`'s is).

The likelihood is that of [`SplinePSFModel`](@ref) summed over both images: Poisson for
`IdealCamera`, Poisson-shifted `(d + v) log(μ + v) - (μ + v)` with `v` the read-noise
variance for `SCMOSCamera`. The fit is the same Levenberg-Marquardt fit on the full 6 × 6
Fisher matrix with the same stop test and the same z starts. The start uses both images:
edge backgrounds, the two centroids weighted by transmission times signal (an image with
zero transmission sets only its own background), and a z scan of the Poisson likelihood of
both images (for sCMOS too, as for `SplinePSFModel`; the fit itself uses the sCMOS
likelihood). Uncertainties are the CRLB from the inverse Fisher matrix, and the p-value is
the Pearson statistic over both images' pixels against χ² with `2 * box^2 - 6` degrees of
freedom. CPU, CUDA and AMDGPU.

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
    A_px, t_px = _pixel_registration(model, camera1, camera2)
    A_px_inv = inv(A_px)
    return JointSplinePSFModel{T, typeof(v1), typeof(v2)}(
        (v1, v2), model.transmission, model.z_offset, model.registration, model.z_range,
        model.starts, SMatrix{2, 2, T}(A_px), SVector{2, T}(t_px), SMatrix{2, 2, T}(A_px_inv)
    )
end

# View 2's registration in camera pixels, in Float64: (A_px, t_px) with view-2 pixel centre
# = A_px * reference pixel centre + t_px (1-based).
function _pixel_registration(model::JointSplinePSFModel, camera1, camera2)
    e1, p1 = _uniform_pixel_grid(camera1, 1)
    e2, p2 = _uniform_pixel_grid(camera2, 2)
    A = model.registration[:, SVector(1, 2)]
    t = model.registration[:, 3]
    A_px = (A .* p1') ./ p2
    half = SVector(0.5, 0.5)
    t_px = (A * e1 + t - e2) ./ p2 - A_px * half + half
    return A_px, t_px
end

# A camera's first pixel edge and pixel size per axis (x, y), in Float64. The joint fit
# places every pixel of both cameras at edge + (index - 1/2) * pixel size, so edges that
# are not evenly spaced (to 1 % of a pixel anywhere on the sensor) are refused: the
# registration would be applied at the wrong camera positions. The pixel size is the mean
# over the sensor: the first pixel's width alone carries the rounding of Float32 edges far
# from zero, which would add up across the sensor.
function _uniform_pixel_grid(camera, view::Int)
    axis(edges, name) = begin
        n = length(edges) - 1
        e = Float64(first(edges))
        p = (Float64(last(edges)) - e) / n
        (n >= 1 && isfinite(e) && isfinite(p) && p > 0) || throw(
            ArgumentError("view $view's camera has no positive finite $name pixel size")
        )
        for (k, edge) in enumerate(edges)
            abs(Float64(edge) - (e + (k - 1) * p)) <= 0.01 * p || throw(
                ArgumentError(
                    "view $view's camera has unevenly spaced $name pixel edges: edge " *
                        "$k is at $edge um, and evenly spaced pixels of $p um put it " *
                        "at $(e + (k - 1) * p) um; a joint fit needs evenly spaced pixels"
                )
            )
        end
        (e, p)
    end
    ex, px = axis(camera.pixel_edges_x, "x")
    ey, py = axis(camera.pixel_edges_y, "y")
    return SVector(ex, ey), SVector(px, py)
end

# Each pair's ROIs must lie on their own cameras, and ROI 2 must be where the registration
# puts ROI 1: the corners enter the likelihood (`_joint_offset`), so a wrong pair is not an
# output shift but a wrong fit. The centre of ROI 1 mapped to camera 2 must lie within
# rounding of the centre of ROI 2 on each axis: half a camera-1 pixel mapped through A_px
# (the emitter is within half a pixel of its ROI's centre pixel), half a camera-2 pixel
# (ROI 2 centred to the nearest pixel), and half a pixel more for an even box, whose centre
# lies between pixels.
function _validate_pair_geometry(model::JointSplinePSFModel, batch1::ROIBatch, batch2::ROIBatch)
    box = size(batch1.data, 1)
    for (v, b) in ((1, batch1), (2, batch2))
        sensor = (length(b.camera.pixel_edges_y) - 1, length(b.camera.pixel_edges_x) - 1)
        _validate_roi_corners(box, b.x_corners, b.y_corners, sensor; what = "view $v: ROI")
    end
    A_px, t_px = _pixel_registration(model, batch1.camera, batch2.camera)
    tol = SVector(
        0.5 * (abs(A_px[1, 1]) + abs(A_px[1, 2])) + 1.0 + 1.0e-6,
        0.5 * (abs(A_px[2, 1]) + abs(A_px[2, 2])) + 1.0 + 1.0e-6,
    )
    half = (box - 1) / 2
    for k in eachindex(batch1.x_corners)
        c1 = SVector(batch1.x_corners[k] + half, batch1.y_corners[k] + half)
        c2 = SVector(batch2.x_corners[k] + half, batch2.y_corners[k] + half)
        d = A_px * c1 + t_px - c2
        (abs(d[1]) <= tol[1] && abs(d[2]) <= tol[2]) || throw(
            ArgumentError(
                "pair $k: the registration puts the centre of ROI 1 (corner x = " *
                    "$(batch1.x_corners[k]), y = $(batch1.y_corners[k])) " *
                    "$(round(d[1]; digits = 2)) px in x and $(round(d[2]; digits = 2)) " *
                    "px in y from the centre of ROI 2 (corner x = $(batch2.x_corners[k]), " *
                    "y = $(batch2.y_corners[k])), beyond the $(round(tol[1]; digits = 2)) " *
                    "and $(round(tol[2]; digits = 2)) px that rounding allows. batch1 " *
                    "must be the reference channel's ROIs, and each batch's corners must " *
                    "place its ROIs on its own camera."
            )
        )
    end
    return nothing
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

_resolve_constraints(
    constraints::ParameterConstraints{6}, model::JointSplinePSFModel, box_size
) = _z_range_constraints(constraints, model)

function default_constraints(model::JointSplinePSFModel, box_size)
    lower, upper = _default_position_bounds(box_size)
    return ParameterConstraints{6}(
        SVector{6, Float32}(lower, lower, model.z_range[1], 1.0f0, 0.01f0, 0.01f0),
        SVector{6, Float32}(upper, upper, model.z_range[2], Inf32, Inf32, Inf32),
        SVector{6, Float32}(1.0f0, 1.0f0, 0.05f0, Inf32, Inf32, Inf32)
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
# the reference view), weighted by transmission times in-box sum; N from the sums of the
# views that receive light over the summed transmission; z from a Poisson likelihood scan
# over both images. A view with zero transmission holds no light of the emitter, so its
# sum above the background is noise: it sets its own background and nothing else. With no
# signal in a lit view the position starts at the centre of ROI 1.
@inline function _fit_start(
        roi, box_size::Int, model::JointSplinePSFModel{T}, corner_x, corner_y
    ) where {T}
    bg1, s1, x1, y1 = _roi_moments(roi[1], box_size, T)
    bg2, s2, x2, y2 = _roi_moments(roi[2], box_size, T)
    o = _joint_offset(model, corner_x, corner_y)
    p2 = model.A_px_inv * (SVector{2, T}(x2, y2) - o)
    tr1, tr2 = model.transmission
    w1, w2 = tr1 * s1, tr2 * s2
    w = w1 + w2
    centre = T(box_size + 1) / 2
    x = w > zero(T) ? (w1 * x1 + w2 * p2[1]) / w : centre
    y = w > zero(T) ? (w1 * y1 + w2 * p2[2]) / w : centre
    lit = ifelse(tr1 > zero(T), s1, zero(T)) + ifelse(tr2 > zero(T), s2, zero(T))
    N = max(lit, one(T)) / (tr1 + tr2)
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
