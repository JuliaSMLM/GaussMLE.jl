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
in ROI 2. Each ROI must contain its emitter's position in its image; that is the caller's
to ensure, and `fit` checks only that the two ROIs of a pair overlap. A pair whose corners
are wrong is a wrong fit, and beyond that requirement `fit` cannot tell it from a valid
pair by the corners alone:

- Refused: a ROI that does not lie on its camera, and a pair whose ROI 1 the registration
  maps entirely outside its ROI 2, so that no position is in both.
- Warned about, and fitted unchanged: pairs whose ROI 2 is farther from where the
  registration puts ROI 1 than two ROIs each placed on the pixel that holds the emitter can
  be (one camera-1 pixel mapped to camera 2 plus one camera-2 pixel, per axis, between the
  mapped centre of ROI 1 and the centre of ROI 2). The warning comes when the median over
  at least 5 pairs is beyond that, or single pairs are beyond 1.5 times that; pairs with a
  ROI at a sensor edge, which sits inward of its emitter, are not counted. A wrong
  registration, swapped batches and a crop with the wrong origin give this warning when
  they are not refused and move the ROIs beyond that limit; a smaller error, such as a crop
  origin one pixel off, can give none. So
  do valid pairs placed another way, such as on the brightest pixel of a PSF whose peak is
  off its emitter.
- Reported: every pair's offset, in `smld.metadata["pair_offsets_px"]` (2 × n, x and y in
  camera-2 pixels: the centre of ROI 1 mapped to camera 2, minus the centre of ROI 2), so a
  caller who knows how the ROIs were placed can apply a limit of its own.
- Fitted, with infinite uncertainties, as a fit on a position bound is: a fit whose emitter
  ends so far from a ROI's centre, in an image that receives light, that some pixel of
  that ROI is beyond the image's table (zero beyond `lateral_range` plus about one pixel).
  With a box as wide as the table reaches, that includes an emitter in the outermost
  pixels of a sensor.

An image with zero transmission holds no light of the emitter: its table need not reach
the box and its ROI is not judged for a cut-off PSF. If it is the second image, its ROI may
lie anywhere on its camera and the pair is neither refused nor warned about. The reference
ROI is judged whatever its transmission, because the fitted position is expressed in its
pixels and bounded near its box (the default bounds reach about a pixel beyond it).

A ROI 2 that is a pixel or two from where it should be is fitted as given, with finite
uncertainties and a wrong position. The p-value shows a large error of this kind, but at a
few hundred photons a one-pixel error mostly passes it: the pairing is the caller's to get
right.

The camera pixel sizes may differ from `pixel_size` and from each other: each view's table
is integrated again on its camera's pixel grid, on every `fit` call for a view whose camera
pixel size is not `pixel_size` (as `SplinePSFModel`'s is).

The likelihood is that of [`SplinePSFModel`](@ref) summed over both images: Poisson for
`IdealCamera`, Poisson-shifted `(d + v) log(μ + v) - (μ + v)` with `v` the read-noise
variance for `SCMOSCamera`. The fit is the same Levenberg-Marquardt fit on the full 6 × 6
Fisher matrix with the same stop test and the same z starts (with `starts = 1` the fit
relies on the scan alone, which can take the wrong side of focus where two z fit the start
almost equally well; the default starts are there for that). The start uses both images:
edge backgrounds, the two centroids weighted by transmission times signal (an image with
zero transmission sets only its own background), and a z scan of the Poisson likelihood of
the images that receive light (for sCMOS too, as for `SplinePSFModel`; the fit itself uses
the sCMOS likelihood). Uncertainties are the CRLB from the inverse Fisher matrix, and the p-value is
the Pearson statistic over both images' pixels against χ² with `2 * box^2 - 6` degrees of
freedom (all six parameters counted, also when bounds hold some fixed, as for every model).
CPU, CUDA and AMDGPU.

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
# camera's pixel size, and the registration in camera pixels. One pixel size per camera
# axis serves the tables, the registration and the output positions
# (`_uniform_pixel_grid`, `_mean_pixel_size`): a table at one size and a registration at
# another would place the emitter differently in the two. Camera microns of pixel
# centre c (1-based) are e + (c - 1/2) p, e the camera's first pixel edge and p its pixel
# size, per axis; with X2 = A X1 + t in microns, c2 = P2^-1 A P1 c1 + P2^-1 (A e1 + t - e2)
# - P2^-1 A P1 / 2 + 1/2, P the diagonal pixel sizes.
function _model_for_cameras(model::JointSplinePSFModel{T}, camera1, camera2) where {T}
    _, p1 = _uniform_pixel_grid(camera1, 1)
    _, p2 = _uniform_pixel_grid(camera2, 2)
    v1 = to_pixel_units(model.views[1], p1[1], p1[2])
    v2 = to_pixel_units(model.views[2], p2[1], p2[2])
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
# are not evenly spaced (to 1 % of a pixel anywhere on the sensor, plus the precision of
# the stored edges) are refused: the
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
        # an edge is stored to the precision of its type at its magnitude
        tol = 0.01 * p + Float64(eps(float(max(abs(first(edges)), abs(last(edges))))))
        for (k, edge) in enumerate(edges)
            abs(Float64(edge) - (e + (k - 1) * p)) <= tol || throw(
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

# A camera's mean pixel size per axis (x, y) in its edges' own type: the pixel size of
# `_uniform_pixel_grid`, for the conversion of fitted positions to microns.
function _mean_pixel_size(camera)
    ex, ey = camera.pixel_edges_x, camera.pixel_edges_y
    return (
        (last(ex) - first(ex)) / (length(ex) - 1), (last(ey) - first(ey)) / (length(ey) - 1),
    )
end

# Pairs away from the sensor edges needed before the median offset of a batch is judged
# (`_validate_pair_geometry`).
const JOINT_MEDIAN_MIN_PAIRS = 5

# The median of a vector of numbers (GaussMLE does not depend on Statistics).
function _median(v::AbstractVector)
    s = sort(v)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

# Whether the registration maps some point of ROI 1 into ROI 2. In camera-2 pixels ROI 1 is
# a parallelogram with half-edges u and v (the columns of A_px times box / 2), its centre
# d from the centre of ROI 2, a square of half-width box / 2. Two convex shapes are apart
# exactly when one of their edge normals separates them: the two camera axes for the
# square, the normals of u and v for the parallelogram.
function _rois_overlap(A_px, d, box::Int)
    hw = box / 2
    u = SVector(A_px[1, 1], A_px[2, 1]) * hw
    v = SVector(A_px[1, 2], A_px[2, 2]) * hw
    for n in (SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(-u[2], u[1]), SVector(-v[2], v[1]))
        reach = abs(n[1] * u[1] + n[2] * u[2]) + abs(n[1] * v[1] + n[2] * v[2]) +
            hw * (abs(n[1]) + abs(n[2]))
        abs(n[1] * d[1] + n[2] * d[2]) <= reach * (1 + 1.0e-9) || return false
    end
    return true
end

# The corners enter the likelihood (`_joint_offset`), so a wrong pair is a wrong fit, not an
# output shift. What the corners alone can show is checked here, before anything reads the
# data, from the offset of each pair: the centre of ROI 1 mapped to camera 2, minus the
# centre of ROI 2.
#
# Refused:
# - a ROI that does not lie on its own camera, for both camera types;
# - a pair whose ROI 1 maps entirely outside its ROI 2 (`_rois_overlap`): no position is in
#   both, and `fit` requires each ROI to contain its emitter's position in its image.
#   Footprints within 1e-9 of their reach of touching count as touching.
#
# Warned about, because a valid pair can be like this and the corners cannot tell:
# - a median offset, over at least JOINT_MEDIAN_MIN_PAIRS pairs, beyond one camera-1 pixel
#   mapped through A_px plus one camera-2 pixel, per axis. That is the most that two ROIs
#   each placed on the pixel that holds the emitter give (an even box has its centre half a
#   pixel from that pixel's centre, and every emitter of a batch can sit at the same place
#   in its pixel). Swapped batches, corners left at 1, a crop with the wrong origin and a
#   ROI 2 moved for every pair shift the median.
# - single pairs beyond 1.5 times that, when the median is within it.
# ROIs placed some other way (on a brightest pixel of a PSF whose peak is off its emitter,
# say) also give such offsets and are fitted correctly, so the fit continues and no result
# is changed. A ROI that touches a sensor edge sits inward of its emitter, as a box finder
# places it, by up to half a box; pairs with such a ROI are left out of both counts. With a
# second image of zero transmission nothing is judged: its ROI holds no light of the
# emitter.
# Returns every pair's offset (2 x n, x and y in camera-2 pixels), which `fit` puts in the
# result's metadata so a caller can apply a limit of its own.
function _validate_pair_geometry(model::JointSplinePSFModel, batch1::ROIBatch, batch2::ROIBatch)
    box = size(batch1.data, 1)
    for (v, b) in ((1, batch1), (2, batch2))
        sensor = (length(b.camera.pixel_edges_y) - 1, length(b.camera.pixel_edges_x) - 1)
        _validate_roi_corners(box, b.x_corners, b.y_corners, sensor; what = "view $v: ROI")
    end
    A_px, t_px = _pixel_registration(model, batch1.camera, batch2.camera)
    row = SVector(abs(A_px[1, 1]) + abs(A_px[1, 2]), abs(A_px[2, 1]) + abs(A_px[2, 2]))
    centre = (box + 1) / 2   # a ROI's centre in its own pixel coordinates
    # the last corner at which a ROI is on its sensor, per view and axis (x, y)
    last1 = SVector(
        length(batch1.camera.pixel_edges_x) - box, length(batch1.camera.pixel_edges_y) - box
    )
    last2 = SVector(
        length(batch2.camera.pixel_edges_x) - box, length(batch2.camera.pixel_edges_y) - box
    )
    tol = row .+ 1.0 .+ 1.0e-6
    offsets = Matrix{Float32}(undef, 2, length(batch1.x_corners))
    # a second image with zero transmission holds no light of the emitter: where its ROI
    # lies says nothing about the pair. The reference ROI is judged whatever its
    # transmission: the fitted position is in its pixels, bounded near its box.
    lit = model.transmission[2] > 0
    dx = Float64[]  # offsets of the pairs away from the sensor edges
    dy = Float64[]
    far, first_far, first_d = 0, 0, SVector(0.0, 0.0)
    for k in eachindex(batch1.x_corners)
        k1 = SVector(Int(batch1.x_corners[k]), Int(batch1.y_corners[k]))
        k2 = SVector(Int(batch2.x_corners[k]), Int(batch2.y_corners[k]))
        d = A_px * (k1 .- 1 .+ centre) + t_px - (k2 .- 1 .+ centre)
        offsets[1, k], offsets[2, k] = d[1], d[2]
        lit || continue
        _rois_overlap(A_px, d, box) || throw(
            ArgumentError(
                "pair $k: the registration maps ROI 1 (corner x = $(k1[1]), y = $(k1[2])) " *
                    "entirely outside ROI 2 (corner x = $(k2[1]), y = $(k2[2])): its " *
                    "centre lands $(round(d[1]; digits = 1)) px in x and " *
                    "$(round(d[2]; digits = 1)) px in y from ROI 2's centre, so no " *
                    "position is in both ROIs. Each ROI must contain its emitter's " *
                    "position in its image. batch1 must be the reference channel's ROIs, " *
                    "each batch's corners must place its ROIs on its own camera, and the " *
                    "registration must be the one between these cameras."
            )
        )
        (any(k1 .== 1) || any(k1 .== last1) || any(k2 .== 1) || any(k2 .== last2)) &&
            continue
        push!(dx, d[1])
        push!(dy, d[2])
        if abs(d[1]) > 1.5 * tol[1] || abs(d[2]) > 1.5 * tol[2]
            far += 1
            first_far == 0 && ((first_far, first_d) = (k, d))
        end
    end
    n = length(dx)
    advice = "ROIs each placed on the pixel that holds their emitter stay within " *
        "$(round(tol[1]; digits = 2)) px in x and $(round(tol[2]; digits = 2)) px in y. " *
        "Check the registration against these cameras, which batch is the reference " *
        "channel's, the order of the pairs, each batch's corners (a crop's origin) and " *
        "how the ROIs were placed. The pairs are fitted as given; " *
        "smld.metadata[\"pair_offsets_px\"] holds every pair's offset."
    if n >= JOINT_MEDIAN_MIN_PAIRS &&
            (abs(_median(dx)) > tol[1] || abs(_median(dy)) > tol[2])
        @warn "JointSplinePSFModel: the registration puts the centres of ROI 1 a median " *
            "$(round(_median(dx); digits = 2)) px in x and " *
            "$(round(_median(dy); digits = 2)) px in y from the centres of ROI 2, over " *
            "the $n pairs away from the sensor edges. " * advice
    elseif far > 0
        @warn "JointSplinePSFModel: in $far of the $n pairs away from the sensor edges " *
            "(the first is pair $first_far, at $(round(first_d[1]; digits = 2)) px in x " *
            "and $(round(first_d[2]; digits = 2)) px in y), the registration puts the " *
            "centre of ROI 1 more than $(round(1.5 * tol[1]; digits = 2)) px in x or " *
            "$(round(1.5 * tol[2]; digits = 2)) px in y from the centre of ROI 2. " * advice
    end
    return offsets
end

# Whether `view`'s table reaches every pixel of a box x box ROI for an emitter at ROI
# position (x, y), by the kernel's own test: `_spline_value_derivatives` returns zero where
# abs((pixel - position) * pixel size) exceeds the table's lateral radius on an axis.
function _view_covers(view::SplinePSFModel{T}, x::T, y::T, box::Int) where {T}
    rx, ry = view.pixel_psf.lateral_radius
    px, py = view.pixel_size
    return !(
        abs((1 - x) * px) > rx || abs((box - x) * px) > rx ||
            abs((1 - y) * py) > ry || abs((box - y) * py) > ry
    )
end

# A fit whose emitter, in an image that receives light, ends so far from its ROI's centre
# that some pixel of that ROI is beyond the image's table was fitted with a PSF cut off
# there (the table is zero beyond lateral_range plus about one pixel). Its uncertainties are
# set to Inf and its covariances to zero, as for a fit on a position bound, so the result is
# not mistaken for a localization. With a ROI centred on its emitter this does not happen
# (the half-diagonal check); it does for an emitter at a ROI's edge, as a ROI 2 placed a
# few pixels from where the registration puts the emitter gives. Each view's position is
# computed as the kernel computes it (`_joint_offset`, `_view_context`). An image with zero
# transmission has no PSF to cut off: its expected count is its background everywhere.
function _mark_uncovered_fits!(
        uncertainties::Matrix{Float32}, covariances::Matrix{Float32},
        results::Matrix{Float32}, model::JointSplinePSFModel{T},
        x_corners::AbstractMatrix, y_corners::AbstractMatrix, box::Int
    ) where {T}
    none = (nothing, nothing)
    for k in axes(results, 2)
        θ = SVector{6, T}(ntuple(i -> T(results[i, k]), 6))
        cx = (x_corners[1, k], x_corners[2, k])
        cy = (y_corners[1, k], y_corners[2, k])
        o = _joint_offset(model, cx, cy)
        covered = true
        for v in 1:2
            view, _, _, _, _, _, tr, θv = _view_context(model, v, θ, o, none, none, cx, cy)
            tr > zero(T) || continue
            covered &= _view_covers(view, θv[1], θv[2], box)
        end
        if !covered
            uncertainties[:, k] .= Inf32
            covariances[:, k] .= 0.0f0
        end
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
_table_arrays(model::JointSplinePSFModel) =
    (_table_arrays(model.views[1])..., _table_arrays(model.views[2])...)

# An image with zero transmission has no PSF in the fit (its expected count is its
# background), so its table need not reach the box.
function _validate_psf_box(model::JointSplinePSFModel{T}, box_size::Integer) where {T}
    for v in 1:2
        model.transmission[v] > zero(T) && _validate_psf_box(model.views[v], box_size)
    end
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

# L(θa) - L(θb) over both views (`_objective_difference`), one loop over the views as in
# `_lm_pass`.
@inline function _objective_difference(
        θa, θb, roi, box_size::Int, model::JointSplinePSFModel{T}, camera_model, variance_map,
        corner_x, corner_y
    ) where {T}
    o = _joint_offset(model, corner_x, corner_y)
    acc = (zero(T), zero(T), zero(T))
    for v in 1:2
        ca = _view_context(model, v, θa, o, roi, variance_map, corner_x, corner_y)
        θbv = _view_context(model, v, θb, o, roi, variance_map, corner_x, corner_y)[8]
        acc = _view_objective_difference(acc, box_size, camera_model, ca..., θbv)
    end
    return acc
end

@noinline function _view_objective_difference(
        acc, box_size::Int, camera_model, vmodel, r, vmap, cx, cy, A, tr, θav, θbv
    )
    @inbounds for j in 1:box_size, i in 1:box_size
        ua, _ = compute_pixel_derivatives(i, j, θav, vmodel)
        ub, _ = compute_pixel_derivatives(i, j, θbv, vmodel)
        w = _readout_variance(camera_model, ua, vmap, cx, cy, i, j)
        acc = _diff_accumulate(acc, r[i, j], ua, ub, w)
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

# The scan ranks z by the Poisson log-likelihood of the images that receive light, each
# pixel in the deviance form the fit itself climbs (`_fit_objective`): r log(μ) - μ less its
# value at μ = r, which is the same at every z. The plain sum carries the background's
# r log(r) - r in every pixel, which uses up the Float32 sum's digits and hides the
# differences between z; an image with zero transmission (μ its background at every z)
# would add nothing but that, so it is left out.
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
            vmodel, r, _, _, _, _, tr, θv = _view_context(
                model, v, θ, o, roi, roi, (0, 0), (0, 0)
            )
            if tr > zero(T)
                @inbounds for j in 1:box_size, i in 1:box_size
                    μ = evaluate_psf(vmodel, i, j, θv)
                    ll += _fit_objective(T(r[i, j]), μ, zero(T))
                end
            end
        end
        if ll > best_ll
            best_z, best_ll = z, ll
        end
    end
    return SVector{6, T}(x, y, best_z, N, bg1, bg2)
end

# As SplinePSFModel: the scan start, then `starts - 1` fixed z starts and, with at least one
# fixed start, the end of z_range that `_end_start` picks, keeping the fit with the strictly
# higher objective (the earlier start on a tie; the end start only by `_end_start_wins`).
# One `_lm_fit` call site, so the kernel holds one copy of the fit.
@inline function _optimize(
        roi, box_size::Int, psf_model::JointSplinePSFModel{T}, camera_model, variance_map,
        corner_x, corner_y, constraints, iterations::Int, θ0
    ) where {T}
    θ_best = θ0
    obj_best = T(-Inf)
    stopped_best = false
    m = psf_model.starts - 1
    for q in -1:m
        z = θ0[3]
        if q == m
            m >= 1 || break
            near_end, z = _end_start(psf_model, θ_best[3])
            near_end || break
        elseif q >= 0
            z = _spline_start_z(psf_model, q, m)
        end
        θs = Base.setindex(θ0, z, 3)
        θ, obj, _, stopped = _lm_fit(
            roi, box_size, psf_model, camera_model, variance_map, corner_x, corner_y,
            constraints, iterations, θs
        )
        better = if q < 0
            true
        elseif q < m
            obj > obj_best
        else
            _end_start_replaces(
                θ, obj, θ_best, obj_best, roi, box_size, psf_model, camera_model,
                variance_map, corner_x, corner_y
            )
        end
        if better
            θ_best = θ
            obj_best = obj
            stopped_best = stopped
        end
    end
    return θ_best, stopped_best
end
