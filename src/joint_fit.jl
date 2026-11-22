# fit for JointSplinePSFModel: one emitter in two images, from a pair of ROIBatches.

# A JointSplinePSFModel fits a pair of batches; every single-batch or array fit refuses it.
_joint_only() = throw(
    ArgumentError(
        "a JointSplinePSFModel fits a pair of ROIBatches, one per view: " *
            "fit((batch1, batch2), fitter)"
    )
)
to_pixel_units(::JointSplinePSFModel, args...) = _joint_only()
_model_for_camera(::JointSplinePSFModel, camera) = _joint_only()
_array_fit_pixel_size(::JointSplinePSFModel) = _joint_only()

# Images per fit a model expects; the kernel runner checks it against its data.
_model_views(::PSFModel) = 1
_model_views(::JointSplinePSFModel) = 2

# The camera kinds a pair can have; both batches must have the same one.
_joint_camera_kind(::SMLMData.SCMOSCamera) = Val(true)
_joint_camera_kind(::SMLMData.IdealCamera) = Val(false)
_joint_camera_kind(camera) = throw(ArgumentError("unsupported camera type $(typeof(camera))"))

# View `view`'s data for the kernel, `(data, variance_map)`: raw counts for an IdealCamera
# (Poisson; the map is unused), electrons and the read-noise variance map for an
# SCMOSCamera (the ROIs lie on the camera: `_validate_pair_geometry`). The gain must be
# finite and positive (a zero gain makes every count zero), and the counts the kernel gets
# must be finite: a non-finite offset, or a count beyond Float32, would otherwise come back
# as a fit.
function _joint_view_data(batch::ROIBatch, view::Int)
    data = batch.data
    validate_fit_input(data)
    box = size(data, 1)
    camera = batch.camera
    if camera isa SMLMData.SCMOSCamera
        all(g -> isfinite(g) && g > zero(g), camera.gain) || throw(
            ArgumentError(
                "view $view: the camera gain must be finite and positive (e-/ADU) at " *
                    "every pixel"
            )
        )
        variance_map = extract_variance_map(camera, Float32)
        electrons = to_electrons(data, camera, batch.x_corners, batch.y_corners)
        counts = convert(Array{Float32, 3}, electrons)
    else
        variance_map = zeros(Float32, box, box)
        counts = convert(Array{Float32, 3}, data)
    end
    k = findfirst(!isfinite, counts)
    k === nothing || throw(
        ArgumentError(
            "view $view: ROI $(k[3]) has the count $(counts[k]) at pixel ($(k[1]), " *
                "$(k[2])) after the camera calibration and conversion to Float32; the " *
                "data and the camera's offset and gain must give finite counts"
        )
    )
    return counts, variance_map
end

"""
    fit((batch1, batch2)::Tuple{ROIBatch, ROIBatch}, fitter::GaussMLEConfig) ->
        (BasicSMLD, GaussMLEFitInfo)

Fit one emitter to two images at once with a [`JointSplinePSFModel`](@ref) (the only model
that takes a pair). `batch1` holds the reference view's ROIs and `batch2` the other view's,
pair `k` being the same emitter in the same frame: equal numbers of ROIs, equal square ROI
sizes and equal frame indices, else `ArgumentError`. `batch1` is the channel
`mc.reference` of the model's `MultiChannelPSF`, whichever index it has. Each batch's
corners place its ROIs on its own camera, whose calibration applies to its pixels; both
cameras must be of one type (`IdealCamera` or `SCMOSCamera`) with evenly spaced pixel
edges. The corners enter the fit, so every ROI must lie on its camera and the pairs must
follow the model's registration (see [`JointSplinePSFModel`](@ref) for the checks); a fit
whose PSF table does not reach every pixel of its ROIs reports infinite uncertainties.

Returns a `BasicSMLD` of [`Emitter3DFitJoint`](@ref) on `batch1`'s camera, positions in its
microns, and the fit information (`n_fits` counts pairs).
"""
function fit(batches::Tuple{ROIBatch, ROIBatch}, fitter::GaussMLEConfig)
    t0 = time_ns()
    model = fitter.psf_model
    model isa JointSplinePSFModel || throw(
        ArgumentError(
            "a pair of ROIBatches is fitted with a JointSplinePSFModel, got $(typeof(model))"
        )
    )
    batch1, batch2 = batches
    n1, n2 = size(batch1.data, 3), size(batch2.data, 3)
    n1 == n2 || throw(ArgumentError("the two batches hold $n1 and $n2 ROIs; pair them one to one"))
    size(batch1.data, 1) == size(batch2.data, 1) || throw(
        ArgumentError(
            "the two batches' ROIs are $(size(batch1.data, 1)) and " *
                "$(size(batch2.data, 1)) pixels; use one ROI size"
        )
    )
    batch1.frame_indices == batch2.frame_indices || throw(
        ArgumentError("the two batches' frame indices differ; pair ROIs of the same frame")
    )

    scmos1 = _joint_camera_kind(batch1.camera)
    scmos1 === _joint_camera_kind(batch2.camera) || throw(
        ArgumentError(
            "the two batches' cameras are $(nameof(typeof(batch1.camera))) and " *
                "$(nameof(typeof(batch2.camera))); use one camera type"
        )
    )
    _validate_pair_geometry(model, batch1, batch2)
    data1, map1 = _joint_view_data(batch1, 1)
    data2, map2 = _joint_view_data(batch2, 2)

    n_fits = n1
    box_size = size(data1, 1)
    n_params = length(model)
    data = Array{Float32, 4}(undef, box_size, box_size, 2, n_fits)
    data[:, :, 1, :] .= data1
    data[:, :, 2, :] .= data2
    x_corners = Matrix{Int32}(undef, 2, n_fits)
    y_corners = Matrix{Int32}(undef, 2, n_fits)
    x_corners[1, :] .= batch1.x_corners
    x_corners[2, :] .= batch2.x_corners
    y_corners[1, :] .= batch1.y_corners
    y_corners[2, :] .= batch2.y_corners
    variance_maps = (map1, map2)

    psf_pixels = _model_for_cameras(model, batch1.camera, batch2.camera)

    results = Matrix{Float32}(undef, n_params, n_fits)
    uncertainties = Matrix{Float32}(undef, n_params, n_fits)
    covariances = Matrix{Float32}(undef, 3, n_fits)  # [σ_xy, σ_xz, σ_yz]
    chi2 = Vector{Float32}(undef, n_fits)
    converged = Vector{UInt8}(undef, n_fits)

    actual_backend, device_id, actual_batch_size, actual_n_batches = _run_mle_kernel!(
        results, uncertainties, covariances, chi2,
        data, psf_pixels, scmos1, variance_maps, x_corners, y_corners,
        fitter; converged
    )

    _mark_uncovered_fits!(
        uncertainties, covariances, results, psf_pixels, x_corners, y_corners, box_size
    )

    # Pearson statistic over both images' pixels
    pvalues = _gof_pvalues(chi2, 2 * box_size^2 - n_params)
    loc_result = create_localization_result(
        results, uncertainties, covariances, pvalues, batch1, model
    )

    elapsed_s = (time_ns() - t0) / 1.0e9
    memory_per_batch = estimate_batch_memory(
        actual_batch_size, box_size, n_params;
        map_bytes = _map_bytes(variance_maps), views = 2
    ) + _psf_table_bytes(psf_pixels)
    info = GaussMLEFitInfo(
        elapsed_s, actual_backend, device_id, n_fits, count(!iszero, converged),
        actual_batch_size, actual_n_batches, memory_per_batch
    )
    return (to_smld(loc_result, batch1), info)
end

"""
    fit((batch1, batch2)::Tuple{ROIBatch, ROIBatch}; psf_model, iterations, backend = :auto,
        ...) -> (BasicSMLD, GaussMLEFitInfo)

Keyword form of the pair fit: builds a `GaussMLEConfig` from the keywords, which match its
fields; `psf_model` (a [`JointSplinePSFModel`](@ref)) is required.
"""
function fit(
        batches::Tuple{ROIBatch, ROIBatch};
        psf_model,
        iterations = _default_iterations(psf_model),
        backend = :auto,
        constraints = nothing,
        batch_size = 10_000,
        auto_timeout = 300.0,
        gpu_timeout = Inf,
        on_wait = nothing
    )
    fitter = GaussMLEConfig(;
        psf_model, backend, iterations, constraints, batch_size, auto_timeout,
        gpu_timeout, on_wait
    )
    return fit(batches, fitter)
end

function to_emitter(
        ::JointSplinePSFModel{T},
        result::LocalizationResult{T},
        idx::Int,
        camera::SMLMData.AbstractCamera;
        dataset::Int = 1,
        track_id::Int = 0,
        id::Int = idx
    ) where {T}
    e = _to_emitter_xyznb(result, idx, camera; dataset, track_id, id)
    return Emitter3DFitJoint{T}(
        e.x, e.y, e.z, e.photons, e.bg, e.σ_x, e.σ_y, e.σ_z, e.σ_xy, e.σ_xz, e.σ_yz,
        e.σ_photons, e.σ_bg, e.pvalue, e.frame, e.dataset, e.track_id, e.id,
        result.parameters[6, idx], result.uncertainties[6, idx]
    )
end
