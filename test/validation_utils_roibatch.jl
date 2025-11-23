"""
ROIBatch-based validation (cleaner, no dummy corner issues)
"""

using Random
using Statistics
using SMLMData
using GaussMLE

"""
Generate test data as ROIBatch using the simulator (guarantees consistent corners)
"""
function generate_roi_batch_validation(
    psf_model::GaussMLE.PSFModel,
    n_rois::Int;
    box_size::Int = 15,
    n_photons::Float32 = 2000.0f0,
    background::Float32 = 1.0f0,
    seed::Int = 42
)
    # Create camera (100nm pixels for test)
    pixel_size = 0.1f0  # 100nm pixels
    camera_size = 512
    camera = IdealCamera(camera_size, camera_size, pixel_size)

    # Create true parameters matrix
    Random.seed!(seed)
    n_params = length(psf_model)
    true_params_matrix = Matrix{Float32}(undef, n_params, n_rois)

    # Fill with realistic values
    for i in 1:n_rois
        # Position variation around center
        x_roi = Float32(box_size/2 + 0.5 * randn())
        y_roi = Float32(box_size/2 + 0.5 * randn())

        if psf_model isa AstigmaticXYZNB
            z_val = Float32(-250 + 500 * rand())  # ±250nm
            true_params_matrix[:, i] = [x_roi, y_roi, z_val,
                                        n_photons * (0.8f0 + 0.4f0 * rand()),
                                        background * (0.8f0 + 0.4f0 * rand())]
        else
            error("Only AstigmaticXYZNB supported for now")
        end
    end

    # Generate ROIBatch using simulator
    roi_batch = generate_roi_batch(
        camera, psf_model;
        n_rois = n_rois,
        roi_size = box_size,
        true_params = true_params_matrix,
        seed = seed
    )

    return roi_batch, true_params_matrix
end

"""
Validate using ROIBatch (corners are consistent by construction)
"""
function validate_roibatch_fitting(
    psf_model::GaussMLE.PSFModel,
    n_rois::Int = 1000;
    box_size::Int = 15,
    device = GaussMLE.CPU(),
    n_photons::Float32 = 2000.0f0,
    background::Float32 = 1.0f0,
    verbose::Bool = false,
    seed::Int = 42
)
    # Generate ROIBatch with known true params
    roi_batch, true_params_matrix = generate_roi_batch_validation(
        psf_model, n_rois;
        box_size = box_size,
        n_photons = n_photons,
        background = background,
        seed = seed
    )

    # Fit
    fitter = GaussMLEFitter(psf_model=psf_model, device=device, iterations=20)
    smld = fit(fitter, roi_batch)

    # Extract fitted params in ROI coordinates (no dummy corners needed!)
    pixel_size = roi_batch.camera.pixel_edges_x[2] - roi_batch.camera.pixel_edges_x[1]

    fitted_x_roi = Float32[]
    fitted_y_roi = Float32[]
    fitted_z = Float32[]

    for (i, e) in enumerate(smld.emitters)
        # Convert back to ROI coords using ACTUAL corners from roi_batch
        x_cam_pixels = e.x / pixel_size + 1
        y_cam_pixels = e.y / pixel_size + 1

        x_roi = x_cam_pixels - roi_batch.x_corners[i] + 1
        y_roi = y_cam_pixels - roi_batch.y_corners[i] + 1

        push!(fitted_x_roi, x_roi)
        push!(fitted_y_roi, y_roi)
        push!(fitted_z, e.z / pixel_size)  # z in pixels
    end

    # True params from matrix
    true_x = true_params_matrix[1, :]
    true_y = true_params_matrix[2, :]
    true_z = true_params_matrix[3, :]

    # Compute biases
    bias_x = mean(fitted_x_roi - true_x)
    bias_y = mean(fitted_y_roi - true_y)
    bias_z = mean(fitted_z - true_z)

    if verbose
        println("Validation Results:")
        println("  X: bias=$(round(bias_x, digits=3))")
        println("  Y: bias=$(round(bias_y, digits=3))")
        println("  Z: bias=$(round(bias_z, digits=3))")
    end

    return (bias_x=bias_x, bias_y=bias_y, bias_z=bias_z,
            fitted_x=fitted_x_roi, fitted_y=fitted_y_roi, fitted_z=fitted_z,
            true_x=true_x, true_y=true_y, true_z=true_z)
end
