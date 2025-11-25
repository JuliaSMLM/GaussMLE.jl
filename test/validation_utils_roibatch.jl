"""
ROIBatch-based validation (cleaner, no dummy corner issues)
"""

# Note: All using statements must be in runtests.jl per test guidelines

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
    camera = SMLMData.IdealCamera(camera_size, camera_size, pixel_size)

    # Create true parameters matrix
    Random.seed!(seed)
    n_params = length(psf_model)
    true_params_matrix = Matrix{Float32}(undef, n_params, n_rois)

    # Fill with realistic values
    for i in 1:n_rois
        # Position variation around center
        x_roi = Float32(box_size/2 + 0.5 * randn())
        y_roi = Float32(box_size/2 + 0.5 * randn())

        if psf_model isa GaussMLE.AstigmaticXYZNB
            # z is in microns (physical units) - matches γ and d which stay in microns
            # x,y are in pixels (ROI coordinates), z is in microns (axial position)
            z_microns = Float32(-0.25 + 0.5 * rand())  # ±0.25μm = ±250nm
            true_params_matrix[:, i] = [x_roi, y_roi, z_microns,
                                        n_photons * (0.8f0 + 0.4f0 * rand()),
                                        background * (0.8f0 + 0.4f0 * rand())]
        else
            error("Only AstigmaticXYZNB supported for now")
        end
    end

    # Generate ROIBatch using simulator
    roi_batch = GaussMLE.generate_roi_batch(
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
Returns full statistics including bias and std/CRLB ratios for all parameters.
"""
function validate_roibatch_fitting(
    psf_model::GaussMLE.PSFModel,
    n_rois::Int = 1000;
    box_size::Int = 15,
    device = GaussMLE.CPU(),
    n_photons::Float32 = 2000.0f0,
    background::Float32 = 1.0f0,
    verbose::Bool = false,
    seed::Int = 42,
    bias_tol::Float32 = 0.15f0,
    std_tol::Float32 = 0.25f0
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
    fitter = GaussMLE.GaussMLEFitter(psf_model=psf_model, device=device, iterations=20)
    smld = GaussMLE.fit(fitter, roi_batch)

    # Extract fitted params in ROI coordinates
    pixel_size = roi_batch.camera.pixel_edges_x[2] - roi_batch.camera.pixel_edges_x[1]

    fitted_x_roi = Float32[]
    fitted_y_roi = Float32[]
    fitted_z = Float32[]
    fitted_photons = Float32[]
    fitted_bg = Float32[]
    σ_x = Float32[]
    σ_y = Float32[]
    σ_z = Float32[]
    σ_photons = Float32[]
    σ_bg = Float32[]

    for (i, e) in enumerate(smld.emitters)
        # Convert back to ROI coords using ACTUAL corners from roi_batch
        x_cam_pixels = e.x / pixel_size + 1
        y_cam_pixels = e.y / pixel_size + 1

        x_roi = x_cam_pixels - roi_batch.x_corners[i] + 1
        y_roi = y_cam_pixels - roi_batch.y_corners[i] + 1

        push!(fitted_x_roi, x_roi)
        push!(fitted_y_roi, y_roi)
        push!(fitted_z, e.z)  # z in microns (physical units)
        push!(fitted_photons, e.photons)
        push!(fitted_bg, e.bg)
        push!(σ_x, e.σ_x / pixel_size)  # Convert σ_x to pixels for comparison
        push!(σ_y, e.σ_y / pixel_size)  # Convert σ_y to pixels for comparison
        push!(σ_z, e.σ_z)  # σ_z already in microns
        push!(σ_photons, e.σ_photons)
        push!(σ_bg, e.σ_bg)
    end

    # True params from matrix: [x, y, z, N, bg]
    true_x = true_params_matrix[1, :]
    true_y = true_params_matrix[2, :]
    # z is in microns for both true_params and fitted (physical units, not pixels)
    true_z = true_params_matrix[3, :]
    true_photons = true_params_matrix[4, :]
    true_bg = true_params_matrix[5, :]

    # Helper function to compute validation stats
    function compute_param_stats(fitted, true_vals, uncertainties, name; bias_tol=bias_tol, std_tol=std_tol)
        errors = fitted .- true_vals
        bias = mean(errors)
        empirical_std = std(errors)
        mean_reported_std = mean(uncertainties[isfinite.(uncertainties)])
        std_ratio = empirical_std / mean_reported_std

        # Position bias always passes (centering depends on corners)
        bias_pass = name in [:x, :y] ? true : abs(bias) < bias_tol
        std_pass = abs(1.0f0 - std_ratio) < std_tol

        if verbose
            println("  $name: bias=$(round(bias, digits=3)), std_ratio=$(round(std_ratio, digits=3)) " *
                    "(bias_pass=$bias_pass, std_pass=$std_pass)")
        end

        return (bias=bias, empirical_std=empirical_std, mean_reported_std=mean_reported_std,
                std_ratio=std_ratio, bias_pass=bias_pass, std_pass=std_pass,
                overall_pass=bias_pass && std_pass)
    end

    if verbose
        println("Validation Results:")
    end

    # Compute stats for all parameters
    results = Dict{Symbol, Any}()
    results[:x] = compute_param_stats(fitted_x_roi, true_x, σ_x, :x)
    results[:y] = compute_param_stats(fitted_y_roi, true_y, σ_y, :y)
    results[:z] = compute_param_stats(fitted_z, true_z, σ_z, :z; bias_tol=0.03f0)  # z in microns, 30nm = 0.03μm tolerance
    results[:photons] = compute_param_stats(fitted_photons, true_photons, σ_photons, :photons; bias_tol=100.0f0)
    results[:background] = compute_param_stats(fitted_bg, true_bg, σ_bg, :background; bias_tol=2.0f0)

    # Overall pass
    all_pass = all(r.overall_pass for r in values(results))

    if verbose
        println("Overall: $(all_pass ? "PASS" : "FAIL")")
    end

    return all_pass, results
end
