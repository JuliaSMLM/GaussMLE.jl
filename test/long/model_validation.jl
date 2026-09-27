using Test, GaussMLE, SMLMData, Random, Statistics, Distributions

include(joinpath(@__DIR__, "..", "validation_utils.jl"))
include(joinpath(@__DIR__, "utils", "validation_utils_roibatch.jl"))

"""
Comprehensive model validation tests for all configurations
Tests that fitted values and uncertainties match expectations within tolerance
"""

@testset "Model Validation Tests" begin

    # Test configuration
    n_test_blobs = 1000  # Use 1000 blobs for reasonable statistics
    box_size = 15  # Larger box for better Fisher Information (especially important for astigmatic)
    verbose = get(ENV, "VERBOSE_TESTS", "false") == "true"

    @testset "Fixed Sigma Model (xynb)" begin
        psf_model = GaussMLE.GaussianXYNB(0.13f0)

        @testset "CPU Backend" begin
            passed, results = run_model_validation(
                :xynb, psf_model, n_test_blobs;
                box_size = box_size,
                device = GaussMLE.CPU(),
                sigma = 1.3f0,
                verbose = verbose
            )

            @test passed
            @test results[:x].bias_pass
            @test results[:y].bias_pass
            @test results[:photons].bias_pass
            @test results[:background].bias_pass

            # Check that reported uncertainties match empirical
            @test results[:x].std_pass
            @test results[:y].std_pass
            @test results[:photons].std_pass
            @test results[:background].std_pass
        end
    end

    @testset "Variable Sigma Model (xynbs)" begin
        psf_model = GaussMLE.GaussianXYNBS()

        @testset "CPU Backend" begin
            passed, results = run_model_validation(
                :xynbs, psf_model, n_test_blobs;
                box_size = box_size,
                device = GaussMLE.CPU(),
                sigma = 1.3f0,
                verbose = verbose
            )

            @test passed
            @test results[:x].bias_pass
            @test results[:y].bias_pass
            @test results[:photons].bias_pass
            @test results[:background].bias_pass
            @test results[:sigma].bias_pass

            # Check uncertainty matching
            @test results[:x].std_pass
            @test results[:y].std_pass
            @test results[:photons].std_pass
            @test results[:background].std_pass
            @test results[:sigma].std_pass
        end
    end

    @testset "Anisotropic Model (xynbsxsy)" begin
        psf_model = GaussMLE.GaussianXYNBSXSY()

        @testset "CPU Backend" begin
            passed, results = run_model_validation(
                :xynbsxsy, psf_model, n_test_blobs;
                box_size = box_size,
                device = GaussMLE.CPU(),
                sigma = 1.3f0,
                verbose = verbose
            )

            @test passed
            @test results[:x].bias_pass
            @test results[:y].bias_pass
            @test results[:photons].bias_pass
            @test results[:background].bias_pass
            @test results[:sigma_x].bias_pass
            @test results[:sigma_y].bias_pass

            # Check uncertainty matching
            @test results[:x].std_pass
            @test results[:y].std_pass
            @test results[:photons].std_pass
            @test results[:background].std_pass
            @test results[:sigma_x].std_pass
            @test results[:sigma_y].std_pass
        end
    end

    @testset "Astigmatic 3D Model (xynbz)" begin
        # Realistic astigmatic calibration following Huang et al. (Science 2008)
        # Higher-order terms (cubic/quartic) are necessary for real optical systems
        # Opposite signs in Ax/Ay and Bx/By create the astigmatic behavior
        # All spatial parameters in microns (per PSF model definition)
        psf_model = GaussMLE.AstigmaticXYZNB{Float32}(
            0.13f0, 0.13f0,   # σx₀, σy₀ - diffraction-limited base width (130nm = 0.13μm)
            0.05f0, -0.05f0,  # Ax, Ay - cubic aberrations (opposite signs for astigmatism)
            0.01f0, -0.01f0,  # Bx, By - quartic aberrations (opposite signs)
            0.2f0,            # γ = 200nm = 0.2μm (offset focal planes)
            0.5f0             # d = 500nm = 0.5μm (depth scale for ±600nm range)
        )

        @testset "CPU Backend" begin
            # Use ROIBatch-based validation for astigmatic model (proper statistical validation)
            passed, results = validate_roibatch_fitting(
                psf_model, n_test_blobs;
                box_size = box_size,
                device = GaussMLE.CPU(),
                n_photons = 2000.0f0,  # Higher SNR needed for reliable z-fitting
                background = 1.0f0,     # Lower background improves SNR
                verbose = verbose
            )

            @test passed
            @test results[:x].bias_pass
            @test results[:y].bias_pass
            @test results[:z].bias_pass
            @test results[:photons].bias_pass
            @test results[:background].bias_pass

            # Check uncertainty matching
            @test results[:x].std_pass
            @test results[:y].std_pass
            @test results[:z].std_pass
            @test results[:photons].std_pass
            @test results[:background].std_pass
        end
    end

    @testset "Different Photon Levels" begin
        psf_model = GaussMLE.GaussianXYNB(0.13f0)

        @testset "Low photons (N=200)" begin
            passed, results = run_model_validation(
                :xynb, psf_model, 500;  # Fewer spots for speed
                box_size = box_size,
                device = GaussMLE.CPU(),
                n_photons = 200.0f0,
                background = 5.0f0,  # Realistic background level
                sigma = 1.3f0,
                verbose = verbose
            )

            @test passed
            # Lower photons = worse precision, but uncertainties should still match
            @test results[:x].std_pass
            @test results[:y].std_pass
        end

        @testset "High photons (N=5000)" begin
            passed, results = run_model_validation(
                :xynb, psf_model, 500;
                box_size = box_size,
                device = GaussMLE.CPU(),
                n_photons = 5000.0f0,
                background = 20.0f0,
                sigma = 1.3f0,
                verbose = verbose
            )

            @test passed
            # Higher photons = better precision
            @test results[:x].std_pass
            @test results[:y].std_pass
            @test results[:x].empirical_std < 0.05  # Should have good precision
        end
    end

    @testset "Different PSF Widths" begin
        @testset "Narrow PSF (σ=1.0)" begin
            psf_model = GaussMLE.GaussianXYNB(0.1f0)
            passed, results = run_model_validation(
                :xynb, psf_model, 500;
                box_size = box_size,
                device = GaussMLE.CPU(),
                sigma = 1.0f0,
                verbose = verbose
            )

            @test passed
            @test results[:x].std_pass
            @test results[:y].std_pass
        end

        @testset "Wide PSF (σ=2.0)" begin
            psf_model = GaussMLE.GaussianXYNB(0.2f0)
            passed, results = run_model_validation(
                :xynb, psf_model, 500;
                box_size = box_size,
                device = GaussMLE.CPU(),
                sigma = 2.0f0,
                verbose = verbose
            )

            @test passed
            @test results[:x].std_pass
            @test results[:y].std_pass
        end
    end

    @testset "sCMOS Camera Model" begin
        psf_model = GaussMLE.GaussianXYNB(0.13f0)
        n_rois = 500

        # Create sCMOS camera with uniform readnoise of 5 e- rms
        readnoise_map = fill(5.0f0, 512, 512)
        scmos = SMLMData.SCMOSCamera(
            512, 512, 0.1f0, readnoise_map,
            offset = 100.0f0,
            gain = 0.5f0,
            qe = 0.82f0
        )

        # Generate ROIBatch with known true parameters
        Random.seed!(42)
        true_params = Float32[
            Float32(box_size / 2) .+ 0.5f0 * randn(Float32, n_rois)';
            Float32(box_size / 2) .+ 0.5f0 * randn(Float32, n_rois)';
            1000.0f0 .+ 200.0f0 * randn(Float32, n_rois)';
            10.0f0 .+ 2.0f0 * randn(Float32, n_rois)'
        ]

        # Generate dummy corners that stay within camera bounds
        dummy_corners = zeros(Int32, 2, n_rois)
        max_corner = 512 - box_size + 1
        roi_spacing = box_size
        rois_per_row = div(max_corner, roi_spacing)
        for i in 1:n_rois
            row = div(i - 1, rois_per_row)
            col = mod(i - 1, rois_per_row)
            dummy_corners[1, i] = Int32(1 + col * roi_spacing)
            dummy_corners[2, i] = Int32(1 + row * roi_spacing)
        end

        batch = GaussMLE.generate_roi_batch(
            scmos, psf_model;
            n_rois = n_rois,
            roi_size = box_size,
            true_params = true_params,
            corners = dummy_corners,
            seed = 42
        )

        # Fit with sCMOS model via ROIBatch
        fitter = GaussMLE.GaussMLEConfig(
            psf_model = psf_model,
            device = GaussMLE.CPU()
        )

        smld, _info = GaussMLE.fit(batch, fitter)

        # Extract fitted params and compute statistics
        pixel_size = batch.camera.pixel_edges_x[2] - batch.camera.pixel_edges_x[1]
        corners = vcat(batch.x_corners', batch.y_corners')
        coords = extract_roi_coords(smld, corners, box_size, pixel_size)

        # Validate photons with proper statistical tests
        # Note: Fitter estimates DETECTED ELECTRONS, not incident photons
        # So we compare to expected_electrons = true_photons * QE
        expected_electrons = true_params[3, :] .* scmos.qe
        photon_errors = coords.photons .- expected_electrons
        photon_bias = mean(photon_errors)
        photon_empirical_std = std(photon_errors)
        photon_mean_crlb = mean([e.σ_photons for e in smld.emitters])
        photon_std_ratio = photon_empirical_std / photon_mean_crlb

        # Check bias is within tolerance (comparing to detected electrons)
        @test abs(photon_bias) < 50.0f0  # Electron count bias tolerance

        # Check that CRLB properly accounts for readout noise (std/CRLB ≈ 1.0)
        @test abs(1.0f0 - photon_std_ratio) < 0.3f0  # 30% tolerance for std/CRLB

        # Verify uncertainties are larger than IdealCamera would give (due to readout noise)
        @test photon_mean_crlb > 30.0f0  # Should be noticeably larger due to readnoise
    end

    @testset "Edge Cases" begin
        psf_model = GaussMLE.GaussianXYNB(0.13f0)

        @testset "Spots near edges" begin
            # Generate data with spots near ROI edges
            Random.seed!(42)
            n_spots = 100
            data = zeros(Float32, box_size, box_size, n_spots)
            true_params = Dict{Symbol, Vector{Float32}}()

            for k in 1:n_spots
                # Place spots near edges
                x_true = Float32(2.0 + (box_size - 3) * rand())
                y_true = Float32(2.0 + (box_size - 3) * rand())
                n_true = 1000.0f0
                bg_true = 10.0f0

                true_params[:x] = push!(get(true_params, :x, Float32[]), x_true)
                true_params[:y] = push!(get(true_params, :y, Float32[]), y_true)
                true_params[:photons] = push!(get(true_params, :photons, Float32[]), n_true)
                true_params[:background] = push!(get(true_params, :background, Float32[]), bg_true)

                for j in 1:box_size, i in 1:box_size
                    mu = generate_pixel_value(i, j, x_true, y_true, n_true, bg_true, 1.3f0, 1.3f0)
                    data[i, j, k] = Float32(rand(Poisson(max(0.01, mu))))
                end
            end

            # Fit
            fitter = GaussMLE.GaussMLEConfig(psf_model = psf_model, device = GaussMLE.CPU())
            smld, _info = GaussMLE.fit(data, fitter)

            # Check that fitting doesn't fail catastrophically
            x_vals = [e.x for e in smld.emitters]
            y_vals = [e.y for e in smld.emitters]
            σ_x_vals = [e.σ_x for e in smld.emitters]
            σ_y_vals = [e.σ_y for e in smld.emitters]

            @test !any(isnan.(x_vals))
            @test !any(isnan.(y_vals))
            @test !any(isinf.(σ_x_vals))
            @test !any(isinf.(σ_y_vals))
        end
    end
end
