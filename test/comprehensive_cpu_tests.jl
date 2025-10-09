"""
Comprehensive CPU model tests for all PSF models with both ideal and sCMOS cameras
"""

@testset "Comprehensive CPU Model Tests" begin
    
    # Test configuration
    n_test_blobs = 500  # Reduced for faster testing
    box_size = 11
    verbose = get(ENV, "VERBOSE_TESTS", "false") == "true"
    
    # Create sCMOS variance map (readout noise)
    variance_map = ones(Float32, box_size, box_size) * 25.0f0  # 5 e- readout noise
    scmos_camera = GaussMLE.SCMOSCameraInternal(variance_map)
    
    @testset "Fixed Sigma Model (xynb)" begin
        psf_model = GaussMLE.GaussianXYNB(1.3f0)
        
        @testset "Ideal Camera" begin
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
            @test results[:x].std_pass
            @test results[:y].std_pass
            @test results[:photons].std_pass
            @test results[:background].std_pass
        end
        
        @testset "sCMOS Camera" begin
            # Generate data with sCMOS noise
            data, true_params = generate_test_data(:xynb, n_test_blobs, box_size; sigma=1.3f0)
            
            # Add readout noise
            for k in 1:size(data, 3)
                data[:, :, k] .+= randn(Float32, box_size, box_size) .* sqrt.(variance_map)
            end
            
            # Fit with sCMOS model
            fitter = GaussMLE.GaussMLEFitter(
                psf_model = psf_model,
                camera_model = scmos_camera,
                device = GaussMLE.CPU(),
                iterations = 20
            )
            
            results = GaussMLE.fit(fitter, data)
            
            # Validate with looser tolerances for sCMOS
            x_result = validate_fitting_results(
                results, true_params, :x;
                bias_tol = 0.1f0,
                std_tol = 0.35f0,  # More tolerance for sCMOS
                verbose = verbose
            )
            
            y_result = validate_fitting_results(
                results, true_params, :y;
                bias_tol = 0.1f0,
                std_tol = 0.35f0,
                verbose = verbose
            )
            
            @test x_result.bias_pass
            @test y_result.bias_pass
            # sCMOS uncertainties are expected to be different but should still be reasonable
            @test x_result.mean_reported_std > 0.03f0 && x_result.mean_reported_std < 0.15f0
            @test y_result.mean_reported_std > 0.03f0 && y_result.mean_reported_std < 0.15f0
        end
    end
    
    @testset "Variable Sigma Model (xynbs)" begin
        psf_model = GaussMLE.GaussianXYNBS()
        
        @testset "Ideal Camera" begin
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
            @test results[:x].std_pass
            @test results[:y].std_pass
            @test results[:photons].std_pass
            @test results[:background].std_pass
            @test results[:sigma].std_pass
        end
        
        @testset "sCMOS Camera" begin
            # Generate data with sCMOS noise
            data, true_params = generate_test_data(:xynbs, n_test_blobs, box_size; sigma=1.3f0)
            
            # Add readout noise
            for k in 1:size(data, 3)
                data[:, :, k] .+= randn(Float32, box_size, box_size) .* sqrt.(variance_map)
            end
            
            # Fit with sCMOS model
            fitter = GaussMLE.GaussMLEFitter(
                psf_model = psf_model,
                camera_model = scmos_camera,
                device = GaussMLE.CPU(),
                iterations = 20
            )
            
            results = GaussMLE.fit(fitter, data)
            
            # Validate key parameters
            x_result = validate_fitting_results(
                results, true_params, :x;
                bias_tol = 0.1f0,
                std_tol = 0.35f0,
                verbose = verbose
            )
            
            @test x_result.bias_pass
            @test x_result.mean_reported_std > 0.03f0 && x_result.mean_reported_std < 0.15f0
        end
    end
    
    @testset "Anisotropic Model (xynbsxsy)" begin
        psf_model = GaussMLE.GaussianXYNBSXSY()
        
        @testset "Ideal Camera" begin
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
            @test results[:x].std_pass
            @test results[:y].std_pass
            @test results[:photons].std_pass
            @test results[:background].std_pass
            @test results[:sigma_x].std_pass
            @test results[:sigma_y].std_pass
        end
        
        @testset "sCMOS Camera" begin
            # Generate data with sCMOS noise
            data, true_params = generate_test_data(:xynbsxsy, n_test_blobs, box_size; sigma=1.3f0)
            
            # Add readout noise
            for k in 1:size(data, 3)
                data[:, :, k] .+= randn(Float32, box_size, box_size) .* sqrt.(variance_map)
            end
            
            # Fit with sCMOS model
            fitter = GaussMLE.GaussMLEFitter(
                psf_model = psf_model,
                camera_model = scmos_camera,
                device = GaussMLE.CPU(),
                iterations = 20
            )
            
            results = GaussMLE.fit(fitter, data)
            
            # Validate key parameters
            x_result = validate_fitting_results(
                results, true_params, :x;
                bias_tol = 0.1f0,
                std_tol = 0.35f0,
                verbose = verbose
            )
            
            @test x_result.bias_pass
            @test x_result.mean_reported_std > 0.03f0 && x_result.mean_reported_std < 0.15f0
        end
    end
    
    @testset "Photon Level Sensitivity" begin
        psf_model = GaussMLE.GaussianXYNB(1.3f0)
        
        @testset "Low photons (N=200) - Ideal" begin
            passed, results = run_model_validation(
                :xynb, psf_model, 200;
                box_size = box_size,
                device = GaussMLE.CPU(),
                n_photons = 200.0f0,
                background = 5.0f0,
                sigma = 1.3f0,
                verbose = verbose
            )
            
            @test passed
            # Lower photons = worse precision
            @test results[:x].empirical_std > 0.08f0  # Should have worse precision
            @test results[:x].std_pass
            @test results[:y].std_pass
        end
        
        @testset "High photons (N=5000) - Ideal" begin
            # Use more samples for better statistics at high SNR
            passed, results = run_model_validation(
                :xynb, psf_model, 500;  # Increased from 200 for better statistics
                box_size = box_size,
                device = GaussMLE.CPU(),
                n_photons = 5000.0f0,
                background = 20.0f0,
                sigma = 1.3f0,
                verbose = verbose
            )

            @test passed
            # Higher photons = better precision
            @test results[:x].empirical_std < 0.03f0  # Should have excellent precision
            @test results[:x].std_pass
            @test results[:y].std_pass
        end
        
        @testset "Low photons (N=200) - sCMOS" begin
            # Generate data with low photons
            data, true_params = generate_test_data(:xynb, 200, box_size; 
                                                  n_photons=200.0f0, 
                                                  background=5.0f0, 
                                                  sigma=1.3f0)
            
            # Add readout noise
            for k in 1:size(data, 3)
                data[:, :, k] .+= randn(Float32, box_size, box_size) .* sqrt.(variance_map)
            end
            
            # Fit with sCMOS model
            fitter = GaussMLE.GaussMLEFitter(
                psf_model = psf_model,
                camera_model = scmos_camera,
                device = GaussMLE.CPU(),
                iterations = 20
            )
            
            results = GaussMLE.fit(fitter, data)
            
            # With low photons and readout noise, precision should be poor
            x_result = validate_fitting_results(
                results, true_params, :x;
                bias_tol = 0.2f0,  # More tolerance for low SNR
                std_tol = 0.5f0,
                verbose = verbose
            )
            
            @test x_result.bias_pass
            @test x_result.empirical_std > 0.1f0  # Should have poor precision
        end
    end
    
    @testset "PSF Width Variations" begin
        @testset "Narrow PSF (σ=1.0) - Ideal" begin
            psf_model = GaussMLE.GaussianXYNB(1.0f0)
            passed, results = run_model_validation(
                :xynb, psf_model, 200;
                box_size = box_size,
                device = GaussMLE.CPU(),
                sigma = 1.0f0,
                verbose = verbose
            )
            
            @test passed
            @test results[:x].std_pass
            @test results[:y].std_pass
            # Narrower PSF should give better localization
            @test results[:x].empirical_std < 0.06f0
        end
        
        @testset "Wide PSF (σ=2.0) - Ideal" begin
            psf_model = GaussMLE.GaussianXYNB(2.0f0)
            passed, results = run_model_validation(
                :xynb, psf_model, 200;
                box_size = box_size,
                device = GaussMLE.CPU(),
                sigma = 2.0f0,
                verbose = verbose
            )
            
            @test passed
            @test results[:x].std_pass
            @test results[:y].std_pass
            # Wider PSF should have worse localization
            @test results[:x].empirical_std > 0.07f0
        end
    end
end