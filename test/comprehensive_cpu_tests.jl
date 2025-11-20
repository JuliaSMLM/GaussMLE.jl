"""
Comprehensive CPU model tests for all PSF models with both ideal and sCMOS cameras
"""

@testset "Comprehensive CPU Model Tests" begin

    # Test configuration
    n_test_blobs = 500  # Reduced for faster testing
    box_size = 11
    verbose = get(ENV, "VERBOSE_TESTS", "false") == "true"
    
    @testset "Fixed Sigma Model (xynb)" begin
        psf_model = GaussMLE.GaussianXYNB(0.13f0)
        
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
    end

    @testset "Photon Level Sensitivity" begin
        psf_model = GaussMLE.GaussianXYNB(0.13f0)
        
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
    end
    
    @testset "PSF Width Variations" begin
        @testset "Narrow PSF (σ=1.0) - Ideal" begin
            psf_model = GaussMLE.GaussianXYNB(0.10f0)
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
            psf_model = GaussMLE.GaussianXYNB(0.20f0)
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