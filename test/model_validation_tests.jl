"""
Comprehensive model validation tests for all configurations
Tests that fitted values and uncertainties match expectations within tolerance
"""

@testset "Model Validation Tests" begin
    
    # Test configuration
    n_test_spots = 1000  # Use 1000 spots for reasonable statistics
    box_size = 11
    verbose = get(ENV, "VERBOSE_TESTS", "false") == "true"
    
    @testset "Fixed Sigma Model (xynb)" begin
        psf_model = GaussMLE.GaussianXYNB(1.3f0)
        
        @testset "CPU Backend" begin
            passed, results = run_model_validation(
                :xynb, psf_model, n_test_spots;
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
                :xynbs, psf_model, n_test_spots;
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
                :xynbsxsy, psf_model, n_test_spots;
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
        # Simple calibration for testing
        psf_model = GaussMLE.AstigmaticXYZNB{Float32}(
            1.3f0, 1.3f0,  # σx₀, σy₀
            0.0f0, 0.0f0,  # Ax, Ay
            0.0f0, 0.0f0,  # Bx, By
            0.0f0,         # γ
            500.0f0        # d
        )
        
        @testset "CPU Backend" begin
            passed, results = run_model_validation(
                :xynbz, psf_model, n_test_spots;
                box_size = box_size,
                device = GaussMLE.CPU(),
                sigma = 1.3f0,
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
        psf_model = GaussMLE.GaussianXYNB(1.3f0)
        
        @testset "Low photons (N=200)" begin
            passed, results = run_model_validation(
                :xynb, psf_model, 500;  # Fewer spots for speed
                box_size = box_size,
                device = GaussMLE.CPU(),
                n_photons = 200.0f0,
                background = 5.0f0,
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
            psf_model = GaussMLE.GaussianXYNB(1.0f0)
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
            psf_model = GaussMLE.GaussianXYNB(2.0f0)
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
        psf_model = GaussMLE.GaussianXYNB(1.3f0)
        
        # Create variance map (readout noise)
        variance_map = ones(Float32, box_size, box_size) * 25.0f0  # 5 e- readout noise
        camera_model = GaussMLE.SCMOSCamera(variance_map)
        
        # Generate test data with sCMOS noise
        data, true_params = generate_test_data(:xynb, 500, box_size; sigma=1.3f0)
        
        # Add readout noise
        for k in 1:size(data, 3)
            data[:, :, k] .+= randn(Float32, box_size, box_size) .* sqrt(25.0f0)
        end
        
        # Fit with sCMOS model
        fitter = GaussMLE.GaussMLEFitter(
            psf_model = psf_model,
            camera_model = camera_model,
            device = GaussMLE.CPU()
        )
        
        results = GaussMLE.fit(fitter, data)
        
        # Validate
        x_result = validate_fitting_results(
            results, true_params, :x;
            bias_tol = 0.1f0,
            std_tol = 0.3f0,
            verbose = verbose
        )
        
        @test x_result.overall_pass
        @test x_result.std_pass  # Uncertainties should account for readout noise
    end
    
    @testset "Edge Cases" begin
        psf_model = GaussMLE.GaussianXYNB(1.3f0)
        
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
            fitter = GaussMLE.GaussMLEFitter(psf_model = psf_model, device = GaussMLE.CPU())
            results = GaussMLE.fit(fitter, data)
            
            # Check that fitting doesn't fail catastrophically
            @test !any(isnan.(results.x))
            @test !any(isnan.(results.y))
            @test !any(isinf.(results.x_error))
            @test !any(isinf.(results.y_error))
        end
    end
end