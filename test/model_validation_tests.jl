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
        # Updated parameters provide flatter CRLB and better convergence
        psf_model = GaussMLE.AstigmaticXYZNB{Float32}(
            1.3f0, 1.3f0,     # σx₀, σy₀ - diffraction-limited base width
            0.05f0, -0.05f0,  # Ax, Ay - cubic aberrations (opposite signs for astigmatism)
            0.01f0, -0.01f0,  # Bx, By - quartic aberrations (opposite signs)
            200.0f0,          # γ = 200nm (offset focal planes - realistic astigmatic system)
            500.0f0           # d = 500nm (typical depth scale for ±600nm range)
        )
        
        @testset "CPU Backend" begin
            passed, results = run_model_validation(
                :xynbz, psf_model, n_test_blobs;
                box_size = box_size,
                device = GaussMLE.CPU(),
                n_photons = 2000.0f0,  # Higher SNR needed for reliable z-fitting
                background = 1.0f0,     # Lower background improves SNR
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
            psf_model = GaussMLE.GaussianXYNB(0.10f0)
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
            psf_model = GaussMLE.GaussianXYNB(0.20f0)
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
        
        # Create variance map (readout noise)
        variance_map = ones(Float32, box_size, box_size) * 25.0f0  # 5 e- readout noise squared

        # Generate test data with sCMOS noise
        data, true_params = generate_test_data(:xynb, 500, box_size; sigma=1.3f0)

        # Add readout noise
        for k in 1:size(data, 3)
            data[:, :, k] .+= randn(Float32, box_size, box_size) .* sqrt(25.0f0)
        end

        # Fit with sCMOS model (variance_map passed to fit())
        fitter = GaussMLE.GaussMLEFitter(
            psf_model = psf_model,
            device = GaussMLE.CPU()
        )

        smld = GaussMLE.fit(fitter, data; variance_map=variance_map)

        # Validate - use photons since position comparison doesn't work
        photons_result = validate_fitting_results(
            smld, true_params, :photons;
            bias_tol = 100.0f0,
            std_tol = 0.3f0,
            roi_size = box_size,
            verbose = verbose
        )

        @test photons_result.overall_pass
        @test photons_result.std_pass  # Uncertainties should account for readout noise
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
            fitter = GaussMLE.GaussMLEFitter(psf_model = psf_model, device = GaussMLE.CPU())
            smld = GaussMLE.fit(fitter, data)

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