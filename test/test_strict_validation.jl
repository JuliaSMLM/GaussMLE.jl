"""
Strict validation tests for fitting accuracy, bias, and CRLB matching
Using the new camera-aware simulator for reliable test data generation
"""

@testset "Strict Validation Tests" begin
    
    # Helper function to validate fitting results
    function validate_fits(results::GaussMLE.LocalizationResult, 
                           expected_params::Matrix{Float32};
                           param_idx::Int,
                           bias_tol::Float32 = 0.1f0,
                           std_ratio_tol::Float32 = 0.25f0,
                           verbose::Bool = false)
        
        fitted = results.parameters[param_idx, :]
        expected = expected_params[param_idx, :]
        uncertainties = results.uncertainties[param_idx, :]
        
        # Calculate bias
        errors = fitted .- expected
        bias = mean(errors)
        empirical_std = std(errors)
        mean_reported_std = mean(uncertainties)
        std_ratio = empirical_std / mean_reported_std
        
        # Tests
        bias_pass = abs(bias) < bias_tol
        std_pass = abs(1.0f0 - std_ratio) < std_ratio_tol
        
        if verbose
            param_names = ["x", "y", "photons", "background", "sigma", "sigma_x", "sigma_y", "z"]
            println("\nParameter: $(param_names[min(param_idx, length(param_names))])")
            println("  Bias: $(round(bias, digits=4)) (tolerance: ±$bias_tol)")
            println("  Empirical STD: $(round(empirical_std, digits=4))")
            println("  Mean reported STD: $(round(mean_reported_std, digits=4))")
            println("  STD ratio: $(round(std_ratio, digits=3)) (should be ≈1.0)")
            println("  Bias test: $(bias_pass ? "PASS" : "FAIL")")
            println("  STD test: $(std_pass ? "PASS" : "FAIL")")
        end
        
        return (bias=bias, empirical_std=empirical_std, 
                mean_reported_std=mean_reported_std, std_ratio=std_ratio,
                bias_pass=bias_pass, std_pass=std_pass)
    end
    
    @testset "GaussianXYNB - Standard Conditions" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        n_rois = 500

        Random.seed!(42)
        true_params = Float32[
            6.0 .+ 0.5f0 * randn(Float32, n_rois)';
            6.0 .+ 0.5f0 * randn(Float32, n_rois)';
            1000.0 .+ 200.0f0 * randn(Float32, n_rois)';
            10.0 .+ 2.0f0 * randn(Float32, n_rois)'
        ]
        
        batch = GaussMLE.generate_roi_batch(camera, psf; 
                                           n_rois=n_rois,
                                           true_params=true_params,
                                           seed=42)
        
        # Fit
        fitter = GaussMLE.GaussMLEFitter(psf_model=psf, device=GaussMLE.CPU(), iterations=20)
        results = GaussMLE.fit(fitter, batch)
        
        # Validate each parameter
        verbose = get(ENV, "VERBOSE_TESTS", "false") == "true"
        
        x_val = validate_fits(results, true_params, param_idx=1, bias_tol=0.1f0, verbose=verbose)
        y_val = validate_fits(results, true_params, param_idx=2, bias_tol=0.1f0, verbose=verbose)
        n_val = validate_fits(results, true_params, param_idx=3, bias_tol=50.0f0, verbose=verbose)
        b_val = validate_fits(results, true_params, param_idx=4, bias_tol=2.0f0, verbose=verbose)
        
        @test x_val.bias_pass
        @test y_val.bias_pass
        @test n_val.bias_pass
        @test b_val.bias_pass
        
        @test x_val.std_pass
        @test y_val.std_pass
        @test n_val.std_pass
        @test b_val.std_pass
    end
    
    @testset "Low SNR Conditions" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        n_rois = 200

        Random.seed!(43)
        true_params = Float32[
            6.0 .+ 0.3f0 * randn(Float32, n_rois)';
            6.0 .+ 0.3f0 * randn(Float32, n_rois)';
            200.0 .+ 50.0f0 * randn(Float32, n_rois)';
            20.0 .+ 5.0f0 * randn(Float32, n_rois)'
        ]
        
        batch = GaussMLE.generate_roi_batch(camera, psf; 
                                           n_rois=n_rois,
                                           true_params=true_params,
                                           seed=43)
        
        fitter = GaussMLE.GaussMLEFitter(psf_model=psf, device=GaussMLE.CPU(), iterations=20)
        results = GaussMLE.fit(fitter, batch)
        
        # More relaxed tolerances for low SNR
        x_val = validate_fits(results, true_params, param_idx=1, bias_tol=0.2f0, std_ratio_tol=0.35f0)
        
        @test x_val.bias_pass
        @test x_val.std_pass
        
        # Check that uncertainties are appropriately larger
        @test x_val.mean_reported_std > 0.08f0
    end
    
    @testset "Edge Position Tests" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        n_rois = 100
        
        # Generate ROIs near edges
        edge_positions = Float32[]
        for i in 1:n_rois
            if i <= 25
                push!(edge_positions, 2.5f0 + 0.3f0 * randn(Float32))  # Near left/top
            elseif i <= 50  
                push!(edge_positions, 8.5f0 + 0.3f0 * randn(Float32))  # Near right/bottom
            else
                push!(edge_positions, 6.0f0 + 0.3f0 * randn(Float32))  # Center
            end
        end
        
        true_params = Float32[
            edge_positions';
            edge_positions';
            1000.0f0 * ones(Float32, n_rois)';
            10.0f0 * ones(Float32, n_rois)'
        ]
        
        batch = GaussMLE.generate_roi_batch(camera, psf; 
                                           n_rois=n_rois,
                                           true_params=true_params,
                                           seed=44)
        
        fitter = GaussMLE.GaussMLEFitter(psf_model=psf, device=GaussMLE.CPU(), iterations=20)
        results = GaussMLE.fit(fitter, batch)
        
        # Check convergence - no infinite uncertainties
        @test !any(isinf.(results.uncertainties))
        @test !any(isnan.(results.uncertainties))
        
        # Separate validation for edge vs center
        edge_idx = vcat(1:50)
        center_idx = 51:100
        
        edge_bias = mean(results.parameters[1, edge_idx] .- true_params[1, edge_idx])
        center_bias = mean(results.parameters[1, center_idx] .- true_params[1, center_idx])
        
        @test abs(edge_bias) < 0.2f0
        @test abs(center_bias) < 0.1f0
    end
    
    @testset "sCMOS Camera with Variance Map" begin
        # Create sCMOS with spatially varying noise
        variance_map = Float32[
            10.0f0 + 40.0f0 * exp(-((i-128)^2 + (j-128)^2) / 5000.0f0)
            for i in 1:256, j in 1:256
        ]
        
        scmos = GaussMLE.SCMOSCamera(256, 256, 0.1f0, variance_map)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        n_rois = 300

        Random.seed!(45)
        true_params = Float32[
            6.0 .+ 0.5f0 * randn(Float32, n_rois)';
            6.0 .+ 0.5f0 * randn(Float32, n_rois)';
            1000.0 .+ 200.0f0 * randn(Float32, n_rois)';
            10.0 .+ 2.0f0 * randn(Float32, n_rois)'
        ]
        
        batch = GaussMLE.generate_roi_batch(scmos, psf; 
                                           n_rois=n_rois,
                                           true_params=true_params,
                                           seed=45)
        
        fitter = GaussMLE.GaussMLEFitter(psf_model=psf, device=GaussMLE.CPU(), iterations=20)
        results = GaussMLE.fit(fitter, batch)
        
        # sCMOS should still meet specifications, but with slightly relaxed tolerances
        x_val = validate_fits(results, true_params, param_idx=1, 
                             bias_tol=0.15f0, std_ratio_tol=0.30f0)
        
        @test x_val.bias_pass
        @test x_val.std_pass
        
        # Uncertainties should be larger than ideal camera
        @test x_val.mean_reported_std > 0.05f0
    end
    
    @testset "Different PSF Models" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        n_rois = 200

        Random.seed!(46)
        psf_nbs = GaussMLE.GaussianXYNBS()
        true_params_nbs = Float32[
            6.0 .+ 0.5f0 * randn(Float32, n_rois)';
            6.0 .+ 0.5f0 * randn(Float32, n_rois)';
            1000.0 .+ 200.0f0 * randn(Float32, n_rois)';
            10.0 .+ 2.0f0 * randn(Float32, n_rois)';
            1.3f0 .+ 0.2f0 * randn(Float32, n_rois)'
        ]
        
        batch_nbs = GaussMLE.generate_roi_batch(camera, psf_nbs; 
                                               n_rois=n_rois,
                                               true_params=true_params_nbs,
                                               seed=46)
        
        fitter_nbs = GaussMLE.GaussMLEFitter(psf_model=psf_nbs, device=GaussMLE.CPU(), iterations=20)
        results_nbs = GaussMLE.fit(fitter_nbs, batch_nbs)
        
        # Validate sigma parameter (more challenging than position/photons)
        sigma_val = validate_fits(results_nbs, true_params_nbs, param_idx=5,
                                  bias_tol=0.15f0, std_ratio_tol=0.60f0)

        @test sigma_val.bias_pass
        @test sigma_val.std_pass  # Known issue: can be flaky due to sigma parameter estimation difficulty

        Random.seed!(47)
        psf_sxsy = GaussMLE.GaussianXYNBSXSY()
        true_params_sxsy = Float32[
            6.0 .+ 0.5f0 * randn(Float32, n_rois)';
            6.0 .+ 0.5f0 * randn(Float32, n_rois)';
            1000.0 .+ 200.0f0 * randn(Float32, n_rois)';
            10.0 .+ 2.0f0 * randn(Float32, n_rois)';
            1.3f0 .+ 0.15f0 * randn(Float32, n_rois)';
            1.3f0 .+ 0.15f0 * randn(Float32, n_rois)'
        ]
        
        batch_sxsy = GaussMLE.generate_roi_batch(camera, psf_sxsy; 
                                                n_rois=n_rois,
                                                true_params=true_params_sxsy,
                                                seed=47)
        
        fitter_sxsy = GaussMLE.GaussMLEFitter(psf_model=psf_sxsy, device=GaussMLE.CPU(), iterations=20)
        results_sxsy = GaussMLE.fit(fitter_sxsy, batch_sxsy)
        
        @test !any(isinf.(results_sxsy.uncertainties))
        
        # Basic validation for anisotropic model
        x_val_sxsy = validate_fits(results_sxsy, true_params_sxsy, param_idx=1, 
                                   bias_tol=0.15f0, std_ratio_tol=0.35f0)
        @test x_val_sxsy.bias_pass
    end
    
    @testset "Photon Level Sensitivity" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        photon_levels = [100.0f0, 500.0f0, 2000.0f0, 10000.0f0]
        # Just check that precision improves with photon count (qualitative test)
        # Skip exact values since they depend on PSF model, pixel size, etc.
        
        precision_values = Float32[]
        
        for photons in photon_levels
            n_rois = 100
            true_params = Float32[
                6.0f0 * ones(Float32, n_rois)';  # Fixed position
                6.0f0 * ones(Float32, n_rois)';
                photons * ones(Float32, n_rois)';
                10.0f0 * ones(Float32, n_rois)'
            ]
            
            batch = GaussMLE.generate_roi_batch(camera, psf; 
                                               n_rois=n_rois,
                                               true_params=true_params,
                                               seed=48)
            
            fitter = GaussMLE.GaussMLEFitter(psf_model=psf, device=GaussMLE.CPU(), iterations=20)
            results = GaussMLE.fit(fitter, batch)
            
            mean_σ_x = mean(results.uncertainties[1, :])
            push!(precision_values, mean_σ_x)
            
            # Verify CRLB is being calculated correctly
            empirical_std = std(results.parameters[1, :] .- true_params[1, :])
            @test empirical_std ≈ mean_σ_x rtol=0.6  # Allow more tolerance
        end
        
        # Check that precision improves with photon count
        @test precision_values[1] > precision_values[2] > precision_values[3] > precision_values[4]
        @test precision_values[1] < 0.5  # Reasonable upper bound
        @test precision_values[4] < 0.05  # Good precision at high photon count
    end
end