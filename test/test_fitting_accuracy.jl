"""
Test fitting accuracy using the new camera-aware simulator
"""

using GaussMLE
using SMLMData
using Test
using Random
using Statistics

@testset "Fitting Accuracy Tests" begin
    
    @testset "PSF Models with IdealCamera" begin
        # Test each PSF model
        camera = SMLMData.IdealCamera(256, 256, 0.1)
        
        psf_models = [
            ("GaussianXYNB", GaussianXYNB(1.3f0), 4),
            ("GaussianXYNBS", GaussianXYNBS(), 5),
            ("GaussianXYNBSXSY", GaussianXYNBSXSY(), 6),
        ]
        
        for (name, psf, n_params) in psf_models
            @testset "$name" begin
                # Generate test data with known parameters
                batch = generate_roi_batch(camera, psf; 
                                          n_rois=100, 
                                          xy_variation=0.5f0,
                                          seed=42)
                
                # Create fitter
                fitter = GaussMLEFitter(
                    psf_model=psf,
                    device=CPU(),
                    iterations=20
                )
                
                # Fit
                results = fit(fitter, batch)
                
                # Check basic properties
                @test results isa LocalizationResult
                @test results.n_fits == 100
                @test size(results.parameters) == (n_params, 100)
                
                # Check parameter recovery (should be near defaults)
                mean_x = mean(results.parameters[1, :])
                mean_y = mean(results.parameters[2, :])
                mean_photons = mean(results.parameters[3, :])
                mean_bg = mean(results.parameters[4, :])
                
                @test mean_x ≈ 6.0 atol=0.5
                @test mean_y ≈ 6.0 atol=0.5
                @test mean_photons ≈ 1000.0 rtol=0.2
                @test mean_bg ≈ 10.0 rtol=0.3
                
                # Check uncertainties are reasonable
                mean_x_err = mean(results.uncertainties[1, :])
                mean_y_err = mean(results.uncertainties[2, :])
                
                @test mean_x_err > 0.02 && mean_x_err < 0.2
                @test mean_y_err > 0.02 && mean_y_err < 0.2
                
                # No infinite uncertainties
                @test !any(isinf.(results.uncertainties))
                @test !any(isnan.(results.uncertainties))
            end
        end
    end
    
    @testset "sCMOS Camera Noise Model" begin
        # Create cameras
        variance_map = ones(Float32, 256, 256) * 25.0f0
        scmos = SCMOSCamera(256, 256, 0.1f0, variance_map)
        ideal = SMLMData.IdealCamera(256, 256, 0.1)
        
        psf = GaussianXYNB(1.3f0)
        
        # Generate data with same parameters but different cameras
        batch_ideal = generate_roi_batch(ideal, psf; n_rois=50, seed=42)
        batch_scmos = generate_roi_batch(scmos, psf; n_rois=50, seed=42)
        
        # Fit both
        fitter = GaussMLEFitter(psf_model=psf, device=CPU(), iterations=20)
        results_ideal = fit(fitter, batch_ideal)
        results_scmos = fit(fitter, batch_scmos)
        
        # sCMOS should have larger uncertainties
        mean_err_ideal = mean(results_ideal.uncertainties[1, :])
        mean_err_scmos = mean(results_scmos.uncertainties[1, :])
        
        @test mean_err_scmos > mean_err_ideal
        
        # But both should recover parameters reasonably
        @test mean(results_ideal.parameters[3, :]) ≈ 1000.0 rtol=0.3
        @test mean(results_scmos.parameters[3, :]) ≈ 1000.0 rtol=0.4
    end
    
    @testset "Photon Level Sensitivity" begin
        camera = SMLMData.IdealCamera(256, 256, 0.1)
        psf = GaussianXYNB(1.3f0)
        fitter = GaussMLEFitter(psf_model=psf, device=CPU(), iterations=20)
        
        # Test different photon levels
        photon_levels = [200.0f0, 1000.0f0, 5000.0f0]
        
        for photons in photon_levels
            @testset "N=$photons photons" begin
                # Custom parameters with specific photon count
                params = Float32[6.0 6.0 photons 10.0]'
                batch = generate_roi_batch(camera, psf; 
                                         true_params=repeat(params, 1, 20),
                                         seed=42)
                
                results = fit(fitter, batch)
                
                # Check photon recovery
                mean_photons = mean(results.parameters[3, :])
                @test mean_photons ≈ photons rtol=0.3
                
                # Higher photons = better precision
                mean_x_err = mean(results.uncertainties[1, :])
                if photons == 200.0f0
                    @test mean_x_err > 0.08  # Lower precision
                elseif photons == 5000.0f0
                    @test mean_x_err < 0.04  # Higher precision
                end
            end
        end
    end
    
    @testset "Edge Cases" begin
        camera = SMLMData.IdealCamera(256, 256, 0.1)
        psf = GaussianXYNB(1.3f0)
        fitter = GaussMLEFitter(psf_model=psf, device=CPU(), iterations=20)
        
        @testset "ROIs near edges" begin
            # Generate ROIs with positions near edges
            edge_params = Float32[
                2.0 2.0 1000.0 10.0;  # Near top-left
                9.0 9.0 1000.0 10.0;  # Near bottom-right
                2.0 9.0 1000.0 10.0;  # Near bottom-left
                9.0 2.0 1000.0 10.0;  # Near top-right
            ]'
            
            batch = generate_roi_batch(camera, psf; 
                                      true_params=edge_params,
                                      seed=42)
            
            results = fit(fitter, batch)
            
            # Should still fit reasonably
            @test all(results.parameters[3, :] .> 500)  # Photons recovered
            @test !any(isinf.(results.uncertainties))
        end
        
        @testset "Very dim ROIs" begin
            # Very few photons
            dim_params = Float32[6.0 6.0 50.0 5.0]'
            batch = generate_roi_batch(camera, psf; 
                                      true_params=repeat(dim_params, 1, 10),
                                      seed=42)
            
            results = fit(fitter, batch)
            
            # Should complete without errors
            @test results.n_fits == 10
            # But uncertainties will be large
            @test mean(results.uncertainties[1, :]) > 0.1
        end
    end
    
    @testset "Coordinate Transformations" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussianXYNB(1.3f0)
        
        # Generate with known corners
        corners = Int32[100 200; 150 250]'
        batch = generate_roi_batch(camera, psf; 
                                 n_rois=2,
                                 corners=corners,
                                 xy_variation=0.0f0,
                                 seed=42)
        
        fitter = GaussMLEFitter(psf_model=psf, device=CPU(), iterations=20)
        results = fit(fitter, batch)
        
        # Check coordinate transformation
        # ROI center at 6.0 + corner = camera position
        @test results.x_camera[1] ≈ 105.0 atol=1.0  # 100 + 6 - 1
        @test results.y_camera[1] ≈ 155.0 atol=1.0  # 150 + 6 - 1
        
        # Convert to SMLD and check physical coordinates
        smld = to_smld(results, batch)
        
        # Physical = (camera - 1) * pixel_size
        @test smld.emitters[1].x ≈ (results.x_camera[1] - 1) * 0.1 atol=0.1
        @test smld.emitters[1].y ≈ (results.y_camera[1] - 1) * 0.1 atol=0.1
    end
end