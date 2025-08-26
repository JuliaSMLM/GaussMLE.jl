"""
Test the new camera-aware simulator
"""

using GaussMLE
using SMLMData
using Test
using Random
using Statistics

@testset "Camera-Aware Simulator Tests" begin
    
    @testset "Basic Generation with IdealCamera" begin
        # Create camera and PSF model
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        # Generate with defaults
        batch = generate_roi_batch(camera, psf; n_rois=10, seed=42)
        
        @test batch isa ROIBatch
        @test length(batch) == 10
        @test batch.roi_size == 11  # Default
        @test batch.camera === camera
        
        # Check that data was generated
        @test size(batch.data) == (11, 11, 10)
        @test all(batch.data .>= 0)  # No negative values
        @test any(batch.data .> 0)   # Some signal
    end
    
    @testset "PSF-Specific Defaults" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        
        # Test each PSF model gets correct defaults
        psf_xynb = GaussMLE.GaussianXYNB(1.3f0)
        batch = generate_roi_batch(camera, psf_xynb; n_rois=1, xy_variation=0.0f0, seed=42)
        
        # Fit and check parameters are near defaults
        fitter = GaussMLE.GaussMLEFitter(psf_model=psf_xynb, device=GaussMLE.CPU(), iterations=20)
        results = GaussMLE.fit(fitter, batch)
        
        # Should be near center with default photons/background
        @test results.parameters[1, 1] ≈ 6.0 atol=1.0  # x
        @test results.parameters[2, 1] ≈ 6.0 atol=1.0  # y
        @test results.parameters[3, 1] ≈ 1000.0 rtol=0.3  # photons
        @test results.parameters[4, 1] ≈ 10.0 rtol=0.5  # background
    end
    
    @testset "SCMOSCamera Noise" begin
        # Create sCMOS camera with variance map
        variance_map = ones(Float32, 100, 100) * 25.0f0  # 5 e- readout noise
        scmos = SCMOSCamera(100, 100, 0.1f0, variance_map)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        # Generate data
        batch_scmos = generate_roi_batch(scmos, psf; n_rois=5, seed=42)
        
        @test batch_scmos.camera isa SCMOSCamera
        @test batch_scmos.camera.readnoise_variance[1,1] == 25.0f0
        
        # Generate same data with ideal camera for comparison
        ideal = SMLMData.IdealCamera(100, 100, 0.1)
        batch_ideal = generate_roi_batch(ideal, psf; n_rois=5, seed=42)
        
        # sCMOS should have more variation due to readout noise
        # (this is a statistical test, may occasionally fail)
        @test std(batch_scmos.data) > std(batch_ideal.data) * 0.9
    end
    
    @testset "Corner Generation Modes" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        # Test random corners
        batch_random = generate_roi_batch(camera, psf; 
                                         n_rois=10, corner_mode=:random, seed=42)
        corners_random = batch_random.corners
        
        # Test grid corners
        batch_grid = generate_roi_batch(camera, psf; 
                                       n_rois=9, corner_mode=:grid, seed=42)
        corners_grid = batch_grid.corners
        
        # Grid should be more regular (check x coordinates are sorted in each row)
        x_coords = sort(corners_grid[1, :])
        @test length(unique(diff(x_coords))) <= 3  # At most 3 different spacings
        
        # Test clustered corners
        batch_clustered = generate_roi_batch(camera, psf; 
                                            n_rois=20, corner_mode=:clustered, seed=42)
        @test size(batch_clustered.corners) == (2, 20)
    end
    
    @testset "Custom Parameters" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        # Provide custom parameters
        custom_params = Float32[
            5.0 7.0;  # x
            5.0 7.0;  # y
            2000.0 500.0;  # photons
            20.0 5.0   # background
        ]
        
        batch = generate_roi_batch(camera, psf; true_params=custom_params)
        
        @test length(batch) == 2  # Inferred from params
        
        # Fit and verify we recover the custom parameters
        fitter = GaussMLEFitter(psf_model=psf, device=CPU(), iterations=20)
        results = fit(fitter, batch)
        
        @test results.parameters[3, 1] ≈ 2000.0 rtol=0.3  # First ROI photons
        @test results.parameters[3, 2] ≈ 500.0 rtol=0.3   # Second ROI photons
    end
    
    @testset "Position Variation" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        # Generate with no variation
        batch_fixed = generate_roi_batch(camera, psf; 
                                        n_rois=10, xy_variation=0.0f0, seed=42)
        
        # Generate with variation
        batch_varied = generate_roi_batch(camera, psf; 
                                         n_rois=10, xy_variation=1.0f0, seed=43)
        
        # Fit both
        fitter = GaussMLEFitter(psf_model=psf, device=CPU(), iterations=20)
        results_fixed = fit(fitter, batch_fixed)
        results_varied = fit(fitter, batch_varied)
        
        # Fixed should all be very close to 6.0
        x_positions_fixed = results_fixed.parameters[1, :]
        @test std(x_positions_fixed) < 0.5
        @test mean(x_positions_fixed) ≈ 6.0 atol=0.5
        
        # Varied should have more spread
        x_positions_varied = results_varied.parameters[1, :]
        @test std(x_positions_varied) > 0.2  # Some variation
    end
    
    @testset "Frame Indices" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        # Custom frame indices
        frame_indices = Int32[1, 1, 2, 2, 3]
        batch = generate_roi_batch(camera, psf; 
                                  n_rois=5, frame_indices=frame_indices)
        
        @test batch.frame_indices == frame_indices
        
        # Default should be all frame 1
        batch_default = generate_roi_batch(camera, psf; n_rois=5)
        @test all(batch_default.frame_indices .== 1)
    end
end