"""
Test SCMOSCamera and camera-based dispatch
"""

using GaussMLE
using SMLMData
using Test
using Random
using StaticArrays

@testset "SCMOSCamera and Camera Dispatch" begin
    
    @testset "SCMOSCamera Construction" begin
        # Create variance map
        nx, ny = 100, 100
        pixel_size = 0.1  # 100nm
        variance_map = ones(Float32, nx, ny) * 25.0f0  # 5 e- readout noise
        
        # Create SCMOSCamera
        scmos = SCMOSCamera(nx, ny, pixel_size, variance_map)
        
        @test scmos isa SMLMData.AbstractCamera
        @test scmos isa SCMOSCamera
        @test length(scmos.pixel_edges_x) == nx + 1
        @test length(scmos.pixel_edges_y) == ny + 1
        @test size(scmos.readnoise_variance) == (nx, ny)
        
        # Check pixel size
        @test scmos.pixel_edges_x[2] - scmos.pixel_edges_x[1] ≈ pixel_size
    end
    
    @testset "ROIBatch with Different Cameras" begin
        Random.seed!(42)
        
        # Create test data
        roi_size = 7
        n_rois = 3
        data = randn(Float32, roi_size, roi_size, n_rois)
        corners = Matrix{Int32}(undef, 2, n_rois)
        corners[1, :] = [10, 20, 30]
        corners[2, :] = [15, 25, 35]
        frame_indices = Int32[1, 1, 2]
        
        # Test with IdealCamera
        ideal_cam = SMLMData.IdealCamera(100, 100, 0.1)
        batch_ideal = ROIBatch(data, corners, frame_indices, ideal_cam)
        
        @test batch_ideal.camera isa SMLMData.IdealCamera
        @test length(batch_ideal) == n_rois
        
        # Test with SCMOSCamera
        variance_map = ones(Float32, 100, 100) * 10.0f0
        scmos_cam = SCMOSCamera(100, 100, 0.1, variance_map)
        batch_scmos = ROIBatch(data, corners, frame_indices, scmos_cam)
        
        @test batch_scmos.camera isa SCMOSCamera
        @test batch_scmos.camera.readnoise_variance[1,1] ≈ 10.0f0
    end
    
    @testset "Camera-based Dispatch" begin
        Random.seed!(42)
        
        # Generate simple test data
        roi_size = 7
        n_rois = 2
        data = ones(Float32, roi_size, roi_size, n_rois) * 100  # Bright uniform signal
        corners = Matrix{Int32}(undef, 2, n_rois)
        corners[1, :] = [10, 20]
        corners[2, :] = [10, 20]
        frame_indices = Int32[1, 1]
        
        # Create fitter
        psf_model = GaussMLE.GaussianXYNB(1.3f0)
        fitter = GaussMLEFitter(
            psf_model = psf_model,
            device = CPU(),
            iterations = 5  # Few iterations for test
        )
        
        # Test with IdealCamera
        ideal_cam = SMLMData.IdealCamera(100, 100, 0.1)
        batch_ideal = ROIBatch(data, corners, frame_indices, ideal_cam)
        
        results_ideal = fit(fitter, batch_ideal)
        @test results_ideal isa LocalizationResult
        @test results_ideal.n_fits == n_rois
        @test length(results_ideal.x_camera) == n_rois
        
        # Test with SCMOSCamera
        variance_map = ones(Float32, 100, 100) * 25.0f0
        scmos_cam = SCMOSCamera(100, 100, 0.1, variance_map)
        batch_scmos = ROIBatch(data, corners, frame_indices, scmos_cam)
        
        results_scmos = fit(fitter, batch_scmos)
        @test results_scmos isa LocalizationResult
        @test results_scmos.n_fits == n_rois
        
        # sCMOS should have larger uncertainties due to readout noise
        @test mean(results_scmos.uncertainties[1, :]) > mean(results_ideal.uncertainties[1, :])
    end
    
    @testset "Conversion to SMLMData" begin
        # Create simple results
        roi_size = 7
        n_rois = 2
        data = ones(Float32, roi_size, roi_size, n_rois)
        corners = Matrix{Int32}(undef, 2, n_rois)
        corners[1, :] = [10, 20]
        corners[2, :] = [15, 25]
        frame_indices = Int32[1, 2]
        
        # Use SCMOSCamera
        variance_map = ones(Float32, 100, 100) * 10.0f0
        scmos_cam = SCMOSCamera(100, 100, 0.1, variance_map)
        batch = ROIBatch(data, corners, frame_indices, scmos_cam)
        
        # Create fake results
        parameters = Float32[4.0 5.0; 4.0 5.0; 1000 1100; 10 11]
        uncertainties = ones(Float32, 4, 2) * 0.01f0
        log_likelihoods = Float32[-100, -110]
        
        result = create_localization_result(
            parameters, uncertainties, log_likelihoods,
            batch, GaussMLE.GaussianXYNB(1.3f0)
        )
        
        # Convert to SMLD using camera from batch
        smld = to_smld(result, batch)
        
        @test smld isa SMLMData.BasicSMLD
        @test length(smld) == 2
        @test smld.camera === scmos_cam  # Should use the same camera
        
        # Check coordinate conversion
        emitter = smld.emitters[1]
        # ROI coord 4.0 at corner 10 = camera pixel 13
        # In microns: (13 - 1) * 0.1 = 1.2
        @test emitter.x ≈ 1.3 atol=0.01
    end
end