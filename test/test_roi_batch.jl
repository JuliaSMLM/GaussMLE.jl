"""
Test ROIBatch functionality and SMLMData integration
"""

using GaussMLE
using SMLMData
using Test
using Random
using Statistics
using StaticArrays

@testset "ROIBatch and SMLMData Integration" begin
    
    @testset "ROIBatch Construction" begin
        # Create test data
        roi_size = 7
        n_rois = 10
        data = randn(Float32, roi_size, roi_size, n_rois)
        corners = Matrix{Int32}(undef, 2, n_rois)
        corners[1, :] = collect(1:10:91)  # x corners
        corners[2, :] = collect(1:10:91)  # y corners
        frame_indices = Int32.(1:n_rois)
        
        # Create ROIBatch
        batch = ROIBatch(data, corners, frame_indices)
        
        @test length(batch) == n_rois
        @test batch.roi_size == roi_size
        @test size(batch.corners) == (2, n_rois)
        
        # Test indexing
        roi = batch[1]
        @test roi isa SingleROI
        @test size(roi.data) == (roi_size, roi_size)
        @test roi.corner == [1, 1]
        @test roi.frame_idx == 1
        
        # Test iteration
        count = 0
        for roi in batch
            count += 1
            @test roi isa SingleROI
        end
        @test count == n_rois
    end
    
    @testset "SingleROI to ROIBatch conversion" begin
        # Create single ROIs
        rois = [
            SingleROI(randn(Float32, 7, 7), SVector{2,Int32}(10, 20), Int32(1)),
            SingleROI(randn(Float32, 7, 7), SVector{2,Int32}(30, 40), Int32(2)),
            SingleROI(randn(Float32, 7, 7), SVector{2,Int32}(50, 60), Int32(3))
        ]
        
        # Convert to batch
        batch = ROIBatch(rois)
        
        @test length(batch) == 3
        @test batch.roi_size == 7
        @test batch.corners[:, 1] == [10, 20]
        @test batch.frame_indices[2] == 2
    end
    
    @testset "Coordinate Transformations" begin
        # Test ROI to camera coordinate conversion
        x_roi, y_roi = 3.5f0, 4.2f0
        x_corner, y_corner = Int32(100), Int32(200)
        
        x_cam, y_cam = roi_to_camera_coords(x_roi, y_roi, x_corner, y_corner)
        
        @test x_cam ≈ 102.5f0  # 100 + 3.5 - 1
        @test y_cam ≈ 203.2f0  # 200 + 4.2 - 1
    end
    
    @testset "Fitting with ROIBatch" begin
        Random.seed!(42)
        
        # Generate test data
        roi_size = 7
        n_rois = 5
        true_x_roi = 4.0f0
        true_y_roi = 4.0f0
        true_photons = 1000.0f0
        true_bg = 10.0f0
        sigma = 1.3f0
        
        # Create ROIs with different corners
        data = zeros(Float32, roi_size, roi_size, n_rois)
        corners = Matrix{Int32}(undef, 2, n_rois)
        frame_indices = Int32.(1:n_rois)
        
        for k in 1:n_rois
            # Set corners (spread across camera)
            corners[1, k] = 10 * k
            corners[2, k] = 20 * k
            
            # Generate Gaussian blob
            for j in 1:roi_size, i in 1:roi_size
                psf_x = GaussMLE.GaussLib.integral_gaussian_1d(i, true_x_roi, sigma)
                psf_y = GaussMLE.GaussLib.integral_gaussian_1d(j, true_y_roi, sigma)
                expected = true_bg + true_photons * psf_x * psf_y
                data[i, j, k] = rand() < expected ? 1.0f0 : 0.0f0
            end
        end
        
        batch = ROIBatch(data, corners, frame_indices)
        
        # Create fitter
        psf_model = GaussMLE.GaussianXYNB(sigma)
        fitter = GaussMLEFitter(
            psf_model = psf_model,
            device = CPU(),
            iterations = 20
        )
        
        # Fit without variance map (ideal camera)
        results = fit(fitter, batch)
        
        @test results isa LocalizationResult
        @test results.n_fits == n_rois
        @test length(results.x_camera) == n_rois
        @test length(results.y_camera) == n_rois
        
        # Check that camera coordinates are computed correctly
        for k in 1:n_rois
            x_roi_fit = results.parameters[1, k]
            y_roi_fit = results.parameters[2, k]
            
            expected_x_cam = corners[1, k] + x_roi_fit - 1
            expected_y_cam = corners[2, k] + y_roi_fit - 1
            
            @test results.x_camera[k] ≈ expected_x_cam atol=0.01
            @test results.y_camera[k] ≈ expected_y_cam atol=0.01
        end
    end
    
    @testset "sCMOS Variance Map Integration" begin
        Random.seed!(42)
        
        # Create simple test case
        roi_size = 7
        camera_size = 100
        
        # Single ROI at corner (10, 20)
        data = ones(Float32, roi_size, roi_size, 1)  # Need 3D array
        corners = Matrix{Int32}(undef, 2, 1)
        corners[:, 1] = [10, 20]
        frame_indices = Int32[1]
        batch = ROIBatch(data, corners, frame_indices)
        
        # Create variance map with pattern
        variance_map = zeros(Float32, camera_size, camera_size)
        # Set high variance in the ROI region
        for j in 20:26, i in 10:16
            variance_map[i, j] = 25.0f0  # 5 e- readout noise squared
        end
        
        # Create fitter
        fitter = GaussMLEFitter(
            psf_model = GaussMLE.GaussianXYNB(1.3f0),
            device = CPU(),
            iterations = 10
        )
        
        # This should work without error
        # The kernel will use the variance map at the correct camera positions
        @test_nowarn fit(fitter, batch, variance_map)
    end
    
    @testset "SMLMData Conversion" begin
        # Create test results
        n_fits = 3
        parameters = Float32[
            3.5 4.5 5.5;  # x in ROI
            4.0 5.0 6.0;  # y in ROI
            1000 1100 1200;  # photons
            10 11 12  # background
        ]
        uncertainties = Float32[
            0.01 0.01 0.01;
            0.01 0.01 0.01;
            50 55 60;
            1 1 1
        ]
        log_likelihoods = Float32[-100, -110, -120]
        
        # Create ROIBatch context
        corners = Matrix{Int32}(undef, 2, 3)
        corners[1, :] = [10, 30, 50]
        corners[2, :] = [20, 40, 60]
        frame_indices = Int32[1, 1, 2]
        data = zeros(Float32, 7, 7, 3)  # dummy data
        batch = ROIBatch(data, corners, frame_indices)
        
        # Create LocalizationResult
        result = create_localization_result(
            parameters, uncertainties, log_likelihoods,
            batch, GaussMLE.GaussianXYNB(1.3f0)
        )
        
        # Create SMLMData camera (100nm pixels)
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        
        # Convert to Emitter2DFit
        emitter = to_emitter2dfit(result, 1, camera)
        
        @test emitter isa SMLMData.Emitter2DFit
        @test emitter.frame == 1
        
        # Check coordinate conversion (pixel to microns)
        # ROI coord 3.5 at corner 10 = camera pixel 12.5
        # In microns: (12.5 - 1) * 0.1 = 1.15
        @test emitter.x ≈ 1.15 atol=0.001
        # ROI coord 4.0 at corner 20 = camera pixel 23.0
        # In microns: (23.0 - 1) * 0.1 = 2.2
        @test emitter.y ≈ 2.2 atol=0.001
        
        # Check other parameters
        @test emitter.photons ≈ 1000
        @test emitter.bg ≈ 10
        
        # Convert to full SMLD
        smld = to_smld(result, camera)
        
        @test smld isa SMLMData.BasicSMLD
        @test length(smld) == 3
        @test smld.n_frames == 2  # max frame index
        @test smld.camera === camera
    end
end