"""
Consolidated test of new simulator and ROIBatch features
"""

@testset "New Features Tests" begin
    
    @testset "Basic Simulator with IdealCamera" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        batch = GaussMLE.generate_roi_batch(camera, psf; n_rois=10, seed=42)
        
        @test batch isa GaussMLE.ROIBatch
        @test length(batch) == 10
        @test batch.roi_size == 11
        @test batch.camera === camera
        @test size(batch.data) == (11, 11, 10)
    end
    
    @testset "SCMOSCamera Support" begin
        # Create sCMOS camera
        variance_map = ones(Float32, 256, 256) * 25.0f0
        scmos = GaussMLE.SCMOSCamera(256, 256, 0.1f0, variance_map)
        
        @test scmos isa SMLMData.AbstractCamera
        @test scmos.readnoise_variance[1,1] == 25.0f0
        
        # Generate data with sCMOS
        psf = GaussMLE.GaussianXYNB(1.3f0)
        batch = GaussMLE.generate_roi_batch(scmos, psf; n_rois=5, seed=42)
        
        @test batch.camera === scmos
    end
    
    @testset "Fitting with ROIBatch" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        batch = GaussMLE.generate_roi_batch(camera, psf; n_rois=20, seed=42)
        
        fitter = GaussMLE.GaussMLEFitter(
            psf_model = psf,
            device = GaussMLE.CPU(),
            iterations = 20
        )
        
        results = GaussMLE.fit(fitter, batch)
        
        @test results isa GaussMLE.LocalizationResult
        @test results.n_fits == 20
        @test size(results.parameters) == (4, 20)
        @test size(results.uncertainties) == (4, 20)
        
        # Check coordinate transformations work
        @test length(results.x_camera) == 20
        @test length(results.y_camera) == 20
        
        # Check reasonable results
        @test mean(results.parameters[3, :]) ≈ 1000.0 rtol=0.5
        @test all(0.02 .< results.uncertainties[1, :] .< 0.2)
    end
    
    @testset "SMLMData Conversion" begin
        camera = SMLMData.IdealCamera(256, 256, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        # Known corners for testing
        corners = Matrix{Int32}(Int32[10 20; 30 40]')
        batch = GaussMLE.generate_roi_batch(camera, psf; 
                                           n_rois=2, 
                                           corners=corners,
                                           xy_variation=0.0f0,
                                           seed=42)
        
        fitter = GaussMLE.GaussMLEFitter(psf_model=psf, device=GaussMLE.CPU(), iterations=20)
        results = GaussMLE.fit(fitter, batch)
        
        # Convert to SMLD
        smld = GaussMLE.to_smld(results, batch)
        
        @test smld isa SMLMData.BasicSMLD
        @test length(smld) == 2
        @test smld.camera === camera
        
        # Check coordinate conversion
        # ROI position ~6 + corner - 1 = camera position
        # corners[1,1] = 10, corners[2,1] = 30 (x,y for first ROI)
        @test results.x_camera[1] ≈ 15.0 atol=2.0  # 10 + 6 - 1
        @test results.y_camera[1] ≈ 25.0 atol=2.0  # 20 + 6 - 1 (second is y, not 30)
        
        # Physical = (camera - 1) * pixel_size
        @test smld.emitters[1].x ≈ (results.x_camera[1] - 1) * 0.1 atol=0.2
    end
    
    @testset "Different PSF Models" begin
        camera = SMLMData.IdealCamera(256, 256, 0.1)
        
        psf_models = [
            GaussMLE.GaussianXYNB(1.3f0),
            GaussMLE.GaussianXYNBS(),
            GaussMLE.GaussianXYNBSXSY()
        ]
        
        for psf in psf_models
            batch = GaussMLE.generate_roi_batch(camera, psf; n_rois=5, seed=42)
            fitter = GaussMLE.GaussMLEFitter(psf_model=psf, device=GaussMLE.CPU(), iterations=20)
            results = GaussMLE.fit(fitter, batch)
            
            @test results.n_fits == 5
            @test !any(isinf.(results.uncertainties))
            @test !any(isnan.(results.uncertainties))
        end
    end
end