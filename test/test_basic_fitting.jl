"""
Basic fitting tests using the new camera-aware simulator
"""

@testset "Basic Fitting Tests" begin
    
    @testset "IdealCamera with GaussianXYNB" begin
        # Create camera and PSF model
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        # Generate ROI batch
        batch = GaussMLE.generate_roi_batch(camera, psf; n_rois=50, seed=42)
        
        # Create fitter and fit
        fitter = GaussMLE.GaussMLEFitter(psf_model=psf, device=GaussMLE.CPU(), iterations=20)
        results = GaussMLE.fit(fitter, batch)
        
        # Basic checks
        @test results isa GaussMLE.LocalizationResult
        @test results.n_fits == 50
        @test size(results.parameters) == (4, 50)
        
        # Check parameter recovery
        @test mean(results.parameters[1, :]) ≈ 6.0 atol=1.0
        @test mean(results.parameters[2, :]) ≈ 6.0 atol=1.0
        @test mean(results.parameters[3, :]) ≈ 1000.0 rtol=0.3
        @test mean(results.parameters[4, :]) ≈ 10.0 rtol=0.5
        
        # Check uncertainties are reasonable
        @test all(0.03 .< results.uncertainties[1, :] .< 0.15)
        @test all(0.03 .< results.uncertainties[2, :] .< 0.15)
        @test !any(isinf.(results.uncertainties))
        @test !any(isnan.(results.uncertainties))
    end
    
    @testset "SCMOSCamera with GaussianXYNB" begin
        # Create sCMOS camera with variance map
        variance_map = ones(Float32, 256, 256) * 25.0f0
        scmos = GaussMLE.SCMOSCamera(256, 256, 0.1f0, variance_map)
        psf = GaussMLE.GaussianXYNB(1.3f0)
        
        # Generate ROI batch
        batch = GaussMLE.generate_roi_batch(scmos, psf; n_rois=50, seed=42)
        
        # Create fitter and fit
        fitter = GaussMLE.GaussMLEFitter(psf_model=psf, device=GaussMLE.CPU(), iterations=20)
        results = GaussMLE.fit(fitter, batch)
        
        # Basic checks
        @test results isa GaussMLE.LocalizationResult
        @test results.n_fits == 50
        
        # Check parameter recovery (looser tolerances for sCMOS)
        @test mean(results.parameters[3, :]) ≈ 1000.0 rtol=0.4
        @test mean(results.parameters[4, :]) ≈ 10.0 rtol=0.6
        
        # Uncertainties should be larger than ideal camera
        @test mean(results.uncertainties[1, :]) > 0.05
    end
    
    @testset "Raw data fitting (backward compatibility)" begin
        # Generate data using old method
        Random.seed!(42)
        n_photons = 1000.0f0
        background = 10.0f0
        sigma = 1.3f0
        box_size = 11
        n_rois = 20
        
        data = zeros(Float32, box_size, box_size, n_rois)
        for k in 1:n_rois
            x_true = 6.0f0 + randn(Float32) * 0.5f0
            y_true = 6.0f0 + randn(Float32) * 0.5f0
            
            for j in 1:box_size, i in 1:box_size
                dx = Float32(i) - x_true
                dy = Float32(j) - y_true
                gauss = exp(-0.5f0 * (dx^2 + dy^2) / sigma^2)
                mu = n_photons * gauss / (2π * sigma^2) + background
                data[i, j, k] = Float32(rand(Poisson(max(0.01, mu))))
            end
        end
        
        # Fit using raw data interface
        psf = GaussMLE.GaussianXYNB(1.3f0)
        fitter = GaussMLE.GaussMLEFitter(psf_model=psf, device=GaussMLE.CPU(), iterations=20)
        results = GaussMLE.fit(fitter, data)
        
        # Check results type and properties
        @test results isa GaussMLE.GaussMLEResults
        @test hasproperty(results, :parameters)
        @test results.x isa Vector{Float32}
        
        # Check recovery
        @test mean(results.x) ≈ 6.0 atol=1.0
        @test mean(results.y) ≈ 6.0 atol=1.0
        @test mean(results.photons) ≈ n_photons rtol=0.3
        @test mean(results.background) ≈ background rtol=0.5
        
        # Check uncertainties
        @test all(0.03 .< results.x_error .< 0.15)
        @test all(0.03 .< results.y_error .< 0.15)
    end
    
    @testset "Different PSF models" begin
        camera = SMLMData.IdealCamera(256, 256, 0.1)
        
        psf_models = [
            GaussMLE.GaussianXYNB(1.3f0),
            GaussMLE.GaussianXYNBS(),
            GaussMLE.GaussianXYNBSXSY()
        ]
        
        for psf in psf_models
            batch = GaussMLE.generate_roi_batch(camera, psf; n_rois=10, seed=42)
            fitter = GaussMLE.GaussMLEFitter(psf_model=psf, device=GaussMLE.CPU(), iterations=20)
            results = GaussMLE.fit(fitter, batch)
            
            @test results.n_fits == 10
            @test !any(isinf.(results.uncertainties))
            @test !any(isnan.(results.uncertainties))
        end
    end
end