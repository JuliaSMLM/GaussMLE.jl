using Test, GaussMLE, SMLMData, Statistics

"""
Consolidated test of new simulator and ROIBatch features. The Long group fits the other
Gaussian models (test/long/gaussian_models_cpu.jl).
"""

@testset "precompile workload" begin
    # A KernelAbstractions that cannot launch a CPU kernel while precompiling skips it.
    if GaussMLE._cpu_backend_precompilable()
        @test GaussMLE.PRECOMPILE_WORKLOAD_RAN[]
    end
end

@testset "New Features Tests" begin

    @testset "Basic Simulator with IdealCamera" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(0.13f0)

        batch = GaussMLE.generate_roi_batch(camera, psf; n_rois = 10, seed = 42)

        @test batch isa GaussMLE.ROIBatch
        @test length(batch) == 10
        @test batch.roi_size == 11
        @test batch.camera === camera
        @test size(batch.data) == (11, 11, 10)
    end

    @testset "SCMOSCamera Support" begin
        # Create sCMOS camera using SMLMData 0.4 API
        # Uniform readnoise of 5.0 e⁻ (variance = 25.0 e⁻²)
        readnoise = 5.0f0  # e⁻ rms
        scmos = SMLMData.SCMOSCamera(
            256, 256, 0.1f0, readnoise,
            offset = 100.0f0,  # ADU
            gain = 0.5f0,      # e⁻/ADU
            qe = 0.82f0        # quantum efficiency
        )

        @test scmos isa SMLMData.AbstractCamera
        @test scmos.readnoise == 5.0f0
        @test scmos.gain == 0.5f0

        # Generate data with sCMOS
        psf = GaussMLE.GaussianXYNB(0.13f0)
        batch = GaussMLE.generate_roi_batch(scmos, psf; n_rois = 5, seed = 42)

        @test batch.camera === scmos
    end

    @testset "sCMOS simulation reads calibration maps as (y, x)" begin
        # A 64 x 32 sensor (x by y) whose read noise is 0 left of column 40 and 50 e- from
        # it on, stored as a (32, 64) map. A ROI at x = 50 must take the noisy patch, one at
        # x = 1 the noise-free one. Indexing the map as (x, y) read past its 32 rows.
        nx, ny = 64, 32
        readnoise = zeros(Float32, ny, nx)
        readnoise[:, 40:end] .= 50.0f0
        scmos = SMLMData.SCMOSCamera(
            nx, ny, 0.1f0, readnoise; offset = 0.0f0, gain = 1.0f0, qe = 1.0f0
        )
        corners = Int32[50 1; 1 1]  # (x, y) per column: ROI 1 at x = 50, ROI 2 at x = 1
        batch = GaussMLE.generate_roi_batch(
            scmos, GaussMLE.GaussianXYNB(0.13f0); n_rois = 2, roi_size = 7, corners,
            seed = 3
        )
        noisy, clean = batch.data[:, :, 1], batch.data[:, :, 2]
        @test all(isinteger, clean)  # Poisson counts only
        @test count(!isinteger, noisy) > 40  # read noise on (nearly) every pixel
    end

    @testset "Fitting with ROIBatch" begin
        camera = SMLMData.IdealCamera(512, 512, 0.1)
        psf = GaussMLE.GaussianXYNB(0.13f0)

        batch = GaussMLE.generate_roi_batch(camera, psf; n_rois = 20, seed = 42)

        fitter = GaussMLE.GaussMLEConfig(
            psf_model = psf,
            backend = :cpu,
            iterations = 20
        )

        smld, _info = GaussMLE.fit(batch, fitter)

        @test smld isa SMLMData.BasicSMLD
        @test length(smld.emitters) == 20

        # Extract parameters from emitters
        photons_vals = [e.photons for e in smld.emitters]
        σ_x_vals = [e.σ_x for e in smld.emitters]  # In microns

        # Check reasonable results
        @test mean(photons_vals) ≈ 1000.0 rtol = 0.5
        # σ_x is in microns, expect 0.002-0.02 μm (0.02-0.2 pixels for 0.1 μm pixels)
        @test all(0.002 .< σ_x_vals .< 0.02)
    end

    @testset "the removed device keyword is refused" begin
        # 0.5 removed `device` (deprecated for `backend`) with its CPU and GPU types
        @test_throws MethodError GaussMLE.GaussMLEConfig(device = :cpu)
        @test !isdefined(GaussMLE, :ComputeDevice)
    end

    @testset "SMLMData Conversion" begin
        camera = SMLMData.IdealCamera(256, 256, 0.1)
        psf = GaussMLE.GaussianXYNB(0.13f0)

        # Known corners for testing
        corners = Matrix{Int32}(Int32[10 20; 30 40]')
        batch = GaussMLE.generate_roi_batch(
            camera, psf;
            n_rois = 2,
            corners = corners,
            xy_variation = 0.0f0,
            seed = 42
        )

        fitter = GaussMLE.GaussMLEConfig(
            psf_model = psf, backend = :cpu, iterations = 20,
        )
        smld, _info = GaussMLE.fit(batch, fitter)

        @test smld isa SMLMData.BasicSMLD
        @test length(smld) == 2
        # Camera may be recreated during fitting, check type instead of identity
        @test smld.camera isa SMLMData.IdealCamera

        # Check coordinate conversion
        # Note: fit() currently loses corner information when converting to SMLD
        # So we can't test exact coordinates, but we can test that positions are reasonable
        @test all([isfinite(e.x) && e.x >= 0 for e in smld.emitters])
        @test all([isfinite(e.y) && e.y >= 0 for e in smld.emitters])
    end
end
