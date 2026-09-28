# SplinePSFModel (moved from PSFLearning's GaussMLE bridge): its constructors, PSFModel methods,
# CPU-only backend dispatch and the square-pixel check. The GPU group checks :auto on a GPU host;
# the Long group checks bias and CRLB.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays

@testset "SplinePSFModel" begin

    # Unaberrated scalar PSF on a coarse grid, as in PSFLearning's bridge tests
    spline = SplinePSF(
        ScalarPSF(1.4, 0.68, 1.518);
        lateral_range = 1.0, axial_range = 0.5, lateral_step = 0.05, axial_step = 0.1
    )
    model(; kwargs...) = SplinePSFModel(spline; pixel_size = 0.1, z_range = (-0.5, 0.5), kwargs...)

    @testset "construction from a SplinePSF" begin
        gpsf = model()
        @test gpsf isa SplinePSFModel{Float32}
        @test gpsf isa GaussMLE.PSFModel{5, Float32}
        @test gpsf.spline_psf === spline  # used directly, no re-tabulation
        @test gpsf.pixel_size ≈ 0.1f0
        @test gpsf.z_range == (-0.5f0, 0.5f0)
        @test gpsf.px_scale > 0
        @test length(gpsf) == 5

        # z_range defaults to the spline's own range and must lie within it
        @test SplinePSFModel(spline; pixel_size = 0.1).z_range == gpsf.z_range
        @test_throws ArgumentError SplinePSFModel(spline; pixel_size = 0.1, z_range = (-1.5, 1.5))

        # A 2D SplinePSF has no z
        spline2d = SplinePSF(AiryPSF(1.4, 0.68), -0.5:0.1:0.5, -0.5:0.1:0.5)
        @test_throws ArgumentError SplinePSFModel(spline2d; pixel_size = 0.1)
    end

    @testset "construction from any callable" begin
        astig(x, y, z) = exp(-x^2 / (2 * (0.13 + 0.1z)^2) - y^2 / (2 * (0.13 - 0.1z)^2))
        gpsf = SplinePSFModel(astig; pixel_size = 0.1, z_range = (-0.5, 0.5))
        @test gpsf isa SplinePSFModel{Float32}
        @test gpsf.z_range == (-0.5f0, 0.5f0)
        @test gpsf.lateral_range == 1.5f0
        # Normalized so the pixel-summed PSF at z = 0 is 1
        n_px = ceil(Int, 1.5 / 0.1)
        @test sum(astig(0.1ix, 0.1iy, 0.0) for ix in (-n_px):n_px, iy in (-n_px):n_px) *
            gpsf.px_scale ≈ 1 rtol = 1.0e-5
        @test_throws ArgumentError SplinePSFModel(astig; pixel_size = 0.1, z_range = (0.5, -0.5))
    end

    @testset "evaluate_psf" begin
        px_gpsf = GaussMLE.to_pixel_units(model(), 0.1)

        # Center of ROI, z=0, N=1000, bg=10
        θ = SVector{5, Float32}(6.0f0, 6.0f0, 0.0f0, 1000.0f0, 10.0f0)

        # At the center pixel, PSF should be > bg
        val_center = GaussMLE.evaluate_psf(px_gpsf, 6, 6, θ)
        @test val_center > 10.0f0

        # At a far edge pixel, should be near bg
        val_edge = GaussMLE.evaluate_psf(px_gpsf, 1, 1, θ)
        @test val_edge > 9.0f0
        @test val_edge < val_center

        # PSF is symmetric: (5,6) ≈ (7,6)
        val_left = GaussMLE.evaluate_psf(px_gpsf, 5, 6, θ)
        val_right = GaussMLE.evaluate_psf(px_gpsf, 7, 6, θ)
        @test val_left ≈ val_right rtol = 0.01

        # N=0 should give bg
        θ_nobg = SVector{5, Float32}(6.0f0, 6.0f0, 0.0f0, 0.0f0, 10.0f0)
        @test GaussMLE.evaluate_psf(px_gpsf, 6, 6, θ_nobg) ≈ 10.0f0
    end

    @testset "compute_pixel_derivatives" begin
        px_gpsf = GaussMLE.to_pixel_units(model(), 0.1)

        θ = SVector{5, Float32}(6.0f0, 6.0f0, 0.0f0, 1000.0f0, 10.0f0)
        val, dudt, d2udt2 = GaussMLE.compute_pixel_derivatives(6, 6, θ, px_gpsf)

        # Model value should match evaluate_psf
        @test val ≈ GaussMLE.evaluate_psf(px_gpsf, 6, 6, θ) rtol = 1.0e-4

        # At center, x and y derivatives should be ≈ 0 (peak)
        @test abs(dudt[1]) < 10.0f0
        @test abs(dudt[2]) < 10.0f0

        # N derivative = PSF fraction at this pixel (positive)
        @test dudt[4] > 0
        @test dudt[4] ≈ (val - θ[5]) / θ[4] rtol = 0.1

        # bg derivative = 1
        @test dudt[5] ≈ 1.0f0

        # Second derivatives: x and y should be negative at peak (concave)
        @test d2udt2[1] < 0
        @test d2udt2[2] < 0

        # N and bg second derivatives = 0
        @test d2udt2[4] == 0.0f0
        @test d2udt2[5] == 0.0f0
    end

    @testset "simple_initialize" begin
        # Create a synthetic ROI (11x11) with signal at center
        box_size = 11
        roi = fill(10.0f0, box_size, box_size)
        center = (box_size + 1) / 2
        for j in 1:box_size, i in 1:box_size
            r2 = (i - center)^2 + (j - center)^2
            roi[i, j] += 1000.0f0 * exp(-r2 / 4.0f0)
        end

        θ = GaussMLE.simple_initialize(roi, box_size, model())

        @test length(θ) == 5
        @test θ[1] ≈ center atol = 1.0
        @test θ[2] ≈ center atol = 1.0
        @test θ[3] ≈ 0.0f0
        @test θ[4] > 0
        @test θ[5] > 0
    end

    @testset "to_pixel_units" begin
        gpsf = model()

        gpsf_px = GaussMLE.to_pixel_units(gpsf, 0.127)
        @test gpsf_px.pixel_size ≈ 0.127f0
        @test gpsf_px.z_range == gpsf.z_range
        @test gpsf_px.px_scale > 0
        @test gpsf_px.px_scale != gpsf.px_scale
    end

    @testset "default_constraints" begin
        # PSFLearning's test used (-0.8, 0.8); a SplinePSF z_range must lie within the spline's
        # own range, which is ±0.5 here
        gpsf = model(z_range = (-0.4, 0.4))

        constraints = GaussMLE.default_constraints(gpsf, 11)
        @test constraints isa GaussMLE.ParameterConstraints{5}
        @test constraints.lower[3] ≈ -0.4f0
        @test constraints.upper[3] ≈ 0.4f0
    end

    @testset "lateral_range stored and used" begin
        gpsf = model(lateral_range = 2.0)
        @test gpsf.lateral_range ≈ 2.0f0

        # to_pixel_units should preserve lateral_range
        gpsf2 = GaussMLE.to_pixel_units(gpsf, 0.05)
        @test gpsf2.lateral_range ≈ 2.0f0
        @test gpsf2.pixel_size ≈ 0.05f0
        # px_scale should be recomputed with the correct lateral_range
        @test gpsf2.px_scale > 0
        @test gpsf2.px_scale != gpsf.px_scale
    end

    @testset "GaussMLEConfig construction" begin
        gpsf = model()
        fitter = GaussMLEConfig(psf_model = gpsf, iterations = 20, backend = :cpu)
        @test fitter isa GaussMLEConfig
        @test fitter.psf_model === gpsf
        @test fitter.iterations == 20
    end

    # Noise-free ROIs from an astigmatic spline (an unaberrated PSF is symmetric in z, so a fit
    # started at z = 0 cannot leave it)
    zc = ZernikeCoefficients(15)
    zc.phase[6] = 0.5  # vertical astigmatism, 0.5 rad RMS
    astig_spline = SplinePSF(
        ScalarPSF(1.4, 0.68, 1.518; zernike_coeffs = zc);
        lateral_range = 1.0, axial_range = 0.5, lateral_step = 0.05, axial_step = 0.1
    )
    gpsf = SplinePSFModel(astig_spline; pixel_size = 0.1)
    box, n = 11, 4
    truth = [(6.0, 6.0, 0.2), (5.5, 6.5, -0.2), (6.3, 5.8, 0.0), (6.0, 6.0, 0.1)]
    data = zeros(Float32, box, box, n)
    for (k, (x, y, z)) in enumerate(truth), j in 1:box, i in 1:box
        data[i, j, k] = GaussMLE.evaluate_psf(gpsf, i, j, SVector{5, Float32}(x, y, z, 2000, 10))
    end
    batch(camera) = ROIBatch(data, ones(Int32, n), ones(Int32, n), Int32.(1:n), camera)

    @testset "fit(::ROIBatch) on the CPU" begin
        smld, info = fit(batch(IdealCamera(64, 64, 0.1)); psf_model = gpsf, backend = :cpu)
        @test info.backend == :cpu
        @test length(smld.emitters) == n
        @test smld.emitters[1] isa Emitter3DFitGaussMLE{Float32}
        for (e, (x, y, z)) in zip(smld.emitters, truth)
            @test e.x ≈ (x - 0.5) * 0.1 atol = 0.005
            @test e.y ≈ (y - 0.5) * 0.1 atol = 0.005
            @test e.z ≈ z atol = 0.02
            @test e.photons ≈ 2000 rtol = 0.05
        end
    end

    @testset "backend dispatch: CPU only" begin
        @test GaussMLE.gpu_compatible(gpsf) == false
        @test GaussMLE.gpu_compatible(GaussianXYNB(0.13f0)) == true
        @test GaussMLE.gpu_compatible(AstigmaticXYZNB{Float32}(0.13f0, 0.13f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.25f0, 0.4f0))
        # :gpu is refused before any CUDA check, so this holds on hosts without a GPU too
        @test_throws ArgumentError fit(batch(IdealCamera(64, 64, 0.1)); psf_model = gpsf, backend = :gpu)
    end

    @testset "non-square pixels" begin
        rect = (0.127, 0.116)  # PSFLearning's default config
        @test_throws ArgumentError fit(batch(IdealCamera(64, 64, rect)); psf_model = gpsf, backend = :cpu)
        @test_throws ArgumentError fit(batch(SCMOSCamera(64, 64, rect, 1.6)); psf_model = gpsf, backend = :cpu)
        # Gaussian models keep their current behaviour
        smld, _ = fit(batch(IdealCamera(64, 64, rect)); psf_model = GaussianXYNB(0.13f0), backend = :cpu)
        @test length(smld.emitters) == n
    end

end
