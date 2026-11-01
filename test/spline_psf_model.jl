# SplinePSFModel (moved from PSFLearning's GaussMLE bridge): its constructors, PSFModel
# methods, backend dispatch and rectangular pixels. The GPU group checks :auto
# on a GPU host; the Long group checks bias and CRLB.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, Random

# Share one Float32-only callable type to cover dispatch and limit Core compilation.
struct SplineTestPSF
    linear::Bool
    x_shift::Float64
    y_shift::Float64
    z_min::Float64
    z_max::Float64
end

function (psf::SplineTestPSF)(x::Float32, y::Float32, z::Float32)
    psf.z_min <= z <= psf.z_max || error("evaluated outside the callable's valid z range")
    dx, dy = x - psf.x_shift * z, y - psf.y_shift * z
    if psf.linear
        # An identifiable change that is linear in z and positive over the full grid.
        return exp(-(dx^2 + dy^2) / (2 * 0.13^2)) *
            (1 + z * tanh((dx^2 - dy^2) / 0.02))
    end
    return exp(-dx^2 / (2 * (0.13 + 0.1z)^2) - dy^2 / (2 * (0.13 - 0.1z)^2))
end

const astig_shape = SplineTestPSF(false, 0.0, 0.0, -Inf, Inf)
const linear_shape = SplineTestPSF(true, 0.0, 0.0, -Inf, Inf)

function roi_at(m, z; x = 6, y = 6, box = 11, photons = 1000)
    return [
        GaussMLE.evaluate_psf(m, i, j, SVector{5, Float32}(x, y, z, photons, 10))
            for i in 1:box, j in 1:box
    ]
end

function batch_at(m, z; x = 6, y = 6, box = 11, photons = 1000)
    roi = roi_at(m, z; x, y, box, photons)
    data = reshape(roi, size(roi)..., 1)
    return ROIBatch(data, Int32[1], Int32[1], Int32[1], IdealCamera(64, 64, m.pixel_size))
end

@testset "SplinePSFModel" begin

    # Unaberrated scalar PSF on a coarse grid, as in PSFLearning's bridge tests
    spline = SplinePSF(
        ScalarPSF(1.4, 0.68, 1.518);
        lateral_range = 1.0, axial_range = 0.5, lateral_step = 0.05, axial_step = 0.1
    )
    model(; kwargs...) = SplinePSFModel(
        spline; pixel_size = 0.1, z_range = (-0.5, 0.5), kwargs...
    )

    @testset "construction from a SplinePSF" begin
        gpsf = model()
        @test gpsf isa SplinePSFModel{Float32}
        @test gpsf isa GaussMLE.PSFModel{5, Float32}
        @test gpsf.spline_psf === spline  # retained as the point PSF for rebuilds
        @test gpsf.pixel_size == (0.1f0, 0.1f0)
        @test gpsf.z_range == (-0.5f0, 0.5f0)
        @test gpsf.px_scale > 0
        @test length(gpsf) == 5

        # z_range defaults to the spline's own range and must lie within it
        @test SplinePSFModel(spline; pixel_size = 0.1).z_range == gpsf.z_range
        @test_throws ArgumentError SplinePSFModel(
            spline; pixel_size = 0.1, z_range = (-1.5, 1.5)
        )

        for zr in (
                (0.4, -0.4), (0.1, 0.1), (NaN, 0.4), (-0.4, Inf),
                (0.25, 0.25 + 1.0e-10), (0.2, 0.205),
            )
            @test_throws ArgumentError SplinePSFModel(
                spline; pixel_size = 0.1, z_range = zr
            )
        end

        # A 2D SplinePSF has no z
        spline2d = SplinePSF(AiryPSF(1.4, 0.68), -0.5:0.1:0.5, -0.5:0.1:0.5)
        @test_throws ArgumentError SplinePSFModel(spline2d; pixel_size = 0.1)
    end

    @testset "construction from any callable" begin
        gpsf = SplinePSFModel(astig_shape; pixel_size = 0.1, z_range = (-0.5, 0.5))
        @test gpsf isa SplinePSFModel{Float32}
        @test gpsf.z_range == (-0.5f0, 0.5f0)
        @test gpsf.lateral_range == 1.5f0
        # Normalized so the pixel-summed PSF at z = 0 is 1
        n_px = ceil(Int, 1.5 / 0.1)
        reference_sum = sum(
            astig_shape(0.1f0 * ix, 0.1f0 * iy, 0.0f0)
                for ix in (-n_px):n_px, iy in (-n_px):n_px
        )
        @test reference_sum * gpsf.px_scale ≈ 1 rtol = 1.0e-5
        for zr in (
                (0.5, -0.5), (0.1, 0.1), (-Inf, 0.5), (-0.5, NaN),
                (0.25, 0.25 + 1.0e-10), (0.2, 0.205),
            )
            @test_throws ArgumentError SplinePSFModel(
                astig_shape; pixel_size = 0.1, z_range = zr
            )
        end
    end

    @testset "invalid pixel size and lateral range" begin
        # Invalid dimensions must be rejected before the callable is sampled.
        no_sampling(x, y, z) = error("invalid dimensions reached PSF sampling")
        for value in (0.0, -0.01, NaN, Inf, -Inf, 1.0e-50, 1.0e40)
            @test_throws ArgumentError SplinePSFModel(
                no_sampling; pixel_size = value, z_range = (-0.5, 0.5)
            )
            @test_throws ArgumentError SplinePSFModel(
                no_sampling; pixel_size = 0.1, z_range = (-0.5, 0.5), lateral_range = value
            )
            for pixel_size in ((value, 0.1), (0.1, value))
                @test_throws ArgumentError SplinePSFModel(
                    no_sampling; pixel_size, z_range = (-0.5, 0.5)
                )
            end
        end
    end

    @testset "sCMOS fit recovers microns and electrons" begin
        m = SplinePSFModel(astig_shape; pixel_size = 0.1, z_range = (-0.5, 0.5))
        box, sensor_size = 9, 32
        truth = ((5.2, 4.7, 0.2), (4.6, 5.3, -0.2))
        x_corners, y_corners = Int32[4, 19], Int32[16, 6]
        readnoise = [
            sqrt(1.0f0 + 0.5f0 * (i - 1) + 0.05f0 * (j - 1))
                for i in 1:sensor_size, j in 1:sensor_size
        ]
        offset, gain = 100.0f0, 0.5f0
        camera = SCMOSCamera(
            sensor_size, sensor_size, 0.1f0, readnoise;
            offset, gain, qe = 1.0f0
        )
        data = zeros(Float32, box, box, length(truth))
        for (k, (x, y, z)) in enumerate(truth)
            electrons = roi_at(m, z; x, y, box, photons = 2000)
            data[:, :, k] .= electrons ./ gain .+ offset  # Noise-free raw ADU
        end
        roi_batch = ROIBatch(data, x_corners, y_corners, Int32[1, 2], camera)
        smld, info = fit(roi_batch; psf_model = m, backend = :cpu, iterations = 40)
        @test info.backend == :cpu
        @test length(smld.emitters) == length(truth)
        for (k, (e, (x, y, z))) in enumerate(zip(smld.emitters, truth))
            @test e.x ≈ (x_corners[k] + x - 1.5) * 0.1 atol = 0.001
            @test e.y ≈ (y_corners[k] + y - 1.5) * 0.1 atol = 0.001
            @test e.z ≈ z atol = 0.001
            @test e.photons ≈ 2000 rtol = 0.01  # Electrons, before ADU conversion
            @test e.bg ≈ 10 rtol = 0.01
            @test all(isfinite, (e.σ_x, e.σ_y, e.σ_z, e.σ_photons, e.σ_bg))
        end
    end

    @testset "fit recovers the z-scan start far from focus" begin
        amodel = SplinePSFModel(astig_shape; pixel_size = 0.1, z_range = (-0.5, 0.5))
        for z in (-0.4f0, 0.4f0)
            smld, _ = fit(batch_at(amodel, z); psf_model = amodel, backend = :cpu)
            @test only(smld.emitters).z ≈ z atol = 0.001
        end
    end

    @testset "fit near both axial bounds with a linear-in-z PSF" begin
        lmodel = SplinePSFModel(linear_shape; pixel_size = 0.1, z_range = (-0.5, 0.5))
        # The scan starts at the bound; false curvature used to send the step outward.
        for z in (-0.5f0, -0.499f0, -0.497f0, 0.497f0, 0.499f0, 0.5f0)
            smld, _ = fit(batch_at(lmodel, z); psf_model = lmodel, backend = :cpu)
            @test only(smld.emitters).z ≈ z atol = 0.0002
        end
    end

    @testset "fit preserves axial position covariances" begin
        moving = SplineTestPSF(false, 0.2, -0.1, -Inf, Inf)
        m = SplinePSFModel(moving; pixel_size = 0.1, z_range = (-0.5, 0.5))
        smld, _ = fit(batch_at(m, 0.2); psf_model = m, backend = :cpu, iterations = 40)
        e = only(smld.emitters)
        @test e.z ≈ 0.2 atol = 0.002
        @test isfinite(e.σ_xz) && isfinite(e.σ_yz)
        @test max(abs(e.σ_xz), abs(e.σ_yz)) > 1.0e-7
    end

    @testset "custom z constraints intersect the model range" begin
        m = SplinePSFModel(astig_shape; pixel_size = 0.1, z_range = (-0.5, 0.5))
        source = SplinePSFModel(astig_shape; pixel_size = 0.1, z_range = (-0.6, 0.6))
        defaults = GaussMLE.default_constraints(m, 11)
        custom(lo, hi) = GaussMLE.ParameterConstraints{5}(
            SVector{5, Float32}(-1, -1, lo, 1, 0.01),
            SVector{5, Float32}(13, 13, hi, Inf, Inf), defaults.max_step
        )
        wider = custom(-1, 1)
        for z in (-0.55, 0.55)
            smld, _ = fit(
                batch_at(source, z); psf_model = m, backend = :cpu, constraints = wider
            )
            fitted_z = only(smld.emitters).z
            @test m.z_range[1] <= fitted_z <= m.z_range[2]
            @test fitted_z ≈ clamp(z, m.z_range...) atol = 0.001
        end
        # A narrower user interval is preserved, and a disjoint interval is refused.
        smld, _ = fit(
            batch_at(m, 0.4); psf_model = m, backend = :cpu, constraints = custom(0.1, 0.2)
        )
        @test only(smld.emitters).z ≈ 0.2 atol = 0.001
        @test_throws ArgumentError fit(
            batch_at(m, 0.2); psf_model = m, backend = :cpu, constraints = custom(0.6, 0.8)
        )
    end

    @testset "fit rescales to another square pixel size" begin
        # The truth is sampled on the new camera grid, independently of fit's rescaling.
        m = SplinePSFModel(
            astig_shape; pixel_size = 0.1, z_range = (-0.5, 0.5), lateral_range = 2.0
        )
        truth_model = SplinePSFModel(
            astig_shape; pixel_size = 0.127, z_range = (-0.5, 0.5), lateral_range = 2.0
        )
        smld, _ = fit(
            batch_at(truth_model, 0.2; x = 5.4, y = 6.2); psf_model = m, backend = :cpu
        )
        e = only(smld.emitters)
        @test e.x ≈ (5.4 - 0.5) * 0.127 atol = 0.001
        @test e.y ≈ (6.2 - 0.5) * 0.127 atol = 0.001
        @test e.z ≈ 0.2 atol = 0.001
        @test e.photons ≈ 1000 rtol = 0.01
    end

    @testset "Float32-only callable normalization and rescaling" begin
        # Each endpoint nearest zero rounds outward in Float32, so move it inward before
        # normalization. The same callable type exercises construction and fitting.
        for (zr, z) in (((0.20000001, 0.5), 0.3), ((-0.5, -0.20000001), -0.3))
            bounded = SplineTestPSF(false, 0.0, 0.0, zr...)
            m = SplinePSFModel(bounded; pixel_size = 0.1, z_range = zr)
            truth_model = SplinePSFModel(bounded; pixel_size = 0.127, z_range = zr)
            smld, _ = fit(batch_at(truth_model, z); psf_model = m, backend = :cpu)
            @test only(smld.emitters).z ≈ z atol = 0.001
            @test only(smld.emitters).photons ≈ 1000 rtol = 0.01
        end
        for value in (0.0, NaN, Inf)
            @test_throws ArgumentError SplinePSFModel(
                (x, y, z) -> value; pixel_size = 0.1, z_range = (-0.5, 0.5)
            )
        end
    end

    @testset "default fits clamp to lateral and axial bounds" begin
        # Truncated tails converge slowly, and in a 7-pixel box this fit stalls inside the
        # bound (x = -0.86 at any iteration count); an 11-pixel box reaches it. Keep z close
        # to zero so a width change cannot absorb the lateral displacement.
        m = SplinePSFModel(astig_shape; pixel_size = 0.1, z_range = (-0.01, 0.01))
        box = 11
        center = Float32((box + 1) / 2)
        for axis in 1:2, position in (-1.5f0, Float32(box + 2.5))
            x, y = axis == 1 ? (position, center) : (center, position)
            smld, _ = fit(
                batch_at(m, 0; x, y, box, photons = 5000);
                psf_model = m, backend = :cpu, iterations = 500
            )
            e = only(smld.emitters)
            # Corner (1, 1): invert the output's conversion from ROI pixels to microns.
            fitted = (e.x, e.y)[axis] / m.pixel_size[axis] + 0.5f0
            bound = clamp(position, -1.0f0, Float32(box + 2))
            @test fitted ≈ bound atol = 0.001f0 rtol = 0
            @test m.z_range[1] <= e.z <= m.z_range[2]
        end

        m = SplinePSFModel(astig_shape; pixel_size = 0.1, z_range = (-0.4, 0.4))
        source = SplinePSFModel(astig_shape; pixel_size = 0.1, z_range = (-0.5, 0.5))
        for z in (-0.45, 0.45)
            smld, _ = fit(batch_at(source, z); psf_model = m, backend = :cpu)
            fitted_z = only(smld.emitters).z
            @test m.z_range[1] <= fitted_z <= m.z_range[2]
            @test fitted_z ≈ clamp(z, m.z_range...) atol = 0.001
        end
    end

    @testset "GaussMLEConfig construction" begin
        gpsf = model()
        fitter = GaussMLEConfig(psf_model = gpsf, iterations = 20, backend = :cpu)
        @test fitter isa GaussMLEConfig
        @test fitter.psf_model === gpsf
        @test fitter.iterations == 20
    end

    # Noise-free ROIs from an astigmatic spline (an unaberrated PSF is symmetric in z, so a
    # fit started at z = 0 cannot leave it)
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
        data[i, j, k] = GaussMLE.evaluate_psf(
            gpsf, i, j, SVector{5, Float32}(x, y, z, 2000, 10)
        )
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

    @testset "fitted photons are total photons at every z" begin
        px, box = 0.1, 13
        edges = collect(0:box) .* px
        truth = (
            (7.2, 6.7, -0.4), (6.6, 7.3, -0.2), (7.1, 7.4, 0.0),
            (7.35, 6.8, 0.2), (6.8, 7.15, 0.4),
        )
        n = length(truth)
        data = zeros(Float32, box, box, n)
        for (k, (x, y, z)) in enumerate(truth)
            emitters = [Emitter3D((x - 0.5) * px, (y - 0.5) * px, z, 2000.0)]
            data[:, :, k] .= MicroscopePSFs.integrate_pixels(
                astig_spline, edges, edges, emitters; sampling = 8
            ) .+ 10
        end
        roi_batch = ROIBatch(
            data, ones(Int32, n), ones(Int32, n), Int32.(1:n), IdealCamera(64, 64, px)
        )
        smld, _ = fit(roi_batch; psf_model = gpsf, backend = :cpu, iterations = 40)
        for (k, (x, y, z)) in enumerate(truth)
            e = smld.emitters[k]
            @test abs(e.photons / 2000 - 1) <= 0.005
            @test e.z ≈ z atol = 0.003
            @test e.x ≈ (x - 0.5) * px atol = 0.001
            @test e.y ≈ (y - 0.5) * px atol = 0.001
        end
    end

    @testset "expected pixel counts are the pixel integral of the point PSF" begin
        px = 0.1
        xr, yr, zr = -1.2:0.02:1.2, -1.2:0.02:1.2, -0.6:0.05:0.6
        σx(z) = 0.13 * sqrt(1 + ((z - 0.25) / 0.4)^2)
        σy(z) = 0.13 * sqrt(1 + ((z + 0.25) / 0.4)^2)
        # Tabulate the point density in [y, x, z] order, without pixel integration.
        stack = [
            1 / (2π * σx(z) * σy(z)) *
                exp(-dx^2 / (2 * σx(z)^2) - dy^2 / (2 * σy(z)^2))
                for dy in yr, dx in xr, z in zr
        ]
        m = SplinePSFModel(SplinePSF(stack, xr, yr, zr); pixel_size = px)
        # A raw bead stack or an incorrectly scaled density must not redefine N.
        @test_throws ArgumentError SplinePSFModel(
            SplinePSF(100 .* stack, xr, yr, zr); pixel_size = px
        )

        # The Gaussian's erf integrals provide independent, noise-free pixel counts.
        truth_model = GaussMLE.to_pixel_units(
            AstigmaticXYZNB{Float32}(0.13, 0.13, 0, 0, 0, 0, 0.25, 0.4), px
        )
        box = 13
        truth = (
            (7.2, 6.7, -0.4), (6.6, 7.3, -0.3), (7.1, 7.4, 0.0),
            (7.35, 6.8, 0.3), (6.8, 7.15, 0.4),
        )
        n = length(truth)
        data = zeros(Float32, box, box, n)
        for (k, (x, y, z)) in enumerate(truth)
            data[:, :, k] .= roi_at(truth_model, z; x, y, box, photons = 2000)
        end
        roi_batch = ROIBatch(
            data, ones(Int32, n), ones(Int32, n), Int32.(1:n), IdealCamera(64, 64, px)
        )
        smld, _ = fit(roi_batch; psf_model = m, backend = :cpu, iterations = 40)
        for (k, (x, y, z)) in enumerate(truth)
            e = smld.emitters[k]
            @test e.x ≈ (x - 0.5) * px atol = 0.001
            @test e.y ≈ (y - 0.5) * px atol = 0.001
            @test e.z ≈ z atol = 0.002
            @test abs(e.photons / 2000 - 1) <= 0.002
        end
    end

    @testset "pixel integration is rebuilt for the camera pixel size" begin
        px = 0.108
        xr, yr, zr = -1.2:0.02:1.2, -1.2:0.02:1.2, -0.6:0.05:0.6
        σx(z) = 0.13 * sqrt(1 + ((z - 0.25) / 0.4)^2)
        σy(z) = 0.13 * sqrt(1 + ((z + 0.25) / 0.4)^2)
        stack = [
            1 / (2π * σx(z) * σy(z)) *
                exp(-dx^2 / (2 * σx(z)^2) - dy^2 / (2 * σy(z)^2))
                for dy in yr, dx in xr, z in zr
        ]
        m = SplinePSFModel(SplinePSF(stack, xr, yr, zr); pixel_size = 0.1)
        truth_model = GaussMLE.to_pixel_units(
            AstigmaticXYZNB{Float32}(0.13, 0.13, 0, 0, 0, 0, 0.25, 0.4), px
        )
        box = 13
        truth = ((7.2, 6.7, -0.4), (6.6, 7.3, 0.0), (7.35, 6.8, 0.4))
        n = length(truth)
        data = zeros(Float32, box, box, n)
        for (k, (x, y, z)) in enumerate(truth)
            data[:, :, k] .= roi_at(truth_model, z; x, y, box, photons = 2000)
        end
        roi_batch = ROIBatch(
            data, ones(Int32, n), ones(Int32, n), Int32.(1:n), IdealCamera(64, 64, px)
        )
        smld, _ = fit(roi_batch; psf_model = m, backend = :cpu, iterations = 40)
        for (k, (x, y, z)) in enumerate(truth)
            e = smld.emitters[k]
            @test e.x ≈ (x - 0.5) * px atol = 0.001
            @test e.y ≈ (y - 0.5) * px atol = 0.001
            @test e.z ≈ z atol = 0.002
            @test abs(e.photons / 2000 - 1) <= 0.002
        end
    end

    @testset "backend dispatch: same GPU requirement as Gaussian models" begin
        @test GaussMLEConfig(psf_model = gpsf, backend = :gpu).backend == :gpu
        if GaussMLE._gpu_vendor() === nothing
            roi_batch = batch(IdealCamera(64, 64, 0.1))
            failures = map((GaussianXYNB(0.13f0), gpsf)) do psf
                try
                    fit(roi_batch; psf_model = psf, backend = :gpu)
                catch err
                    err
                end
            end
            @test all(err -> err isa ErrorException, failures)
            @test sprint(showerror, failures[1]) == sprint(showerror, failures[2])
        end
    end

    @testset "non-square pixels" begin
        rect = (0.127, 0.116)  # PSFLearning's default config
        for camera in (IdealCamera(64, 64, rect), SCMOSCamera(64, 64, rect, 1.6))
            smld, _ = fit(batch(camera); psf_model = gpsf, backend = :cpu)
            @test length(smld.emitters) == n
        end
        # Gaussian models keep their current behaviour
        smld, _ = fit(
            batch(IdealCamera(64, 64, rect));
            psf_model = GaussianXYNB(0.13f0), backend = :cpu
        )
        @test length(smld.emitters) == n
    end

    @testset "pixel integration on a rectangular camera" begin
        px_x, px_y = 0.1313, 0.1258
        box = 13
        edges_x, edges_y = collect(0:box) .* px_x, collect(0:box) .* px_y
        truth = ((7.2, 6.7, -0.4), (6.6, 7.3, 0.0), (7.35, 6.8, 0.4))
        n = length(truth)
        data = zeros(Float32, box, box, n)
        for (k, (x, y, z)) in enumerate(truth)
            emitters = [Emitter3D((x - 0.5) * px_x, (y - 0.5) * px_y, z, 2000.0)]
            data[:, :, k] .= MicroscopePSFs.integrate_pixels(
                astig_spline, edges_x, edges_y, emitters; sampling = 8
            ) .+ 10
        end
        camera = IdealCamera(64, 64, (px_x, px_y))
        roi_batch = ROIBatch(data, ones(Int32, n), ones(Int32, n), Int32.(1:n), camera)
        # gpsf was built at 0.1 µm square pixels; fitting must rebuild both axes.
        smld, _ = fit(roi_batch; psf_model = gpsf, backend = :cpu, iterations = 40)
        for (k, (x, y, z)) in enumerate(truth)
            e = smld.emitters[k]
            @test e.x ≈ (x - 0.5) * px_x atol = 0.001
            @test e.y ≈ (y - 0.5) * px_y atol = 0.001
            @test e.z ≈ z atol = 0.003
            @test abs(e.photons / 2000 - 1) <= 0.005
        end

        direct = SplinePSFModel(astig_spline; pixel_size = (px_x, px_y))
        one_roi = ROIBatch(data[:, :, 1:1], Int32[1], Int32[1], Int32[1], camera)
        direct_smld, _ = fit(one_roi; psf_model = direct, backend = :cpu, iterations = 40)
        e = only(direct_smld.emitters)
        for field in (:x, :y, :z, :photons, :bg)
            expected = getproperty(smld.emitters[1], field)
            @test getproperty(e, field) ≈ expected atol = 1.0e-6 rtol = 1.0e-5
        end
    end

    @testset "fit(::Array) fits at the model's pixel size" begin
        px_x, px_y = 0.13, 0.12
        box = 13
        edges_x, edges_y = collect(0:box) .* px_x, collect(0:box) .* px_y
        x, y, z = 7.2, 6.7, 0.2
        emitters = [Emitter3D((x - 0.5) * px_x, (y - 0.5) * px_y, z, 2000.0)]
        roi = MicroscopePSFs.integrate_pixels(
            astig_spline, edges_x, edges_y, emitters; sampling = 8
        ) .+ 10
        data = reshape(Float32.(roi), box, box, 1)
        direct = SplinePSFModel(astig_spline; pixel_size = (px_x, px_y))
        smld_a, _ = fit(
            data, GaussMLEConfig(psf_model = direct, backend = :cpu, iterations = 40)
        )
        camera = IdealCamera(64, 64, (px_x, px_y))
        batch1 = ROIBatch(data, Int32[1], Int32[1], Int32[1], camera)
        smld_b, _ = fit(batch1; psf_model = direct, backend = :cpu, iterations = 40)
        ea, eb = only(smld_a.emitters), only(smld_b.emitters)
        @test ea.z ≈ eb.z rtol = 1.0e-5
        @test ea.photons ≈ eb.photons rtol = 1.0e-5
        @test ea.x ≈ (x - 0.5) * px_x atol = 0.001
        @test ea.y ≈ (y - 0.5) * px_y atol = 0.001
    end

    @testset "bright emitters converge in the default iterations" begin
        px, box = 0.1, 13
        edges = collect(0:box) .* px
        truth = ((7.2, 6.7, -0.4), (7.1, 7.4, 0.0), (6.8, 7.15, 0.4))
        n = length(truth)
        data = zeros(Float32, box, box, n)
        for (k, (x, y, z)) in enumerate(truth)
            emitters = [Emitter3D((x - 0.5) * px, (y - 0.5) * px, z, 3.0e5)]
            data[:, :, k] .= MicroscopePSFs.integrate_pixels(
                astig_spline, edges, edges, emitters; sampling = 8
            ) .+ 50
        end
        smld, _ = fit(data, GaussMLEConfig(psf_model = gpsf, backend = :cpu))
        for (k, (x, y, z)) in enumerate(truth)
            e = smld.emitters[k]
            @test abs(e.photons / 3.0e5 - 1) <= 0.005
            @test e.z ≈ z atol = 0.003
            # fit(::Array) tiles the ROIs horizontally: ROI k starts (k - 1) boxes along x
            @test e.x ≈ (x - 0.5 + (k - 1) * box) * px atol = 0.001
            @test e.y ≈ (y - 0.5) * px atol = 0.001
        end
    end

    @testset "sCMOS fits end at a maximum of the fit objective" begin
        px, box, sensor_size = 0.1, 11, 40
        edges = collect(0:box) .* px
        truth = ((6.2, 5.8, -0.3), (5.7, 6.4, 0.2), (6.0, 6.0, 0.35), (6.4, 5.6, -0.1))
        n = length(truth)
        x_corners, y_corners = Int32[3, 25, 6, 28], Int32[5, 8, 24, 27]
        readnoise = [
            sqrt(1.0f0 + 0.02f0 * (i - 1) + 0.05f0 * (j - 1))
                for i in 1:sensor_size, j in 1:sensor_size
        ]
        camera = SCMOSCamera(
            sensor_size, sensor_size, Float32(px), readnoise; offset = 0.0f0, gain = 1.0f0, qe = 1.0f0
        )
        rng = MersenneTwister(7)
        data = zeros(Float32, box, box, n)
        for (k, (x, y, z)) in enumerate(truth)
            emitters = [Emitter3D((x - 0.5) * px, (y - 0.5) * px, z, 2000.0)]
            mean_img = MicroscopePSFs.integrate_pixels(
                astig_spline, edges, edges, emitters; sampling = 8
            ) .+ 10
            for j in 1:box, i in 1:box
                λ = Float64(mean_img[i, j])
                # Poisson draw by Knuth's product (λ is below 150 here)
                threshold, count, prod = exp(-λ), 0, rand(rng)
                while prod > threshold
                    count += 1
                    prod *= rand(rng)
                end
                data[i, j, k] = count + readnoise[y_corners[k] + i - 1, x_corners[k] + j - 1] *
                    randn(rng)
            end
        end
        roi_batch = ROIBatch(data, x_corners, y_corners, Int32.(1:n), camera)
        smld, _ = fit(roi_batch; psf_model = gpsf, backend = :cpu)

        # Poisson-shifted objective of ROI k at θ = (x, y, z, photons, bg)
        function objective(k, θ)
            s = 0.0
            for j in 1:box, i in 1:box
                μ = Float64(GaussMLE.evaluate_psf(gpsf, i, j, SVector{5, Float32}(θ)))
                v = Float64(readnoise[y_corners[k] + i - 1, x_corners[k] + j - 1])^2
                s += (data[i, j, k] + v) * log(μ + v) - (μ + v)
            end
            return s
        end
        for (k, e) in enumerate(smld.emitters)
            θ = [
                e.x / px + 1.5 - x_corners[k], e.y / px + 1.5 - y_corners[k], e.z,
                e.photons, e.bg,
            ]
            scale = [1.0e-3, 1.0e-3, 1.0e-4, 1.0e-3 * θ[4], 1.0e-3 * θ[5]]
            base = objective(k, θ)
            for p in 1:5, sgn in (-1, 1)
                moved = copy(θ)
                moved[p] += sgn * scale[p]
                @test objective(k, moved) - base <= 1.0e-3
            end
        end
    end

end
