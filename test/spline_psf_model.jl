# SplinePSFModel (moved from PSFLearning's GaussMLE bridge): its constructors, PSFModel
# methods, backend dispatch, rectangular pixels, Array pixel size, bright emitters, and
# that IdealCamera fits (including negative pixels, and truths beyond a z_range end) end at
# a maximum of the fit objective. The sCMOS spline kernel costs about 15 s to compile on
# Julia 1.11, so its tests are in the Long group (test/long/spline_psf_scmos.jl), as are
# bias and CRLB; the GPU group checks :auto on a GPU host.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, Random
using SMLMSim: poisson_noise

include(joinpath(@__DIR__, "long", "utils", "spline_psf_helpers.jl"))

# An isbits stand-in for a device array: Adapt turns the coefficient Array into one.
struct PtrArray{T} <: AbstractArray{T, 3}
    ptr::Ptr{T}
    dims::NTuple{3, Int}
end
Base.size(a::PtrArray) = a.dims
Base.getindex(a::PtrArray, i::Int) = unsafe_load(a.ptr, i)
GaussMLE.Adapt.adapt_storage(::Type{PtrArray}, a::Array{T, 3}) where {T} =
    PtrArray{T}(pointer(a), size(a))

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
        # The memory estimate counts the table that the GPU poll counts.
        table_bytes = 4 * length(gpsf.pixel_psf.coefficients)
        @test info.memory_per_batch ==
            GaussMLE.estimate_batch_memory(n, box, 5) + table_bytes
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
        smld, _ = fit(batch(IdealCamera(64, 64, rect)); psf_model = gpsf, backend = :cpu)
        @test length(smld.emitters) == n
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
        x, y, z = 7.2, 6.7, 0.2
        direct = SplinePSFModel(astig_spline; pixel_size = (px_x, px_y))
        roi = [
            GaussMLE.evaluate_psf(direct, i, j, SVector{5, Float32}(x, y, z, 2000, 10))
                for i in 1:box, j in 1:box
        ]
        data = reshape(Float32.(roi), box, box, 1)
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
                astig_spline, edges, edges, emitters; sampling = 4
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

    @testset "IdealCamera fits with negative pixels end at a maximum of the objective" begin
        px, box = 0.1, 11
        truth = ((6.2, 5.8, -0.3), (5.7, 6.4, 0.2), (6.0, 6.0, 0.35), (6.4, 5.6, -0.1))
        n = length(truth)
        Random.seed!(11)
        # Counts at background 6 with 4 subtracted (an offset-subtracted camera): the
        # background is 2 and over a tenth of the pixels are negative.
        mu = zeros(Float32, box, box, n)
        for (k, (x, y, z)) in enumerate(truth), j in 1:box, i in 1:box
            mu[i, j, k] = GaussMLE.evaluate_psf(
                gpsf, i, j, SVector{5, Float32}(x, y, z, 1000, 6)
            )
        end
        data = Float32.(poisson_noise(mu)) .- 4
        @test count(<(0), data) > 0.05 * length(data)
        corners = ones(Int32, n)
        roi_batch = ROIBatch(data, corners, corners, Int32.(1:n), IdealCamera(64, 64, px))
        smld, _ = fit(roi_batch; psf_model = gpsf, backend = :cpu)
        check_objective_maximum(smld, gpsf, data, (k, i, j) -> 0, corners, corners, px)
    end

    @testset "IdealCamera fits beyond a z_range end finish on the bound" begin
        check_fits_beyond_z_range(astig_spline, gpsf, :ideal)
    end

    @testset "a NaN pixel stays in its own ROI and cannot index out of the table" begin
        camera = IdealCamera(64, 64, 0.1)
        config = GaussMLEConfig(psf_model = gpsf, backend = :cpu)
        poisoned = copy(data)
        poisoned[3, 4, 1] = NaN32
        smld, _ = fit(batch(camera), config)
        corners = ones(Int32, n)
        smld_nan, _ = fit(ROIBatch(poisoned, corners, corners, Int32.(1:n), camera), config)
        for k in 2:n
            @test smld_nan.emitters[k].x == smld.emitters[k].x
            @test smld_nan.emitters[k].z == smld.emitters[k].z
        end
        # Device code must not truncate a NaN coordinate: PTX and GCN convert it to 0, but
        # the compiler may treat unsafe_trunc(Int, NaN) as poison. A direct check, because
        # a fit cannot show which index was formed.
        @test GaussMLE._spline_interval(NaN32, 5) == (0, 0.0f0)
        @test GaussMLE._spline_interval(2.5f0, 5) == (2, 0.5f0)
    end


    @testset "starts: default count, validation, fixed start positions" begin
        # Plain callables keep the construction cheap; only z_range matters here.
        build(zr; kwargs...) = SplinePSFModel(
            astig_shape; pixel_size = 0.1, z_range = zr, lateral_range = 0.5, kwargs...
        )
        # one fixed start per 0.4 µm over the central 80 % of the range, plus the scan start
        for (zr, expected) in (((-1, 1), 6), ((-0.6, 0.6), 5), ((-0.5, 0.5), 4))
            @test build(zr).starts == expected
            @test GaussMLE._default_spline_starts(build(zr).z_range) == expected
        end
        @test model().starts == 4  # from a SplinePSF
        @test model(; starts = 1).starts == 1
        @test model(; starts = 7).starts == 7
        @test_throws ArgumentError build((-0.5, 0.5); starts = 0)
        @test_throws ArgumentError model(; starts = -2)
        @test occursin("starts=4", sprint(show, model()))

        # starts survive pixel-size rescaling and Adapt, and the adapted model is isbits
        m = build((-0.5, 0.5); starts = 3)
        @test GaussMLE.to_pixel_units(m, 0.11).starts == 3
        adapted = GaussMLE.Adapt.adapt(PtrArray, m)
        @test adapted.starts == 3
        @test isbitstype(typeof(adapted))

        # m = 1 fixed start at the center, else m starts evenly over the central 80 %
        m1 = build((-0.4, 0.6); starts = 2)
        @test GaussMLE._spline_start_z(m1, 0, 1) ≈ 0.1f0
        m4 = build((-1, 1))
        zs = [GaussMLE._spline_start_z(m4, q, 5) for q in 0:4]
        @test zs ≈ Float32[-0.8, -0.4, 0, 0.4, 0.8]
        m3 = build((-0.6, 0.6))
        @test [GaussMLE._spline_start_z(m3, q, 4) for q in 0:3] ≈
            Float32[-0.48, -0.16, 0.16, 0.48]
    end

    @testset "starts: the default objective is never below one start" begin
        Random.seed!(11)
        one_start = SplinePSFModel(
            astig_shape; pixel_size = 0.1, z_range = (-0.5, 0.5), starts = 1
        )
        multi = SplinePSFModel(astig_shape; pixel_size = 0.1, z_range = (-0.5, 0.5))
        box = 11
        cons = GaussMLE.default_constraints(multi, box)
        vmap = zeros(Float32, box, box)
        objective(roi, θ) = sum(
            GaussMLE._fit_objective(
                roi[i, j], GaussMLE.evaluate_psf(multi, i, j, θ), 0.0f0
            ) for j in 1:box, i in 1:box
        )
        for z in (-0.45f0, -0.2f0, 0.3f0, 0.45f0)
            roi = Float32.(poisson_noise(roi_at(multi, z; photons = 300)))
            θ0 = GaussMLE.simple_initialize(roi, box, multi)
            fits = map((one_start, multi)) do mdl
                GaussMLE._optimize(
                    roi, box, mdl, Val(false), vmap, Int32(1), Int32(1), cons, 20, θ0
                )
            end
            # the tolerance covers Float32 summation order: the kernel compares its own sums
            @test objective(roi, fits[2]) >= objective(roi, fits[1]) - 1.0f-3
        end
    end

    @testset "fit refuses a ROI beyond the lateral range" begin
        # The table is zero beyond lateral_range plus one pixel, so a ROI whose
        # half-diagonal exceeds lateral_range would be fitted with a truncated PSF.
        n_roi = 2
        function rois(mdl, box)
            data = zeros(Float32, box, box, n_roi)
            c = (box + 1) / 2
            for k in 1:n_roi, j in 1:box, i in 1:box
                data[i, j, k] = GaussMLE.evaluate_psf(
                    mdl, i, j, SVector{5, Float32}(c + 0.3k, c, 0.1, 2000, 10)
                )
            end
            return data
        end
        ideal_batch(data, camera) = ROIBatch(
            data, ones(Int32, n_roi), ones(Int32, n_roi), Int32.(1:n_roi), camera
        )
        cpu_fit(b, mdl) = fit(b; psf_model = mdl, backend = :cpu)
        camera = IdealCamera(64, 64, 0.1)
        wide = SplinePSFModel(
            astig_spline; pixel_size = 0.1, z_range = (-0.5, 0.5), lateral_range = 2.2
        )
        narrow = SplinePSFModel(
            astig_spline; pixel_size = 0.1, z_range = (-0.5, 0.5), lateral_range = 1.5
        )

        # 31 px of 0.1 um pixels: half-diagonal 2.19 um
        data31 = rois(wide, 31)
        err = try
            cpu_fit(ideal_batch(data31, camera), narrow)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        for text in ("31 px", "2.19", "1.5", "lateral_range")
            @test occursin(text, err.msg)
        end
        @test_throws ArgumentError fit(
            data31, GaussMLEConfig(psf_model = narrow, backend = :cpu)
        )
        @test_throws ArgumentError fit(
            ideal_batch(data31, camera); psf_model = narrow, backend = :auto
        )
        # 21 px: half-diagonal 1.48 um
        @test length(cpu_fit(ideal_batch(rois(narrow, 21), camera), narrow)[1].emitters) ==
            n_roi
        # The same box fits once lateral_range covers it
        smld, _ = cpu_fit(ideal_batch(data31, camera), wide)
        @test length(smld.emitters) == n_roi
        smld_a, _ = fit(data31, GaussMLEConfig(psf_model = wide, backend = :cpu))
        @test length(smld_a.emitters) == n_roi

        # Rectangular pixels: the half-diagonal uses both pixel sizes
        px = (0.1313, 0.1258)  # 21 px: 1.91 um
        rect_camera = IdealCamera(64, 64, px)
        rect(lr) = SplinePSFModel(
            astig_spline; pixel_size = px, z_range = (-0.5, 0.5), lateral_range = lr
        )
        rect_data = rois(rect(2.0), 21)
        err = try
            cpu_fit(ideal_batch(rect_data, rect_camera), rect(1.5))
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("1.91", err.msg)
        # A model built at another pixel size is judged at the camera's pixel size
        @test_throws ArgumentError cpu_fit(ideal_batch(rect_data, rect_camera), narrow)
        smld, _ = cpu_fit(ideal_batch(rect_data, rect_camera), rect(2.0))
        @test length(smld.emitters) == n_roi
    end

end
