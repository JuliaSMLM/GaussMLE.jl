# SplinePSFModel with an SCMOSCamera: raw-ADU recovery, objective maximum for noisy data
# (including pixels with data + read-noise variance <= 0), truths beyond a z_range end, and
# a rectangular-pixel camera. Moved from Core: this kernel costs about 15 s to compile on
# Julia 1.11.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, Random

include(joinpath(@__DIR__, "utils", "spline_psf_helpers.jl"))

@testset "SplinePSFModel with an SCMOSCamera" begin
    zc = ZernikeCoefficients(15)
    zc.phase[6] = 0.5  # vertical astigmatism, 0.5 rad RMS
    astig_spline = SplinePSF(
        ScalarPSF(1.4, 0.68, 1.518; zernike_coeffs = zc);
        lateral_range = 1.0, axial_range = 0.5, lateral_step = 0.05, axial_step = 0.1
    )
    gpsf = SplinePSFModel(astig_spline; pixel_size = 0.1)

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
            sensor_size, sensor_size, Float32(px), readnoise;
            offset = 0.0f0, gain = 1.0f0, qe = 1.0f0
        )
        Random.seed!(7)
        mu = zeros(Float32, box, box, n)
        for (k, (x, y, z)) in enumerate(truth), j in 1:box, i in 1:box
            mu[i, j, k] = GaussMLE.evaluate_psf(
                gpsf, i, j, SVector{5, Float32}(x, y, z, 2000, 10)
            )
        end
        data = scmos_rois(mu, readnoise, x_corners, y_corners, px)
        roi_batch = ROIBatch(data, x_corners, y_corners, Int32.(1:n), camera)
        smld, _ = fit(roi_batch; psf_model = gpsf, backend = :cpu)

        v(k, i, j) = readnoise[y_corners[k] + i - 1, x_corners[k] + j - 1]^2
        check_objective_maximum(smld, gpsf, data, v, x_corners, y_corners, px)
    end


    @testset "sCMOS fits with data + readnoise^2 <= 0 end at an objective maximum" begin
        px, box, sensor_size = 0.1, 11, 40
        truth = ((6.2, 5.8, -0.2), (5.7, 6.4, 0.2), (6.0, 6.0, 0.1), (6.4, 5.6, -0.1))
        n = length(truth)
        x_corners, y_corners = Int32[3, 25, 6, 28], Int32[5, 8, 24, 27]
        # Low background and read noise of about 1 e-: the electrons (offset 0, gain 1) are
        # often negative, and about 5 % of the pixels have d + v <= 0, the objective's
        # s <= 0 branch.
        readnoise = [
            sqrt(1.0f0 + 0.01f0 * (i - 1) + 0.02f0 * (j - 1))
                for i in 1:sensor_size, j in 1:sensor_size
        ]
        camera = SCMOSCamera(
            sensor_size, sensor_size, Float32(px), readnoise;
            offset = 0.0f0, gain = 1.0f0, qe = 1.0f0
        )
        Random.seed!(13)
        mu = zeros(Float32, box, box, n)
        for (k, (x, y, z)) in enumerate(truth), j in 1:box, i in 1:box
            mu[i, j, k] = GaussMLE.evaluate_psf(
                gpsf, i, j, SVector{5, Float32}(x, y, z, 800, 0.5)
            )
        end
        data = scmos_rois(mu, readnoise, x_corners, y_corners, px)
        v(k, i, j) = readnoise[y_corners[k] + i - 1, x_corners[k] + j - 1]^2
        nonpositive = [data[i, j, k] + v(k, i, j) <= 0 for i in 1:box, j in 1:box, k in 1:n]
        @test count(nonpositive) > 0.02 * length(data)
        roi_batch = ROIBatch(data, x_corners, y_corners, Int32.(1:n), camera)
        smld, _ = fit(roi_batch; psf_model = gpsf, backend = :cpu)
        check_objective_maximum(smld, gpsf, data, v, x_corners, y_corners, px)
    end

    @testset "SCMOSCamera fits beyond a z_range end finish on the bound" begin
        check_fits_beyond_z_range(astig_spline, gpsf, :scmos)
    end

    @testset "non-square pixels" begin
        rect = (0.127, 0.116)  # PSFLearning's default config
        box, n = 11, 4
        data = zeros(Float32, box, box, n)
        for k in 1:n, j in 1:box, i in 1:box
            data[i, j, k] = GaussMLE.evaluate_psf(
                gpsf, i, j, SVector{5, Float32}(6, 6, 0.1k - 0.2, 2000, 10)
            )
        end
        corners = ones(Int32, n)
        camera = SCMOSCamera(64, 64, rect, 1.6)
        batch = ROIBatch(data, corners, corners, Int32.(1:n), camera)
        smld, _ = fit(batch; psf_model = gpsf, backend = :cpu)
        @test length(smld.emitters) == n
    end
end
