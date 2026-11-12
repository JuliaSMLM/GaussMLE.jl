# JointSplinePSFModel: one emitter fitted to two images at once, from a two-channel
# MicroscopePSFs MultiChannelPSF. Construction and its refusals, a fit of simulated pairs on
# two IdealCameras with different pixel sizes and a mirrored registration (near the truth),
# the pair checks, and the emitter type. Bias, scatter against the CRLB, p-values, the
# sCMOS joint fit, the joint CRLB against each view's and a view with zero transmission are
# in Long (joint_spline_psf.jl); CPU/GPU agreement is in the GPU group. MultiChannelPSF
# exists from MicroscopePSFs 0.5.7.
using Test, GaussMLE, MicroscopePSFs, SMLMData, Random, Statistics
using SMLMSim: poisson_noise

include(joinpath(@__DIR__, "long", "utils", "joint_helpers.jl"))

if !isdefined(MicroscopePSFs, :MultiChannelPSF)
    @info "joint_spline_psf.jl: MicroscopePSFs $(pkgversion(MicroscopePSFs)) has no " *
        "MultiChannelPSF (0.5.7 and later), so JointSplinePSFModel is not defined"
else
    @testset "JointSplinePSFModel" begin
        spline = joint_test_spline()
        I23 = [1.0 0 0; 0 1 0]
        # View 2 mirrored in x and shifted 0.3 um in y, 0.15 um deeper
        reg = [-1.0 0.0 16.0; 0.0 1.0 0.3]
        mc = MultiChannelPSF(;
            channels = [spline, spline], registration = [I23, reg],
            transmission = [0.45, 0.55], z_offset = [0.0, 0.15]
        )

        @testset "construction from a two-channel MultiChannelPSF" begin
            jm = JointSplinePSFModel(mc; pixel_size = 0.1)
            @test jm isa JointSplinePSFModel{Float32}
            @test jm isa GaussMLE.PSFModel{6, Float32}
            @test length(jm) == 6
            @test jm.transmission == (0.45f0, 0.55f0)
            # By default all that both channels cover: channel 2 reaches z + 0.15 <= 0.6
            @test jm.z_range[1] ≈ -0.6f0 && jm.z_range[2] ≈ 0.45f0
            @test jm.views[2].z_range[1] ≈ -0.45f0 && jm.views[2].z_range[2] ≈ 0.6f0
            # z bounds are rounded inward to Float32, as for SplinePSFModel
            zr = JointSplinePSFModel(mc; pixel_size = 0.1, z_range = (-0.3, 0.3)).z_range
            @test zr[1] ≈ -0.3f0 && zr[2] ≈ 0.3f0 && -0.3 <= zr[1] && zr[2] <= 0.3
            # The reference channel is view 1
            mc2 = MultiChannelPSF(;
                channels = [spline, spline], registration = [reg, I23],
                transmission = [0.3, 0.7], z_offset = [0.15, 0.0], reference = 2
            )
            jm2 = JointSplinePSFModel(mc2; pixel_size = 0.1)
            @test jm2.transmission == (0.7f0, 0.3f0)
            @test jm2.z_offset == (0.0f0, 0.15f0)

            one = MultiChannelPSF(; channels = [spline])
            three = MultiChannelPSF(; channels = [spline, spline, spline])
            analytic = MultiChannelPSF(; channels = [spline, ScalarPSF(1.4, 0.68, 1.518)])
            singular = MultiChannelPSF(;
                channels = [spline, spline], registration = [I23, [1.0 1.0 0; 1.0 1.0 0]]
            )
            for bad in (one, three, analytic, singular)
                @test_throws ArgumentError JointSplinePSFModel(bad; pixel_size = 0.1)
            end
            # A z_range channel 2 does not cover (z + 0.15 beyond its +0.6)
            @test_throws ArgumentError JointSplinePSFModel(
                mc; pixel_size = 0.1, z_range = (-0.3, 0.5)
            )
            @test_throws "JointSplinePSFModel" SplinePSFModel(mc; pixel_size = 0.1)
        end

        # Camera 1: the left 64 columns at 0.1 um; camera 2: a crop at 0.12 um pixels whose
        # first column is the sensor's 65th, so its pixel edges start at 7.68 um
        cam1 = IdealCamera(1:64, 1:128, 0.1)
        cam2 = IdealCamera(65:128, 1:128, 0.12)
        box, n, photons, bgs = 9, 40, 3000.0, (8.0, 12.0)
        jm = JointSplinePSFModel(mc; pixel_size = 0.1)
        rng = Xoshiro(11)
        Random.seed!(11)  # SMLMSim's noise draws from the global generator
        truths = [(2 + 3rand(rng), 2 + 8rand(rng), 0.8rand(rng) - 0.4) for _ in 1:n]
        mus, corners = joint_expected(mc, (cam1, cam2), truths; photons, bgs, box)
        frames = Int32.(1:n)
        b1 = ROIBatch(Float32.(poisson_noise(mus[1])), corners[1]..., frames, cam1)
        b2 = ROIBatch(Float32.(poisson_noise(mus[2])), corners[2]..., frames, cam2)

        @testset "fit of simulated pairs" begin
            smld, info = fit((b1, b2); psf_model = jm, backend = :cpu)
            es = smld.emitters
            @test es isa Vector{Emitter3DFitJoint{Float32}}
            @test info.n_fits == n
            @test smld.camera === cam1
            # Positions in camera 1's microns near the truth: errors in CRLB units average
            # about 0.8 when the fit is right; a wrong registration or pixel size is many
            for (row, f, sf) in ((1, :x, :σ_x), (2, :y, :σ_y), (3, :z, :σ_z))
                z = [
                    (getfield(e, f) - t[row]) / getfield(e, sf) for (e, t) in zip(es, truths)
                ]
                @test mean(abs, z) < 1.2
            end
            @test mean(e.photons for e in es) ≈ photons rtol = 0.05
            @test mean(e.bg for e in es) ≈ bgs[1] rtol = 0.05
            @test mean(e.bg2 for e in es) ≈ bgs[2] rtol = 0.05
        end

        @testset "pairs are checked" begin
            fitter = GaussMLEConfig(psf_model = jm, backend = :cpu)
            sub(b, r) = ROIBatch(
                b.data[:, :, r], b.x_corners[r], b.y_corners[r], b.frame_indices[r], b.camera
            )
            @test_throws ArgumentError fit((b1, sub(b2, 1:(n - 1))), fitter)
            small = ROIBatch(b2.data[2:8, 2:8, :], b2.x_corners, b2.y_corners, frames, cam2)
            @test_throws ArgumentError fit((b1, small), fitter)
            shifted = ROIBatch(b2.data, b2.x_corners, b2.y_corners, frames .+ Int32(1), cam2)
            @test_throws ArgumentError fit((b1, shifted), fitter)
            scmos = SCMOSCamera(64, 128, 0.12f0, 1.0f0)
            @test_throws ArgumentError fit(
                (b1, ROIBatch(b2.data, b2.x_corners, b2.y_corners, frames, scmos)), fitter
            )
            # A joint model needs a pair; a pair needs a joint model
            @test_throws ArgumentError fit(b1, fitter)
            @test_throws ArgumentError fit(b1.data, fitter)
            @test_throws ArgumentError fit(
                (b1, b2), GaussMLEConfig(psf_model = GaussianXYNB(0.13f0))
            )
        end

        @testset "Emitter3DFitJoint" begin
            names = fieldnames(Emitter3DFitJoint)
            @test names[1:18] == fieldnames(Emitter3DFitGaussMLE)
            @test names[19:20] == (:bg2, :σ_bg2)
            @test ismutabletype(Emitter3DFitJoint)
            vals = Dict(f => Float32(k) for (k, f) in enumerate(names) if k <= 14 || k > 18)
            e = Emitter3DFitJoint(; vals..., frame = 3, id = 7)
            @test e isa Emitter3DFitJoint{Float32}
            @test e.bg2 == 19.0f0 && e.σ_bg2 == 20.0f0 && e.frame == 3 && e.id == 7
            @test e.dataset == 1 && e.track_id == 0
        end
    end
end
