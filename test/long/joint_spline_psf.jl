# JointSplinePSFModel accuracy on simulated pairs: bias, scatter against the CRLB and p-value
# calibration for an IdealCamera pair and an SCMOSCamera pair, each pair two crops of one
# sensor with different pixel sizes (and for sCMOS different calibration), view 2 mirrored
# in x and focused 0.15 um deeper; the joint CRLB below either view's alone; a view with
# zero transmission reproducing the one-view fit (the same likelihood). Data follow
# MicroscopePSFs' MultiChannelPSF forward law (test/long/utils/joint_helpers.jl) with
# SMLMSim noise. MultiChannelPSF exists from MicroscopePSFs 0.5.7.
using Test, GaussMLE, MicroscopePSFs, SMLMData, Random, Statistics
using SMLMSim: poisson_noise

include(joinpath(@__DIR__, "utils", "joint_helpers.jl"))
include(joinpath(@__DIR__, "utils", "spline_psf_helpers.jl"))

if !isdefined(MicroscopePSFs, :MultiChannelPSF)
    @info "long/joint_spline_psf.jl: MicroscopePSFs $(pkgversion(MicroscopePSFs)) has no " *
        "MultiChannelPSF (0.5.7 and later), so JointSplinePSFModel is not defined"
else
    @testset "JointSplinePSFModel accuracy" begin
        spline = joint_test_spline()
        mc = MultiChannelPSF(;
            channels = [spline, spline],
            registration = [[1.0 0 0; 0 1 0], [-1.0 0.0 16.0; 0.0 1.0 0.3]],
            transmission = [0.45, 0.55], z_offset = [0.0, 0.15]
        )
        jm = JointSplinePSFModel(mc; pixel_size = 0.1)
        box, n, photons, bgs = 9, 1500, 3000.0, (8.0, 12.0)
        frames = Int32.(1:n)

        # Both views of one sensor: columns 1:64 at 0.1 um and a crop starting at column 65
        # at 0.12 um (its pixel edges start at 7.68 um)
        edges(first, count, px) = collect(((first - 1):(first - 1 + count)) .* px)
        rn1 = Float32[1.0 + 0.4 * sin(i / 7) + 0.3 * cos(j / 5) for i in 1:128, j in 1:64]
        rn2 = Float32[1.2 + 0.5 * cos(i / 9) + 0.2 * sin(j / 4) for i in 1:128, j in 1:64]
        cameras = Dict(
            "IdealCamera" => (IdealCamera(1:64, 1:128, 0.1), IdealCamera(65:128, 1:128, 0.12)),
            "SCMOSCamera" => (
                SCMOSCamera(
                    edges(1, 64, 0.1f0), edges(1, 128, 0.1f0);
                    readnoise = rn1, offset = 100.0f0, gain = 0.5f0, qe = 1.0f0
                ),
                SCMOSCamera(
                    edges(65, 64, 0.12f0), edges(1, 128, 0.12f0);
                    readnoise = rn2, offset = 90.0f0, gain = 0.8f0, qe = 1.0f0
                ),
            ),
        )

        for name in ("IdealCamera", "SCMOSCamera")
            @testset "$name pair" begin
                cam1, cam2 = cameras[name]
                rng = Xoshiro(21)
                Random.seed!(21)  # SMLMSim's noise draws from the global generator
                truths = [(2 + 3rand(rng), 2 + 8rand(rng), 0.8rand(rng) - 0.4) for _ in 1:n]
                mus, corners = joint_expected(mc, (cam1, cam2), truths; photons, bgs, box)
                data = ntuple(2) do v
                    cam = (cam1, cam2)[v]
                    cam isa IdealCamera && return Float32.(poisson_noise(mus[v]))
                    px = cam.pixel_edges_x[2] - cam.pixel_edges_x[1]
                    return scmos_rois(
                        mus[v], cam.readnoise, corners[v]..., px;
                        offset = cam.offset, gain = cam.gain
                    )
                end
                b1 = ROIBatch(data[1], corners[1]..., frames, cam1)
                b2 = ROIBatch(data[2], corners[2]..., frames, cam2)
                smld, info = fit((b1, b2); psf_model = jm, backend = :cpu)
                es = smld.emitters
                @test info.n_fits == n
                @test info.n_converged >= 0.99n
                for (row, f, sf) in ((1, :x, :σ_x), (2, :y, :σ_y), (3, :z, :σ_z))
                    err = [Float64(getfield(e, f)) for e in es] .- getindex.(truths, row)
                    crlb = sqrt(mean(Float64(getfield(e, sf))^2 for e in es))
                    @test abs(mean(err)) < 3 * std(err) / sqrt(n)
                    @test 0.93 < std(err) / crlb < 1.07
                end
                @test mean(e.photons for e in es) ≈ photons rtol = 0.01
                @test mean(e.bg for e in es) ≈ bgs[1] rtol = 0.01
                @test mean(e.bg2 for e in es) ≈ bgs[2] rtol = 0.01
                @test 0.03 < mean(e.pvalue < 0.05 for e in es) < 0.07
                # Both images inform every parameter: the joint CRLB is below either view's
                for (v, b) in ((1, b1), (2, b2))
                    alone, _ = fit(b; psf_model = jm.views[v], backend = :cpu)
                    @test mean(e.σ_x for e in es) < mean(e.σ_x for e in alone.emitters)
                    @test mean(e.σ_z for e in es) < mean(e.σ_z for e in alone.emitters)
                end
            end
        end

        @testset "a view with zero transmission leaves the one-view fit" begin
            # View 2 receives no light: the joint fit's x, y, z and photons are view 1's
            # own SplinePSFModel fit, to the stop rule (each within 0.014 CRLB of the max)
            mc0 = MultiChannelPSF(;
                channels = [spline, spline],
                registration = [[1.0 0 0; 0 1 0], [-1.0 0.0 16.0; 0.0 1.0 0.3]],
                transmission = [1.0, 0.0], z_offset = [0.0, 0.15]
            )
            jm0 = JointSplinePSFModel(mc0; pixel_size = 0.1)
            cam1, cam2 = cameras["IdealCamera"]
            m = 300
            rng = Xoshiro(23)
            Random.seed!(23)
            truths = [(2 + 3rand(rng), 2 + 8rand(rng), 0.8rand(rng) - 0.4) for _ in 1:m]
            mus0, corners0 = joint_expected(mc0, (cam1, cam2), truths; photons, bgs, box)
            fr = Int32.(1:m)
            c1 = ROIBatch(Float32.(poisson_noise(mus0[1])), corners0[1]..., fr, cam1)
            c2 = ROIBatch(Float32.(poisson_noise(mus0[2])), corners0[2]..., fr, cam2)
            joint, _ = fit((c1, c2); psf_model = jm0, backend = :cpu)
            alone, _ = fit(c1; psf_model = jm0.views[1], backend = :cpu)
            for (f, sf) in ((:x, :σ_x), (:y, :σ_y), (:z, :σ_z), (:photons, :σ_photons))
                d = [
                    abs(getfield(a, f) - getfield(b, f)) / getfield(b, sf)
                        for (a, b) in zip(joint.emitters, alone.emitters)
                ]
                @test maximum(d) < 0.05
            end
        end
    end
end
