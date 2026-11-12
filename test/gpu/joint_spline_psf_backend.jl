# JointSplinePSFModel on CPU and GPU: the same simulated pairs (IdealCamera and SCMOSCamera,
# each pair two crops of one sensor with different pixel sizes, view 2 mirrored and focused
# deeper) fitted on both backends, in several GPU batches, agree in CRLB units.
# MultiChannelPSF exists from MicroscopePSFs 0.5.7.
using Test, GaussMLE, MicroscopePSFs, SMLMData, Random, Statistics
using SMLMSim: poisson_noise

include(joinpath(@__DIR__, "utils", "gpu_vendor.jl"))
include(joinpath(@__DIR__, "..", "long", "utils", "joint_helpers.jl"))
include(joinpath(@__DIR__, "..", "long", "utils", "spline_psf_helpers.jl"))

if !isdefined(MicroscopePSFs, :MultiChannelPSF)
    @info "gpu/joint_spline_psf_backend.jl: MicroscopePSFs $(pkgversion(MicroscopePSFs)) " *
        "has no MultiChannelPSF (0.5.7 and later), so JointSplinePSFModel is not defined"
else
    @testset "JointSplinePSFModel on CPU and GPU" begin
        @test gpu_backend() !== nothing
        spline = joint_test_spline()
        mc = MultiChannelPSF(;
            channels = [spline, spline],
            registration = [[1.0 0 0; 0 1 0], [-1.0 0.0 16.0; 0.0 1.0 0.3]],
            transmission = [0.45, 0.55], z_offset = [0.0, 0.15]
        )
        jm = JointSplinePSFModel(mc; pixel_size = 0.1)
        box, n, photons, bgs = 9, 300, 3000.0, (8.0, 12.0)
        frames = Int32.(1:n)
        edges(first, count, px) = collect(((first - 1):(first - 1 + count)) .* px)
        rn1 = Float32[1.0 + 0.4 * sin(i / 7) + 0.3 * cos(j / 5) for i in 1:128, j in 1:64]
        rn2 = Float32[1.2 + 0.5 * cos(i / 9) + 0.2 * sin(j / 4) for i in 1:128, j in 1:64]
        pairs = (
            (IdealCamera(1:64, 1:128, 0.1), IdealCamera(65:128, 1:128, 0.12)),
            (
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
        fields = ((:x, :σ_x), (:y, :σ_y), (:z, :σ_z), (:photons, :σ_photons),
            (:bg, :σ_bg), (:bg2, :σ_bg2))
        @testset "$(typeof(cams[1]).name.name)" for cams in pairs
            rng = Xoshiro(31)
            Random.seed!(31)  # SMLMSim's noise draws from the global generator
            truths = [(2 + 3rand(rng), 2 + 8rand(rng), 0.8rand(rng) - 0.4) for _ in 1:n]
            mus, corners = joint_expected(mc, cams, truths; photons, bgs, box)
            data = ntuple(2) do v
                cam = cams[v]
                cam isa IdealCamera && return Float32.(poisson_noise(mus[v]))
                px = cam.pixel_edges_x[2] - cam.pixel_edges_x[1]
                return scmos_rois(
                    mus[v], cam.readnoise, corners[v]..., px; offset = cam.offset,
                    gain = cam.gain
                )
            end
            b1 = ROIBatch(data[1], corners[1]..., frames, cams[1])
            b2 = ROIBatch(data[2], corners[2]..., frames, cams[2])
            cpu, cpu_info = fit((b1, b2); psf_model = jm, backend = :cpu)
            gpu, gpu_info = fit(
                (b1, b2); psf_model = jm, backend = :gpu, batch_size = 128,
                gpu_timeout = 60.0
            )
            @test cpu_info.backend == :cpu
            @test gpu_info.backend == :gpu
            @test gpu_info.n_batches == 3
            # Each start stops within about 0.014 CRLB of the maximum of its quadratic
            # model, and Float32 differences decide accepts near the optimum, so the
            # backends stop a little apart, as for SplinePSFModel
            for (f, sf) in fields
                a = [getfield(e, f) for e in cpu.emitters]
                b = [getfield(e, f) for e in gpu.emitters]
                s = [getfield(e, sf) for e in cpu.emitters]
                @test all(isfinite, a) && all(isfinite, b) && all(isfinite, s)
                d = abs.(a .- b) ./ s
                @test maximum(d) <= 0.05
                @test count(<=(0.02), d) / length(d) >= 0.99
            end
            @test [e.pvalue for e in cpu.emitters] ≈ [e.pvalue for e in gpu.emitters] atol = 0.01
        end
    end
end
