# After a GPU fit the process holds none of the fit's device memory: the fit frees its own
# device arrays (data, corners, outputs, the variance map, the device copy of a PSF table)
# and trims the pool, without waiting on the garbage collector. Checked with the collector
# switched off during the fit, so only an explicit free can pass. CUDA only: AMDGPU has no
# matching pool queries, and its free and reclaim calls run in every GPU fit of this group.
using Test, GaussMLE, CUDA, SMLMData, MicroscopePSFs, Random

include(joinpath(@__DIR__, "..", "long", "utils", "joint_helpers.jl"))

if CUDA.functional()
    @testset "GPU memory after a fit" begin
        Random.seed!(7)
        rois(box, n) = begin
            data = 10.0f0 .+ 5.0f0 .* rand(Float32, box, box, n)
            c = (box + 1) ÷ 2
            data[(c - 1):(c + 1), (c - 1):(c + 1), :] .+= 100.0f0
            data
        end
        corners(n) = fill(Int32(20), n)
        ideal = IdealCamera(64, 64, 0.1)
        scmos = SCMOSCamera(64, 64, 0.1f0, 1.5f0)
        # one batch of 200,000 11 x 11 ROIs: about 100 MiB of device arrays
        n = 200_000
        big = rois(11, n)
        cases = (
            (
                "GaussianXYNB, IdealCamera",
                ROIBatch(big, corners(n), corners(n), Int32.(1:n), ideal),
                GaussianXYNB(0.13f0), n,
            ),
            (
                "GaussianXYNB, SCMOSCamera (a variance map on the device)",
                ROIBatch(big, corners(n), corners(n), Int32.(1:n), scmos),
                GaussianXYNB(0.13f0), n,
            ),
            (
                "SplinePSFModel (a PSF table on the device)",
                ROIBatch(rois(9, 20_000), corners(20_000), corners(20_000), Int32.(1:20_000), ideal),
                SplinePSFModel(joint_test_spline(); pixel_size = 0.1), 20_000,
            ),
        )
        for (name, batch, model, nb) in cases
            @testset "$name" begin
                run() = fit(batch; psf_model = model, backend = :gpu, batch_size = nb, iterations = 20)
                run()  # compile, and let earlier device garbage go
                GC.gc(true)
                CUDA.reclaim()
                used = CUDA.used_memory()
                GC.enable(false)
                info = try
                    run()[2]
                finally
                    GC.enable(true)
                end
                @test info.backend == :gpu
                @test info.n_batches == 1
                @test CUDA.used_memory() <= used
                # the pool is trimmed: far less cached than the batch needed
                @test CUDA.cached_memory() <= 64 * 2^20
            end
        end
    end
end
