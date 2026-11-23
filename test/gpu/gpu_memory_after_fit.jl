# After a GPU fit the process holds none of the fit's device memory: the fit frees its own
# device arrays (data, corners, outputs, the variance map, the device copy of a PSF table)
# and returns the pool's unused memory, without waiting on the garbage collector. Checked
# with the collector switched off during the fit, so only an explicit free can pass. CUDA
# only: AMDGPU has no matching pool queries, and its free and reclaim calls run in every GPU
# fit of this group.
using Test, GaussMLE, CUDA, SMLMData, MicroscopePSFs, Random

include(joinpath(@__DIR__, "..", "long", "utils", "joint_helpers.jl"))

if CUDA.functional() && CUDA.used_memory() === missing
    # JULIA_CUDA_MEMORY_POOL=none: no pool, so no pool counters to compare
    @info "GPU memory after a fit: CUDA.jl uses no memory pool here; nothing to measure"
elseif CUDA.functional()
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
        # 200,000 11 x 11 ROIs: about 100 MiB of device arrays in one batch
        n = 200_000
        big = rois(11, n)
        ideal_batch = ROIBatch(big, corners(n), corners(n), Int32.(1:n), ideal)
        # unused pool memory a fit may leave: the trim returns it all; where CUDA.jl lacks
        # the trim, the package's threshold for a full reclaim
        limit = GaussMLE._cuda_trim_available() ? 64 * 2^20 : GaussMLE._GPU_CACHE_TRIM_BYTES
        cases = (
            ("GaussianXYNB, IdealCamera", ideal_batch, GaussianXYNB(0.13f0), n, false),
            (
                "GaussianXYNB, SCMOSCamera (a variance map on the device)",
                ROIBatch(big, corners(n), corners(n), Int32.(1:n), scmos),
                GaussianXYNB(0.13f0), n, false,
            ),
            (
                "SplinePSFModel (a PSF table on the device)",
                ROIBatch(rois(9, 20_000), corners(20_000), corners(20_000), Int32.(1:20_000), ideal),
                SplinePSFModel(joint_test_spline(); pixel_size = 0.1), 20_000, false,
            ),
            ("GaussianXYNB, four batches", ideal_batch, GaussianXYNB(0.13f0), n ÷ 4, false),
            # CUDA.jl defers the free of memory last used on the per-thread stream
            (
                "GaussianXYNB on the per-thread stream", ideal_batch, GaussianXYNB(0.13f0),
                n, true,
            ),
        )
        for (name, batch, model, nb, per_thread) in cases
            @testset "$name" begin
                fit1() = fit(batch; psf_model = model, backend = :gpu, batch_size = nb, iterations = 20)
                run() = per_thread ? CUDA.stream!(fit1, CUDA.per_thread_stream()) : fit1()
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
                @test info.n_batches == cld(size(batch.data, 3), nb)
                @test CUDA.used_memory() <= used
                # the pool's unused memory is returned
                @test CUDA.cached_memory() - CUDA.used_memory() <= limit
            end
        end
    end
end
