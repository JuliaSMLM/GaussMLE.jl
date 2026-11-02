# The GPU retry loop against real CUDA faults, each injected in a child Julia process so no
# method is overwritten in the group's own process. The child inherits this process's
# environment and project.
using Test, GaussMLE, CUDA

if CUDA.functional()
    # Run utils/retry_fault_child.jl; return its RESULT line.
    function run_child(scenario, backend; env = String[])
        script = joinpath(@__DIR__, "utils", "retry_fault_child.jl")
        command = addenv(
            `$(Base.julia_cmd()) --project=$(Base.active_project()) $script $scenario
            $backend`,
            env,
        )
        output = IOBuffer()
        run(pipeline(ignorestatus(command); stdout = output, stderr = devnull))
        lines = filter(startswith("RESULT"), split(String(take!(output)), '\n'))
        return isempty(lines) ? "" : only(lines)
    end

    @testset "GPU retry faults (CUDA)" begin
        @testset "Total device memory query" begin
            totals = GaussMLE._gpu_total_memory(GaussMLE.CUDAGPU())
            @test length(totals) == length(CUDA.devices())
            @test all(>(0), totals)
        end

        @testset "Launch failure (719) under :auto falls back to the CPU" begin
            line = run_child("launch_failed", :auto)
            @test occursin("backend = :cpu", line)
            @test occursin("n = 200", line)
            @test parse(Float64, match(r"seconds=([\d.]+)", line)[1]) < 20
        end

        @testset "Launch failure (719) under :gpu raises the CuError at once" begin
            line = run_child("launch_failed", :gpu)
            @test occursin("CuError", line)
            @test occursin("code = $(Int(CUDA.ERROR_LAUNCH_FAILED))", line)
            @test parse(Float64, match(r"seconds=([\d.]+)", line)[1]) < 20
        end

        @testset "Out-of-memory under a hard limit halves the batch" begin
            # The ROI array of one 200,000-ROI batch of box 15 (180 MB) cannot fit a limit
            # of 60 % of its size; half the batch, with its outputs, can. The driver's pool
            # rounds a small limit up (to 32 MiB on CUDA 12), so the batch must be large.
            # (The estimate used for polling is larger than what the batch allocates.)
            data_bytes = 15 * 15 * 200_000 * sizeof(Float32)
            limit = string(round(Int, 0.6 * data_bytes))
            line = run_child(
                "oom", :gpu; env = ["JULIA_CUDA_HARD_MEMORY_LIMIT=$limit"],
            )
            @test occursin("backend = :gpu", line)
            @test parse(Int, match(r"n_batches = (\d+)", line)[1]) > 1
            @test parse(Float64, match(r"seconds=([\d.]+)", line)[1]) < 30
        end

        # In this process, not a child: a child would see this process's context as
        # another process on the device and, correctly, wait. A filler leaves about 150 MB
        # free, so the request for 200,000 ROIs of box 15 exceeds the free memory of an idle
        # device, which never reaches the total: the poll says so and the batch shrinks at
        # once. It needs one visible device with no other process on it; another process
        # makes the poll wait instead, the intended behavior, which is not tested here.
        idle = length(CUDA.devices()) == 1 && GaussMLE._other_process_present(0) === false
        if idle
            @testset "A request above an idle device's free memory halves the batch" begin
                psf_model = GaussianXYNB(0.13f0)
                camera = IdealCamera(64, 64, 0.1)
                warm = GaussMLE.generate_roi_batch(
                    camera, psf_model; n_rois = 200, roi_size = 15, seed = 1
                )
                fit(
                    convert(Array{Float32, 3}, warm.data),
                    GaussMLEConfig(; psf_model, backend = :gpu)
                )
                n = 200_000
                batch = GaussMLE.generate_roi_batch(
                    camera, psf_model; n_rois = n, roi_size = 15, seed = 1
                )
                data = convert(Array{Float32, 3}, batch.data)
                for backend in (:gpu, :auto)
                    GC.gc(true)
                    CUDA.reclaim()
                    filler = CuArray{UInt8}(undef, CUDA.free_memory() - 150 * 1024^2)
                    CUDA.synchronize()
                    fitter = GaussMLEConfig(;
                        psf_model, backend, batch_size = n, gpu_timeout = 10.0,
                        auto_timeout = 10.0,
                    )
                    t = @elapsed smld, info = fit(data, fitter)
                    CUDA.unsafe_free!(filler)
                    @test info.backend == :gpu
                    @test info.n_batches > 1
                    @test length(smld.emitters) == n
                    @test t < 9  # a regression waits for the 10 s timeout
                end
                CUDA.reclaim()
            end
        else
            @info "retry_faults.jl: the idle-device test needs one visible GPU with no " *
                "other process on it"
        end
    end
else
    @info "retry_faults.jl covers CUDA only; CUDA is not functional here"
end
