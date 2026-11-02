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

        # The request (200,000 ROIs of box 15) exceeds the free memory of an idle device,
        # which never reaches the total: the poll says so and the batch shrinks at once.
        # Another process on the device would make the poll wait instead (gpu_timeout 10 s).
        @testset "A request above an idle device's free memory halves the batch" begin
            for backend in (:gpu, :auto)
                line = run_child("idle_full", backend)
                @test occursin("backend = :gpu", line)
                @test parse(Int, match(r"n_batches = (\d+)", line)[1]) > 1
                @test parse(Float64, match(r"seconds=([\d.]+)", line)[1]) < 9
            end
        end
    end
else
    @info "retry_faults.jl covers CUDA only; CUDA is not functional here"
end
