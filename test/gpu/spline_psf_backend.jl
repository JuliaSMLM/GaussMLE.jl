# SplinePSFModel on a CUDA host: backend = :auto goes straight to the CPU. Before the
# gpu_compatible trait, the GPU loop failed to compile the model, released the context and
# polled again until auto_timeout (300 s) before falling back.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, CUDA, Logging

@testset "SplinePSFModel under :auto on a GPU host" begin
    @test CUDA.functional()

    spline = SplinePSF(
        ScalarPSF(1.4, 0.68, 1.518);
        lateral_range = 1.0, axial_range = 0.5, lateral_step = 0.05, axial_step = 0.1
    )
    model = SplinePSFModel(spline; pixel_size = 0.1)
    box, n = 11, 20
    data = zeros(Float32, box, box, n)
    for k in 1:n, j in 1:box, i in 1:box
        data[i, j, k] = GaussMLE.evaluate_psf(
            model, i, j, SVector{5, Float32}(6, 6, 0.1, 2000, 10)
        )
    end
    batch = ROIBatch(
        data, ones(Int32, n), ones(Int32, n), Int32.(1:n), IdealCamera(64, 64, 0.1)
    )

    # First call, compile included; no GPU warning (a GPU attempt would log one)
    t = @elapsed begin
        smld, info = @test_logs min_level = Logging.Warn fit(
            batch; psf_model = model, backend = :auto
        )
    end
    @test info.backend == :cpu
    @test t < 30
    @test length(smld.emitters) == n
end
