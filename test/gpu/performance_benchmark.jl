# Performance benchmark with a correctness check: all 4 PSF models x 2 cameras x CPU and
# GPU, fits per second, and empirical std / CRLB within [0.8, 1.2] for every parameter.
using Test, GaussMLE, SMLMData, Random, Statistics, Printf
using MicroscopePSFs: MicroscopePSFs

include(joinpath(@__DIR__, "utils", "gpu_vendor.jl"))
# extract_roi_coords
include(joinpath(@__DIR__, "..", "long", "utils", "validation_utils.jl"))
include(joinpath(@__DIR__, "utils", "performance_benchmark.jl"))

model_names = ("GaussianXYNB", "GaussianXYNBS", "GaussianXYNBSXSY", "AstigmaticXYZNB")
devices = gpu_backend() === nothing ? (:cpu,) : (:cpu, :gpu)
expected_configs = Set(Iterators.product(model_names, (:ideal, :scmos), devices))

results = run_comprehensive_benchmark()
actual_configs = Set(
    (r.config.model_name, r.config.camera_symbol, r.config.device_symbol) for r in results
)
@test length(results) == length(expected_configs)
@test actual_configs == expected_configs
@test all(r -> r.fits_per_second > 0, results)

# std/CRLB within 20% of optimal; allows for statistical variation
for r in results
    for (param, stats) in r.param_stats
        ratio_ok = isfinite(stats.std_crlb_ratio) && 0.8 <= stats.std_crlb_ratio <= 1.2
        if !ratio_ok
            @warn "$(r.config.model_name)-$(r.config.camera_symbol)-\
                $(r.config.device_symbol): $param has std/CRLB=$(stats.std_crlb_ratio) \
                (outside [0.8, 1.2])"
        end
        @test ratio_ok
    end
end

@testset "Spline / AstigmaticXYZNB throughput" begin
    comparison = run_spline_comparison_benchmark()
    expected = Set(Iterators.product(
        ("AstigmaticXYZNB", "SplinePSFModel"), (13, 15), (:ideal, :scmos), devices
    ))
    actual = Set((r.model, r.box, r.camera, r.device) for r in comparison)
    @test length(comparison) == length(expected)
    @test actual == expected
    for row in comparison
        @test row.backend == row.device
        @test row.n_fits == 10_000
        @test isfinite(row.fits_per_second) && row.fits_per_second > 0
        @test row.finite
    end
end
