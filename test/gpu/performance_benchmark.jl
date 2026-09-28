# Performance benchmark with a correctness check: all 4 PSF models x 2 cameras x CPU and
# GPU, fits per second, and empirical std / CRLB within [0.8, 1.2] for every parameter.
using Test, GaussMLE, SMLMData, CUDA, Random, Statistics, Printf

# extract_roi_coords
include(joinpath(@__DIR__, "..", "long", "utils", "validation_utils.jl"))
include(joinpath(@__DIR__, "utils", "performance_benchmark.jl"))

results = run_comprehensive_benchmark()
@test !isempty(results)
@test all(r -> r.fits_per_second > 0, results)

# std/CRLB within 20% of optimal; allows for statistical variation
for r in results
    for (param, stats) in r.param_stats
        if isfinite(stats.std_crlb_ratio)
            ratio_ok = 0.8 <= stats.std_crlb_ratio <= 1.2
            if !ratio_ok
                @warn "$(r.config.model_name)-$(r.config.camera_symbol)-\
                    $(r.config.device_symbol): $param has std/CRLB=$(stats.std_crlb_ratio) \
                    (outside [0.8, 1.2])"
            end
            @test ratio_ok
        end
    end
end
