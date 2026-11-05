# SplinePSFModel starts: one Levenberg-Marquardt fit climbs to the maximum of the basin it
# starts in, so a ROI whose scan start lies in the wrong basin ends there. The default
# adds fixed z starts and keeps the higher objective. The Long group checks this on
# aberrated vector-PSF ROIs (test/long/spline_psf_multistart.jl); this is the Core case,
# on a callable PSF with two z wells.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, Random

include(joinpath(@__DIR__, "long", "utils", "spline_psf_helpers.jl"))

@testset "SplinePSFModel: fixed z starts leave the wrong basin" begin
    px, box = 0.1, 11
    model_one = SplinePSFModel(
        two_basin_shape; pixel_size = px, z_range = (-0.5, 0.5), starts = 1
    )
    model_default = SplinePSFModel(two_basin_shape; pixel_size = px, z_range = (-0.5, 0.5))
    @test model_default.starts > 1

    # A noiseless ROI at the deep well, fitted from a start in the shallow one (the scan
    # start of a noisy defocused ROI can be there)
    z_true = 0.35f0
    roi = roi_at(model_one, z_true; box, photons = 2000)
    θ0 = SVector{5, Float32}(6, 6, -0.35, 2000, 10)
    cons = GaussMLE.default_constraints(model_one, box)
    optimize(model) = GaussMLE._optimize(
        roi, box, model, Val(false), zeros(Float32, box, box), Int32(1), Int32(1), cons,
        20, θ0
    )

    @test optimize(model_one)[3] < 0  # the single start stays in the lower well
    @test optimize(model_default)[3] ≈ z_true atol = 0.005
end
