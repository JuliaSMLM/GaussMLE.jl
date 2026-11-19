# AstigmaticXYZNB fits each ROI a second time from the first fit's end point with z negated
# and keeps the higher objective. With weak astigmatism (a focal offset of 0.05 um against a
# depth scale of 0.4 um) the likelihood has a second maximum near the mirrored z, and one
# start ends in the lower of the two for some emitters. The Long group measures this
# against an independent maximizer; this Core test (GitHub CI runs Core only) guards the
# second start.
using Test, GaussMLE, StaticArrays, Random
using SMLMSim: poisson_noise

@testset "AstigmaticXYZNB's second start reaches the higher of the two z maxima" begin
    psf = AstigmaticXYZNB{Float32}(
        0.13f0, 0.13f0, 0.05f0, 0.05f0, 0.3f0, 0.3f0, 0.05f0, 0.4f0
    )
    pm = GaussMLE.to_pixel_units(psf, 0.1f0)
    box, n = 13, 300
    rng = Xoshiro(11)
    Random.seed!(11)  # SMLMSim's noise draws from the global generator
    truth = [
        SVector{5, Float32}(
            7 + rand(rng) - 0.5f0, 7 + rand(rng) - 0.5f0,
            (rand(rng) < 0.5 ? -1 : 1) * (0.2f0 + 0.3f0 * rand(rng)), 300, 2
        ) for _ in 1:n
    ]
    mu = [GaussMLE.evaluate_psf(pm, i, j, truth[k]) for i in 1:box, j in 1:box, k in 1:n]
    data = Float32.(poisson_noise(mu))
    cons = GaussMLE.default_constraints(pm, box)
    vm = zeros(Float32, box, box)
    iterations = GaussMLE._default_iterations(pm)
    objective(roi, θ) = sum(
        GaussMLE._fit_objective(
            roi[i, j], GaussMLE.compute_pixel_derivatives(i, j, θ, pm)[1], 0.0f0
        ) for j in 1:box, i in 1:box
    )
    # Objective below the higher of the two maxima (each reached from the truth's z and from
    # its mirror), for one start and for the released fit
    one_start = zeros(n)
    released = zeros(n)
    for k in 1:n
        roi = view(data, :, :, k)
        lm(θ0) = GaussMLE._lm_fit(
            roi, box, pm, Val(false), vm, Int32(1), Int32(1), cons, iterations, θ0
        )[1]
        best = max(
            objective(roi, lm(truth[k])),
            objective(roi, lm(Base.setindex(truth[k], -truth[k][3], 3)))
        )
        θ0 = GaussMLE.simple_initialize(roi, box, pm)
        one_start[k] = best - objective(roi, lm(θ0))
        θ = GaussMLE._optimize(
            roi, box, pm, Val(false), vm, Int32(1), Int32(1), cons, iterations, θ0
        )[1]
        released[k] = best - objective(roi, θ)
    end
    @test count(>(0.5), one_start) > 0  # one start ends in the lower maximum for some ROIs
    @test count(>(0.5), released) == 0
end
