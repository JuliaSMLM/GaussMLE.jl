# The IdealCamera p-value is calibrated at the low backgrounds of SMLM data. The Poisson
# deviance is not chi-squared at 1 to 2 photons per pixel (a 0.05 cut rejected 17 to 32 % of
# correct fits); the Pearson statistic is. Boxes are rendered by SMLMSim (exact pixel
# integrals of a 0.13 um Gaussian at sampling 8, Poisson noise), one emitter at 1000 photons
# within one pixel of the center, fitted with the true model; and the same with a second
# emitter in the box, which the one-emitter model must reject.
using Test, GaussMLE, SMLMData, SMLMSim, MicroscopePSFs, Random

@testset "IdealCamera p-value calibration" begin
    px, box, n, photons = 0.1, 15, 2000, 1000.0
    camera = IdealCamera(box, box, px)
    center = (box + 1) / 2

    function render(bg; second, seed)
        rng = Xoshiro(seed)
        Random.seed!(seed)  # SMLMSim's noise draws from the global generator
        emitters = Emitter2DFit{Float64}[]
        for k in 1:n
            x = center + 2rand(rng) - 1
            y = center + 2rand(rng) - 1
            push!(
                emitters,
                Emitter2DFit{Float64}(
                    (x - 0.5) * px, (y - 0.5) * px, photons, 0.0, 0.0, 0.0, 0.0, 0.0;
                    frame = k
                )
            )
            second && push!(
                emitters,
                Emitter2DFit{Float64}(
                    (x + 3.5) * px, (y - 0.5) * px, photons / 3, 0.0, 0.0, 0.0, 0.0, 0.0;
                    frame = k
                )
            )
        end
        images, _ = gen_images(
            BasicSMLD(emitters, camera, n, 1), GaussianPSF(0.13);
            bg = bg, poisson_noise = true, sampling = 8
        )
        ids = Int32.(1:n)
        return ROIBatch(Float32.(images), ones(Int32, n), ones(Int32, n), ids, camera)
    end

    fraction_below(batch, level) = begin
        smld, _ = fit(batch; psf_model = GaussianXYNB(0.13f0), backend = :cpu)
        sum(e -> e.pvalue < level, smld.emitters) / length(smld.emitters)
    end

    for bg in (1.0, 2.0)
        @testset "bg $bg photons per pixel" begin
            f = fraction_below(render(bg; second = false, seed = 5), 0.05f0)
            @info "bg $bg: fraction with p < 0.05, true model $f"
            @test 0.035 <= f <= 0.065
            f2 = fraction_below(render(bg; second = true, seed = 6), 0.05f0)
            @info "bg $bg: fraction with p < 0.05, two emitters $f2"
            @test f2 > 0.9
        end
    end
end
