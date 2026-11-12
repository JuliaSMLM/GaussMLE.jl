# Regression: KA once chose one large work-group for a 10,000-ROI sCMOS batch and crashed
# POCL. Core runs the same check on a smaller batch to stay fast.
using Test, GaussMLE, SMLMData

@testset "Large sCMOS batch on CPU" begin
    n_rois = 10_000
    camera = SCMOSCamera(
        512, 512, 0.1f0, fill(5.0f0, 512, 512);
        offset = 100.0f0, gain = 0.5f0, qe = 0.82f0
    )
    psf_model = GaussianXYNBS()
    batch = GaussMLE.generate_roi_batch(
        camera, psf_model;
        n_rois = n_rois, roi_size = 11, seed = 42,
        corner_mode = :grid, min_spacing = 1
    )
    fitter = GaussMLEConfig(psf_model = psf_model, backend = :cpu)
    smld, _info = fit(batch, fitter)

    @test length(smld.emitters) == n_rois
    @test all(e -> isfinite(e.x) && isfinite(e.y), smld.emitters)
end
