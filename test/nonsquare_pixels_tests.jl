# Non-square camera pixels: every model either fits in per-axis pixel units or refuses.
# The data are noise-free and exact: a separable Gaussian in microns integrated over each
# pixel, evaluated through the per-axis pixel-unit model (σx/px_x, σy/px_y).

const NSQ_PX = (0.127f0, 0.116f0)
const NSQ_SIG = 0.13f0
const NSQ_BOX = 11
const NSQ_N, NSQ_BG = 2000.0f0, 10.0f0
const NSQ_AST = (0.13f0, 0.13f0, 0.05f0, -0.05f0, 0.01f0, -0.01f0, 0.2f0, 0.5f0)
const NSQ_XY = (
    (6.0f0, 6.0f0), (5.7f0, 6.4f0), (6.3f0, 5.8f0), (5.9f0, 5.6f0), (6.2f0, 6.3f0),
)
const NSQ_Z = (-0.3f0, -0.15f0, 0.0f0, 0.15f0, 0.3f0)

# A model that only knows square pixels, to exercise the generic 3-arg fallback.
struct NSQSquareOnlyPSF <: GaussMLE.PSFModel{4, Float32} end
GaussMLE.to_pixel_units(psf::NSQSquareOnlyPSF, pixel_size::Real) = psf

function nsq_batch(model_pixels, θs, camera)
    n = length(θs)
    data = zeros(Float32, NSQ_BOX, NSQ_BOX, n)
    for (k, θ) in enumerate(θs), j in 1:NSQ_BOX, i in 1:NSQ_BOX
        data[i, j, k] = GaussMLE.evaluate_psf(model_pixels, i, j, θ)
    end
    return ROIBatch(data, ones(Int32, n), ones(Int32, n), Int32.(1:n), camera)
end

nsq_camera() = IdealCamera(64, 64, NSQ_PX)

# Exact 2D data and the matching pixel-unit parameters (x, y, N, bg, σx, σy)
function nsq_2d_batch(camera = nsq_camera())
    θs = [
        GaussMLE.Params{6}(x, y, NSQ_N, NSQ_BG, NSQ_SIG / NSQ_PX[1], NSQ_SIG / NSQ_PX[2])
            for (x, y) in NSQ_XY
    ]
    return nsq_batch(GaussianXYNBSXSY(), θs, camera), θs
end

# CRLB of x and y in microns for fixed-σ fitting, from the exact model's Fisher information
function nsq_crlb_xy(θ6)
    F = zeros(4, 4)
    for j in 1:NSQ_BOX, i in 1:NSQ_BOX
        μ, dudt, _ = GaussMLE.compute_pixel_derivatives(i, j, θ6, GaussianXYNBSXSY())
        g = Float64.(dudt[1:4])
        F .+= g * g' ./ μ
    end
    c = sqrt.(diag(inv(F)))
    return c[1] * NSQ_PX[1], c[2] * NSQ_PX[2]
end

nsq_truth_um(x, y) = ((x - 0.5f0) * NSQ_PX[1], (y - 0.5f0) * NSQ_PX[2])

function nsq_error(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

@testset "Non-square pixels" begin
    @testset "to_pixel_units per axis" begin
        # Square pixels keep the existing model and numerics
        sq = GaussMLE.to_pixel_units(GaussianXYNB(0.13f0), 0.1f0, 0.1f0)
        @test sq isa GaussianXYNB{Float32}
        @test sq.σ ≈ 1.3f0

        # The generic fallback forwards on square pixels and refuses non-square ones
        square_only = NSQSquareOnlyPSF()
        @test GaussMLE.to_pixel_units(square_only, 0.1f0, 0.1f0) === square_only
        @test_throws ArgumentError GaussMLE.to_pixel_units(square_only, NSQ_PX...)

        astig = GaussMLE.to_pixel_units(AstigmaticXYZNB{Float32}(NSQ_AST...), NSQ_PX...)
        @test astig.σx₀ ≈ NSQ_AST[1] / NSQ_PX[1]
        @test astig.σy₀ ≈ NSQ_AST[2] / NSQ_PX[2]
        @test (astig.γ, astig.d) == (NSQ_AST[7], NSQ_AST[8])
    end

    @testset "GaussianXYNB" begin
        batch, θs = nsq_2d_batch()
        smld, _ = fit(batch; psf_model = GaussianXYNB(NSQ_SIG), backend = :cpu)
        for (e, θ) in zip(smld.emitters, θs)
            tx, ty = nsq_truth_um(θ[1], θ[2])
            @test e.x ≈ tx atol = 1.0e-4
            @test e.y ≈ ty atol = 1.0e-4
            @test e.photons ≈ NSQ_N rtol = 1.0e-3
            @test e.bg ≈ NSQ_BG rtol = 1.0e-3
            cx, cy = nsq_crlb_xy(θ)
            @test e.σ_x ≈ cx rtol = 1.0e-3
            @test e.σ_y ≈ cy rtol = 1.0e-3
        end
    end

    @testset "GaussianXYNBSXSY (regression)" begin
        batch, θs = nsq_2d_batch()
        smld, _ = fit(batch; psf_model = GaussianXYNBSXSY(), backend = :cpu)
        for e in smld.emitters
            @test e.σx ≈ NSQ_SIG rtol = 1.0e-3
            @test e.σy ≈ NSQ_SIG rtol = 1.0e-3
            @test e.photons ≈ NSQ_N rtol = 1.0e-3
        end
    end

    @testset "AstigmaticXYZNB" begin
        pix = AstigmaticXYZNB{Float32}(
            NSQ_AST[1] / NSQ_PX[1], NSQ_AST[2] / NSQ_PX[2], NSQ_AST[3:end]...
        )
        θs = [
            GaussMLE.Params{5}(x, y, z, NSQ_N, NSQ_BG)
                for ((x, y), z) in zip(NSQ_XY, NSQ_Z)
        ]
        batch = nsq_batch(pix, θs, nsq_camera())
        model = AstigmaticXYZNB{Float32}(NSQ_AST...)
        smld, _ = fit(batch; psf_model = model, backend = :cpu, iterations = 50)
        for (e, θ) in zip(smld.emitters, θs)
            tx, ty = nsq_truth_um(θ[1], θ[2])
            @test e.x ≈ tx atol = 1.0e-4
            @test e.y ≈ ty atol = 1.0e-4
            @test e.z ≈ θ[3] atol = 1.0e-3
            @test e.photons ≈ NSQ_N rtol = 1.0e-3
        end
    end

    @testset "GaussianXYNBS refuses" begin
        for camera in (nsq_camera(), SCMOSCamera(64, 64, NSQ_PX, 1.6f0))
            batch, _ = nsq_2d_batch(camera)
            err = nsq_error(() -> fit(batch; psf_model = GaussianXYNBS(), backend = :cpu))
            @test err isa ArgumentError
            @test err isa ArgumentError && occursin("GaussianXYNBSXSY", err.msg)
        end
    end

    @testset "generate_roi_batch per axis" begin
        n = 400
        params = zeros(Float32, 4, n)
        params[1, :] .= 6.0f0
        params[2, :] .= 6.0f0
        params[3, :] .= NSQ_N
        params[4, :] .= NSQ_BG
        batch = generate_roi_batch(
            nsq_camera(), GaussianXYNB(NSQ_SIG);
            n_rois = n, true_params = params, corners = ones(Int32, 2, n), seed = 7
        )
        smld, _ = fit(batch; psf_model = GaussianXYNBSXSY(), backend = :cpu)
        @test mean(e.σx for e in smld.emitters) ≈ NSQ_SIG rtol = 1.0e-2
        @test mean(e.σy for e in smld.emitters) ≈ NSQ_SIG rtol = 1.0e-2
    end

    if GPU_AVAILABLE
        @testset "GaussianXYNB on GPU" begin
            batch, θs = nsq_2d_batch()
            smld, _ = fit(batch; psf_model = GaussianXYNB(NSQ_SIG), backend = :gpu)
            for (e, θ) in zip(smld.emitters, θs)
                tx, ty = nsq_truth_um(θ[1], θ[2])
                @test e.x ≈ tx atol = 1.0e-4
                @test e.y ≈ ty atol = 1.0e-4
                @test e.photons ≈ NSQ_N rtol = 1.0e-3
            end
        end
    end
end
