# The CPU path for the Gaussian models Core does not fit (Core runs GaussianXYNB and keeps
# the Julia 1.11 compile time of each kernel out of its budget): fit(::ROIBatch) for
# GaussianXYNBS and GaussianXYNBSXSY, and the unified kernel called directly.
using Test, GaussMLE, SMLMData, KernelAbstractions

@testset "Gaussian models on the CPU" begin

    @testset "Different PSF Models" begin
        camera = SMLMData.IdealCamera(256, 256, 0.1)

        psf_models = [
            GaussMLE.GaussianXYNB(0.13f0),
            GaussMLE.GaussianXYNBS(),
            GaussMLE.GaussianXYNBSXSY(),
        ]

        for psf in psf_models
            batch = GaussMLE.generate_roi_batch(camera, psf; n_rois = 5, seed = 42)
            fitter = GaussMLE.GaussMLEConfig(
                psf_model = psf, backend = :cpu,
            )
            smld, _info = GaussMLE.fit(batch, fitter)

            @test length(smld.emitters) == 5
            # Check that uncertainties are finite
            σ_x_vals = [e.σ_x for e in smld.emitters]
            σ_photons_vals = [e.σ_photons for e in smld.emitters]
            @test !any(isinf.(σ_x_vals))
            @test !any(isnan.(σ_x_vals))
            @test !any(isinf.(σ_photons_vals))
            @test !any(isnan.(σ_photons_vals))
        end
    end

    # Test configuration
    n_test_blobs = 100
    box_size = 7
    iterations = 100  # the Gaussian models' default cap

    # Generate test data
    function generate_simple_test_data(n_blobs, box_size)
        data = zeros(Float32, box_size, box_size, n_blobs)
        center = Float32((box_size + 1) / 2)

        for k in 1:n_blobs
            # Simple Gaussian blob
            for j in 1:box_size, i in 1:box_size
                dx = Float32(i) - center
                dy = Float32(j) - center
                gaussian = 1000.0f0 * exp(-(dx^2 + dy^2) / (2 * 1.3f0^2))
                data[i, j, k] = gaussian + 10.0f0  # Add background
            end
            # Add some noise
            data[:, :, k] .+= randn(Float32, box_size, box_size) * 5.0f0
        end

        return data
    end

    @testset "Unified kernel, variable-width models" begin
        data = generate_simple_test_data(n_test_blobs, box_size)

        @testset "GaussianXYNBS (N=5)" begin
            psf_model = GaussMLE.GaussianXYNBS{Float32}()
            constraints = GaussMLE.default_constraints(psf_model, box_size)

            results = Matrix{Float32}(undef, 5, n_test_blobs)
            uncertainties = Matrix{Float32}(undef, 5, n_test_blobs)
            covariances = Matrix{Float32}(undef, 3, n_test_blobs)
            chi2 = Vector{Float32}(undef, n_test_blobs)

            backend = KernelAbstractions.CPU()
            kernel = GaussMLE.unified_gaussian_mle_kernel!(
                backend, GaussMLE.CPU_WORKGROUP_SIZE
            )

            # Create dummy corners and variance for new kernel signature
            x_corners = Int32[1 + (i - 1) * box_size for i in 1:n_test_blobs]
            y_corners = fill(Int32(1), n_test_blobs)
            variance_map = zeros(Float32, 512, 512)

            kernel(
                results, uncertainties, covariances, chi2,
                zeros(UInt8, size(results, 2)),
                data, psf_model, Val(false), variance_map, x_corners, y_corners,
                constraints, iterations,
                ndrange = n_test_blobs
            )

            @test all(isfinite.(results))
            @test all(uncertainties .> 0)
            @test all(isfinite.(covariances))
            @test all(isfinite.(chi2) .& (chi2 .>= 0))
        end

        @testset "GaussianXYNBSXSY (N=6)" begin
            psf_model = GaussMLE.GaussianXYNBSXSY{Float32}()
            constraints = GaussMLE.default_constraints(psf_model, box_size)

            results = Matrix{Float32}(undef, 6, n_test_blobs)
            uncertainties = Matrix{Float32}(undef, 6, n_test_blobs)
            covariances = Matrix{Float32}(undef, 3, n_test_blobs)
            chi2 = Vector{Float32}(undef, n_test_blobs)

            backend = KernelAbstractions.CPU()
            kernel = GaussMLE.unified_gaussian_mle_kernel!(
                backend, GaussMLE.CPU_WORKGROUP_SIZE
            )

            # Create dummy corners and variance for new kernel signature
            x_corners = Int32[1 + (i - 1) * box_size for i in 1:n_test_blobs]
            y_corners = fill(Int32(1), n_test_blobs)
            variance_map = zeros(Float32, 512, 512)

            kernel(
                results, uncertainties, covariances, chi2,
                zeros(UInt8, size(results, 2)),
                data, psf_model, Val(false), variance_map, x_corners, y_corners,
                constraints, iterations,
                ndrange = n_test_blobs
            )

            @test all(isfinite.(results))
            @test all(uncertainties .> 0)
            @test all(isfinite.(covariances))
            @test all(isfinite.(chi2) .& (chi2 .>= 0))
        end
    end

end
