# Unified kernel on the CPU backend: each Gaussian model fits a simple blob batch.
using Test, GaussMLE, KernelAbstractions, LinearAlgebra

@testset "Unified Kernel Tests" begin

    # Test configuration
    n_test_blobs = 100
    box_size = 7
    iterations = 20

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

    @testset "CPU Unified Kernel" begin
        data = generate_simple_test_data(n_test_blobs, box_size)

        # Test with different PSF models
        @testset "GaussianXYNB (N=4)" begin
            psf_model = GaussMLE.GaussianXYNB(0.13f0)
            constraints = GaussMLE.default_constraints(psf_model, box_size)

            # Allocate output arrays
            results = Matrix{Float32}(undef, 4, n_test_blobs)
            uncertainties = Matrix{Float32}(undef, 4, n_test_blobs)
            covariances = Matrix{Float32}(undef, 3, n_test_blobs)
            log_likelihoods = Vector{Float32}(undef, n_test_blobs)

            # Run unified kernel on CPU
            backend = KernelAbstractions.CPU()
            kernel = GaussMLE.unified_gaussian_mle_kernel!(backend)

            # Create dummy corners and variance for new kernel signature
            x_corners = Int32[1 + (i - 1) * box_size for i in 1:n_test_blobs]
            y_corners = fill(Int32(1), n_test_blobs)
            variance_map = zeros(Float32, 512, 512)

            kernel(
                results, uncertainties, covariances, log_likelihoods,
                data, psf_model, Val(false), variance_map, x_corners, y_corners,
                constraints, iterations,
                ndrange = n_test_blobs
            )

            # Check results are reasonable
            @test all(isfinite.(results))
            @test all(uncertainties .> 0)
            @test all(isfinite.(covariances))
            @test all(isfinite.(log_likelihoods))

            # Check parameters are in expected ranges
            @test all(2 .< results[1, :] .< 6)  # x position
            @test all(2 .< results[2, :] .< 6)  # y position
            @test all(100 .< results[3, :] .< 20000)  # photons (integrated Gaussian)
            @test all(0 .< results[4, :] .< 100)  # background
        end

        @testset "GaussianXYNBS (N=5)" begin
            psf_model = GaussMLE.GaussianXYNBS{Float32}()
            constraints = GaussMLE.default_constraints(psf_model, box_size)

            results = Matrix{Float32}(undef, 5, n_test_blobs)
            uncertainties = Matrix{Float32}(undef, 5, n_test_blobs)
            covariances = Matrix{Float32}(undef, 3, n_test_blobs)
            log_likelihoods = Vector{Float32}(undef, n_test_blobs)

            backend = KernelAbstractions.CPU()
            kernel = GaussMLE.unified_gaussian_mle_kernel!(backend)

            # Create dummy corners and variance for new kernel signature
            x_corners = Int32[1 + (i - 1) * box_size for i in 1:n_test_blobs]
            y_corners = fill(Int32(1), n_test_blobs)
            variance_map = zeros(Float32, 512, 512)

            kernel(
                results, uncertainties, covariances, log_likelihoods,
                data, psf_model, Val(false), variance_map, x_corners, y_corners,
                constraints, iterations,
                ndrange = n_test_blobs
            )

            @test all(isfinite.(results))
            @test all(uncertainties .> 0)
            @test all(isfinite.(covariances))
            @test all(isfinite.(log_likelihoods))
        end

        @testset "GaussianXYNBSXSY (N=6)" begin
            psf_model = GaussMLE.GaussianXYNBSXSY{Float32}()
            constraints = GaussMLE.default_constraints(psf_model, box_size)

            results = Matrix{Float32}(undef, 6, n_test_blobs)
            uncertainties = Matrix{Float32}(undef, 6, n_test_blobs)
            covariances = Matrix{Float32}(undef, 3, n_test_blobs)
            log_likelihoods = Vector{Float32}(undef, n_test_blobs)

            backend = KernelAbstractions.CPU()
            kernel = GaussMLE.unified_gaussian_mle_kernel!(backend)

            # Create dummy corners and variance for new kernel signature
            x_corners = Int32[1 + (i - 1) * box_size for i in 1:n_test_blobs]
            y_corners = fill(Int32(1), n_test_blobs)
            variance_map = zeros(Float32, 512, 512)

            kernel(
                results, uncertainties, covariances, log_likelihoods,
                data, psf_model, Val(false), variance_map, x_corners, y_corners,
                constraints, iterations,
                ndrange = n_test_blobs
            )

            @test all(isfinite.(results))
            @test all(uncertainties .> 0)
            @test all(isfinite.(covariances))
            @test all(isfinite.(log_likelihoods))
        end
    end

    @testset "Covariance vs Float64 reference (asymmetric fixture)" begin
        # Noise-free spots near the ROI corners: truncation makes sigma_xy clearly nonzero
        # (correlation about 3%) with a sign set by the corner, so zero or sign-flipped
        # covariance output fails. The GPU group runs the same check on the CUDA kernel.
        psf_model = GaussMLE.GaussianXYNB(1.3f0)  # kernel units: pixels
        psf64 = GaussMLE.GaussianXYNB(1.3)
        roi = 7
        n_photons, bg = 2000.0, 10.0
        positions = [(2.5, 2.5), (2.5, 5.5), (5.5, 2.5), (5.5, 5.5), (2.2, 3.0), (3.0, 2.2)]
        n = length(positions)
        pixel(i, j, θ) = GaussMLE._evaluate_psf_pixel(psf64, i, j, θ)
        data = zeros(Float32, roi, roi, n)
        for (k, (x, y)) in enumerate(positions), j in 1:roi, i in 1:roi
            data[i, j, k] = pixel(i, j, [x, y, n_photons, bg])
        end

        # Reference: inverse Poisson Fisher information at the true parameters, in Float64,
        # with central-difference derivatives
        function reference_cov_xy(x, y)
            θ = [x, y, n_photons, bg]
            F = zeros(4, 4)
            for j in 1:roi, i in 1:roi
                g = map(1:4) do p
                    δ = zeros(4)
                    δ[p] = 1.0e-6 * max(1.0, abs(θ[p]))
                    (pixel(i, j, θ + δ) - pixel(i, j, θ - δ)) / (2 * δ[p])
                end
                F .+= g * g' ./ pixel(i, j, θ)
            end
            return inv(F)[1, 2]
        end
        cov_ref = [reference_cov_xy(x, y) for (x, y) in positions]
        @test all(abs.(cov_ref) .> 1.0e-5)  # fixture sanity: far above Float32 roundoff

        results = zeros(Float32, 4, n)
        uncertainties = zeros(Float32, 4, n)
        covariances = zeros(Float32, 3, n)
        log_likelihoods = zeros(Float32, n)
        kernel = GaussMLE.unified_gaussian_mle_kernel!(KernelAbstractions.CPU())
        kernel(
            results, uncertainties, covariances, log_likelihoods,
            data, psf_model, Val(false), zeros(Float32, 512, 512),
            Int32[1 + (k - 1) * roi for k in 1:n], fill(Int32(1), n),
            GaussMLE.default_constraints(psf_model, roi), iterations;
            ndrange = n
        )
        @test covariances[1, :] ≈ cov_ref rtol = 1.0e-3
    end

end
