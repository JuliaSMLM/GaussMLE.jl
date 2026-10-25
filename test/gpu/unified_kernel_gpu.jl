"""
GPU kernel tests: the unified kernel on CUDA, and CPU vs GPU agreement (GPU group; the CPU
backend is in test/unified_kernel_cpu.jl)
"""

using GaussMLE
using Test
using CUDA
using Statistics
using KernelAbstractions
using LinearAlgebra

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

    @testset "GPU Unified Kernel" begin
        data = generate_simple_test_data(n_test_blobs, box_size)

        @testset "GaussianXYNB GPU" begin
            psf_model = GaussMLE.GaussianXYNB(0.13f0)
            constraints = GaussMLE.default_constraints(psf_model, box_size)

            # Move data to GPU
            d_data = CuArray(data)

            # Allocate GPU output arrays
            d_results = CUDA.zeros(Float32, 4, n_test_blobs)
            d_uncertainties = CUDA.zeros(Float32, 4, n_test_blobs)
            d_covariances = CUDA.zeros(Float32, 3, n_test_blobs)
            d_log_likelihoods = CUDA.zeros(Float32, n_test_blobs)

            # Create dummy corners and variance for new kernel signature
            d_x_corners = CuArray(Int32[1 + (i - 1) * box_size for i in 1:n_test_blobs])
            d_y_corners = CuArray(fill(Int32(1), n_test_blobs))
            d_variance_map = CUDA.zeros(Float32, 512, 512)

            # Run unified kernel on GPU
            backend = CUDABackend()
            kernel = GaussMLE.unified_gaussian_mle_kernel!(backend)

            kernel(
                d_results, d_uncertainties, d_covariances, d_log_likelihoods,
                d_data, psf_model, Val(false), d_variance_map, d_x_corners, d_y_corners,
                constraints, iterations,
                ndrange = n_test_blobs
            )

            # Wait for completion
            CUDA.synchronize()

            # Copy results back
            results = Array(d_results)
            uncertainties = Array(d_uncertainties)
            covariances = Array(d_covariances)
            log_likelihoods = Array(d_log_likelihoods)

            # Check results
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

        @testset "CPU vs GPU Consistency" begin
            psf_model = GaussMLE.GaussianXYNB(0.13f0)
            constraints = GaussMLE.default_constraints(psf_model, box_size)

            # Create dummy corners and variance for new kernel signature
            x_corners = Int32[1 + (i - 1) * box_size for i in 1:n_test_blobs]
            y_corners = fill(Int32(1), n_test_blobs)
            variance_map = zeros(Float32, 512, 512)

            # Run on CPU
            results_cpu = Matrix{Float32}(undef, 4, n_test_blobs)
            uncertainties_cpu = Matrix{Float32}(undef, 4, n_test_blobs)
            covariances_cpu = Matrix{Float32}(undef, 3, n_test_blobs)
            log_likelihoods_cpu = Vector{Float32}(undef, n_test_blobs)

            backend_cpu = KernelAbstractions.CPU()
            kernel_cpu = GaussMLE.unified_gaussian_mle_kernel!(backend_cpu)
            kernel_cpu(
                results_cpu, uncertainties_cpu, covariances_cpu, log_likelihoods_cpu,
                data, psf_model, Val(false), variance_map, x_corners, y_corners,
                constraints, iterations,
                ndrange = n_test_blobs
            )

            # Run on GPU
            d_data = CuArray(data)
            d_results = CUDA.zeros(Float32, 4, n_test_blobs)
            d_uncertainties = CUDA.zeros(Float32, 4, n_test_blobs)
            d_covariances = CUDA.zeros(Float32, 3, n_test_blobs)
            d_log_likelihoods = CUDA.zeros(Float32, n_test_blobs)
            d_x_corners = CuArray(x_corners)
            d_y_corners = CuArray(y_corners)
            d_variance_map = CuArray(variance_map)

            backend_gpu = CUDABackend()
            kernel_gpu = GaussMLE.unified_gaussian_mle_kernel!(backend_gpu)
            kernel_gpu(
                d_results, d_uncertainties, d_covariances, d_log_likelihoods,
                d_data, psf_model, Val(false), d_variance_map, d_x_corners, d_y_corners,
                constraints, iterations,
                ndrange = n_test_blobs
            )

            CUDA.synchronize()

            results_gpu = Array(d_results)
            uncertainties_gpu = Array(d_uncertainties)
            covariances_gpu = Array(d_covariances)
            log_likelihoods_gpu = Array(d_log_likelihoods)

            # Compare results (should be very close)
            @test results_cpu ≈ results_gpu rtol = 1.0e-4
            @test uncertainties_cpu ≈ uncertainties_gpu rtol = 1.0e-3
            # σ_xy of a centred symmetric blob is Float32 roundoff around zero, so compare
            # it on the scale of the x/y variances rather than relative to itself.
            var_xy = maximum(abs2, uncertainties_cpu[1:2, :])
            @test covariances_cpu ≈ covariances_gpu rtol = 1.0e-3 atol = 1.0e-3 * var_xy
            @test log_likelihoods_cpu ≈ log_likelihoods_gpu rtol = 1.0e-4
        end
    end

    @testset "Covariance vs Float64 reference (asymmetric fixture)" begin
        # Noise-free spots near the ROI corners: truncation makes sigma_xy clearly nonzero
        # (correlation about 3%) with a sign set by the corner, so zero or sign-flipped
        # covariance output fails. The random fixture above only checks roundoff near zero;
        # the CPU kernel's half of this check is in Core (test/unified_kernel_cpu.jl).
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

        function kernel_cov_xy(backend, to)
            out = (
                to(zeros(Float32, 4, n)), to(zeros(Float32, 4, n)),
                to(zeros(Float32, 3, n)), to(zeros(Float32, n)),
            )
            kernel = GaussMLE.unified_gaussian_mle_kernel!(backend)
            kernel(
                out..., to(data), psf_model, Val(false), to(zeros(Float32, 512, 512)),
                to(Int32[1 + (k - 1) * roi for k in 1:n]), to(fill(Int32(1), n)),
                GaussMLE.default_constraints(psf_model, roi), iterations;
                ndrange = n
            )
            KernelAbstractions.synchronize(backend)
            return Array(out[3])[1, :]
        end
        @test kernel_cov_xy(CUDABackend(), CuArray) ≈ cov_ref rtol = 1.0e-3
    end

end
