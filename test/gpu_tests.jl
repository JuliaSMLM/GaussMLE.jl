"""
GPU kernel tests - tests the unified kernel on both CPU and GPU
"""

using GaussMLE
using Test
using CUDA
using Statistics
using KernelAbstractions

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
            log_likelihoods = Vector{Float32}(undef, n_test_blobs)
            
            # Run unified kernel on CPU
            backend = KernelAbstractions.CPU()
            kernel = GaussMLE.unified_gaussian_mle_kernel!(backend)
            
            kernel(results, uncertainties, log_likelihoods,
                   data, psf_model, GaussMLE.IdealCamera(), nothing,
                   constraints, iterations,
                   ndrange=n_test_blobs)
            
            # Check results are reasonable
            @test all(isfinite.(results))
            @test all(uncertainties .> 0)
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
            log_likelihoods = Vector{Float32}(undef, n_test_blobs)
            
            backend = KernelAbstractions.CPU()
            kernel = GaussMLE.unified_gaussian_mle_kernel!(backend)
            
            kernel(results, uncertainties, log_likelihoods,
                   data, psf_model, GaussMLE.IdealCamera(), nothing,
                   constraints, iterations,
                   ndrange=n_test_blobs)
            
            @test all(isfinite.(results))
            @test all(uncertainties .> 0)
            @test all(isfinite.(log_likelihoods))
        end
        
        @testset "GaussianXYNBSXSY (N=6)" begin
            psf_model = GaussMLE.GaussianXYNBSXSY{Float32}()
            constraints = GaussMLE.default_constraints(psf_model, box_size)
            
            results = Matrix{Float32}(undef, 6, n_test_blobs)
            uncertainties = Matrix{Float32}(undef, 6, n_test_blobs)
            log_likelihoods = Vector{Float32}(undef, n_test_blobs)
            
            backend = KernelAbstractions.CPU()
            kernel = GaussMLE.unified_gaussian_mle_kernel!(backend)
            
            kernel(results, uncertainties, log_likelihoods,
                   data, psf_model, GaussMLE.IdealCamera(), nothing,
                   constraints, iterations,
                   ndrange=n_test_blobs)
            
            @test all(isfinite.(results))
            @test all(uncertainties .> 0)
            @test all(isfinite.(log_likelihoods))
        end
    end
    
    @testset "GPU Unified Kernel" begin
        if CUDA.functional()
            data = generate_simple_test_data(n_test_blobs, box_size)
            
            @testset "GaussianXYNB GPU" begin
                psf_model = GaussMLE.GaussianXYNB(0.13f0)
                constraints = GaussMLE.default_constraints(psf_model, box_size)
                
                # Move data to GPU
                d_data = CuArray(data)
                
                # Allocate GPU output arrays
                d_results = CUDA.zeros(Float32, 4, n_test_blobs)
                d_uncertainties = CUDA.zeros(Float32, 4, n_test_blobs)
                d_log_likelihoods = CUDA.zeros(Float32, n_test_blobs)
                
                # Run unified kernel on GPU
                backend = CUDABackend()
                kernel = GaussMLE.unified_gaussian_mle_kernel!(backend)
                
                kernel(d_results, d_uncertainties, d_log_likelihoods,
                       d_data, psf_model, GaussMLE.IdealCamera(), nothing,
                       constraints, iterations,
                       ndrange=n_test_blobs)
                
                # Wait for completion
                CUDA.synchronize()
                
                # Copy results back
                results = Array(d_results)
                uncertainties = Array(d_uncertainties)
                log_likelihoods = Array(d_log_likelihoods)
                
                # Check results
                @test all(isfinite.(results))
                @test all(uncertainties .> 0)
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
                
                # Run on CPU
                results_cpu = Matrix{Float32}(undef, 4, n_test_blobs)
                uncertainties_cpu = Matrix{Float32}(undef, 4, n_test_blobs)
                log_likelihoods_cpu = Vector{Float32}(undef, n_test_blobs)
                
                backend_cpu = KernelAbstractions.CPU()
                kernel_cpu = GaussMLE.unified_gaussian_mle_kernel!(backend_cpu)
                kernel_cpu(results_cpu, uncertainties_cpu, log_likelihoods_cpu,
                          data, psf_model, GaussMLE.IdealCamera(), nothing,
                          constraints, iterations,
                          ndrange=n_test_blobs)
                
                # Run on GPU
                d_data = CuArray(data)
                d_results = CUDA.zeros(Float32, 4, n_test_blobs)
                d_uncertainties = CUDA.zeros(Float32, 4, n_test_blobs)
                d_log_likelihoods = CUDA.zeros(Float32, n_test_blobs)
                
                backend_gpu = CUDABackend()
                kernel_gpu = GaussMLE.unified_gaussian_mle_kernel!(backend_gpu)
                kernel_gpu(d_results, d_uncertainties, d_log_likelihoods,
                          d_data, psf_model, GaussMLE.IdealCamera(), nothing,
                          constraints, iterations,
                          ndrange=n_test_blobs)
                
                CUDA.synchronize()
                
                results_gpu = Array(d_results)
                uncertainties_gpu = Array(d_uncertainties)
                log_likelihoods_gpu = Array(d_log_likelihoods)
                
                # Compare results (should be very close)
                @test results_cpu ≈ results_gpu rtol=1e-4
                @test uncertainties_cpu ≈ uncertainties_gpu rtol=1e-3
                @test log_likelihoods_cpu ≈ log_likelihoods_gpu rtol=1e-4
            end
        else
            @test_skip "GPU not available"
        end
    end
    
end