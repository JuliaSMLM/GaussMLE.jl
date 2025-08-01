using GaussMLE
using Test
using Statistics
using LinearAlgebra
using CUDA

@testset "GPU Backend Tests" begin
    
    # Test the GPU test scenario generation
    @testset "GPU Test Scenarios" begin
        scenarios = GaussMLE.GaussSim.generate_gpu_test_scenarios(Float32)
        @test length(scenarios) >= 6
        @test all(s -> s.n_rois > 0, scenarios)
        @test all(s -> s.boxsize > 0, scenarios)
    end
    
    # Test streaming batch generator
    @testset "Streaming Batch Generator" begin
        scenario = GaussMLE.GaussSim.GPUTestScenario{Float32}(
            "test", 7, 1000, (300f0, 700f0), (1f0, 3f0), 0.1f0, 1.3f0, nothing
        )
        gen = GaussMLE.GaussSim.StreamingBatchGenerator(scenario, 100)
        
        total_rois = 0
        while true
            batch = GaussMLE.GaussSim.next_batch!(gen)
            if batch === nothing
                break
            end
            total_rois += size(batch.data, 3)
            @test size(batch.data, 1) == 7
            @test size(batch.data, 2) == 7
        end
        @test total_rois == 1000
    end
    
    # Test GPU backend interface
    @testset "GPU Backend Interface" begin
        # Test backend selection
        backend = GaussMLE.select_backend()
        @test backend isa GaussMLE.FittingBackend
        @test GaussMLE.backend_name(backend) in ["CUDA", "CPU"]
    end
    
    # Test CUDA backend
    @testset "CUDA Backend" begin
        if CUDA.functional()
            backend = GaussMLE.CUDABackend()
            @test backend isa GaussMLE.FittingBackend
            
            # Create realistic synthetic Gaussian test data
            n_rois = 10
            data = zeros(Float32, 7, 7, n_rois)
            
            # Generate proper Gaussian spots
            for k in 1:n_rois
                x_true = 3.0f0 + rand(Float32) * 1.0f0  # 3-4 range
                y_true = 3.0f0 + rand(Float32) * 1.0f0  # 3-4 range
                intensity = 800f0 + rand(Float32) * 400f0  # 800-1200 photons
                bg = 8f0 + rand(Float32) * 4f0  # 8-12 background
                
                for i in 1:7
                    for j in 1:7
                        dx = Float32(j) - x_true
                        dy = Float32(i) - y_true
                        gauss = intensity * exp(-(dx^2 + dy^2) / (2*1.3f0^2)) / (2π*1.3f0^2)
                        data[i, j, k] = bg + gauss
                    end
                end
            end
            
            # Test GPU fitting
            θ_gpu, Σ_gpu = GaussMLE.fitstack_gpu(data, :xynb, backend)
            @test length(θ_gpu) == n_rois
            @test length(Σ_gpu) == n_rois
            
            # Compare with CPU
            θ_cpu, Σ_cpu = GaussMLE.fitstack(data, :xynb)
            
            # Position results should be very close (most important for SMLM)
            max_x_diff = maximum(abs(θ_gpu[i].x - θ_cpu[i].x) for i in 1:n_rois)
            max_y_diff = maximum(abs(θ_gpu[i].y - θ_cpu[i].y) for i in 1:n_rois)
            @test max_x_diff < 0.1  # Sub-pixel accuracy required
            @test max_y_diff < 0.1  # Sub-pixel accuracy required
        else
            @test_skip "CUDA not available"
        end
    end
    
    # Test Metal backend (will fail until implemented)
    @testset "Metal Backend" begin
        @test_skip begin
            if GaussMLE.metal_available()
                backend = GaussMLE.MetalBackend()
                @test backend isa GaussMLE.FittingBackend
                
                # Similar tests as CUDA
                scenario = GaussMLE.GaussSim.GPUTestScenario{Float32}(
                    "metal_test", 7, 1000, (500f0, 500f0), (2f0, 2f0), 0.1f0, 1.3f0, nothing
                )
                gen = GaussMLE.GaussSim.StreamingBatchGenerator(scenario, 1000)
                batch = GaussMLE.GaussSim.next_batch!(gen)
                
                θ_gpu, Σ_gpu = GaussMLE.fitstack_gpu(batch.data, :xynb, backend)
                θ_cpu, Σ_cpu = GaussMLE.GaussFit.fitstack(batch.data, :xynb)
                
                valid, msg = GaussMLE.GaussSim.validate_gpu_results(
                    θ_cpu, Σ_cpu, θ_gpu, Σ_gpu, scenario
                )
                @test valid
            end
        end
    end
    
    # Test batching system (will fail until implemented)
    @testset "Batching System" begin
        @test_skip begin
            # Test large dataset that requires batching
            scenario = GaussMLE.GaussSim.GPUTestScenario{Float32}(
                "batch_test", 7, 1_000_000, (500f0, 500f0), (2f0, 2f0), 0.1f0, 1.3f0, nothing
            )
            
            backend = GaussMLE.select_backend()
            batch_config = GaussMLE.BatchConfig(
                max_batch_size = 100_000,
                n_streams = 4,
                pinned_memory = true,
                overlap_compute = true
            )
            
            # This should handle batching automatically
            gen = GaussMLE.GaussSim.StreamingBatchGenerator(scenario, batch_config.max_batch_size)
            all_results = []
            
            while true
                batch = GaussMLE.GaussSim.next_batch!(gen)
                if batch === nothing
                    break
                end
                θ, Σ = GaussMLE.fitstack_gpu(batch.data, :xynb, backend, batch_config)
                push!(all_results, (θ, Σ))
            end
            
            @test length(all_results) == 10  # 1M / 100K = 10 batches
        end
    end
    
    # Test performance requirements (will fail until implemented)
    @testset "Performance Requirements" begin
        @test_skip begin
            backend = GaussMLE.select_backend()
            
            # Small benchmark
            scenario = GaussMLE.GaussSim.GPUTestScenario{Float32}(
                "perf_test", 7, 100_000, (500f0, 500f0), (2f0, 2f0), 0.1f0, 1.3f0, nothing
            )
            
            # Benchmark GPU
            gpu_bench = GaussMLE.GaussSim.benchmark_scenario(
                scenario, 
                data -> GaussMLE.fitstack_gpu(data, :xynb, backend)
            )
            
            # Benchmark CPU (single thread)
            cpu_bench = GaussMLE.GaussSim.benchmark_scenario(
                scenario,
                data -> GaussMLE.GaussFit.fitstack(data, :xynb)
            )
            
            speedup = cpu_bench.mean_time / gpu_bench.mean_time
            
            # Check performance targets
            if backend isa GaussMLE.CUDABackend
                @test speedup >= 20  # Minimum 20x speedup
                @test gpu_bench.rois_per_second >= 500_000  # 500K ROIs/sec minimum
            elseif backend isa GaussMLE.MetalBackend
                @test speedup >= 10  # Minimum 10x speedup
                @test gpu_bench.rois_per_second >= 200_000  # 200K ROIs/sec minimum
            else  # CPU backend
                @test speedup >= 2  # Multi-threaded should be faster
            end
        end
    end
    
    # Test variance-weighted fitting (will fail until implemented)
    @testset "Variance-weighted GPU Fitting" begin
        @test_skip begin
            backend = GaussMLE.select_backend()
            
            # Generate test data with variance map
            scenario = GaussMLE.GaussSim.GPUTestScenario{Float32}(
                "variance_test", 7, 10_000, (500f0, 500f0), (2f0, 2f0), 0.1f0, 1.3f0, nothing
            )
            
            gen = GaussMLE.GaussSim.StreamingBatchGenerator(scenario, 10_000)
            batch = GaussMLE.GaussSim.next_batch!(gen)
            
            # Generate sCMOS variance map
            variance_map = GaussMLE.GaussSim.generate_scmos_variance_map(
                7, 10_000, Float32
            )
            
            # Fit with variance weighting
            θ_gpu, Σ_gpu = GaussMLE.fitstack_gpu(
                batch.data, :xynb, backend, 
                variance=variance_map
            )
            
            # Should complete without error
            @test length(θ_gpu) == 10_000
            @test length(Σ_gpu) == 10_000
        end
    end
    
    # Test edge cases (will fail until implemented)
    @testset "Edge Cases" begin
        @test_skip begin
            backend = GaussMLE.select_backend()
            
            # Empty data
            empty_data = zeros(Float32, 7, 7, 0)
            θ, Σ = GaussMLE.fitstack_gpu(empty_data, :xynb, backend)
            @test isempty(θ)
            @test isempty(Σ)
            
            # Single ROI
            single_roi = randn(Float32, 7, 7, 1)
            θ, Σ = GaussMLE.fitstack_gpu(single_roi, :xynb, backend)
            @test length(θ) == 1
            @test length(Σ) == 1
            
            # Very large ROI
            large_roi = randn(Float32, 21, 21, 100)
            θ, Σ = GaussMLE.fitstack_gpu(large_roi, :xynb, backend)
            @test length(θ) == 100
            
            # Non-square ROI (should error)
            @test_throws ArgumentError GaussMLE.fitstack_gpu(
                randn(Float32, 7, 9, 10), :xynb, backend
            )
        end
    end
end

