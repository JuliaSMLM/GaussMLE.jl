using GaussMLE
using Test
using Statistics
using LinearAlgebra

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
    
    # Test GPU backend interface (will fail until implemented)
    @testset "GPU Backend Interface" begin
        @test_skip begin
            # Test backend selection
            backend = GaussMLE.select_backend()
            @test backend isa GaussMLE.FittingBackend
            
            # Test backend capabilities
            @test GaussMLE.supports_streaming(backend)
            @test GaussMLE.max_batch_size(backend) > 0
        end
    end
    
    # Test CUDA backend (will fail until implemented)
    @testset "CUDA Backend" begin
        @test_skip begin
            if CUDA.functional()
                backend = GaussMLE.CUDABackend()
                @test backend isa GaussMLE.FittingBackend
                
                # Test small batch fitting
                scenario = GaussMLE.GaussSim.GPUTestScenario{Float32}(
                    "cuda_test", 7, 1000, (500f0, 500f0), (2f0, 2f0), 0.1f0, 1.3f0, nothing
                )
                gen = GaussMLE.GaussSim.StreamingBatchGenerator(scenario, 1000)
                batch = GaussMLE.GaussSim.next_batch!(gen)
                
                # Fit with CUDA backend
                θ_gpu, Σ_gpu = GaussMLE.fitstack_gpu(batch.data, :xynb, backend)
                
                # Compare with CPU reference
                θ_cpu, Σ_cpu = GaussMLE.GaussFit.fitstack(batch.data, :xynb)
                
                # Validate results
                valid, msg = GaussMLE.GaussSim.validate_gpu_results(
                    θ_cpu, Σ_cpu, θ_gpu, Σ_gpu, scenario
                )
                @test valid
            end
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

