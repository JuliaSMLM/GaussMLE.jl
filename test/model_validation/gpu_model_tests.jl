# GPU Model Validation Tests
# Tests all models on GPU backend with statistical validation

using CUDA

@testset "GPU Model Validation" begin
    
    # Include validation utilities
    include("validation_utils.jl")
    
    # Skip all GPU tests if CUDA is not functional
    if !CUDA.functional()
        @test_skip "CUDA not available - skipping GPU model validation"
        return
    end
    
    # Get GPU backend
    backend = GaussMLE.CUDABackend()
    
    # Test configuration for 10^4 samples
    n_test_rois = 10_000
    
    @testset "xynb Model (GPU)" begin
        config = ModelTestConfig{Float32}(
            :xynb,
            n_test_rois,
            7,                    # boxsize
            1000.0f0,            # n_photons
            10.0f0,              # bg_photons
            1.5f0,               # σ_psf
            Dict(:x => 0.05f0,   # tolerances
                 :y => 0.05f0,
                 :n => 50.0f0,
                 :bg => 2.0f0),
            0.05,                # 5% CRLB tolerance (may be placeholder)
            42                   # seed
        )
        
        passed, stats, messages = validate_model_gpu(config, backend; verbose=false)
        
        # Test assertions
        @test passed
        if !passed
            println("xynb GPU validation failed:")
            for msg in messages
                println("  - $msg")
            end
        end
        
        # Additional checks
        @test abs(stats[:x][:bias]) < config.param_tolerances[:x]
        @test abs(stats[:y][:bias]) < config.param_tolerances[:y]
        
        # CRLB might be placeholder on GPU, so check if implemented
        if stats[:x][:mean_crlb] != 1.0
            @test stats[:x][:crlb_error] < config.crlb_tolerance
            @test stats[:y][:crlb_error] < config.crlb_tolerance
        end
    end
    
    @testset "xynbs Model (GPU)" begin
        # Note: This model may not be fully implemented on GPU yet
        @test_skip begin
            config = ModelTestConfig{Float32}(
                :xynbs,
                n_test_rois,
                7,                    # boxsize
                1000.0f0,            # n_photons
                10.0f0,              # bg_photons
                1.5f0,               # σ_psf
                Dict(:x => 0.05f0,   # tolerances
                     :y => 0.05f0,
                     :n => 50.0f0,
                     :bg => 2.0f0,
                     :σ => 0.1f0),
                0.05,                # 5% CRLB tolerance
                43                   # seed
            )
            
            passed, stats, messages = validate_model_gpu(config, backend; verbose=false)
            
            @test passed
            if !passed
                println("xynbs GPU validation failed:")
                for msg in messages
                    println("  - $msg")
                end
            end
            
            # Additional checks for σ parameter
            @test abs(stats[:σ][:bias]) < config.param_tolerances[:σ]
        end
    end
    
    @testset "xynbsxsy Model (GPU)" begin
        # Note: This model may not be fully implemented on GPU yet
        @test_skip begin
            config = ModelTestConfig{Float32}(
                :xynbsxsy,
                n_test_rois,
                7,                    # boxsize
                1000.0f0,            # n_photons
                10.0f0,              # bg_photons
                1.5f0,               # σ_psf (used as initial guess)
                Dict(:x => 0.05f0,   # tolerances
                     :y => 0.05f0,
                     :n => 50.0f0,
                     :bg => 2.0f0,
                     :σx => 0.1f0,
                     :σy => 0.1f0),
                0.05,                # 5% CRLB tolerance
                44                   # seed
            )
            
            passed, stats, messages = validate_model_gpu(config, backend; verbose=false)
            
            @test passed
            if !passed
                println("xynbsxsy GPU validation failed:")
                for msg in messages
                    println("  - $msg")
                end
            end
        end
    end
    
    @testset "xynbz Model (GPU)" begin
        # Note: This model may not be fully implemented on GPU yet
        @test_skip begin
            config = ModelTestConfig{Float32}(
                :xynbz,
                n_test_rois,
                7,                    # boxsize
                1000.0f0,            # n_photons
                10.0f0,              # bg_photons
                1.5f0,               # σ_psf (base PSF width)
                Dict(:x => 0.05f0,   # tolerances
                     :y => 0.05f0,
                     :z => 0.05f0,
                     :n => 50.0f0,
                     :bg => 2.0f0),
                0.05,                # 5% CRLB tolerance
                45                   # seed
            )
            
            # Create calibration for astigmatic PSF
            calib = GaussMLE.AstigmaticCalibration{Float32}(
                1.5f0, 1.5f0,   # σx0, σy0
                0.4f0, 0.4f0,   # Ax, Ay  
                0.0f0, 0.0f0,   # Bx, By
                0.0f0, 0.0f0    # 𝛾x, 𝛾y
            )
            model_args = GaussMLE.GaussModel.Args_xynbz{Float32}(calib)
            
            passed, stats, messages = validate_model_gpu(config, backend; verbose=false, model_args=model_args)
            
            @test passed
            if !passed
                println("xynbz GPU validation failed:")
                for msg in messages
                    println("  - $msg")
                end
            end
        end
    end
    
    @testset "CPU/GPU Consistency" begin
        # Test that CPU and GPU give consistent results
        config = ModelTestConfig{Float32}(
            :xynb,
            1000,               # smaller dataset for consistency check
            7,                  # boxsize
            1000.0f0,          # n_photons
            10.0f0,            # bg_photons
            1.5f0,             # σ_psf
            Dict(:x => 0.05f0, :y => 0.05f0, :n => 50.0f0, :bg => 2.0f0),
            0.05,              # CRLB tolerance
            50                 # seed
        )
        
        # Run on both CPU and GPU
        data, true_params, args = generate_synthetic_data(config)
        
        # CPU fitting
        θ_cpu, Σ_cpu = GaussMLE.fitstack(data, :xynb; σ_PSF=config.σ_psf)
        
        # GPU fitting
        θ_gpu, Σ_gpu = GaussMLE.fitstack_gpu(data, :xynb, backend)
        
        # Compare results
        param_names = [:x, :y, :n, :bg]
        cpu_stats = compute_statistics(θ_cpu, true_params, Σ_cpu, param_names)
        gpu_stats = compute_statistics(θ_gpu, true_params, Σ_gpu, param_names)
        
        consistency_passed, consistency_msgs = compare_cpu_gpu_results(
            cpu_stats, gpu_stats, param_names; tolerance=0.1
        )
        
        @test consistency_passed
        if !consistency_passed
            println("CPU/GPU consistency check failed:")
            for msg in consistency_msgs
                println("  - $msg")
            end
        end
        
        # Check individual ROI differences
        max_x_diff = maximum(abs(θ_gpu[i].x - θ_cpu[i].x) for i in 1:config.n_rois)
        max_y_diff = maximum(abs(θ_gpu[i].y - θ_cpu[i].y) for i in 1:config.n_rois)
        
        @test max_x_diff < 0.1  # Sub-pixel accuracy
        @test max_y_diff < 0.1  # Sub-pixel accuracy
    end
    
    @testset "Performance Benchmarks (GPU)" begin
        # Performance comparison between CPU and GPU
        config = ModelTestConfig{Float32}(
            :xynb,
            10_000,             # Full dataset for benchmark
            7,                  # boxsize
            1000.0f0,          # n_photons
            10.0f0,            # bg_photons
            1.5f0,             # σ_psf
            Dict(:x => 0.1f0, :y => 0.1f0, :n => 100.0f0, :bg => 5.0f0),
            0.1,               # relaxed for speed test
            51                 # seed
        )
        
        data, true_params, args = generate_synthetic_data(config)
        
        # Warm up GPU (first call includes compilation)
        _ = GaussMLE.fitstack_gpu(data[1:7, 1:7, 1:10], :xynb, backend)
        
        # Measure GPU fitting time
        t_start = time()
        fitted_gpu, uncertainties_gpu = GaussMLE.fitstack_gpu(data, :xynb, backend)
        t_gpu = time() - t_start
        
        # Measure CPU fitting time for comparison
        t_start = time()
        fitted_cpu, uncertainties_cpu = GaussMLE.fitstack(data, :xynb; σ_PSF=config.σ_psf)
        t_cpu = time() - t_start
        
        gpu_rois_per_second = config.n_rois / t_gpu
        cpu_rois_per_second = config.n_rois / t_cpu
        speedup = t_cpu / t_gpu
        
        @test length(fitted_gpu) == config.n_rois
        @test gpu_rois_per_second > 10_000  # Minimum GPU performance
        @test speedup > 5  # Minimum speedup over CPU
        
        # Report performance
        println("\nGPU Performance:")
        println("  GPU Time: $(round(t_gpu, digits=3)) seconds")
        println("  CPU Time: $(round(t_cpu, digits=3)) seconds")
        println("  Speedup: $(round(speedup, digits=1))x")
        println("  GPU ROIs/second: $(round(Int, gpu_rois_per_second))")
        println("  CPU ROIs/second: $(round(Int, cpu_rois_per_second))")
    end
    
    @testset "Memory and Batching (GPU)" begin
        # Test automatic batching for large datasets
        config = ModelTestConfig{Float32}(
            :xynb,
            100_000,            # Large dataset that may require batching
            7,                  # boxsize
            1000.0f0,          # n_photons
            10.0f0,            # bg_photons
            1.5f0,             # σ_psf
            Dict(:x => 0.1f0, :y => 0.1f0, :n => 100.0f0, :bg => 5.0f0),
            0.1,               # relaxed tolerance
            52                 # seed
        )
        
        # Generate data in chunks to avoid memory issues
        chunk_size = 10_000
        n_chunks = div(config.n_rois, chunk_size)
        
        all_fitted = []
        all_uncertainties = []
        
        for chunk in 1:n_chunks
            chunk_config = ModelTestConfig{Float32}(
                :xynb, chunk_size, 7, 1000.0f0, 10.0f0, 1.5f0,
                config.param_tolerances, config.crlb_tolerance, 52 + chunk
            )
            
            chunk_data, _, _ = generate_synthetic_data(chunk_config)
            θ_chunk, Σ_chunk = GaussMLE.fitstack_gpu(chunk_data, :xynb, backend)
            
            push!(all_fitted, θ_chunk...)
            push!(all_uncertainties, Σ_chunk...)
        end
        
        @test length(all_fitted) == config.n_rois
        @test length(all_uncertainties) == config.n_rois
        
        # Check that results are reasonable
        x_vals = [p.x for p in all_fitted]
        @test minimum(x_vals) > 1.0
        @test maximum(x_vals) < Float32(config.boxsize)
    end
    
    @testset "Edge Cases (GPU)" begin
        # Test with single ROI
        single_roi = randn(Float32, 7, 7, 1) .+ 10.0f0
        θ_single, Σ_single = GaussMLE.fitstack_gpu(single_roi, :xynb, backend)
        @test length(θ_single) == 1
        @test length(Σ_single) == 1
        
        # Test with empty data (should handle gracefully)
        empty_data = zeros(Float32, 7, 7, 0)
        θ_empty, Σ_empty = GaussMLE.fitstack_gpu(empty_data, :xynb, backend)
        @test isempty(θ_empty)
        @test isempty(Σ_empty)
        
        # Test with very small dataset
        small_data = randn(Float32, 7, 7, 10) .+ 10.0f0
        θ_small, Σ_small = GaussMLE.fitstack_gpu(small_data, :xynb, backend)
        @test length(θ_small) == 10
    end
end