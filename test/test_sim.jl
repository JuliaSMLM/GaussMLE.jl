# Tests for simulation module

@testset "GaussSim internals" begin
    @testset "genstack - data generation" begin
        # Test basic generation without noise
        boxsz = 7
        nboxes = 100
        roi_stack, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynb; poissonnoise=false)
        
        @test size(roi_stack) == (boxsz, boxsz, nboxes)
        @test length(θ_true) == nboxes
        @test all(isa.(θ_true, GaussMLE.θ_xynb))
        
        # Test with Poisson noise
        roi_stack_noise, θ_true_noise, args_noise = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynb; poissonnoise=true)
        @test size(roi_stack_noise) == (boxsz, boxsz, nboxes)
        @test !all(roi_stack .== roi_stack_noise)  # Should be different with noise
        
        # Test with different model
        roi_stack_s, θ_true_s, args_s = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynbs; poissonnoise=true)
        @test all(isa.(θ_true_s, GaussMLE.θ_xynbs))
    end
    
    @testset "genstack - parameter ranges" begin
        boxsz = 7
        nboxes = 1000
        
        # Test that generated parameters are within reasonable ranges
        roi_stack, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynb)
        
        x_vals = getproperty.(θ_true, :x)
        y_vals = getproperty.(θ_true, :y)
        n_vals = getproperty.(θ_true, :n)
        bg_vals = getproperty.(θ_true, :bg)
        
        # Position should be within box
        @test all(1 .<= x_vals .<= boxsz)
        @test all(1 .<= y_vals .<= boxsz)
        
        # Intensity and background should be positive
        @test all(n_vals .> 0)
        @test all(bg_vals .>= 0)
    end
end