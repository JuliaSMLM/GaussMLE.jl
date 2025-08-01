# User-facing API tests
# Tests all exported functions that users interact with

@testset "Public API" begin
    
    @testset "fitstack - Basic fitting" begin
        # Simulate a stack of boxes with Poisson noise
        T = Float32 # Data type
        boxsz = 7 # Box size
        nboxes = Int(1e5) # Number of boxes
        roi_stack, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynb; T=T, poissonnoise=true)

        # Fit all boxes in the stack using the new unified API
        θ_found, Σ_found = GaussMLE.fitstack(roi_stack, :xynb; σ_PSF=args.σ_PSF)

        # Compare the true and found parameters
        μ_x_mc = mean(getproperty.(θ_found, :x))
        σ_x_mc = std(getproperty.(θ_found, :x))
        σ_x_reported = mean(getproperty.(Σ_found, :σ_x))

        μ_y_mc = mean(getproperty.(θ_found, :y))
        σ_y_mc = std(getproperty.(θ_found, :y))
        σ_y_reported = mean(getproperty.(Σ_found, :σ_y))

        μ_n_mc = mean(getproperty.(θ_found, :n))
        σ_n_mc = std(getproperty.(θ_found, :n))
        σ_n_reported = mean(getproperty.(Σ_found, :σ_n))

        μ_bg_mc = mean(getproperty.(θ_found, :bg))
        σ_bg_mc = std(getproperty.(θ_found, :bg))
        σ_bg_reported = mean(getproperty.(Σ_found, :σ_bg))

        # Check if the means and standard deviations are close to the true values
        @test isapprox(μ_x_mc, θ_true[1].x, atol=1e-1)
        @test isapprox(σ_x_mc, σ_x_reported, atol=1e-1)
        
        @test isapprox(μ_y_mc, θ_true[1].y, atol=1e-1)
        @test isapprox(σ_y_mc, σ_y_reported, atol=1e-1)
        
        @test isapprox(μ_n_mc, θ_true[1].n, atol=1e1)
        @test isapprox(σ_n_mc, σ_n_reported,atol=1e1)
        
        @test isapprox(μ_bg_mc, θ_true[1].bg, atol=1e-1)
        @test isapprox(σ_bg_mc, σ_bg_reported, atol=1e-1)
    end

    @testset "Model types" begin
        # Test that model types are exported and constructible
        @test GaussMLE.θ_xynb{Float32} <: GaussMLE.GaussMLEParams{Float32}
        @test GaussMLE.θ_xynbs{Float32} <: GaussMLE.GaussMLEParams{Float32}
        
        # Test that we can create instances
        θ1 = GaussMLE.θ_xynb(1.0f0, 2.0f0, 100.0f0, 1.0f0)
        @test isa(θ1, GaussMLE.GaussMLEParams)
        
        θ2 = GaussMLE.θ_xynbs(1.0f0, 2.0f0, 100.0f0, 1.0f0, 1.3f0)
        @test isa(θ2, GaussMLE.GaussMLEParams)
    end

    @testset "fitstack with different models" begin
        # Test fitting with GaussXyNb model
        boxsz = 7
        nboxes = 100
        roi_stack, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynb; poissonnoise=true)
        θ_found, Σ_found = GaussMLE.fitstack(roi_stack, :xynb; σ_PSF=args.σ_PSF)
        @test length(θ_found) == nboxes
        @test all(isa.(θ_found, GaussMLE.θ_xynb))
        
        # Test fitting with GaussXyNbS model
        roi_stack, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynbs; poissonnoise=true)
        θ_found, Σ_found = GaussMLE.fitstack(roi_stack, :xynbs; σ_PSF=args.σ_PSF)
        @test length(θ_found) == nboxes
        @test all(isa.(θ_found, GaussMLE.θ_xynbs))
    end
end