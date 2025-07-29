using GaussMLE
using SpecialFunctions
using LinearAlgebra
using Statistics 
using Test

@testset "GaussMLE.jl" begin

    # Importing specific functions for testing from the provided baselibrary.jl content
    
    @testset "baselibrary" begin

        using GaussMLE.GaussLib: integral_gaussian_1d, compute_alpha, derivative_alpha_z, 
                    second_derivative_alpha_z, center_of_mass_2d, gaussian_max_min_2d
    
        # Test for integral_gaussian_1d
        @test integral_gaussian_1d(1, 2.0, 1.0) ≈ 0.2417303374571288
        
        # Test for compute_alpha function
        z, Ax, Bx, d = 1.0, 2.0, 3.0, 4.0
        @test compute_alpha(z, Ax, Bx, d) ≈ 1.0 + (z / d)^2 + Ax * (z / d)^3 + Bx * (z / d)^4
        
        # Test for derivative_alpha_z function
        @test derivative_alpha_z(z, Ax, Bx, d) ≈ 2.0 * z / d^2 + 3.0 * Ax * z^2 / d^3 + 4.0 * Bx * z^3 / d^4
        
        # Test for second_derivative_alpha_z function
        @test second_derivative_alpha_z(z, Ax, Bx, d) ≈ 2.0 / d^2 + 6.0 * Ax * z / d^3 + 12.0 * Bx * z^2 / d^4
        
        # Test for center_of_mass_2d (assuming the function signature hasn't changed)
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        @test all(center_of_mass_2d(3, data) .≈ (2.1333333333333333, 2.4))

        # Test for gauss_f_max_min_2D (assuming the function signature hasn't changed)
        @test all(gaussian_max_min_2d(3, 1.0, data) .≈ (6.985605655276496, 3.0143943447235038))
        
        
    end

    @testset "Gaussian blob fitting" begin

        # Simulate a stack of boxes with Poisson noise
        T = Float32 # Data type
        boxsz = 7 # Box size
        nboxes = Int(1e5) # Number of boxes
        out, θ_true, args = GaussMLE.GaussSim.genstack(boxsz, nboxes, :xynb; T=T, poissonnoise=true)

        # Fit all boxes in the stack
        θ_found, Σ_found = GaussMLE.GaussFit.fitstack(out, :xynb, args)

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

end

# Include GPU tests if requested
if get(ENV, "GAUSSMLE_TEST_GPU", "false") == "true"
    include("gpu_tests.jl")
else
    @info "GPU tests skipped. Set GAUSSMLE_TEST_GPU=true to run them."
end
