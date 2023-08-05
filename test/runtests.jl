using GaussMLE
using SpecialFunctions
using LinearAlgebra
using Test

@testset "GaussMLE.jl" begin

    # Importing specific functions for testing from the provided baselibrary.jl content
    using GaussMLE: integral_gaussian_1d, compute_alpha, derivative_alpha_z, 
                    second_derivative_alpha_z, center_of_mass_2d, gaussian_max_min_2d, matrix_inverse!
    
    @testset "baselibrary" begin
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
        
        # Test for matrix_inverse (assuming the function signature hasn't changed)
        m = [4.0 2.0; 2.0 2.0]
        m_inv_la = inv(m)
        m_inverse, m_inverse_diag = matrix_inverse!(m, 2)
        @test all(m_inverse .≈ m_inv_la)
        @test all(m_inverse_diag .≈ diag(m_inv_la))
    end
end
