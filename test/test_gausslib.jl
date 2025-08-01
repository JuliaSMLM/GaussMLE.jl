# Internal function tests for GaussLib module

@testset "GaussLib internals" begin
    using GaussMLE.GaussLib: integral_gaussian_1d, compute_alpha, derivative_alpha_z, 
                second_derivative_alpha_z, center_of_mass_2d, gaussian_max_min_2d

    @testset "1D Gaussian integral" begin
        @test integral_gaussian_1d(1, 2.0, 1.0) ≈ 0.2417303374571288
        @test integral_gaussian_1d(0, 0.0, 1.0) ≈ 0.5  # Center of Gaussian
        @test integral_gaussian_1d(1, 0.0, 0.5) > integral_gaussian_1d(1, 0.0, 1.0)  # Narrower Gaussian
    end
    
    @testset "Alpha computation functions" begin
        z, Ax, Bx, d = 1.0, 2.0, 3.0, 4.0
        
        # Test compute_alpha
        @test compute_alpha(z, Ax, Bx, d) ≈ 1.0 + (z / d)^2 + Ax * (z / d)^3 + Bx * (z / d)^4
        @test compute_alpha(0.0, Ax, Bx, d) ≈ 1.0  # At z=0
        
        # Test first derivative
        @test derivative_alpha_z(z, Ax, Bx, d) ≈ 2.0 * z / d^2 + 3.0 * Ax * z^2 / d^3 + 4.0 * Bx * z^3 / d^4
        @test derivative_alpha_z(0.0, Ax, Bx, d) ≈ 0.0  # At z=0
        
        # Test second derivative
        @test second_derivative_alpha_z(z, Ax, Bx, d) ≈ 2.0 / d^2 + 6.0 * Ax * z / d^3 + 12.0 * Bx * z^2 / d^4
        @test second_derivative_alpha_z(0.0, Ax, Bx, d) ≈ 2.0 / d^2  # At z=0
    end
    
    @testset "2D operations" begin
        # Test center of mass
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        @test all(center_of_mass_2d(3, data) .≈ (2.1333333333333333, 2.4))
        
        # Test with uniform data
        uniform_data = ones(9)
        com_x, com_y = center_of_mass_2d(3, uniform_data)
        @test com_x ≈ 2.0  # Center of 3x3 grid
        @test com_y ≈ 2.0

        # Test Gaussian max/min
        @test all(gaussian_max_min_2d(3, 1.0, data) .≈ (6.985605655276496, 3.0143943447235038))
    end
end