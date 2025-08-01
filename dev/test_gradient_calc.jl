#!/usr/bin/env julia

using GaussMLE
using Printf

# Test gradient calculation manually
roi_size = 7
x = 4.0
y = 4.0
n = 1000.0
bg = 10.0
sigma = 1.5

println("Manual Gradient Calculation Test")
println("Parameters: x=$x, y=$y, n=$n, bg=$bg")
println()

# Generate data
roi = zeros(roi_size, roi_size)
for i in 1:roi_size
    for j in 1:roi_size
        xi = Float64(i)
        yi = Float64(j)
        dx = xi - x
        dy = yi - y
        gauss = n * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
        roi[i, j] = bg + gauss
    end
end

# Calculate gradients at true parameters
grad_x = 0.0
grad_y = 0.0
grad_n = 0.0
grad_bg = 0.0

hess_xx = 0.0
hess_yy = 0.0
hess_nn = 0.0
hess_bb = 0.0

println("Pixel contributions (center 3x3):")
for i in 3:5
    for j in 3:5
        xi = Float64(i)
        yi = Float64(j)
        dx = xi - x
        dy = yi - y
        
        # Model value
        gaussian = n * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
        model_val = bg + gaussian
        data_val = roi[i, j]
        
        # Derivatives
        d_x = gaussian * dx / sigma^2
        d_y = gaussian * dy / sigma^2
        d_n = gaussian / n  # Note: should be gaussian/n not just gaussian
        d_bg = 1.0
        
        # For MLE under Poisson noise
        residual = (data_val - model_val) / model_val
        weight = 1.0 / model_val
        
        # Accumulate
        global grad_x += residual * d_x
        global grad_y += residual * d_y
        global grad_n += residual * d_n
        global grad_bg += residual * d_bg
        
        global hess_xx += weight * d_x * d_x
        global hess_yy += weight * d_y * d_y
        global hess_nn += weight * d_n * d_n
        global hess_bb += weight * d_bg * d_bg
        
        @printf("  (%d,%d): data=%.1f, model=%.1f, dx=%.2f, dy=%.2f, d_x=%.3f\n", 
                i, j, data_val, model_val, dx, dy, d_x)
    end
end

println("\nTotal gradients (should be ~0 at true params):")
println("  grad_x = $grad_x")
println("  grad_y = $grad_y")
println("  grad_n = $grad_n")
println("  grad_bg = $grad_bg")

println("\nDiagonal Hessian elements:")
println("  hess_xx = $hess_xx")
println("  hess_yy = $hess_yy")
println("  hess_nn = $hess_nn")
println("  hess_bb = $hess_bb")

# Now test with perturbed parameters
println("\n" * ("=" ^ 50))
println("Test with perturbed x = 3.5 (should have positive grad_x)")

x_pert = 3.5
grad_x_pert = 0.0

for i in 1:roi_size
    for j in 1:roi_size
        xi = Float64(i)
        yi = Float64(j)
        dx = xi - x_pert
        dy = yi - y
        
        gaussian = n * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
        model_val = bg + gaussian
        data_val = roi[i, j]
        
        d_x = gaussian * dx / sigma^2
        residual = (data_val - model_val) / model_val
        
        global grad_x_pert += residual * d_x
    end
end

println("grad_x at x=3.5: $grad_x_pert (should be positive to push x toward 4.0)")

# Test the other direction
x_pert = 4.5
grad_x_pert = 0.0

for i in 1:roi_size
    for j in 1:roi_size
        xi = Float64(i)
        yi = Float64(j)
        dx = xi - x_pert
        dy = yi - y
        
        gaussian = n * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
        model_val = bg + gaussian
        data_val = roi[i, j]
        
        d_x = gaussian * dx / sigma^2
        residual = (data_val - model_val) / model_val
        
        global grad_x_pert += residual * d_x
    end
end

println("grad_x at x=4.5: $grad_x_pert (should be negative to push x toward 4.0)")