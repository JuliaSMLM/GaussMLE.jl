using Pkg
Pkg.activate("dev")

#
# Debug coordinate system and initialization

using GaussMLE
using CUDA
using Printf

# Parameters (adjust these as needed)
roi_size = 7      # Size of ROI to debug
verbose = true    # Print detailed output

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)  # Create if it doesn't exist

println("=== Coordinate System Debug ===")
println("This script debugs coordinate system and initialization")
println("Parameters: roi_size=$roi_size")
println("=" ^ 50)

# Test simple center of mass calculation
roi = zeros(Float32, roi_size, roi_size)

# Place a single bright pixel at known location
test_positions = [
    (3, 3, "center-1"),
    (4, 4, "center"),
    (5, 5, "center+1"),
    (1, 1, "corner (1,1)"),
    (7, 7, "corner (7,7)"),
    (1, 4, "edge case")
]

for (test_i, test_j, desc) in test_positions
    # Clear ROI
    fill!(roi, 10.0)  # background
    
    # Add bright spot
    roi[test_i, test_j] = 100.0
    
    println("\nTest: Bright pixel at ($test_i, $test_j) - $desc")
    
    # Manual center of mass calculation (1-based)
    bg = 10.0
    sum_above_bg = 0.0
    sum_x_weighted = 0.0
    sum_y_weighted = 0.0
    
    for i in 1:roi_size
        for j in 1:roi_size
            val_above_bg = max(roi[i,j] - bg, 0.0)
            sum_above_bg += val_above_bg
            sum_x_weighted += val_above_bg * j  # Column index (x)
            sum_y_weighted += val_above_bg * i  # Row index (y)
        end
    end
    
    # 1-based center of mass
    com_x_1based = sum_x_weighted / sum_above_bg
    com_y_1based = sum_y_weighted / sum_above_bg
    
    # Convert to 0-based for model
    com_x_0based = com_x_1based - 1
    com_y_0based = com_y_1based - 1
    
    println("  1-based COM: x=$com_x_1based, y=$com_y_1based")
    println("  0-based COM: x=$com_x_0based, y=$com_y_0based")
    println("  Expected 0-based: x=$(test_j-1), y=$(test_i-1)")
    
    # Test with GPU kernel
    d_data = CuArray(reshape(roi, roi_size, roi_size, 1))
    d_params = CUDA.zeros(Float32, 4, 1)
    d_crlb = CUDA.zeros(Float32, 4, 1)
    
    GaussMLE.GaussGPU.launch_cuda_fit_kernel!(d_data, d_params, d_crlb, Int32(roi_size))
    CUDA.synchronize()
    
    params = Array(d_params)
    println("  GPU result: x=$(params[1,1]), y=$(params[2,1])")
end

# Test with realistic Gaussian blob
println("\n" * "=" * 50)
println("Realistic Gaussian Test")
println("=" * 50)

# Parameters in 0-based coordinates
x_true = 3.2
y_true = 2.8
n_true = 1000.0
bg_true = 10.0
sigma = 1.5

roi = zeros(Float32, roi_size, roi_size)
for i in 1:roi_size
    for j in 1:roi_size
        # Convert to 0-based for calculation
        xi = i - 1
        yi = j - 1
        dx = xi - x_true
        dy = yi - y_true
        gauss = n_true * exp(-(dx^2 + dy^2) / (2 * sigma^2)) / (2π * sigma^2)
        roi[i, j] = bg_true + gauss
    end
end

println("True parameters (0-based): x=$x_true, y=$y_true")

# Show ROI
println("\nROI values:")
for i in 1:roi_size
    for j in 1:roi_size
        @printf("%6.1f ", roi[i,j])
    end
    println()
end

# Calculate center of mass manually
sum_val = sum(roi)
min_val = minimum(roi)
bg_est = min_val

sum_above_bg = 0.0
sum_x_weighted = 0.0
sum_y_weighted = 0.0

for i in 1:roi_size
    for j in 1:roi_size
        val_above_bg = max(roi[i,j] - bg_est, 0.0)
        sum_above_bg += val_above_bg
        sum_x_weighted += val_above_bg * j  # 1-based column
        sum_y_weighted += val_above_bg * i  # 1-based row
    end
end

com_x_1based = sum_x_weighted / sum_above_bg
com_y_1based = sum_y_weighted / sum_above_bg
com_x_0based = com_x_1based - 1
com_y_0based = com_y_1based - 1

println("\nManual calculation:")
println("  Background estimate: $bg_est")
println("  1-based COM: x=$com_x_1based, y=$com_y_1based")
println("  0-based COM: x=$com_x_0based, y=$com_y_0based")

# Test GPU
d_data = CuArray(reshape(roi, roi_size, roi_size, 1))
d_params = CUDA.zeros(Float32, 4, 1)
d_crlb = CUDA.zeros(Float32, 4, 1)

GaussMLE.GaussGPU.launch_cuda_fit_kernel!(d_data, d_params, d_crlb, Int32(roi_size))
CUDA.synchronize()

params = Array(d_params)
println("\nGPU fit result:")
println("  x=$(params[1,1]), y=$(params[2,1]), n=$(params[3,1]), bg=$(params[4,1])")
println("  Error: x=$(abs(params[1,1] - x_true)), y=$(abs(params[2,1] - y_true))")