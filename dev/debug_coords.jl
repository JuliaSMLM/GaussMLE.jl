using GaussMLE
using Random
using Distributions

# Simple test: Generate one spot at known position
box_size = 11
x_true = 6.0f0  # Center of 11x11 box
y_true = 6.0f0
N_true = 1000.0f0
bg_true = 10.0f0
σ = 1.3f0

# Generate data using integrated Gaussian
data = zeros(Float32, box_size, box_size)
for j in 1:box_size, i in 1:box_size
    dx = Float32(i) - x_true
    dy = Float32(j) - y_true
    psf_x = GaussMLE.integrated_gaussian_1d(dx, σ)
    psf_y = GaussMLE.integrated_gaussian_1d(dy, σ)
    mu = bg_true + N_true * psf_x * psf_y
    data[i, j] = Float32(rand(Poisson(mu)))
end

# Fit with our model
println("True parameters:")
println("  x = $x_true")
println("  y = $y_true")
println("  N = $N_true")
println("  bg = $bg_true")

data_3d = reshape(data, box_size, box_size, 1)
fitter = GaussMLE.GaussMLEFitter(
    psf_model = GaussMLE.GaussianXYNB(σ),
    device = GaussMLE.CPU()
)

results = GaussMLE.fit(fitter, data_3d)

println("\nFitted parameters:")
println("  x = $(results.x[1])")
println("  y = $(results.y[1])")
println("  N = $(results.photons[1])")
println("  bg = $(results.background[1])")

println("\nErrors:")
println("  x error = $(results.x[1] - x_true)")
println("  y error = $(results.y[1] - y_true)")