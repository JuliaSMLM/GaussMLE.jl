using Pkg
Pkg.activate("dev")

using GaussMLE
using Printf
using Statistics
using CairoMakie

# Parameters (adjust these as needed)
box_size = 11  # Size of ROI for each PSF
n_photons = 10000.0f0  # High photon count for noiseless data
background = 100.0f0  # Background level
z_range = (-0.8, 0.8)  # Z range in microns
z_step = 0.1  # Step size in microns
plot_size = (1000, 800)  # Figure size
iterations = 50  # Fitting iterations

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)
x_true = Float32(box_size/2 + 0.5)
y_true = Float32(box_size/2 + 0.5)

# Set up the astigmatic PSF model
astigmatic_model = GaussMLE.AstigmaticXYZNB{Float32}(
    1.3f0, 1.3f0,  # σx₀, σy₀ - base PSF widths (symmetric)
    0.0f0, 0.0f0,  # Ax, Ay - no cubic terms for symmetric behavior
    0.0f0, 0.0f0,  # Bx, By - no quartic terms for pure quadratic
    250.0f0,       # γ - focal plane offset (500nm total separation)
    400.0f0        # d - depth scale (smaller = faster change)
)

# Set up anisotropic model for fitting (xynbsxsy)
anisotropic_model = GaussMLE.GaussianXYNBSXSY()

# Z positions to test (in microns, then convert to nm)
z_positions_um = range(z_range[1], z_range[2], step=z_step)
z_positions_nm = Float32.(z_positions_um * 1000)  # Convert to nm

# Storage for results
n_z = length(z_positions_nm)
fitted_sx = zeros(Float32, n_z)
fitted_sy = zeros(Float32, n_z)
fitted_sx_error = zeros(Float32, n_z)
fitted_sy_error = zeros(Float32, n_z)
expected_sx = zeros(Float32, n_z)
expected_sy = zeros(Float32, n_z)

println("="^80)
println("Astigmatic PSF Analysis: Fitting with Anisotropic Model")
println("="^80)
println()
println("Generating noiseless astigmatic PSF images from z = -1 to +1 μm")
println("Fitting each with anisotropic (σx, σy) model")
println()

# Header for results table
@printf("%-8s %-10s %-10s %-10s %-10s %-10s %-10s\n", 
        "z (μm)", "True σx", "True σy", "Fit σx", "Fit σy", "Error σx", "Error σy")
println("-"^80)

for (idx, z_nm) in enumerate(z_positions_nm)
    z_um = z_nm / 1000
    
    # Calculate expected sigmas from astigmatic model
    zx = z_nm - astigmatic_model.γ
    zy = z_nm + astigmatic_model.γ
    zx_norm = zx / astigmatic_model.d
    zy_norm = zy / astigmatic_model.d
    alpha_x = 1.0f0 + zx_norm^2 + astigmatic_model.Ax * zx_norm^3 + astigmatic_model.Bx * zx_norm^4
    alpha_y = 1.0f0 + zy_norm^2 + astigmatic_model.Ay * zy_norm^3 + astigmatic_model.By * zy_norm^4
    
    # Ensure alpha is positive
    alpha_x = max(0.1f0, alpha_x)
    alpha_y = max(0.1f0, alpha_y)
    
    sigma_x_true = astigmatic_model.σx₀ * sqrt(alpha_x)
    sigma_y_true = astigmatic_model.σy₀ * sqrt(alpha_y)
    
    expected_sx[idx] = sigma_x_true
    expected_sy[idx] = sigma_y_true
    
    # Generate noiseless data using astigmatic model
    data = zeros(Float32, box_size, box_size)
    θ_true = GaussMLE.Params{5}(x_true, y_true, z_nm, n_photons, background)
    
    for j in 1:box_size, i in 1:box_size
        data[i, j] = GaussMLE.evaluate_psf(astigmatic_model, i, j, θ_true)
    end
    
    # Reshape for fitting (needs 3D array)
    data_3d = reshape(data, box_size, box_size, 1)
    
    # Fit with anisotropic model
    fitter = GaussMLE.GaussMLEFitter(
        psf_model = anisotropic_model,
        device = GaussMLE.CPU(),
        iterations = iterations  # Use parameter from top
    )
    
    results = GaussMLE.fit(fitter, data_3d)
    
    # Extract fitted parameters
    fitted_sx[idx] = results.sigma_x[1]
    fitted_sy[idx] = results.sigma_y[1]
    fitted_sx_error[idx] = results.sigma_x_error[1]
    fitted_sy_error[idx] = results.sigma_y_error[1]
    
    # Print results
    @printf("%-8.2f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n",
            z_um, sigma_x_true, sigma_y_true, 
            fitted_sx[idx], fitted_sy[idx],
            fitted_sx_error[idx], fitted_sy_error[idx])
end

println()
println("="^80)
println("Summary Statistics")
println("="^80)

# Calculate errors
sx_errors = fitted_sx .- expected_sx
sy_errors = fitted_sy .- expected_sy

@printf("σx fit error: mean = %.4f, std = %.4f, max = %.4f\n",
        mean(sx_errors), std(sx_errors), maximum(abs.(sx_errors)))
@printf("σy fit error: mean = %.4f, std = %.4f, max = %.4f\n",
        mean(sy_errors), std(sy_errors), maximum(abs.(sy_errors)))

# Check if the crossing point is correct
crossing_idx = findfirst(expected_sx .> expected_sy)
if !isnothing(crossing_idx)
    z_cross = z_positions_nm[crossing_idx] / 1000
    println()
    @printf("σx/σy crossing point: z ≈ %.2f μm\n", z_cross)
end

# Save results to CSV file for plotting
z_um = z_positions_nm ./ 1000

output_file = joinpath(output_dir, "astigmatic_psf_data.csv")
open(output_file, "w") do io
    println(io, "z_um,expected_sx,expected_sy,fitted_sx,fitted_sy,error_sx,error_sy")
    for i in 1:n_z
        println(io, @sprintf("%.2f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                 z_um[i], expected_sx[i], expected_sy[i], 
                 fitted_sx[i], fitted_sy[i],
                 fitted_sx[i] - expected_sx[i], fitted_sy[i] - expected_sy[i]))
    end
end
println()
println("Data saved to $(output_file)")
println("This can be plotted with any plotting software")


# Check quality of fit
println("="^80)
println("Fit Quality Assessment")
println("="^80)

good_fit_count = 0
for idx in 1:n_z
    sx_match = abs(fitted_sx[idx] - expected_sx[idx]) < 0.05
    sy_match = abs(fitted_sy[idx] - expected_sy[idx]) < 0.05
    if sx_match && sy_match
        global good_fit_count += 1
    end
end

@printf("Good fits (error < 0.05 pixels): %d/%d (%.1f%%)\n", 
        good_fit_count, n_z, 100 * good_fit_count / n_z)

if good_fit_count == n_z
    println("✓ All fits match expected values - astigmatic PSF model is working correctly!")
else
    println("⚠ Some fits don't match - there may be issues with the model or fitting")
end

# Create the plot using CairoMakie
println()
println("="^80)
println("Creating Plot with CairoMakie")
println("="^80)

fig = Figure(size=plot_size)

# Main plot
ax = Axis(fig[1, 1],
    xlabel = "Z Position (μm)",
    ylabel = "PSF Width (pixels)",
    title = "Astigmatic PSF: Expected vs Fitted σx and σy",
    xlabelsize = 14,
    ylabelsize = 14,
    titlesize = 16
)

# Plot expected values
lines!(ax, z_um, expected_sx, label = "Expected σx", color = :blue, linewidth = 2, linestyle = :dash)
lines!(ax, z_um, expected_sy, label = "Expected σy", color = :red, linewidth = 2, linestyle = :dash)

# Plot fitted values
scatter!(ax, z_um, fitted_sx, label = "Fitted σx", color = :blue, markersize = 8)
scatter!(ax, z_um, fitted_sy, label = "Fitted σy", color = :red, markersize = 8)

# Add legend
axislegend(ax, position = :lt)

# Add a grid
ax.xgridvisible = true
ax.ygridvisible = true

# Error subplot
ax2 = Axis(fig[2, 1],
    xlabel = "Z Position (μm)",
    ylabel = "Error (pixels)",
    title = "Fitting Error",
    xlabelsize = 12,
    ylabelsize = 12,
    titlesize = 14
)

lines!(ax2, z_um, fitted_sx .- expected_sx, label = "σx error", color = :blue, linewidth = 2)
lines!(ax2, z_um, fitted_sy .- expected_sy, label = "σy error", color = :red, linewidth = 2)
hlines!(ax2, [0.0], color = :black, linestyle = :dash, alpha = 0.5)

axislegend(ax2, position = :rt)

# Adjust layout
fig[1, 1] = ax
fig[2, 1] = ax2
rowsize!(fig.layout, 1, Relative(0.65))
rowsize!(fig.layout, 2, Relative(0.35))

# Save the figure
plot_file = joinpath(output_dir, "astigmatic_psf_analysis.png")
save(plot_file, fig)
println("Plot saved to $(plot_file)")
println()
println("Analysis complete!")