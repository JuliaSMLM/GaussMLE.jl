using Pkg
Pkg.activate("dev")

using GaussMLE
using Printf
using Statistics
using Random
using CairoMakie

# Parameters (adjust these as needed)
box_size = 9  # Larger box for astigmatic PSFs
n_photons_levels = [500.0f0, 1000.0f0, 2000.0f0, 5000.0f0]
background = 5.0f0  # More realistic background level
z_test_positions = Float32[-400, -200, -100, 0, 100, 200, 400]  # nm
n_realizations = 200  # Number of spots per condition
iterations = 50  # Fitting iterations
seed = 42
plot_size = (1200, 900)

# Output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("="^80)
println("Z-Fitting Performance Test with CRLB Analysis")
println("="^80)
println()
println("Testing astigmatic PSF z-fitting accuracy and precision")
println("Comparing empirical uncertainties to CRLB predictions")
println()

# Set up the astigmatic PSF model with optimized parameters
psf_model = GaussMLE.AstigmaticXYZNB{Float32}(
    1.3f0, 1.3f0,  # σx₀, σy₀ - base PSF widths
    0.0f0, 0.0f0,  # Ax, Ay - no cubic terms for symmetric behavior
    0.0f0, 0.0f0,  # Bx, By - no quartic terms
    250.0f0,       # γ - focal plane offset (500nm total separation)
    400.0f0        # d - depth scale
)

# Results storage
results_data = Dict{Float32, Dict}()

for n_photons in n_photons_levels
    println("="^80)
    println("Testing with N = $(Int(n_photons)) photons")
    println("="^80)
    
    # Storage for this photon level
    z_bias = Float32[]
    z_std_empirical = Float32[]
    z_std_crlb = Float32[]
    z_positions_tested = Float32[]
    
    x_bias = Float32[]
    x_std_empirical = Float32[]
    x_std_crlb = Float32[]
    
    for z_true in z_test_positions
        println(@sprintf("\nZ = %6.0f nm:", z_true))
        
        # Generate multiple realizations
        spots = GaussMLE.SimulatedSpot{Float32}[]
        for i in 1:n_realizations
            push!(spots, GaussMLE.SimulatedSpot{Float32}(
                Float32(box_size/2 + 0.5),  # Center x
                Float32(box_size/2 + 0.5),  # Center y
                n_photons,
                background,
                nothing, nothing, nothing,
                z_true
            ))
        end
        
        # Generate data
        Random.seed!(seed)
        data, true_params = GaussMLE.generate_spots_data(
            psf_model,
            spots,
            box_size
        )
        
        # Fit
        fitter = GaussMLE.GaussMLEFitter(
            psf_model = psf_model,
            device = GaussMLE.CPU(),
            iterations = iterations
        )
        
        results = GaussMLE.fit(fitter, data)
        
        # Calculate statistics for z
        z_fitted = results.z
        z_errors_reported = results.z_error
        
        # Remove any Inf or NaN values
        valid_idx = isfinite.(z_fitted) .& isfinite.(z_errors_reported)
        z_fitted_valid = z_fitted[valid_idx]
        z_errors_valid = z_errors_reported[valid_idx]
        
        if length(z_fitted_valid) > 10  # Need enough valid fits
            # Z statistics
            z_mean = mean(z_fitted_valid)
            z_std_emp = std(z_fitted_valid)
            z_mean_crlb = mean(z_errors_valid)
            z_bias_val = z_mean - z_true
            
            push!(z_positions_tested, z_true)
            push!(z_bias, z_bias_val)
            push!(z_std_empirical, z_std_emp)
            push!(z_std_crlb, z_mean_crlb)
            
            # X statistics (for comparison)
            x_fitted_valid = results.x[valid_idx]
            x_errors_valid = results.x_error[valid_idx]
            x_mean = mean(x_fitted_valid)
            x_std_emp = std(x_fitted_valid)
            x_mean_crlb = mean(x_errors_valid)
            x_bias_val = x_mean - true_params[:x][1]
            
            push!(x_bias, x_bias_val)
            push!(x_std_empirical, x_std_emp)
            push!(x_std_crlb, x_mean_crlb)
            
            # CRLB ratio
            crlb_ratio_z = z_std_emp / z_mean_crlb
            crlb_ratio_x = x_std_emp / x_mean_crlb
            
            @printf("  Z: bias=%7.2f nm, σ_emp=%6.2f nm, σ_CRLB=%6.2f nm, ratio=%5.3f\n",
                    z_bias_val, z_std_emp, z_mean_crlb, crlb_ratio_z)
            @printf("  X: bias=%7.4f px, σ_emp=%6.4f px, σ_CRLB=%6.4f px, ratio=%5.3f\n",
                    x_bias_val, x_std_emp, x_mean_crlb, crlb_ratio_x)
            
            # Check quality
            if abs(crlb_ratio_z - 1.0) > 0.2
                println("  ⚠ Z CRLB ratio deviates from 1.0")
            end
            if abs(z_bias_val) > z_mean_crlb
                println("  ⚠ Z bias exceeds CRLB uncertainty")
            end
        else
            println("  ⚠ Too many failed fits ($(n_realizations - length(z_fitted_valid))/$(n_realizations))")
        end
    end
    
    # Store results for this photon level
    results_data[n_photons] = Dict(
        :z_positions => z_positions_tested,
        :z_bias => z_bias,
        :z_std_empirical => z_std_empirical,
        :z_std_crlb => z_std_crlb,
        :x_bias => x_bias,
        :x_std_empirical => x_std_empirical,
        :x_std_crlb => x_std_crlb
    )
end

# Create comprehensive plots
println()
println("="^80)
println("Creating Performance Plots")
println("="^80)

fig = Figure(size=plot_size)

# Plot 1: Z Bias vs Position for different photon levels
ax1 = Axis(fig[1, 1],
    xlabel = "Z Position (nm)",
    ylabel = "Z Bias (nm)",
    title = "Z-Fitting Bias vs Position",
    xlabelsize = 12,
    ylabelsize = 12
)

colors = [:blue, :red, :green, :orange]
markers = [:circle, :rect, :diamond, :star5]

for (idx, n_photons) in enumerate(n_photons_levels)
    if haskey(results_data, n_photons)
        data = results_data[n_photons]
        scatter!(ax1, data[:z_positions], data[:z_bias],
                label = "N = $(Int(n_photons))",
                color = colors[idx],
                marker = markers[idx],
                markersize = 10)
    end
end
hlines!(ax1, [0.0], color = :black, linestyle = :dash, alpha = 0.5)
axislegend(ax1, position = :lt)

# Plot 2: Z Uncertainty Comparison (Empirical vs CRLB)
ax2 = Axis(fig[1, 2],
    xlabel = "Z Position (nm)",
    ylabel = "Z Uncertainty (nm)",
    title = "Z Uncertainty: Empirical vs CRLB",
    xlabelsize = 12,
    ylabelsize = 12
)

for (idx, n_photons) in enumerate(n_photons_levels)
    if haskey(results_data, n_photons)
        data = results_data[n_photons]
        # Empirical
        lines!(ax2, data[:z_positions], data[:z_std_empirical],
              label = "Emp N=$(Int(n_photons))",
              color = colors[idx],
              linewidth = 2)
        # CRLB
        lines!(ax2, data[:z_positions], data[:z_std_crlb],
              label = "CRLB N=$(Int(n_photons))",
              color = colors[idx],
              linewidth = 2,
              linestyle = :dash)
    end
end
axislegend(ax2, position = :rt, nbanks = 2)

# Plot 3: CRLB Ratio for Z
ax3 = Axis(fig[2, 1],
    xlabel = "Z Position (nm)",
    ylabel = "σ_empirical / σ_CRLB",
    title = "Z CRLB Ratio (should be ≈ 1)",
    xlabelsize = 12,
    ylabelsize = 12
)

for (idx, n_photons) in enumerate(n_photons_levels)
    if haskey(results_data, n_photons)
        data = results_data[n_photons]
        ratio = data[:z_std_empirical] ./ data[:z_std_crlb]
        scatter!(ax3, data[:z_positions], ratio,
                label = "N = $(Int(n_photons))",
                color = colors[idx],
                marker = markers[idx],
                markersize = 10)
    end
end
hlines!(ax3, [1.0], color = :black, linewidth = 2)
hlines!(ax3, [0.8, 1.2], color = :black, linestyle = :dash, alpha = 0.5)
ylims!(ax3, 0.5, 1.5)
axislegend(ax3, position = :rt)

# Plot 4: X vs Z Precision Comparison
ax4 = Axis(fig[2, 2],
    xlabel = "Z Position (nm)",
    ylabel = "Localization Precision (nm for Z, pixels for X)",
    title = "X and Z Localization Precision",
    xlabelsize = 12,
    ylabelsize = 12,
    yscale = log10
)

# Focus on one photon level for clarity
n_photons_display = 1000.0f0
if haskey(results_data, n_photons_display)
    data = results_data[n_photons_display]
    
    # Z precision in nm
    lines!(ax4, data[:z_positions], data[:z_std_crlb],
          label = "Z CRLB (nm)",
          color = :blue,
          linewidth = 2)
    scatter!(ax4, data[:z_positions], data[:z_std_empirical],
            label = "Z Empirical (nm)",
            color = :blue,
            markersize = 8)
    
    # X precision in pixels (convert to nm for comparison, assuming 100nm/pixel)
    pixel_to_nm = 100.0f0
    x_crlb_nm = data[:x_std_crlb] .* pixel_to_nm
    x_emp_nm = data[:x_std_empirical] .* pixel_to_nm
    
    lines!(ax4, data[:z_positions], x_crlb_nm,
          label = "X CRLB (nm)",
          color = :red,
          linewidth = 2)
    scatter!(ax4, data[:z_positions], x_emp_nm,
            label = "X Empirical (nm)",
            color = :red,
            markersize = 8)
end
axislegend(ax4, position = :rt)

# Save figure
plot_file = joinpath(output_dir, "z_fitting_crlb_analysis.png")
save(plot_file, fig)
println("Plots saved to $(plot_file)")

# Summary statistics
println()
println("="^80)
println("Summary Statistics")
println("="^80)

for n_photons in n_photons_levels
    if haskey(results_data, n_photons)
        data = results_data[n_photons]
        
        # Calculate overall statistics
        mean_bias = mean(abs.(data[:z_bias]))
        max_bias = maximum(abs.(data[:z_bias]))
        mean_ratio = mean(data[:z_std_empirical] ./ data[:z_std_crlb])
        
        println(@sprintf("\nN = %d photons:", Int(n_photons)))
        println(@sprintf("  Mean |bias|: %.2f nm", mean_bias))
        println(@sprintf("  Max |bias|:  %.2f nm", max_bias))
        println(@sprintf("  Mean CRLB ratio: %.3f", mean_ratio))
        
        if mean_ratio > 1.2
            println("  ⚠ Empirical uncertainty exceeds CRLB by >20%")
        elseif mean_ratio < 0.8
            println("  ⚠ Empirical uncertainty below CRLB by >20%")
        else
            println("  ✓ CRLB ratio within acceptable range")
        end
        
        if mean_bias > 20.0
            println("  ⚠ Mean bias exceeds 20 nm")
        else
            println("  ✓ Bias within acceptable range")
        end
    end
end

println()
println("="^80)
println("Z-Fitting CRLB Analysis Complete")
println("="^80)

# Save numerical results to CSV
csv_file = joinpath(output_dir, "z_fitting_crlb_data.csv")
open(csv_file, "w") do io
    println(io, "n_photons,z_position,z_bias,z_std_empirical,z_std_crlb,crlb_ratio,x_bias,x_std_empirical,x_std_crlb")
    for n_photons in n_photons_levels
        if haskey(results_data, n_photons)
            data = results_data[n_photons]
            for i in 1:length(data[:z_positions])
                println(io, @sprintf("%d,%.1f,%.3f,%.3f,%.3f,%.3f,%.5f,%.5f,%.5f",
                    Int(n_photons),
                    data[:z_positions][i],
                    data[:z_bias][i],
                    data[:z_std_empirical][i],
                    data[:z_std_crlb][i],
                    data[:z_std_empirical][i] / data[:z_std_crlb][i],
                    data[:x_bias][i],
                    data[:x_std_empirical][i],
                    data[:x_std_crlb][i]))
            end
        end
    end
end
println("\nNumerical data saved to $(csv_file)")