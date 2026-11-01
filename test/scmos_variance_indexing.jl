# Regression test: the sCMOS variance map is indexed at camera coordinates (corner + ROI
# pixel), with x and y not swapped. Moved from the Monte Carlo validation file, which is now
# in Long.
using Test, GaussMLE, SMLMData, Random, Statistics

box_size = 15

@testset "sCMOS Variance Map Spatial Indexing" begin
    # This test verifies that the variance map is indexed correctly using
    # camera coordinates (corner + roi_position), not just ROI-local coordinates.
    #
    # Design: Asymmetric gradient where variance varies 10x faster in y than x.
    # Two groups of ROIs placed at strategic positions will show inverted
    # uncertainty ratios if x/y indexing is swapped.

    psf_model = GaussMLE.GaussianXYNB(0.13f0)
    camera_size = 512

    # Create asymmetric gradient variance map:
    # var[i,j] = base + α_y*(i-1) + α_x*(j-1)
    # where α_y = 0.5 (steep) and α_x = 0.05 (shallow)
    # This makes y-index 10x more important than x-index
    base_var = 1.0f0
    α_y = 0.5f0   # Variance increases 0.5 e⁻² per row
    α_x = 0.05f0  # Variance increases 0.05 e⁻² per column

    readnoise_map = Matrix{Float32}(undef, camera_size, camera_size)
    for j in 1:camera_size, i in 1:camera_size
        variance = base_var + α_y * (i - 1) + α_x * (j - 1)
        readnoise_map[i, j] = sqrt(variance)  # SCMOSCamera takes std dev
    end

    scmos = SMLMData.SCMOSCamera(
        camera_size, camera_size, 0.1f0, readnoise_map,
        offset = 100.0f0,
        gain = 1.0f0,   # Simplify: 1 e⁻/ADU
        qe = 1.0f0      # Simplify: 100% QE
    )

    # Strategic positions to detect x/y swap:
    # Group A: low-y, high-x (y_corner=50, x_corner=400)
    #   Correct variance ≈ 1 + 0.5*57 + 0.05*407 = 1 + 28.5 + 20.35 ≈ 50 e⁻²
    # Group B: high-y, low-x (y_corner=400, x_corner=50)
    #   Correct variance ≈ 1 + 0.5*407 + 0.05*57 = 1 + 203.5 + 2.85 ≈ 207 e⁻²
    #
    # If x/y swapped: A would see ~207, B would see ~50 (ratio inverts)

    n_per_group = 50
    n_rois = 2 * n_per_group

    # Fixed positions within ROI (center)
    Random.seed!(123)
    true_params = Matrix{Float32}(undef, 4, n_rois)
    for i in 1:n_rois
        true_params[1, i] = Float32(box_size / 2 + 0.3 * randn())  # x in ROI
        true_params[2, i] = Float32(box_size / 2 + 0.3 * randn())  # y in ROI
        true_params[3, i] = 2000.0f0  # High photons for good SNR
        true_params[4, i] = 5.0f0     # Background
    end

    # Corners: first half at low-variance position, second half at high-variance
    corners = zeros(Int32, 2, n_rois)
    for i in 1:n_per_group
        # Group A: low-y (row 50), high-x (column 400)
        corners[1, i] = Int32(400)  # x_corner
        corners[2, i] = Int32(50)   # y_corner
    end
    for i in (n_per_group + 1):n_rois
        # Group B: high-y (row 400), low-x (column 50)
        corners[1, i] = Int32(50)   # x_corner
        corners[2, i] = Int32(400)  # y_corner
    end

    # Generate and fit
    batch = GaussMLE.generate_roi_batch(
        scmos, psf_model;
        n_rois = n_rois,
        roi_size = box_size,
        true_params = true_params,
        corners = corners,
        seed = 123
    )

    fitter = GaussMLE.GaussMLEConfig(psf_model = psf_model, device = GaussMLE.CPU())
    smld, _info = GaussMLE.fit(batch, fitter)

    # Extract uncertainties for each group
    σ_x_A = [smld.emitters[i].σ_x for i in 1:n_per_group]
    σ_x_B = [smld.emitters[i].σ_x for i in (n_per_group + 1):n_rois]

    mean_σ_A = mean(σ_x_A)
    mean_σ_B = mean(σ_x_B)

    # Compute expected variance at center of each ROI group
    # Group A center: (y=50+7, x=400+7) = (57, 407)
    # Group B center: (y=400+7, x=50+7) = (407, 57)
    center_offset = box_size ÷ 2
    var_A = base_var + α_y * (50 + center_offset - 1) + α_x * (400 + center_offset - 1)
    var_B = base_var + α_y * (400 + center_offset - 1) + α_x * (50 + center_offset - 1)

    # Expected ratio of uncertainties (σ ∝ √variance in readnoise-dominated regime)
    expected_ratio = sqrt(var_B / var_A)  # Should be ~2.0
    actual_ratio = mean_σ_B / mean_σ_A

    # Key test: If indexing is correct, Group B (high-y position) should have
    # LARGER uncertainties than Group A (low-y position)
    # If x/y swapped, the ratio would be inverted (<1 instead of >1)

    @test actual_ratio > 1.5  # Must be > 1, definitively catches x/y swap
    @test actual_ratio < 3.0  # Sanity check, shouldn't be too extreme

    # Check ratio is reasonably close to expected (within 30%)
    # Note: Not exact because Poisson variance also contributes
    @test abs(actual_ratio - expected_ratio) / expected_ratio < 0.4

    # Additional check: verify the absolute magnitudes are reasonable
    # At high variance (207 e⁻²), readnoise dominates over Poisson (~5 e⁻)
    @test mean_σ_B > mean_σ_A  # Sanity check
end

# The kernel indexes the variance map by corner, so an ROI off the sensor is a device-side
# bounds error: fit(::ROIBatch) rejects it on the host before any launch, naming the ROI.
@testset "ROIs outside the sCMOS sensor are rejected before any launch" begin
    psf_model = GaussMLE.GaussianXYNB(0.13f0)
    sensor, box, n = 32, 7, 3
    scmos = SMLMData.SCMOSCamera(
        sensor, sensor, 0.1f0, fill(2.0f0, sensor, sensor);
        offset = 100.0f0, gain = 1.0f0, qe = 1.0f0
    )
    last_corner = sensor - box + 1
    corners = Int32[1 last_corner 8; 1 last_corner 9]  # x row, y row
    batch = GaussMLE.generate_roi_batch(
        scmos, psf_model; n_rois = n, roi_size = box, corners, seed = 5
    )
    with_corners(x, y) = ROIBatch(batch.data, x, y, batch.frame_indices, scmos)

    # ROIs touching the sensor edges are on the sensor.
    smld, _ = fit(batch; psf_model, backend = :cpu)
    @test length(smld.emitters) == n

    # (x corners, y corners, index of the first bad ROI)
    bad = (
        (Int32[1, 8, 0], Int32[1, 1, 9], 3),    # x corner < 1
        (Int32[1, 8, 8], Int32[1, 1, -4], 3),   # y corner < 1
        (Int32[1, 8, 8], Int32[1, 27, 9], 2),   # y corner + box - 1 > sensor
        (Int32[1, 27, 8], Int32[1, 8, 9], 2),   # x corner + box - 1 > sensor
        (Int32[40, 27, 8], Int32[1, 8, 9], 1),  # the first of several bad ROIs
    )
    for (x, y, index) in bad, backend in (:cpu, :gpu, :auto)
        err = try
            fit(with_corners(x, y); psf_model, backend)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test err isa ArgumentError && occursin("ROI $index ", err.msg)
    end

    # fit(::Array; variance_map) tiles ROIs at x = 1, 1 + box, ...: a map one box wide holds
    # only the first.
    for backend in (:cpu, :gpu, :auto)
        err = try
            fit(
                batch.data[:, :, 1:2], GaussMLEConfig(; psf_model, backend);
                variance_map = fill(2.0f0, box, box)
            )
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test err isa ArgumentError && occursin("ROI 2 ", err.msg)
    end

    # An IdealCamera never indexes calibration by corner and converts corners to microns
    # linearly, so ROIs off its nominal sensor still fit (fit(::Array) tiles ROIs far
    # beyond its stand-in camera this way).
    ideal = ROIBatch(
        batch.data, Int32[1, 100, 5000], Int32[1, 1, 1], batch.frame_indices,
        IdealCamera(sensor, sensor, 0.1f0)
    )
    smld, _ = fit(ideal; psf_model, backend = :cpu)
    @test all(e -> isfinite(e.x) && isfinite(e.y), smld.emitters)
    @test smld.emitters[3].x > smld.emitters[2].x > smld.emitters[1].x
end
