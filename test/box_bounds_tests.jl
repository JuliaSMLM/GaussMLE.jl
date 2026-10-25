# Exact, noise-free pixel-integrated PSF data with a single ROI at corner (1, 1).
function box_bounds_batch(box_size, x)
    camera = SMLMData.IdealCamera(box_size, box_size, 0.1f0)
    pixel_size = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    psf_pixels = GaussMLE.to_pixel_units(GaussianXYNB(0.13f0), pixel_size)
    y = Float32((box_size + 1) / 2)
    parameters = GaussMLE.Params{4}(Float32(x), y, 5000.0f0, 10.0f0)
    data = Array{Float32}(undef, box_size, box_size, 1)
    for j in 1:box_size, i in 1:box_size
        data[i, j, 1] = GaussMLE.evaluate_psf(psf_pixels, i, j, parameters)
    end
    return SMLMData.ROIBatch(data, Int32[1], Int32[1], Int32[1], camera)
end

function box_bounds_roi_position(smld, x_corner, y_corner)
    emitter = only(smld.emitters)
    camera = smld.camera
    pixel_size_x = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    pixel_size_y = camera.pixel_edges_y[2] - camera.pixel_edges_y[1]
    # Invert src/roi_batch.jl: microns -> camera pixels -> ROI pixels.
    x_camera = (emitter.x - camera.pixel_edges_x[1]) / pixel_size_x + 0.5f0
    y_camera = (emitter.y - camera.pixel_edges_y[1]) / pixel_size_y + 0.5f0
    return x_camera - x_corner + 1, y_camera - y_corner + 1
end

@testset "Box-dependent parameter bounds" begin
    @testset "Default fit recovers x = 14 in a 15-pixel ROI" begin
        batch = box_bounds_batch(15, 14.0f0)
        config = GaussMLEConfig(psf_model = GaussianXYNB(0.13f0), backend = :cpu)

        @testset "ROIBatch input" begin
            smld, info = fit(batch, config)
            x, y = box_bounds_roi_position(smld, batch.x_corners[1], batch.y_corners[1])
            @test x ≈ 14.0f0 atol = 0.01f0 rtol = 0
            @test y ≈ 8.0f0 atol = 0.001f0 rtol = 0
            @test info.backend == :cpu
        end

        @testset "Array input" begin
            smld, info = fit(batch.data, config)
            # fit(Array) supplies corner (1, 1) for its first ROI and its own camera.
            x, y = box_bounds_roi_position(smld, 1, 1)
            @test x ≈ 14.0f0 atol = 0.01f0 rtol = 0
            @test y ≈ 8.0f0 atol = 0.001f0 rtol = 0
            @test info.backend == :cpu
        end
    end

    @testset "Symmetric bounds preserve other limits and step sizes" begin
        # Snapshot the non-position bounds and all max steps from version 0.4.0.
        cases = (
            (
                GaussianXYNB(0.13f0),
                (1.0f0, 0.01f0), (Inf32, Inf32),
                (1.0f0, 1.0f0, 100.0f0, 2.0f0),
            ),
            (
                GaussianXYNBS(),
                (1.0f0, 0.01f0, 0.3f0), (Inf32, Inf32, 5.0f0),
                (1.0f0, 1.0f0, 100.0f0, 2.0f0, 0.5f0),
            ),
            (
                GaussianXYNBSXSY(),
                (1.0f0, 0.01f0, 0.3f0, 0.3f0), (Inf32, Inf32, 5.0f0, 5.0f0),
                (1.0f0, 1.0f0, 100.0f0, 2.0f0, 0.5f0, 0.5f0),
            ),
            (
                AstigmaticXYZNB{Float32}(
                    0.13f0, 0.13f0, 0.05f0, 0.05f0, 0.3f0, 0.3f0, 0.05f0, 0.4f0
                ),
                (-1.0f0, 1.0f0, 0.01f0), (1.0f0, Inf32, Inf32),
                (1.0f0, 1.0f0, 0.1f0, 100.0f0, 2.0f0),
            ),
        )
        for (model, lower_tail, upper_tail, max_step) in cases
            @testset "$(typeof(model)), n = $n" for n in (7, 9, 11, 15)
                constraints = GaussMLE.default_constraints(model, n)
                for axis in 1:2
                    lower = constraints.lower[axis]
                    upper = constraints.upper[axis]
                    @test lower == -1.0f0
                    @test upper == Float32(n + 2)
                    @test (lower + upper) / 2 == (n + 1) / 2
                    @test 0.5f0 - lower == 1.5f0
                    @test upper - (n + 0.5f0) == 1.5f0
                end
                @test Tuple(constraints.lower[3:end]) == lower_tail
                @test Tuple(constraints.upper[3:end]) == upper_tail
                @test Tuple(constraints.max_step) == max_step
            end
        end
    end

    @testset "Default fits clamp to the actual box bounds" begin
        # Truncated tails converge slowly; leave constraints at their default value.
        config = GaussMLEConfig(
            psf_model = GaussianXYNB(0.13f0), backend = :cpu, iterations = 500
        )
        @testset "n = $n" for n in (7, 9, 11, 15)
            @testset "High edge" begin
                batch = box_bounds_batch(n, Float32(n + 3))
                smld, info = fit(batch, config)
                x, _ = box_bounds_roi_position(
                    smld, batch.x_corners[1], batch.y_corners[1]
                )
                @test x ≈ Float32(n + 2) atol = 0.001f0 rtol = 0
                @test info.backend == :cpu
            end

            @testset "Low edge" begin
                batch = box_bounds_batch(n, -2.0f0)
                smld, info = fit(batch, config)
                x, _ = box_bounds_roi_position(
                    smld, batch.x_corners[1], batch.y_corners[1]
                )
                @test x ≈ -1.0f0 atol = 0.001f0 rtol = 0
                @test info.backend == :cpu
            end
        end
    end

    @testset "Explicit constraints are preserved and enforced" begin
        model = GaussianXYNB(0.13f0)
        batch = box_bounds_batch(15, 14.0f0)
        eleven_pixel_constraints = GaussMLE.default_constraints(model, 11)
        custom_constraints = GaussMLE.ParameterConstraints{4}(
            GaussMLE.Params{4}(-1.0f0, -1.0f0, 1.0f0, 0.01f0),
            GaussMLE.Params{4}(12.0f0, 17.0f0, Inf32, Inf32),
            GaussMLE.Params{4}(1.0f0, 1.0f0, 100.0f0, 2.0f0),
        )
        @testset "$name" for (name, c) in (
                ("11-pixel constraints", eleven_pixel_constraints),
                ("Hand-built x upper bound of 12", custom_constraints),
            )
            config = GaussMLEConfig(psf_model = model, backend = :cpu, constraints = c)
            @test config.constraints === c
            smld, info = fit(batch, config)
            x, _ = box_bounds_roi_position(smld, batch.x_corners[1], batch.y_corners[1])
            @test x ≈ c.upper[1] atol = 0.001f0 rtol = 0
            @test config.constraints === c
            @test info.backend == :cpu
        end
    end

    @testset "Omitted constraints remain deferred, including keyword fit" begin
        @test GaussMLEConfig().constraints === nothing
        @test GaussMLEConfig(backend = :cpu).constraints === nothing
        batch = box_bounds_batch(15, 14.0f0)
        smld, info = fit(batch; backend = :cpu)
        x, y = box_bounds_roi_position(smld, batch.x_corners[1], batch.y_corners[1])
        @test x ≈ 14.0f0 atol = 0.01f0 rtol = 0
        @test y ≈ 8.0f0 atol = 0.001f0 rtol = 0
        @test info.backend == :cpu
    end
end
