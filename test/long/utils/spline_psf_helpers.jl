# Helpers shared by test/spline_psf_model.jl (Core) and test/long/spline_psf_scmos.jl: the
# including file has `using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, Random`.
using SMLMSim: poisson_noise, scmos_noise

# Share one Float32-only callable type to cover dispatch and limit Core compilation.
struct SplineTestPSF
    linear::Bool
    x_shift::Float64
    y_shift::Float64
    z_min::Float64
    z_max::Float64
    basins::Bool
end
SplineTestPSF(linear, x_shift, y_shift, z_min, z_max) =
    SplineTestPSF(linear, x_shift, y_shift, z_min, z_max, false)

function (psf::SplineTestPSF)(x::Float32, y::Float32, z::Float32)
    psf.z_min <= z <= psf.z_max || error("evaluated outside the callable's valid z range")
    dx, dy = x - psf.x_shift * z, y - psf.y_shift * z
    if psf.basins
        # Isotropic, with the same energy at every z. The width is 0.2 um except in two
        # wells: 0.12 um at z = +0.35 and a shallower 0.16 um at z = -0.35, so the
        # likelihood in z has a maximum in each, and the lower one is a local maximum.
        w = 0.2 - 0.08 * exp(-((z - 0.35) / 0.1)^2) - 0.04 * exp(-((z + 0.35) / 0.1)^2)
        return exp(-(dx^2 + dy^2) / (2 * w^2)) / (2 * pi * w^2)
    end
    if psf.linear
        # An identifiable change that is linear in z and positive over the full grid.
        return exp(-(dx^2 + dy^2) / (2 * 0.13^2)) *
            (1 + z * tanh((dx^2 - dy^2) / 0.02))
    end
    return exp(-dx^2 / (2 * (0.13 + 0.1z)^2) - dy^2 / (2 * (0.13 - 0.1z)^2))
end

const astig_shape = SplineTestPSF(false, 0.0, 0.0, -Inf, Inf)
const linear_shape = SplineTestPSF(true, 0.0, 0.0, -Inf, Inf)
const two_basin_shape = SplineTestPSF(false, 0.0, 0.0, -Inf, Inf, true)

function roi_at(m, z; x = 6, y = 6, box = 11, photons = 1000)
    return [
        GaussMLE.evaluate_psf(m, i, j, SVector{5, Float32}(x, y, z, photons, 10))
            for i in 1:box, j in 1:box
    ]
end

function batch_at(m, z; x = 6, y = 6, box = 11, photons = 1000)
    roi = roi_at(m, z; x, y, box, photons)
    data = reshape(roi, size(roi)..., 1)
    return ROIBatch(data, Int32[1], Int32[1], Int32[1], IdealCamera(64, 64, m.pixel_size))
end

# Expected counts mu (box, box, n) of ROIs at camera corners to raw sCMOS data: SMLMSim's
# scmos_noise on each ROI's own slice of the sensor calibration.
function scmos_rois(mu, readnoise, x_corners, y_corners, px; offset = 0.0f0, gain = 1.0f0)
    box = size(mu, 1)
    data = similar(mu, Float32)
    for k in axes(mu, 3)
        rows = y_corners[k]:(y_corners[k] + box - 1)
        columns = x_corners[k]:(x_corners[k] + box - 1)
        roi_camera = SCMOSCamera(
            box, box, Float32(px), readnoise[rows, columns]; offset, gain, qe = 1.0f0
        )
        data[:, :, k] .= scmos_noise(mu[:, :, k], roi_camera)
    end
    return data
end

# Each fitted ROI ends at a maximum of the objective the fit climbs, (d + v) log(mu + v) -
# (mu + v) with v(k, i, j) the read-noise variance (zero: Poisson). The check is a
# coordinatewise perturbation; data is (box, box, n) and corners are camera positions.
function check_objective_maximum(smld, model, data, v, x_corners, y_corners, px)
    box = size(data, 1)
    function objective(k, θ)
        s = 0.0
        for j in 1:box, i in 1:box
            μ = Float64(GaussMLE.evaluate_psf(model, i, j, SVector{5, Float32}(θ)))
            vk = Float64(v(k, i, j))
            s += (data[i, j, k] + vk) * log(μ + vk) - (μ + vk)
        end
        return s
    end
    for (k, e) in enumerate(smld.emitters)
        θ = [
            e.x / px + 1.5 - x_corners[k], e.y / px + 1.5 - y_corners[k], e.z,
            e.photons, e.bg,
        ]
        scale = [1.0e-3, 1.0e-3, 1.0e-4, 1.0e-3 * θ[4], 1.0e-3 * θ[5]]
        base = objective(k, θ)
        for p in 1:5, sgn in (-1, 1)
            moved = copy(θ)
            moved[p] += sgn * scale[p]
            @test objective(k, moved) - base <= 1.0e-3
        end
    end
    return nothing
end

# ROIs whose truth lies beyond the ends of a narrower model's z_range, fitted with that
# model (IdealCamera or SCMOSCamera): z ends on the bound and the other parameters at the
# bounded optimum. `source_model` renders the truth over the spline's full range; the
# fitted model stops at +-0.4 um, so the objective's gradient in z points out of the box.
# With every parameter free, the coupled photon and background steps assumed z moved, and
# the fit stalled short of the bounded optimum.
function check_fits_beyond_z_range(spline, source_model, camera_kind)
    px, box, sensor_size = 0.1, 11, 40
    m = SplinePSFModel(spline; pixel_size = px, z_range = (-0.4, 0.4))
    truth = (
        (6.2, 5.8, -0.5), (5.7, 6.4, -0.45), (6.0, 6.0, -0.5), (6.4, 5.6, -0.45),
        (6.2, 5.8, 0.5), (5.7, 6.4, 0.45), (6.0, 6.0, 0.5), (6.4, 5.6, 0.45),
    )
    n = length(truth)
    x_corners = Int32[3, 25, 6, 28, 4, 24, 7, 27]
    y_corners = Int32[5, 8, 24, 27, 6, 9, 23, 26]
    mu = zeros(Float32, box, box, n)
    for (k, (x, y, z)) in enumerate(truth), j in 1:box, i in 1:box
        mu[i, j, k] = GaussMLE.evaluate_psf(
            source_model, i, j, SVector{5, Float32}(x, y, z, 2000, 10)
        )
    end
    readnoise = [
        sqrt(1.0f0 + 0.02f0 * (i - 1) + 0.05f0 * (j - 1))
            for i in 1:sensor_size, j in 1:sensor_size
    ]
    Random.seed!(17)
    if camera_kind == :ideal
        data = Float32.(poisson_noise(mu))
        camera = IdealCamera(sensor_size, sensor_size, px)
        v = (k, i, j) -> 0
    else
        data = scmos_rois(mu, readnoise, x_corners, y_corners, px)
        camera = SCMOSCamera(
            sensor_size, sensor_size, Float32(px), readnoise;
            offset = 0.0f0, gain = 1.0f0, qe = 1.0f0
        )
        v = (k, i, j) -> readnoise[y_corners[k] + i - 1, x_corners[k] + j - 1]^2
    end
    roi_batch = ROIBatch(data, x_corners, y_corners, Int32.(1:n), camera)
    smld, _ = fit(roi_batch; psf_model = m, backend = :cpu)
    for (e, (_, _, z)) in zip(smld.emitters, truth)
        @test e.z == (z < 0 ? m.z_range[1] : m.z_range[2])
    end
    check_objective_maximum(smld, m, data, v, x_corners, y_corners, px)
    return nothing
end
