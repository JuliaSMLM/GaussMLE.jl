# Simulated pairs for JointSplinePSFModel tests (Core joint_spline_psf.jl, Long and GPU).
# The including file has `using GaussMLE, MicroscopePSFs, SMLMData`. Expected counts follow
# MicroscopePSFs' MultiChannelPSF forward law: view v records N * transmission[v] photons of
# channel v's PSF, integrated over its camera's pixels at channel_position(mc, v, X, Y, z).
# Noise is added by the caller with SMLMSim.

# A spline of an astigmatic scalar PSF (so each view alone has one z maximum) tabulated to
# ±1.0 um laterally and ±0.6 um in z, on a 0.1 um grid (simulation and fit use the same
# spline, so a coarse grid only makes the tabulation quick)
function joint_test_spline()
    zc = ZernikeCoefficients(15)
    zc.phase[6] = 0.5
    return SplinePSF(
        ScalarPSF(1.4, 0.68, 1.518; zernike_coeffs = zc);
        lateral_range = 1.0, axial_range = 0.6, lateral_step = 0.1, axial_step = 0.1
    )
end

# Expected counts `(mu1, mu2)` (box, box, n) and corners `((x1, y1), (x2, y2))` of the two
# views of emitters at reference camera positions `truths` = [(X, Y, z), ...] in microns.
# Each view's ROI is centred on the emitter's position in that view to the nearest pixel.
function joint_expected(mc, cameras, truths; photons, bgs, box, sampling = 4)
    n = length(truths)
    mus = (zeros(Float32, box, box, n), zeros(Float32, box, box, n))
    corners = ntuple(_ -> (zeros(Int32, n), zeros(Int32, n)), 2)
    for (k, (X, Y, z)) in enumerate(truths), v in 1:2
        Xv, Yv, zv = channel_position(mc, v, X, Y, z)
        cam = cameras[v]
        ex0, ey0 = cam.pixel_edges_x[1], cam.pixel_edges_y[1]
        px = cam.pixel_edges_x[2] - ex0
        py = cam.pixel_edges_y[2] - ey0
        cx = round(Int32, (Xv - ex0) / px - box / 2) + Int32(1)
        cy = round(Int32, (Yv - ey0) / py - box / 2) + Int32(1)
        corners[v][1][k], corners[v][2][k] = cx, cy
        ex = ex0 .+ ((cx - 1):(cx - 1 + box)) .* px
        ey = ey0 .+ ((cy - 1):(cy - 1 + box)) .* py
        em = [Emitter3D(Xv, Yv, zv, photons * mc.transmission[v])]
        mus[v][:, :, k] .= bgs[v] .+ MicroscopePSFs.integrate_pixels(
            mc.channels[v], collect(ex), collect(ey), em; sampling
        )
    end
    return mus, corners
end
