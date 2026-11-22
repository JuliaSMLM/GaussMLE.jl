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

# The pixel edges (um) of `count` pixels of size `px` whose first is pixel `first` of a
# sensor with its corner at 0: `edges(65, 64, 0.12)` starts at 7.68 um. Cameras that are
# crops are built from these, `IdealCamera(edges_x, edges_y)`: a camera built from a pixel
# range (`IdealCamera(65:128, ...)`) starts at 0 whatever the range's first pixel.
joint_edges(first, count, px) = collect(((first - 1):(first - 1 + count)) .* px)

# Expected counts `(mu1, mu2)` (box, box, n) and corners `((x1, y1), (x2, y2))` of the two
# views of emitters at reference camera positions `truths` = [(X, Y, z), ...] in microns.
# Each view's ROI is centred on the emitter's position in that view to the nearest pixel,
# and its pixel edges are the camera's own, so a ROI off its camera is a BoundsError.
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
        ex = Float64.(cam.pixel_edges_x[cx:(cx + box)])
        ey = Float64.(cam.pixel_edges_y[cy:(cy + box)])
        em = [Emitter3D(Xv, Yv, zv, photons * mc.transmission[v])]
        mus[v][:, :, k] .= bgs[v] .+ MicroscopePSFs.integrate_pixels(
            mc.channels[v], ex, ey, em; sampling
        )
    end
    return mus, corners
end
