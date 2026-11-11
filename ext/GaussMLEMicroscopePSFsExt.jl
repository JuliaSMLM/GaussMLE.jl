# SplinePSFModel from a MicroscopePSFs SplinePSF (e.g. a learned PSF read with load_psf).
module GaussMLEMicroscopePSFsExt

using GaussMLE: GaussMLE
using MicroscopePSFs: MicroscopePSFs

function GaussMLE._compute_psf_scale(
        psf::MicroscopePSFs.AbstractPSF, pixel_size::NTuple{2, T},
        lateral_range::T, z_ref::T
    ) where {T}
    # MicroscopePSFs supplies photon densities in µm^-2, so only the pixel area is
    # needed to convert the table's averages to fractions of the total photons.
    pixel_area = pixel_size[1] * pixel_size[2]
    photon_sum = pixel_area *
        GaussMLE._spline_reference_sum(psf, pixel_size, lateral_range, z_ref)
    (isfinite(photon_sum) && T(0.5) <= photon_sum <= T(1.05)) || throw(
        ArgumentError(
            "MicroscopePSFs PSF is not a normalized photon density in µm^-2: " *
                "pixel-integrated sum on the lateral_range patch at z = $z_ref is " *
                "$photon_sum, outside [0.5, 1.05]. Normalize the input stack as a " *
                "photon density before constructing SplinePSFModel."
        )
    )
    return pixel_area
end

# A SplinePSF is zero outside its own grid.
GaussMLE._lateral_limits(psf::MicroscopePSFs.SplinePSF) =
    (psf.x_min, psf.x_max, psf.y_min, psf.y_max)

function GaussMLE.SplinePSFModel(
        psf::MicroscopePSFs.SplinePSF;
        pixel_size::Union{Real, Tuple{Real, Real}},
        z_range = (psf.z_min, psf.z_max),
        lateral_range::Union{Nothing, Real} = nothing,
        starts::Union{Nothing, Integer} = nothing
    )
    psf.z_range === nothing &&
        throw(ArgumentError("SplinePSFModel needs a 3D SplinePSF; this one is 2D"))
    # By default the largest lateral_range the spline's own grid covers at this pixel size
    # (the table reaches one pixel beyond lateral_range); a larger one is refused.
    if lateral_range === nothing
        px = GaussMLE._spline_pixel_size(Float32, pixel_size)
        largest = GaussMLE._largest_lateral_range(GaussMLE._lateral_reach(psf), px)
        largest > 0 || throw(
            ArgumentError(
                "the SplinePSF covers x $(psf.x_min) to $(psf.x_max) and y $(psf.y_min) " *
                    "to $(psf.y_max) um, less than one $(px[1]) x $(px[2]) um pixel " *
                    "from its center; tabulate it wider"
            )
        )
        lateral_range = largest
    end
    # The fitted interval must lie within the point spline's own z range (1e-6 slack).
    # The shared constructor integrates it on the camera's pixel grid.
    (z_range[1] >= psf.z_min - 1.0e-6 && z_range[2] <= psf.z_max + 1.0e-6) || throw(
        ArgumentError(
            "z_range $z_range must lie within the SplinePSF's own (z_min, z_max) = " *
                "($(psf.z_min), $(psf.z_max))"
        )
    )
    return GaussMLE._spline_psf_model(psf, pixel_size, z_range, psf, lateral_range, starts)
end

end # module
