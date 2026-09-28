# SplinePSFModel from a MicroscopePSFs SplinePSF (e.g. a learned PSF read with load_psf).
module GaussMLEMicroscopePSFsExt

using GaussMLE: GaussMLE
using MicroscopePSFs: MicroscopePSFs

function GaussMLE.SplinePSFModel(
        psf::MicroscopePSFs.SplinePSF;
        pixel_size::Real,
        z_range = (psf.z_min, psf.z_max),
        lateral_range::Real = 1.5
    )
    psf.z_range === nothing &&
        throw(ArgumentError("SplinePSFModel needs a 3D SplinePSF; this one is 2D"))
    # Re-sampling an already-tabulated grid: z_range must lie within the spline's own range
    # (1e-6 slack), so GaussMLE samples exactly the coordinate the spline was exported on
    (z_range[1] >= psf.z_min - 1.0e-6 && z_range[2] <= psf.z_max + 1.0e-6) || throw(
        ArgumentError(
            "z_range $z_range must lie within the SplinePSF's own (z_min, z_max) = " *
                "($(psf.z_min), $(psf.z_max))"
        )
    )
    return GaussMLE._spline_psf_model(psf, pixel_size, z_range, psf, lateral_range)
end

end # module
