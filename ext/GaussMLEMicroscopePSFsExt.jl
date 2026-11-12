# SplinePSFModel from a MicroscopePSFs SplinePSF (e.g. a learned PSF read with load_psf), and
# JointSplinePSFModel from a two-channel MultiChannelPSF.
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

# A SplinePSF is zero outside its own grid. A PlanePSF evaluates the PSF it wraps at z + dz,
# so it keeps that PSF's lateral grid and shifts its z range by -dz. Other PSFs (analytic
# ones) have no grid.
GaussMLE._lateral_limits(psf::MicroscopePSFs.SplinePSF) =
    (psf.x_min, psf.x_max, psf.y_min, psf.y_max)

_z_limits(psf::MicroscopePSFs.SplinePSF) = (Float64(psf.z_min), Float64(psf.z_max))
_z_limits(::Any) = nothing

# A model of a PSF tabulated on a grid with z range zl: z_range must lie within zl, and by
# default lateral_range is the largest the grid covers at this pixel size (the table reaches
# one pixel beyond lateral_range); a larger one is refused by the shared constructor.
function _tabulated_model(psf, zl, what; pixel_size, z_range, lateral_range, starts)
    if lateral_range === nothing
        px = GaussMLE._spline_pixel_size(Float32, pixel_size)
        largest = GaussMLE._largest_lateral_range(GaussMLE._lateral_reach(psf), px)
        lim = GaussMLE._lateral_limits(psf)
        largest > 0 || throw(
            ArgumentError(
                "the $what covers x $(lim[1]) to $(lim[2]) and y $(lim[3]) to $(lim[4]) " *
                    "um, less than one $(px[1]) x $(px[2]) um pixel from its center; " *
                    "tabulate it wider"
            )
        )
        lateral_range = largest
    end
    # The fitted interval must lie within the point spline's own z range (1e-6 slack).
    # The shared constructor integrates it on the camera's pixel grid.
    (z_range[1] >= zl[1] - 1.0e-6 && z_range[2] <= zl[2] + 1.0e-6) || throw(
        ArgumentError(
            "z_range $z_range must lie within the $what's own (z_min, z_max) = " *
                "($(zl[1]), $(zl[2]))"
        )
    )
    limits = (z_min = zl[1], z_max = zl[2])
    return GaussMLE._spline_psf_model(
        psf, pixel_size, z_range, limits, lateral_range, starts
    )
end

function GaussMLE.SplinePSFModel(
        psf::MicroscopePSFs.SplinePSF;
        pixel_size::Union{Real, Tuple{Real, Real}},
        z_range = (psf.z_min, psf.z_max),
        lateral_range::Union{Nothing, Real} = nothing,
        starts::Union{Nothing, Integer} = nothing
    )
    psf.z_range === nothing &&
        throw(ArgumentError("SplinePSFModel needs a 3D SplinePSF; this one is 2D"))
    return _tabulated_model(
        psf, _z_limits(psf), "SplinePSF"; pixel_size, z_range, lateral_range, starts
    )
end

# PlanePSF and MultiChannelPSF exist from MicroscopePSFs 0.5.7; with 0.5.6 the methods
# for SplinePSF above are all there is.
@static if isdefined(MicroscopePSFs, :PlanePSF)
    GaussMLE._lateral_limits(p::MicroscopePSFs.PlanePSF) = GaussMLE._lateral_limits(p.psf)

    function _z_limits(p::MicroscopePSFs.PlanePSF)
        inner = _z_limits(p.psf)
        inner === nothing && return nothing
        return (inner[1] - Float64(p.dz), inner[2] - Float64(p.dz))
    end

    # A PlanePSF of a SplinePSF keeps the spline's grid, with z shifted by -dz; a
    # PlanePSF of an analytic PSF is fitted as any callable (z_range required,
    # lateral_range 1.5 by default).
    function GaussMLE.SplinePSFModel(
            psf::MicroscopePSFs.PlanePSF;
            pixel_size::Union{Real, Tuple{Real, Real}},
            z_range = _z_limits(psf),
            lateral_range::Union{Nothing, Real} = nothing,
            starts::Union{Nothing, Integer} = nothing
        )
        zl = _z_limits(psf)
        if zl === nothing
            z_range === nothing && throw(
                ArgumentError("z_range is required for a PlanePSF of a PSF with no z range")
            )
            limits = (z_min = Float64(z_range[1]), z_max = Float64(z_range[2]))
            return GaussMLE._spline_psf_model(
                psf, pixel_size, z_range, limits, something(lateral_range, 1.5), starts
            )
        end
        return _tabulated_model(
            psf, zl, "PlanePSF's SplinePSF (z shifted by -dz)";
            pixel_size, z_range, lateral_range, starts
        )
    end
end

@static if isdefined(MicroscopePSFs, :MultiChannelPSF)
    # One SplinePSFModel fits one PSF in one ROI; JointSplinePSFModel fits two channels.
    function GaussMLE.SplinePSFModel(::MicroscopePSFs.MultiChannelPSF; kwargs...)
        throw(
            ArgumentError(
                "SplinePSFModel fits one channel: build one model per channel " *
                    "(SplinePSFModel(mc.channels[k]; ...)) and fit each channel's ROIs, " *
                    "or fit two channels at once with JointSplinePSFModel(mc; ...)"
            )
        )
    end

    # Two channels, the reference first. Each view is a SplinePSFModel of its channel in
    # the channel's own z (reference z plus its z_offset), with the SplinePSF checks; the
    # joint z range is in reference z, by default all that both channels cover.
    function GaussMLE.JointSplinePSFModel(
            mc::MicroscopePSFs.MultiChannelPSF;
            pixel_size::Union{Real, Tuple{Real, Real}},
            z_range = nothing,
            lateral_range::Union{Nothing, Real} = nothing,
            starts::Union{Nothing, Integer} = nothing
        )
        n = length(mc.channels)
        n == 2 || throw(
            ArgumentError("JointSplinePSFModel fits two channels; this MultiChannelPSF has $n")
        )
        order = mc.reference == 1 ? (1, 2) : (2, 1)
        channels = (mc.channels[order[1]], mc.channels[order[2]])
        limits = (_z_limits(channels[1]), _z_limits(channels[2]))
        for v in 1:2
            limits[v] === nothing && throw(
                ArgumentError(
                    "channel $(order[v]) is a $(typeof(channels[v]).name.name); each " *
                        "channel must be a 3D SplinePSF (or a PlanePSF of one)"
                )
            )
        end
        offset = (Float64(mc.z_offset[order[1]]), Float64(mc.z_offset[order[2]]))
        cover = (
            max(limits[1][1] - offset[1], limits[2][1] - offset[2]),
            min(limits[1][2] - offset[1], limits[2][2] - offset[2]),
        )
        zr = z_range === nothing ? cover : z_range
        (length(zr) == 2 && isfinite(zr[1]) && isfinite(zr[2]) && zr[1] < zr[2]) || throw(
            ArgumentError(
                "z_range $zr must be finite and increasing; with these z offsets the " *
                    "channels both cover $cover"
            )
        )
        views = ntuple(
            v -> _tabulated_model(
                channels[v], limits[v],
                "channel $(order[v])'s SplinePSF (in its own z, reference z plus " *
                    "z_offset $(offset[v]))";
                pixel_size, z_range = (zr[1] + offset[v], zr[2] + offset[v]),
                lateral_range, starts
            ),
            2
        )
        # The views' Float32 z ranges in reference z; each view clamps z to its own.
        T = Float32
        zj = (
            max(views[1].z_range[1] - T(offset[1]), views[2].z_range[1] - T(offset[2])),
            min(views[1].z_range[2] - T(offset[1]), views[2].z_range[2] - T(offset[2])),
        )
        return GaussMLE._joint_spline_model(
            views, (mc.transmission[order[1]], mc.transmission[order[2]]), offset,
            mc.registration[order[2]], zj, GaussMLE._spline_starts(starts, zj)
        )
    end
end

end # module
