"""
Camera calibration helpers: raw sCMOS counts (ADU) to electrons.

Users pass SMLMData.IdealCamera or SMLMData.SCMOSCamera; the fit's noise model is chosen
from the camera type (see `_readout_variance` in unified_kernel.jl).
"""

# Preprocessing helpers for SMLMData.SCMOSCamera
"""
    to_electrons(data_adu, camera::SMLMData.SCMOSCamera)

Convert raw ADU data to electrons using camera calibration.
Applies: electrons = (ADU - offset) × gain

For scalar offset/gain, broadcasts directly.
For per-pixel calibration, requires x_corners and y_corners to index correctly.
"""
function to_electrons(data_adu::AbstractArray, camera)
    # Scalar offset/gain - broadcast directly
    if camera.offset isa Real && camera.gain isa Real
        return (data_adu .- camera.offset) .* camera.gain
    else
        error("Per-pixel sCMOS calibration requires corners. Use to_electrons(data, \
            camera, x_corners, y_corners)")
    end
end

"""
    to_electrons(data_adu, camera, x_corners, y_corners)

Convert ROI data from ADU to electrons using per-pixel camera calibration.
Extracts calibration values at correct sensor positions using corner coordinates.
"""
function to_electrons(
        data_adu::AbstractArray{T, 3}, camera, x_corners::AbstractVector,
        y_corners::AbstractVector
    ) where {T}
    # Scalar offset/gain - broadcast directly
    if camera.offset isa Real && camera.gain isa Real
        return (data_adu .- camera.offset) .* camera.gain
    end

    box_size = size(data_adu, 1)
    n_rois = size(data_adu, 3)
    data_electrons = similar(data_adu, Float32)

    for roi_idx in 1:n_rois
        x_corner = x_corners[roi_idx]
        y_corner = y_corners[roi_idx]

        for j in 1:box_size  # x within ROI
            for i in 1:box_size  # y within ROI
                # Map to camera coordinates (1-based)
                cam_y = i + y_corner - 1
                cam_x = j + x_corner - 1

                # Get calibration at this camera position
                offset_val = camera.offset isa Real ? camera.offset :
                    camera.offset[cam_y, cam_x]
                gain_val = camera.gain isa Real ? camera.gain : camera.gain[cam_y, cam_x]

                # Convert ADU → electrons
                data_electrons[i, j, roi_idx] = (data_adu[i, j, roi_idx] - offset_val) *
                    gain_val
            end
        end
    end

    return data_electrons
end
