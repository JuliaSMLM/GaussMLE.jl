# Compile the first CPU fit of every model and camera at package precompile time, so a fresh
# process does not pay for it. Everything runs on `backend = :cpu` and never creates a GPU
# context.

# A plain function standing in for a tabulated PSF: an astigmatic Gaussian.
_precompile_psf(x, y, z) = exp(-x^2 / (2 * (0.13 + 0.1z)^2) - y^2 / (2 * (0.13 - 0.1z)^2))

# Whether the CPU backend of the installed KernelAbstractions can launch a kernel while
# precompiling. Version 0.9 runs kernels on Julia threads. Version 0.10 compiles them
# through POCL, which is absent on some platforms.
function _cpu_backend_precompilable()
    Sys.iswindows() && v"1.11-" <= VERSION < v"1.12-" && return false
    isdefined(KernelAbstractions, :POCL) || return true
    pocl = getfield(KernelAbstractions, :POCL)
    return pocl.nanoOpenCL.pocl_standalone_jll.is_available() &&
        pocl.SPIRV_LLVM_Backend_jll.is_available() && pocl.SPIRV_Tools_jll.is_available()
end

# Four 7x7 ROIs on `camera` at expected counts of `model` with parameters θ0 (x moved by
# one pixel per ROI), in ADU for an sCMOS camera.
function _precompile_batch(camera, model, θ0::SVector{N, Float32}) where {N}
    pixel_model = _model_for_camera(model, camera)
    data = Array{Float32}(undef, 7, 7, 4)
    for k in 1:4, j in 1:7, i in 1:7
        θ = Base.setindex(θ0, θ0[1] + 0.25f0 * (k - 2), 1)
        mu = evaluate_psf(pixel_model, i, j, θ)
        data[i, j, k] = camera isa SMLMData.SCMOSCamera ? mu / 0.5f0 + 100.0f0 : mu
    end
    corners = Int32[1, 8, 15, 22]
    return ROIBatch(data, corners, ones(Int32, 4), Int32[1, 2, 3, 4], camera)
end

@setup_workload begin
    ideal = IdealCamera(32, 32, 0.1f0)
    scmos = SCMOSCamera(
        32, 32, 0.1f0, fill(2.0f0, 32, 32); offset = 100.0f0, gain = 0.5f0, qe = 1.0f0
    )
    models = (
        (GaussianXYNB(0.13f0), SVector{4, Float32}(4, 4, 500, 5)),
        (GaussianXYNBS(), SVector{5, Float32}(4, 4, 500, 5, 1.3)),
        (GaussianXYNBSXSY(), SVector{6, Float32}(4, 4, 500, 5, 1.3, 1.3)),
        (
            AstigmaticXYZNB{Float32}(0.13, 0.13, 0.05, -0.05, 0.01, -0.01, 0.2, 0.5),
            SVector{5, Float32}(4, 4, 0.1, 500, 5),
        ),
    )
    if _cpu_backend_precompilable()
        try
            # PoCL's kernel cache stays out of the user's cache directory, and it starts one
            # thread rather than one per core.
            mktempdir() do cache_dir
                env = (
                    "POCL_CACHE_DIR" => cache_dir,
                    "POCL_CPU_MAX_CU_COUNT" => "1", "POCL_MAX_PTHREAD_COUNT" => "1",
                )
                withenv(env...) do
                    @compile_workload begin
                        for (model, θ0) in models, camera in (ideal, scmos)
                            fit(
                                _precompile_batch(camera, model, θ0);
                                psf_model = model, backend = :cpu, iterations = 3
                            )
                        end
                        spline = SplinePSFModel(
                            _precompile_psf; pixel_size = 0.1, z_range = (-0.3, 0.3)
                        )
                        θ0 = SVector{5, Float32}(4, 4, 0.1, 500, 5)
                        for camera in (ideal, scmos)
                            fit(
                                _precompile_batch(camera, spline, θ0);
                                psf_model = spline, backend = :cpu, iterations = 3
                            )
                        end
                    end
                end
            end
        catch err
            # A broken CPU backend must not keep the package from loading.
            @debug "Failed to fit during precompilation" exception = (
                err, catch_backtrace(),
            )
        end
    end
end
