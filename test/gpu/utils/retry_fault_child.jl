# Child process for retry_faults.jl: one fit under an injected fault, one result line on
# stdout. Run in its own process so the injected method never touches the group's process.
# ARGS: scenario ("launch_failed" | "oom"), backend (:auto or :gpu or :cpu).
using GaussMLE, CUDA, SMLMData

scenario = ARGS[1]
backend = Symbol(ARGS[2])

if scenario == "launch_failed"
    # A real CuError with code 719 (launch failed) inside the GPU attempt; the cleanup
    # and later calls stay real.
    function GaussMLE._ka_gpu_backend(::GaussMLE.CUDAGPU)
        throw(CUDA.CuError(CUDA.ERROR_LAUNCH_FAILED))
    end
end

n_rois = scenario == "oom" ? 20_000 : 200
roi_size = 15
psf_model = GaussianXYNB(0.13f0)
batch = GaussMLE.generate_roi_batch(
    IdealCamera(64, 64, 0.1), psf_model; n_rois, roi_size, seed = 1
)
data = convert(Array{Float32, 3}, batch.data)
fitter = GaussMLEConfig(;
    psf_model, backend, batch_size = n_rois, auto_timeout = 20.0, gpu_timeout = 20.0
)
t = @elapsed result = try
    smld, info = fit(data, fitter)
    (; backend = info.backend, n_batches = info.n_batches, n = length(smld.emitters))
catch e
    (; error = string(typeof(e)), code = e isa CUDA.CuError ? Int(e.code) : 0)
end
println("RESULT ", result, " seconds=", round(t; digits = 2))
