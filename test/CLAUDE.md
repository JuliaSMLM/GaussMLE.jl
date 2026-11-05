# Testing Guidelines for test/

GaussMLE follows the lab's standard test layout (admiral decisions 0007 to 0010).

## Layout

| Group | Folder | Contents | Runs on |
|---|---|---|---|
| Core | `test/*.jl` | API smoke and CPU kernel for GaussianXYNB (a 1,000-ROI sCMOS batch), CRLB of bright emitters (N = 1e3, 1e5, 1e6) against a Float64 reference for all four Gaussian models and SplinePSFModel, SplinePSFModel fixed z starts leaving a wrong z well, sCMOS variance-map indexing regression and rejection of ROIs off the sensor, default bounds from the actual box, SplinePSFModel with IdealCamera (CPU fit, pixel integration, total photons at every z, density guard, dispatch, rectangular pixels, Array pixel size, bright emitters, objective maximum with negative pixels, truths beyond a z_range end finishing on the bound, a NaN pixel) | GitHub CI (Julia min and 1); at most 2 min |
| QA | `test/qa/` | Aqua, ExplicitImports, docstrings for every public name, docstring doctests | GitHub CI, its own job |
| GPU | `test/gpu/` | GPU kernel, CPU vs GPU agreement, performance benchmark with std/CRLB checks, vendor routing and resource-error classification for the retry loop, CUDA free-memory query fallback, NVML device under `CUDA_VISIBLE_DEVICES`, contention check, spline CPU/GPU agreement and GPU bias/CRLB per z band, spline/AstigmaticXYZNB throughput at box 13 and 15, spline `:auto` uses GPU | CUDA or AMD GPU |
| Long | `test/long/` | Monte Carlo validation: bias and std vs CRLB for the four Gaussian models with both cameras; the 10,000-ROI sCMOS batch POCL work-group regression; the CPU path of GaussianXYNBS and GaussianXYNBSXSY; SplinePSFModel with an SCMOSCamera (raw-ADU recovery, objective maximum including pixels with data + read-noise variance <= 0, truths beyond a z_range end, rectangular pixels); separate SplinePSFModel CPU validation with IdealCamera using MicroscopePSFs.integrate_pixels data, including photon bias per z band | lab machine |

Spline tests treat the callable as a point PSF. A MicroscopePSFs PSF is a normalized photon
density in µm^-2, so N is total photons and an input stack must have that normalization.
A plain callable is normalized on the reference plane: N is the photons within the
`lateral_range` patch at `z_ref`, and its energy must be the same at every z. The Core
total-photon test and Long spline tests render independently with
`MicroscopePSFs.integrate_pixels(...; sampling = 8)`.

Each kernel specialization (model, camera) costs 3 to 20 s to compile on Julia 1.11 under
`--check-bounds=yes`, which `Pkg.test` uses, and that compile is a large part of Core's time. Core
therefore fits only GaussianXYNB (both cameras) and SplinePSFModel with IdealCamera; the
other models and the sCMOS spline run in Long. Measure Core's time at the Julia floor
(1.11) with the machine's load average noted.

Camera noise in spline fixtures comes from SMLMSim (`poisson_noise`, `scmos_noise`, test
extras in the root and GPU environments), not local code (lab decision 0032). Expected
counts come from `GaussMLE.evaluate_psf` or `MicroscopePSFs.integrate_pixels`. SMLMSim's
noise draws from the global generator, so a fixture calls `Random.seed!`.
`test/long/utils/spline_psf_helpers.jl` holds the spline helpers that Core's
`spline_psf_model.jl` and Long's `spline_psf_scmos.jl` share.

- `test/runtests.jl` is the lab template, identical in every package: never edit it.
- `test/test_groups.toml` declares the groups.
- `test/qa/qa.jl` is the lab template; a check is opted out only with the reason beside it.
- Each file runs in its own module and `@testset`, so each file starts with its own `using` lines.
- Helpers a file `include`s go in a `utils/` subfolder of its group folder (not run as tests).
  A helper two groups share lives in one group's helper folder and the other includes it by
  relative path: `test/long/utils/validation_utils.jl` is also included by the GPU benchmark.
  Never put a helper in `test/` itself: every `test/*.jl` runs as a Core file.
- `test/gpu/gpu_memory_query.jl` and `test/gpu/nvml_device_index.jl` cover CUDA only.
  They check free-memory fallback, contention and NVML indices under `CUDA_VISIBLE_DEVICES`.
  Their bodies run when `CUDA.functional()`; otherwise an `@info` line names the file
  and explains that it covers the CUDA vendor only.

## Running

```bash
julia --project -e 'using Pkg; Pkg.test()'                   # Core
GROUP=QA julia --project -e 'using Pkg; Pkg.test()'          # one group; "GPU,Long" for several
GROUP=Everything julia --project -e 'using Pkg; Pkg.test()'  # every group this machine can run
```

## Rules

- **No @test_skip**: a failing test is information; fix the code or the test.
- Core stays under 2 minutes on a GitHub runner, counting first-call compile. A test that pushes
  it over moves to Long (CPU) or GPU.
- New or changed public behaviour gets tests of its documented contract; a bug fix gets one
  regression test. No tests of private helpers, no duplicates of behaviour tested elsewhere.
- Exception: GaussMLE's GPU memory, contention and poll helpers (`_gpu_free_memory`,
  `_gpu_contended`, `_poll_gpu`) are tested directly because their failure paths cannot
  be reached through `fit` on a lab machine: NVML memory info is missing only on GPUs
  such as the DGX Spark GB10, and a failed poll needs more GPU memory than exists.
  A contended GPU needs another process at high compute on the same GPU, which a test
  cannot arrange, so `_gpu_contended` injects the process and utilization queries.
- Exception: `_gpu_vendor` is tested directly because `fit` does not expose which vendor
  ran: `info.backend` is `:gpu` for both CUDA and AMD.
- Exception: `_release_gpu_context` runs only after a GPU failure mid-fit, which `fit`
  cannot reach on a lab machine, and its catch once hid a renamed CUDA function. Test
  it directly in a separate Julia process to preserve the test process's CUDA context.
- Report the test count per group before and after every change.
- Statistical tests seed their RNG or compare on a tolerance that holds for any draw.
