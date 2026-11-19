# The Gaussian fits end at the maximum of the likelihood they climb. Every fit is compared
# with an independent Float64 maximizer of the same objective on simulated ROIs; Core's
# fisher_crlb.jl compares a fit only with the same Levenberg-Marquardt code run for 500
# passes, which would pass a fault the two share.
#
# The reference is written here, not taken from GaussMLE: Float64 pixel-integrated expected
# counts (error-function differences; the astigmatic widths from the documented formula), the
# analytic gradient of the objective sum_pixels (d + v) log(mu + v) - (mu + v) (v = 0 for an
# IdealCamera, the read-noise variance for an SCMOSCamera), and Newton steps on its observed
# Hessian (central differences of the gradient) with damping, backtracking and GaussMLE's
# default bounds, to a Newton decrement below 1e-12. It starts from the fit's end point and
# from the truth; the higher objective is the maximum. A fit's distance is the largest
# |fit_k - max_k| / CRLB_k over its parameters, with the CRLB from the Float64 Fisher matrix
# at the maximum. A fit is scored when its simulated emitter has 500 photons or more and a
# position CRLB at the truth (the Float64 Fisher matrix) of at most 0.2 px: the selection
# uses the truth alone, so it cannot depend on the fit under test. Every condition must
# score at least min_scored of its fits, and 99.9 % of the scored ones must lie within
# 0.1 CRLB of the maximum.
using Test, GaussMLE, SMLMData, SMLMSim, Random, LinearAlgebra, SpecialFunctions
using StaticArrays, Printf

const BOX = 13
const PIXEL = 0.1
const SENSOR = 64
const N_ROIS = 2000
const CENTER = (BOX + 1) / 2
const MAX_DISTANCE = 0.1
const MIN_SHARE = 0.999
# Share of a condition's fits that must be scored. The selection uses the truth alone, so
# the count is fixed by the simulation: every fit at 2,000 photons and above, but at 500
# photons and background 10 the defocused astigmatic emitters (13 to 18 % of them) have a
# position CRLB above 0.2 px.
min_scored(photons) = photons >= 2000 ? 0.99 : 0.8

# ---- Float64 reference model ----------------------------------------------------------------

# kind is :xynb, :xynbs, :xynbsxsy or :astig; c holds the model's fixed calibration in pixel
# units: the width (:xynb) or σx₀, σy₀, Ax, Ay, Bx, By, γ, d (:astig, γ and d in microns).
struct RefModel
    kind::Symbol
    c::Vector{Float64}
end

# Parameters are x, y, N, bg [, σ | σx, σy] in ROI pixels, or x, y, z, N, bg with z in microns
indices(m::RefModel) = m.kind === :astig ? (x = 1, y = 2, N = 4, bg = 5) :
    (x = 1, y = 2, N = 3, bg = 4)

# σx, σy and their derivatives with respect to each parameter
function widths(m::RefModel, θ)
    n = length(θ)
    dsx, dsy = zeros(n), zeros(n)
    if m.kind === :xynb
        σx = σy = m.c[1]
    elseif m.kind === :xynbs
        σx = σy = θ[5]
        dsx[5] = dsy[5] = 1
    elseif m.kind === :xynbsxsy
        σx, σy = θ[5], θ[6]
        dsx[5] = 1
        dsy[6] = 1
    else
        σx0, σy0, Ax, Ay, Bx, By, γ, d = m.c
        z = θ[3]
        ux, uy = (z - γ) / d, (z + γ) / d
        αx = 1 + ux^2 + Ax * ux^3 + Bx * ux^4
        αy = 1 + uy^2 + Ay * uy^3 + By * uy^4
        σx, σy = σx0 * sqrt(αx), σy0 * sqrt(αy)
        dsx[3] = σx0 / (2 * sqrt(αx)) * (2ux + 3Ax * ux^2 + 4Bx * ux^3) / d
        dsy[3] = σy0 / (2 * sqrt(αy)) * (2uy + 3Ay * uy^2 + 4By * uy^3) / d
    end
    return σx, σy, dsx, dsy
end

stdnormal(t) = exp(-t^2 / 2) / sqrt(2π)

# Pixel integral of a 1D Gaussian centered at c over [i - 1/2, i + 1/2], and its derivatives
# with respect to c and σ
function pixel1d(i, c, σ)
    a, b = (i - 0.5 - c) / σ, (i + 0.5 - c) / σ
    p = (erf(b / sqrt(2)) - erf(a / sqrt(2))) / 2
    dc = (stdnormal(a) - stdnormal(b)) / σ
    dσ = (a * stdnormal(a) - b * stdnormal(b)) / σ
    return p, dc, dσ
end

# Expected counts mu[i, j] (i = row = y, j = column = x) and dmu[k, i, j]
function expected(m::RefModel, θ)
    n = length(θ)
    ix = indices(m)
    σx, σy, dsx, dsy = widths(m, θ)
    N, bg = θ[ix.N], θ[ix.bg]
    mu = zeros(BOX, BOX)
    dmu = zeros(n, BOX, BOX)
    for j in 1:BOX, i in 1:BOX
        px, dpx, dsigx = pixel1d(j, θ[ix.x], σx)
        py, dpy, dsigy = pixel1d(i, θ[ix.y], σy)
        mu[i, j] = bg + N * px * py
        dmu[ix.x, i, j] = N * dpx * py
        dmu[ix.y, i, j] = N * px * dpy
        dmu[ix.N, i, j] = px * py
        dmu[ix.bg, i, j] = 1
        for k in 1:n
            (dsx[k] == 0 && dsy[k] == 0) && continue
            dmu[k, i, j] += N * (dsigx * dsx[k] * py + px * dsigy * dsy[k])
        end
    end
    return mu, dmu
end

# Objective (up to a constant in θ) and gradient. The objective is written as the deviance
# s log(1 + r / s) - r, s = d + v, r = mu - d, so that its rounding is small.
function objgrad(m::RefModel, θ, data, v)
    mu, dmu = expected(m, θ)
    L = 0.0
    g = zeros(length(θ))
    for j in 1:BOX, i in 1:BOX
        d, vij, μ = data[i, j], v[i, j], mu[i, j]
        var = μ + vij
        var > 0 || return -Inf, fill(NaN, length(θ))
        s = d + vij
        L += s > 0 ? s * log1p((μ - d) / s) - (μ - d) : s * log(var) - var
        w = (d - μ) / var
        for k in eachindex(g)
            g[k] += dmu[k, i, j] * w
        end
    end
    return L, g
end

# Observed Hessian of the objective, central differences of the analytic gradient
function hessian(m::RefModel, θ, data, v)
    n = length(θ)
    H = zeros(n, n)
    for k in 1:n
        h = 1.0e-6 * max(abs(θ[k]), m.kind === :astig && k == 3 ? 0.1 : 1.0)
        θp, θm = copy(θ), copy(θ)
        θp[k] += h
        θm[k] -= h
        H[:, k] = (objgrad(m, θp, data, v)[2] - objgrad(m, θm, data, v)[2]) / (2h)
    end
    return (H + H') / 2
end

function fisher(m::RefModel, θ, v)
    mu, dmu = expected(m, θ)
    n = length(θ)
    F = zeros(n, n)
    for j in 1:BOX, i in 1:BOX
        dm = @view dmu[:, i, j]
        F .+= (dm * dm') ./ (mu[i, j] + v[i, j])
    end
    return F
end

# Projected, damped Newton ascent from θ0 within [lo, hi]. A parameter on a bound whose
# gradient points out of the box is held. Returns (θ, objective, converged): converged when
# the Newton decrement g' (-H)⁻¹ g over the free parameters is below 1e-12.
function maximize(m::RefModel, θ0, data, v, lo, hi; maxiter = 300)
    n = length(θ0)
    θ = clamp.(θ0, lo, hi)
    L, g = objgrad(m, θ, data, v)
    isfinite(L) || return θ, L, false
    for _ in 1:maxiter
        free = [!((θ[k] <= lo[k] && g[k] < 0) || (θ[k] >= hi[k] && g[k] > 0)) for k in 1:n]
        A = -hessian(m, θ, data, v)[free, free]
        gf = g[free]
        # Damp until A is positive definite (a Levenberg term, relative to the diagonal)
        δf = zeros(length(gf))
        dec = Inf
        μ = 0.0
        while true
            Ad = A + μ * Diagonal(max.(abs.(diag(A)), 1.0e-12))
            ch = cholesky(Symmetric(Ad); check = false)
            if issuccess(ch)
                δf = ch \ gf
                dec = dot(gf, δf)
                break
            end
            μ = iszero(μ) ? 1.0e-6 : 10μ
            μ > 1.0e12 && return θ, L, false
        end
        dec < 1.0e-12 && return θ, L, true
        δ = zeros(n)
        δ[free] = δf
        # Backtracking. Near the maximum a step whose gain is below the objective's rounding
        # is taken, since it can neither be confirmed nor refused.
        tol = 64 * eps() * (abs(L) + 1)
        t = 1.0
        moved = false
        while t > 1.0e-14
            θn = clamp.(θ + t * δ, lo, hi)
            Ln, gn = objgrad(m, θn, data, v)
            if isfinite(Ln) && (Ln > L || (dec < 1.0e-3 && Ln >= L - tol))
                θ, L, g = θn, Ln, gn
                moved = true
                break
            end
            t /= 2
        end
        moved || return θ, L, false
    end
    return θ, L, false
end

# ---- GaussMLE conditions --------------------------------------------------------------------

struct Spec
    name::String
    model::Any            # the GaussMLE model, in microns
    kind::Symbol
    draw::Function        # (rng, photons, bg) -> truth parameter vector
end

typical_astigmatic() = AstigmaticXYZNB{Float32}(0.13, 0.13, 0.05, 0.05, 0.3, 0.3, 0.05, 0.4)

specs = [
    Spec(
        "GaussianXYNB", GaussianXYNB(0.13f0), :xynb,
        (rng, N, bg) -> [CENTER + rand(rng) - 0.5, CENTER + rand(rng) - 0.5, N, bg]
    ),
    Spec(
        "GaussianXYNBS", GaussianXYNBS(), :xynbs,
        (rng, N, bg) -> [
            CENTER + rand(rng) - 0.5, CENTER + rand(rng) - 0.5, N, bg,
            1.1 + 0.4 * rand(rng),
        ]
    ),
    Spec(
        "GaussianXYNBSXSY", GaussianXYNBSXSY(), :xynbsxsy,
        (rng, N, bg) -> [
            CENTER + rand(rng) - 0.5, CENTER + rand(rng) - 0.5, N, bg,
            1.1 + 0.4 * rand(rng), 1.1 + 0.4 * rand(rng),
        ]
    ),
    Spec(
        "AstigmaticXYZNB", typical_astigmatic(), :astig,
        (rng, N, bg) -> [
            CENTER + rand(rng) - 0.5, CENTER + rand(rng) - 0.5, rand(rng) - 0.5, N, bg,
        ]
    ),
]

# The reference model for a GaussMLE model on a camera with this pixel size
function refmodel(spec::Spec, pixel_size)
    pm = GaussMLE.to_pixel_units(spec.model, pixel_size)
    c = if spec.kind === :xynb
        [Float64(pm.σ)]
    elseif spec.kind === :astig
        Float64[pm.σx₀, pm.σy₀, pm.Ax, pm.Ay, pm.Bx, pm.By, pm.γ, pm.d]
    else
        Float64[]
    end
    return RefModel(spec.kind, c), pm
end

# Fitted parameters in the reference's order and units (ROI pixels; z in microns), from an
# emitter and the ROI's corner and the camera's pixel geometry
function fitted_parameters(e, kind, camera, corner_x, corner_y)
    px = camera.pixel_edges_x[2] - camera.pixel_edges_x[1]
    py = camera.pixel_edges_y[2] - camera.pixel_edges_y[1]
    x = (e.x - camera.pixel_edges_x[1]) / px + 1.5 - corner_x
    y = (e.y - camera.pixel_edges_y[1]) / py + 1.5 - corner_y
    kind === :xynb && return Float64[x, y, e.photons, e.bg]
    kind === :xynbs && return Float64[x, y, e.photons, e.bg, e.σ / px]
    kind === :xynbsxsy && return Float64[x, y, e.photons, e.bg, e.σx / px, e.σy / py]
    return Float64[x, y, e.z, e.photons, e.bg]
end

# Whether a fit is scored, decided on the simulated truth alone: 500 photons or more and a
# position CRLB at the truth of at most 0.2 px
function scored(truth, kind, crlb_truth)
    photons = truth[kind === :astig ? 4 : 3]
    return photons >= 500 && crlb_truth[1] <= 0.2 && crlb_truth[2] <= 0.2
end

struct Result
    distance::Float64
    scored::Bool
    from_truth::Bool      # the truth start reached a higher objective than the fit's end point did
    reference_ok::Bool    # a reference run met the Newton decrement test
    gap::Float64          # objective of the maximum minus that at the fit's end point
    θfit::Vector{Float64}
    θmax::Vector{Float64}
    crlb::Vector{Float64}
end

# Simulate n ROIs of one condition, fit them through the public API and compare each fit with
# the reference maximum
function run_condition(spec::Spec, camera_kind::Symbol, photons, bg, seed)
    rng = Xoshiro(seed)
    Random.seed!(seed)  # SMLMSim's noise draws from the global generator
    readnoise = 1.0 .+ rand(rng, SENSOR, SENSOR)  # 1 to 2 e-, about 1.5
    camera = camera_kind === :ideal ? IdealCamera(SENSOR, SENSOR, PIXEL) :
        SCMOSCamera(SENSOR, SENSOR, PIXEL, readnoise; offset = 0.0, gain = 1.0)
    ref, pm = refmodel(spec, PIXEL)
    n = N_ROIS
    npar = length(spec.draw(rng, photons, bg))
    truth = zeros(npar, n)
    x_corners = Int32[rand(rng, 1:(SENSOR - BOX + 1)) for _ in 1:n]
    y_corners = Int32[rand(rng, 1:(SENSOR - BOX + 1)) for _ in 1:n]
    mu = zeros(Float32, BOX, BOX, n)
    for k in 1:n
        truth[:, k] = spec.draw(rng, photons, bg)
        θ32 = SVector{npar, Float32}(truth[:, k])
        for j in 1:BOX, i in 1:BOX
            mu[i, j, k] = GaussMLE.evaluate_psf(pm, i, j, θ32)
        end
    end
    data = zeros(Float32, BOX, BOX, n)
    v = zeros(BOX, BOX, n)  # read-noise variance, e-^2
    if camera_kind === :ideal
        data .= Float32.(poisson_noise(mu))
    else
        for k in 1:n
            rows = y_corners[k]:(y_corners[k] + BOX - 1)
            columns = x_corners[k]:(x_corners[k] + BOX - 1)
            roi_camera = SCMOSCamera(
                BOX, BOX, PIXEL, readnoise[rows, columns]; offset = 0.0, gain = 1.0
            )
            data[:, :, k] .= scmos_noise(mu[:, :, k], roi_camera)
            v[:, :, k] .= readnoise[rows, columns] .^ 2
        end
    end

    batch = ROIBatch(data, x_corners, y_corners, Int32.(1:n), camera)
    smld, _ = fit(batch, GaussMLEConfig(psf_model = spec.model, backend = :cpu))

    cons = GaussMLE.default_constraints(pm, BOX)
    lo, hi = Float64.(cons.lower), Float64.(cons.upper)
    results = Vector{Result}(undef, n)
    Threads.@threads for k in 1:n
        e = smld.emitters[k]
        roi = Float64.(data[:, :, k])
        vk = v[:, :, k]
        θfit = fitted_parameters(e, spec.kind, camera, x_corners[k], y_corners[k])
        θa, La, oka = maximize(ref, θfit, roi, vk, lo, hi)
        θb, Lb, okb = maximize(ref, truth[:, k], roi, vk, lo, hi)
        pick_b = (okb && !oka) || (oka == okb && Lb > La)
        θmax, Lmax = pick_b ? (θb, Lb) : (θa, La)
        CRLB = sqrt.(diag(inv(Symmetric(fisher(ref, θmax, vk)))))
        distance = maximum(abs.(clamp.(θfit, lo, hi) .- θmax) ./ CRLB)
        crlb_truth = sqrt.(diag(inv(Symmetric(fisher(ref, truth[:, k], vk)))))
        results[k] = Result(
            isnan(distance) ? Inf : distance, scored(truth[:, k], spec.kind, crlb_truth),
            okb && Lb - La > 1.0e-6,
            oka || okb,
            Lmax - objgrad(ref, clamp.(θfit, lo, hi), roi, vk)[1], θfit, θmax, CRLB
        )
    end
    return results
end

# ---- the test -------------------------------------------------------------------------------

@testset "Gaussian fits end at the maximum of the objective" begin
    # At 1000 photons. GaussMLE's expected counts are Float32 erf differences, whose error is
    # about 2e-8 photons per photon (2e-3 counts at 1e5 photons, 1e-4 relative to a
    # background of 2), so the two models agree to 1e-5 only below about 3000 photons.
    @testset "the Float64 model equals GaussMLE.evaluate_psf" begin
        rng = Xoshiro(3)
        for spec in specs
            ref, pm = refmodel(spec, PIXEL)
            worst = 0.0
            for _ in 1:20
                θ = spec.draw(rng, 1000.0, 2 + 8rand(rng))
                mu, _ = expected(ref, θ)
                θ32 = SVector{length(θ), Float32}(θ)
                for j in 1:BOX, i in 1:BOX
                    worst = max(
                        worst, abs(GaussMLE.evaluate_psf(pm, i, j, θ32) - mu[i, j]) / mu[i, j]
                    )
                end
            end
            @test worst < 1.0e-5
        end
    end

    conditions = [(N, bg) for N in (500.0, 2000.0, 1.0e4, 1.0e5) for bg in (2.0, 10.0)]
    println(
        "\n", rpad("model", 18), rpad("camera", 7), rpad("photons", 9), rpad("bg", 5),
        rpad("fits", 6), rpad("scored", 7), rpad("within", 8), rpad("worst", 9),
        rpad("truth-higher", 12), "ref failed"
    )
    seed = 100
    for spec in specs, camera_kind in (:ideal, :scmos)
        @testset "$(spec.name), $camera_kind camera" begin
            for (photons, bg) in conditions
                seed += 1
                results = run_condition(spec, camera_kind, photons, bg, seed)
                kept = filter(r -> r.scored, results)
                within = count(r -> r.distance <= MAX_DISTANCE, kept)
                share = isempty(kept) ? 0.0 : within / length(kept)
                worst = isempty(kept) ? NaN : maximum(r -> r.distance, kept)
                @printf(
                    "%-18s%-7s%-9.0f%-5.0f%-6d%-7d%-8s%-9.3g%-12d%d\n", spec.name, camera_kind,
                    photons, bg, length(results), length(kept),
                    @sprintf("%.2f%%", 100share), worst,
                    count(r -> r.from_truth, kept), count(r -> !r.reference_ok, results)
                )
                if share < MIN_SHARE
                    for r in sort(kept; by = r -> -r.distance)[1:min(3, end)]
                        @info "worst fits" distance = r.distance objective_gap = r.gap fit =
                            round.(r.θfit; digits = 3) maximum = round.(r.θmax; digits = 3)
                    end
                end
                @testset "photons $photons, bg $bg" begin
                    @test length(kept) >= min_scored(photons) * length(results)
                    @test all(r -> r.reference_ok, kept)
                    @test share >= MIN_SHARE
                end
            end
        end
    end
end
