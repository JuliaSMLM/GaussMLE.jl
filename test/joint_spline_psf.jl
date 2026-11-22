# JointSplinePSFModel: one emitter fitted to two images at once, from a two-channel
# MicroscopePSFs MultiChannelPSF. Construction and its refusals, a fit of simulated pairs on
# two IdealCameras with different pixel sizes and a mirrored registration (near the truth),
# the pair checks, the expected counts and their derivatives against MicroscopePSFs'
# forward law at a rotated, scaled and mirrored registration, the start with a dark image,
# and the emitter type. Bias, scatter against the CRLB, p-values, the sCMOS joint fit, the
# joint CRLB against each view's and a view with zero transmission are in Long
# (joint_spline_psf.jl); CPU/GPU agreement is in the GPU group. MultiChannelPSF exists from
# MicroscopePSFs 0.5.7.
using Test, GaussMLE, MicroscopePSFs, SMLMData, StaticArrays, Random, Statistics
using SMLMSim: poisson_noise

include(joinpath(@__DIR__, "long", "utils", "joint_helpers.jl"))

if !isdefined(MicroscopePSFs, :MultiChannelPSF)
    @info "joint_spline_psf.jl: MicroscopePSFs $(pkgversion(MicroscopePSFs)) has no " *
        "MultiChannelPSF (0.5.7 and later), so JointSplinePSFModel is not defined"
else
    @testset "JointSplinePSFModel" begin
        spline = joint_test_spline()
        I23 = [1.0 0 0; 0 1 0]
        # View 2 mirrored in x and shifted 0.3 um in y, 0.15 um deeper
        reg = [-1.0 0.0 16.0; 0.0 1.0 0.3]
        mc = MultiChannelPSF(;
            channels = [spline, spline], registration = [I23, reg],
            transmission = [0.45, 0.55], z_offset = [0.0, 0.15]
        )

        @testset "construction from a two-channel MultiChannelPSF" begin
            jm = JointSplinePSFModel(mc; pixel_size = 0.1)
            @test jm isa JointSplinePSFModel{Float32}
            @test jm isa GaussMLE.PSFModel{6, Float32}
            @test length(jm) == 6
            @test jm.transmission == (0.45f0, 0.55f0)
            # By default all that both channels cover: channel 2 reaches z + 0.15 <= 0.6
            @test jm.z_range[1] ≈ -0.6f0 && jm.z_range[2] ≈ 0.45f0
            @test jm.views[2].z_range[1] ≈ -0.45f0 && jm.views[2].z_range[2] ≈ 0.6f0
            # z bounds are rounded inward to Float32, as for SplinePSFModel
            zr = JointSplinePSFModel(mc; pixel_size = 0.1, z_range = (-0.3, 0.3)).z_range
            @test zr[1] ≈ -0.3f0 && zr[2] ≈ 0.3f0 && -0.3 <= zr[1] && zr[2] <= 0.3
            # The reference channel is view 1
            mc2 = MultiChannelPSF(;
                channels = [spline, spline], registration = [reg, I23],
                transmission = [0.3, 0.7], z_offset = [0.15, 0.0], reference = 2
            )
            jm2 = JointSplinePSFModel(mc2; pixel_size = 0.1)
            @test jm2.transmission == (0.7f0, 0.3f0)
            @test jm2.z_offset == (0.0f0, 0.15f0)

            one = MultiChannelPSF(; channels = [spline])
            three = MultiChannelPSF(; channels = [spline, spline, spline])
            analytic = MultiChannelPSF(; channels = [spline, ScalarPSF(1.4, 0.68, 1.518)])
            singular = MultiChannelPSF(;
                channels = [spline, spline], registration = [I23, [1.0 1.0 0; 1.0 1.0 0]]
            )
            for bad in (one, three, analytic, singular)
                @test_throws ArgumentError JointSplinePSFModel(bad; pixel_size = 0.1)
            end
            # A reference registration within MicroscopePSFs' tolerance of the identity
            # but not equal to it: positions are reported on the reference camera
            near = MultiChannelPSF(;
                channels = [spline, spline],
                registration = [[1.0 + 1.0e-9 0 0; 0 1 0], reg]
            )
            @test_throws "not exactly the identity" JointSplinePSFModel(near; pixel_size = 0.1)
            # A 2D SplinePSF channel has no z to fit
            flat = SplinePSF(AiryPSF(1.4, 0.68), -1.0:0.1:1.0, -1.0:0.1:1.0)
            @test_throws "no z axis" JointSplinePSFModel(
                MultiChannelPSF(; channels = [spline, flat]); pixel_size = 0.1
            )
            # A z_range channel 2 does not cover (z + 0.15 beyond its +0.6)
            @test_throws ArgumentError JointSplinePSFModel(
                mc; pixel_size = 0.1, z_range = (-0.3, 0.5)
            )
            @test_throws "JointSplinePSFModel" SplinePSFModel(mc; pixel_size = 0.1)
        end

        # Camera 1: the left 64 columns at 0.1 um; camera 2: a crop at 0.12 um pixels whose
        # first column is the sensor's 65th, so its pixel edges start at 7.68 um
        cam1 = IdealCamera(joint_edges(1, 64, 0.1), joint_edges(1, 128, 0.1))
        cam2 = IdealCamera(joint_edges(65, 64, 0.12), joint_edges(1, 128, 0.12))
        box, n, photons, bgs = 9, 40, 3000.0, (8.0, 12.0)
        jm = JointSplinePSFModel(mc; pixel_size = 0.1)
        rng = Xoshiro(11)
        Random.seed!(11)  # SMLMSim's noise draws from the global generator
        truths = [(2 + 3rand(rng), 2 + 8rand(rng), 0.8rand(rng) - 0.4) for _ in 1:n]
        mus, corners = joint_expected(mc, (cam1, cam2), truths; photons, bgs, box)
        frames = Int32.(1:n)
        b1 = ROIBatch(Float32.(poisson_noise(mus[1])), corners[1]..., frames, cam1)
        b2 = ROIBatch(Float32.(poisson_noise(mus[2])), corners[2]..., frames, cam2)

        @testset "fit of simulated pairs" begin
            smld, info = fit((b1, b2); psf_model = jm, backend = :cpu)
            es = smld.emitters
            @test es isa Vector{Emitter3DFitJoint{Float32}}
            @test info.n_fits == n
            @test smld.camera === cam1
            # Positions in camera 1's microns near the truth: errors in CRLB units average
            # about 0.8 when the fit is right; a wrong registration or pixel size is many
            for (row, f, sf) in ((1, :x, :σ_x), (2, :y, :σ_y), (3, :z, :σ_z))
                z = [
                    (getfield(e, f) - t[row]) / getfield(e, sf) for (e, t) in zip(es, truths)
                ]
                @test mean(abs, z) < 1.2
            end
            @test mean(e.photons for e in es) ≈ photons rtol = 0.05
            @test mean(e.bg for e in es) ≈ bgs[1] rtol = 0.05
            @test mean(e.bg2 for e in es) ≈ bgs[2] rtol = 0.05
        end

        @testset "pairs are checked" begin
            fitter = GaussMLEConfig(psf_model = jm, backend = :cpu)
            sub(b, r) = ROIBatch(
                b.data[:, :, r], b.x_corners[r], b.y_corners[r], b.frame_indices[r], b.camera
            )
            @test_throws ArgumentError fit((b1, sub(b2, 1:(n - 1))), fitter)
            small = ROIBatch(b2.data[2:8, 2:8, :], b2.x_corners, b2.y_corners, frames, cam2)
            @test_throws ArgumentError fit((b1, small), fitter)
            shifted = ROIBatch(b2.data, b2.x_corners, b2.y_corners, frames .+ Int32(1), cam2)
            @test_throws ArgumentError fit((b1, shifted), fitter)
            scmos = SCMOSCamera(64, 128, 0.12f0, 1.0f0)
            @test_throws "use one camera type" fit(
                (b1, ROIBatch(b2.data, b2.x_corners, b2.y_corners, frames, scmos)), fitter
            )
            # The corners enter the fit. Impossible pairs are refused: corners left at 1
            # map ROI 1 entirely outside ROI 2 here
            view2(xc, yc = b2.y_corners; camera = cam2) =
                ROIBatch(b2.data, xc, yc, frames, camera)
            ones32 = fill(Int32(1), n)
            @test_throws "entirely outside ROI 2" fit(
                (ROIBatch(b1.data, ones32, ones32, frames, cam1), view2(ones32, ones32)),
                fitter
            )
            # Footprints that overlap on both camera axes but not along the diagonal of a
            # rotated registration are apart as well
            turned = JointSplinePSFModel(
                MultiChannelPSF(;
                    channels = [spline, spline],
                    registration = [I23, [1.0 -1.0 4.75; 1.0 1.0 1.85]]
                ); pixel_size = 0.1
            )
            cam30 = IdealCamera(joint_edges(1, 30, 0.1), joint_edges(1, 30, 0.1))
            blank(cx, cy) = ROIBatch(zeros(Float32, 9, 9, 1), Int32[cx], Int32[cy], Int32[1], cam30)
            @test_throws "entirely outside ROI 2" GaussMLE._validate_pair_geometry(
                turned, blank(1, 11), blank(22, 22)
            )
            # Pairs that only look wrong are warned about and fitted as given, no result
            # changed: a ROI 2 moved by 4 pixels in every pair (the median), in one pair
            # of the batch, and in every pair of a batch too small for a median. A valid
            # batch gives no warning.
            warn = Base.CoreLogging.Warn
            @test_logs min_level = warn fit((b1, b2), fitter)
            # swapped batches: the ROIs still overlap here, 6 pixels apart in y
            @test_logs (:warn, r"a median .* batch1 is not the reference channel's ROIs") match_mode = :any fit(
                (b2, b1), fitter
            )
            @test_logs (:warn, r"a median -4") match_mode = :any fit(
                (b1, view2(b2.x_corners .+ Int32(4))), fitter
            )
            inner(i) = 1 < b1.x_corners[i] < 64 - box + 1 &&
                1 < b1.y_corners[i] < 128 - box + 1 &&
                1 < b2.x_corners[i] < 64 - box - 3 && 1 < b2.y_corners[i] < 128 - box + 1
            four = findall(inner, 1:n)[1:4]
            k = four[1]
            xs = copy(b2.x_corners)
            xs[k] += Int32(4)
            got = Ref{Any}()
            @test_logs (:warn, Regex("in 1 of the \\d+ pairs away from the sensor edges \\(the first is pair $k\\)")) match_mode = :any (
                got[] = fit((b1, view2(xs)), fitter)
            )
            @test all(e -> isfinite(e.σ_x) && isfinite(e.σ_bg2), got[][1].emitters)
            @test_logs (:warn, r"in 4 of the 4 pairs") match_mode = :any fit(
                (
                    sub(b1, four),
                    ROIBatch(
                        b2.data[:, :, four], b2.x_corners[four] .+ Int32(4),
                        b2.y_corners[four], frames[four], cam2
                    ),
                ), fitter
            )
            # Every ROI lies on its own camera, for an IdealCamera too
            @test_throws "view 2: ROI" fit((b1, view2(b2.x_corners .+ Int32(60))), fitter)
            # Pixel edges that are not evenly spaced (one column 0.05 um wider)
            ex = joint_edges(65, 64, 0.12)
            ex[30:end] .+= 0.05
            uneven = IdealCamera(ex, joint_edges(1, 128, 0.12))
            @test_throws "unevenly spaced x pixel edges" fit(
                (b1, view2(b2.x_corners; camera = uneven)), fitter
            )
            # Evenly spaced edges stored as Float32 far from zero are within their own
            # precision of even spacing
            far32 = IdealCamera(
                Float32.(16384 .+ (0:64) .* 0.1), Float32.(joint_edges(1, 128, 0.1))
            )
            @test GaussMLE._uniform_pixel_grid(far32, 1)[2][1] ≈ 0.1 rtol = 1.0e-4
            # A gain that is not finite and positive, and counts that are not finite after
            # the camera calibration
            scmos_pair(gain2, offset2 = 0.0f0) = (
                ROIBatch(
                    b1.data, b1.x_corners, b1.y_corners, frames,
                    SCMOSCamera(
                        Float32.(joint_edges(1, 64, 0.1)), Float32.(joint_edges(1, 128, 0.1));
                        readnoise = 1.0f0, offset = 0.0f0, gain = 1.0f0, qe = 1.0f0
                    )
                ),
                view2(
                    b2.x_corners;
                    camera = SCMOSCamera(
                        Float32.(joint_edges(65, 64, 0.12)),
                        Float32.(joint_edges(1, 128, 0.12));
                        readnoise = 1.0f0, offset = offset2, gain = gain2, qe = 1.0f0
                    )
                ),
            )
            @test fit(scmos_pair(1.0f0), fitter)[2].n_fits == n
            @test_throws "view 2: the camera gain" fit(scmos_pair(0.0f0), fitter)
            @test_throws "view 2: the camera gain" fit(scmos_pair(NaN32), fitter)
            @test_throws "view 2: ROI 1 has the count" fit(scmos_pair(1.0f0, NaN32), fitter)
            # Constraints of another model's size
            five = GaussMLE.default_constraints(jm.views[1], box)
            @test_throws "constraints for 5 parameters" fit(
                (b1, b2), GaussMLEConfig(psf_model = jm, backend = :cpu, constraints = five)
            )
            # A joint model needs a pair; a pair needs a joint model
            @test_throws ArgumentError fit(b1, fitter)
            @test_throws ArgumentError fit(b1.data, fitter)
            @test_throws ArgumentError fit(
                (b1, b2), GaussMLEConfig(psf_model = GaussianXYNB(0.13f0))
            )
        end

        @testset "pairs at a sensor edge, an even box, a ROI 2 off its emitter" begin
            # ROIs clamped at a sensor edge leave their emitters off centre, in one view or
            # both, and an even box has no centre pixel: such pairs are valid and fitted.
            boxe = 10
            edge = [
                (3.03, 0.12, 0.0), (6.3, 5.0, 0.1), (2.5, 12.7, -0.1), (6.32, 0.15, 0.0),
            ]
            rnge = Xoshiro(13)
            Random.seed!(13)
            inner = [(2 + 3rand(rnge), 2 + 8rand(rnge), 0.6rand(rnge) - 0.3) for _ in 1:16]
            te = vcat(edge, inner)
            mue, ce = joint_expected(mc, (cam1, cam2), te; photons, bgs, box = boxe)
            @test ce[1][2][1] == 1 && ce[1][1][2] == 64 - boxe + 1  # clamped
            fe = Int32.(1:length(te))
            se, _ = fit(
                (
                    ROIBatch(Float32.(poisson_noise(mue[1])), ce[1]..., fe, cam1),
                    ROIBatch(Float32.(poisson_noise(mue[2])), ce[2]..., fe, cam2),
                ),
                GaussMLEConfig(psf_model = jm, backend = :cpu)
            )
            # finite uncertainties, and x and y within 5 of them of the truth
            near(sm, ts) = all(e -> isfinite(e.σ_x) && isfinite(e.σ_y), sm.emitters) && all(
                abs(e.x - t[1]) < 5 * e.σ_x && abs(e.y - t[2]) < 5 * e.σ_y
                    for (e, t) in zip(sm.emitters, ts)
            )
            @test near(se, te)

            # A batch whose pairs are all clamped at one sensor edge (their ROI centres
            # about 2 pixels apart in y under the registration) is as valid: fitted, and
            # no warning, since pairs at an edge are not judged on their offset
            warn = Base.CoreLogging.Warn
            top = [(1.5 + 0.5i, 0.1 + 0.02i, 0.0) for i in 1:8]
            mut, ct = joint_expected(mc, (cam1, cam2), top; photons, bgs, box)
            @test all(==(1), ct[1][2]) && all(==(1), ct[2][2])
            ft = Int32.(1:length(top))
            got = Ref{Any}()
            @test_logs min_level = warn (
                got[] = fit(
                    (
                        ROIBatch(Float32.(poisson_noise(mut[1])), ct[1]..., ft, cam1),
                        ROIBatch(Float32.(poisson_noise(mut[2])), ct[2]..., ft, cam2),
                    ),
                    GaussMLEConfig(psf_model = jm, backend = :cpu)
                )
            )
            @test near(got[][1], top)

            # A registration 2 camera-2 pixels wrong in y is warned about from the pairs
            # away from the edges, although most pairs of the batch sit at the top edge
            mcw = MultiChannelPSF(;
                channels = [spline, spline],
                registration = [I23, [-1.0 0.0 16.0; 0.0 1.0 0.54]],
                transmission = [0.45, 0.55], z_offset = [0.0, 0.15]
            )
            mixed = vcat(top, inner[1:6])
            mum, cm = joint_expected(mc, (cam1, cam2), mixed; photons, bgs, box)
            fm = Int32.(1:length(mixed))
            @test_logs (:warn, r"a median .* over the 6 pairs away from the sensor edges") match_mode = :any fit(
                (ROIBatch(mum[1], cm[1]..., fm, cam1), ROIBatch(mum[2], cm[2]..., fm, cam2)),
                GaussMLEConfig(psf_model = JointSplinePSFModel(mcw; pixel_size = 0.1), backend = :cpu)
            )

            # A ROI 2 placed 4 pixels from its emitter, in one pair of a batch that follows
            # the registration: fitted, but the emitter sits on ROI 2's first column, so
            # its far columns lie beyond the table (lateral_range 0.65 um plus a pixel) and
            # the fit reports infinite uncertainties. Two halves of one sensor at 0.1 um.
            camL = IdealCamera(joint_edges(1, 64, 0.1), joint_edges(1, 128, 0.1))
            camR = IdealCamera(joint_edges(65, 64, 0.1), joint_edges(1, 128, 0.1))
            mch = MultiChannelPSF(;
                channels = [spline, spline], registration = [I23, [-1.0 0.0 12.8; 0.0 1.0 0.0]],
                transmission = [0.45, 0.55]
            )
            jmh = JointSplinePSFModel(mch; pixel_size = 0.1, lateral_range = 0.65)
            # the first emitter on a pixel centre in both views (column 34 and 31)
            th = vcat([(3.35, 6.05, 0.0)], [(2 + 3rand(rnge), 2 + 8rand(rnge), 0.0) for _ in 1:19])
            muh, ch = joint_expected(mch, (camL, camR), th; photons, bgs, box)
            cxs = copy(ch[2][1])
            cxs[1] += Int32(4)
            ex = Float64.(camR.pixel_edges_x[cxs[1]:(cxs[1] + box)])
            ey = Float64.(camR.pixel_edges_y[ch[2][2][1]:(ch[2][2][1] + box)])
            X2, Y2, z2 = channel_position(mch, 2, th[1]...)
            m2 = copy(muh[2])
            m2[:, :, 1] .= bgs[2] .+ MicroscopePSFs.integrate_pixels(
                spline, ex, ey, [Emitter3D(X2, Y2, z2, photons * 0.55)]; sampling = 4
            )
            fh = Int32.(1:length(th))
            @test_logs (:warn, r"in 1 of the 20 pairs") match_mode = :any (
                got[] = fit(
                    (ROIBatch(muh[1], ch[1]..., fh, camL), ROIBatch(m2, cxs, ch[2][2], fh, camR)),
                    GaussMLEConfig(psf_model = jmh, backend = :cpu)
                )
            )
            sh = got[][1]
            @test isinf(sh.emitters[1].σ_x) && isinf(sh.emitters[1].σ_bg2)
            @test all(e -> isfinite(e.σ_x) && isfinite(e.σ_bg2), sh.emitters[2:end])

            # An image with zero transmission has no PSF to cut off. Its ROI is clamped at
            # camera R's first column with the emitter in that column, so its far columns
            # are beyond the table; the lit ROI is centred. The fits keep their
            # uncertainties.
            mcd = MultiChannelPSF(;
                channels = [spline, spline], registration = [I23, [-1.0 0.0 9.65; 0.0 1.0 0.0]],
                transmission = [1.0, 0.0]
            )
            td = [(3.2 + 0.008i, 5.0 + 0.3i, 0.0) for i in 1:5]
            mud, cdk = joint_expected(mcd, (camL, camR), td; photons, bgs, box)
            @test all(==(1), cdk[2][1])
            fd = Int32.(1:length(td))
            sd, _ = fit(
                (
                    ROIBatch(Float32.(poisson_noise(mud[1])), cdk[1]..., fd, camL),
                    ROIBatch(Float32.(poisson_noise(mud[2])), cdk[2]..., fd, camR),
                ),
                GaussMLEConfig(
                    psf_model = JointSplinePSFModel(mcd; pixel_size = 0.1, lateral_range = 0.65),
                    backend = :cpu
                )
            )
            @test near(sd, td)

            # One pixel size per camera serves the tables, the registration and the output.
            # Float32 edges far from zero make the first pixel's width differ from the mean
            # by 1e-3 of a pixel, which over 2,000 pixels is 0.2 um.
            camF = IdealCamera(
                Float32.(4096 .+ (0:2048) .* 0.1), Float32.(joint_edges(1, 128, 0.1))
            )
            mcf = MultiChannelPSF(;
                channels = [spline, spline],
                registration = [I23, [-1.0 0.0 4306.95; 0.0 1.0 0.3]],
                transmission = [0.45, 0.55]
            )
            tf = [(4295.95, 6.05, 0.0)]
            muf, cf = joint_expected(mcf, (camF, cam2), tf; photons, bgs, box)
            sf, _ = fit(
                (
                    ROIBatch(Float32.(muf[1]), cf[1]..., Int32[1], camF),
                    ROIBatch(Float32.(muf[2]), cf[2]..., Int32[1], cam2),
                ),
                GaussMLEConfig(psf_model = JointSplinePSFModel(mcf; pixel_size = 0.1), backend = :cpu)
            )
            @test sf.emitters[1].x ≈ 4295.95 atol = 0.003
            @test sf.emitters[1].y ≈ 6.05 atol = 0.003
        end

        @testset "start with an image that receives no light" begin
            # An image with zero transmission holds no light of the emitter: its counts
            # above the background (here one bright pixel off centre) are noise and must
            # not move the start. Position and photons start from the lit image alone.
            for dark in (1, 2)
                lit = 3 - dark
                mcd = MultiChannelPSF(;
                    channels = [spline, spline], registration = [I23, reg],
                    transmission = dark == 2 ? [1.0, 0.0] : [0.0, 1.0],
                    z_offset = [0.0, 0.15]
                )
                pm = GaussMLE._model_for_cameras(
                    JointSplinePSFModel(mcd; pixel_size = 0.1), cam1, cam2
                )
                mud, cd = joint_expected(mcd, (cam1, cam2), truths[1:1]; photons, bgs, box)
                noise = fill(1000.0f0, box, box)
                noise[2, 3] = 9000.0f0
                roi = ntuple(v -> v == dark ? noise : mud[v][:, :, 1], 2)
                cx = (cd[1][1][1], cd[2][1][1])
                cy = (cd[1][2][1], cd[2][2][1])
                θ = GaussMLE._fit_start(roi, box, pm, cx, cy)
                _, s, x, y = GaussMLE._roi_moments(roi[lit], box, Float32)
                at = lit == 1 ? [x, y] :
                    pm.A_px_inv * ([x, y] .- GaussMLE._joint_offset(pm, cx, cy))
                @test θ[1] ≈ at[1] atol = 1.0e-3
                @test θ[2] ≈ at[2] atol = 1.0e-3
                @test θ[4] ≈ s rtol = 1.0e-5
                @test θ[4 + dark] ≈ 1000.0f0 rtol = 0.05
                # nor does the dark image's level choose the z start: its likelihood is
                # the same at every z, and at 1e9 counts would hide the lit image's
                zstart(level) = GaussMLE._fit_start(
                    ntuple(v -> v == dark ? fill(level, box, box) : mud[v][:, :, 1], 2),
                    box, pm, cx, cy
                )[3]
                @test zstart(1.0f9) == zstart(5.0f0)
            end
        end

        @testset "forward law at a rotated, scaled and mirrored registration" begin
            # No fit decides this: the kernel's expected counts are compared with
            # MicroscopePSFs' integrate_pixels at channel_position, and its derivatives
            # with central differences. The registration is not symmetric (rotation 28
            # degrees, scales 1.06 and 0.93, a mirror in y), the two channels have
            # different PSFs without mirror symmetry, camera 2 has non-square pixels, both
            # cameras' edges start away from zero, and channel 2 is 0.12 um shallower: a
            # transposed chain rule, swapped axes or a wrong origin changes the numbers.
            shaped(phases) = begin
                zc = ZernikeCoefficients(15)
                for (k, v) in phases
                    zc.phase[k] = v
                end
                SplinePSF(
                    ScalarPSF(1.4, 0.68, 1.518; zernike_coeffs = zc);
                    lateral_range = 1.2, axial_range = 0.6, lateral_step = 0.1,
                    axial_step = 0.1
                )
            end
            s1 = shaped((5 => 0.45, 7 => 0.3))
            s2 = shaped((6 => -0.35, 8 => 0.35))
            θr = deg2rad(28)
            A = [cos(θr) -sin(θr); sin(θr) cos(θr)] * [1.06 0.0; 0.0 -0.93]
            camA = IdealCamera(
                collect(3.2 .+ (0:64) .* 0.1), collect(1.5 .+ (0:128) .* 0.1)
            )
            camB = IdealCamera(
                collect(20.0 .+ (0:96) .* 0.11), collect(0.7 .+ (0:128) .* 0.13)
            )
            # the middle of camera A's emitters lands on the middle of camera B
            t = [20.0 + 48 * 0.11, 0.7 + 64 * 0.13] - A * [6.4, 7.5]
            mcg = MultiChannelPSF(;
                channels = [s1, s2], registration = [I23, hcat(A, t)],
                transmission = [0.4, 0.5], z_offset = [0.0, -0.12]
            )
            jmg = JointSplinePSFModel(mcg; pixel_size = 0.1)
            pm = GaussMLE._model_for_cameras(jmg, camA, camB)
            boxg, photonsg, bgsg = 11, 4000.0, (6.0, 9.0)
            rngg = Xoshiro(5)
            truthsg = [
                (5.0 + 2.5rand(rngg), 5.0 + 5rand(rngg), 0.7rand(rngg) - 0.35) for _ in 1:12
            ]
            musg, cg = joint_expected(
                mcg, (camA, camB), truthsg; photons = photonsg, bgs = bgsg, box = boxg
            )
            # the kernel's counts and derivatives of both views at θ
            function kernel_counts(θ, cx, cy)
                o = GaussMLE._joint_offset(pm, cx, cy)
                blank = (zeros(Float32, boxg, boxg), zeros(Float32, boxg, boxg))
                u = (zeros(Float32, boxg, boxg), zeros(Float32, boxg, boxg))
                du = (zeros(Float32, boxg, boxg, 6), zeros(Float32, boxg, boxg, 6))
                for v in 1:2
                    vm, _, _, _, _, Av, tr, θv = GaussMLE._view_context(
                        pm, v, θ, o, blank, blank, cx, cy
                    )
                    for j in 1:boxg, i in 1:boxg
                        u[v][i, j], d = GaussMLE._joint_view_pixel(vm, v, i, j, θv, Av, tr)
                        du[v][i, j, :] .= d
                    end
                end
                return u, du
            end
            count_error = [0.0, 0.0]
            derivative_error = zeros(6)
            steps = (0.01f0, 0.01f0, 0.005f0, 1.0f0, 0.1f0, 0.1f0)
            for (k, (X, Y, z)) in enumerate(truthsg)
                cx = (cg[1][1][k], cg[2][1][k])
                cy = (cg[1][2][k], cg[2][2][k])
                θ = SVector{6, Float32}(
                    (X - 3.2) / 0.1 + 0.5 - (cx[1] - 1), (Y - 1.5) / 0.1 + 0.5 - (cy[1] - 1),
                    z, photonsg, bgsg[1], bgsg[2]
                )
                u, du = kernel_counts(θ, cx, cy)
                for v in 1:2
                    peak = maximum(musg[v][:, :, k]) - bgsg[v]
                    count_error[v] = max(
                        count_error[v], maximum(abs, u[v] .- musg[v][:, :, k]) / peak
                    )
                end
                for p in 1:6
                    e = SVector{6, Float32}(ntuple(q -> q == p ? steps[p] : 0.0f0, 6))
                    up, _ = kernel_counts(θ + e, cx, cy)
                    um, _ = kernel_counts(θ - e, cx, cy)
                    for v in 1:2
                        fd = (up[v] .- um[v]) ./ (2 * steps[p])
                        an = du[v][:, :, p]
                        derivative_error[p] = max(
                            derivative_error[p],
                            maximum(abs, fd .- an) / max(maximum(abs, an), 1.0f-6)
                        )
                    end
                end
            end
            # 0.4 % and 0.6 % of the peak when right (the table's interpolation error);
            # the transposed linear part gives tens of percent
            @test maximum(count_error) < 0.015
            # 3e-4 at most when right
            @test maximum(derivative_error) < 2.0e-3
            # A fit of the expected counts returns the truth (within 0.06 CRLB when right;
            # the transposed linear part is 89 CRLB off)
            fr = Int32.(1:length(truthsg))
            smld, _ = fit(
                (ROIBatch(musg[1], cg[1]..., fr, camA), ROIBatch(musg[2], cg[2]..., fr, camB));
                psf_model = jmg, backend = :cpu
            )
            for (row, f, sf) in ((1, :x, :σ_x), (2, :y, :σ_y), (3, :z, :σ_z))
                d = [
                    abs(getfield(e, f) - tr[row]) / getfield(e, sf)
                        for (e, tr) in zip(smld.emitters, truthsg)
                ]
                @test maximum(d) < 0.15
            end
        end

        @testset "Emitter3DFitJoint" begin
            names = fieldnames(Emitter3DFitJoint)
            @test names[1:18] == fieldnames(Emitter3DFitGaussMLE)
            @test names[19:20] == (:bg2, :σ_bg2)
            @test ismutabletype(Emitter3DFitJoint)
            vals = Dict(f => Float32(k) for (k, f) in enumerate(names) if k <= 14 || k > 18)
            e = Emitter3DFitJoint(; vals..., frame = 3, id = 7)
            @test e isa Emitter3DFitJoint{Float32}
            @test e.bg2 == 19.0f0 && e.σ_bg2 == 20.0f0 && e.frame == 3 && e.id == 7
            @test e.dataset == 1 && e.track_id == 0
        end
    end
end
