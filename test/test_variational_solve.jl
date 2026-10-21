# End-to-end tests of the variational problem (Phase 9): a synthetic
# dual-Doppler pair gridded through the REAL `WindGridAccumulator` on the
# Springsteel quadrature mish, turned into a `VariationalProblem`
# (background, B = S∘D∘F, super-obs + continuity), solved, and diagnosed.

using Random
using LinearAlgebra
using TOML
using Springsteel

# Template config with a small [grid.springsteel] box covering the synthetic
# radar pair's overlap (6×6×3 cells over 24×24×3 km, mubar 3 → 18×18×9 mish),
# W Dirichlet at the surface, the 3-unknown accumulation, and a working
# [variational] block; `mutate!` tweaks the raw TOML dict.
function _solve_load(mutate! = identity)
    path = tempname() * ".toml"
    base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
    for (ax, lo, hi, c) in (("i", -12000.0, 12000.0, 6), ("j", -12000.0, 12000.0, 6),
                            ("k", 0.0, 3000.0, 3))
        base["grid"]["springsteel"][ax]["min"] = lo
        base["grid"]["springsteel"][ax]["max"] = hi
        base["grid"]["springsteel"][ax]["cells"] = c
    end
    base["grid"]["springsteel"]["k"]["bc"]["W"] = Dict("min" => "dirichlet")
    base["synthesis"]["include_vertical_velocity"] = true
    base["synthesis"]["max_elevation"] = 90.0
    base["variational"] = Dict{String,Any}(
        "enabled" => true,
        "state" => ["U", "V", "W"],
        "background" => "zero",
        "mass_continuity_sigma_per_s" => 1.0e-4,
        # The weak-background (inflation 10) solves below need ≈ 2.2k–3.7k
        # CG iterations with continuity at the nodes (3.8k–5.3k on the mish);
        # the others ≈ 0.3k.
        "max_inner_iterations" => 10000,
        "inner_tolerance" => 1.0e-6,
        # Filter lengths ≈ 1 cell (4 km horizontal, 1 km vertical) and the
        # template's σ_b (10 m/s for U/V, 2 m/s for W). The mish filter F is
        # variance-normalised (diag(F Fᵀ) = 1), so these σ are the actual
        # per-point background-error standard deviations. (Before that
        # normalisation the row-normalised F shrank white control noise to
        # std ≈ 0.05, and this test had to configure 100 / 20 m/s.)
        "background_error" => Dict(
            "default" => Dict("sigma" => 10.0, "filter_length_m" => [4000.0, 4000.0, 1000.0]),
            "W"       => Dict("sigma" => 2.0,  "filter_length_m" => [4000.0, 4000.0, 1000.0])),
    )
    mutate!(base)
    open(path, "w") do io; TOML.print(io, base); end
    return DaishoParameters(path)
end

# A copy of a stage-1 SynthesisOutput with some fields replaced.
_s1_with(s1; kw...) = Daisho.SynthesisOutput(; merge(
    (; (f => getfield(s1, f) for f in fieldnames(Daisho.SynthesisOutput))...), (; kw...))...)

# Grid every sweep of every volume through the real accumulator exactly like
# `_grid_volumes_spectral!` (Springsteel quadrature lattice, compute_roi).
function _solve_grid(p, vols, ref_lat, ref_lon)
    sgrid = Daisho.create_radar_grid(p)
    spec = Daisho.build_springsteel_grid_spec(sgrid;
        reference_latitude = ref_lat, reference_longitude = ref_lon)
    acc = Daisho.build_accumulator(spec, p)
    roi = Daisho.compute_roi(sgrid; h_factor = p.gridding.horizontal_roi_factor,
                                    v_factor = p.gridding.vertical_roi_factor)
    for vol in vols, s in eachindex(vol.sweeps)
        grid_sweep!(acc, vol, s, p; roi = (roi[1], roi[2]))
    end
    radar_grid = Daisho.finalize_grid(acc.scalar)
    return sgrid, spec, acc, radar_grid
end

@testset "Variational solve (Phase 9)" begin
    t_file = time()
    rng = MersenneTwister(5)
    ref_lat, ref_lon = 16.0, -24.0
    # Nondivergent linear wind, w = 0: ∂u/∂x + ∂v/∂y = 0, ∂v/∂x − ∂u/∂y = −3e-4 s⁻¹.
    a_uy, a_vx = 2.0e-4, -1.0e-4
    wind = (x, y, z) -> (8.0 + a_uy * y, -3.0 + a_vx * x, 0.0)
    vort_true = a_vx - a_uy
    volA, volB = make_synthetic_dual_doppler_field(wind; ref_lat = ref_lat, ref_lon = ref_lon,
                                                   dbz = (x, y, z) -> 30.0)
    # Helper default geometry: elevations 0.5–8°, ranges 2–35 km, radar A
    # 0.2° south and radar B 0.2° west of the reference point.
    @test maximum(s.fixed_angle for s in volA.sweeps) == 8.0

    p = _solve_load()
    sgrid, spec, acc, radar_grid = _solve_grid(p, (volA, volB), ref_lat, ref_lon)
    @test acc isa Daisho.WindGridAccumulator
    @test acc.n_unknowns == 3

    t_build = @elapsed prob = Daisho.VariationalProblem(acc, radar_grid, sgrid, p)
    state = prob.state
    nz, ny, nx = state.dims
    X, Y, Z = spec.x_axis, spec.y_axis, spec.z_axis
    @test (length(Z), length(Y), length(X)) == state.dims
    @test Z ≈ state.z_mish
    stage1 = prob.stage1
    q = stage1.quality_flag
    truth(v) = [wind(X[i], Y[j], Z[k])[v] for k in 1:nz, j in 1:ny, i in 1:nx]
    Ut, Vt = truth(1), truth(2)

    @testset "problem construction" begin
        @test prob.background.mode == :zero
        @test all(iszero, prob.background.x)
        @test prob.background.sigma == Daisho.background_sigma_vector(state)
        @test Daisho.nobs(prob.obsset) > 0
        @test Daisho.nobs(prob.obsset) == Daisho.nobs(prob.observations)
        @test Set(Daisho.kinds(prob.observations)) == Set([:radial_velocity, :mass_continuity])
        @test prob.nobs_vr isa Array{Int8,3}
        @test size(prob.nobs_vr) == state.dims
        @test all(in(0:3), prob.nobs_vr)
        @test sum(Int, prob.nobs_vr) == Daisho.nobs(prob.observations.groups[1])
        # Well observed (stage-1 solvable) points carry 2–3 super-obs rows.
        @test all(in(2:3), prob.nobs_vr[q .== 0])
        @test count(==(3), prob.nobs_vr) > 0
        @test size(prob.C) == (size(prob.S, 1), length(state))
        @test Daisho.adjoint_relative_error(prob.C; rng = rng) < 1e-12
        # Filter length in metres → cells (≈ 1 cell on every axis).
        @test Daisho.filter_length_cells(prob.F, :i, 1) ≈ 1.0
        @test Daisho.filter_length_cells(prob.F, :k, 3) ≈ 1.0
        @info "Phase 9 geometry" dims = state.dims stage1_ok = count(==(0), q) stage1_singular = count(==(Daisho.QFLAG_SINGULAR), q) stage1_sigma = count(x -> x in (1, 2, 3), q) stage1_nodata = count(==(Daisho.QFLAG_NODATA), q) nobs = Daisho.nobs(prob.obsset) build_seconds = t_build
    end

    t_solve = @elapsed r = Daisho.solve_variational(prob)
    Ua = Daisho.analysis_field(r, "U")
    Va = Daisho.analysis_field(r, "V")
    Wa = Daisho.analysis_field(r, "W")
    ok = q .== 0
    # Stage-1 could not solve (singular / baseline or σ-flagged) but gates
    # reached the point: the observed region the variational analysis fills.
    filled = (q .!= 0) .& (stage1.n_gates .> 0)

    @testset "zero-background solve" begin
        @test r.converged
        @test r.stats.outer_iterations == 1
        @test length(r.analysis) == length(state)
        @test r.background == prob.background.x
        @test Daisho.analysis_field(r, :wind_u) == Ua
        @test Daisho.background_field(r, "U") == zeros(state.dims)
        # Zero background: analysis = background + D F ξ exactly on the mish
        # up to the spline projection; the increment is the unsmoothed field.
        @test Daisho.increment_field(r, "U") == reshape(r.increment[1:state.n_mish], state.dims)
        eU_ok = maximum(abs.(Ua[ok] .- Ut[ok])); eV_ok = maximum(abs.(Va[ok] .- Vt[ok]))
        @test eU_ok < 0.3
        @test eV_ok < 0.3
        nfill = count(filled)
        @test nfill >= 1
        # Some of them are strictly singular (single look direction), not just σ-flagged.
        nsing = count(filled .& (q .== Daisho.QFLAG_SINGULAR))
        @test nsing >= 1
        eU_f = maximum(abs.(Ua[filled] .- Ut[filled])); eV_f = maximum(abs.(Va[filled] .- Vt[filled]))
        @test eU_f < 0.5
        @test eV_f < 0.5
        # Truth has w = 0 everywhere.
        eW = maximum(abs.(Wa[ok .| filled]))
        @test eW < 0.5
        mc = Daisho.mass_continuity_residual(prob, r)
        # Continuity is enforced at the nodes (the default lattice): the
        # residual there is at the solver tolerance; on the mish, between the
        # constraint points, it is only as small as the spline makes it.
        @test p.variational.mass_continuity_lattice == "nodes"
        @test Daisho.nobs(prob.observations.groups[2]) == state.n_nodes
        mcn = Daisho.mass_continuity_residual(prob, r; lattice = :nodes)
        @test size(mcn) == state.node_dims
        mcn_rms = sqrt(sum(abs2, mcn) / length(mcn))
        @test mcn_rms < 1e-5
        mc_rms = sqrt(sum(abs2, mc) / length(mc))
        @test size(mc) == state.dims
        @test mc_rms < 1e-4
        @test Set(keys(r.cost_by_group)) == Set([:radial_velocity, :mass_continuity])
        @test all(>=(0.0), values(r.cost_by_group))
        @info "Phase 9 zero-background solve" cg = repr(r.stats.inner_iterations) solve_seconds = t_solve max_err_U_ok = eU_ok max_err_V_ok = eV_ok n_filled = nfill n_filled_singular = nsing max_err_U_filled = eU_f max_err_V_filled = eV_f max_abs_W = eW mc_rms_nodes = mcn_rms mc_rms_mish = mc_rms cost_by_group = repr(r.cost_by_group)
    end

    @testset "evaluate_analysis! parity and derivative diagnostics" begin
        sg = prob.state_grid
        Daisho.evaluate_analysis!(sg, r)
        for (v, name) in enumerate(Daisho.variable_names(state))
            col = sg.params.vars[name]
            @test maximum(abs.(sg.physical[:, col, 1] .- vec(Daisho.analysis_field(r, v)))) < 1e-12
        end
        der = Daisho.analysis_derivatives(prob, r)
        @test Set(keys(der)) == Set(["U", "V", "W"])
        @test maximum(abs.(der["U"].value .- Ua)) < 1e-12
        # Derivative slabs agree with Springsteel's own gridTransform! slots
        # (2 = ∂x, 4 = ∂y, 6 = ∂z).
        colU = sg.params.vars["U"]
        @test maximum(abs.(vec(der["U"].dx) .- sg.physical[:, colU, 2])) < 1e-10
        @test maximum(abs.(vec(der["U"].dy) .- sg.physical[:, colU, 4])) < 1e-10
        @test maximum(abs.(vec(der["U"].dz) .- sg.physical[:, colU, 6])) < 1e-10
        div = Daisho.horizontal_divergence(prob, r)
        vort = Daisho.vertical_vorticity(prob, r)
        @test div ≈ der["U"].dx .+ der["V"].dy
        @test vort ≈ der["V"].dx .- der["U"].dy
        mc = Daisho.mass_continuity_residual(prob, r)
        dl = [Daisho.dlnrho_dz(prob.reference, z) for z in state.z_mish]
        @test mc ≈ div .+ der["W"].dz .+ der["W"].value .* reshape(dl, :, 1, 1)
        # Interior (one cell in from the lateral boundaries, well-observed
        # levels) vorticity vs the analytic −3e-4 s⁻¹.
        interior = falses(state.dims)
        for i in 1:nx, j in 1:ny, k in 1:nz
            interior[k, j, i] = (-8000.0 < X[i] < 8000.0) && (-8000.0 < Y[j] < 8000.0) && ok[k, j, i]
        end
        @test count(interior) > 0
        vort_err = sqrt(sum(abs2, vort[interior] .- vort_true) / count(interior))
        div_rms = sqrt(sum(abs2, div[interior]) / count(interior))
        @test vort_err < 0.2 * abs(vort_true)
        @test div_rms < 0.2 * abs(vort_true)
        @info "Phase 9 derivative diagnostics" vort_rms_err = vort_err vort_true = vort_true div_rms = div_rms
    end

    obs_region = ok .| filled
    iu, iv, iw = Daisho.wind_indices(state)
    σU, σW = 10.0, 2.0
    # Max |jump| between adjacent mish points along each axis of a 3-D field.
    maxjump(a) = maximum(maximum(abs.(diff(a; dims = d))) for d in 1:3)
    maxdiff(a, b, m) = maximum(abs.(a[m] .- b[m]))

    @testset "synthesis background (smooth fit + smoothed σ map)" begin
        psyn = _solve_load(b -> (b["variational"]["background"] = "synthesis"))
        t_bg = @elapsed bg = Daisho.Background(state, prob.state_grid, stage1, psyn)
        @test bg.mode == :synthesis
        xu = Daisho.state_view(bg.x, state, iu); su = Daisho.state_view(bg.sigma, state, iu)
        xv = Daisho.state_view(bg.x, state, iv); sv = Daisho.state_view(bg.sigma, state, iv)
        # Smooth fit: close to stage 1 at good points, finite and smooth
        # everywhere (no step to zero at flagged points).
        fitU = maxdiff(xu, stage1.comp1, ok); fitV = maxdiff(xv, stage1.comp2, ok)
        @test fitU < 0.3
        @test fitV < 0.3
        @test all(isfinite, bg.x)
        jU = maxjump(xu); jV = maxjump(xv)
        @test jU < 1.0
        @test jV < 1.0
        @test all(iszero, Daisho.state_view(bg.x, state, iw))
        # σ map: clamped to [σ/10, σ] (inflation 1), smoother than the raw
        # stage-1/config map, and equal to the definition.
        @test all(s -> σU / 10 - 1e-12 <= s <= σU + 1e-12, su)
        @test all(s -> σU / 10 - 1e-12 <= s <= σU + 1e-12, sv)
        @test all(==(σW), Daisho.state_view(bg.sigma, state, iw))
        raw = [ok[idx] ? stage1.sigma1[idx] : σU for idx in CartesianIndices(ok)]
        rawc = clamp.(raw, σU / 10, σU)
        @test maxjump(su) < maxjump(rawc)
        FU = Daisho.MishFilter(prob.state_grid, ["U"], Daisho.filter_lengths_cells(prob.state_grid,
                 ["U"], Daisho.filter_lengths_m(state)[:, iu:iu]); normalisation = :row)
        sm = zeros(state.n_mish)
        Daisho.apply!(sm, FU, vec(raw .^ 2))
        @test vec(su) ≈ clamp.(sqrt.(max.(sm, 0.0)), σU / 10, σU)
        # Dispatch: the 3-argument form cannot build the smooth fit (no state
        # grid); the high-level problem builds exactly this background.
        @test_throws ArgumentError Daisho.Background(state, stage1, psyn)
        @test_throws ArgumentError Daisho.Background(state, prob.state_grid, nothing, psyn)

        prob1 = Daisho.VariationalProblem(acc, radar_grid, sgrid, psyn)
        @test prob1.background.mode == :synthesis
        @test prob1.background.x ≈ bg.x rtol = 1e-10
        @test prob1.background.sigma ≈ bg.sigma rtol = 1e-12
        r1 = Daisho.solve_variational(prob1)
        @test r1.converged
        U1 = Daisho.analysis_field(r1, "U"); V1 = Daisho.analysis_field(r1, "V")
        e1 = max(maxdiff(U1, Ut, obs_region), maxdiff(V1, Vt, obs_region))
        @test e1 < 0.5

        # Large inflation: a weak background, so the analysis approaches the
        # zero-background one on the observed region. Two measured limits
        # (2026-09-23, this setup) set the inflation and the tolerance:
        # - With the variance-normalised F the zero background's σ = 10 m/s
        #   is a real prior that pulls the edges of the data toward 0 (the
        #   zero-background analysis is itself up to 0.25 m/s from the truth
        #   at filled points). A much weaker background therefore converges
        #   to a slightly DIFFERENT answer, not the zero-background one. With
        #   inflation 100 and CG fully converged (rtol 1e-6) the max
        #   difference is 0.338 m/s (synthesis) / 0.351 m/s
        #   (synthesis_uniform_sigma), both 0.21 m/s from the truth. That
        #   is not σ-map roughness: synthesis_uniform_sigma has no σ map.
        # - Inflation 100 makes σ_b 100–1000 m/s, and CG needs 21.6k / 28.5k
        #   iterations (35 s / 49 s) without a preconditioner. Inflation 10
        #   converges in ≈ 3.8k / 5.3k iterations and is still weak (σ_b of
        #   U/V 10–84 m/s here vs the zero background's 10 m/s).
        pinf = _solve_load(b -> (b["variational"]["background"] = "synthesis";
                                 b["variational"]["background_error_inflation"] = 10.0))
        prob_inf = Daisho.VariationalProblem(acc, radar_grid, sgrid, pinf)
        @test prob_inf.background.x ≈ bg.x rtol = 1e-10
        sui = Daisho.state_view(prob_inf.background.sigma, state, iu)
        @test sui ≈ 10 .* su
        @test all(==(10 * σW), Daisho.state_view(prob_inf.background.sigma, state, iw))
        r_inf = Daisho.solve_variational(prob_inf)
        @test r_inf.converged
        Ui = Daisho.analysis_field(r_inf, "U"); Vi = Daisho.analysis_field(r_inf, "V")
        d_inf = max(maxdiff(Ui, Ua, obs_region), maxdiff(Vi, Va, obs_region))
        e_inf = max(maxdiff(Ui, Ut, obs_region), maxdiff(Vi, Vt, obs_region))
        # 2026-10-20, continuity at the nodes (one constraint per cell instead
        # of 27 on the mish): the weaker coupling lets the prior matter more
        # at the data edges, so the two analyses differ by 0.36 m/s (was 0.24
        # with mish continuity) while BOTH move closer to the truth (0.15 m/s
        # here vs 0.26 for the zero background at filled points). CG: 2.2k
        # iterations (was 3.8k).
        @test d_inf < 0.45    # measured 0.36 (nodes), 0.24 (mish)
        @test e_inf < 0.3     # measured 0.15 (nodes), 0.11 (mish)
        @info "Phase 9b synthesis background" build_seconds = t_bg fit_max_err_U = fitU fit_max_err_V = fitV max_jump_U = jU max_jump_V = jV sigma_U_range = extrema(su) sigma_jump_smoothed = maxjump(su) sigma_jump_raw = maxjump(rawc) inflation1_max_err = e1 inflation10_max_diff_vs_zero = d_inf inflation10_max_err = e_inf cg_1 = repr(r1.stats.inner_iterations) cg_inf = repr(r_inf.stats.inner_iterations)
    end

    @testset "synthesis_uniform_sigma background" begin
        pun = _solve_load(b -> (b["variational"]["background"] = "synthesis_uniform_sigma"))
        bg = Daisho.Background(state, prob.state_grid, stage1, pun)
        @test bg.mode == :synthesis_uniform_sigma
        @test bg.sigma == Daisho.background_sigma_vector(state)
        xu = Daisho.state_view(bg.x, state, iu)
        @test maxdiff(xu, stage1.comp1, ok) < 0.3
        @test all(iszero, Daisho.state_view(bg.x, state, iw))
        r1 = Daisho.solve_variational(Daisho.VariationalProblem(acc, radar_grid, sgrid, pun))
        @test r1.converged
        e1 = max(maxdiff(Daisho.analysis_field(r1, "U"), Ut, obs_region),
                 maxdiff(Daisho.analysis_field(r1, "V"), Vt, obs_region))
        @test e1 < 0.5
        # Weak background: see the inflation note in the "synthesis" testset
        # (the 0.1 m/s target is not reachable: converged inflation 100 gives
        # 0.351 m/s, inflation 10 gives 0.26 m/s, because the zero
        # background's σ = 10 m/s is itself a real prior at the data edges).
        pinf = _solve_load(b -> (b["variational"]["background"] = "synthesis_uniform_sigma";
                                 b["variational"]["background_error_inflation"] = 10.0))
        prob_inf = Daisho.VariationalProblem(acc, radar_grid, sgrid, pinf)
        @test prob_inf.background.sigma == 10 .* Daisho.background_sigma_vector(state)
        r_inf = Daisho.solve_variational(prob_inf)
        @test r_inf.converged
        Ui = Daisho.analysis_field(r_inf, "U"); Vi = Daisho.analysis_field(r_inf, "V")
        d_inf = max(maxdiff(Ui, Ua, obs_region), maxdiff(Vi, Va, obs_region))
        e_inf = max(maxdiff(Ui, Ut, obs_region), maxdiff(Vi, Vt, obs_region))
        @test d_inf < 0.45    # measured 0.38 (nodes, see the note above), 0.26 (mish)
        @test e_inf < 0.3     # measured 0.17 (nodes), 0.13 (mish)
        @info "Phase 9b synthesis_uniform_sigma background" inflation1_max_err = e1 inflation10_max_diff_vs_zero = d_inf inflation10_max_err = e_inf cg_inf = repr(r_inf.stats.inner_iterations)
    end

    @testset "synthesis_profile background" begin
        ppr = _solve_load(b -> (b["variational"]["background"] = "synthesis_profile"))
        bg = Daisho.Background(state, stage1, ppr)
        @test bg.mode == :synthesis_profile
        @test Daisho.Background(state, prob.state_grid, stage1, ppr).x == bg.x
        @test bg.sigma == Daisho.background_sigma_vector(state)
        xu = Daisho.state_view(bg.x, state, iu); xv = Daisho.state_view(bg.x, state, iv)
        @test all(iszero, Daisho.state_view(bg.x, state, iw))
        for k in 1:nz
            @test all(==(xu[k, 1, 1]), xu[k, :, :])
            @test all(==(xv[k, 1, 1]), xv[k, :, :])
        end
        # One level by hand: the plain mean of the good stage-1 points.
        k0 = argmax([count(ok[k, :, :]) for k in 1:nz])
        m = ok[k0, :, :]
        @test xu[k0, 1, 1] ≈ sum(stage1.comp1[k0, :, :][m]) / count(m)
        @test xv[k0, 1, 1] ≈ sum(stage1.comp2[k0, :, :][m]) / count(m)
        rp = Daisho.solve_variational(Daisho.VariationalProblem(acc, radar_grid, sgrid, ppr))
        @test rp.converged
        ep = max(maxdiff(Daisho.analysis_field(rp, "U"), Ut, obs_region),
                 maxdiff(Daisho.analysis_field(rp, "V"), Vt, obs_region))
        @test ep < 0.5

        # A level with no good points is interpolated linearly in z between
        # its neighbours; an empty top level takes the nearest level's value.
        with_flags(qf) = _s1_with(stage1; quality_flag = qf)
        levmean(k) = sum(stage1.comp1[k, :, :][ok[k, :, :]]) / count(ok[k, :, :])
        kmid = 1 + nz ÷ 2
        @test all(k -> count(ok[k, :, :]) > 0, (kmid - 1, kmid + 1, nz - 1))
        qf = copy(stage1.quality_flag)
        qf[kmid, :, :] .= Daisho.QFLAG_SINGULAR
        qf[nz, :, :] .= Daisho.QFLAG_NODATA
        bgq = Daisho.Background(state, with_flags(qf), ppr)
        xq = Daisho.state_view(bgq.x, state, iu)
        z = state.z_mish
        wgt = (z[kmid] - z[kmid - 1]) / (z[kmid + 1] - z[kmid - 1])
        @test xq[kmid, 3, 4] ≈ (1 - wgt) * levmean(kmid - 1) + wgt * levmean(kmid + 1)
        @test xq[nz, 2, 2] ≈ levmean(nz - 1)
        @test xq[1, 1, 1] ≈ levmean(1)
        # All flagged → zeros and a warning.
        qall = fill(Daisho.QFLAG_NODATA, size(qf))
        bg0 = @test_logs (:warn, r"no stage-1 point") Daisho.Background(state, with_flags(qall), ppr)
        @test all(iszero, bg0.x)
        @info "Phase 9b synthesis_profile background" max_err = ep level = k0 u_mean = xu[k0, 1, 1]
    end

    @testset "solver options, Background errors, in-memory constructor" begin
        pso = _solve_load(b -> (b["variational"]["max_outer_iterations"] = 3;
                                b["variational"]["max_inner_iterations"] = 77;
                                b["variational"]["inner_tolerance"] = 1e-5;
                                b["variational"]["outer_tolerance"] = 2e-3))
        o = Daisho.solver_options(pso)
        @test o isa Daisho.VarSolverOptions
        @test (o.max_outer, o.max_inner, o.inner_tolerance, o.outer_tolerance) == (3, 77, 1e-5, 2e-3)

        # Zero background from the config; synthesis config without stage 1 → error.
        bz = Daisho.Background(state, p)
        @test bz.mode == :zero && all(iszero, bz.x)
        @test Daisho.Background(state, nothing, p).sigma == bz.sigma
        psyn = _solve_load(b -> (b["variational"]["background"] = "synthesis"))
        @test_throws ArgumentError Daisho.Background(state, nothing, psyn)
        ppr = _solve_load(b -> (b["variational"]["background"] = "synthesis_profile"))
        @test_throws ArgumentError Daisho.Background(state, nothing, ppr)
        @test_throws ArgumentError Daisho.Background(state, prob.state_grid, nothing, ppr)
        pun = _solve_load(b -> (b["variational"]["background"] = "synthesis_uniform_sigma"))
        @test_throws ArgumentError Daisho.Background(state, stage1, pun)
        # "zero" ignores a supplied stage-1 product.
        @test Daisho.Background(state, stage1, p).mode == :zero
        @test Daisho.Background(state, prob.state_grid, stage1, p).x == zeros(length(state))
        # Frame check: stage-1 components must be the state's u/v names.
        s1rot = _s1_with(stage1; component_names = ("UR", "VT"))
        @test_throws ArgumentError Daisho.Background(state, s1rot, ppr)
        @test_throws ArgumentError Daisho.Background(state, prob.state_grid, s1rot, psyn)
        # Raw constructor validation.
        n = length(state)
        @test_throws DimensionMismatch Daisho.Background(zeros(n), ones(n - 1), :zero)
        @test_throws ArgumentError Daisho.Background(zeros(n), zeros(n), :zero)
        @test_throws ArgumentError Daisho.Background(fill(NaN, n), ones(n), :zero)
        @test_throws ArgumentError Daisho.Background(zeros(n), ones(n), :bogus)
        # Stage-1 product on the wrong grid → DimensionMismatch.
        spec_small = GridSpec(shape = :volume_3d, reference_latitude = ref_lat,
            reference_longitude = ref_lon, x_axis = [0.0, 1000.0], y_axis = [0.0, 1000.0],
            z_axis = [500.0])
        acc_small = Daisho.WindGridAccumulator(spec_small, "VEL"; n_unknowns = 3)
        @test_throws DimensionMismatch Daisho.Background(state, prob.state_grid, Daisho.finalize_wind(acc_small, p), psyn)
        @test_throws DimensionMismatch Daisho.Background(state, Daisho.finalize_wind(acc_small, p), ppr)

        # In-memory constructor from parts reproduces the high-level problem.
        prob2 = Daisho.VariationalProblem(state, prob.state_grid, prob.observations,
                                          Daisho.Background(state, p), prob.reference, p;
                                          nobs_vr = prob.nobs_vr)
        @test prob2.stage1 === nothing
        r2 = Daisho.solve_variational(prob2)
        @test r2.analysis ≈ r.analysis rtol = 1e-10
        @test Daisho.analysis_coefficients!(zeros(length(r2.coefficients)), prob2.cf) == r2.coefficients
    end

    t_total = time() - t_file
    @info "Phase 9 test file runtime" seconds = t_total threads = Threads.nthreads()
end
