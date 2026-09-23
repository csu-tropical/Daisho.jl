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
        "max_inner_iterations" => 500,
        "inner_tolerance" => 1.0e-6,
        # Filter lengths ≈ 1 cell (4 km horizontal, 1 km vertical). The
        # row-normalised mish filter F (F·1 = 1) shrinks the amplitude of
        # white control noise: here std(F ξ) ≈ 0.05, so the effective
        # background-error σ of D·F·ξ is ≈ 0.05 × the configured sigma. The
        # configured 100 / 20 m/s give ≈ 5 / 1 m/s effective (the template's
        # 10 / 2 would give ≈ 0.5 / 0.1 m/s and pull the analysis ~0.5 m/s
        # toward the zero background at the edges of the data).
        "background_error" => Dict(
            "default" => Dict("sigma" => 100.0, "filter_length_m" => [4000.0, 4000.0, 1000.0]),
            "W"       => Dict("sigma" => 20.0,  "filter_length_m" => [4000.0, 4000.0, 1000.0])),
    )
    mutate!(base)
    open(path, "w") do io; TOML.print(io, base); end
    return DaishoParameters(path)
end

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
        mc_rms = sqrt(sum(abs2, mc) / length(mc))
        @test mc_rms < 1e-5
        @test Set(keys(r.cost_by_group)) == Set([:radial_velocity, :mass_continuity])
        @test all(>=(0.0), values(r.cost_by_group))
        @info "Phase 9 zero-background solve" cg = repr(r.stats.inner_iterations) solve_seconds = t_solve max_err_U_ok = eU_ok max_err_V_ok = eV_ok n_filled = nfill n_filled_singular = nsing max_err_U_filled = eU_f max_err_V_filled = eV_f max_abs_W = eW mc_rms = mc_rms cost_by_group = repr(r.cost_by_group)
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

    @testset "synthesis background" begin
        iu, iv, iw = Daisho.wind_indices(state)
        # The Background picks stage-1 values/σ where quality_flag == 0 and
        # the config values elsewhere.
        pinf = _solve_load(b -> (b["variational"]["background"] = "synthesis";
                                 b["variational"]["background_error_inflation"] = 100.0))
        bg = Daisho.Background(state, stage1, pinf)
        @test bg.mode == :synthesis
        xu = Daisho.state_view(bg.x, state, iu); su = Daisho.state_view(bg.sigma, state, iu)
        xv = Daisho.state_view(bg.x, state, iv); sv = Daisho.state_view(bg.sigma, state, iv)
        @test xu[ok] == stage1.comp1[ok]
        @test xv[ok] == stage1.comp2[ok]
        @test all(iszero, xu[.!ok]) && all(iszero, xv[.!ok])
        @test su[ok] ≈ max.(100.0 .* stage1.sigma1[ok], 1e-3 * 100.0)
        @test sv[ok] ≈ max.(100.0 .* stage1.sigma2[ok], 1e-3 * 100.0)
        @test all(==(100.0), su[.!ok]) && all(==(100.0), sv[.!ok])
        @test all(iszero, Daisho.state_view(bg.x, state, iw))
        @test all(==(20.0), Daisho.state_view(bg.sigma, state, iw))
        # Dispatch on the config: "synthesis" + stage1 → synthesis mode.
        @test Daisho.Background(state, stage1, pinf).x == bg.x

        # Large inflation was expected to make the background weak and so
        # reproduce the zero-background answer. It does NOT (max |ΔU|, |ΔV| ≈ 5.6, 7.1
        # m/s on the observed region; CG converged, a true minimiser): the
        # :synthesis background is a step (stage-1 values where
        # quality_flag == 0, 0 elsewhere) and its σ map is rough point to
        # point (inflation·stage-1 σ ∈ [33, 194] beside the config 100). D
        # multiplies the smooth filtered control pointwise, so a smooth
        # increment through a rough σ map, or one cancelling the step's
        # spline projection, needs a rough control that F (Gaussian, σ ≈ 1
        # cell) can barely produce; the minimiser trades that prior cost
        # against misfit. Each ingredient alone shows it (diagnosed with
        # this setup): stage-1 x + uniform σ → 5.7 m/s, zero x + the rough
        # σ map → 4.6 m/s, zero x + iid σ ∈ [33, 194] → 12 m/s. Recorded as
        # broken pending a design decision (e.g. smooth the σ map / fill
        # the background smoothly across flagged points).
        prob_inf = Daisho.VariationalProblem(acc, radar_grid, sgrid, pinf)
        @test prob_inf.background.mode == :synthesis
        r_inf = Daisho.solve_variational(prob_inf)
        @test r_inf.converged
        # Compared on the observed region: at the few no-data points the
        # analysis is the (different) background itself.
        obs_region = ok .| filled
        dU = maximum(abs.(Daisho.analysis_field(r_inf, "U")[obs_region] .- Ua[obs_region]))
        dV = maximum(abs.(Daisho.analysis_field(r_inf, "V")[obs_region] .- Va[obs_region]))
        @test_broken dU < 0.1
        @test_broken dV < 0.1

        # Inflation 1 (stage-1 σ used as-is, the double-counting case) is
        # still close to the truth on the observed region.
        p1 = _solve_load(b -> (b["variational"]["background"] = "synthesis"))
        r1 = Daisho.solve_variational(Daisho.VariationalProblem(acc, radar_grid, sgrid, p1))
        @test r1.converged
        U1 = Daisho.analysis_field(r1, "U"); V1 = Daisho.analysis_field(r1, "V")
        e1 = max(maximum(abs.(U1[obs_region] .- Ut[obs_region])),
                 maximum(abs.(V1[obs_region] .- Vt[obs_region])))
        @test e1 < 0.5
        @info "Phase 9 synthesis background" inflation100_max_diff = max(dU, dV) inflation1_max_err = e1 cg_inf = repr(r_inf.stats.inner_iterations) cg_1 = repr(r1.stats.inner_iterations)
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
        @test_throws DimensionMismatch Daisho.Background(state, Daisho.finalize_wind(acc_small, p), psyn)

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
