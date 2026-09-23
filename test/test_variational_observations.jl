# Tests for the variational observation types, their conversion to backends,
# and the super-obs / continuity / scalar-value builders (Phase 8).

using Random
using LinearAlgebra
using TOML
using Springsteel

# Template config with a working [variational] block (state U, V, W, DBZ)
# on a small [grid.springsteel] box `cells` over `extent` metres, W Dirichlet
# at the surface, include_vertical_velocity = true; `mutate!` tweaks it.
function _obs_load(mutate! = identity; cells = (4, 4, 3), extent = (4000.0, 4000.0, 3000.0),
                   state = ["U", "V", "W", "DBZ"], velocity_variance = 0.25)
    path = tempname() * ".toml"
    base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
    for (ax, c, L) in zip(("i", "j", "k"), cells, extent)
        base["grid"]["springsteel"][ax]["cells"] = c
        base["grid"]["springsteel"][ax]["min"] = 0.0
        base["grid"]["springsteel"][ax]["max"] = L
    end
    "W" in state && (base["grid"]["springsteel"]["k"]["bc"]["W"] = Dict("min" => "dirichlet"))
    base["synthesis"]["include_vertical_velocity"] = true
    base["synthesis"]["velocity_variance"] = velocity_variance
    var = Dict{String,Any}(
        "enabled" => true,
        "state" => state,
        "background_error" => Dict{String,Any}(
            "default" => Dict("sigma" => 10.0, "filter_length_m" => [1000.0, 1000.0, 1000.0])))
    "W" in state && (var["background_error"]["W"] = Dict("sigma" => 2.0, "filter_length_m" => [1000.0, 1000.0, 1000.0]))
    "DBZ" in state && (var["scalar_obs"] = Dict("DBZ" => Dict("sigma" => 5.0, "undetect_value" => -10.0)))
    base["variational"] = var
    mutate!(base)
    open(path, "w") do io; TOML.print(io, base); end
    return DaishoParameters(path)
end

function _obs_setup(p)
    sgrid = Daisho.create_radar_grid(p)
    state = Daisho.StateVector(p, sgrid)
    state_grid = Daisho.create_state_grid(p, sgrid, state)
    spec = Daisho.build_springsteel_grid_spec(sgrid; reference_latitude = 40.0, reference_longitude = -105.0)
    return sgrid, state, state_grid, spec
end

_obs_acc(spec; N = 3) = Daisho.WindGridAccumulator(spec, "VEL"; n_unknowns = N)

# Unit look vector g of a gate at (az, el) (radians), via the accumulator's own helper.
_look(az, el, N = 3) = Daisho._wind_coeffs!(zeros(N), Float64(az), Float64(el), N)

# Add gates (w, g, ṽ) to the accumulator planes at mish point (k, j, i).
function _add_gates!(acc, k, j, i, gates)
    N = acc.n_unknowns
    for (w, g, vt) in gates
        for a in 1:N, b in a:N
            acc.AtWA[Daisho._packed_index(a, b, N), k, j, i] += w * g[a] * g[b]
        end
        for a in 1:N
            acc.AtWb[a, k, j, i] += w * vt * g[a]
        end
        acc.weight_total[k, j, i] += w
        acc.gate_count[k, j, i] += Int32(1)
    end
    return acc
end

# (e, y, rinv) rows of the super-obs at one point.
_rows(group, loc) = [(collect(o.e), o.value, Daisho.obs_inverse_variance(o))
                     for o in group.obs if o.loc == loc]

_ovars(o, state) = [t.var for t in Daisho.obs_terms(o, state)]

@testset "Variational observations (Phase 8)" begin
    rng = MersenneTwister(4)
    p = _obs_load()
    sgrid, state, state_grid, spec = _obs_setup(p)
    iu, iv, iw = Daisho.wind_indices(state)
    nz, ny, nx = state.dims
    σ2 = p.synthesis.velocity_variance
    @test σ2 == 0.25

    @testset "Locations and terms" begin
        @test Daisho.mish_flat(Daisho.MishIndex(1, 1, 1), state.dims) == 1
        @test Daisho.mish_flat(Daisho.MishIndex(2, 3, 4), state.dims) == 2 + 2nz + 3nz * ny
        @test Daisho.mish_flat(Daisho.MishIndex(nz, ny, nx), state.dims) == state.n_mish
        x = collect(1.0:length(state))
        @test x[Daisho.mish_flat(Daisho.MishIndex(2, 3, 4), state.dims)] == Daisho.state_view(x, state, 1)[2, 3, 4]
        @test_throws ArgumentError Daisho.mish_flat(Daisho.MishIndex(nz + 1, 1, 1), state.dims)
        @test_throws ArgumentError Daisho.mish_flat(Daisho.MishIndex(1, 0, 1), state.dims)
        @test_throws ArgumentError Daisho.ScalarValueObs(Daisho.MishIndex(1, 1, 1), 1, 0.0, 0.0)
        @test_throws ArgumentError Daisho.ScalarValueObs(Daisho.MishIndex(1, 1, 1), 1, NaN, 1.0)
        o = Daisho.ScalarValueObs(Daisho.MishIndex(1, 2, 3), 4, 7.5, 2.0)
        @test Daisho.obs_value(o) == 7.5 && Daisho.obs_sigma(o) == 2.0
        @test Daisho.obs_inverse_variance(o) == 0.25
        @test Daisho.obs_kind(o) == :scalar_value
        @test Daisho.is_linear(o)
        @test Daisho.obs_terms(o, state) == [Daisho.ObsTerm(4, 0, 1.0)]
        @test Daisho.obs_slots(o, state) == [(4, 1)]
    end

    # ── 1. Super-obs identity ──────────────────────────────────────────────
    @testset "Super-obs identity (3×3)" begin
        acc = _obs_acc(spec)
        k, j, i = 2, 3, 1
        gates = [(0.2 + rand(rng), _look(2π * rand(rng), 0.6 * rand(rng)), 10 * randn(rng)) for _ in 1:5]
        _add_gates!(acc, k, j, i, gates)
        grp, cnt = Daisho.radial_velocity_superobs(acc, state, p)
        @test grp isa Daisho.ObsGroup{Daisho.RadialVelocitySuperObs}
        @test grp.kind == :radial_velocity
        @test Daisho.nobs(grp) == 3 == cnt[k, j, i]
        @test sum(cnt) == 3 && eltype(cnt) == Int8 && size(cnt) == state.dims
        @test Daisho.radial_velocity_superobs_count(acc, state, p) == cnt
        @test Daisho.build_radial_velocity_superobs(acc, state, p).obs == grp.obs
        rows = _rows(grp, Daisho.MishIndex(k, j, i))
        S = sum(w * g * g' for (w, g, _) in gates)
        b = sum(w * vt * g for (w, g, vt) in gates)
        gatecost(x) = sum(w * (vt - dot(g, x))^2 for (w, g, vt) in gates) / σ2
        rowcost(x) = sum(r * (dot(e, x) - y)^2 for (e, y, r) in rows)
        rowgrad(x) = sum(2r * (dot(e, x) - y) * e for (e, y, r) in rows)
        x1, x2 = randn(rng, 3), randn(rng, 3)
        lhs = rowcost(x1) - rowcost(x2)
        rhs = gatecost(x1) - gatecost(x2)
        id_res = abs(lhs - rhs) / abs(rhs)
        gr_res = maximum(norm(rowgrad(x) - 2 * (S * x - b) / σ2) / norm(2 * (S * x - b) / σ2) for x in (x1, x2))
        @info "Super-obs identity residuals" cost = id_res gradient = gr_res
        @test id_res < 1e-10
        @test gr_res < 1e-10
        # Every row has u, v, w terms with the eigenvector weights.
        for o in grp.obs
            @test _ovars(o, state) == [iu, iv, iw]
            @test [t.weight for t in Daisho.obs_terms(o, state)] == collect(o.e)
            @test abs(norm(collect(o.e)) - 1) < 1e-12
        end
        # error_inflation scales every inverse variance by 1/inflation.
        p2 = _obs_load(b -> (b["variational"]["radial_velocity"] = Dict("error_inflation" => 2.0)))
        grp2 = Daisho.build_radial_velocity_superobs(acc, state, p2)
        @test [Daisho.obs_inverse_variance(o) for o in grp2.obs] ≈
              [Daisho.obs_inverse_variance(o) / 2 for o in grp.obs] rtol = 1e-14
        # Empty / zero-weight points emit nothing.
        acc0 = _obs_acc(spec)
        acc0.gate_count[1, 1, 1] = Int32(2)   # gates but zero weight
        @test Daisho.nobs(Daisho.build_radial_velocity_superobs(acc0, state, p)) == 0
        # Dims must match the state mish.
        spec_bad = Daisho.GridSpec(shape = :volume_3d, reference_latitude = 0.0, reference_longitude = 0.0,
                                   x_axis = [0.0], y_axis = [0.0], z_axis = [0.0])
        @test_throws DimensionMismatch Daisho.build_radial_velocity_superobs(_obs_acc(spec_bad), state, p)
    end

    # ── 2. Degenerate geometries ───────────────────────────────────────────
    @testset "Single look, baseline, guard" begin
        acc = _obs_acc(spec)
        g = _look(0.7, 0.3)
        ws = 0.5 .+ rand(rng, 4); vs = 5 .+ randn(rng, 4)
        _add_gates!(acc, 1, 1, 1, [(ws[m], g, vs[m]) for m in 1:4])
        # Two horizontal looks (el = 0 exactly): N = 3 but w is unobserved.
        _add_gates!(acc, 2, 2, 2, [(1.0, _look(0.3, 0.0), 3.0), (2.0, _look(1.9, 0.0), -1.0)])
        # Two horizontal looks plus a nearly coincident third one with a
        # tiny elevation: numerically singular third direction.
        gs = [_look(0.3, 0.0), _look(1.9, 0.0), _look(0.3, 1e-7)]
        _add_gates!(acc, 3, 2, 2, [(1.0, gs[1], 3.0), (1.0, gs[2], -1.0), (1.0, gs[3], 3.0)])
        grp, cnt = Daisho.radial_velocity_superobs(acc, state, p)
        @test cnt[1, 1, 1] == 1 && cnt[2, 2, 2] == 2 && cnt[3, 2, 2] == 2
        @test Daisho.nobs(grp) == 5
        r1 = _rows(grp, Daisho.MishIndex(1, 1, 1))
        e, y, r = r1[1]
        sgn = sign(dot(e, g))
        @test abs(abs(dot(e, g)) - 1) < 1e-12
        @test y * sgn ≈ sum(ws .* vs) / sum(ws) rtol = 1e-12
        @test r ≈ sum(ws) / σ2 rtol = 1e-12
        # Baseline rows: no w term (structurally absent third unknown).
        for o in grp.obs
            o.loc == Daisho.MishIndex(2, 2, 2) || continue
            @test o.e[3] == 0.0
            @test !(iw in _ovars(o, state))
        end
        # Guard off (1e-20): the tiny eigen-direction is emitted.
        p0 = _obs_load(b -> (b["variational"]["radial_velocity"] = Dict("eigenvalue_guard_relative" => 1e-20)))
        cnt0 = Daisho.radial_velocity_superobs_count(acc, state, p0)
        @test cnt0[3, 2, 2] == 3
    end

    # ── 3. Horizontal approximation and N checks ────────────────────────────
    @testset "Horizontal approximation" begin
        acc = _obs_acc(spec)
        gates = [(0.2 + rand(rng), _look(2π * rand(rng), 0.5 * rand(rng)), 10 * randn(rng)) for _ in 1:5]
        _add_gates!(acc, 1, 2, 3, gates)
        ph = _obs_load(b -> (b["variational"]["radial_velocity"] = Dict("horizontal_approximation" => true)))
        grp = Daisho.build_radial_velocity_superobs(acc, state, ph)
        @test Daisho.nobs(grp) == 2
        rows = _rows(grp, Daisho.MishIndex(1, 2, 3))
        for o in grp.obs
            @test o.e[3] == 0.0
            @test _ovars(o, state) == [iu, iv]
        end
        S2 = sum(w * g[1:2] * g[1:2]' for (w, g, _) in gates)
        b2 = sum(w * vt * g[1:2] for (w, g, vt) in gates)
        gatecost(x) = sum(w * (vt - dot(g[1:2], x))^2 for (w, g, vt) in gates) / σ2
        rowcost(x) = sum(r * (dot(e[1:2], x) - y)^2 for (e, y, r) in rows)
        x1, x2 = randn(rng, 2), randn(rng, 2)
        @test abs((rowcost(x1) - rowcost(x2)) - (gatecost(x1) - gatecost(x2))) < 1e-10 * abs(gatecost(x1) - gatecost(x2))
        g1 = sum(2r * (dot(e[1:2], x1) - y) * e[1:2] for (e, y, r) in rows)
        @test norm(g1 - 2 * (S2 * x1 - b2) / σ2) < 1e-10 * norm(2 * (S2 * x1 - b2) / σ2)
        # An N = 2 accumulator with horizontal_approximation works too.
        acc2 = _obs_acc(spec; N = 2)
        _add_gates!(acc2, 1, 2, 3, [(w, g[1:2], vt) for (w, g, vt) in gates])
        grp2 = Daisho.build_radial_velocity_superobs(acc2, state, ph)
        @test Daisho.nobs(grp2) == 2
        @test sort([o.value^2 for o in grp2.obs]) ≈ sort([o.value^2 for o in grp.obs]) rtol = 1e-12
        # N = 2 without the approximation but with W in the state → error.
        err = try
            Daisho.build_radial_velocity_superobs(acc2, state, p); nothing
        catch ex
            ex
        end
        @test err isa ArgumentError
        @test occursin("include_vertical_velocity", err.msg)
        # A state without W cannot represent an N = 3 row without the approximation.
        pnw = _obs_load(; state = ["U", "V"]) do b
            b["variational"]["mass_continuity"] = false
        end
        snw = Daisho.StateVector(pnw, Daisho.create_radar_grid(pnw))
        @test_throws ArgumentError Daisho.build_radial_velocity_superobs(acc, snw, pnw)
    end

    # ── 4. Mass continuity ─────────────────────────────────────────────────
    @testset "Mass continuity" begin
        ref = Daisho.ExponentialReferenceProfile(1.2, 8500.0, NaN, NaN)
        grp = Daisho.build_mass_continuity_obs(state, ref, p)
        @test grp isa Daisho.ObsGroup{Daisho.MassContinuityObs}
        @test grp.kind == :mass_continuity
        @test Daisho.nobs(grp) == state.n_mish
        @test all(o -> Daisho.obs_value(o) == 0.0, grp.obs)
        @test all(o -> Daisho.obs_sigma(o) == p.variational.mass_continuity_sigma_per_s, grp.obs)
        # Analytic anelastic field: u = −(x/2)(w' + w·dlnρ), v = −(y/2)(…).
        H = 8500.0; Htop = 3000.0; w0 = 5.0
        dl = -1 / H
        xs = spec.x_axis; ys = spec.y_axis; zs = state.z_mish
        maxres = 0.0
        for o in grp.obs[rand(rng, 1:length(grp.obs), 12)]
            k, j, i = o.loc.k, o.loc.j, o.loc.i
            x, y, z = xs[i], ys[j], zs[k]
            w = w0 * sin(π * z / Htop); wz = w0 * π / Htop * cos(π * z / Htop)
            ux = -(wz + w * dl) / 2; vy = -(wz + w * dl) / 2
            sample = Dict((iu, 1) => ux, (iv, 2) => vy, (iw, 3) => wz, (iw, 0) => w)
            res = sum(t.weight * sample[(t.var, t.deriv)] for t in Daisho.obs_terms(o, state))
            maxres = max(maxres, abs(res))
            @test o.dlnrho_dz == dl
        end
        @test maxres < 1e-12
        # dlnρ = 0 drops the w value term.
        o0 = Daisho.MassContinuityObs(Daisho.MishIndex(1, 1, 1), 1e-4, 0.0)
        @test length(Daisho.obs_terms(o0, state)) == 3
        # Disabled → empty; state without W → error.
        pmc = _obs_load(b -> (b["variational"]["mass_continuity"] = false))
        @test Daisho.nobs(Daisho.build_mass_continuity_obs(state, ref, pmc)) == 0
        pnw = _obs_load(; state = ["U", "V"]) do b
            b["variational"]["mass_continuity"] = false
        end
        snw = Daisho.StateVector(pnw, Daisho.create_radar_grid(pnw))
        @test_throws ArgumentError Daisho.obs_terms(o0, snw)
    end

    # ── 5. Scalar value obs ────────────────────────────────────────────────
    @testset "Scalar value obs" begin
        fidx = Daisho.field_index_dict(p)
        c = fidx["DBZ"]
        rg = fill(p.io.fill_value, length(fidx), nz, ny, nx)
        rg[c, :, :, :] .= 20.0 .+ randn(rng, nz, ny, nx)
        rg[c, 1, 1, 1] = p.io.fill_value
        rg[c, 2, 1, 1] = NaN
        rg[c, 1, 2, 1] = p.io.undetect
        rg[c, 1, 3, 1] = p.io.undetect
        grp = Daisho.build_scalar_value_obs(rg, state, p)
        @test grp isa Daisho.ObsGroup{Daisho.ScalarValueObs}
        @test grp.kind == :scalar_value
        @test Daisho.nobs(grp) == state.n_mish - 2
        idbz = state.index["DBZ"]
        @test all(o -> o.var == idbz && Daisho.obs_sigma(o) == 5.0, grp.obs)
        und = [o for o in grp.obs if o.loc in (Daisho.MishIndex(1, 2, 1), Daisho.MishIndex(1, 3, 1))]
        @test length(und) == 2 && all(o -> Daisho.obs_value(o) == -10.0, und)
        o = only(o for o in grp.obs if o.loc == Daisho.MishIndex(3, 2, 4))
        @test Daisho.obs_value(o) == rg[c, 3, 2, 4]
        @test !any(o -> o.loc in (Daisho.MishIndex(1, 1, 1), Daisho.MishIndex(2, 1, 1)), grp.obs)
        @test_throws DimensionMismatch Daisho.build_scalar_value_obs(rg[:, 1:2, :, :], state, p)
        # A state without gridded fields → empty group.
        puv = _obs_load(; state = ["U", "V", "W"])
        suv = Daisho.StateVector(puv, Daisho.create_radar_grid(puv))
        @test Daisho.nobs(Daisho.build_scalar_value_obs(rg, suv, puv)) == 0
    end

    # ── 6. Wind speed (nonlinear) ──────────────────────────────────────────
    @testset "WindSpeedObs" begin
        o = Daisho.WindSpeedObs(Daisho.MishIndex(2, 2, 2), 9.0, 0.5)
        @test !Daisho.is_linear(o)
        @test Daisho.obs_kind(o) == :wind_speed
        @test Daisho.obs_slots(o, state) == [(iu, 1), (iv, 1)]
        @test [t.var for t in Daisho.obs_terms(o, state)] == [iu, iv]
        smp = [3.0, -4.0]
        @test Daisho.model_value(o, smp, state) ≈ 5.0
        lw = Daisho.linearize(o, smp, state)
        ε = 1e-6
        fd = [(Daisho.model_value(o, smp .+ ε .* ((1:2) .== q), state) -
               Daisho.model_value(o, smp .- ε .* ((1:2) .== q), state)) / (2ε) for q in 1:2]
        @test maximum(abs, collect(lw) .- fd) < 1e-8
        @test collect(Daisho.linearize(o, [0.0, 0.0], state)) == [0.0, 0.0]

        # Through build_backends → ObsSet on a smooth quadratic (u, v) field.
        names = Daisho.variable_names(state)
        xs = spec.x_axis; ys = spec.y_axis; zs = state.z_mish
        uf(x, y, z) = 8.0 + 3e-4 * x - 2e-8 * y^2 + 1e-4 * z
        vf(x, y, z) = -3.0 + 2e-4 * y + 1e-8 * x * z
        xm = zeros(length(state))
        U = Daisho.state_view(xm, state, iu); V = Daisho.state_view(xm, state, iv)
        for i in 1:nx, j in 1:ny, k in 1:nz
            U[k, j, i] = uf(xs[i], ys[j], zs[k]); V[k, j, i] = vf(xs[i], ys[j], zs[k])
        end
        Sp = Daisho.SplineProjection(state_grid, names)
        a = zeros(size(Sp, 1)); Daisho.apply!(a, Sp, xm)
        locs = [Daisho.MishIndex(rand(rng, 1:nz), rand(rng, 1:ny), rand(rng, 1:nx)) for _ in 1:20]
        wobs = [Daisho.WindSpeedObs(l, 9.0 + randn(rng), 0.5) for l in locs]
        set = Daisho.ObservationSet()
        Daisho.push_group!(set, Daisho.ObsGroup(:wind_speed, wobs))
        bks = Daisho.build_backends(set, state, state_grid)
        @test length(bks) == 1 && bks[1] isa Daisho.MishObsBackend
        @test !Daisho.is_linear(bks[1])
        @test bks[1].slots == [(iu, 1), (iv, 1)]
        obsset = Daisho.build_obsset(set, state, state_grid; nchunks = 1)
        yh = zeros(Daisho.nobs(obsset))
        Daisho.evaluate!(yh, obsset, a; linearize = true)
        be = obsset.backends[1]
        @test be.slots == [(iu, 1), (iv, 1)]
        expected = [hypot(uf(xs[l.i], ys[l.j], zs[l.k]), vf(xs[l.i], ys[l.j], zs[l.k])) for l in locs][be.perm]
        relerr = maximum(abs.(yh .- expected) ./ expected)
        @info "WindSpeedObs spline evaluation" relerr
        @test relerr < 1e-6
        # Tangent weights at a are (u/s, v/s).
        m = 3; l = locs[be.perm[m]]
        uu, vv = uf(xs[l.i], ys[l.j], zs[l.k]), vf(xs[l.i], ys[l.j], zs[l.k])
        @test be.term_w[be.term_ptr[m]] ≈ uu / hypot(uu, vv) rtol = 1e-6
        Daisho.linearize!(obsset, a)
        @test Daisho.adjoint_relative_error(obsset; rng = rng) < 1e-12
    end

    # ── 8. PointObs / scattered backend / mixed sets ───────────────────────
    @testset "PointObs and mixed sets" begin
        names = Daisho.variable_names(state)
        pts = [(500.0 + 3000 * rand(rng), 500.0 + 3000 * rand(rng), 300.0 + 2400 * rand(rng)) for _ in 1:6]
        pobs = [Daisho.PointObs(Daisho.XYZ(q...), randn(rng), 0.3,
                                [Daisho.ObsTerm(iu, 0, 1.0), Daisho.ObsTerm(iw, 3, -0.5)]) for q in pts]
        @test Daisho.obs_kind(pobs[1]) == :point
        @test Daisho.obs_kind(Daisho.PointObs(Daisho.XYZ(1.0, 1.0, 1.0), 0.0, 1.0,
                                              [Daisho.ObsTerm(1, 0, 1.0)]; kind = :sonde)) == :sonde
        sobs = [Daisho.ScalarValueObs(Daisho.MishIndex(1, j, 2), state.index["DBZ"], 10.0 * j, 5.0) for j in 1:ny]
        set = Daisho.ObservationSet(Daisho.ObsGroup[Daisho.ObsGroup(:point, pobs),
                                                    Daisho.ObsGroup(:scalar_value, sobs),
                                                    Daisho.ObsGroup(:point, Daisho.PointObs[])])
        @test Daisho.nobs(set) == 6 + ny
        @test Daisho.kinds(set) == [:point, :scalar_value, :point]
        bks = Daisho.build_backends(set, state, state_grid)
        @test length(bks) == 2
        @test bks[1] isa Daisho.ScatteredObsBackend && bks[2] isa Daisho.MishObsBackend
        obsset = Daisho.build_obsset(set, state, state_grid)
        @test Daisho.nobs(obsset) == 6 + ny
        @test Daisho.obs_rinv(obsset)[1:6] ≈ fill(1 / 0.09, 6) rtol = 1e-14
        @test Daisho.adjoint_relative_error(obsset; rng = rng) < 1e-12
        # The scattered path equals a direct ScatteredObsBackend.
        direct = Daisho.ScatteredObsBackend(:point, state_grid, names,
            [q[c] for q in pts, c in 1:3], [Daisho.obs_value(o) for o in pobs],
            fill(1 / 0.09, 6), [[(iu, 1, 1.0), (iw, 4, -0.5)] for _ in pobs])
        @test bks[1].H == direct.H
        # cost_by_group labels the backends and sums repeated kinds.
        res = randn(rng, Daisho.nobs(obsset))
        cbg = Daisho.cost_by_group(set, obsset, res)
        cbb = Daisho.cost_by_backend(obsset, res)
        @test cbg[:point] ≈ cbb[1] && cbg[:scalar_value] ≈ cbb[2]
        set2 = Daisho.ObservationSet(Daisho.ObsGroup[Daisho.ObsGroup(:point, pobs[1:3]),
                                                     Daisho.ObsGroup(:point, pobs[4:6])])
        os2 = Daisho.build_obsset(set2, state, state_grid)
        c2 = Daisho.cost_by_group(set2, os2, res[1:6])
        @test collect(keys(c2)) == [:point]
        @test c2[:point] ≈ sum(Daisho.cost_by_backend(os2, res[1:6]))
        # Nonlinear obs at an XYZ location are refused.
        bad = Daisho.ObservationSet(Daisho.ObsGroup[Daisho.ObsGroup(:wind_speed,
            [Daisho.WindSpeedObs(Daisho.XYZ(100.0, 100.0, 100.0), 5.0, 1.0)])])
        @test_throws ArgumentError Daisho.build_backends(bad, state, state_grid)
    end

    # ── 7. End to end through the solver ───────────────────────────────────
    @testset "End to end (3 radars + continuity)" begin
        Lx, Ly, Lz = 10_000.0, 10_000.0, 5000.0
        amp = 10.0
        pe = _obs_load(; cells = (8, 8, 5), extent = (Lx, Ly, Lz), state = ["U", "V", "W"],
                       velocity_variance = (0.01 * amp)^2)
        sg, st, stg, sp = _obs_setup(pe)
        ref = Daisho.build_reference_profile(pe, sg)
        H = pe.reference_state.density_scale_height_m
        @test Daisho.dlnrho_dz(ref, 1000.0) ≈ -1 / H
        # Anelastic Beltrami-type truth: ρ̄·(u, v, w) is divergence-free.
        k = 2π / Lx; l = 2π / Ly; m = π / (2Lz)
        A = 10.0; B = -6.0; Cc = -(A * k + B * l) / m
        uf(x, y, z) = A * sin(k * x) * cos(l * y) * cos(m * z) * exp(z / H)
        vf(x, y, z) = B * cos(k * x) * sin(l * y) * cos(m * z) * exp(z / H)
        wf(x, y, z) = Cc * cos(k * x) * cos(l * y) * sin(m * z) * exp(z / H)
        acc = Daisho.WindGridAccumulator(sp, "VEL"; n_unknowns = 3)
        radars = [(-5000.0, -5000.0, 0.0), (15_000.0, -4000.0, 0.0), (-4000.0, 15_000.0, 0.0)]
        xs, ys, zs = sp.x_axis, sp.y_axis, st.z_mish
        nz_, ny_, nx_ = st.dims
        for i in 1:nx_, j in 1:ny_, kk in 1:nz_
            x, y, z = xs[i], ys[j], zs[kk]
            tru = (uf(x, y, z), vf(x, y, z), wf(x, y, z))
            gates = map(radars) do r
                dx, dy, dz = x - r[1], y - r[2], z - r[3]
                g = _look(atan(dx, dy), atan(dz, hypot(dx, dy)))
                (1.0, g, dot(g, collect(tru)))
            end
            _add_gates!(acc, kk, j, i, gates)
        end
        oset = Daisho.build_observation_set(acc, nothing, st, ref, pe)
        @test Daisho.kinds(oset) == [:radial_velocity, :mass_continuity]
        @test Daisho.nobs(oset.groups[2]) == st.n_mish
        @test Daisho.nobs(oset.groups[1]) == 3 * st.n_mish
        obsset = Daisho.build_obsset(oset, st, stg)
        names = Daisho.variable_names(st)
        nm = st.n_mish
        S = Daisho.SplineProjection(stg, names)
        D = Daisho.DiagonalScaling(Union{Float64,Vector{Float64}}[amp, amp, amp], nm)
        # σ_b (`amp`) and the CG limits were tuned with the row-normalised
        # filter (see _solv_chain in test_variational_solver.jl).
        F = Daisho.MishFilter(stg, names, fill(1.0, 3, 3); normalisation = :row)
        cf = Daisho.IncrementalCostFunction(Daisho.ChainOp(S, D, F), obsset, zeros(size(S, 1)))
        t = @elapsed stats = Daisho.solve_incremental!(cf, Daisho.VarSolverOptions(max_inner = 1000, inner_tolerance = 1e-6))
        a = zeros(size(S, 1)); Daisho.analysis_coefficients!(a, cf)
        E = Daisho.SplineEvaluation(stg, names, [(1, 1), (2, 1), (3, 1)])
        mv = zeros(size(E, 1)); Daisho.apply!(mv, E, a)
        dx, dy, dz = Lx / 8, Ly / 8, Lz / 5
        errs = Float64[]
        for (v, f) in enumerate((uf, vf, wf))
            ana = reshape(mv[(v-1)*nm+1:v*nm], st.dims)
            num = 0.0; den = 0.0
            for i in 1:nx_, j in 1:ny_, kk in 1:nz_
                x, y, z = xs[i], ys[j], zs[kk]
                (dx < x < Lx - dx && dy < y < Ly - dy && dz < z < Lz - dz) || continue
                num += (ana[kk, j, i] - f(x, y, z))^2; den += f(x, y, z)^2
            end
            push!(errs, sqrt(num / den))
        end
        yh = zeros(Daisho.nobs(obsset)); Daisho.evaluate!(yh, obsset, a)
        cbg = Daisho.cost_by_group(oset, obsset, Daisho.obs_values(obsset) .- yh)
        @info "End-to-end (builders)" errs = repr(errs) cg = repr(stats.inner_iterations) Jo = repr(cbg) seconds = t
        @test stats.converged
        @test all(errs .< 2e-2)
        @test Set(keys(cbg)) == Set([:radial_velocity, :mass_continuity])
    end
end
