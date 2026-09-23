# Tests for the variational state vector (Phase 7): role resolution, layout,
# background-error/BC resolution, and the derived state grid.

using TOML
using Springsteel
using Springsteel: CubicBSpline

# Stub non-identity output frame, mirroring `_RotatedFrame` in
# test_wind_synthesis.jl:5-11, for exercising frame-based role resolution
# independently of the config loader's hard "U"/"V" name requirement.
struct _StateFrame <: Daisho.SynthesisFrame end
Daisho.component_names(::_StateFrame) = ("C1", "C2")
Daisho.rotation_at(::_StateFrame, x::Real, y::Real, z::Real) = [1.0 0.0; 0.0 1.0]

@testset "Variational state vector (Phase 7)" begin

    # Load the bundled template with a small [grid.springsteel] (4x3x3 cells)
    # and a working [variational] block (state = U, V, W, DBZ, WIDTH; DBZ
    # carries the `reflectivity` tag in the template, WIDTH does not), then
    # apply `mutate!` for per-test tweaks.
    function _state_load(mutate!)
        path = tempname() * ".toml"
        base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
        base["grid"]["springsteel"]["i"]["cells"] = 4
        base["grid"]["springsteel"]["j"]["cells"] = 3
        base["grid"]["springsteel"]["k"]["cells"] = 3
        base["variational"] = Dict(
            "enabled" => true,
            "state" => ["U", "V", "W", "DBZ", "WIDTH"],
            "scalar_obs" => Dict(
                "DBZ"   => Dict("sigma" => 5.0, "undetect_value" => -10.0),
                "WIDTH" => Dict("sigma" => 1.0, "undetect_value" => 0.0),
            ),
            "background_error" => Dict(
                "default" => Dict("sigma" => 10.0, "filter_length_m" => [2000.0, 2000.0, 500.0]),
                "W"       => Dict("sigma" => 2.0,  "filter_length_m" => [2000.0, 2000.0, 500.0]),
            ),
        )
        base["grid"]["springsteel"]["k"]["bc"]["W"] = Dict("min" => "dirichlet")
        mutate!(base)
        open(path, "w") do io; TOML.print(io, base); end
        return DaishoParameters(path)
    end

    p = _state_load(identity)
    sgrid = Daisho.create_radar_grid(p)
    s = Daisho.StateVector(p, sgrid)

    @testset "role resolution" begin
        @test s.vars[s.index["U"]].role == :wind_u
        @test s.vars[s.index["V"]].role == :wind_v
        @test s.vars[s.index["W"]].role == :wind_w
        @test s.vars[s.index["DBZ"]].role == :reflectivity   # reflectivity-tagged
        @test s.vars[s.index["WIDTH"]].role == :scalar       # no reflectivity tag

        # Explicit override wins over automatic reflectivity-tag detection.
        p_over = _state_load(base -> begin
            base["variational"]["roles"] = Dict("DBZ" => "scalar")
        end)
        s_over = Daisho.StateVector(p_over, Daisho.create_radar_grid(p_over))
        @test s_over.vars[s_over.index["DBZ"]].role == :scalar

        # Duplicate singular role -> ArgumentError (DBZ overridden onto W's
        # `wind_w` role; the config loader only checks role-name membership,
        # not uniqueness across state variables).
        p_dup = _state_load(base -> begin
            base["variational"]["roles"] = Dict("DBZ" => "wind_w")
        end)
        @test_throws ArgumentError Daisho.StateVector(p_dup, Daisho.create_radar_grid(p_dup))

        # Unknown name (frame mismatch): the config loader hard-requires
        # literal "U"/"V" in `state` regardless of frame, so build
        # `VariationalParameters` directly to exercise `StateVector`'s own
        # frame-based resolution and its failure mode.
        fields = Daisho.FieldSpec[]
        moments = MomentParameters(fields)
        cfg2 = SpringsteelGridConfig(geometry = "RRR", mubar = 3, quadrature = :gauss,
            i = SpringsteelAxisConfig(min = -1000.0, max = 1000.0, cells = 2),
            j = SpringsteelAxisConfig(min = -1000.0, max = 1000.0, cells = 2),
            k = SpringsteelAxisConfig(min = 0.0, max = 1000.0, cells = 2))
        grid2 = GridParameters(springsteel = cfg2)
        vp_bad = VariationalParameters(enabled = true, state = ["U", "V"],
            mass_continuity = false,
            background_error = Dict("default" => BackgroundErrorEntry(10.0, (2000.0, 2000.0, 500.0))))
        p_frame = DaishoParameters(moments, QCParameters(), GriddingParameters(), grid2,
            IOParameters(), SynthesisParameters(), EchoProductsParameters(),
            HybridScanParameters(), ReferenceStateParameters(), vp_bad, Set{Symbol}())
        sgrid_frame = Daisho.create_radar_grid(cfg2, Dict("U" => 1, "V" => 2))
        @test_throws ArgumentError Daisho.StateVector(p_frame, sgrid_frame; frame = _StateFrame())

        # The frame's own component names resolve fine.
        vp_ok = VariationalParameters(enabled = true, state = ["C1", "C2"],
            mass_continuity = false,
            background_error = Dict("default" => BackgroundErrorEntry(10.0, (2000.0, 2000.0, 500.0))))
        p_frame2 = DaishoParameters(moments, QCParameters(), GriddingParameters(), grid2,
            IOParameters(), SynthesisParameters(), EchoProductsParameters(),
            HybridScanParameters(), ReferenceStateParameters(), vp_ok, Set{Symbol}())
        sgrid_frame2 = Daisho.create_radar_grid(cfg2, Dict("C1" => 1, "C2" => 2))
        s_frame = Daisho.StateVector(p_frame2, sgrid_frame2; frame = _StateFrame())
        @test s_frame.vars[s_frame.index["C1"]].role == :wind_u
        @test s_frame.vars[s_frame.index["C2"]].role == :wind_v
    end

    @testset "source / units / scalar_obs / background_error" begin
        dbz = s.vars[s.index["DBZ"]]
        @test dbz.source == :gridded_field
        @test dbz.units == "dBZ"
        @test dbz.obs_sigma == 5.0
        @test dbz.undetect_value == -10.0

        width = s.vars[s.index["WIDTH"]]
        @test width.source == :gridded_field
        @test width.units == "1"
        @test width.obs_sigma == 1.0
        @test width.undetect_value == 0.0

        u = s.vars[s.index["U"]]
        @test u.source == :retrieved
        @test u.units == "m s-1"
        @test isnan(u.obs_sigma)
        @test isnan(u.undetect_value)

        w = s.vars[s.index["W"]]
        @test w.background_error isa Daisho.UniformSigma
        @test w.background_error.sigma == 2.0
        @test w.filter_length_m == (2000.0, 2000.0, 500.0)
        @test u.background_error isa Daisho.UniformSigma
        @test u.background_error.sigma == 10.0
        @test dbz.background_error.sigma == 10.0

        M = Daisho.filter_lengths_m(s)
        @test size(M) == (3, Daisho.nvars(s))
        vW = s.index["W"]
        # config order (i, j, k) = (2000, 2000, 500); filter_lengths_m rows
        # are (k, j, i).
        @test M[:, vW] == [500.0, 2000.0, 2000.0]
    end

    @testset "layout" begin
        @test s.dims == (sgrid.params.kDim, sgrid.params.jDim, sgrid.params.iDim)
        @test Daisho.nvars(s) == 5
        @test s.n_mish == prod(s.dims)
        @test length(s) == Daisho.nvars(s) * s.n_mish
        @test Daisho.variable_names(s) == ["U", "V", "W", "DBZ", "WIDTH"]
        @test Daisho.state_index_dict(s) == s.index

        x = collect(1.0:length(s))
        vW = s.index["W"]
        view_w = Daisho.state_view(x, s, "W")
        @test size(view_w) == s.dims
        @test view_w === Daisho.state_view(x, s, vW)
        @test view_w === Daisho.state_view(x, s, :wind_w)

        for (k, j, i) in ((1, 1, 1), (2, 3, 4), (s.dims[1], s.dims[2], s.dims[3]))
            flat = Daisho.daisho_to_springsteel_index(i, j, k, s.dims[2], s.dims[1])
            @test view_w[k, j, i] == x[s.offsets[vW] + flat]
        end
    end

    @testset "wind_indices / has_role / role_index" begin
        iu, iv, iw = Daisho.wind_indices(s)
        @test iu == s.index["U"]
        @test iv == s.index["V"]
        @test iw == s.index["W"]
        @test Daisho.has_role(s, :wind_u)
        @test Daisho.has_role(s, :wind_w)
        @test Daisho.has_role(s, :reflectivity)
        @test Daisho.role_index(s, :reflectivity) == s.index["DBZ"]
        @test !Daisho.has_role(s, :scalar)   # :scalar is not singular; never tracked
        @test_throws ArgumentError Daisho.role_index(s, :scalar)

        # No `:wind_w` -> `wind_indices` third element is `nothing`.
        p_nomc = _state_load(base -> begin
            base["variational"]["state"] = ["U", "V", "DBZ"]
            base["variational"]["mass_continuity"] = false
            base["variational"]["background_error"] = Dict(
                "default" => Dict("sigma" => 10.0, "filter_length_m" => [2000.0, 2000.0, 500.0]))
            base["variational"]["scalar_obs"] = Dict(
                "DBZ" => Dict("sigma" => 5.0, "undetect_value" => -10.0))
            delete!(base["grid"]["springsteel"]["k"]["bc"], "W")
        end)
        s_nomc = Daisho.StateVector(p_nomc, Daisho.create_radar_grid(p_nomc))
        @test !Daisho.has_role(s_nomc, :wind_w)
        @test Daisho.wind_indices(s_nomc)[3] === nothing

        # mass_continuity = true with no wind_w role -> ArgumentError (role,
        # not name — the config loader already blocks `mass_continuity=true`
        # without the literal name "W"; this is the StateVector-level guard
        # for the same behaviour under a `[variational.roles]` override away
        # from `wind_w`).
        p_mc_bad = _state_load(base -> begin
            base["variational"]["roles"] = Dict("W" => "scalar")
        end)
        @test_throws ArgumentError Daisho.StateVector(p_mc_bad, Daisho.create_radar_grid(p_mc_bad))
    end

    @testset "background_sigma_vector" begin
        bsv = Daisho.background_sigma_vector(s)
        @test length(bsv) == length(s)
        for name in ("U", "V", "DBZ", "WIDTH")
            v = s.index[name]
            rng = (s.offsets[v] + 1):(s.offsets[v] + s.n_mish)
            @test all(==(10.0), bsv[rng])
        end
        vW = s.index["W"]
        rngW = (s.offsets[vW] + 1):(s.offsets[vW] + s.n_mish)
        @test all(==(2.0), bsv[rngW])
    end

    @testset "BC resolution" begin
        vW = s.index["W"]
        w = s.vars[vW]
        @test w.bc_min[3] == DirichletBC(0.0)   # k axis is index 3 of (i, j, k)
        @test w.bc_max[3] == NaturalBC()
        @test w.bc_min[1] == NaturalBC()
        @test w.bc_min[2] == NaturalBC()

        u = s.vars[s.index["U"]]
        @test u.bc_min[3] == NaturalBC()
        @test u.bc_max[3] == NaturalBC()

        # Inhomogeneous BC (rank-3, ahat/R3X) is refused. Not reachable via
        # TOML (`_parse_bc`'s vocabulary never produces a rank-3
        # inhomogeneous BC — only `zeros`, which is homogeneous), so build a
        # `VariationalParameters`/`SpringsteelGridConfig` directly.
        bad_bc = Springsteel.BoundaryConditions(1.0, 2.0, 3.0, nothing)   # rank 3, inhomogeneous
        cfg_bad = SpringsteelGridConfig(geometry = "RRR", mubar = 3, quadrature = :gauss,
            i = SpringsteelAxisConfig(min = -1000.0, max = 1000.0, cells = 2),
            j = SpringsteelAxisConfig(min = -1000.0, max = 1000.0, cells = 2),
            k = SpringsteelAxisConfig(min = 0.0, max = 1000.0, cells = 2,
                bc_min = Dict{String,BoundaryConditions}("default" => NaturalBC(), "W" => bad_bc)))
        vp_bad = VariationalParameters(enabled = true, state = ["U", "V", "W"],
            background_error = Dict("default" => BackgroundErrorEntry(10.0, (2000.0, 2000.0, 500.0))))
        grid_bad = GridParameters(springsteel = cfg_bad)
        p_bad = DaishoParameters(MomentParameters(Daisho.FieldSpec[]), QCParameters(),
            GriddingParameters(), grid_bad, IOParameters(), SynthesisParameters(),
            EchoProductsParameters(), HybridScanParameters(), ReferenceStateParameters(),
            vp_bad, Set{Symbol}())
        sgrid_bad = Daisho.create_radar_grid(cfg_bad, Dict("U" => 1, "V" => 2, "W" => 3))
        err = try
            Daisho.StateVector(p_bad, sgrid_bad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("ahat", err.msg)
    end

    @testset "create_state_grid" begin
        state_grid = Daisho.create_state_grid(p, sgrid, s)

        @test state_grid.params.iDim == sgrid.params.iDim
        @test state_grid.params.jDim == sgrid.params.jDim
        @test state_grid.params.kDim == sgrid.params.kDim
        @test state_grid.params.b_iDim == sgrid.params.b_iDim
        @test state_grid.params.b_jDim == sgrid.params.b_jDim
        @test state_grid.params.b_kDim == sgrid.params.b_kDim
        @test state_grid.params.i_regular_out == sgrid.params.i_regular_out
        @test state_grid.params.j_regular_out == sgrid.params.j_regular_out
        @test state_grid.params.k_regular_out == sgrid.params.k_regular_out
        @test isempty(state_grid.params.spline_filter)
        @test state_grid.params.vars == Daisho.state_index_dict(s)

        @test Vector{Float64}(state_grid.kbasis.data[1, 1, 1].mishPoints) == s.z_mish
        @test Vector{Float64}(state_grid.ibasis.data[1, 1, 1].mishPoints) ==
              Vector{Float64}(sgrid.ibasis.data[1, 1, 1].mishPoints)
        @test Vector{Float64}(state_grid.jbasis.data[1, 1, 1].mishPoints) ==
              Vector{Float64}(sgrid.jbasis.data[1, 1, 1].mishPoints)

        vW = state_grid.params.vars["W"]
        @test state_grid.kbasis.data[1, 1, vW].params.BCL == CubicBSpline.R1T0
        @test state_grid.kbasis.data[1, 1, vW].params.BCR == CubicBSpline.R0
        vU = state_grid.params.vars["U"]
        @test state_grid.kbasis.data[1, 1, vU].params.BCL == CubicBSpline.R0

        # Also exercises the `AxisSpline`/`_has_ahat` refusal path (Phase 2):
        # since every resolved BC here is homogeneous, construction succeeds.
        proj = Daisho.SplineProjection(state_grid, Daisho.variable_names(s))
        ncoef_per_var = state_grid.params.b_kDim * state_grid.params.b_jDim * state_grid.params.b_iDim
        @test size(proj) == (Daisho.nvars(s) * ncoef_per_var, length(s))
    end

    @testset "IdentityTransform round trip" begin
        t = Daisho.IdentityTransform()
        @test Daisho.to_physical(t, 3.5, 1234.0) == 3.5
        @test Daisho.to_control(t, 3.5, 1234.0) == 3.5
        @test Daisho.d_physical_d_control(t, 3.5, 1234.0) == 1.0
    end

    @testset "z_mish sanity" begin
        @test issorted(s.z_mish)
        @test length(s.z_mish) == s.dims[1]
        @test s.z_mish == Vector{Float64}(sgrid.kbasis.data[1, 1, 1].mishPoints)
    end

    @testset "BackgroundErrorSpec" begin
        us = Daisho.UniformSigma(2.5)
        @test_throws ArgumentError Daisho.UniformSigma(0.0)
        @test_throws ArgumentError Daisho.UniformSigma(-1.0)
        dims = (2, 3, 4)
        @test Daisho.sigma_field(us, dims) == fill(2.5, dims)
        @test Daisho.sigma_vector(us, dims) == vec(fill(2.5, dims))

        fs = Daisho.FieldSigma(fill(1.5, dims))
        @test Daisho.sigma_field(fs, dims) === fs.sigma
        @test_throws DimensionMismatch Daisho.sigma_field(fs, (2, 3, 5))
        @test_throws ArgumentError Daisho.FieldSigma(fill(-1.0, dims))
    end
end
