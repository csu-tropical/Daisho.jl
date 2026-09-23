using TOML

@testset "Variational config (Phase 6)" begin

    # Load the bundled template, apply `mutate!` to the parsed Dict, write it
    # to a temp file, and return the caught error (or `nothing` on success).
    function _var_err(mutate!)
        path = tempname() * ".toml"
        base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
        mutate!(base)
        open(path, "w") do io; TOML.print(io, base); end
        try
            DaishoParameters(path)
            return nothing
        catch e
            return e
        end
    end

    # Load the bundled template, apply `mutate!`, write it, and load it.
    function _var_load(mutate!)
        path = tempname() * ".toml"
        base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
        mutate!(base)
        open(path, "w") do io; TOML.print(io, base); end
        return DaishoParameters(path)
    end

    @testset "template presence / absence" begin
        p = DaishoParameters()
        @test :variational in p.provided
        @test p.variational.enabled == false
        @test p.variational.state == ["U", "V", "W"]
        @test p.variational.background == "zero"
        @test p.variational.mass_continuity == true
        @test p.variational.filter_type == "gaussian"
        @test haskey(p.variational.background_error, "default")
        @test haskey(p.variational.background_error, "W")

        # Block removed entirely -> defaults, not in provided. `VariationalParameters`
        # has no custom `==` (it holds Vector/Dict fields, so the default `==`
        # is identity-only) — compare the fields that matter instead.
        p2 = _var_load(base -> delete!(base, "variational"))
        @test !(:variational in p2.provided)
        d = VariationalParameters()
        @test p2.variational.enabled == d.enabled
        @test p2.variational.state == d.state
        @test p2.variational.background == d.background
        @test p2.variational.mass_continuity == d.mass_continuity
        @test p2.variational.filter_type == d.filter_type
        @test p2.variational.background_error == d.background_error
        @test p2.variational.radial_velocity == d.radial_velocity
        @test p2.variational.scalar_obs == d.scalar_obs
        @test p2.variational.roles == d.roles
    end

    @testset "full round trip" begin
        p = _var_load(base -> begin
            base["variational"] = Dict(
                "enabled" => true,
                "state" => ["U", "V", "W", "DBZ"],
                "background" => "synthesis",
                "background_error_inflation" => 2.0,
                "mass_continuity" => true,
                "mass_continuity_sigma_per_s" => 2.0e-4,
                "max_outer_iterations" => 3,
                "max_inner_iterations" => 150,
                "inner_tolerance" => 1.0e-5,
                "outer_tolerance" => 1.0e-2,
                "output_background" => true,
                "output_increment" => true,
                "output_mish" => true,
                "filter_type" => "lanczos",
                "background_error" => Dict(
                    "default" => Dict("sigma" => 10.0, "filter_length_m" => [2000.0, 2000.0, 500.0]),
                    "W"       => Dict("sigma" => 2.0,  "filter_length_m" => [1500.0, 1500.0, 0.0]),
                    "DBZ"     => Dict("sigma" => 5.0,  "filter_length_m" => [1000.0, 1000.0, 250.0]),
                ),
                "radial_velocity" => Dict(
                    "error_inflation" => 1.5,
                    "horizontal_approximation" => true,
                    "eigenvalue_guard_relative" => 1.0e-8,
                ),
                "scalar_obs" => Dict(
                    "DBZ" => Dict("sigma" => 5.0, "undetect_value" => -10.0),
                ),
                "roles" => Dict("W" => "wind_w"),
            )
        end)
        vp = p.variational
        @test :variational in p.provided
        @test vp.enabled == true
        @test vp.state == ["U", "V", "W", "DBZ"]
        @test vp.background == "synthesis"
        @test vp.background_error_inflation == 2.0
        @test vp.mass_continuity == true
        @test vp.mass_continuity_sigma_per_s == 2.0e-4
        @test vp.max_outer_iterations == 3
        @test vp.max_inner_iterations == 150
        @test vp.inner_tolerance == 1.0e-5
        @test vp.outer_tolerance == 1.0e-2
        @test vp.output_background == true
        @test vp.output_increment == true
        @test vp.output_mish == true
        @test vp.filter_type == "lanczos"
        @test vp.background_error["default"] == BackgroundErrorEntry(10.0, (2000.0, 2000.0, 500.0))
        @test vp.background_error["W"] == BackgroundErrorEntry(2.0, (1500.0, 1500.0, 0.0))
        @test vp.background_error["DBZ"] == BackgroundErrorEntry(5.0, (1000.0, 1000.0, 250.0))
        @test vp.radial_velocity == RadialVelocityObsParameters(1.5, true, 1.0e-8)
        @test vp.scalar_obs["DBZ"] == ScalarObsEntry(5.0, -10.0)
        @test vp.roles == Dict("W" => :wind_w)
    end

    @testset "background_error_for" begin
        p = DaishoParameters()
        @test Daisho.background_error_for(p.variational, "W") ==
            p.variational.background_error["W"]
        @test Daisho.background_error_for(p.variational, "U") ==
            p.variational.background_error["default"]

        vp = VariationalParameters(state = ["U", "V", "W"],
            background_error = Dict("W" => BackgroundErrorEntry(3.0, (1.0, 1.0, 1.0))))
        @test Daisho.background_error_for(vp, "W") == BackgroundErrorEntry(3.0, (1.0, 1.0, 1.0))
        err = try
            Daisho.background_error_for(vp, "U")
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("U", err.msg)
        @test occursin("default", err.msg)
    end

    @testset "validation: state" begin
        err = _var_err(base -> base["variational"]["state"] = String[])
        @test err isa ArgumentError
        @test occursin("state", err.msg) && occursin("non-empty", err.msg)

        err = _var_err(base -> base["variational"]["state"] = ["U", "V", "U"])
        @test err isa ArgumentError
        @test occursin("unique", err.msg)

        err = _var_err(base -> base["variational"]["state"] = ["V", "W"])
        @test err isa ArgumentError
        @test occursin("\"U\"", err.msg)

        err = _var_err(base -> base["variational"]["state"] = ["U", "W"])
        @test err isa ArgumentError
        @test occursin("\"V\"", err.msg)

        err = _var_err(base -> base["variational"]["state"] = ["U", "V", "W", "BOGUS"])
        @test err isa ArgumentError
        @test occursin("BOGUS", err.msg)
        @test occursin("[fields]", err.msg)
    end

    @testset "validation: mass_continuity requires W" begin
        err = _var_err(base -> begin
            base["variational"]["state"] = ["U", "V"]
            base["variational"]["mass_continuity"] = true
        end)
        @test err isa ArgumentError
        @test occursin("mass_continuity", err.msg) && occursin("\"W\"", err.msg)
    end

    @testset "validation: scalar_obs coverage" begin
        # Field in state with no scalar_obs entry.
        err = _var_err(base -> base["variational"]["state"] = ["U", "V", "W", "DBZ"])
        @test err isa ArgumentError
        @test occursin("DBZ", err.msg) && occursin("scalar_obs", err.msg)

        # scalar_obs entry for a name not in state.
        err = _var_err(base -> begin
            base["variational"]["scalar_obs"] = Dict(
                "DBZ" => Dict("sigma" => 5.0, "undetect_value" => -10.0))
        end)
        @test err isa ArgumentError
        @test occursin("DBZ", err.msg)

        # scalar_obs sigma must be positive.
        err = _var_err(base -> begin
            base["variational"]["state"] = ["U", "V", "W", "DBZ"]
            base["variational"]["scalar_obs"] = Dict(
                "DBZ" => Dict("sigma" => 0.0, "undetect_value" => -10.0))
        end)
        @test err isa ArgumentError
        @test occursin("sigma", err.msg) && occursin("positive", err.msg)

        # Valid round trip.
        p = _var_load(base -> begin
            base["variational"]["state"] = ["U", "V", "W", "DBZ"]
            base["variational"]["scalar_obs"] = Dict(
                "DBZ" => Dict("sigma" => 5.0, "undetect_value" => -10.0))
        end)
        @test p.variational.scalar_obs["DBZ"] == ScalarObsEntry(5.0, -10.0)
    end

    @testset "validation: background_error resolution" begin
        err = _var_err(base -> begin
            base["variational"]["background_error"] = Dict(
                "W" => Dict("sigma" => 2.0, "filter_length_m" => [2000.0, 2000.0, 500.0]))
        end)
        @test err isa ArgumentError
        @test occursin("\"U\"", err.msg) || occursin("U", err.msg)
        @test occursin("background_error", err.msg)

        # Unknown key naming a non-state, non-default entry.
        err = _var_err(base -> begin
            base["variational"]["background_error"]["BOGUS"] =
                Dict("sigma" => 1.0, "filter_length_m" => [1.0, 1.0, 1.0])
        end)
        @test err isa ArgumentError
        @test occursin("BOGUS", err.msg)
        @test occursin("[variational.background_error]", err.msg)

        # sigma must be positive.
        err = _var_err(base -> begin
            base["variational"]["background_error"]["default"]["sigma"] = 0.0
        end)
        @test err isa ArgumentError
        @test occursin("sigma", err.msg) && occursin("positive", err.msg)

        # filter_length_m wrong length.
        err = _var_err(base -> begin
            base["variational"]["background_error"]["default"]["filter_length_m"] = [1.0, 1.0]
        end)
        @test err isa ArgumentError
        @test occursin("filter_length_m", err.msg)

        # filter_length_m negative entry.
        err = _var_err(base -> begin
            base["variational"]["background_error"]["default"]["filter_length_m"] = [-1.0, 1.0, 1.0]
        end)
        @test err isa ArgumentError
        @test occursin("filter_length_m", err.msg)
        @test occursin("non-negative", err.msg)

        # Unknown key inside a background_error entry table.
        err = _var_err(base -> begin
            base["variational"]["background_error"]["default"]["bogus"] = 1.0
        end)
        @test err isa ArgumentError
        @test occursin("[variational.background_error.default]", err.msg)
    end

    @testset "validation: background / filter_type" begin
        err = _var_err(base -> base["variational"]["background"] = "bogus")
        @test err isa ArgumentError
        @test occursin("background", err.msg)

        err = _var_err(base -> base["variational"]["filter_type"] = "bogus")
        @test err isa ArgumentError
        @test occursin("filter_type", err.msg)

        p = _var_load(base -> base["variational"]["filter_type"] = "lanczos")
        @test p.variational.filter_type == "lanczos"
    end

    @testset "validation: scalar guards" begin
        err = _var_err(base -> base["variational"]["background_error_inflation"] = 0.0)
        @test err isa ArgumentError
        @test occursin("background_error_inflation", err.msg)

        err = _var_err(base -> base["variational"]["mass_continuity_sigma_per_s"] = 0.0)
        @test err isa ArgumentError
        @test occursin("mass_continuity_sigma_per_s", err.msg)

        err = _var_err(base -> base["variational"]["max_outer_iterations"] = 0)
        @test err isa ArgumentError
        @test occursin("max_outer_iterations", err.msg)

        err = _var_err(base -> base["variational"]["max_inner_iterations"] = 0)
        @test err isa ArgumentError
        @test occursin("max_inner_iterations", err.msg)

        err = _var_err(base -> base["variational"]["inner_tolerance"] = 0.0)
        @test err isa ArgumentError
        @test occursin("inner_tolerance", err.msg)

        err = _var_err(base -> base["variational"]["inner_tolerance"] = 1.0)
        @test err isa ArgumentError
        @test occursin("inner_tolerance", err.msg)

        err = _var_err(base -> base["variational"]["outer_tolerance"] = 0.0)
        @test err isa ArgumentError
        @test occursin("outer_tolerance", err.msg)

        err = _var_err(base -> base["variational"]["outer_tolerance"] = 1.0)
        @test err isa ArgumentError
        @test occursin("outer_tolerance", err.msg)
    end

    @testset "validation: radial_velocity" begin
        err = _var_err(base -> begin
            base["variational"]["radial_velocity"]["error_inflation"] = 0.0
        end)
        @test err isa ArgumentError
        @test occursin("error_inflation", err.msg)

        err = _var_err(base -> begin
            base["variational"]["radial_velocity"]["eigenvalue_guard_relative"] = 1.0
        end)
        @test err isa ArgumentError
        @test occursin("eigenvalue_guard_relative", err.msg)

        err = _var_err(base -> begin
            base["variational"]["radial_velocity"]["eigenvalue_guard_relative"] = -0.1
        end)
        @test err isa ArgumentError
        @test occursin("eigenvalue_guard_relative", err.msg)

        # 0.0 is allowed (boundary of [0, 1)).
        p = _var_load(base -> begin
            base["variational"]["radial_velocity"]["eigenvalue_guard_relative"] = 0.0
        end)
        @test p.variational.radial_velocity.eigenvalue_guard_relative == 0.0

        # Unknown key.
        err = _var_err(base -> begin
            base["variational"]["radial_velocity"]["bogus"] = 1.0
        end)
        @test err isa ArgumentError
        @test occursin("[variational.radial_velocity]", err.msg)
    end

    @testset "validation: roles" begin
        err = _var_err(base -> base["variational"]["roles"] = Dict("BOGUS" => "wind_w"))
        @test err isa ArgumentError
        @test occursin("BOGUS", err.msg)
        @test occursin("[variational.roles]", err.msg)

        err = _var_err(base -> base["variational"]["roles"] = Dict("W" => "not_a_role"))
        @test err isa ArgumentError
        @test occursin("not_a_role", err.msg)

        p = _var_load(base -> base["variational"]["roles"] = Dict("W" => "wind_w"))
        @test p.variational.roles == Dict("W" => :wind_w)
    end

    @testset "validation: unknown top-level key" begin
        err = _var_err(base -> base["variational"]["bogus"] = 1.0)
        @test err isa ArgumentError
        @test occursin("[variational]", err.msg)
        @test occursin("bogus", err.msg)
    end

    @testset "validation: scalar_obs unknown key in entry table" begin
        err = _var_err(base -> begin
            base["variational"]["state"] = ["U", "V", "W", "DBZ"]
            base["variational"]["scalar_obs"] = Dict(
                "DBZ" => Dict("sigma" => 5.0, "undetect_value" => -10.0, "bogus" => 1.0))
        end)
        @test err isa ArgumentError
        @test occursin("[variational.scalar_obs.DBZ]", err.msg)
    end

    @testset "samurai migration keys" begin
        err = _var_err(base -> base["variational"]["mc_weight"] = 1.0e-4)
        @test err isa ArgumentError
        @test occursin("mc_weight", err.msg)
        @test occursin("mass_continuity_sigma_per_s", err.msg)

        err = _var_err(base -> base["variational"]["num_iterations"] = 5)
        @test err isa ArgumentError
        @test occursin("num_iterations", err.msg)
        @test occursin("max_outer_iterations", err.msg)

        err = _var_err(base -> begin
            base["variational"]["background_error"]["bg_rhou_error"] = 1.0
        end)
        @test err isa ArgumentError
        @test occursin("bg_rhou_error", err.msg)

        err = _var_err(base -> begin
            base["variational"]["background_error"]["bg_rhov_error"] = 1.0
        end)
        @test err isa ArgumentError
        @test occursin("bg_rhov_error", err.msg)

        err = _var_err(base -> begin
            base["variational"]["background_error"]["bg_rhow_error"] = 1.0
        end)
        @test err isa ArgumentError
        @test occursin("bg_rhow_error", err.msg)

        err = _var_err(base -> begin
            base["variational"]["background_error"]["i_filter_length"] = 2000.0
        end)
        @test err isa ArgumentError
        @test occursin("i_filter_length", err.msg)
        @test occursin("filter_length_m", err.msg)

        err = _var_err(base -> begin
            base["variational"]["background_error"]["j_filter_length"] = 2000.0
        end)
        @test err isa ArgumentError
        @test occursin("j_filter_length", err.msg)

        err = _var_err(base -> begin
            base["variational"]["background_error"]["k_filter_length"] = 500.0
        end)
        @test err isa ArgumentError
        @test occursin("k_filter_length", err.msg)

        err = _var_err(base -> begin
            base["variational"]["radial_velocity"]["horizontal_radar_appx"] = true
        end)
        @test err isa ArgumentError
        @test occursin("horizontal_radar_appx", err.msg)
        @test occursin("horizontal_approximation", err.msg)
    end

    @testset "[grid.springsteel.k.bc] state-variable BC override" begin
        # Accepted: "W" is a state variable (default template state).
        p = _var_load(base -> begin
            base["grid"]["springsteel"]["k"]["bc"]["W"] = Dict("min" => "dirichlet")
        end)
        @test "W" in p.variational.state
        @test p.grid.springsteel.k.bc_min["W"] == DirichletBC(0.0)
        sgrid = Daisho.create_radar_grid(p)
        @test sgrid !== nothing

        # Rejected: "W" is neither a [fields] name nor a state variable.
        p2 = _var_load(base -> begin
            base["variational"]["state"] = ["U", "V"]
            base["variational"]["mass_continuity"] = false
            base["variational"]["background_error"] = Dict(
                "default" => Dict("sigma" => 10.0, "filter_length_m" => [2000.0, 2000.0, 500.0]))
            base["grid"]["springsteel"]["k"]["bc"]["W"] = Dict("min" => "dirichlet")
        end)
        @test !("W" in p2.variational.state)
        err2 = try
            Daisho.create_radar_grid(p2)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("W", err2.msg)
        @test occursin("does not match any configured", err2.msg)
    end
end
