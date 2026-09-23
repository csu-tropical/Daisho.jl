using TOML

@testset "Reference density (Phase 1)" begin

    @testset "ExponentialReferenceProfile" begin
        ρ0 = 1.2
        H = 8500.0

        # No temperature configured (NaN sentinel).
        r = ExponentialReferenceProfile(ρ0, H, NaN, 0.0065)
        @test reference_density(r, 0.0) == ρ0
        @test reference_density(r, H) ≈ ρ0 / exp(1)
        for z in (0.0, 250.0, 1000.0, 5000.0, 12000.0)
            @test dlnrho_dz(r, z) == -1.0 / H
        end
        @test reference_temperature_kelvin(r, 0.0) === nothing
        @test freezing_level_m(r) === nothing
        err = try
            temperature_celsius(r, 0.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("temperature", err.msg)

        # With a temperature lapse configured.
        T0 = 300.0
        Γ = 0.0065
        r2 = ExponentialReferenceProfile(ρ0, H, T0, Γ)
        @test reference_temperature_kelvin(r2, 0.0) == T0
        @test reference_temperature_kelvin(r2, 1000.0) ≈ T0 - Γ * 1000.0
        expected_fl = (T0 - 273.15) / Γ
        @test freezing_level_m(r2) ≈ expected_fl
        @test temperature_celsius(r2, 0.0) ≈ T0 - 273.15
        @test temperature_celsius(r2, expected_fl) ≈ 0.0 atol = 1e-8

        # Zero lapse rate: never crosses freezing (guarded, not a division by zero).
        r3 = ExponentialReferenceProfile(ρ0, H, T0, 0.0)
        @test freezing_level_m(r3) === nothing
    end

    @testset "SoundingReferenceProfile" begin
        sgrid = Daisho.create_radar_grid("RRR", Dict("DBZ" => 1);
            xmin=-5000.0, xmax=5000.0, xdim=4,
            ymin=-5000.0, ymax=5000.0, ydim=4,
            zmin=0.0, zmax=6000.0, zdim=6)

        # samurai/Scythe sounding format: surface line "p_sfc theta q_v" (hPa, K,
        # g/kg), then "z theta q_v" per level (z metres, q_v g/kg). Levels run well
        # above the grid top (6000 m) to 15 km; theta increases and q_v decreases
        # with height; the profile is built to cross 0 degC (273.15 K) around 3 km.
        path = tempname()
        open(path, "w") do io
            println(io, "1010.0 300.0 12.0")
            zs     = [500.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0, 9000.0, 12000.0, 15000.0]
            thetas = [300.5, 301.0,  302.0,  303.0,  304.0,  306.0,  311.0,  318.0,   330.0]
            qvs    = [10.0,  8.0,    5.0,    3.0,    2.0,    1.0,    0.3,    0.05,    0.01]
            for i in eachindex(zs)
                println(io, "$(zs[i]) $(thetas[i]) $(qvs[i])")
            end
        end

        prof = SoundingReferenceProfile(path, sgrid; moisture=true)

        n = length(prof.z)
        @test n == length(prof.rho) == length(prof.drho_dz) == length(prof.temperature_k)
        @test n > 0

        # Density positive and monotonically decreasing with height.
        @test all(prof.rho .> 0.0)
        @test all(diff(prof.rho) .< 0.0)

        # dlnrho_dz negative everywhere.
        @test all(dlnrho_dz(prof, z) < 0.0 for z in prof.z)

        # Stored slot-2 derivative matches a centered finite difference of rho on
        # the mish to within 5% in the interior (spectrally consistent, never
        # finite-differenced itself — this is just a cross-check).
        for i in 2:(n - 1)
            fd = (prof.rho[i+1] - prof.rho[i-1]) / (prof.z[i+1] - prof.z[i-1])
            @test isapprox(prof.drho_dz[i], fd; rtol=0.05)
        end

        # Exact at mish points.
        for i in 1:n
            @test reference_density(prof, prof.z[i]) == prof.rho[i]
        end

        # Hermite sampling strictly between two mish levels lies between the
        # neighbouring stored values.
        for i in 1:(n - 1)
            zm = 0.5 * (prof.z[i] + prof.z[i+1])
            val = reference_density(prof, zm)
            lo, hi = minmax(prof.rho[i], prof.rho[i+1])
            @test lo <= val <= hi
        end

        # Clamped outside the mish range.
        @test reference_density(prof, prof.z[1] - 500.0) == prof.rho[1]
        @test reference_density(prof, prof.z[end] + 500.0) == prof.rho[end]

        # freezing_level_m is bracketed by the two (mish) levels whose stored
        # temperature straddles 0 degC.
        fl = freezing_level_m(prof)
        @test fl !== nothing
        idx = findfirst(i -> (prof.temperature_k[i] - 273.15) *
                              (prof.temperature_k[i+1] - 273.15) <= 0.0, 1:(n - 1))
        @test idx !== nothing
        @test prof.z[idx] <= fl <= prof.z[idx+1]

        # temperature_celsius agrees with reference_temperature_kelvin and, at
        # z <= z[1] (clamped), with the surface-most stored mish temperature.
        @test temperature_celsius(prof, prof.z[1]) ≈ reference_temperature_kelvin(prof, prof.z[1]) - 273.15
        @test temperature_celsius(prof, 0.0) ≈ prof.temperature_k[1] - 273.15

        # Dry build (moisture=false): rho == ref_rho_d only, no vapor contribution.
        prof_dry = SoundingReferenceProfile(path, sgrid; moisture=false)
        @test all(prof_dry.rho .> 0.0)
        @test all(diff(prof_dry.rho) .< 0.0)
        # Moist density includes the vapor partial density, so it should exceed
        # the dry density at the same (matching) mish points.
        @test all(prof.rho .>= prof_dry.rho)

        rm(path; force=true)
    end

    @testset "[reference_state] config" begin
        # The bundled template now ships a [reference_state] block (see
        # config/defaults.toml), so DaishoParameters() has it in `provided`.
        p = DaishoParameters()
        @test :reference_state in p.provided
        @test p.reference_state.source == "exponential"
        @test build_reference_profile(p, nothing) isa ExponentialReferenceProfile

        # Absent from a user file -> default exponential, not in provided.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            delete!(base, "reference_state")
            open(path, "w") do f; TOML.print(f, base); end
            p0 = DaishoParameters(path)
            @test :reference_state ∉ p0.provided
            @test p0.reference_state.source == "exponential"
            @test p0.reference_state.density_surface_kg_m3 == ReferenceStateParameters().density_surface_kg_m3
            @test build_reference_profile(p0, nothing) isa ExponentialReferenceProfile
        end

        # Present block round-trips (exponential values).
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["reference_state"] = Dict(
                "source" => "exponential",
                "density_surface_kg_m3" => 1.3,
                "density_scale_height_m" => 9000.0,
                "temperature_surface_k" => 295.0,
                "lapse_rate_k_m" => 0.006,
            )
            open(path, "w") do f; TOML.print(f, base); end
            p2 = DaishoParameters(path)
            @test :reference_state in p2.provided
            @test p2.reference_state.source == "exponential"
            @test p2.reference_state.density_surface_kg_m3 == 1.3
            @test p2.reference_state.density_scale_height_m == 9000.0
            @test p2.reference_state.temperature_surface_k == 295.0
            @test p2.reference_state.lapse_rate_k_m == 0.006
            @test build_reference_profile(p2, nothing) isa ExponentialReferenceProfile
        end

        # Round trip with a sounding source + moisture flag.
        sounding_path = tempname()
        open(sounding_path, "w") do sio
            println(sio, "1010.0 300.0 12.0")
            for zi in 500.0:500.0:15000.0
                println(sio, "$(zi) $(300.0 + 0.005 * zi) 5.0")
            end
        end
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["reference_state"] = Dict(
                "source" => "sounding",
                "sounding_file" => sounding_path,
                "moisture" => false,
            )
            open(path, "w") do f; TOML.print(f, base); end
            p3 = DaishoParameters(path)
            @test p3.reference_state.source == "sounding"
            @test p3.reference_state.sounding_file == sounding_path
            @test p3.reference_state.moisture == false

            sgrid = Daisho.create_radar_grid("RRR", Dict("DBZ" => 1);
                xmin=-5000.0, xmax=5000.0, xdim=4,
                ymin=-5000.0, ymax=5000.0, ydim=4,
                zmin=0.0, zmax=6000.0, zdim=6)
            prof = build_reference_profile(p3, sgrid)
            @test prof isa SoundingReferenceProfile
        end
        rm(sounding_path; force=true)

        # Unknown key -> ArgumentError naming [reference_state].
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["reference_state"] = Dict("source" => "exponential", "bogus" => 1.0)
            open(path, "w") do f; TOML.print(f, base); end
            err = try
                DaishoParameters(path)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("[reference_state]", err.msg)
            @test occursin("bogus", err.msg)
        end

        # Bad source -> error.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["reference_state"] = Dict("source" => "not_a_real_source")
            open(path, "w") do f; TOML.print(f, base); end
            @test_throws ArgumentError DaishoParameters(path)
        end

        # source = "sounding" with no/missing file -> error.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["reference_state"] = Dict("source" => "sounding")
            open(path, "w") do f; TOML.print(f, base); end
            @test_throws ArgumentError DaishoParameters(path)
        end
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["reference_state"] = Dict("source" => "sounding",
                                           "sounding_file" => "/no/such/sounding.txt")
            open(path, "w") do f; TOML.print(f, base); end
            @test_throws ArgumentError DaishoParameters(path)
        end

        # Non-positive density guards.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["reference_state"] = Dict("source" => "exponential",
                                           "density_surface_kg_m3" => 0.0)
            open(path, "w") do f; TOML.print(f, base); end
            @test_throws ArgumentError DaishoParameters(path)
        end
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["reference_state"] = Dict("source" => "exponential",
                                           "density_scale_height_m" => -1.0)
            open(path, "w") do f; TOML.print(f, base); end
            @test_throws ArgumentError DaishoParameters(path)
        end
    end
end
