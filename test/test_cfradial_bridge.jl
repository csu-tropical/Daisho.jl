@testset "CfRadial bridge layer" begin
    @testset "synthetic Volume → legacy radar → Volume round trip" begin
        t0 = DateTime(2024, 9, 3, 15, 0, 0)
        n_gates = 5
        sweeps = SweepGroup[]
        for s in 1:2
            n_rays_s = 4
            sweep = SweepGroup(
                sweep_number = s - 1,
                sweep_mode = "azimuth_surveillance",
                fixed_angle = Float64(s),
                time = [t0 + Second((s-1)*n_rays_s + r) for r in 0:(n_rays_s-1)],
                range = collect(Float64, 400.0:100.0:(400.0 + 100.0 * (n_gates - 1))),
                azimuth = collect(Float64, range(0.0, 270.0; length=n_rays_s)),
                elevation = fill(Float64(s), n_rays_s),
                nyquist_velocity = fill(25.0, n_rays_s),
            )
            for (k, name) in enumerate(("DBZ", "VEL"))
                data = Float32.(reshape(1:(n_rays_s * n_gates), n_rays_s, n_gates)) .+ Float32(k)
                add_field!(sweep, name, data, FieldMetadata(units=name, fill_value=-32768.0))
            end
            push!(sweeps, sweep)
        end
        v = Volume(
            instrument_name = "TEST",
            time_coverage_start = t0,
            time_coverage_end = t0 + Second(8),
            latitude = 16.0, longitude = -24.0, altitude = 50.0,
            sweeps = sweeps,
        )

        legacy, names = as_legacy_radar(v)
        @test names == sort(["DBZ", "VEL"])
        @test length(legacy.azimuth) == 8
        @test legacy.swpstart == Float32[0, 4]
        @test legacy.swpend == Float32[3, 7]
        @test size(legacy.moments) == (8 * n_gates, 2)

        # Verify a known cell: sweep 1, ray 1, gate 1. DBZ index by names.
        col_dbz = findfirst(==("DBZ"), names)
        idx = (1 - 1) * n_gates + 1
        @test legacy.moments[idx, col_dbz] ≈ Float64(v.sweeps[1].fields["DBZ"].data[1, 1])

        # Reverse: legacy → Volume
        rt = as_volume(legacy; field_names = names, instrument_name = "TEST")
        @test length(rt.sweeps) == length(v.sweeps)
        for i in eachindex(v.sweeps)
            @test rt.sweeps[i].fixed_angle ≈ v.sweeps[i].fixed_angle
            @test length(rt.sweeps[i].time) == length(v.sweeps[i].time)
            for name in names
                d_in = v.sweeps[i].fields[name].data
                d_rt = rt.sweeps[i].fields[name].data
                @test all(isapprox.(d_in, d_rt; atol=1e-3))
            end
        end
    end

    @testset "moment-column ordering: explicit field_names" begin
        # Regression for the silent column-ordering bug: when alphabetical order
        # of the volume's fields disagrees with TOML order, alphabetical layout
        # would put ZDR data into the DBZ column and vice versa.
        t0 = DateTime(2024, 9, 3, 15, 0, 0)
        n_rays_s = 4
        n_gates = 3
        sweep = SweepGroup(
            sweep_number = 0,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 1.0,
            time = [t0 + Second(r) for r in 0:(n_rays_s-1)],
            range = collect(Float64, 400.0:100.0:(400.0 + 100.0 * (n_gates - 1))),
            azimuth = collect(Float64, range(0.0, 270.0; length=n_rays_s)),
            elevation = fill(1.0, n_rays_s),
            nyquist_velocity = fill(25.0, n_rays_s),
        )
        # Two fields with distinctly different value ranges so column→field
        # binding is testable by value alone.
        dbz_data = fill(Float32(30.0), n_rays_s, n_gates)
        zdr_data = fill(Float32(2.0), n_rays_s, n_gates)
        add_field!(sweep, "DBZ", dbz_data, FieldMetadata(units="dBZ", fill_value=-32768.0))
        add_field!(sweep, "ZDR", zdr_data, FieldMetadata(units="dB",  fill_value=-32768.0))
        v = Volume(
            instrument_name = "TEST",
            time_coverage_start = t0,
            time_coverage_end = t0 + Second(n_rays_s),
            latitude = 16.0, longitude = -24.0, altitude = 50.0,
            sweeps = [sweep],
        )

        # Reverse-alphabetical TOML order: column 1 must be ZDR.
        legacy, names = as_legacy_radar(v; field_names = ["ZDR", "DBZ"])
        @test names == ["ZDR", "DBZ"]
        col_one_value = legacy.moments[1, 1]
        col_two_value = legacy.moments[1, 2]
        @test col_one_value ≈ 2.0
        @test col_two_value ≈ 30.0

        # A field listed but absent in the sweep stays missing.
        legacy2, names2 = as_legacy_radar(v; field_names = ["DBZ", "MISSING_FIELD"])
        @test names2 == ["DBZ", "MISSING_FIELD"]
        @test legacy2.moments[1, 1] ≈ 30.0
        @test ismissing(legacy2.moments[1, 2])
    end

    @testset "moment-column ordering: alphabetical default" begin
        t0 = DateTime(2024, 9, 3, 15, 0, 0)
        n_rays_s = 2
        n_gates = 2
        sweep = SweepGroup(
            sweep_number = 0,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 1.0,
            time = [t0 + Second(r) for r in 0:(n_rays_s-1)],
            range = collect(Float64, 400.0:100.0:(400.0 + 100.0 * (n_gates - 1))),
            azimuth = collect(Float64, range(0.0, 180.0; length=n_rays_s)),
            elevation = fill(1.0, n_rays_s),
            nyquist_velocity = fill(25.0, n_rays_s),
        )
        add_field!(sweep, "ZDR", fill(Float32(2.0), n_rays_s, n_gates),
                   FieldMetadata(units="dB", fill_value=-32768.0))
        add_field!(sweep, "DBZ", fill(Float32(30.0), n_rays_s, n_gates),
                   FieldMetadata(units="dBZ", fill_value=-32768.0))
        v = Volume(
            instrument_name = "TEST",
            time_coverage_start = t0,
            time_coverage_end = t0 + Second(n_rays_s),
            latitude = 0.0, longitude = 0.0, altitude = 0.0,
            sweeps = [sweep],
        )

        legacy, names = as_legacy_radar(v)
        # Default: alphabetical, so DBZ → col 1, ZDR → col 2.
        @test names == ["DBZ", "ZDR"]
        @test legacy.moments[1, 1] ≈ 30.0
        @test legacy.moments[1, 2] ≈ 2.0
    end

    @testset "platform velocity passthrough" begin
        # Regression for hard-coded zeros. Build a mobile-platform sweep with
        # nonzero per-ray velocity in the Georeference; expect those to land in
        # ew/ns/w_platform.
        t0 = DateTime(2024, 9, 3, 15, 0, 0)
        n_rays_s = 4
        n_gates = 2
        ew = collect(Float64, [1.0, 2.0, 3.0, 4.0])
        ns = collect(Float64, [10.0, 20.0, 30.0, 40.0])
        w  = collect(Float64, [0.1, 0.2, 0.3, 0.4])
        gr = Georeference(
            latitude = fill(20.0, n_rays_s),
            longitude = fill(-150.0, n_rays_s),
            altitude = fill(3000.0, n_rays_s),
            eastward_velocity = ew,
            northward_velocity = ns,
            vertical_velocity = w,
        )
        sweep = SweepGroup(
            sweep_number = 0,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 1.0,
            time = [t0 + Second(r) for r in 0:(n_rays_s-1)],
            range = collect(Float64, 400.0:100.0:(400.0 + 100.0 * (n_gates - 1))),
            azimuth = collect(Float64, range(0.0, 270.0; length=n_rays_s)),
            elevation = fill(1.0, n_rays_s),
            nyquist_velocity = fill(25.0, n_rays_s),
            georeference = gr,
        )
        add_field!(sweep, "DBZ", fill(Float32(20.0), n_rays_s, n_gates),
                   FieldMetadata(units="dBZ", fill_value=-32768.0))
        v = Volume(
            instrument_name = "AIRBORNE_TEST",
            platform_type = "aircraft",
            time_coverage_start = t0,
            time_coverage_end = t0 + Second(n_rays_s),
            latitude = 20.0, longitude = -150.0, altitude = 3000.0,
            sweeps = [sweep],
        )

        legacy, _ = as_legacy_radar(v)
        @test legacy.ew_platform == Float32.(ew)
        @test legacy.ns_platform == Float32.(ns)
        @test legacy.w_platform  == Float32.(w)

        # Sweep without Georeference still gets zeros.
        sweep_no_gr = SweepGroup(
            sweep_number = 0,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 1.0,
            time = [t0 + Second(r) for r in 0:(n_rays_s-1)],
            range = collect(Float64, 400.0:100.0:(400.0 + 100.0 * (n_gates - 1))),
            azimuth = collect(Float64, range(0.0, 270.0; length=n_rays_s)),
            elevation = fill(1.0, n_rays_s),
            nyquist_velocity = fill(25.0, n_rays_s),
        )
        add_field!(sweep_no_gr, "DBZ", fill(Float32(20.0), n_rays_s, n_gates),
                   FieldMetadata(units="dBZ", fill_value=-32768.0))
        v_stat = Volume(
            instrument_name = "STATIONARY_TEST",
            time_coverage_start = t0,
            time_coverage_end = t0 + Second(n_rays_s),
            latitude = 0.0, longitude = 0.0, altitude = 0.0,
            sweeps = [sweep_no_gr],
        )
        legacy_stat, _ = as_legacy_radar(v_stat)
        @test all(legacy_stat.ew_platform .== 0.0f0)
        @test all(legacy_stat.ns_platform .== 0.0f0)
        @test all(legacy_stat.w_platform  .== 0.0f0)
    end

    @testset "v1 fixture → legacy radar shape" begin
        if !isfile(FIXTURE_V1)
            @info "Skipping bridge fixture round-trip"
            return
        end
        v = read_cfradial(FIXTURE_V1)
        legacy, names = as_legacy_radar(v)
        n_total = sum(length(s.time) for s in v.sweeps)
        @test length(legacy.azimuth) == n_total
        @test length(legacy.range) == 2447
        @test size(legacy.moments) == (n_total * 2447, length(names))
        @test legacy.swpstart[1] == 0
        @test legacy.swpend[end] == n_total - 1
    end
end
