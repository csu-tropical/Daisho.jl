function _build_synthetic_volume(; n_sweeps=2, n_rays=8, n_gates=20,
                                  fields=("DBZ", "VEL"))
    t0 = DateTime(2024, 9, 3, 15, 0, 0)
    sweeps = SweepGroup[]
    for s in 1:n_sweeps
        rays = [t0 + Second((s - 1) * n_rays + r) for r in 0:(n_rays - 1)]
        sweep = SweepGroup(
            sweep_number = s - 1,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = Float64(s),
            time = rays,
            range = collect(Float64, 400.0:100.0:(400.0 + 100.0 * (n_gates - 1))),
            azimuth = collect(Float64, range(0.0, 360.0; length=n_rays + 1))[1:n_rays],
            elevation = fill(Float64(s), n_rays),
            range_meters_to_first_gate = 400.0,
            range_meters_between_gates = 100.0,
        )
        for fname in fields
            data = Float32.(reshape(1:(n_rays * n_gates), n_rays, n_gates)) .+ 0.1f0 * s
            md = FieldMetadata(units = (fname == "DBZ" ? "dBZ" : "m/s"),
                               long_name = fname, fill_value = -32768.0)
            add_field!(sweep, fname, data, md)
        end
        push!(sweeps, sweep)
    end
    return Volume(
        title = "Test",
        instrument_name = "SYN",
        site_name = "SYN",
        source = "synthetic",
        history = "test",
        time_coverage_start = t0,
        time_coverage_end = t0 + Second(n_sweeps * n_rays),
        latitude = 16.886,
        longitude = -24.988,
        altitude = 50.0,
        sweeps = sweeps,
    )
end

@testset "CfRadial writer" begin

    @testset "synthetic round-trip" begin
        v = _build_synthetic_volume()
        tmp = tempname() * ".nc"
        write_cfradial(v, tmp)
        @test isfile(tmp)
        rt = read_cfradial(tmp)
        @test length(rt.sweeps) == length(v.sweeps)
        @test rt.instrument_name == v.instrument_name
        @test rt.latitude ≈ v.latitude
        @test rt.longitude ≈ v.longitude
        @test rt.altitude ≈ v.altitude
        for i in eachindex(v.sweeps)
            @test rt.sweeps[i].fixed_angle ≈ v.sweeps[i].fixed_angle
            @test length(rt.sweeps[i].time) == length(v.sweeps[i].time)
            @test length(rt.sweeps[i].range) == length(v.sweeps[i].range)
            @test all(isapprox.(rt.sweeps[i].azimuth, v.sweeps[i].azimuth; atol=1e-3))
            for fname in keys(v.sweeps[i].fields)
                @test haskey(rt.sweeps[i].fields, fname)
                d_in = v.sweeps[i].fields[fname].data
                d_rt = rt.sweeps[i].fields[fname].data
                @test size(d_in) == size(d_rt)
                @test all(isapprox.(d_in, d_rt; atol=1e-3))
                @test rt.sweeps[i].fields[fname].metadata.units ==
                      v.sweeps[i].fields[fname].metadata.units
            end
        end
        rm(tmp)
    end

    @testset "synthetic v1 round-trip preserves optional groups" begin
        # Read a fully-populated synthetic v1 file (radar calibration,
        # georeference correction, frequency, radar parameters, per-ray optional
        # vars, extra attrs) into a Volume and write it back out. Writing a Volume
        # that carries these optionals exercises the optional-field write paths
        # (calibration / georeference_correction groups, frequency dimension, the
        # per-ray optional variables) that the minimal synthetic volume skips —
        # paths otherwise only hit by the gitignored real fixtures.
        src = tempname() * ".nc"
        build_synthetic_cfradial_v1(src)
        v = read_cfradial(src)
        @test v.radar_calibration !== nothing        # source is rich
        tmp = tempname() * ".nc"
        write_cfradial(v, tmp)
        @test isfile(tmp)
        rt = read_cfradial(tmp)
        @test length(rt.sweeps) == length(v.sweeps)
        @test rt.radar_calibration !== nothing       # calibration survived the round-trip
        @test !isempty(rt.radar_calibration.entries)
        for fname in keys(v.sweeps[1].fields)
            @test haskey(rt.sweeps[1].fields, fname)
        end
        rm(src); rm(tmp)
    end

    @testset "synthetic v2 round-trip" begin
        # Read a fully-populated synthetic v2 file into a Volume, augment each
        # sweep with a georeference carrying `georefs_applied` (set explicitly so
        # the test is robust to the source), and write it back
        # with `write_extras=true`. This exercises writer paths the minimal
        # synthetic volume skips and that are otherwise only hit by the gitignored
        # real fixtures: volume-level `extra_attrs`, per-field metadata
        # `extra_attrs`, the `radar_monitoring` group, and `georefs_applied`.
        src = tempname() * ".nc"
        build_synthetic_cfradial_v2(src; n_sweeps = 2, rays_per_sweep = 4,
                                    n_gates = 5, field_names = ["DBZ", "VEL"])
        v = read_cfradial(src)

        # Source Volume is rich: volume-level + per-field extras and monitoring.
        @test haskey(v.extra_attrs, "non_spec_attr")
        @test haskey(v.sweeps[1].fields["DBZ"].metadata.extra_attrs, "custom_field_attr")
        @test v.sweeps[1].radar_monitoring !== nothing

        # Inject a georeference (with georefs_applied) into every sweep so the
        # writer's georeference/georefs_applied path runs on write.
        newsweeps = SweepGroup[]
        for s in v.sweeps
            n = length(s.time)
            geo = Georeference(
                latitude = fill(v.latitude, n),
                longitude = fill(v.longitude, n),
                altitude = fill(v.altitude, n),
                heading = fill(90.0, n),
                georefs_applied = fill(true, n),
            )
            kw = Dict(f => getfield(s, f) for f in fieldnames(SweepGroup))
            kw[:georeference] = geo
            push!(newsweeps, SweepGroup(; kw...))
        end
        vkw = Dict(f => getfield(v, f) for f in fieldnames(Volume))
        vkw[:sweeps] = newsweeps
        vw = Volume(; vkw...)

        tmp = tempname() * ".nc"
        write_cfradial(vw, tmp; write_extras = true)
        @test isfile(tmp)

        # Round-trip read: georeference, monitoring, and extras all survive.
        rt = read_cfradial(tmp)
        @test length(rt.sweeps) == length(vw.sweeps)
        @test haskey(rt.extra_attrs, "non_spec_attr")                       # volume extra_attrs written
        @test haskey(rt.sweeps[1].fields["DBZ"].metadata.extra_attrs,
                     "custom_field_attr")                                   # per-field extra_attrs written
        @test rt.sweeps[1].radar_monitoring !== nothing                     # radar_monitoring group written
        @test all(≈(70.0), rt.sweeps[1].radar_monitoring.measured_transmit_power_h)
        @test rt.sweeps[1].georeference !== nothing                         # georeference survived (v2 reader fix)
        @test length(rt.sweeps[1].georeference.latitude) == length(rt.sweeps[1].time)
        # Fill/sentinel semantics preserved across the round-trip
        dbz = rt.sweeps[1].fields["DBZ"].data
        @test isnan(dbz[1, 1])
        @test dbz[1, 2] ≈ -9999.0 atol = 1e-3
        for fname in keys(vw.sweeps[1].fields)
            @test haskey(rt.sweeps[1].fields, fname)
        end

        # Also inspect the raw NetCDF to confirm the writer emitted the
        # georeference sub-group + georefs_applied.
        NCDatasets.NCDataset(tmp, "r") do nc
            g1 = nc.group["sweep_0001"]
            @test haskey(g1.group, "georeference")
            @test haskey(g1.group["georeference"], "georefs_applied")
            @test haskey(g1.group, "radar_monitoring")
        end

        rm(src); rm(tmp)
    end

    @testset "real-fixture v2 round-trip" begin
        if !isfile(FIXTURE_V2)
            @info "Skipping v2 round-trip (fixture absent)"
            return
        end
        v = read_cfradial(FIXTURE_V2)
        tmp = tempname() * ".nc"
        write_cfradial(v, tmp)
        rt = read_cfradial(tmp)
        @test length(rt.sweeps) == length(v.sweeps)
        @test rt.instrument_name == v.instrument_name
        for i in eachindex(v.sweeps)
            @test length(rt.sweeps[i].time) == length(v.sweeps[i].time)
            for fname in keys(v.sweeps[i].fields)
                @test haskey(rt.sweeps[i].fields, fname)
            end
        end
        rm(tmp)
    end

    @testset "real-fixture v1 round-trip" begin
        if !isfile(FIXTURE_V1)
            @info "Skipping v1 round-trip (fixture absent)"
            return
        end
        v = read_cfradial(FIXTURE_V1)
        tmp = tempname() * ".nc"
        write_cfradial(v, tmp)
        rt = read_cfradial(tmp)
        @test length(rt.sweeps) == length(v.sweeps)
        for i in eachindex(v.sweeps)
            @test length(rt.sweeps[i].time) == length(v.sweeps[i].time)
            @test length(rt.sweeps[i].range) == length(v.sweeps[i].range)
        end
        rm(tmp)
    end

    @testset "update_cfradial on v1 fixture" begin
        if !isfile(FIXTURE_V1)
            @info "Skipping update_cfradial v1 (fixture absent)"
            return
        end
        v = read_cfradial(FIXTURE_V1)
        # Add a synthetic field to the first sweep.
        s = v.sweeps[1]
        new_data = fill(Float32(42.0), length(s.time), length(s.range))
        add_field!(s, "DBZ_DAISHO_QC", new_data,
                   FieldMetadata(units="dBZ", long_name="Daisho QC test", fill_value=-32768.0))
        tmp = tempname() * ".nc"
        update_cfradial(FIXTURE_V1, tmp, v; fields=["DBZ_DAISHO_QC"])
        rt = read_cfradial(tmp)
        @test haskey(rt.sweeps[1].fields, "DBZ_DAISHO_QC")
        rt_data = rt.sweeps[1].fields["DBZ_DAISHO_QC"].data
        @test size(rt_data) == (length(s.time), length(s.range))
        @test all(isapprox.(rt_data, 42.0f0; atol=1e-3))
        # Original DBZ should still match the source.
        @test haskey(rt.sweeps[1].fields, "DBZ")
        rm(tmp)
    end

    @testset "update_cfradial on v2 fixture" begin
        if !isfile(FIXTURE_V2)
            @info "Skipping update_cfradial v2 (fixture absent)"
            return
        end
        v = read_cfradial(FIXTURE_V2)
        s = v.sweeps[1]
        new_data = fill(Float32(7.0), length(s.time), length(s.range))
        add_field!(s, "DBZ_DAISHO_QC", new_data,
                   FieldMetadata(units="dBZ", fill_value=-32768.0))
        tmp = tempname() * ".nc"
        update_cfradial(FIXTURE_V2, tmp, v; fields=["DBZ_DAISHO_QC"])
        rt = read_cfradial(tmp)
        @test haskey(rt.sweeps[1].fields, "DBZ_DAISHO_QC")
        rm(tmp)
    end

    @testset "update_cfradial layout-mismatch error" begin
        if !isfile(FIXTURE_V1)
            return
        end
        v = read_cfradial(FIXTURE_V1)
        # Drop one ray from sweep 1.
        s = v.sweeps[1]
        bad = SweepGroup(
            sweep_number = s.sweep_number,
            sweep_mode = s.sweep_mode,
            fixed_angle = s.fixed_angle,
            time = s.time[1:end-1],
            range = s.range,
            azimuth = s.azimuth[1:end-1],
            elevation = s.elevation[1:end-1],
        )
        v2 = Volume(
            title = v.title, institution = v.institution, source = v.source,
            history = v.history, instrument_name = v.instrument_name,
            site_name = v.site_name,
            time_coverage_start = v.time_coverage_start,
            time_coverage_end = v.time_coverage_end,
            latitude = v.latitude, longitude = v.longitude, altitude = v.altitude,
            sweeps = vcat([bad], v.sweeps[2:end]),
        )
        tmp = tempname() * ".nc"
        @test_throws ErrorException update_cfradial(FIXTURE_V1, tmp, v2)
        isfile(tmp) && rm(tmp)
    end
end
