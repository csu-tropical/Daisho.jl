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
