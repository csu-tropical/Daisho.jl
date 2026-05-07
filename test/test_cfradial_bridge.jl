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
