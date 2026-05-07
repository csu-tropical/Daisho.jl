@testset "CfRadial validate_spec" begin
    t0 = DateTime(2024, 9, 3, 15, 0, 0)

    function _good_volume()
        s = SweepGroup(
            sweep_number = 0,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 1.0,
            time = [t0, t0 + Second(1)],
            range = [400.0, 500.0],
            azimuth = [0.0, 90.0],
            elevation = [1.0, 1.0],
        )
        add_field!(s, "DBZ", zeros(Float32, 2, 2),
                   FieldMetadata(units="dBZ", fill_value=-32768.0))
        return Volume(
            title = "T", institution = "I", source = "S", history = "h",
            instrument_name = "RAD", site_name = "SITE",
            time_coverage_start = t0,
            time_coverage_end = t0 + Second(10),
            latitude = 16.0, longitude = -24.0, altitude = 50.0,
            sweeps = [s],
        )
    end

    @testset "valid volume passes" begin
        r = validate_spec(_good_volume())
        @test isempty(r.errors)
    end

    @testset "missing instrument_name is an error" begin
        s = SweepGroup(
            sweep_number = 0,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 1.0,
            time = [t0],
            range = [400.0],
            azimuth = [0.0],
            elevation = [1.0],
        )
        v = Volume(
            title = "T", institution = "I", source = "S", history = "h",
            instrument_name = "",
            site_name = "SITE",
            time_coverage_start = t0,
            time_coverage_end = t0 + Second(1),
            latitude = 16.0, longitude = -24.0, altitude = 50.0,
            sweeps = [s],
        )
        r = validate_spec(v)
        @test any(occursin("instrument_name", e) for e in r.errors)
        @test_throws ArgumentError validate_spec(v; strict=true)
    end

    @testset "no sweeps is an error" begin
        v = Volume(
            instrument_name = "X",
            time_coverage_start = t0,
            time_coverage_end = t0 + Second(1),
            latitude = 0.0, longitude = 0.0, altitude = 0.0,
            sweeps = SweepGroup[],
        )
        r = validate_spec(v)
        @test any(occursin("no sweeps", e) for e in r.errors)
        @test_throws ArgumentError validate_spec(v; strict=true)
    end

    @testset "missing units / _FillValue produces warnings" begin
        v = _good_volume()
        # Replace DBZ with a unit-less field.
        s = v.sweeps[1]
        s.fields["DBZ"] = Daisho.Field(zeros(Float32, 2, 2), FieldMetadata())
        r = validate_spec(v)
        @test any(occursin("units", w) for w in r.warnings)
        @test any(occursin("_FillValue", w) for w in r.warnings)
    end
end
