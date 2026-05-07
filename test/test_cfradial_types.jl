@testset "CfRadial type model" begin
    t0 = DateTime(2024, 9, 3, 15, 0, 0)
    t1 = DateTime(2024, 9, 3, 15, 5, 0)

    @testset "FieldMetadata defaults" begin
        md = FieldMetadata()
        @test md.standard_name === nothing
        @test md.long_name === nothing
        @test md.units === nothing
        @test md.fill_value === nothing
        @test md.undetect === nothing
        @test md.sampling_ratio == 1.0
        @test md.is_discrete == false
        @test md.field_folds == false
        @test md.is_quality_field == false
        @test md.qualified_variables == String[]
        @test md.ancillary_variables == String[]
        @test md.extra_attrs isa Dict{String,Any}
    end

    @testset "Field constructors" begin
        data = zeros(Float32, 3, 4)
        fld = Daisho.Field(data)
        @test size(fld.data) == (3, 4)
        @test fld.metadata isa FieldMetadata
        fld2 = Daisho.Field(data, FieldMetadata(units="dBZ"))
        @test fld2.metadata.units == "dBZ"
    end

    @testset "SweepGroup defaults + add_field!" begin
        sweep = SweepGroup(
            sweep_number = 0,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 1.0,
            time = [t0, t0 + Second(1), t0 + Second(2)],
            range = [400.0, 500.0, 600.0, 700.0],
            azimuth = [0.0, 90.0, 180.0],
            elevation = [1.0, 1.0, 1.0],
        )
        @test sweep.follow_mode == "none"
        @test sweep.prt_mode == "fixed"
        @test sweep.polarization_mode == "horizontal"
        @test isempty(sweep.fields)
        @test n_rays(sweep) == 3
        @test field_names(sweep) == String[]

        data = ones(Float32, 3, 4)
        add_field!(sweep, "DBZ", data, FieldMetadata(units="dBZ"))
        @test has_field(sweep, "DBZ")
        @test field_names(sweep) == ["DBZ"]
        @test sweep.fields["DBZ"].metadata.units == "dBZ"

        # Wrong shape rejected
        @test_throws DimensionMismatch add_field!(sweep, "BAD", ones(Float32, 2, 2))

        @test remove_field!(sweep, "DBZ")
        @test !has_field(sweep, "DBZ")
        @test !remove_field!(sweep, "MISSING")
    end

    @testset "Volume basics + iteration" begin
        sweep = SweepGroup(
            sweep_number = 0,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 1.0,
            time = [t0, t0 + Second(1)],
            range = [400.0, 500.0],
            azimuth = [0.0, 90.0],
            elevation = [1.0, 1.0],
        )
        sweep2 = SweepGroup(
            sweep_number = 1,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 2.0,
            time = [t0 + Second(2), t0 + Second(3), t0 + Second(4)],
            range = [400.0, 500.0],
            azimuth = [0.0, 120.0, 240.0],
            elevation = [2.0, 2.0, 2.0],
        )
        add_field!(sweep, "DBZ", zeros(Float32, 2, 2))
        add_field!(sweep2, "VEL", zeros(Float32, 3, 2))
        v = Volume(
            time_coverage_start = t0,
            time_coverage_end = t1,
            latitude = 16.886,
            longitude = -24.988,
            altitude = 50.0,
            sweeps = [sweep, sweep2],
        )
        @test length(v) == 2
        @test v[1] === sweep
        @test v[2] === sweep2
        collected = [s.sweep_number for s in v]
        @test collected == [0, 1]
        @test n_rays(v) == 5
        @test field_names(v) == ["DBZ", "VEL"]
        @test has_field(v, "DBZ")
        @test !has_field(v, "MISSING")
    end

    @testset "RadarCalibration / Georeference / GeoreferenceCorrection defaults" begin
        e = RadarCalibrationEntry(pulse_width = 1e-6, noise_hc = -100.0)
        @test e.pulse_width == 1e-6
        @test e.noise_hc == -100.0
        @test e.zdr_correction === nothing

        cal = RadarCalibration(entries = [e])
        @test length(cal.entries) == 1
        @test cal.calib_index === nothing

        gc = GeoreferenceCorrection(azimuth_correction = 0.5)
        @test gc.azimuth_correction == 0.5
        @test gc.elevation_correction === nothing

        g = Georeference(latitude = [16.0, 16.1],
                         longitude = [-24.0, -24.1],
                         altitude = [50.0, 50.0])
        @test length(g.latitude) == 2
        @test g.heading === nothing
    end
end
