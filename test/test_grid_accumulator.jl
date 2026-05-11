@testset "GridAccumulator" begin
    # Reusable mini grid_spec factory.
    function _spec(shape::Symbol; n_x=4, n_y=3, n_z=2)
        return GridSpec(
            shape = shape,
            reference_latitude = 16.0,
            reference_longitude = -24.0,
            x_axis = collect(Float64, 1.0:n_x),
            y_axis = collect(Float64, 1.0:n_y),
            z_axis = collect(Float64, 1.0:n_z),
            lat_axis = shape === :latlon_3d ? collect(Float64, 1.0:n_y) : nothing,
            lon_axis = shape === :latlon_3d ? collect(Float64, 1.0:n_x) : nothing,
            rhi_azimuth = shape === :rhi_2d ? 90.0 : nothing,
        )
    end

    @testset "constructors and shapes" begin
        fields = ["DBZ", "VEL"]
        gtype = Dict("DBZ" => :linear, "VEL" => :weighted)

        for (shape, expected_dims) in (
            (:volume_3d,    (2, 2, 3, 4)),
            (:latlon_3d,    (2, 2, 3, 4)),
            (:rhi_2d,       (2, 2, 4)),
            (:ppi_2d,       (2, 3, 4)),
            (:composite_2d, (2, 3, 4)),
            (:column_1d,    (2, 2)),
        )
            spec = _spec(shape)
            acc = GridAccumulator(spec, fields, gtype)
            @test size(acc.weighted_sum) == expected_dims
            @test size(acc.weight_total) == expected_dims
            @test size(acc.coverage)     == expected_dims
            @test all(acc.weighted_sum .== 0.0)
            @test all(acc.weight_total .== 0.0)
            @test all(acc.coverage     .== Int8(0))
            @test acc.fields == fields
            @test acc.grid_type == gtype
            @test acc.field_folds == [false, false]
            @test isempty(acc.sweeps)
            @test acc.schema_version == Daisho.GRID_ACCUMULATOR_SCHEMA_VERSION
        end

        # Custom field_folds.
        spec = _spec(:volume_3d)
        acc = GridAccumulator(spec, fields, gtype; field_folds = [false, true])
        @test acc.field_folds == [false, true]

        # field_folds length mismatch is rejected.
        @test_throws ArgumentError GridAccumulator(spec, fields, gtype; field_folds = [false])

        # Missing grid_type entry is rejected.
        bad_gtype = Dict("DBZ" => :linear)
        @test_throws ArgumentError GridAccumulator(spec, fields, bad_gtype)

        # Unsupported shape is rejected by the dims helper (indirectly).
        bad_spec = GridSpec(
            shape = :not_a_shape,
            reference_latitude = 0.0, reference_longitude = 0.0,
            x_axis = [0.0], y_axis = [0.0], z_axis = [0.0])
        @test_throws ArgumentError GridAccumulator(bad_spec, fields, gtype)
    end

    @testset "DaishoParameters constructor uses TOML order" begin
        p = DaishoParameters()
        spec = _spec(:volume_3d)
        acc = GridAccumulator(spec, p)
        @test acc.fields == p.moments.fields
        @test acc.grid_type == p.moments.grid_type
        @test length(acc.field_folds) == length(p.moments.fields)
        @test all(.!acc.field_folds)
    end

    @testset "JLD2 round-trip" begin
        spec = _spec(:volume_3d)
        fields = ["DBZ", "VEL"]
        gtype  = Dict("DBZ" => :linear, "VEL" => :weighted)
        acc = GridAccumulator(spec, fields, gtype; field_folds = [false, true])
        acc.weighted_sum[1, 1, 1, 1] = 42.0
        acc.weighted_sum[2, 2, 2, 3] = -1.5
        acc.weight_total[1, 1, 1, 1] = 7.0
        acc.weight_total[2, 2, 2, 3] = 3.0
        acc.coverage[1, 1, 1, 1]     = Int8(2)
        acc.coverage[2, 2, 2, 3]     = Int8(1)
        push!(acc.sweeps, SweepProvenance(
            instrument_name = "SEAPOL",
            scan_name = "SUR",
            sweep_number = 0,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 0.5,
            time_start = DateTime(2024, 9, 3, 15, 0, 0),
            time_end   = DateTime(2024, 9, 3, 15, 0, 30),
            source_file = "/tmp/test.nc",
            ref_latitude = 16.0,
            ref_longitude = -24.0,
            ref_altitude = 50.0,
        ))

        path = tempname() * ".jld2"
        try
            save_accumulator(path, acc)
            rt = load_accumulator(path)
            @test rt.fields == acc.fields
            @test rt.grid_type == acc.grid_type
            @test rt.field_folds == acc.field_folds
            @test rt.weighted_sum == acc.weighted_sum
            @test rt.weight_total == acc.weight_total
            @test rt.coverage == acc.coverage
            @test length(rt.sweeps) == 1
            @test rt.sweeps[1].instrument_name == "SEAPOL"
            @test rt.sweeps[1].fixed_angle == 0.5
            @test rt.sweeps[1].time_start == DateTime(2024, 9, 3, 15, 0, 0)
            @test rt.schema_version == Daisho.GRID_ACCUMULATOR_SCHEMA_VERSION
            @test rt.grid_spec.x_axis == acc.grid_spec.x_axis
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "merge_accumulators! happy path" begin
        spec = _spec(:volume_3d)
        fields = ["DBZ", "VEL"]
        gtype  = Dict("DBZ" => :weighted, "VEL" => :weighted)

        dst = GridAccumulator(spec, fields, gtype)
        src = GridAccumulator(spec, fields, gtype)

        src.weighted_sum[1, 1, 1, 1] = 4.0
        src.weight_total[1, 1, 1, 1] = 2.0
        src.coverage[1, 1, 1, 1]     = Int8(2)
        src.weighted_sum[2, 2, 2, 3] = -1.0
        src.weight_total[2, 2, 2, 3] = 0.5
        src.coverage[2, 2, 2, 3]     = Int8(1)
        push!(src.sweeps, SweepProvenance(sweep_number = 1))

        merge_accumulators!(dst, src)
        @test dst.weighted_sum[1, 1, 1, 1] == 4.0
        @test dst.weight_total[1, 1, 1, 1] == 2.0
        @test dst.coverage[1, 1, 1, 1]     == Int8(2)
        @test dst.weighted_sum[2, 2, 2, 3] == -1.0
        @test dst.weight_total[2, 2, 2, 3] == 0.5
        @test dst.coverage[2, 2, 2, 3]     == Int8(1)
        @test length(dst.sweeps) == 1
        @test dst.sweeps[1].sweep_number == 1

        # Merge a second source on top: weights/sums accumulate; coverage takes max.
        src2 = GridAccumulator(spec, fields, gtype)
        src2.weighted_sum[1, 1, 1, 1] = 6.0
        src2.weight_total[1, 1, 1, 1] = 3.0
        src2.coverage[1, 1, 1, 1]     = Int8(1)  # weaker coverage, dst stays at 2
        push!(src2.sweeps, SweepProvenance(sweep_number = 2))
        merge_accumulators!(dst, src2)
        @test dst.weighted_sum[1, 1, 1, 1] == 10.0
        @test dst.weight_total[1, 1, 1, 1] == 5.0
        @test dst.coverage[1, 1, 1, 1]     == Int8(2)
        @test length(dst.sweeps) == 2
    end

    @testset "merge_accumulators! strict compatibility" begin
        fields = ["DBZ"]
        gtype  = Dict("DBZ" => :weighted)
        spec   = _spec(:volume_3d)
        dst    = GridAccumulator(spec, fields, gtype)

        # grid_spec mismatch.
        spec_b = _spec(:volume_3d; n_x = 5)
        @test_throws ArgumentError merge_accumulators!(
            dst, GridAccumulator(spec_b, fields, gtype))

        # fields mismatch.
        @test_throws ArgumentError merge_accumulators!(
            dst, GridAccumulator(spec, ["VEL"], Dict("VEL" => :weighted)))

        # grid_type mismatch.
        @test_throws ArgumentError merge_accumulators!(
            dst, GridAccumulator(spec, fields, Dict("DBZ" => :linear)))

        # field_folds mismatch.
        @test_throws ArgumentError merge_accumulators!(
            dst, GridAccumulator(spec, fields, gtype; field_folds = [true]))
    end

    @testset "merge_accumulators! rejects field-folds fields" begin
        spec = _spec(:volume_3d)
        fields = ["VEL"]
        gtype  = Dict("VEL" => :weighted)
        dst = GridAccumulator(spec, fields, gtype; field_folds = [true])
        src = GridAccumulator(spec, fields, gtype; field_folds = [true])
        err = nothing
        try
            merge_accumulators!(dst, src)
        catch e
            err = e
        end
        @test err isa ArgumentError
        @test occursin("wind retrieval", err.msg)
    end

    @testset "merge_accumulators! :nearest picks higher weight per cell" begin
        spec = _spec(:volume_3d)
        fields = ["RHOHV"]
        gtype  = Dict("RHOHV" => :nearest)
        dst = GridAccumulator(spec, fields, gtype)
        src = GridAccumulator(spec, fields, gtype)

        dst.weighted_sum[1, 1, 1, 1] = 0.95
        dst.weight_total[1, 1, 1, 1] = 0.3
        src.weighted_sum[1, 1, 1, 1] = 0.85
        src.weight_total[1, 1, 1, 1] = 0.7   # higher weight wins

        # Second cell: dst higher, should stay.
        dst.weighted_sum[1, 2, 2, 2] = 0.99
        dst.weight_total[1, 2, 2, 2] = 0.9
        src.weighted_sum[1, 2, 2, 2] = 0.1
        src.weight_total[1, 2, 2, 2] = 0.05

        merge_accumulators!(dst, src)
        @test dst.weighted_sum[1, 1, 1, 1] == 0.85
        @test dst.weight_total[1, 1, 1, 1] == 0.7
        @test dst.weighted_sum[1, 2, 2, 2] == 0.99
        @test dst.weight_total[1, 2, 2, 2] == 0.9
    end
end
