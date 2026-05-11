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

    @testset "grid_sweep!: legacy-equivalent Volume → grid" begin
        # Build a small synthetic Volume that can be gridded through both paths
        # and assert the new accumulator path agrees with the legacy worker
        # within numerical tolerance.
        p = DaishoParameters()
        # Use a small grid so the test is fast.
        new_cart = Daisho.CartesianGridParameters(
            xmin = -1500.0, xincr = 500.0, xdim = 7,
            ymin = -1500.0, yincr = 500.0, ydim = 7,
            zmin = 500.0,   zincr = 500.0, zdim = 3)
        p = Daisho.DaishoParameters(p.moments, p.qc, p.gridding,
            Daisho.GridParameters(cartesian = new_cart,
                                  latlon = p.grid.latlon,
                                  rhi = p.grid.rhi,
                                  spectral = p.grid.spectral),
            p.io)

        # Two-sweep synthetic volume with rich content — DBZ/VEL/SQI/DBZ_QC
        # exists with reasonable defaults. Distribute SQI=0.5 so missing_key
        # check engages.
        v = synthetic_volume(n_sweeps = 2, n_rays = 36, n_gates = 6)

        # Legacy path: explicitly go through `as_legacy_radar` and the
        # `radar`-typed driver. The Volume-typed driver no longer goes
        # through the bridge after Phase 4, so we'd be comparing against
        # ourselves if we called it here.
        legacy_file = tempname() * "_legacy.nc"
        try
            legacy_r, _ = as_legacy_radar(v; field_names = p.moments.fields)
            Daisho.grid_radar_volume(legacy_r, legacy_file, v.time_coverage_start, p)
            ds = NCDataset(legacy_file, "r")
            try
                # Reconstruct legacy radar_grid from variables. The writer
                # writes one variable per moment; we can read them back.
                legacy_grids = Dict{String,Array{Float64,4}}()
                for fname in p.moments.fields
                    if haskey(ds, fname)
                        raw = ds[fname][:, :, :, :]
                        legacy_grids[fname] = Float64.(coalesce.(raw, -32768.0))
                    end
                end

                # Accumulator path.
                gs = GridSpec(
                    shape = :volume_3d,
                    reference_latitude = v.latitude,
                    reference_longitude = v.longitude,
                    x_axis = collect(Float64, new_cart.xmin .+
                        (0:(new_cart.xdim-1)) .* new_cart.xincr),
                    y_axis = collect(Float64, new_cart.ymin .+
                        (0:(new_cart.ydim-1)) .* new_cart.yincr),
                    z_axis = collect(Float64, new_cart.zmin .+
                        (0:(new_cart.zdim-1)) .* new_cart.zincr),
                )
                acc = GridAccumulator(gs, p)
                for i in eachindex(v.sweeps)
                    grid_sweep!(acc, v, i, p)
                end
                grid = finalize_grid(acc)

                # Asserting cell-by-cell agreement on every cell would lock in
                # legacy "nearest gate SQI" coverage bug pixels. Instead, check
                # that cells where both paths report a valid value (i.e.
                # neither is -32768 nor -9999) agree closely.
                for (m, fname) in enumerate(p.moments.fields)
                    haskey(legacy_grids, fname) || continue
                    leg = legacy_grids[fname]
                    # legacy_grids dims: (X, Y, Z, time). Permute to (Z, Y, X).
                    leg_zyx = permutedims(leg[:, :, :, 1], (3, 2, 1))
                    for ix in CartesianIndices((new_cart.zdim, new_cart.ydim, new_cart.xdim))
                        a = grid[m, ix]
                        b = leg_zyx[ix]
                        if a > -9000 && b > -9000
                            mode = p.moments.grid_type[fname]
                            tol = mode === :linear ? 1e-3 : 1e-6
                            @test isapprox(a, b; atol = tol)
                        end
                    end
                    @test length(acc.sweeps) == length(v.sweeps)
                end
            finally
                close(ds)
            end
        finally
            isfile(legacy_file) && rm(legacy_file)
        end
    end

    @testset "grid_sweep! per-sweep additivity" begin
        # Grid sweep 1 alone, sweep 2 alone, vs both into the same accumulator.
        p = DaishoParameters()
        new_cart = Daisho.CartesianGridParameters(
            xmin = -1500.0, xincr = 500.0, xdim = 7,
            ymin = -1500.0, yincr = 500.0, ydim = 7,
            zmin = 500.0,   zincr = 500.0, zdim = 3)
        p = Daisho.DaishoParameters(p.moments, p.qc, p.gridding,
            Daisho.GridParameters(cartesian = new_cart,
                                  latlon = p.grid.latlon,
                                  rhi = p.grid.rhi,
                                  spectral = p.grid.spectral),
            p.io)
        v = synthetic_volume(n_sweeps = 2, n_rays = 36, n_gates = 6)

        gs = GridSpec(
            shape = :volume_3d,
            reference_latitude = v.latitude,
            reference_longitude = v.longitude,
            x_axis = collect(Float64, new_cart.xmin .+
                (0:(new_cart.xdim-1)) .* new_cart.xincr),
            y_axis = collect(Float64, new_cart.ymin .+
                (0:(new_cart.ydim-1)) .* new_cart.yincr),
            z_axis = collect(Float64, new_cart.zmin .+
                (0:(new_cart.zdim-1)) .* new_cart.zincr),
        )

        acc_combined = GridAccumulator(gs, p)
        grid_sweep!(acc_combined, v, 1, p)
        grid_sweep!(acc_combined, v, 2, p)
        grid_combined = finalize_grid(acc_combined)

        acc_a = GridAccumulator(gs, p)
        grid_sweep!(acc_a, v, 1, p)
        acc_b = GridAccumulator(gs, p)
        grid_sweep!(acc_b, v, 2, p)
        merge_accumulators!(acc_a, acc_b)
        grid_merged = finalize_grid(acc_a)

        for ix in CartesianIndices(size(grid_combined))
            a = grid_combined[ix]
            b = grid_merged[ix]
            # Both can be -32768 (no cov), -9999 (cov 1 no data), or values.
            if a > -9000 && b > -9000
                @test isapprox(a, b; atol = 1e-3)
            else
                @test a == b
            end
        end
    end

    @testset "any-in-range SQI coverage" begin
        # Construct a sweep where the gate VERTICALLY nearest to a chosen
        # gridpoint has missing SQI, but a farther in-range gate has
        # non-missing SQI. The legacy worker would punch a -32768 hole here;
        # the new worker must mark coverage = 1.
        p = DaishoParameters()
        # One-sweep volume, simple ranges/azimuths/elevations.
        t0 = DateTime(2024, 9, 3, 15, 0, 0)
        n_rays_s = 4
        n_gates_s = 4
        sweep = SweepGroup(
            sweep_number = 0,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 1.0,
            time = [t0 + Second(r) for r in 0:(n_rays_s-1)],
            range = collect(Float64, 500.0:500.0:(500.0 * n_gates_s)),
            azimuth = collect(Float64, [0.0, 90.0, 180.0, 270.0]),
            elevation = fill(1.0, n_rays_s),
            nyquist_velocity = fill(25.0, n_rays_s),
        )
        # SQI all-missing except for the FAR gates of one ray. Put a
        # non-missing SQI at (ray=2, gate=4).
        sqi = fill(Float32(NaN), n_rays_s, n_gates_s)
        sqi[2, 4] = 0.6f0
        dbz = fill(Float32(NaN), n_rays_s, n_gates_s)
        dbz[2, 4] = 35.0f0
        add_field!(sweep, "SQI", sqi, FieldMetadata(units="", fill_value=-32768.0))
        add_field!(sweep, "DBZ", dbz, FieldMetadata(units="dBZ", fill_value=-32768.0))
        v = Volume(
            instrument_name = "TEST",
            time_coverage_start = t0,
            time_coverage_end = t0 + Second(n_rays_s),
            latitude = 16.0, longitude = -24.0, altitude = 50.0,
            sweeps = [sweep],
        )

        # Make a small grid so the in-range query catches all of ray 2's gates.
        gs = GridSpec(
            shape = :volume_3d,
            reference_latitude = v.latitude,
            reference_longitude = v.longitude,
            x_axis = [0.0, 1000.0, 2000.0],
            y_axis = [-500.0, 0.0, 500.0],
            z_axis = [0.0, 50.0],
        )
        acc = GridAccumulator(gs, ["DBZ", "SQI"],
                              Dict("DBZ" => :linear, "SQI" => :weighted))
        grid_sweep!(acc, v, 1, p)

        # At least one cell should have coverage == 1 (any-in-range gate with
        # non-missing SQI). The legacy worker would have set it to 0 for cells
        # whose vertically-nearest gate had missing SQI.
        @test any(acc.coverage .== Int8(1)) || any(acc.coverage .== Int8(2))
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
