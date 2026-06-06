# Reference copy of the pre-unification `_grid_sweep_3d!` per-sweep worker. Held
# here (as test_wind_synthesis.jl does for its kernels) so the unified
# `_grid_sweep_products_3d!` can be asserted byte-identical on a fixture.
function _ref_grid_sweep_3d!(accum, sweep, p, ref_latitude, ref_longitude, ref_altitude)
    g  = accum.grid_spec
    gd = p.gridding
    missing_key = Daisho.field_with_tag(p, :define_scanned)
    valid_key   = Daisho.field_with_tag(p, :define_detection)

    TM = Daisho.CoordRefSystems.shift(
        Daisho.TransverseMercator{1.0, g.reference_latitude, Daisho.WGS84Latest},
        lonₒ = g.reference_longitude)
    grid_origin, radar_zyx, beams, n_gates_s, _ =
        Daisho._sweep_zyx_and_beams(sweep, ref_latitude, ref_longitude, ref_altitude, TM)
    balltree = Daisho._sweep_balltree_yx(radar_zyx, beams)
    gridpoints = Daisho._materialize_gridpoints_3d(g, TM, grid_origin)

    n_fields = length(accum.fields)
    nx = length(g.x_axis); ny = length(g.y_axis); nz = length(g.z_axis)

    horizontal_roi = if g.shape === :latlon_3d
        latrad = g.reference_latitude * pi / 180.0
        fac_lat = 111.13209 - 0.56605 * cos(2.0 * latrad)
        fac_lon = 111.41513 * cos(latrad)
        deg_km = sqrt(fac_lat^2 + fac_lon^2)
        degincr = ny >= 2 ? (g.lat_axis[2] - g.lat_axis[1]) :
                  (nx >= 2 ? (g.lon_axis[2] - g.lon_axis[1]) : 0.01)
        deg_km * 1000.0 * degincr * 0.75
    else
        xincr = nx >= 2 ? (g.x_axis[2] - g.x_axis[1]) : 0.0
        xincr * 0.75
    end
    zincr = nz >= 2 ? (g.z_axis[2] - g.z_axis[1]) : 0.0
    vertical_roi = zincr * 0.75
    beam_inflation  = gd.beam_inflation
    power_threshold = gd.power_threshold

    for ii in CartesianIndices((ny, nx))
        j_y, i_x = ii.I
        yx_point = [gridpoints[1, j_y, i_x, 2], gridpoints[1, j_y, i_x, 3]]
        eff_h_roi = horizontal_roi; eff_v_roi = vertical_roi
        if beam_inflation > 0.0
            origin_dist = Daisho.euclidean(yx_point, [0.0, 0.0])
            eff_h_roi = max(beam_inflation * origin_dist, horizontal_roi)
            eff_v_roi = max(beam_inflation * origin_dist, vertical_roi)
        end
        gates = Daisho.inrange(balltree, yx_point, eff_h_roi)
        isempty(gates) && continue
        for k_z in 1:nz
            grid_z = gridpoints[k_z, j_y, i_x, 1]
            any_scanned = false
            for g_flat in gates
                abs(beams[g_flat, 4] - grid_z) > eff_v_roi && continue
                ray = Daisho._ray_of(g_flat, n_gates_s)
                gate = Daisho._gate_in_ray(g_flat, n_gates_s)
                if !ismissing(Daisho._gate_value(sweep, missing_key, ray, gate))
                    any_scanned = true; break
                end
            end
            if any_scanned
                for m in 1:n_fields
                    accum.coverage[m, k_z, j_y, i_x] == Int8(0) &&
                        (accum.coverage[m, k_z, j_y, i_x] = Int8(1))
                end
            else
                continue
            end
            for g_flat in gates
                ray = Daisho._ray_of(g_flat, n_gates_s)
                gate_in = Daisho._gate_in_ray(g_flat, n_gates_s)
                ismissing(Daisho._gate_value(sweep, valid_key, ray, gate_in)) && continue
                _, _, total_weight = Daisho._gate_grid_geometry(grid_z, yx_point,
                    radar_zyx, beams, g_flat, horizontal_roi, vertical_roi, power_threshold)
                total_weight > 0.0 || continue
                for m in 1:n_fields
                    fname = accum.fields[m]
                    v = Daisho._gate_value(sweep, fname, ray, gate_in)
                    ismissing(v) && continue
                    mode = accum.grid_type[fname]
                    accum.coverage[m, k_z, j_y, i_x] = Int8(2)
                    if mode === :linear
                        linear_z = 10.0 ^ (v / 10.0)
                        accum.weighted_sum[m, k_z, j_y, i_x] += total_weight * linear_z
                        accum.weight_total[m, k_z, j_y, i_x] += total_weight
                    elseif mode === :nearest
                        if total_weight > accum.weight_total[m, k_z, j_y, i_x]
                            accum.weighted_sum[m, k_z, j_y, i_x] = v
                            accum.weight_total[m, k_z, j_y, i_x] = total_weight
                        end
                    else
                        accum.weighted_sum[m, k_z, j_y, i_x] += total_weight * v
                        accum.weight_total[m, k_z, j_y, i_x] += total_weight
                    end
                end
            end
        end
    end
    return accum
end

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

    @testset "DaishoParameters constructor uses sorted field order" begin
        p = DaishoParameters()
        spec = _spec(:volume_3d)
        acc = GridAccumulator(spec, p)
        ordered = Daisho._ordered_fields(p)
        @test acc.fields == [fs.name for fs in ordered]
        @test acc.grid_type == Dict(fs.name => Daisho.interp_of(fs) for fs in ordered)
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
            p.io, p.provided)

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
            ordered_names = [fs.name for fs in Daisho._ordered_fields(p)]
            legacy_r, _ = as_legacy_radar(v; field_names = ordered_names)
            Daisho.grid_radar_volume(legacy_r, legacy_file, v.time_coverage_start, p)
            ds = NCDataset(legacy_file, "r")
            try
                # Reconstruct legacy radar_grid from variables. The writer
                # writes one variable per moment; we can read them back.
                legacy_grids = Dict{String,Array{Float64,4}}()
                for fname in ordered_names
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
                for (m, fname) in enumerate(acc.fields)
                    haskey(legacy_grids, fname) || continue
                    leg = legacy_grids[fname]
                    # legacy_grids dims: (X, Y, Z, time). Permute to (Z, Y, X).
                    leg_zyx = permutedims(leg[:, :, :, 1], (3, 2, 1))
                    for ix in CartesianIndices((new_cart.zdim, new_cart.ydim, new_cart.xdim))
                        a = grid[m, ix]
                        b = leg_zyx[ix]
                        if a > -9000 && b > -9000
                            mode = acc.grid_type[fname]
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
            p.io, p.provided)
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

    @testset "ScalarGridAccumulator rename + GridAccumulator alias" begin
        @test GridAccumulator === Daisho.ScalarGridAccumulator
        @test ScalarGridAccumulator <: Daisho.FieldAccumulator
        spec = _spec(:volume_3d)
        acc = ScalarGridAccumulator(spec, ["DBZ"], Dict("DBZ" => :weighted))
        @test acc isa GridAccumulator
        @test acc isa Daisho.FieldAccumulator
        # The alias keyword constructor still resolves to the same type.
        @test GridAccumulator(spec, ["DBZ"], Dict("DBZ" => :weighted)) isa
              ScalarGridAccumulator
    end

    @testset "unified traversal matches legacy 3D worker exactly" begin
        # The unified `_grid_sweep_products_3d!` must reproduce the pre-refactor
        # `_grid_sweep_3d!` (held as `_ref_grid_sweep_3d!`) byte-for-byte.
        p = DaishoParameters()
        # synthetic_volume gates sit at low altitude (~7–55 m, ranges 400–1500 m),
        # so the grid must hug the radar near the surface to capture them.
        v = synthetic_volume(n_sweeps = 2, n_rays = 72, n_gates = 12)
        for shape in (:volume_3d, :latlon_3d)
            gs = if shape === :volume_3d
                GridSpec(shape = :volume_3d,
                    reference_latitude = v.latitude, reference_longitude = v.longitude,
                    x_axis = collect(Float64, -1200.0:300.0:1200.0),
                    y_axis = collect(Float64, -1200.0:300.0:1200.0),
                    z_axis = [0.0, 60.0, 120.0])
            else
                GridSpec(shape = :latlon_3d,
                    reference_latitude = v.latitude, reference_longitude = v.longitude,
                    x_axis = collect(Float64, 1:9), y_axis = collect(Float64, 1:9),
                    z_axis = [0.0, 60.0, 120.0],
                    lat_axis = collect(Float64, v.latitude .+ (-0.008:0.002:0.008)),
                    lon_axis = collect(Float64, v.longitude .+ (-0.008:0.002:0.008)))
            end

            acc_new = GridAccumulator(gs, p)
            for s in eachindex(v.sweeps); grid_sweep!(acc_new, v, s, p); end

            acc_ref = GridAccumulator(gs, p)
            for s in eachindex(v.sweeps)
                sweep = v.sweeps[s]
                _ref_grid_sweep_3d!(acc_ref, sweep, p,
                    Float64(v.latitude), Float64(v.longitude), Float64(v.altitude))
            end

            @test acc_new.weighted_sum == acc_ref.weighted_sum
            @test acc_new.weight_total == acc_ref.weight_total
            @test acc_new.coverage     == acc_ref.coverage
            # Non-vacuous: the fixture actually fills cells.
            @test any(acc_new.coverage .== Int8(2))
        end
    end

    @testset "beamwidth scales the angular gate weight" begin
        @test Daisho._beam_coef(1.0) == Daisho.BEAM_COEF_1DEG
        @test Daisho._beam_coef(2.0) ≈ Daisho.BEAM_COEF_1DEG / 2
        @test Daisho._beam_coef(0.0) == Daisho.BEAM_COEF_1DEG    # non-positive ⇒ 1° fallback
        @test Daisho._beam_coef(-3.0) == Daisho.BEAM_COEF_1DEG

        p = DaishoParameters()
        v = synthetic_volume(n_sweeps = 2, n_rays = 72, n_gates = 12)
        spec = GridSpec(shape = :volume_3d,
            reference_latitude = v.latitude, reference_longitude = v.longitude,
            x_axis = collect(Float64, -1200.0:300.0:1200.0),
            y_axis = collect(Float64, -1200.0:300.0:1200.0), z_axis = [0.0, 60.0, 120.0])
        rl, ro, ra = Float64(v.latitude), Float64(v.longitude), Float64(v.altitude)

        # beam_width = 1.0 reproduces the default (no-kwarg, legacy 79.43) gridding.
        acc_def = GridAccumulator(spec, p); acc_bw1 = GridAccumulator(spec, p)
        for s in eachindex(v.sweeps)
            grid_sweep!(acc_def, v.sweeps[s], p; ref_latitude = rl, ref_longitude = ro,
                        ref_altitude = ra)
            grid_sweep!(acc_bw1, v.sweeps[s], p; ref_latitude = rl, ref_longitude = ro,
                        ref_altitude = ra, beam_width = 1.0)
        end
        @test acc_def.weighted_sum == acc_bw1.weighted_sum
        @test acc_def.coverage == acc_bw1.coverage
        @test any(acc_def.coverage .== Int8(2))   # non-vacuous baseline

        # A wider 2° beam admits a superset of gates at higher angular weight ⇒
        # strictly more total weight, at least as many covered cells, and a
        # different weighted field.
        acc_bw2 = GridAccumulator(spec, p)
        for s in eachindex(v.sweeps)
            grid_sweep!(acc_bw2, v.sweeps[s], p; ref_latitude = rl, ref_longitude = ro,
                        ref_altitude = ra, beam_width = 2.0)
        end
        @test sum(acc_bw2.weight_total) > sum(acc_def.weight_total)
        @test count(>(Int8(0)), acc_bw2.coverage) >= count(>(Int8(0)), acc_def.coverage)
        @test acc_bw2.weighted_sum != acc_def.weighted_sum
    end

    @testset "Volume overload reads beamwidth from radar_parameters" begin
        p = DaishoParameters()
        v = synthetic_volume(n_sweeps = 1, n_rays = 72, n_gates = 12)
        sweep = v.sweeps[1]
        spec = GridSpec(shape = :volume_3d,
            reference_latitude = v.latitude, reference_longitude = v.longitude,
            x_axis = collect(Float64, -1200.0:300.0:1200.0),
            y_axis = collect(Float64, -1200.0:300.0:1200.0), z_axis = [0.0, 60.0, 120.0])

        vol_bw = Volume(latitude = v.latitude, longitude = v.longitude,
            altitude = v.altitude, time_coverage_start = v.time_coverage_start,
            time_coverage_end = v.time_coverage_end,
            instrument_name = "BW", sweeps = [sweep],
            radar_parameters = RadarParameters(beam_width_h = 2.0))
        vol_no = Volume(latitude = v.latitude, longitude = v.longitude,
            altitude = v.altitude, time_coverage_start = v.time_coverage_start,
            time_coverage_end = v.time_coverage_end,
            instrument_name = "NOBW", sweeps = [sweep])
        @test Daisho._volume_beam_width(vol_bw) == 2.0
        @test Daisho._volume_beam_width(vol_no) == 1.0   # absent ⇒ legacy default

        # Volume overload's beamwidth matches the explicit sweep-level kwarg.
        acc_v = GridAccumulator(spec, p); grid_sweep!(acc_v, vol_bw, 1, p)
        acc_s = GridAccumulator(spec, p)
        grid_sweep!(acc_s, sweep, p; ref_latitude = Float64(v.latitude),
            ref_longitude = Float64(v.longitude), ref_altitude = Float64(v.altitude),
            beam_width = 2.0)
        @test acc_v.weighted_sum == acc_s.weighted_sum
        @test acc_v.coverage == acc_s.coverage
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

    @testset "[io] sentinels are authoritative" begin
        @testset "GridAccumulator(spec, p) carries custom [io]" begin
            p0 = DaishoParameters()
            p = Daisho.DaishoParameters(p0.moments, p0.qc, p0.gridding, p0.grid,
                Daisho.IOParameters(fill_value = -777.0, undetect = -8888.0),
                p0.provided)
            acc = GridAccumulator(_spec(:volume_3d), p)
            @test acc.fill_value == -777.0
            @test acc.undetect == -8888.0
        end

        @testset "finalize_grid emits exactly the accumulator sentinels" begin
            acc = GridAccumulator(_spec(:volume_3d), ["DBZ"],
                Dict("DBZ" => :weighted);
                fill_value = -777.0, undetect = -8888.0)
            # Valid cell: scanned + detected, real weighted value.
            acc.coverage[1, 1, 1, 1]     = Int8(2)
            acc.weight_total[1, 1, 1, 1] = 2.0
            acc.weighted_sum[1, 1, 1, 1] = 7.0      # → 3.5
            # Scanned-no-echo cell → undetect.
            acc.coverage[1, 2, 2, 2]     = Int8(1)
            # Unscanned cell → true missing (coverage 0, the default).
            grid = finalize_grid(acc)
            @test grid[1, 1, 1, 1] == 3.5
            @test grid[1, 2, 2, 2] == -8888.0
            @test grid[1, 2, 3, 4] == -777.0
        end

        @testset "merge refuses mismatched sentinels" begin
            spec = _spec(:volume_3d)
            a = GridAccumulator(spec, ["DBZ"], Dict("DBZ" => :weighted);
                fill_value = -32768.0, undetect = -9999.0)
            b = GridAccumulator(spec, ["DBZ"], Dict("DBZ" => :weighted);
                fill_value = -777.0, undetect = -9999.0)
            @test_throws ArgumentError merge_accumulators!(a, b)
            c = GridAccumulator(spec, ["DBZ"], Dict("DBZ" => :weighted);
                fill_value = -32768.0, undetect = -8888.0)
            @test_throws ArgumentError merge_accumulators!(a, c)
        end

        @testset "load_accumulator rejects schema mismatch" begin
            spec = _spec(:column_1d)
            acc = GridAccumulator(spec, ["DBZ"], Dict("DBZ" => :weighted))
            bad = Daisho.GridAccumulator(
                grid_spec = acc.grid_spec, fields = acc.fields,
                grid_type = acc.grid_type, field_folds = acc.field_folds,
                weighted_sum = acc.weighted_sum, weight_total = acc.weight_total,
                coverage = acc.coverage, sweeps = acc.sweeps,
                schema_version = 1,
                fill_value = acc.fill_value, undetect = acc.undetect)
            mktemp() do path, io
                close(io)
                save_accumulator(path, bad)
                err = try; load_accumulator(path); nothing; catch e; e; end
                @test err isa ArgumentError
                @test occursin("schema version", err.msg)
            end
        end

        @testset "gridded NetCDF carries custom _FillValue/_Undetect" begin
            p0 = DaishoParameters()
            small_cart = Daisho.CartesianGridParameters(
                xmin = -1500.0, xincr = 500.0, xdim = 7,
                ymin = -1500.0, yincr = 500.0, ydim = 7,
                zmin = 500.0,   zincr = 500.0, zdim = 3)
            mk(io) = Daisho.DaishoParameters(p0.moments, p0.qc, p0.gridding,
                Daisho.GridParameters(cartesian = small_cart,
                                      latlon = p0.grid.latlon,
                                      rhi = p0.grid.rhi,
                                      spectral = p0.grid.spectral),
                io, p0.provided)
            v = synthetic_volume(n_sweeps = 1, n_rays = 12, n_gates = 4)
            # Custom sentinels.
            p = mk(Daisho.IOParameters(fill_value = -777.0, undetect = -8888.0))
            f1 = tempname() * "_customio.nc"
            try
                Daisho.grid_radar_volume(v, f1, v.time_coverage_start, p)
                ds = NCDataset(f1, "r")
                try
                    @test haskey(ds, "DBZ")
                    a = ds["DBZ"].attrib
                    @test Float32(a["_FillValue"]) == Float32(-777.0)
                    @test Float32(a["_Undetect"]) == Float32(-8888.0)
                finally
                    close(ds)
                end
            finally
                rm(f1, force = true)
            end
            # Default sentinels regression guard.
            pd = mk(Daisho.IOParameters())
            f2 = tempname() * "_defaultio.nc"
            try
                Daisho.grid_radar_volume(v, f2, v.time_coverage_start, pd)
                ds = NCDataset(f2, "r")
                try
                    a = ds["DBZ"].attrib
                    @test Float32(a["_FillValue"]) == Float32(-32768.0)
                    @test Float32(a["_Undetect"]) == Float32(-9999.0)
                finally
                    close(ds)
                end
            finally
                rm(f2, force = true)
            end
        end
    end
end
