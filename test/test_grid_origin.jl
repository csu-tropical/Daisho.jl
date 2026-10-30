# Tests for the [grid.origin] override and the moving (storm-relative) frame
# (issue #18).

@testset "Grid origin and moving frame" begin
    t_ref = DateTime(2024, 9, 3, 15, 0, 0)

    # DaishoParameters with a small Cartesian grid and the given origin table.
    function _p_origin(origin::OriginParameters = OriginParameters();
                       cart = Daisho.CartesianGridParameters(
                           xmin = -1500.0, xincr = 500.0, xdim = 7,
                           ymin = -1500.0, yincr = 500.0, ydim = 7,
                           zmin = 500.0,   zincr = 500.0, zdim = 3))
        p = DaishoParameters()
        return Daisho.DaishoParameters(p.moments, p.qc, p.gridding,
            Daisho.GridParameters(cartesian = cart, latlon = p.grid.latlon,
                                  rhi = p.grid.rhi, springsteel = p.grid.springsteel,
                                  origin = origin),
            p.io, p.provided)
    end

    @testset "resolve_grid_origin" begin
        @test resolve_grid_origin(OriginParameters(), 16.9, -25.0) == (16.9, -25.0)
        o = OriginParameters(latitude_deg = 18.0, longitude_deg = 150.0)
        @test resolve_grid_origin(o, 16.9, -25.0) == (18.0, 150.0)
        @test resolve_grid_origin(_p_origin(o), 16.9, -25.0) == (18.0, 150.0)
    end

    @testset "build_grid_spec honours the origin override on every shape" begin
        v = synthetic_volume(n_sweeps = 1)                  # radar at (16.886, -24.988)
        p0 = _p_origin()
        o = OriginParameters(latitude_deg = 17.0, longitude_deg = -24.5)
        p1 = _p_origin(o)
        for shape in (:volume_3d, :ppi_2d, :composite_2d, :column_1d, :rhi_2d)
            s0 = build_grid_spec(shape, v, p0)
            s1 = build_grid_spec(shape, v, p1)
            @test (s0.reference_latitude, s0.reference_longitude) == (16.886, -24.988)
            @test (s1.reference_latitude, s1.reference_longitude) == (17.0, -24.5)
            @test s1.reference_time === nothing && s1.motion_east_mps == 0.0
        end
        # latlon keeps its degincr snap, applied to the resolved origin.
        s = build_grid_spec(:latlon_3d, v, _p_origin(OriginParameters(
            latitude_deg = 17.013, longitude_deg = -24.507)))
        dinc = p1.grid.latlon.degincr
        @test s.reference_latitude ≈ 17.013 - rem(17.013, dinc)
        @test s.reference_longitude ≈ -24.507 - rem(-24.507, dinc)
        # Origin equal to the radar position: identical spec (no-op).
        same = _p_origin(OriginParameters(latitude_deg = 16.886, longitude_deg = -24.988))
        @test Daisho._grid_spec_equal(build_grid_spec(:volume_3d, v, same),
                                      build_grid_spec(:volume_3d, v, p0))
    end

    @testset "frame fields: propagate, merge strictly, reject on RHI" begin
        v = synthetic_volume(n_sweeps = 1)
        om = OriginParameters(reference_time = t_ref, motion_east_mps = 8.0,
                              motion_north_mps = -3.0)
        pm = _p_origin(om)
        s = build_grid_spec(:volume_3d, v, pm)
        @test s.reference_time == t_ref
        @test (s.motion_east_mps, s.motion_north_mps) == (8.0, -3.0)
        # Motion with the default origin: origin stays the radar position.
        @test (s.reference_latitude, s.reference_longitude) == (16.886, -24.988)
        # Accumulators built under different frames refuse to merge.
        s0 = build_grid_spec(:volume_3d, v, _p_origin())
        @test !Daisho._grid_spec_equal(s, s0)
        a = ScalarGridAccumulator(s, pm); b = ScalarGridAccumulator(s0, _p_origin())
        @test_throws ArgumentError merge_accumulators!(a, b)
        # RHI: an origin override is fine, a nonzero motion is not.
        @test_throws ArgumentError build_grid_spec(:rhi_2d, v, pm)
        acc = ScalarGridAccumulator(GridSpec(shape = :rhi_2d, reference_latitude = 16.886,
            reference_longitude = -24.988, x_axis = [0.0, 250.0], y_axis = [0.0],
            z_axis = [0.0, 250.0], rhi_azimuth = 0.0, reference_time = t_ref,
            motion_east_mps = 1.0), pm)
        @test_throws ArgumentError grid_sweep!(acc, v, 1, pm)
        # Schema 3 round-trip keeps the frame.
        f = tempname() * ".jld2"
        try
            save_accumulator(f, a)
            r = load_accumulator(f)
            @test Daisho._grid_spec_equal(r.grid_spec, s)
        finally
            rm(f, force = true)
        end
    end

    @testset "_sweep_zyx_and_beams shifts each beam by −(u, v)·Δt" begin
        v = synthetic_volume(n_sweeps = 1, n_rays = 12, n_gates = 4)
        sw = v.sweeps[1]
        TM = CoordRefSystems.shift(TransverseMercator{1.0, v.latitude, WGS84Latest},
                                   lonₒ = v.longitude)
        args = (sw, v.latitude, v.longitude, v.altitude, TM)
        g0, z0, b0, _, _ = Daisho._sweep_zyx_and_beams(args...)
        # Zero motion, or no frame at all: bit-identical.
        _, zz, bz, _, _ = Daisho._sweep_zyx_and_beams(args...;
            reference_time = t_ref, motion_east_mps = 0.0, motion_north_mps = 0.0)
        @test zz == z0 && bz == b0
        # Moving frame referenced to ray 5's time.
        u, w = 8.0, -3.0
        tr = sw.time[5]
        g1, z1, b1, ng, nr = Daisho._sweep_zyx_and_beams(args...;
            reference_time = tr, motion_east_mps = u, motion_north_mps = w)
        @test g1 == g0 && b1 == b0                  # grid origin, beams unchanged
        for ray in 1:nr, gate in 1:ng
            idx = (ray - 1) * ng + gate
            dt = Dates.value(sw.time[ray] - tr) / 1000
            @test z1[idx][1] == z0[idx][1]                      # altitude unchanged
            @test z1[idx][2] ≈ z0[idx][2] - w * dt atol = 1e-9  # y
            @test z1[idx][3] ≈ z0[idx][3] - u * dt atol = 1e-9  # x
        end
        @test z1[(5 - 1) * ng + 1] == z0[(5 - 1) * ng + 1]      # Δt = 0 ray untouched
    end

    @testset "moving frame collapses a moving echo onto the origin" begin
        # One stationary radar, two sweeps 10 min apart. A compact echo moves
        # at (u, v); it is at P0 at t_ref (the first sweep's middle ray).
        u, w = 8.0, 5.0
        P0 = (4000.0, 1000.0)                                   # (x, y) m
        v = synthetic_volume(n_sweeps = 2, n_rays = 360, n_gates = 110)
        for (s, sw) in enumerate(v.sweeps)
            sw.time .= [t_ref + Minute(10) * (s - 1) + Second(r - 180) for r in 1:360]
        end
        TM = CoordRefSystems.shift(TransverseMercator{1.0, v.latitude, WGS84Latest},
                                   lonₒ = v.longitude)
        for sw in v.sweeps
            _, zyx, beams, ng, nr = Daisho._sweep_zyx_and_beams(sw, v.latitude,
                v.longitude, v.altitude, TM)
            yx = Daisho._sweep_gate_yx(zyx, beams)              # earth-frame gate (y, x)
            for name in ("DBZ", "DBZ_QC")
                d = sw.fields[name].data
                for ray in 1:nr, gate in 1:ng
                    dt = Dates.value(sw.time[ray] - t_ref) / 1000
                    idx = (ray - 1) * ng + gate
                    ex = yx[2, idx] - (P0[1] + u * dt); ey = yx[1, idx] - (P0[2] + w * dt)
                    d[ray, gate] = hypot(ex, ey) < 800.0 ? 30.0f0 : NaN32
                end
            end
        end
        cart = Daisho.CartesianGridParameters(
            xmin = -2000.0, xincr = 500.0, xdim = 33,
            ymin = -4000.0, yincr = 500.0, ydim = 21,
            zmin = 0.0,     zincr = 500.0, zdim = 3)
        # Centroid (x, y) of the detected DBZ cells from one sweep.
        function centroid(p, s)
            spec = build_grid_spec(:volume_3d, v, p)
            acc = ScalarGridAccumulator(spec, p)
            grid_sweep!(acc, v, s, p)
            m = findfirst(==("DBZ"), acc.fields)
            hit = findall(c -> c == Int8(2), acc.coverage[m, :, :, :])
            isempty(hit) && return (NaN, NaN)
            xs = [spec.x_axis[I[3]] for I in hit]; ys = [spec.y_axis[I[2]] for I in hit]
            return (sum(xs) / length(xs), sum(ys) / length(ys))
        end
        p_earth = _p_origin(; cart = cart)
        p_frame = _p_origin(OriginParameters(reference_time = t_ref,
            motion_east_mps = u, motion_north_mps = w); cart = cart)
        e1, e2 = centroid(p_earth, 1), centroid(p_earth, 2)
        f1, f2 = centroid(p_frame, 1), centroid(p_frame, 2)
        # Earth frame: the echo has moved ≈ (u, v)·600 s between sweeps.
        @test e2[1] - e1[1] ≈ u * 600 atol = 500
        @test e2[2] - e1[2] ≈ w * 600 atol = 500
        # Storm-relative frame: both sweeps put it at P0.
        for c in (f1, f2)
            @test c[1] ≈ P0[1] atol = 500
            @test c[2] ≈ P0[2] atol = 500
        end
    end

    @testset "multi-radar Springsteel origin: override replaces the centroid" begin
        vols = [synthetic_volume(n_sweeps = 1, lat = 16.8, lon = -25.0),
                synthetic_volume(n_sweeps = 1, lat = 17.0, lon = -24.8)]
        @test Daisho._volumes_reference_position(vols) == (16.9, -24.9)
        o = OriginParameters(latitude_deg = 17.2, longitude_deg = -24.6)
        @test Daisho._volumes_origin(_p_origin(o), vols) == (17.2, -24.6)
        @test Daisho._volumes_origin(_p_origin(), vols) == (16.9, -24.9)
    end

    @testset "legacy radar path: origin honoured, motion rejected" begin
        v = synthetic_volume(n_sweeps = 1, n_rays = 36, n_gates = 6)
        radar, _ = Daisho.as_legacy_radar(v)
        f0 = tempname() * ".nc"; f1 = tempname() * ".nc"; f2 = tempname() * ".nc"
        try
            # Origin equal to the radar position: identical output.
            Daisho.grid_radar_volume(radar, f0, v.time_coverage_start, _p_origin())
            Daisho.grid_radar_volume(radar, f1, v.time_coverage_start,
                _p_origin(OriginParameters(latitude_deg = v.latitude, longitude_deg = v.longitude)))
            NCDataset(f0) do a
                NCDataset(f1) do b
                    @test isequal(a["DBZ"][:], b["DBZ"][:])
                end
            end
            pm = _p_origin(OriginParameters(reference_time = t_ref, motion_east_mps = 2.0))
            @test_throws ArgumentError Daisho.grid_radar_volume(radar, f2,
                v.time_coverage_start, pm)
        finally
            foreach(f -> rm(f, force = true), (f0, f1, f2))
        end
    end

    @testset "written files record the origin and frame" begin
        v = synthetic_volume(n_sweeps = 1, n_rays = 36, n_gates = 6)
        o = OriginParameters(latitude_deg = 16.9, longitude_deg = -24.9,
            reference_time = t_ref, motion_east_mps = 8.0, motion_north_mps = -3.0)
        f0 = tempname() * ".nc"; f1 = tempname() * ".nc"
        try
            Daisho.grid_radar_volume(v, f0, v.time_coverage_start, _p_origin())
            Daisho.grid_radar_volume(v, f1, v.time_coverage_start, _p_origin(o))
            NCDataset(f0) do ds
                @test !haskey(ds.attrib, "frame_motion_east_mps")
                @test !haskey(ds.attrib, "origin_reference_time")
            end
            NCDataset(f1) do ds
                @test ds.attrib["origin_reference_time"] == "2024-09-03T15:00:00Z"
                @test ds.attrib["frame_motion_east_mps"] == 8.0
                @test ds.attrib["frame_motion_north_mps"] == -3.0
                gm = ds["grid_mapping"].attrib
                @test gm["latitude_of_projection_origin"] ≈ 16.9
                @test gm["longitude_of_central_meridian"] ≈ -24.9
            end
        finally
            foreach(f -> rm(f, force = true), (f0, f1))
        end

        # Multi-radar Springsteel: the override is the written origin.
        p = _p_origin(OriginParameters(latitude_deg = 16.95, longitude_deg = -24.95))
        sgrid = Daisho.create_radar_grid("RRR", Daisho.field_index_dict(p);
            xmin = -1500.0, xmax = 1500.0, xdim = 4, ymin = -1500.0, ymax = 1500.0,
            ydim = 4, zmin = 0.0, zmax = 300.0, zdim = 3)
        f2 = tempname() * ".nc"
        try
            Daisho.grid_radar_volume_spectral([v, synthetic_volume(n_sweeps = 1,
                n_rays = 36, n_gates = 6, lat = 16.8)], f2, nothing, sgrid, p)
            NCDataset(f2) do ds
                @test ds.attrib["reference_latitude"] ≈ 16.95
                @test ds.attrib["reference_longitude"] ≈ -24.95
            end
        finally
            rm(f2, force = true)
        end
    end
end
