@testset "Gridding" begin

    @testset "initialize_regular_grid - 3D" begin
        grid = Daisho.initialize_regular_grid(
            -1000.0, 500.0, 5,   # x: -1000 to 1000
            -1000.0, 500.0, 5,   # y: -1000 to 1000
            0.0, 500.0, 3         # z: 0 to 1000
        )

        @test size(grid) == (3, 5, 5, 3)  # zdim, ydim, xdim, 3

        # Check z values
        @test grid[1, 1, 1, 1] ≈ 0.0
        @test grid[2, 1, 1, 1] ≈ 500.0
        @test grid[3, 1, 1, 1] ≈ 1000.0

        # Check y values
        @test grid[1, 1, 1, 2] ≈ -1000.0
        @test grid[1, 3, 1, 2] ≈ 0.0
        @test grid[1, 5, 1, 2] ≈ 1000.0

        # Check x values
        @test grid[1, 1, 1, 3] ≈ -1000.0
        @test grid[1, 1, 3, 3] ≈ 0.0
        @test grid[1, 1, 5, 3] ≈ 1000.0

        # Z values should be the same for all y, x
        @test grid[2, 1, 1, 1] == grid[2, 3, 4, 1]
    end

    @testset "initialize_regular_grid - 2D" begin
        grid = Daisho.initialize_regular_grid(
            -500.0, 250.0, 5,   # x
            -500.0, 250.0, 5    # y
        )

        @test size(grid) == (5, 5, 2)  # ydim, xdim, 2

        # Check y values
        @test grid[1, 1, 1] ≈ -500.0
        @test grid[3, 1, 1] ≈ 0.0

        # Check x values
        @test grid[1, 1, 2] ≈ -500.0
        @test grid[1, 3, 2] ≈ 0.0
    end

    @testset "initialize_regular_grid - 1D" begin
        grid = Daisho.initialize_regular_grid(0.0, 500.0, 5)

        @test size(grid) == (5,)
        @test grid[1] ≈ 0.0
        @test grid[2] ≈ 500.0
        @test grid[5] ≈ 2000.0
    end

    @testset "initialize_regular_grid - latlon (bug fix verification)" begin
        # This test verifies the j[1] -> j bug fix
        # Should not error at runtime anymore
        grid = Daisho.initialize_regular_grid(
            16.0, -25.0,        # reference lat, lon
            -25.5, 3,           # lonmin, londim
            15.5, 3,            # latmin, latdim
            0.5,                # degincr
            0.0, 500.0, 2       # zmin, zincr, zdim
        )

        @test size(grid) == (2, 3, 3, 3)  # zdim, latdim, londim, 3

        # Z values should vary with first index
        @test grid[1, 1, 1, 1] ≈ 0.0
        @test grid[2, 1, 1, 1] ≈ 500.0

        # Y and X values should come from coordinate transform
        @test !isnan(grid[1, 1, 1, 2])
        @test !isnan(grid[1, 1, 1, 3])
    end

    @testset "initialize_regular_grid - single point" begin
        grid = Daisho.initialize_regular_grid(0.0, 1.0, 1, 0.0, 1.0, 1, 0.0, 1.0, 1)
        @test size(grid) == (1, 1, 1, 3)
        @test grid[1, 1, 1, 1] ≈ 0.0
        @test grid[1, 1, 1, 2] ≈ 0.0
        @test grid[1, 1, 1, 3] ≈ 0.0
    end

    @testset "get_beam_info" begin
        vol = make_synthetic_radar(n_rays=4, n_gates=5)
        vol.azimuth[:] .= [0.0, 90.0, 180.0, 270.0]
        vol.elevation[:] .= 1.0
        vol.altitude[:] .= 50.0

        beams = Daisho.get_beam_info(vol)

        # Should have n_gates * n_rays rows, 4 columns (az, el, range, height)
        @test size(beams, 1) == 20
        @test size(beams, 2) == 4

        # Azimuth should be in radians
        @test beams[1, 1] ≈ 0.0  atol=0.01  # First ray, azimuth 0
        @test beams[6, 1] ≈ deg2rad(90.0) atol=0.01  # Second ray

        # Elevation should be in radians
        @test beams[1, 2] ≈ deg2rad(1.0) atol=0.01

        # Range should match radar range
        @test beams[1, 3] ≈ 400.0f0
        @test beams[2, 3] ≈ 500.0f0

        # Height should be positive
        @test beams[1, 4] > 0.0
    end

    @testset "appx_inverse_projection" begin
        ref_lat = 16.886
        ref_lon = -24.988

        # At the origin, should return the reference point
        lat, lon = Daisho.appx_inverse_projection(ref_lat, ref_lon, [0.0, 0.0])
        @test lat ≈ ref_lat atol=1e-6
        @test lon ≈ ref_lon atol=1e-6

        # A point 1 km north should increase latitude
        lat, lon = Daisho.appx_inverse_projection(ref_lat, ref_lon, [1000.0, 0.0])
        @test lat > ref_lat
        @test abs(lon - ref_lon) < 1e-6

        # A point 1 km east should increase longitude
        lat, lon = Daisho.appx_inverse_projection(ref_lat, ref_lon, [0.0, 1000.0])
        @test lon > ref_lon
        @test abs(lat - ref_lat) < 1e-6
    end

    @testset "appx_inverse_projection - round-trip consistency" begin
        ref_lat = 40.0
        ref_lon = -105.0

        # Test at a known offset
        y_offset = 50000.0  # 50 km north
        x_offset = 30000.0  # 30 km east

        lat, lon = Daisho.appx_inverse_projection(ref_lat, ref_lon, [y_offset, x_offset])

        # The returned lat should be roughly 0.45 degrees north (50km / ~111km/deg)
        @test lat ≈ ref_lat + y_offset / 111000.0 atol=0.1
        # The returned lon should be roughly 0.35 degrees east (30km / ~85km/deg at 40°N)
        @test lon > ref_lon
    end

    @testset "get_radar_zyx" begin
        vol = make_synthetic_radar(n_rays=3, n_gates=5, lat=16.886, lon=-24.988, alt=50.0)

        TM = CoordRefSystems.shift(TransverseMercator{1.0,16.886,WGS84Latest}, lonₒ=-24.988)
        radar_zyx = Daisho.get_radar_zyx(16.886, -24.988, vol, TM)

        # Should have n_gates * n_rays elements
        @test length(radar_zyx) == 15

        # Each element should be a 3-element vector [z, y, x]
        @test length(radar_zyx[1]) == 3

        # Z should be the altitude
        @test radar_zyx[1][1] ≈ 50.0 atol=1.0

        # For a stationary radar at the reference point, y and x should be near 0
        @test abs(radar_zyx[1][2]) < 100.0  # meters
        @test abs(radar_zyx[1][3]) < 100.0
    end

    @testset "radar_arrays" begin
        vol = make_synthetic_radar(n_rays=3, n_gates=5)
        TM = CoordRefSystems.shift(TransverseMercator{1.0,16.886,WGS84Latest}, lonₒ=-24.988)

        grid_origin, radar_zyx, beams = Daisho.radar_arrays(16.886, -24.988, vol, TM)

        @test length(radar_zyx) == 15
        @test size(beams, 1) == 15
        @test size(beams, 2) == 4
    end

    @testset "radar_balltree_yx" begin
        vol = make_synthetic_radar(n_rays=4, n_gates=5)
        TM = CoordRefSystems.shift(TransverseMercator{1.0,16.886,WGS84Latest}, lonₒ=-24.988)

        radar_zyx = Daisho.get_radar_zyx(16.886, -24.988, vol, TM)
        beams = Daisho.get_beam_info(vol)
        balltree = Daisho.radar_balltree_yx(vol, radar_zyx, beams)

        # Query near the radar location - should find some gates
        gates = NearestNeighbors.inrange(balltree, [0.0, 0.0], 50000.0)
        @test length(gates) > 0
    end

    @testset "radar_balltree_r" begin
        vol = make_synthetic_radar(n_rays=4, n_gates=5)
        TM = CoordRefSystems.shift(TransverseMercator{1.0,16.886,WGS84Latest}, lonₒ=-24.988)

        radar_zyx = Daisho.get_radar_zyx(16.886, -24.988, vol, TM)
        beams = Daisho.get_beam_info(vol)
        balltree = Daisho.radar_balltree_r(vol, radar_zyx, beams)

        # Query near range 0 - should find some gates
        gates = NearestNeighbors.inrange(balltree, [500.0], 500.0)
        @test length(gates) > 0
    end

    @testset "Volume overloads dispatch via bridge" begin
        # Smoke-test that the Volume-typed driver methods exist and dispatch.
        v = synthetic_volume(n_sweeps=1, n_rays=8, n_gates=5)
        # The Volume-typed methods are defined; just check that they resolve.
        @test hasmethod(Daisho.grid_radar_volume,
            Tuple{Volume,AbstractString,Any,DaishoParameters})
        @test hasmethod(Daisho.grid_radar_ppi,
            Tuple{Volume,AbstractString,Any,DaishoParameters})
        @test hasmethod(Daisho.grid_radar_rhi,
            Tuple{Volume,AbstractString,Any,DaishoParameters})
        @test hasmethod(Daisho.grid_radar_column,
            Tuple{Volume,AbstractString,Any,DaishoParameters})
        @test hasmethod(Daisho.grid_radar_composite,
            Tuple{Volume,AbstractString,Any,DaishoParameters})
        @test hasmethod(Daisho.grid_radar_latlon_volume,
            Tuple{Volume,AbstractString,Any,DaishoParameters})
        # Bridge: Volume → legacy radar should produce a sensibly-shaped struct.
        legacy, names = as_legacy_radar(v)
        @test length(legacy.azimuth) == 8
        @test "DBZ" in names
    end

end
