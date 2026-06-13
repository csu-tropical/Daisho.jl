@testset "Range gate-inclusion filter" begin
    # Small grid; PPI/volume fall back to [grid.cartesian]. The z axis (0/25/50 m)
    # overlaps the synthetic gates' low altitudes so the 3D volume has coverage.
    # Default [fields]: DBZ (define_detection), VEL (velocity), WIDTH, SQI
    # (define_scanned). `gd_kwargs` set range_minimum/maximum per case.
    function _filter_test_p(; gd_kwargs...)
        pp = DaishoParameters()
        small_cart = Daisho.CartesianGridParameters(
            xmin = -2000.0, xincr = 1000.0, xdim = 5,
            ymin = -2000.0, yincr = 1000.0, ydim = 5,
            zmin = 0.0, zincr = 25.0, zdim = 3)
        gd = Daisho.GriddingParameters(; gd_kwargs...)
        Daisho.DaishoParameters(pp.moments, pp.qc, gd,
            Daisho.GridParameters(cartesian = small_cart,
                                  latlon = pp.grid.latlon,
                                  rhi = pp.grid.rhi,
                                  springsteel = pp.grid.springsteel),
            pp.io, pp.provided)
    end

    pread = _filter_test_p()  # field set + io are invariant across cases
    nvalid_ppi(f) = count(!isnan,
        Daisho.mask_sentinels(Daisho.read_gridded_ppi(f, pread).fields["DBZ"], pread.io))
    nvalid_vol(f) = count(!isnan,
        Daisho.mask_sentinels(Daisho.read_gridded_radar(f, pread).fields["DBZ"], pread.io))

    # synthetic_volume gate slant ranges: 400, 500, …, 1500 m (n_gates = 12).
    v = synthetic_volume(n_sweeps = 2, n_rays = 72, n_gates = 12)
    grid_ppi(f, p) = Daisho.grid_radar_ppi(v, f, v.time_coverage_start, p)
    grid_vol(f, p) = Daisho.grid_radar_volume(v, f, v.time_coverage_start, p)

    @testset "validation (loader)" begin
        @test_throws ArgumentError Daisho._gridding_from_dict(
            Dict("power_threshold" => 0.5, "range_minimum" => 5.0,
                 "range_maximum" => 1.0))
        @test_throws ArgumentError Daisho._gridding_from_dict(
            Dict("power_threshold" => 0.5, "range_minimum" => -1.0))
        gp = Daisho._gridding_from_dict(
            Dict("power_threshold" => 0.5, "range_minimum" => 1000.0,
                 "range_maximum" => 5000.0))
        @test gp.range_minimum == 1000.0 && gp.range_maximum == 5000.0
    end

    @testset "PPI gate filter" begin
        base   = tempname() * "_base.nc"
        capped = tempname() * "_cap.nc"
        noop   = tempname() * "_noop.nc"
        none_hi = tempname() * "_hi.nc"
        none_lo = tempname() * "_lo.nc"
        try
            grid_ppi(base,   _filter_test_p())
            grid_ppi(capped, _filter_test_p(range_maximum = 800.0))
            grid_ppi(noop,   _filter_test_p(range_minimum = 0.0, range_maximum = Inf))
            grid_ppi(none_lo, _filter_test_p(range_maximum = 300.0))   # < min gate (400 m)
            grid_ppi(none_hi, _filter_test_p(range_minimum = 1600.0))  # > max gate (1500 m)
            nb = nvalid_ppi(base)
            @test nb > 0
            @test nvalid_ppi(noop) == nb                 # [0, Inf] default is a no-op
            @test 0 < nvalid_ppi(capped) < nb            # capping at 800 m drops outer cells
            @test nvalid_ppi(none_lo) == 0               # window excludes every gate
            @test nvalid_ppi(none_hi) == 0
        finally
            for f in (base, capped, noop, none_hi, none_lo); isfile(f) && rm(f); end
        end
    end

    @testset "Volume gate filter" begin
        base   = tempname() * "_vbase.nc"
        noop   = tempname() * "_vnoop.nc"
        none_lo = tempname() * "_vlo.nc"
        try
            grid_vol(base,    _filter_test_p())
            grid_vol(noop,    _filter_test_p(range_minimum = 0.0, range_maximum = Inf))
            grid_vol(none_lo, _filter_test_p(range_maximum = 300.0))
            nb = nvalid_vol(base)
            @test nb > 0                                 # z overlaps gate altitudes
            @test nvalid_vol(noop) == nb                 # no-op
            @test nvalid_vol(none_lo) == 0               # all gates excluded
        finally
            for f in (base, noop, none_lo); isfile(f) && rm(f); end
        end
    end
end
