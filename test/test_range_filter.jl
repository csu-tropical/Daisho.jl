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

    @testset "per-product range overrides" begin
        geom = (xmin = -2000.0, xincr = 1000.0, xdim = 5,
                ymin = -2000.0, yincr = 1000.0, ydim = 5,
                zmin = 0.0, zincr = 25.0, zdim = 3)
        cart   = Daisho.CartesianGridParameters(; geom...)
        volcap = Daisho.CartesianGridParameters(; geom..., range_maximum = 800.0)

        # gridding global stays the default (0 / Inf); only volume caps via its table.
        function _p(; gridding = Daisho.GriddingParameters(), volume = nothing)
            Daisho.DaishoParameters(pread.moments, pread.qc, gridding,
                Daisho.GridParameters(cartesian = cart, volume = volume,
                                      latlon = pread.grid.latlon, rhi = pread.grid.rhi,
                                      springsteel = pread.grid.springsteel),
                pread.io, pread.provided)
        end

        @testset "resolution precedence" begin
            p = _p(volume = volcap)
            @test Daisho._range_bounds(p, :volume_3d) == (0.0, 800.0)   # product override
            @test Daisho._range_bounds(p, :ppi_2d) == (0.0, Inf)        # inherits global
            @test Daisho._range_bounds(p, :composite_2d) == (0.0, Inf)
            # A [gridding] global is overridden per-product but kept elsewhere.
            p2 = _p(gridding = Daisho.GriddingParameters(range_maximum = 600.0),
                    volume = volcap)
            @test Daisho._range_bounds(p2, :volume_3d) == (0.0, 800.0)
            @test Daisho._range_bounds(p2, :ppi_2d) == (0.0, 600.0)
            # No override anywhere ⇒ global everywhere (back-compat).
            p3 = _p()
            @test Daisho._range_bounds(p3, :volume_3d) == (0.0, Inf)
            @test Daisho._range_bounds(p3, :ppi_2d) == (0.0, Inf)
        end

        @testset "invalid override raises (named by product)" begin
            bad = Daisho.CartesianGridParameters(; geom..., range_minimum = 5000.0,
                                                 range_maximum = 1000.0)
            @test_throws ArgumentError Daisho._range_bounds(_p(volume = bad), :volume_3d)
        end

        @testset "behavior: volume capped, PPI full-range, one config" begin
            v = synthetic_volume(n_sweeps = 2, n_rays = 72, n_gates = 12)
            # Cap that drops some (not all) volume cells for this small grid.
            pcap = _p(volume = Daisho.CartesianGridParameters(; geom...,
                                                              range_maximum = 1200.0))
            pbase = _p()
            vcap   = tempname() * "_ppr_vcap.nc"
            vbase  = tempname() * "_ppr_vbase.nc"
            ppicap = tempname() * "_ppr_ppicap.nc"
            ppiref = tempname() * "_ppr_ppiref.nc"
            try
                Daisho.grid_radar_volume(v, vcap, v.time_coverage_start, pcap)
                Daisho.grid_radar_volume(v, vbase, v.time_coverage_start, pbase)
                Daisho.grid_radar_ppi(v, ppicap, v.time_coverage_start, pcap)
                Daisho.grid_radar_ppi(v, ppiref, v.time_coverage_start, pbase)
                @test 0 < nvalid_vol(vcap) < nvalid_vol(vbase)   # volume cap drops outer gates
                @test nvalid_ppi(ppicap) == nvalid_ppi(ppiref)   # PPI unaffected by volume cap
            finally
                for f in (vcap, vbase, ppicap, ppiref); isfile(f) && rm(f); end
            end
        end
    end
end
