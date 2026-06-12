@testset "Composite gridding value finalize" begin
    # Small grid params so gridding is fast; default [fields] are
    # DBZ (define_detection, linear_interp), VEL (velocity), WIDTH, SQI.
    p = let pp = DaishoParameters()
        small_cart = Daisho.CartesianGridParameters(
            xmin = -2000.0, xincr = 1000.0, xdim = 5,
            ymin = -2000.0, yincr = 1000.0, ydim = 5,
            zmin = 500.0, zincr = 500.0, zdim = 3)
        Daisho.DaishoParameters(pp.moments, pp.qc, pp.gridding,
            Daisho.GridParameters(cartesian = small_cart,
                                  latlon = pp.grid.latlon,
                                  rhi = pp.grid.rhi,
                                  springsteel = pp.grid.springsteel),
            pp.io, pp.provided)
    end
    v = synthetic_volume(n_sweeps = 2, n_rays = 24, n_gates = 6)

    @testset "composite passes the column-max value through (regression)" begin
        # Composite selects the column-maximum gate per cell and stores that
        # gate's value directly; finalize must pass it through like :nearest,
        # NOT apply each field's interpolation tag. The old finalize divided a
        # :linear field (DBZ) by weight_total and dB-reconverted it, collapsing
        # the composite reflectivity to ~0 dBZ.
        outfile = tempname() * "_comp.nc"
        try
            Daisho.grid_radar_composite(v, outfile, v.time_coverage_start, p)
            g = Daisho.read_gridded_ppi(outfile, p)
            dbz = Daisho.mask_sentinels(g.fields["DBZ"], g.io)
            valid = filter(!isnan, vec(dbz))
            @test !isempty(valid)
            # synthetic_volume DBZ gates span ~21..164 dBZ; the column max must be
            # preserved (well above the ~0 the bug produced) and never exceed the
            # input maximum.
            @test minimum(valid) > 1.0
            @test maximum(valid) <= 164.0 + 1e-3
            @test maximum(valid) > 100.0
        finally
            isfile(outfile) && rm(outfile)
        end
    end
end
