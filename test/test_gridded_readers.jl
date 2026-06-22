@testset "Gridded Fields-API readers" begin
    # Small grid params so gridding is fast; default [fields] are
    # DBZ (define_detection), VEL (velocity), WIDTH, SQI (define_scanned).
    p = let pp = DaishoParameters()
        small_cart = Daisho.CartesianGridParameters(
            xmin = -2000.0, xincr = 1000.0, xdim = 5,
            ymin = -2000.0, yincr = 1000.0, ydim = 5,
            zmin = 500.0, zincr = 500.0, zdim = 3)
        small_rhi = Daisho.RhiGridParameters(
            rmin = 0.0, rincr = 500.0, rdim = 7,
            zmin = 0.0, zincr = 500.0, zdim = 4)
        Daisho.DaishoParameters(pp.moments, pp.qc, pp.gridding,
            Daisho.GridParameters(cartesian = small_cart,
                                  latlon = pp.grid.latlon,
                                  rhi = small_rhi,
                                  springsteel = pp.grid.springsteel),
            pp.io, pp.provided)
    end
    v = synthetic_volume(n_sweeps = 2, n_rays = 24, n_gates = 6)
    names = Set(fs.name for fs in p.moments.fields)

    @testset "read_gridded_rhi(file, p::DaishoParameters)" begin
        outfile = tempname() * "_rhi.nc"
        try
            Daisho.grid_radar_rhi(v, outfile, v.time_coverage_start, p)
            g = Daisho.read_gridded_rhi(outfile, p)
            @test Set(keys(g.fields)) == names              # keyed by name, all p fields
            @test g.io === p.io                             # io carried through
            @test length(g.R) == 7 && length(g.Z) == 4
            @test size(g.fields["DBZ"]) == (length(g.R), length(g.Z))
            @test eltype(g.fields["DBZ"]) == Float32
        finally
            isfile(outfile) && rm(outfile)
        end
    end

    @testset "read_gridded_ppi(file, p::DaishoParameters)" begin
        outfile = tempname() * "_ppi.nc"
        try
            Daisho.grid_radar_ppi(v, outfile, v.time_coverage_start, p)
            g = Daisho.read_gridded_ppi(outfile, p)
            @test Set(keys(g.fields)) == names
            @test g.io === p.io
            @test length(g.X) == 5 && length(g.Y) == 5
            @test size(g.fields["DBZ"]) == (length(g.X), length(g.Y))
        finally
            isfile(outfile) && rm(outfile)
        end
    end

    @testset "read_gridded_radar(file, p::DaishoParameters)" begin
        outfile = tempname() * "_vol.nc"
        try
            Daisho.grid_radar_volume(v, outfile, v.time_coverage_start, p)
            g = Daisho.read_gridded_radar(outfile, p)
            @test Set(keys(g.fields)) == names
            @test size(g.fields["DBZ"]) == (length(g.X), length(g.Y), length(g.Z))
        finally
            isfile(outfile) && rm(outfile)
        end
    end

    @testset "legacy moment_dict reader still functions" begin
        outfile = tempname() * "_rhi2.nc"
        try
            Daisho.grid_radar_rhi(v, outfile, v.time_coverage_start, p)
            md = Daisho.field_index_dict(p)
            R, Z, lat, lon, t0, t1, radardata = Daisho.read_gridded_rhi(outfile, md)
            @test size(radardata, 1) == length(md)
            @test size(radardata, 2) == length(R) * length(Z)
        finally
            isfile(outfile) && rm(outfile)
        end
    end

    @testset "time is written as an unlimited (record) dimension" begin
        # All gridded writers must emit `time` as a record dimension so files can
        # be concatenated (e.g. with `ncrcat`) without `ncks --mk_rec_dmn time`.
        for (suffix, writer) in (
                ("_vol.nc", Daisho.grid_radar_volume),
                ("_ppi.nc", Daisho.grid_radar_ppi),
                ("_rhi.nc", Daisho.grid_radar_rhi))
            outfile = tempname() * suffix
            try
                writer(v, outfile, v.time_coverage_start, p)
                NCDataset(outfile, "r") do ds
                    @test "time" in NCDatasets.unlimited(ds.dim)
                    @test ds.dim["time"] == 1
                    # NCDatasets CF-decodes the time variable back to DateTime.
                    @test ds["time"][:] == [v.time_coverage_start]
                end
            finally
                isfile(outfile) && rm(outfile)
            end
        end
    end

    @testset "record dimension concatenates a second time slice" begin
        outfile = tempname() * "_concat.nc"
        try
            Daisho.grid_radar_volume(v, outfile, v.time_coverage_start, p)
            t1 = v.time_coverage_start
            t2 = t1 + Second(600)
            # Append a second analysis time by growing the unlimited dim.
            NCDataset(outfile, "a") do ds
                ds["time"][2] = t2
                ds["DBZ"][:, :, :, 2] = ds["DBZ"][:, :, :, 1]
            end
            NCDataset(outfile, "r") do ds
                @test ds.dim["time"] == 2
                @test ds["time"][:] == [t1, t2]
                @test size(ds["DBZ"]) == (length(ds["X"]), length(ds["Y"]), length(ds["Z"]), 2)
                # isequal: CF-decoded fill cells are `missing`, which `==` won't compare.
                @test isequal(ds["DBZ"][:, :, :, 2], ds["DBZ"][:, :, :, 1])
            end
        finally
            isfile(outfile) && rm(outfile)
        end
    end

    @testset "mask_sentinels maps both sentinels to NaN, keeps the rest" begin
        io = Daisho.IOParameters()
        a = Float32[io.fill_value, io.undetect, 5.0f0, -3.0f0]
        m = Daisho.mask_sentinels(a, io)
        @test isnan(m[1]) && isnan(m[2])
        @test m[3] == 5.0f0 && m[4] == -3.0f0
        @test eltype(m) == Float32
    end
end
