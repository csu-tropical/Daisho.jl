# Tests for Springsteel.jl integration

using Springsteel

@testset "Springsteel Integration" begin

    # ── daisho_cells_from_gridpoints ──────────────────────────────────────
    @testset "daisho_cells_from_gridpoints" begin
        @test Daisho.daisho_cells_from_gridpoints(9, 3) == 3
        @test Daisho.daisho_cells_from_gridpoints(10, 3) == 4
        @test Daisho.daisho_cells_from_gridpoints(1, 3) == 1
        @test Daisho.daisho_cells_from_gridpoints(0, 3) == 1
    end

    # ── Index mapping ─────────────────────────────────────────────────────
    @testset "Index mapping" begin
        @testset "3D index mapping" begin
            # First point
            @test Daisho.daisho_to_springsteel_index(1, 1, 1, 4, 3) == 1
            # z varies fastest
            @test Daisho.daisho_to_springsteel_index(1, 1, 2, 4, 3) == 2
            @test Daisho.daisho_to_springsteel_index(1, 1, 3, 4, 3) == 3
            # y next
            @test Daisho.daisho_to_springsteel_index(1, 2, 1, 4, 3) == 4
            # x last
            @test Daisho.daisho_to_springsteel_index(2, 1, 1, 4, 3) == 13
            # General point
            @test Daisho.daisho_to_springsteel_index(2, 3, 2, 4, 3) == (2-1)*4*3 + (3-1)*3 + 2
        end

        @testset "2D index mapping" begin
            @test Daisho.daisho_to_springsteel_index_2d(1, 1, 5) == 1
            @test Daisho.daisho_to_springsteel_index_2d(1, 2, 5) == 2
            @test Daisho.daisho_to_springsteel_index_2d(2, 1, 5) == 6
        end
    end

    # ── create_radar_grid ─────────────────────────────────────────────────
    @testset "create_radar_grid" begin
        moment_dict = Dict("DBZ" => 1, "VEL" => 2)

        @testset "R (1D) grid" begin
            sgrid = Daisho.create_radar_grid("R", moment_dict;
                xmin=0.0, xmax=10000.0, xdim=4)
            @test sgrid isa SpringsteelGrid{CartesianGeometry, SplineBasisArray, NoBasisArray, NoBasisArray}
            @test sgrid.params.iDim == 4 * 3  # cells * mubar
            @test sgrid.params.iMin == 0.0
            @test sgrid.params.iMax == 10000.0
            @test length(sgrid.params.vars) == 2
            @test sgrid.params.vars["DBZ"] == 1
            @test sgrid.params.vars["VEL"] == 2
        end

        @testset "RR (2D) grid" begin
            sgrid = Daisho.create_radar_grid("RR", moment_dict;
                xmin=-5000.0, xmax=5000.0, xdim=3,
                ymin=-5000.0, ymax=5000.0, ydim=3)
            @test sgrid isa SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, NoBasisArray}
            @test sgrid.params.iDim == 3 * 3
            @test sgrid.params.jDim == 3 * 3
            @test sgrid.params.iMin == -5000.0
            @test sgrid.params.jMin == -5000.0
        end

        @testset "RRR (3D) grid" begin
            sgrid = Daisho.create_radar_grid("RRR", moment_dict;
                xmin=-50000.0, xmax=50000.0, xdim=5,
                ymin=-50000.0, ymax=50000.0, ydim=5,
                zmin=0.0, zmax=15000.0, zdim=3)
            @test sgrid isa SpringsteelGrid{CartesianGeometry, SplineBasisArray, SplineBasisArray, SplineBasisArray}
            @test sgrid.params.iDim == 5 * 3
            @test sgrid.params.jDim == 5 * 3
            @test sgrid.params.kDim == 3 * 3
            @test sgrid.params.iMin == -50000.0
            @test sgrid.params.iMax == 50000.0
            @test sgrid.params.kMin == 0.0
            @test sgrid.params.kMax == 15000.0
        end

        @testset "Variable count matches moment_dict" begin
            md3 = Dict("DBZ" => 1, "VEL" => 2, "WIDTH" => 3)
            sgrid = Daisho.create_radar_grid("R", md3; xmin=0.0, xmax=100.0, xdim=3)
            @test length(sgrid.params.vars) == 3
            @test size(sgrid.physical, 2) == 3
        end

        @testset "Custom boundary conditions" begin
            sgrid = Daisho.create_radar_grid("R", moment_dict;
                xmin=0.0, xmax=100.0, xdim=3,
                BCL=CubicBSpline.R1T1, BCR=CubicBSpline.R1T1)
            # Grid should be created successfully with custom BCs
            @test sgrid isa SpringsteelGrid
        end

        @testset "Unsupported geometry errors" begin
            @test_throws ErrorException Daisho.create_radar_grid("XYZ", moment_dict;
                xmin=0.0, xmax=100.0, xdim=3)
        end
    end

    # ── get_springsteel_gridpoints_zyx ────────────────────────────────────
    @testset "get_springsteel_gridpoints_zyx" begin
        moment_dict = Dict("DBZ" => 1)

        @testset "3D gridpoints" begin
            sgrid = Daisho.create_radar_grid("RRR", moment_dict;
                xmin=-1000.0, xmax=1000.0, xdim=2,
                ymin=-1000.0, ymax=1000.0, ydim=2,
                zmin=0.0, zmax=1000.0, zdim=2)
            gp = Daisho.get_springsteel_gridpoints_zyx(sgrid)

            iDim = sgrid.params.iDim
            jDim = sgrid.params.jDim
            kDim = sgrid.params.kDim

            @test size(gp) == (kDim, jDim, iDim, 3)
            # z values in slot 1
            @test gp[1, 1, 1, 1] >= 0.0      # z >= zmin
            @test gp[end, 1, 1, 1] <= 1000.0  # z <= zmax
            # x values in slot 3
            @test gp[1, 1, 1, 3] >= -1000.0
            @test gp[1, 1, end, 3] <= 1000.0
            # y values in slot 2
            @test gp[1, 1, 1, 2] >= -1000.0
            @test gp[1, end, 1, 2] <= 1000.0
            # z increases along first dimension
            @test gp[2, 1, 1, 1] > gp[1, 1, 1, 1]
        end

        @testset "2D gridpoints" begin
            sgrid = Daisho.create_radar_grid("RR", moment_dict;
                xmin=-1000.0, xmax=1000.0, xdim=2,
                ymin=-1000.0, ymax=1000.0, ydim=2)
            gp = Daisho.get_springsteel_gridpoints_zyx(sgrid)

            iDim = sgrid.params.iDim
            jDim = sgrid.params.jDim

            @test size(gp) == (jDim, iDim, 2)
            # y in slot 1, x in slot 2
            @test gp[1, 1, 1] >= -1000.0  # y
            @test gp[1, 1, 2] >= -1000.0  # x
        end

        @testset "1D gridpoints" begin
            sgrid = Daisho.create_radar_grid("R", moment_dict;
                xmin=0.0, xmax=1000.0, xdim=3)
            gp = Daisho.get_springsteel_gridpoints_zyx(sgrid)

            @test gp isa Vector{Float64}
            @test length(gp) == sgrid.params.iDim
            @test gp[1] >= 0.0
            @test gp[end] <= 1000.0
        end
    end

    # ── populate_physical! ────────────────────────────────────────────────
    @testset "populate_physical!" begin
        moment_dict = Dict("DBZ" => 1, "VEL" => 2)

        @testset "3D populate" begin
            sgrid = Daisho.create_radar_grid("RRR", moment_dict;
                xmin=-1000.0, xmax=1000.0, xdim=2,
                ymin=-1000.0, ymax=1000.0, ydim=2,
                zmin=0.0, zmax=1000.0, zdim=2)
            iDim = sgrid.params.iDim
            jDim = sgrid.params.jDim
            kDim = sgrid.params.kDim
            n_moments = 2

            # Create a Daisho-format radar_grid
            radar_grid = fill(10.0, n_moments, kDim, jDim, iDim)
            # Set some fill values
            radar_grid[1, 1, 1, 1] = -32768.0   # true missing
            radar_grid[1, 2, 1, 1] = -9999.0    # clear air
            radar_grid[2, 1, 1, 1] = 5.0         # valid VEL

            Daisho.populate_physical!(sgrid, radar_grid, moment_dict)

            # Check true missing → NaN
            idx_111 = Daisho.daisho_to_springsteel_index(1, 1, 1, jDim, kDim)
            @test isnan(sgrid.physical[idx_111, 1, 1])

            # Check clear air → -9999.0 (preserved, NOT NaN)
            idx_112 = Daisho.daisho_to_springsteel_index(1, 1, 2, jDim, kDim)
            @test sgrid.physical[idx_112, 1, 1] == -9999.0

            # Check valid value
            @test sgrid.physical[idx_111, 2, 1] == 5.0

            # Check other valid values
            idx_mid = Daisho.daisho_to_springsteel_index(2, 2, 2, jDim, kDim)
            @test sgrid.physical[idx_mid, 1, 1] == 10.0

            # Only slot 1 should be modified
            @test sgrid.physical[idx_mid, 1, 2] == 0.0  # slot 2 untouched
        end

        @testset "2D populate" begin
            sgrid = Daisho.create_radar_grid("RR", moment_dict;
                xmin=-1000.0, xmax=1000.0, xdim=2,
                ymin=-1000.0, ymax=1000.0, ydim=2)
            iDim = sgrid.params.iDim
            jDim = sgrid.params.jDim

            radar_grid = fill(20.0, 2, jDim, iDim)
            radar_grid[1, 1, 1] = -32768.0
            radar_grid[1, 2, 1] = -9999.0

            Daisho.populate_physical!(sgrid, radar_grid, moment_dict)

            idx = Daisho.daisho_to_springsteel_index_2d(1, 1, jDim)
            @test isnan(sgrid.physical[idx, 1, 1])
            idx2 = Daisho.daisho_to_springsteel_index_2d(1, 2, jDim)
            @test sgrid.physical[idx2, 1, 1] == -9999.0
        end

        @testset "1D populate" begin
            sgrid = Daisho.create_radar_grid("R", moment_dict;
                xmin=0.0, xmax=1000.0, xdim=3)
            iDim = sgrid.params.iDim

            radar_grid = fill(15.0, 2, iDim)
            radar_grid[1, 1] = -32768.0
            radar_grid[1, 3] = -9999.0

            Daisho.populate_physical!(sgrid, radar_grid, moment_dict)

            @test isnan(sgrid.physical[1, 1, 1])
            @test sgrid.physical[3, 1, 1] == -9999.0
            @test sgrid.physical[2, 1, 1] == 15.0
        end
    end

    # ── compute_roi ───────────────────────────────────────────────────────
    @testset "compute_roi" begin
        moment_dict = Dict("DBZ" => 1)

        @testset "3D ROI" begin
            sgrid = Daisho.create_radar_grid("RRR", moment_dict;
                xmin=0.0, xmax=10000.0, xdim=5,
                ymin=0.0, ymax=10000.0, ydim=5,
                zmin=0.0, zmax=5000.0, zdim=3)
            h_roi, v_roi = Daisho.compute_roi(sgrid)
            @test h_roi > 0.0
            @test v_roi > 0.0
            # ROI should be reasonable relative to grid spacing
            # Grid spacing ~ 10000 / (5*3) ≈ 667m, so ROI ≈ 500m
            @test h_roi < 10000.0
            @test v_roi < 5000.0
        end

        @testset "2D ROI" begin
            sgrid = Daisho.create_radar_grid("RR", moment_dict;
                xmin=0.0, xmax=10000.0, xdim=5,
                ymin=0.0, ymax=10000.0, ydim=5)
            roi = Daisho.compute_roi(sgrid)
            @test length(roi) == 1
            @test roi[1] > 0.0
        end

        @testset "1D ROI" begin
            sgrid = Daisho.create_radar_grid("R", moment_dict;
                xmin=0.0, xmax=10000.0, xdim=5)
            roi = Daisho.compute_roi(sgrid)
            @test length(roi) == 1
            @test roi[1] > 0.0
        end
    end

    # ── Spectral round-trip ───────────────────────────────────────────────
    @testset "Spectral round-trip" begin

        @testset "1D linear field exact recovery" begin
            moment_dict = Dict("u" => 1)
            sgrid = Daisho.create_radar_grid("R", moment_dict;
                xmin=0.0, xmax=10.0, xdim=5)
            pts = getGridpoints(sgrid)

            # Set linear field: f(x) = 2x + 1
            for i in 1:sgrid.params.iDim
                sgrid.physical[i, 1, 1] = 2.0 * pts[i] + 1.0
            end

            spectralTransform!(sgrid)
            gridTransform!(sgrid)

            # Values should be exact for cubic splines on a linear field
            for i in 1:sgrid.params.iDim
                @test sgrid.physical[i, 1, 1] ≈ 2.0 * pts[i] + 1.0 atol=1e-10
            end
            # First derivative should be 2.0
            for i in 1:sgrid.params.iDim
                @test sgrid.physical[i, 1, 2] ≈ 2.0 atol=1e-8
            end
        end

        @testset "1D quadratic derivative accuracy" begin
            moment_dict = Dict("u" => 1)
            sgrid = Daisho.create_radar_grid("R", moment_dict;
                xmin=0.0, xmax=10.0, xdim=10)
            pts = getGridpoints(sgrid)

            # Set quadratic field: f(x) = x^2
            for i in 1:sgrid.params.iDim
                sgrid.physical[i, 1, 1] = pts[i]^2
            end

            spectralTransform!(sgrid)
            gridTransform!(sgrid)

            # Second derivative should be 2.0
            for i in 2:sgrid.params.iDim-1  # avoid boundary effects
                @test sgrid.physical[i, 1, 3] ≈ 2.0 atol=0.5
            end
        end

        @testset "2D linear field exact recovery" begin
            moment_dict = Dict("u" => 1)
            sgrid = Daisho.create_radar_grid("RR", moment_dict;
                xmin=0.0, xmax=10.0, xdim=4,
                ymin=0.0, ymax=10.0, ydim=4)
            pts = getGridpoints(sgrid)

            # Set linear field: f(x,y) = 3x + 2y + 1
            for p in 1:size(pts, 1)
                sgrid.physical[p, 1, 1] = 3.0 * pts[p, 1] + 2.0 * pts[p, 2] + 1.0
            end

            spectralTransform!(sgrid)
            gridTransform!(sgrid)

            for p in 1:size(pts, 1)
                expected = 3.0 * pts[p, 1] + 2.0 * pts[p, 2] + 1.0
                @test sgrid.physical[p, 1, 1] ≈ expected atol=1e-8
            end
        end

        @testset "3D linear field exact recovery" begin
            moment_dict = Dict("u" => 1)
            sgrid = Daisho.create_radar_grid("RRR", moment_dict;
                xmin=0.0, xmax=10.0, xdim=3,
                ymin=0.0, ymax=10.0, ydim=3,
                zmin=0.0, zmax=10.0, zdim=3)
            pts = getGridpoints(sgrid)

            # Set linear field: f(x,y,z) = x + 2y + 3z + 1
            for p in 1:size(pts, 1)
                sgrid.physical[p, 1, 1] = pts[p, 1] + 2.0 * pts[p, 2] + 3.0 * pts[p, 3] + 1.0
            end

            spectralTransform!(sgrid)
            gridTransform!(sgrid)

            for p in 1:size(pts, 1)
                expected = pts[p, 1] + 2.0 * pts[p, 2] + 3.0 * pts[p, 3] + 1.0
                @test sgrid.physical[p, 1, 1] ≈ expected atol=1e-6
            end
        end

        @testset "Full pipeline: populate → transform → check" begin
            moment_dict = Dict("u" => 1)
            sgrid = Daisho.create_radar_grid("R", moment_dict;
                xmin=0.0, xmax=10.0, xdim=5)
            iDim = sgrid.params.iDim
            pts = getGridpoints(sgrid)

            # Create a Daisho-format radar_grid with linear ramp
            radar_grid = Array{Float64}(undef, 1, iDim)
            for i in 1:iDim
                radar_grid[1, i] = 3.0 * pts[i] + 5.0
            end

            Daisho.populate_physical!(sgrid, radar_grid, moment_dict)
            spectralTransform!(sgrid)
            gridTransform!(sgrid)

            for i in 1:iDim
                @test sgrid.physical[i, 1, 1] ≈ 3.0 * pts[i] + 5.0 atol=1e-10
            end
        end
    end

    # ── mask_fill_for_transform! / restore_fill_from_mask! ────────────────
    @testset "Fill value masking" begin
        moment_dict = Dict("u" => 1, "v" => 2)
        sgrid = Daisho.create_radar_grid("R", moment_dict;
            xmin=0.0, xmax=10.0, xdim=3)
        iDim = sgrid.params.iDim

        # Set up physical array with fill values
        sgrid.physical[1, 1, 1] = NaN        # true missing
        sgrid.physical[2, 1, 1] = -9999.0    # clear air
        sgrid.physical[3, 1, 1] = 42.0       # valid
        sgrid.physical[1, 2, 1] = 7.0
        sgrid.physical[2, 2, 1] = NaN
        sgrid.physical[3, 2, 1] = -9999.0

        mask = Daisho.mask_fill_for_transform!(sgrid)

        # After masking, NaN and -9999 should be 0.0
        @test sgrid.physical[1, 1, 1] == 0.0
        @test sgrid.physical[2, 1, 1] == 0.0
        @test sgrid.physical[3, 1, 1] == 42.0
        @test mask[1, 1] == Int8(1)  # was NaN
        @test mask[2, 1] == Int8(2)  # was -9999
        @test mask[3, 1] == Int8(0)  # valid

        # Simulate transform (just set some values in other slots)
        sgrid.physical[1, 1, 2] = 99.0  # derivative slot

        Daisho.restore_fill_from_mask!(sgrid, mask)

        # NaN and -9999 should be restored
        @test isnan(sgrid.physical[1, 1, 1])
        @test sgrid.physical[2, 1, 1] == -9999.0
        @test sgrid.physical[3, 1, 1] == 42.0
        # Derivative slots should also be masked
        @test isnan(sgrid.physical[1, 1, 2])
        @test sgrid.physical[2, 1, 2] == -9999.0
    end

    # ── radar_global_attributes ───────────────────────────────────────────
    @testset "radar_global_attributes" begin
        radar = make_synthetic_radar()

        attrs = Daisho.radar_global_attributes(radar, 16.886, -24.988;
            institution="CSU", source="TEST_RADAR")

        @test attrs["Conventions"] == "CF-1.12"
        @test attrs["institution"] == "CSU"
        @test attrs["source"] == "TEST_RADAR"
        @test attrs["reference_latitude"] ≈ 16.886
        @test attrs["reference_longitude"] ≈ -24.988
        @test haskey(attrs, "time_coverage_start")
        @test haskey(attrs, "time_coverage_end")
        @test haskey(attrs, "history")
        @test !haskey(attrs, "platform_heading")  # default heading is missing

        # With heading
        attrs2 = Daisho.radar_global_attributes(radar, 16.886, -24.988;
            heading=45.0)
        @test attrs2["platform_heading"] == 45.0
    end

    # ── write_radar_netcdf ────────────────────────────────────────────────
    @testset "write_radar_netcdf" begin
        moment_dict = Dict("DBZ" => 1, "VEL" => 2)

        @testset "1D NetCDF output" begin
            sgrid = Daisho.create_radar_grid("R", moment_dict;
                xmin=0.0, xmax=10000.0, xdim=4)
            pts = getGridpoints(sgrid)

            for i in 1:sgrid.params.iDim
                sgrid.physical[i, 1, 1] = 20.0 + pts[i] / 1000.0
                sgrid.physical[i, 2, 1] = -5.0 + pts[i] / 5000.0
            end

            spectralTransform!(sgrid)

            radar = make_synthetic_radar()
            outfile = tempname() * ".nc"
            try
                Daisho.write_radar_netcdf(outfile, sgrid, radar, moment_dict;
                    ref_lat=16.886, ref_lon=-24.988, institution="CSU")

                @test isfile(outfile)
                NCDataset(outfile, "r") do ds
                    @test ds.attrib["Conventions"] == "CF-1.12"
                    @test ds.attrib["institution"] == "CSU"
                    @test haskey(ds, "DBZ")
                    @test haskey(ds, "VEL")
                    @test ds["DBZ"].attrib["units"] == "dBZ"
                    @test ds["VEL"].attrib["units"] == "m/s"
                    @test haskey(ds, "grid_mapping")
                    @test ds["grid_mapping"].attrib["grid_mapping_name"] == "transverse_mercator"
                    @test ds["grid_mapping"].attrib["latitude_of_projection_origin"] ≈ 16.886
                    # Coordinate attributes
                    @test ds["x"].attrib["units"] == "m"
                    @test ds["x"].attrib["long_name"] == "easting"
                    # Data round-trip check
                    dbz_data = Array(ds["DBZ"])
                    @test length(dbz_data) > 0
                    @test !all(isnan.(dbz_data))
                end
            finally
                rm(outfile; force=true)
            end
        end

        @testset "2D NetCDF output" begin
            sgrid = Daisho.create_radar_grid("RR", moment_dict;
                xmin=-5000.0, xmax=5000.0, xdim=3,
                ymin=-5000.0, ymax=5000.0, ydim=3)
            pts = getGridpoints(sgrid)

            for p in 1:size(pts, 1)
                sgrid.physical[p, 1, 1] = 25.0
                sgrid.physical[p, 2, 1] = -3.0
            end

            spectralTransform!(sgrid)

            radar = make_synthetic_radar()
            outfile = tempname() * ".nc"
            try
                Daisho.write_radar_netcdf(outfile, sgrid, radar, moment_dict;
                    ref_lat=16.886, ref_lon=-24.988)

                @test isfile(outfile)
                NCDataset(outfile, "r") do ds
                    @test haskey(ds, "DBZ")
                    @test haskey(ds, "VEL")
                end
            finally
                rm(outfile; force=true)
            end
        end

        @testset "3D NetCDF output" begin
            sgrid = Daisho.create_radar_grid("RRR", moment_dict;
                xmin=-5000.0, xmax=5000.0, xdim=2,
                ymin=-5000.0, ymax=5000.0, ydim=2,
                zmin=0.0, zmax=5000.0, zdim=2)
            pts = getGridpoints(sgrid)

            for p in 1:size(pts, 1)
                sgrid.physical[p, 1, 1] = 30.0
                sgrid.physical[p, 2, 1] = -2.0
            end

            spectralTransform!(sgrid)

            radar = make_synthetic_radar()
            outfile = tempname() * ".nc"
            try
                Daisho.write_radar_netcdf(outfile, sgrid, radar, moment_dict;
                    ref_lat=16.886, ref_lon=-24.988)

                @test isfile(outfile)
                NCDataset(outfile, "r") do ds
                    @test haskey(ds, "DBZ")
                    @test haskey(ds, "VEL")
                    @test haskey(ds, "grid_mapping")
                end
            finally
                rm(outfile; force=true)
            end
        end
    end

    # ── Backward compatibility ────────────────────────────────────────────
    @testset "Backward compatibility" begin
        @testset "initialize_regular_grid still works" begin
            grid_3d = Daisho.initialize_regular_grid(-1000.0, 500.0, 5, -1000.0, 500.0, 5, 0.0, 500.0, 3)
            @test size(grid_3d) == (3, 5, 5, 3)
        end
    end


    @testset "Volume spectral overloads dispatch" begin
        @test hasmethod(Daisho.grid_radar_volume_spectral,
            Tuple{Volume,AbstractString,Any,SpringsteelGrid,DaishoParameters})
        @test hasmethod(Daisho.grid_radar_ppi_spectral,
            Tuple{Volume,AbstractString,Any,SpringsteelGrid,DaishoParameters})
        @test hasmethod(Daisho.grid_radar_column_spectral,
            Tuple{Volume,AbstractString,Any,SpringsteelGrid,DaishoParameters})
    end

end
