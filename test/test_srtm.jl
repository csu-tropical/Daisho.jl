# --- Synthetic SRTM DEM helpers (no network, no real tiles required) ---
#
# A bare headerless `.hgt` cannot be opened by `Raster` without geotransform
# metadata, and GDAL's SRTMHGT writer only accepts the standard tile sizes
# (1201x1201, 3601x3601, ...). So we synthesize DEMs two ways:
#   * In-memory `Raster` objects with proper X(lon)/Y(lat) dims covering a full
#     1-degree tile (north-to-south rows) for the Raster/Dict code paths. These
#     keep the raw `Int16` -32768 void sentinel, exercising the `== -32768` branch.
#   * A real 1201x1201 SRTM3 `.hgt` written to disk for the file-reading paths.
#     The SRTMHGT driver tags -32768 as nodata, so on read-back voids surface as
#     `missing`, exercising the `ismissing` branch.

# Build an in-memory SRTM raster covering the 1-degree tile whose SW corner is
# (lat_south, lon_west). Rows run north-to-south to match SRTM convention.
function make_srtm_raster(lat_south::Real, lon_west::Real; n::Int=6,
                          value::Integer=250, void_cell=nothing)
    lons = range(lon_west, lon_west + 1.0, length=n)
    lats = range(lat_south + 1.0, lat_south, length=n)   # north -> south
    data = fill(Int16(value), n, n)
    if void_cell !== nothing
        data[void_cell...] = Int16(-32768)
    end
    return Raster(data, (X(lons), Y(lats)))
end

# Write a real 1201x1201 SRTM3 `.hgt` tile to `dir/name` covering the tile whose
# SW corner is (lat_south, lon_west). A void block is placed in the NW corner.
function write_srtm_tile(dir::String, name::String, lat_south::Real, lon_west::Real;
                         value::Integer=250, high::Integer=900)
    n = 1201
    lons = range(lon_west, lon_west + 1.0, length=n)
    lats = range(lat_south + 1.0, lat_south, length=n)   # north -> south
    data = fill(Int16(value), n, n)
    for i in 1:n, j in 1:n
        if i <= 100 && j <= 100
            data[i, j] = Int16(-32768)      # void block near NW corner
        elseif i >= 1100 && j >= 1100
            data[i, j] = Int16(high)        # high block near SE corner
        end
    end
    path = joinpath(dir, name)
    write(path, Raster(data, (X(lons), Y(lats))))
    return path
end

@testset "SRTM" begin

    @testset "SRTM tile naming convention" begin
        # Test the tile naming logic by checking the coordinate-to-name mapping
        # This tests the logic in read_srtm_elevation_multi and read_srtm_elevation_dict

        # Northern hemisphere, eastern longitude
        lat, lon = 16.5, 25.3
        lat_char = lat >= 0 ? 'N' : 'S'
        lon_char = lon >= 0 ? 'E' : 'W'
        lat_deg = Int(floor(abs(lat)))
        lon_deg = Int(floor(abs(lon)))
        tile_name = string(lat_char, lpad(lat_deg, 2, '0'), lon_char, lpad(lon_deg, 3, '0'))
        @test tile_name == "N16E025"

        # Southern hemisphere, western longitude
        lat, lon = -33.4, -70.6
        lat_char = lat >= 0 ? 'N' : 'S'
        lon_char = lon >= 0 ? 'E' : 'W'
        lat_deg = Int(floor(abs(lat))) + 1  # Adjust for negative
        lon_deg = Int(floor(abs(lon))) + 1  # Adjust for negative
        tile_name = string(lat_char, lpad(lat_deg, 2, '0'), lon_char, lpad(lon_deg, 3, '0'))
        @test tile_name == "S34W071"
    end

    @testset "terrain_height - missing tile directory" begin
        # With a non-existent directory, should handle gracefully
        result = Daisho.terrain_height(joinpath(@__DIR__, "fixtures", "nonexistent_srtm"), 16.886, -24.988)
        @test result == -1.0
    end

    @testset "terrain_height - empty directory" begin
        # Create an empty directory for testing
        empty_dir = joinpath(@__DIR__, "fixtures", "empty_srtm")
        mkpath(empty_dir)

        result = Daisho.terrain_height(empty_dir, 16.886, -24.988)
        @test result == -1.0

        rm(empty_dir, recursive=true, force=true)
    end

    @testset "terrain_height returns Float64" begin
        # Both overloads should return -1.0 (Float64) when no data available
        result = Daisho.terrain_height(joinpath(@__DIR__, "fixtures"), 16.886, -24.988)
        @test result isa Float64
        @test result == -1.0
    end

    @testset "read_srtm_elevation_multi - missing tile returns nothing" begin
        empty_dir = joinpath(@__DIR__, "fixtures", "empty_srtm2")
        mkpath(empty_dir)

        result = Daisho.read_srtm_elevation_multi(empty_dir, 16.886, -24.988)
        @test result === nothing

        rm(empty_dir, recursive=true, force=true)
    end

    @testset "read_srtm_elevation_dict - missing tile returns nothing" begin
        tiles = Dict{String, Rasters.Raster}()
        result = Daisho.read_srtm_elevation_dict(tiles, 16.886, -24.988)
        @test result === nothing
    end

    @testset "terrain_height dict overload - missing tile" begin
        tiles = Dict{String, Rasters.Raster}()
        result = Daisho.terrain_height(tiles, 16.886, -24.988)
        @test result == -1.0
    end

    @testset "read_srtm_elevation_multi - load all tiles" begin
        # Create a temporary directory (no tiles)
        tile_dir = joinpath(@__DIR__, "fixtures", "srtm_load_test")
        mkpath(tile_dir)

        tiles = Daisho.read_srtm_elevation_multi(tile_dir)
        @test isempty(tiles)

        rm(tile_dir, recursive=true, force=true)
    end

    @testset "read_srtm_elevation_rasters - in-memory Raster" begin
        # Tile N16W025 (lat 16..17, lon -25..-24), void in the NW corner cell.
        ras = make_srtm_raster(16, -25; n=6, value=137, void_cell=(1, 1))

        # Interior lookup returns the constant elevation as Float64.
        elev = Daisho.read_srtm_elevation_rasters(ras, 16.5, -24.5)
        @test elev isa Float64
        @test elev == 137.0

        # The raw -32768 sentinel cell (NW corner: lon ~ -25, lat ~ 17) -> nothing.
        @test Daisho.read_srtm_elevation_rasters(ras, 17.0, -25.0) === nothing
    end

    @testset "read_srtm_elevation_dict - hemisphere tile-name routing" begin
        # One tile per hemisphere combination, each with a distinct elevation so
        # we can confirm the coordinate-to-tile-name math routes correctly:
        #   N/E, N/W, S/E, S/W (the lat_deg+1 / lon_deg+1 negative branches).
        tiles = Dict{String, Raster}(
            "N16E025" => make_srtm_raster(16,  25; value=111),  # N, E
            "N16W025" => make_srtm_raster(16, -25; value=222),  # N, W
            "S34E018" => make_srtm_raster(-34, 18; value=333),  # S, E
            "S34W071" => make_srtm_raster(-34, -71; value=444), # S, W
        )

        @test Daisho.read_srtm_elevation_dict(tiles, 16.5,  25.3) == 111.0
        @test Daisho.read_srtm_elevation_dict(tiles, 16.886, -24.988) == 222.0
        @test Daisho.read_srtm_elevation_dict(tiles, -33.4,  18.6) == 333.0
        @test Daisho.read_srtm_elevation_dict(tiles, -33.4, -70.6) == 444.0

        # A coordinate whose tile is not loaded -> ocean -> nothing.
        @test Daisho.read_srtm_elevation_dict(tiles, 0.5, 0.5) === nothing

        # terrain_height(dict, ...) wraps the dict lookup: hit and ocean miss.
        @test Daisho.terrain_height(tiles, -33.4, -70.6) == 444.0
        @test Daisho.terrain_height(tiles, 0.5, 0.5) == -1.0
    end

    @testset "read_srtm_elevation_dict - void cell returns nothing" begin
        ras = make_srtm_raster(16, -25; value=300, void_cell=(1, 1))
        tiles = Dict{String, Raster}("N16W025" => ras)
        # NW corner cell is the raw -32768 void.
        @test Daisho.read_srtm_elevation_dict(tiles, 17.0, -25.0) === nothing
        @test Daisho.terrain_height(tiles, 17.0, -25.0) == -1.0
    end

    @testset "SRTM file readers - on-disk .hgt tile" begin
        # Write a real 1201x1201 SRTM3 tile for N16W025 (lat 16..17, lon -25..-24).
        tile_dir = mktempdir()
        try
            write_srtm_tile(tile_dir, "N16W025.hgt", 16, -25; value=250, high=900)

            # read_srtm_elevation_rasters(filename, lat, lon)
            elev = Daisho.read_srtm_elevation_rasters(joinpath(tile_dir, "N16W025.hgt"),
                                                      16.5, -24.5)
            @test elev isa Float64
            @test elev == 250.0

            # On-disk void block surfaces as `missing` (nodata) -> nothing.
            # NW corner block sits near lon -25, lat 17.
            @test Daisho.read_srtm_elevation_rasters(
                joinpath(tile_dir, "N16W025.hgt"), 16.97, -24.97) === nothing

            # read_srtm_elevation_multi(dir, lat, lon): tile name derived from coords.
            @test Daisho.read_srtm_elevation_multi(tile_dir, 16.5, -24.5) == 250.0

            # High block near the SE corner (lon -24, lat 16).
            @test Daisho.read_srtm_elevation_multi(tile_dir, 16.03, -24.03) == 900.0

            # Coordinate whose tile file is absent -> ocean -> nothing / -1.0.
            @test Daisho.read_srtm_elevation_multi(tile_dir, 16.5, -23.5) === nothing
            @test Daisho.terrain_height(tile_dir, 16.5, -23.5) == -1.0

            # terrain_height(dir, lat, lon) over a present tile.
            th = Daisho.terrain_height(tile_dir, 16.5, -24.5)
            @test th isa Float64
            @test th == 250.0

            # read_srtm_elevation_multi(dir) loads every .hgt into a Dict keyed by base name.
            loaded = Daisho.read_srtm_elevation_multi(tile_dir)
            @test loaded isa Dict{String, Raster}
            @test haskey(loaded, "N16W025")
            @test length(loaded) == 1
            # The loaded raster routes correctly through the dict path.
            @test Daisho.terrain_height(loaded, 16.5, -24.5) == 250.0
        finally
            rm(tile_dir, recursive=true, force=true)
        end
    end

end
