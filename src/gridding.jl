# Radar gridding functions

"""
    initialize_regular_grid(xmin, xincr, xdim, ymin, yincr, ydim, zmin, zincr, zdim) -> Array{Float64, 4}

Initialize a 3D regular Cartesian grid with Z, Y, X coordinates.

Allocates and fills a 4D array where the first three dimensions correspond to Z, Y, X grid indices
and the fourth dimension stores the coordinate values (z, y, x) at each grid point.

# Arguments
- `xmin`: Minimum x-coordinate (meters).
- `xincr`: Grid spacing in the x-direction (meters).
- `xdim`: Number of grid points in the x-direction.
- `ymin`: Minimum y-coordinate (meters).
- `yincr`: Grid spacing in the y-direction (meters).
- `ydim`: Number of grid points in the y-direction.
- `zmin`: Minimum z-coordinate (meters).
- `zincr`: Grid spacing in the z-direction (meters).
- `zdim`: Number of grid points in the z-direction.

# Returns
A `(zdim, ydim, xdim, 3)` `Array{Float64, 4}` where the last dimension stores `[z, y, x]` coordinates.
"""
function initialize_regular_grid(xmin, xincr, xdim, ymin, yincr, ydim, zmin, zincr, zdim)

    # Define and allocate a 3d regular grid
    regular_3d_grid = Array{Float64}(undef,zdim,ydim,xdim,3)
    for i in CartesianIndices(size(regular_3d_grid)[1:3])
        regular_3d_grid[i,1] = zincr * (i[1]-1) + zmin
        regular_3d_grid[i,2] = yincr * (i[2]-1) + ymin
        regular_3d_grid[i,3] = xincr * (i[3]-1) + xmin
    end
    return regular_3d_grid
end

"""
    initialize_regular_grid(xmin, xincr, xdim, ymin, yincr, ydim) -> Array{Float64, 3}

Initialize a 2D regular Cartesian grid with Y, X coordinates.

Allocates and fills a 3D array where the first two dimensions correspond to Y, X grid indices
and the third dimension stores the coordinate values (y, x) at each grid point.

# Arguments
- `xmin`: Minimum x-coordinate (meters).
- `xincr`: Grid spacing in the x-direction (meters).
- `xdim`: Number of grid points in the x-direction.
- `ymin`: Minimum y-coordinate (meters).
- `yincr`: Grid spacing in the y-direction (meters).
- `ydim`: Number of grid points in the y-direction.

# Returns
A `(ydim, xdim, 2)` `Array{Float64, 3}` where the last dimension stores `[y, x]` coordinates.
"""
function initialize_regular_grid(xmin, xincr, xdim, ymin, yincr, ydim)

    # Define and allocate a 2d regular grid
    print("Allocating 2d regular grid with dimensions: y ", ydim, "x", xdim, "\n")
    regular_2d_grid = Array{Float64}(undef,ydim,xdim,2)
    for i in CartesianIndices(size(regular_2d_grid)[1:2])
        regular_2d_grid[i,1] = yincr * (i[1]-1) + ymin
        regular_2d_grid[i,2] = xincr * (i[2]-1) + xmin
    end
    return regular_2d_grid
end

"""
    initialize_regular_grid(zmin, zincr, zdim) -> Array{Float64, 1}

Initialize a 1D regular grid (e.g., vertical altitude levels).

Allocates and fills a 1D array of evenly spaced coordinate values.

# Arguments
- `zmin`: Minimum coordinate value (meters).
- `zincr`: Grid spacing (meters).
- `zdim`: Number of grid points.

# Returns
A `(zdim,)` `Array{Float64, 1}` of coordinate values.
"""
function initialize_regular_grid(zmin, zincr, zdim)

    # Define and allocate a 2d regular grid
    regular_1d_grid = Array{Float64}(undef,zdim)
    for i in CartesianIndices(size(regular_1d_grid))
        regular_1d_grid[i] = zincr * (i[1]-1) + zmin
    end
    return regular_1d_grid
end

"""
    initialize_regular_grid(reference_latitude, reference_longitude, lonmin, londim, latmin, latdim, degincr, zmin, zincr, zdim) -> Array{Float64, 4}

Initialize a 3D regular grid on a latitude-longitude coordinate system with vertical levels.

Constructs a Transverse Mercator projection centered on the reference point, creates a lat/lon grid,
converts it to Cartesian coordinates, and produces a 3D grid array with Z, Y, X values.

# Arguments
- `reference_latitude`: Latitude of the projection origin (degrees).
- `reference_longitude`: Longitude of the projection origin (degrees).
- `lonmin`: Minimum longitude of the grid (degrees).
- `londim`: Number of grid points in the longitude direction.
- `latmin`: Minimum latitude of the grid (degrees).
- `latdim`: Number of grid points in the latitude direction.
- `degincr`: Grid spacing in degrees for both latitude and longitude.
- `zmin`: Minimum altitude (meters).
- `zincr`: Vertical grid spacing (meters).
- `zdim`: Number of vertical grid levels.

# Returns
A `(zdim, latdim, londim, 3)` `Array{Float64, 4}` where the last dimension stores `[z, y, x]`
in Transverse Mercator Cartesian coordinates (meters).
"""
function initialize_regular_grid(reference_latitude, reference_longitude, lonmin, londim, latmin, latdim, degincr, zmin, zincr, zdim)

    # Define and allocate a 3d regular latlon grid
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)

    latlon_grid = Array{Float64}(undef,latdim,londim,2)
    for i in CartesianIndices(size(latlon_grid)[1:2])
        latlon_grid[i,1] = degincr * (i[1]-1) + latmin
        latlon_grid[i,2] = degincr * (i[2]-1) + lonmin
    end
    cartTM = convert.(TM,LatLon.(latlon_grid[:,:,1], latlon_grid[:,:,2]))

    regular_3d_grid = Array{Float64}(undef,zdim,latdim,londim,3)
    for i in CartesianIndices(size(regular_3d_grid)[2:3])
        for j in 1:zdim
            regular_3d_grid[j,i,1] = zincr * (j-1) + zmin
            regular_3d_grid[j,i,2] = ustrip(cartTM[i].y)
            regular_3d_grid[j,i,3] = ustrip(cartTM[i].x)
        end
    end

    return regular_3d_grid
end

"""
    get_radar_zyx(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, radar_volume::radar, projection) -> Vector

Compute radar gate origin positions in Transverse Mercator Cartesian coordinates.

Converts the radar latitude/longitude positions from the volume to the given projection and
constructs a vector of `[z, y, x]` origin coordinates for every gate (range x beam).

# Arguments
- `reference_latitude::AbstractFloat`: Reference latitude (degrees), unused directly but passed for context.
- `reference_longitude::AbstractFloat`: Reference longitude (degrees), unused directly but passed for context.
- `radar_volume::radar`: Radar volume data structure containing latitude, longitude, altitude, and range fields.
- `projection`: A Transverse Mercator projection type used to convert lat/lon to Cartesian coordinates.

# Returns
A matrix-shaped vector of `[z, y, x]` vectors with dimensions `(n_ranges, n_beams)`, giving the beam
origin position for each gate in the radar volume.
"""
function get_radar_zyx(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, radar_volume::radar, projection)

    # Radar locations mapped to transverse mercator with Z,Y,X dimensions
    radar_loc = convert.(projection,LatLon.(radar_volume.latitude, radar_volume.longitude))
    beam_origin = [ [radar_volume.altitude[i], Float64(ustrip(radar_loc[i].y)), Float64(ustrip(radar_loc[i].x))]  for i in eachindex(radar_loc)]
    radar_zyx = [ zyx for j in radar_volume.range, zyx in beam_origin ]
    return radar_zyx

end

"""
    get_beam_info(radar_volume::radar) -> Array{Float64, 2}

Extract beam geometry information for every gate in the radar volume.

Computes azimuth (radians), elevation (radians), range (meters), and beam height (meters) for
each gate, accounting for earth curvature via `beam_height`.

# Arguments
- `radar_volume::radar`: Radar volume data structure containing azimuth, elevation, range, and altitude fields.

# Returns
An `(n_gates, 4)` `Array` where each row contains `[azimuth, elevation, range, height]` for a gate.
Azimuth and elevation are in radians; range and height are in meters.
"""
function get_beam_info(radar_volume::radar)

    # Create an array with all the relevant beam info (azimuth, elevation, range, height)
    beams = [ (deg2rad(radar_volume.azimuth[j]), deg2rad(radar_volume.elevation[j]), radar_volume.range[i],
            beam_height(radar_volume.range[i], radar_volume.elevation[j], radar_volume.altitude[j]))
            for i in eachindex(radar_volume.range), j in eachindex(radar_volume.elevation) ]
    beams = [ beams[i][j] for i in eachindex(beams), j in 1:4]
    return beams

end

"""
    radar_arrays(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, radar_volume::radar, projection) -> Tuple

Compute the grid origin, radar gate positions, and beam geometry arrays for a radar volume.

A convenience function that calls `get_radar_zyx` and `get_beam_info` together, and also
computes the grid origin in the given projection.

# Arguments
- `reference_latitude::AbstractFloat`: Reference latitude for the grid origin (degrees).
- `reference_longitude::AbstractFloat`: Reference longitude for the grid origin (degrees).
- `radar_volume::radar`: Radar volume data structure.
- `projection`: Transverse Mercator projection type.

# Returns
A tuple `(grid_origin, radar_zyx, beams)` where:
- `grid_origin`: The reference point converted to the projection coordinate system.
- `radar_zyx`: Gate origin positions from `get_radar_zyx`.
- `beams`: Beam geometry array from `get_beam_info`.
"""
function radar_arrays(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, radar_volume::radar, projection)

    # Grid origin
    grid_origin = convert(projection,LatLon(reference_latitude, reference_longitude))

    # Radar locations mapped to transverse mercator with Z,Y,X dimensions
    radar_zyx = get_radar_zyx(reference_latitude, reference_longitude, radar_volume, projection)

    # Create an array with all the relevant beam info (azimuth, elevation, range, height)
    beams = get_beam_info(radar_volume)

    return grid_origin, radar_zyx, beams

end

"""
    radar_balltree_yx(radar_volume::radar, radar_zyx::AbstractArray, beams::AbstractArray) -> BallTree

Build a BallTree spatial index for horizontal (Y, X) gate locations.

Computes the surface-projected Y and X positions of every radar gate using the effective earth
radius model, then constructs a `BallTree` for efficient nearest-neighbor and range queries
in the horizontal plane.

# Arguments
- `radar_volume::radar`: Radar volume data structure (used for array dimensions).
- `radar_zyx::AbstractArray`: Gate origin positions as `[z, y, x]` vectors from `get_radar_zyx`.
- `beams::AbstractArray`: Beam geometry array from `get_beam_info` with columns `[azimuth, elevation, range, height]`.

# Returns
A `BallTree` indexed on 2D `(y, x)` gate positions (meters).
"""
function radar_balltree_yx(radar_volume::radar, radar_zyx::AbstractArray, beams::AbstractArray)

    # Create a balltree that has horizontal locations of every gate in Y, X dimension
    gate_yx = zeros(Float64, 2, length(radar_volume.azimuth)*length(radar_volume.range))
    for i in 1:size(beams,1)
        surface_range = Reff * asin(beams[i,3] * cos(beams[i,2]) / (Reff + beams[i,4]))
        gate_yx[1,i] = radar_zyx[i][2] + surface_range * cos(beams[i,1])
        gate_yx[2,i] = radar_zyx[i][3] + surface_range * sin(beams[i,1])
    end
    balltree = BallTree(gate_yx)
    return balltree

end

"""
    radar_balltree_r(radar_volume::radar, radar_zyx::AbstractArray, beams::AbstractArray) -> BallTree

Build a BallTree spatial index for radial (range) gate locations.

Computes the surface-projected radial distance from the origin for every radar gate using the
effective earth radius model, then constructs a `BallTree` for efficient range queries in 1D.
This is used for RHI gridding where the horizontal coordinate is range rather than Y/X.

# Arguments
- `radar_volume::radar`: Radar volume data structure (used for array dimensions).
- `radar_zyx::AbstractArray`: Gate origin positions as `[z, y, x]` vectors from `get_radar_zyx`.
- `beams::AbstractArray`: Beam geometry array from `get_beam_info` with columns `[azimuth, elevation, range, height]`.

# Returns
A `BallTree` indexed on 1D radial distance (meters) from the origin.
"""
function radar_balltree_r(radar_volume::radar, radar_zyx::AbstractArray, beams::AbstractArray)

    # Create a balltree that has horizontal locations of every gate in R dimension
    gate_r = zeros(Float64, 1, length(radar_volume.azimuth)*length(radar_volume.range))
    for i in 1:size(beams,1)
        surface_range = Reff * asin(beams[i,3] * cos(beams[i,2]) / (Reff + beams[i,4]))
        y = radar_zyx[i][2] + surface_range * cos(beams[i,1])
        x = radar_zyx[i][3] + surface_range * sin(beams[i,1])
        gate_r[i] = sqrt(x^2 + y^2)
    end
    balltree = BallTree(gate_r)
    return balltree

end

"""
    appx_inverse_projection(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, yx_point::AbstractArray) -> Tuple{Float64, Float64}

Compute an approximate inverse map projection from Cartesian (y, x) offsets back to latitude/longitude.

Uses an empirical formula (originating from HRD/FCC wireless communication specifications) to convert
meter offsets from a reference point back to geographic coordinates. This is a fast approximation
rather than a rigorous geodetic inverse projection.

# Arguments
- `reference_latitude::AbstractFloat`: Latitude of the reference origin (degrees).
- `reference_longitude::AbstractFloat`: Longitude of the reference origin (degrees).
- `yx_point::AbstractArray`: A 2-element array `[y_offset, x_offset]` in meters from the reference point.

# Returns
A tuple `(lat, lon)` of the approximate latitude and longitude in degrees.
"""
function appx_inverse_projection(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, yx_point::AbstractArray)

    # Approximate lat/lon from SAMURAI formula
    # This formula originated from HRD code, but is originally from FCC wireless communication specs evidently
    latrad = reference_latitude * pi/180.0
    fac_lat = 111.13209 - 0.56605 * cos(2.0 * latrad)
        + 0.00012 * cos(4.0 * latrad) - 0.000002 * cos(6.0 * latrad)
    fac_lon = 111.41513 * cos(latrad)
        - 0.09455 * cos(3.0 * latrad) + 0.00012 * cos(5.0 * latrad)
    lon = reference_longitude + yx_point[2]/(fac_lon*1000.0)
    lat = reference_latitude + yx_point[1]/(fac_lat*1000.0)
    return lat, lon

end

"""
    grid_radar_volume(radar_volume, moment_dict, grid_type_dict, output_file, index_time, xmin, xincr, xdim, ymin, yincr, ydim, zmin, zincr, zdim, power_threshold, missing_key="SQI", valid_key="DBZ", heading=-9999.0, metadata=MetadataParameters())

Grid a radar volume scan onto a 3D Cartesian grid and write the result to a NetCDF file.

This is the high-level driver for Cartesian volume gridding. It initializes the grid, computes
the radius of influence from the grid spacing, calls `grid_volume` for the interpolation, and
writes the output via `write_gridded_radar_volume`.

# Arguments
- `radar_volume`: Radar volume data structure.
- `moment_dict`: Dictionary mapping moment names (e.g., `"DBZ"`) to integer indices.
- `grid_type_dict`: Dictionary mapping moment indices to interpolation type symbols (`:linear`, `:nearest`, or default weighted average).
- `output_file`: Path to the output NetCDF file.
- `index_time`: Reference time for the output dataset.
- `xmin`: Minimum x-coordinate of the grid (meters).
- `xincr`: Grid spacing in x (meters).
- `xdim`: Number of grid points in x.
- `ymin`: Minimum y-coordinate of the grid (meters).
- `yincr`: Grid spacing in y (meters).
- `ydim`: Number of grid points in y.
- `zmin`: Minimum z-coordinate of the grid (meters).
- `zincr`: Grid spacing in z (meters).
- `zdim`: Number of grid points in z.
- `power_threshold`: Minimum beam power weight below which a gate is excluded.
- `missing_key`: Moment name used to determine if a gate has valid signal (default `"SQI"`).
- `valid_key`: Moment name used for valid-data gating (default `"DBZ"`).
- `heading`: Mean heading of the platform in degrees (default `-9999.0` for missing).
- `metadata`: CF-1.12 global attributes forwarded to the writer (default `MetadataParameters()`).
"""
function grid_radar_volume(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
        xmin, xincr, xdim, ymin, yincr, ydim, zmin, zincr, zdim, power_threshold,
        missing_key="SQI", valid_key="DBZ", heading=-9999.0,
        metadata::MetadataParameters=MetadataParameters())

    # Set the reference to the first location in the volume, but could be a parameter
    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    # Initialize the gridpoints
    # This array is slightly different than Springsteel spectral arrays, need to reconcile later
    gridpoints = initialize_regular_grid(xmin, xincr, xdim, ymin, yincr, ydim, zmin, zincr, zdim)

    h_roi = xincr * 0.75
    v_roi = zincr * 0.75

    radar_grid, latlon_grid = grid_volume(reference_latitude, reference_longitude, gridpoints,
        radar_volume, moment_dict, grid_type_dict, h_roi, v_roi, power_threshold,
        missing_key, valid_key; beam_width = radar_volume.beam_width)

    write_gridded_radar_volume(output_file, index_time, radar_volume.time[1],
        radar_volume.time[end], gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude, heading, metadata)

end

"""
    grid_radar_latlon_volume(radar_volume, moment_dict, grid_type_dict, output_file, index_time, lonmin, londim, latmin, latdim, degincr, zmin, zincr, zdim, power_threshold, missing_key="SQI", valid_key="DBZ", heading=-9999.0, metadata=MetadataParameters())

Grid a radar volume scan onto a 3D latitude-longitude grid and write the result to a NetCDF file.

Similar to `grid_radar_volume`, but the horizontal grid is defined in degrees of latitude and longitude
rather than meters. The reference point is derived from the radar location snapped to the nearest
grid increment. Horizontal radius of influence is computed from the degree increment converted to meters.

# Arguments
- `radar_volume`: Radar volume data structure.
- `moment_dict`: Dictionary mapping moment names to integer indices.
- `grid_type_dict`: Dictionary mapping moment indices to interpolation type symbols.
- `output_file`: Path to the output NetCDF file.
- `index_time`: Reference time for the output dataset.
- `lonmin`: Minimum longitude offset from reference (degrees).
- `londim`: Number of grid points in longitude.
- `latmin`: Minimum latitude offset from reference (degrees).
- `latdim`: Number of grid points in latitude.
- `degincr`: Grid spacing in degrees for both latitude and longitude.
- `zmin`: Minimum altitude (meters).
- `zincr`: Vertical grid spacing (meters).
- `zdim`: Number of vertical grid levels.
- `power_threshold`: Minimum beam power weight below which a gate is excluded.
- `missing_key`: Moment name used for signal quality gating (default `"SQI"`).
- `valid_key`: Moment name used for valid-data gating (default `"DBZ"`).
- `heading`: Mean heading of the platform in degrees (default `-9999.0` for missing).
- `metadata`: CF-1.12 global attributes forwarded to the writer (default `MetadataParameters()`).
"""
function grid_radar_latlon_volume(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
    lonmin, londim, latmin, latdim, degincr, zmin, zincr, zdim, power_threshold,
    missing_key="SQI", valid_key="DBZ", heading=-9999.0,
    metadata::MetadataParameters=MetadataParameters())

    # Set the reference to the first location in the volume, but could be a parameter
    #reference_latitude = latmin + Int64(round(latdim/2)) * degincr
    #reference_longitude = lonmin + Int64(round(londim/2)) * degincr

    reference_latitude = radar_volume.latitude[1] - rem(radar_volume.latitude[1], degincr)
    reference_longitude = radar_volume.longitude[1] - rem(radar_volume.longitude[1], degincr)

    latmin = round(reference_latitude + latmin, digits=2)
    lonmin = round(reference_longitude + lonmin, digits=2)

    # Initialize the gridpoints
    # This array is slightly different than Springsteel spectral arrays, need to reconcile later
    gridpoints = initialize_regular_grid(reference_latitude, reference_longitude, lonmin, londim, latmin, latdim, degincr, zmin, zincr, zdim)

    latrad = reference_latitude * pi/180.0
    fac_lat = 111.13209 - 0.56605 * cos(2.0 * latrad)
        + 0.00012 * cos(4.0 * latrad) - 0.000002 * cos(6.0 * latrad)
    fac_lon = 111.41513 * cos(latrad)
        - 0.09455 * cos(3.0 * latrad) + 0.00012 * cos(5.0 * latrad)
    deg_km = sqrt(fac_lat^2 + fac_lon^2)
    h_roi = deg_km * 1000.0 * degincr * 0.75
    v_roi = zincr * 0.75

    radar_grid, latlon_grid = grid_volume(reference_latitude, reference_longitude, gridpoints,
        radar_volume, moment_dict, grid_type_dict, h_roi, v_roi, power_threshold,
        missing_key, valid_key; beam_width = radar_volume.beam_width)

    write_gridded_radar_volume(output_file, index_time, radar_volume.time[1],
        radar_volume.time[end], gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude, heading, metadata)

end

"""
    grid_radar_rhi(radar_volume, moment_dict, grid_type_dict, output_file, index_time, rmin, rincr, rdim, rhi_zmin, rhi_zincr, rhi_zdim, power_threshold, missing_key="SQI", valid_key="DBZ", metadata=MetadataParameters())

Grid a radar RHI (Range-Height Indicator) scan onto a 2D range-height grid and write to a NetCDF file.

Initializes a 2D grid in range and altitude, calls `grid_rhi` for the interpolation, and writes
the output via `write_gridded_radar_rhi`.

# Arguments
- `radar_volume`: Radar volume data structure containing the RHI scan.
- `moment_dict`: Dictionary mapping moment names to integer indices.
- `grid_type_dict`: Dictionary mapping moment indices to interpolation type symbols.
- `output_file`: Path to the output NetCDF file.
- `index_time`: Reference time for the output dataset.
- `rmin`: Minimum range (meters).
- `rincr`: Range grid spacing (meters).
- `rdim`: Number of range grid points.
- `rhi_zmin`: Minimum altitude (meters).
- `rhi_zincr`: Vertical grid spacing (meters).
- `rhi_zdim`: Number of vertical grid points.
- `power_threshold`: Minimum beam power weight below which a gate is excluded.
- `missing_key::String`: Moment name used for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name used for valid-data gating (default `"DBZ"`).
- `metadata`: CF-1.12 global attributes forwarded to the writer (default `MetadataParameters()`).
"""
function grid_radar_rhi(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
        rmin, rincr, rdim, rhi_zmin, rhi_zincr, rhi_zdim, power_threshold,
        missing_key::String="SQI", valid_key::String="DBZ",
        metadata::MetadataParameters=MetadataParameters())

    # Set the reference to the first location in the volume, but could be a parameter
    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    # Initialize the gridpoints
    # This array is slightly different than Springsteel spectral arrays, need to reconcile later
    gridpoints = initialize_regular_grid(rmin, rincr, rdim, rhi_zmin, rhi_zincr, rhi_zdim)

    h_roi = rincr * 0.75
    v_roi = rhi_zincr * 0.75

    radar_grid, latlon_grid = grid_rhi(reference_latitude, reference_longitude, gridpoints,
        radar_volume, moment_dict, grid_type_dict, h_roi, v_roi, power_threshold, missing_key, valid_key;
        beam_width = radar_volume.beam_width)

    write_gridded_radar_rhi(output_file, index_time, radar_volume,
        gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude, metadata)

end

"""
    grid_radar_ppi(radar_volume, moment_dict, grid_type_dict, output_file, index_time, xmin, xincr, xdim, ymin, yincr, ydim, power_threshold, missing_key="SQI", valid_key="DBZ", heading=-9999.0, metadata=MetadataParameters())

Grid a radar PPI (Plan Position Indicator) scan onto a 2D Cartesian grid and write to a NetCDF file.

Initializes a 2D horizontal grid, calls `grid_ppi` for the interpolation, and writes the output
via `write_gridded_radar_ppi`.

# Arguments
- `radar_volume`: Radar volume data structure containing the PPI scan.
- `moment_dict`: Dictionary mapping moment names to integer indices.
- `grid_type_dict`: Dictionary mapping moment indices to interpolation type symbols.
- `output_file`: Path to the output NetCDF file.
- `index_time`: Reference time for the output dataset.
- `xmin`: Minimum x-coordinate of the grid (meters).
- `xincr`: Grid spacing in x (meters).
- `xdim`: Number of grid points in x.
- `ymin`: Minimum y-coordinate of the grid (meters).
- `yincr`: Grid spacing in y (meters).
- `ydim`: Number of grid points in y.
- `power_threshold`: Minimum beam power weight below which a gate is excluded.
- `missing_key`: Moment name used for signal quality gating (default `"SQI"`).
- `valid_key`: Moment name used for valid-data gating (default `"DBZ"`).
- `heading`: Mean heading of the platform in degrees (default `-9999.0` for missing).
- `metadata`: CF-1.12 global attributes forwarded to the writer (default `MetadataParameters()`).
"""
function grid_radar_ppi(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
        xmin, xincr, xdim, ymin, yincr, ydim, power_threshold,
        missing_key="SQI", valid_key="DBZ", heading=-9999.0,
        metadata::MetadataParameters=MetadataParameters())

    # Set the reference to the first location in the volume, but could be a parameter
    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    # Initialize the gridpoints
    # This array is slightly different than Springsteel spectral arrays, need to reconcile later
    gridpoints = initialize_regular_grid(xmin, xincr, xdim, ymin, yincr, ydim)

    h_roi = xincr * 0.75

    radar_grid, latlon_grid = grid_ppi(reference_latitude, reference_longitude, gridpoints,
        radar_volume, moment_dict, grid_type_dict, h_roi, power_threshold, missing_key, valid_key;
        beam_width = radar_volume.beam_width)

    # The legacy `radar` struct has no per-sweep fixed angle; a PPI is a single tilt,
    # so the mean ray elevation is the sweep angle.
    write_gridded_radar_ppi(output_file, index_time, radar_volume,
        gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude, heading, metadata;
        fixed_angle = _mean_elevation(radar_volume))

end

# Mean ray elevation of a legacy `radar` volume, or `nothing` when unavailable.
function _mean_elevation(radar_volume)
    hasproperty(radar_volume, :elevation) || return nothing
    els = radar_volume.elevation
    (els === nothing || isempty(els)) && return nothing
    finite = filter(isfinite, Float64.(els))
    isempty(finite) && return nothing
    return sum(finite) / length(finite)
end

"""
    grid_radar_composite(radar_volume, moment_dict, grid_type_dict, output_file, index_time, xmin, xincr, xdim, ymin, yincr, ydim, missing_key="SQI", valid_key="DBZ", mean_heading=-9999.0, metadata=MetadataParameters())

Grid a radar composite (column-maximum) onto a 2D Cartesian grid and write to a NetCDF file.

Creates a 2D horizontal grid and calls `grid_composite` to select the maximum reflectivity gate
at each horizontal grid point across all elevations. The output is written via `write_gridded_radar_ppi`.

# Arguments
- `radar_volume`: Radar volume data structure.
- `moment_dict`: Dictionary mapping moment names to integer indices.
- `grid_type_dict`: Dictionary mapping moment indices to interpolation type symbols.
- `output_file`: Path to the output NetCDF file.
- `index_time`: Reference time for the output dataset.
- `xmin`: Minimum x-coordinate of the grid (meters).
- `xincr`: Grid spacing in x (meters).
- `xdim`: Number of grid points in x.
- `ymin`: Minimum y-coordinate of the grid (meters).
- `yincr`: Grid spacing in y (meters).
- `ydim`: Number of grid points in y.
- `missing_key`: Moment name used for signal quality gating (default `"SQI"`).
- `valid_key`: Moment name used for valid-data gating (default `"DBZ"`).
- `mean_heading`: Mean heading of the platform in degrees (default `-9999.0` for missing).
- `metadata`: CF-1.12 global attributes forwarded to the writer (default `MetadataParameters()`).
"""
function grid_radar_composite(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
        xmin, xincr, xdim, ymin, yincr, ydim,
        missing_key="SQI", valid_key="DBZ", mean_heading=-9999.0,
        metadata::MetadataParameters=MetadataParameters())

    # Set the reference to the first location in the volume, but could be a parameter
    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    # Initialize the gridpoints
    # This array is slightly different than Springsteel spectral arrays, need to reconcile later
    gridpoints = initialize_regular_grid(xmin, xincr, xdim, ymin, yincr, ydim)

    h_roi = xincr * 0.75

    radar_grid, latlon_grid = grid_composite(reference_latitude, reference_longitude, gridpoints,
        radar_volume, moment_dict, grid_type_dict, h_roi, missing_key, valid_key;
        beam_width = radar_volume.beam_width)

    write_gridded_radar_ppi(output_file, index_time, radar_volume,
        gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude, mean_heading, metadata)

end

"""
    grid_radar_column(radar_volume, moment_dict, grid_type_dict, output_file, index_time, column_zmin, column_zincr, column_zdim, power_threshold, missing_key="SQI", valid_key="DBZ", metadata=MetadataParameters())

Grid a radar volume into a single vertical column profile and write to a NetCDF file.

Initializes a 1D vertical grid at the radar location, calls `grid_column` for the interpolation,
and writes the output via `write_gridded_radar_column`. This is useful for extracting a vertical
profile directly above the radar.

# Arguments
- `radar_volume`: Radar volume data structure.
- `moment_dict`: Dictionary mapping moment names to integer indices.
- `grid_type_dict`: Dictionary mapping moment indices to interpolation type symbols.
- `output_file`: Path to the output NetCDF file.
- `index_time`: Reference time for the output dataset.
- `column_zmin`: Minimum altitude of the column (meters).
- `column_zincr`: Vertical grid spacing (meters).
- `column_zdim`: Number of vertical grid points.
- `power_threshold`: Minimum beam power weight below which a gate is excluded.
- `missing_key::String`: Moment name used for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name used for valid-data gating (default `"DBZ"`).
- `metadata`: CF-1.12 global attributes forwarded to the writer (default `MetadataParameters()`).
"""
function grid_radar_column(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
    column_zmin, column_zincr, column_zdim, power_threshold,
    missing_key::String="SQI", valid_key::String="DBZ",
    metadata::MetadataParameters=MetadataParameters())

    # Set the reference to the first location in the volume, but could be a parameter
    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    # Initialize the gridpoints
    # This array is slightly different than Springsteel spectral arrays, need to reconcile later
    gridpoints = initialize_regular_grid(column_zmin, column_zincr, column_zdim)

    v_roi = column_zincr * 0.75

    radar_grid, latlon_grid = grid_column(reference_latitude, reference_longitude, gridpoints,
        radar_volume, moment_dict, grid_type_dict, v_roi, power_threshold, missing_key, valid_key;
        beam_width = radar_volume.beam_width)

    write_gridded_radar_column(output_file, index_time, radar_volume.time[1],
        radar_volume.time[end], gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude, metadata)

end

"""
    grid_volume(reference_latitude, reference_longitude, gridpoints, radar_volume, moment_dict, grid_type_dict, horizontal_roi, vertical_roi, power_threshold, missing_key="SQI", valid_key="DBZ") -> Tuple{Array{Float64}, Array{Float64}}

Interpolate radar moment data onto a 3D Cartesian grid using beam-weighted averaging.

This is the core gridding engine for 3D volume scans. For each horizontal grid column, a BallTree
is queried to find nearby radar gates within the radius of influence. Vertical matching is then
performed, and weights are computed from the spherical angle difference (beam pattern) and range
ratio. Moments can be interpolated using linear (dBZ-aware), nearest-neighbor, or weighted-average
schemes as specified by `grid_type_dict`. The function is multithreaded over horizontal grid points.

# Arguments
- `reference_latitude::AbstractFloat`: Reference latitude for the Transverse Mercator projection (degrees).
- `reference_longitude::AbstractFloat`: Reference longitude for the Transverse Mercator projection (degrees).
- `gridpoints::AbstractArray`: 4D grid coordinate array from `initialize_regular_grid`.
- `radar_volume::radar`: Radar volume data structure.
- `moment_dict::Dict`: Dictionary mapping moment names to integer indices.
- `grid_type_dict::Dict`: Dictionary mapping moment indices to interpolation symbols (`:linear`, `:nearest`, or default).
- `horizontal_roi::Float64`: Horizontal radius of influence (meters).
- `vertical_roi::Float64`: Vertical radius of influence (meters).
- `power_threshold::Float64`: Minimum beam power weight for a gate to contribute.
- `missing_key::String`: Moment name for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name for valid-data gating (default `"DBZ"`).

# Returns
A tuple `(radar_grid, latlon_grid)` where:
- `radar_grid`: A `(n_moments, zdim, ydim, xdim)` array of gridded moment values. Fill value is `-32768.0` (no data), `-9999.0` (in range but QC'd out).
- `latlon_grid`: A `(ydim, xdim, 2)` array of `[latitude, longitude]` at each horizontal grid point.
"""
function grid_volume(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, gridpoints::AbstractArray,
        radar_volume::radar, moment_dict::Dict, grid_type_dict::Dict, horizontal_roi::Float64, vertical_roi::Float64,
        power_threshold::Float64, missing_key::String="SQI", valid_key::String="DBZ"; beam_width::Real=1.0)

    # Convert the relevant radar information to arrays
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)
    grid_origin, radar_zyx, beams = radar_arrays(reference_latitude, reference_longitude, radar_volume, TM)

    # Gate surface positions + horizontal BallTree (shared with the accumulator
    # path). Edge-referenced beam params for the shared inclusion kernel.
    gate_yx  = _sweep_gate_yx(radar_zyx, beams)
    balltree = BallTree(gate_yx)
    beam_coef = _beam_coef(beam_width)
    beam_cutoff = _beam_cutoff(power_threshold, beam_coef)
    s = sin(beam_cutoff)

    # Allocate the grid for the radar moments and the weights for each radar gate
    # NOTE: the -32768.0 (true missing) / -9999.0 (undetect) literals here and
    # in the sibling legacy `grid_*` workers are the legacy-path defaults. This
    # `radar`/`moment_dict` API has no `DaishoParameters` in scope, so `[io]`
    # fill_value/undetect do NOT reach here; the accumulator/finalize_grid path
    # is the one driven by `[io]`.
    n_moments = length(moment_dict)
    radar_grid = fill(-32768.0,n_moments,size(gridpoints,1),size(gridpoints,2),size(gridpoints,3))
    weights = zeros(Float64,n_moments,size(gridpoints,1),size(gridpoints,2),size(gridpoints,3))

    # Allocate a regular latlon grid for the map
    latlon_grid = Array{Float64}(undef,size(gridpoints,2),size(gridpoints,3),2)

    # Loop through the horizontal indices then do each column
    Threads.@threads for i in CartesianIndices(size(gridpoints)[2:3])

        # Calculate the lat, lon of the gridpoint
        yx_point = gridpoints[1,i,2:3]

        # Using CoordRefSystems pure Julia transform, the Proj.jl wrapper was significantly slower for unknown reasons
        cartTM = convert(TM,Cartesian{WGS84Latest}(grid_origin.x + yx_point[2]u"m", grid_origin.y + yx_point[1]u"m"))
        latlon = convert(LatLon,cartTM)
        latlon_grid[i,1] = ustrip(latlon.lat)
        latlon_grid[i,2] = ustrip(latlon.lon)

        # Edge-referenced conservative query radius; the per-gate d_h/d_v checks
        # inside the shared kernel trim it (see `_gate_grid_geometry`).
        origin_dist = euclidean(yx_point, [0.0, 0.0])
        R_q = (horizontal_roi + origin_dist * s) / (1.0 - s)
        gates = inrange(balltree, yx_point, R_q)

        if !isempty(gates)

            # Found some gates that are within range horizontally
            for j in 1:size(gridpoints,1)

                grid_z = gridpoints[j,i,1]

                # Coverage: any gate within the edge-referenced vertical reach that
                # was scanned (missing_key present) ⇒ flag -9999 (in range, scanned).
                # If none, this grid box is out of range and stays -32768 (missing).
                any_scanned = false
                for gate in gates
                    abs(beams[gate,4] - grid_z) <= vertical_roi + beams[gate,3]*s || continue
                    if !ismissing(radar_volume.moments[gate, moment_dict[missing_key]])
                        any_scanned = true
                        break
                    end
                end
                any_scanned || continue
                for m in 1:n_moments
                    radar_grid[m,j,i] = -9999.0
                end

                # Contributions: valid_key gates, weighted via the shared
                # edge-referenced kernel (identical physics to the 3D accumulator
                # path; Springsteel's spectral path inherits this through here).
                for gate in gates
                    ismissing(radar_volume.moments[gate, moment_dict[valid_key]]) && continue

                    _, _, total_weight = _gate_grid_geometry(grid_z, yx_point,
                        radar_zyx, beams, gate_yx, gate, horizontal_roi,
                        vertical_roi, beam_cutoff, beam_coef)
                    total_weight > 0.0 || continue

                    for m in 1:n_moments
                        if weights[m,j,i] == 0.0
                            # Initialize the radar grid box with 0 since there is a possibility that the beam hit it
                            radar_grid[m,j,i] = 0.0
                        end

                        if !ismissing(radar_volume.moments[gate,m])
                            if grid_type_dict[m] == :linear
                                linear_z = 10.0 ^ (radar_volume.moments[gate,m] / 10.0)
                                radar_grid[m,j,i] += total_weight * linear_z
                                weights[m,j,i] += total_weight
                            elseif grid_type_dict[m] == :nearest
                                if total_weight > weights[m,j,i]
                                    radar_grid[m,j,i] = radar_volume.moments[gate,m]
                                    weights[m,j,i] = total_weight
                                end
                            else
                                radar_grid[m,j,i] += total_weight * radar_volume.moments[gate,m]
                                weights[m,j,i] += total_weight
                            end
                        end
                    end

                end # End of gate loop

                # Divide by the total weight for that gridbox
                for m in 1:n_moments
                    if weights[m,j,i] > 0.0 && grid_type_dict[m] != :nearest
                        radar_grid[m,j,i] /= weights[m,j,i]
                        if grid_type_dict[m] == :linear
                            if radar_grid[m,j,i] > 0.0
                                radar_grid[m,j,i] = 10.0 * log10(radar_grid[m,j,i])
                            else
                                radar_grid[m,j,i] = -9999.0
                            end
                        end
                    elseif weights[m,j,i] == 0.0
                        if radar_grid[m,j,i] == 0.0
                            # Use a different flag to indicate data has been QCed out
                            radar_grid[m,j,i] = -9999.0
                        end
                    end
                end # End of moment loop

            end # End of height loop
        end # End of gate empty test
    end # End of horizontal loop

    # Return the gridded radar array
    return radar_grid, latlon_grid
end

"""
    grid_rhi(reference_latitude, reference_longitude, gridpoints, radar_volume, moment_dict, grid_type_dict, horizontal_roi, vertical_roi, power_threshold, missing_key="SQI", valid_key="DBZ") -> Tuple{Array{Float64}, Array{Float64}}

Interpolate radar moment data onto a 2D range-height grid for an RHI scan using beam-weighted averaging.

Similar to `grid_volume`, but operates on a 2D grid in range and altitude rather than 3D Cartesian space.
A BallTree is constructed on radial distances, and the gridding preserves the RHI azimuth from the scan.
The function is multithreaded over range grid points.

# Arguments
- `reference_latitude::AbstractFloat`: Reference latitude for the Transverse Mercator projection (degrees).
- `reference_longitude::AbstractFloat`: Reference longitude for the Transverse Mercator projection (degrees).
- `gridpoints::AbstractArray`: 3D grid coordinate array from `initialize_regular_grid` (2D version).
- `radar_volume::radar`: Radar volume data structure containing the RHI scan.
- `moment_dict::Dict`: Dictionary mapping moment names to integer indices.
- `grid_type_dict::Dict`: Dictionary mapping moment indices to interpolation symbols.
- `horizontal_roi::Float64`: Range radius of influence (meters).
- `vertical_roi::Float64`: Vertical radius of influence (meters).
- `power_threshold::Float64`: Minimum beam power weight for a gate to contribute.
- `missing_key::String`: Moment name for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name for valid-data gating (default `"DBZ"`).

# Returns
A tuple `(radar_grid, latlon_grid)` where:
- `radar_grid`: A `(n_moments, zdim, rdim)` array of gridded moment values.
- `latlon_grid`: An `(rdim, 2)` array of `[latitude, longitude]` along the RHI azimuth.
"""
function grid_rhi(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, gridpoints::AbstractArray,
        radar_volume::radar, moment_dict::Dict, grid_type_dict::Dict, horizontal_roi::Float64, vertical_roi::Float64,
        power_threshold::Float64, missing_key::String="SQI", valid_key::String="DBZ"; beam_width::Real=1.0)

    # Convert the relevant radar information to arrays
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)
    grid_origin, radar_zyx, beams = radar_arrays(reference_latitude, reference_longitude, radar_volume, TM)

    # Surface radial distance + 1D BallTree (shared with the accumulator path).
    gate_r   = _sweep_gate_r(radar_zyx, beams)
    balltree = BallTree(gate_r)
    beam_coef = _beam_coef(beam_width)
    beam_cutoff = _beam_cutoff(power_threshold, beam_coef)
    s = sin(beam_cutoff)

    # Allocate the grid for the radar moments and the weights for each radar gate
    n_moments = length(moment_dict)
    radar_grid = fill(-32768.0,n_moments,size(gridpoints,1),size(gridpoints,2))
    weights = zeros(Float64,n_moments,size(gridpoints,1),size(gridpoints,2))

    # Allocate a regular latlon grid for the map
    latlon_grid = Array{Float64}(undef,size(gridpoints,2),2)

    # Get the RHI azimuth
    azimuth_rhi = deg2rad(radar_volume.azimuth[1])

    # Loop through the horizontal indices then do each column
    Threads.@threads for i in 1:size(gridpoints)[2]

        # Calculate the lat, lon of the gridpoint
        r_point = gridpoints[1,i,2]
        y_point = r_point * cos(azimuth_rhi)
        x_point = r_point * sin(azimuth_rhi)

        # Using CoordRefSystems pure Julia transform, the Proj.jl wrapper was significantly slower for unknown reasons
        cartTM = convert(TM,Cartesian{WGS84Latest}(grid_origin.x + (x_point)u"m", grid_origin.y + (y_point)u"m"))
        latlon = convert(LatLon,cartTM)
        latlon_grid[i,1] = ustrip(latlon.lat)
        latlon_grid[i,2] = ustrip(latlon.lon)

        # Edge-referenced conservative radial query radius; per-gate d_r/d_v
        # checks below trim it.
        origin_dist = euclidean(r_point, [0.0])
        R_q = (horizontal_roi + origin_dist * s) / (1.0 - s)
        gates = inrange(balltree, [ origin_dist ], R_q)
        if !isempty(gates)

            # Found some gates that are within range horizontally
            for j in 1:size(gridpoints,1)

                grid_z = gridpoints[j,i,1]

                # Coverage: any gate within the edge-referenced vertical reach that
                # was scanned ⇒ flag -9999. If none, stays -32768 (out of range).
                any_scanned = false
                for gate in gates
                    abs(beams[gate,4] - grid_z) <= vertical_roi + beams[gate,3]*s || continue
                    if !ismissing(radar_volume.moments[gate, moment_dict[missing_key]])
                        any_scanned = true
                        break
                    end
                end
                any_scanned || continue
                for m in 1:n_moments
                    radar_grid[m,j,i] = -9999.0
                end

                # Loops through the nearby gates with valid data
                for gate in gates
                    ismissing(radar_volume.moments[gate, moment_dict[valid_key]]) && continue

                    # Calculate the beam intercept to the gridpoint
                    dz = gridpoints[j,i,1] - radar_zyx[gate][1]
                    r = beams[gate,3]

                    # Effective elevation (earth curvature + standard refraction)
                    sine_h = ((dz + Reff)^2 - r^2 - Reff^2) / (2*r*Reff)
                    abs(sine_h) < 1.0 || continue
                    gridpt_el = asin(sine_h)

                    # Edge-referenced inclusion (radial × vertical), no /r.
                    footprint = r * s
                    d_r = abs(gate_r[1,gate] - origin_dist)
                    (d_r <= horizontal_roi + footprint &&
                     abs(beams[gate,4] - grid_z) <= vertical_roi + footprint) || continue

                    # Don't need azimuth, just dx and dy to get the surface range
                    dx = x_point - radar_zyx[gate][3]
                    dy = y_point - radar_zyx[gate][2]

                    # Elevation-only angular weight (RHI is a vertical slice).
                    angle_diff = spherical_angle([beams[gate,1], beams[gate,2]],
                        [beams[gate,1], gridpt_el])
                    angle_weight = exp(-angle_diff * beam_coef)

                    # Range weighting (guarded against the near-radar singularity).
                    gridpt_r = sin(sqrt(dx^2 + dy^2)/Reff) * (Reff + dz) / cos(gridpt_el)
                    r_eff = max(r, 1.0)
                    range_weight = clamp(gridpt_r / r_eff, 0.0, 10.0)
                    if abs(gridpt_r - r) > horizontal_roi || abs(gridpt_r - r) > vertical_roi
                        range_weight = 0.0
                    end
                    total_weight = range_weight * angle_weight

                    # If there is non-zero weight, add it to the grid box
                    if total_weight > 0.0
                        for m in 1:n_moments
                            if weights[m,j,i] == 0.0
                                # Initialize the radar grid box with 0 since there is a possibility that the beam hit it
                                radar_grid[m,j,i] = 0.0
                            end

                            if !ismissing(radar_volume.moments[gate,m])
                                if grid_type_dict[m] == :linear
                                    linear_z = 10.0 ^ (radar_volume.moments[gate,m] / 10.0)
                                    radar_grid[m,j,i] += total_weight * linear_z
                                    weights[m,j,i] += total_weight
                                elseif grid_type_dict[m] == :nearest
                                    if total_weight > weights[m,j,i]
                                        radar_grid[m,j,i] = radar_volume.moments[gate,m]
                                        weights[m,j,i] = total_weight
                                    end
                                else
                                    radar_grid[m,j,i] += total_weight * radar_volume.moments[gate,m]
                                    weights[m,j,i] += total_weight
                                end
                            end
                        end
                    end

                end # End of gate loop

                # Divide by the total weight for that gridbox
                for m in 1:n_moments
                    if weights[m,j,i] > 0.0 && grid_type_dict[m] != :nearest
                        radar_grid[m,j,i] /= weights[m,j,i]
                        if grid_type_dict[m] == :linear
                            if radar_grid[m,j,i] > 0.0
                                radar_grid[m,j,i] = 10.0 * log10(radar_grid[m,j,i])
                            else
                                radar_grid[m,j,i] = -9999.0
                            end
                        end
                    elseif weights[m,j,i] == 0.0
                        if radar_grid[m,j,i] == 0.0
                            # Use a different flag to indicate data has been QCed out
                            radar_grid[m,j,i] = -9999.0
                        end
                    end
                end # End of moment loop

            end # End of height loop
        end # End of gate empty test
    end # End of horizontal loop

    # Return the gridded radar array
    return radar_grid, latlon_grid
end

"""
    grid_ppi(reference_latitude, reference_longitude, gridpoints, radar_volume, moment_dict, grid_type_dict, horizontal_roi, power_threshold, missing_key="SQI", valid_key="DBZ") -> Tuple{Array{Float64}, Array{Float64}}

Interpolate radar moment data onto a 2D Cartesian grid for a PPI scan using beam-weighted averaging.

Similar to `grid_volume`, but operates on a 2D horizontal grid for a single elevation scan. Only
azimuthal angle weighting and range weighting are applied (no vertical matching). The function is
multithreaded over horizontal grid points.

# Arguments
- `reference_latitude::AbstractFloat`: Reference latitude for the Transverse Mercator projection (degrees).
- `reference_longitude::AbstractFloat`: Reference longitude for the Transverse Mercator projection (degrees).
- `gridpoints::AbstractArray`: 3D grid coordinate array from `initialize_regular_grid` (2D version).
- `radar_volume::radar`: Radar volume data structure containing the PPI scan.
- `moment_dict::Dict`: Dictionary mapping moment names to integer indices.
- `grid_type_dict::Dict`: Dictionary mapping moment indices to interpolation symbols.
- `horizontal_roi::Float64`: Horizontal radius of influence (meters).
- `power_threshold::Float64`: Minimum beam power weight for a gate to contribute.
- `missing_key::String`: Moment name for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name for valid-data gating (default `"DBZ"`).

# Returns
A tuple `(radar_grid, latlon_grid)` where:
- `radar_grid`: A `(n_moments, ydim, xdim)` array of gridded moment values.
- `latlon_grid`: A `(ydim, xdim, 2)` array of `[latitude, longitude]` at each grid point.
"""
function grid_ppi(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, gridpoints::AbstractArray,
        radar_volume::radar, moment_dict::Dict, grid_type_dict::Dict, horizontal_roi::Float64,
        power_threshold::Float64, missing_key::String="SQI", valid_key::String="DBZ"; beam_width::Real=1.0)

    # Convert the relevant radar information to arrays
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)
    grid_origin, radar_zyx, beams = radar_arrays(reference_latitude, reference_longitude, radar_volume, TM)

    # Gate surface positions + horizontal BallTree (shared with the accumulator path).
    gate_yx  = _sweep_gate_yx(radar_zyx, beams)
    balltree = BallTree(gate_yx)
    beam_coef = _beam_coef(beam_width)
    beam_cutoff = _beam_cutoff(power_threshold, beam_coef)
    s = sin(beam_cutoff)

    # Allocate the grid for the radar moments and the weights for each radar gate
    n_moments = length(moment_dict)
    radar_grid = fill(-32768.0,n_moments,size(gridpoints,1),size(gridpoints,2))
    weights = zeros(Float64,n_moments,size(gridpoints,1),size(gridpoints,2))

    # Allocate a regular latlon grid for the map
    latlon_grid = Array{Float64}(undef,size(gridpoints,1),size(gridpoints,2),2)

    # Loop through the horizontal indices then do each column
    Threads.@threads for i in CartesianIndices(size(gridpoints)[1:2])

        # Calculate the lat, lon of the gridpoint
        yx_point = gridpoints[i,1:2]

        # Using CoordRefSystems pure Julia transform, the Proj.jl wrapper was significantly slower for unknown reasons
        cartTM = convert(TM,Cartesian{WGS84Latest}(grid_origin.x + yx_point[2]u"m", grid_origin.y + yx_point[1]u"m"))
        latlon = convert(LatLon,cartTM)
        latlon_grid[i,1] = ustrip(latlon.lat)
        latlon_grid[i,2] = ustrip(latlon.lon)

        # Edge-referenced conservative query radius; per-gate d_h trims it below.
        origin_dist = euclidean(yx_point, [0.0, 0.0])
        R_q = (horizontal_roi + origin_dist * s) / (1.0 - s)
        gates = inrange(balltree, yx_point, R_q)

        if !isempty(gates)

            # Found some gates that are within range horizontally
            valid_gates = collect(keys(skipmissing(radar_volume.moments[gates,moment_dict[missing_key]])))
            if !isempty(valid_gates)
                # There is at least one gate in range so set the flags to -9999
                for m in 1:n_moments
                    radar_grid[m,i] = -9999.0
                end
            else
                continue
            end

            # Loops through the nearby gates with valid data
            for gate in gates
                ismissing(radar_volume.moments[gate, moment_dict[valid_key]]) && continue

                r = beams[gate,3]
                # Edge-referenced horizontal inclusion (surface footprint), no /r.
                dh_y = gate_yx[1,gate] - yx_point[1]
                dh_x = gate_yx[2,gate] - yx_point[2]
                sqrt(dh_y^2 + dh_x^2) <= horizontal_roi + r * s || continue

                # Calculate the effective azimuth
                dx = yx_point[2] - radar_zyx[gate][3]
                dy = yx_point[1] - radar_zyx[gate][2]
                gridpt_az = (pi/2.0) - atan(dy, dx)
                if gridpt_az < 0
                    gridpt_az += 2*pi
                end

                # Azimuthal angular weight (elevation held at the beam's own).
                angle_diff = spherical_angle([beams[gate,1], beams[gate,2]],
                    [gridpt_az, beams[gate,2]])
                angle_weight = exp(-angle_diff * beam_coef)

                # Range weighting (guarded against the near-radar singularity).
                gridpt_r = sqrt(dx^2 + dy^2)
                r_eff = max(r, 1.0)
                range_weight = clamp(gridpt_r / r_eff, 0.0, 10.0)
                if abs(gridpt_r - r) > horizontal_roi
                    range_weight = 0.0
                end

                # Multiply weights so that center of beam is 1.0
                total_weight = range_weight * angle_weight

                # If there is non-zero weight, add it to the grid box
                if total_weight > 0.0
                    for m in 1:n_moments
                        if weights[m,i] == 0.0
                            # Initialize the radar grid box with 0 since there is a possibility that the beam hit it
                            radar_grid[m,i] = 0.0
                        end

                        if !ismissing(radar_volume.moments[gate,m])
                            if grid_type_dict[m] == :linear
                                linear_z = 10.0 ^ (radar_volume.moments[gate,m] / 10.0)
                                radar_grid[m,i] += total_weight * linear_z
                                weights[m,i] += total_weight
                            elseif grid_type_dict[m] == :nearest
                                if total_weight > weights[m,i]
                                    radar_grid[m,i] = radar_volume.moments[gate,m]
                                    weights[m,i] = total_weight
                                end
                            else
                                radar_grid[m,i] += total_weight * radar_volume.moments[gate,m]
                                weights[m,i] += total_weight
                            end
                        end
                    end
                end

            end # End of gate loop

            # Divide by the total weight for that gridbox
            for m in 1:n_moments
                if weights[m,i] > 0.0 && grid_type_dict[m] != :nearest
                    radar_grid[m,i] /= weights[m,i]
                    if grid_type_dict[m] == :linear
                        if radar_grid[m,i] > 0.0
                            radar_grid[m,i] = 10.0 * log10(radar_grid[m,i])
                        else
                            radar_grid[m,i] = -9999.0
                        end
                    end
                elseif weights[m,i] == 0.0
                    if radar_grid[m,i] == 0.0
                        # Use a different flag to indicate data has been QCed out
                        radar_grid[m,i] = -9999.0
                    end
                end
            end # End of moment loop
        end # End of gate empty test
    end # End of horizontal loop

    # Return the gridded radar array
    return radar_grid, latlon_grid
end

"""
    grid_composite(reference_latitude, reference_longitude, gridpoints, radar_volume, moment_dict, grid_type_dict, horizontal_roi, missing_key="SQI", valid_key="DBZ") -> Tuple{Array{Float64}, Array{Float64}}

Create a 2D composite (column-maximum) grid from a radar volume.

For each horizontal grid point, finds all nearby gates via a BallTree query and selects the gate
with the maximum value of the `valid_key` moment. All moments for that gate are assigned to the
grid point. This is commonly used to create composite reflectivity maps. The function is multithreaded
over horizontal grid points.

# Arguments
- `reference_latitude::AbstractFloat`: Reference latitude for the Transverse Mercator projection (degrees).
- `reference_longitude::AbstractFloat`: Reference longitude for the Transverse Mercator projection (degrees).
- `gridpoints::AbstractArray`: 3D grid coordinate array from `initialize_regular_grid` (2D version).
- `radar_volume::radar`: Radar volume data structure.
- `moment_dict::Dict`: Dictionary mapping moment names to integer indices.
- `grid_type_dict::Dict`: Dictionary mapping moment indices to interpolation symbols (not used for composite, but kept for interface consistency).
- `horizontal_roi::Float64`: Horizontal radius of influence (meters).
- `missing_key::String`: Moment name for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name used to find the maximum value gate (default `"DBZ"`).

# Returns
A tuple `(radar_grid, latlon_grid)` where:
- `radar_grid`: A `(n_moments, ydim, xdim)` array of composite moment values.
- `latlon_grid`: A `(ydim, xdim, 2)` array of `[latitude, longitude]` at each grid point.
"""
function grid_composite(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, gridpoints::AbstractArray,
        radar_volume::radar, moment_dict::Dict, grid_type_dict::Dict, horizontal_roi::Float64,
        missing_key::String="SQI", valid_key::String="DBZ"; beam_width::Real=1.0, power_threshold::Real=0.5)

    # Convert the relevant radar information to arrays
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)
    grid_origin, radar_zyx, beams = radar_arrays(reference_latitude, reference_longitude, radar_volume, TM)

    # Gate surface positions + horizontal BallTree (shared with the accumulator path).
    # Composite has no beam/range weighting, but gate inclusion is edge-referenced.
    gate_yx  = _sweep_gate_yx(radar_zyx, beams)
    balltree = BallTree(gate_yx)
    beam_cutoff = _beam_cutoff(power_threshold, _beam_coef(beam_width))
    s = sin(beam_cutoff)

    # Allocate the grid for the radar moments and the weights for each radar gate
    n_moments = length(moment_dict)
    radar_grid = fill(-32768.0,n_moments,size(gridpoints,1),size(gridpoints,2))
    weights = zeros(Float64,n_moments,size(gridpoints,1),size(gridpoints,2))

    # Allocate a regular latlon grid for the map
    latlon_grid = Array{Float64}(undef,size(gridpoints,1),size(gridpoints,2),2)

    # Loop through the horizontal indices then do each column
    Threads.@threads for i in CartesianIndices(size(gridpoints)[1:2])

        # Calculate the lat, lon of the gridpoint
        yx_point = gridpoints[i,1:2]

        # Using CoordRefSystems pure Julia transform, the Proj.jl wrapper was significantly slower for unknown reasons
        cartTM = convert(TM,Cartesian{WGS84Latest}(grid_origin.x + yx_point[2]u"m", grid_origin.y + yx_point[1]u"m"))
        latlon = convert(LatLon,cartTM)
        latlon_grid[i,1] = ustrip(latlon.lat)
        latlon_grid[i,2] = ustrip(latlon.lon)

        # Edge-referenced conservative query radius; per-gate d_h trims it below.
        origin_dist = euclidean(yx_point, [0.0, 0.0])
        R_q = (horizontal_roi + origin_dist * s) / (1.0 - s)
        gates = inrange(balltree, yx_point, R_q)

        if !isempty(gates)

            # Found some gates that are within range horizontally
            valid_gates = collect(keys(skipmissing(radar_volume.moments[gates,moment_dict[missing_key]])))
            if !isempty(valid_gates)
                # There is at least one gate in range so set the flags to -9999
                for m in 1:n_moments
                    radar_grid[m,i] = -9999.0
                end
            else
                continue
            end

            # Column maximum over edge-referenced (footprint-included) valid gates.
            best_val = -Inf
            best_gate = 0
            for gate in gates
                vk = radar_volume.moments[gate, moment_dict[valid_key]]
                ismissing(vk) && continue
                r = beams[gate,3]
                dh_y = gate_yx[1,gate] - yx_point[1]
                dh_x = gate_yx[2,gate] - yx_point[2]
                sqrt(dh_y^2 + dh_x^2) <= horizontal_roi + r * s || continue
                if vk > best_val
                    best_val = vk
                    best_gate = gate
                end
            end
            if best_gate != 0
                for m in 1:n_moments
                    if !ismissing(radar_volume.moments[best_gate,m])
                        radar_grid[m,i] = radar_volume.moments[best_gate,m]
                    end
                end # End of moment loop
            end # End of valid gates test
        end # End of gate empty test
    end # End of horizontal loop

    # Return the gridded radar array
    return radar_grid, latlon_grid
end

"""
    grid_column(reference_latitude, reference_longitude, gridpoints, radar_volume, moment_dict, grid_type_dict, vertical_roi, power_threshold, missing_key="SQI", valid_key="DBZ") -> Tuple{Array{Float64}, Array{Float64}}

Interpolate radar moment data onto a 1D vertical column grid using beam-weighted averaging.

Extracts a vertical profile at the radar location by matching gates vertically using elevation angle
weighting and range weighting. This is useful for constructing vertical profiles of radar moments
directly above the radar. The function is multithreaded over vertical grid levels.

# Arguments
- `reference_latitude::AbstractFloat`: Reference latitude for the Transverse Mercator projection (degrees).
- `reference_longitude::AbstractFloat`: Reference longitude for the Transverse Mercator projection (degrees).
- `gridpoints::AbstractArray`: 1D grid coordinate array from `initialize_regular_grid` (1D version).
- `radar_volume::radar`: Radar volume data structure.
- `moment_dict::Dict`: Dictionary mapping moment names to integer indices.
- `grid_type_dict::Dict`: Dictionary mapping moment indices to interpolation symbols.
- `vertical_roi::Float64`: Vertical radius of influence (meters).
- `power_threshold::Float64`: Minimum beam power weight for a gate to contribute.
- `missing_key::String`: Moment name for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name for valid-data gating (default `"DBZ"`).

# Returns
A tuple `(radar_grid, latlon_grid)` where:
- `radar_grid`: A `(n_moments, zdim)` array of gridded moment values.
- `latlon_grid`: A 2-element array `[latitude, longitude]` of the column location.
"""
function grid_column(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, gridpoints::AbstractArray,
    radar_volume::radar, moment_dict::Dict, grid_type_dict::Dict, vertical_roi::Float64,
    power_threshold::Float64, missing_key::String="SQI", valid_key::String="DBZ"; beam_width::Real=1.0)

    # Convert the relevant radar information to arrays
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)
    grid_origin, radar_zyx, beams = radar_arrays(reference_latitude, reference_longitude, radar_volume, TM)

    # Column gridding matches gates by elevation/height only (no horizontal tree).
    beam_coef = _beam_coef(beam_width)
    beam_cutoff = _beam_cutoff(power_threshold, beam_coef)
    s = sin(beam_cutoff)

    # Allocate the grid for the radar moments and the weights for each radar gate
    n_moments = length(moment_dict)
    radar_grid = fill(-32768.0,n_moments,size(gridpoints,1))
    weights = zeros(Float64,n_moments,size(gridpoints,1))

    # Allocate a column grid for the map, but it is just a single point
    latlon_grid = Array{Float64}(undef,2)
    latlon_grid[1] = reference_latitude
    latlon_grid[2] = reference_longitude

    # Loop through the horizontal indices then do each column
    Threads.@threads for i in 1:size(gridpoints)[1]

        grid_z = gridpoints[i]

        # Coverage: any gate within the edge-referenced vertical reach that was
        # scanned ⇒ flag -9999. If none, stays -32768 (out of range).
        any_scanned = false
        for gate in 1:size(beams,1)
            abs(beams[gate,4] - grid_z) <= vertical_roi + beams[gate,3]*s || continue
            if !ismissing(radar_volume.moments[gate, moment_dict[missing_key]])
                any_scanned = true
                break
            end
        end
        any_scanned || continue
        for m in 1:n_moments
            radar_grid[m,i] = -9999.0
        end

        # Loops through the nearby gates with valid data
        valid_gates = collect(keys(skipmissing(radar_volume.moments[:,moment_dict[valid_key]])))
        for gate in valid_gates

            # Calculate the beam intercept to the gridpoint
            dz = gridpoints[i] - radar_zyx[gate][1]
            r = beams[gate,3]

            # Effective elevation (earth curvature + standard refraction)
            sine_h = ((dz + Reff)^2 - r^2 - Reff^2) / (2*r*Reff)
            abs(sine_h) < 1.0 || continue
            gridpt_el = asin(sine_h)

            # Edge-referenced vertical inclusion (beam footprint), no /r.
            abs(beams[gate,4] - grid_z) <= vertical_roi + r * s || continue

            # Elevation-only angular weight.
            angle_diff = spherical_angle([beams[gate,1], beams[gate,2]],
                [beams[gate,1], gridpt_el])
            angle_weight = exp(-angle_diff * beam_coef)

            # Range weighting (guarded against the near-radar singularity).
            gridpt_r = grid_z / cos(beams[gate,2])
            r_eff = max(r, 1.0)
            range_weight = clamp(gridpt_r / r_eff, 0.0, 10.0)
            if abs(gridpt_r - r) > vertical_roi
                range_weight = 0.0
            end

            # Multiply weights so that center of beam is 1.0
            total_weight = range_weight * angle_weight

            # If there is non-zero weight, add it to the grid box
            if total_weight > 0.0
                for m in 1:n_moments
                    if weights[m,i] == 0.0
                        # Initialize the radar grid box with 0 since there is a possibility that the beam hit it
                        radar_grid[m,i] = 0.0
                    end

                    if !ismissing(radar_volume.moments[gate,m])
                        if grid_type_dict[m] == :linear
                            linear_z = 10.0 ^ (radar_volume.moments[gate,m] / 10.0)
                            radar_grid[m,i] += total_weight * linear_z
                            weights[m,i] += total_weight
                        elseif grid_type_dict[m] == :nearest
                            if total_weight > weights[m,i]
                                radar_grid[m,i] = radar_volume.moments[gate,m]
                                weights[m,i] = total_weight
                            end
                        else
                            radar_grid[m,i] += total_weight * radar_volume.moments[gate,m]
                            weights[m,i] += total_weight
                        end
                    end
                end
            end

        end # End of gate loop

        # Divide by the total weight for that gridbox
        for m in 1:n_moments
            if weights[m,i] > 0.0 && grid_type_dict[m] != :nearest
                radar_grid[m,i] /= weights[m,i]
                if grid_type_dict[m] == :linear
                    if radar_grid[m,i] > 0.0
                        radar_grid[m,i] = 10.0 * log10(radar_grid[m,i])
                    else
                        radar_grid[m,i] = -9999.0
                    end
                end
            elseif weights[m,i] == 0.0
                if radar_grid[m,i] == 0.0
                    # Use a different flag to indicate data has been QCed out
                    radar_grid[m,i] = -9999.0
                end
            end
        end # End of moment loop
    end # End of horizontal loop

    # Return the gridded radar array
    return radar_grid, latlon_grid
end

# Build the CF-1.12 global-attribute OrderedDict for a gridded output from a
# user-supplied `MetadataParameters`. `extra` lets a writer append shape-
# specific attrs (e.g., `scan_name` for PPI). `references` is omitted when
# empty, preserving the pre-Issue-#1 behavior of leaving it unwritten unless
# the user opted in.
function _global_attrib_dict(m::MetadataParameters; extra=Pair{String,Any}[])
    attrs = OrderedDict{String,Any}(
        "Conventions"      => m.Conventions,
        "history"          => m.history,
        "institution"      => m.institution,
        "source"           => m.source,
        "instrument"       => m.instrument,
        "title"            => m.title,
        "summary"          => m.summary,
        "creator_name"     => m.creator_name,
        "creator_email"    => m.creator_email,
        "creator_id"       => m.creator_id,
        "project"          => m.project,
        "platform"         => m.platform,
        "keywords"         => m.keywords,
        "processing_level" => m.processing_level,
        "license"          => m.license,
    )
    if !isempty(m.references)
        attrs["references"] = m.references
    end
    for kv in extra
        attrs[String(first(kv))] = last(kv)
    end
    return attrs
end

# Override the shared `common_attrib` sentinels with the run's `[io]` values
# and add the ODIM `_Undetect` attribute (`common_attrib` only carries CF
# `_FillValue`/`missing_value`). `merge` returns a fresh OrderedDict, so the
# module-level `common_attrib` is never mutated.
_with_io_sentinels(attrib, fill_value::Real, undetect::Real) =
    merge(attrib, OrderedDict(
        "_FillValue"    => Float32(fill_value),
        "missing_value" => Float32(fill_value),
        "_Undetect"     => Float32(undetect)))

"""
    write_gridded_radar_volume(file, index_time, start_time, stop_time, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude, reference_longitude, mean_heading, metadata=MetadataParameters(); fill_value=-32768.0, undetect=-9999.0)

Write a gridded 3D radar volume to a CF-1.12 compliant NetCDF file.

Creates a NetCDF file with X, Y, Z dimensions and time, writes grid coordinates, latitude/longitude
fields, a Transverse Mercator grid mapping variable, heading, and all radar moment variables. Any
pre-existing file at the output path is deleted first.

# Arguments
- `file`: Output file path for the NetCDF file.
- `index_time`: Reference time for the time variable.
- `start_time`: Start time of the radar volume scan.
- `stop_time`: Stop time of the radar volume scan.
- `gridpoints`: 4D grid coordinate array from `initialize_regular_grid`.
- `radar_grid`: 4D array `(n_moments, zdim, ydim, xdim)` of gridded moment values.
- `latlon_grid`: 3D array `(ydim, xdim, 2)` of latitude/longitude values.
- `moment_dict`: Dictionary mapping moment names to integer indices.
- `reference_latitude::AbstractFloat`: Latitude of the projection origin (degrees).
- `reference_longitude::AbstractFloat`: Longitude of the projection origin (degrees).
- `mean_heading::AbstractFloat`: Mean platform heading in degrees.
- `metadata::MetadataParameters`: CF-1.12 global attributes (institution, creator, project, platform, …). Defaults to `MetadataParameters()`.
"""
function write_gridded_radar_volume(file, index_time, start_time, stop_time, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, mean_heading::AbstractFloat, metadata::MetadataParameters=MetadataParameters(); fill_value::Real=-32768.0, undetect::Real=-9999.0)

    # Delete any pre-existing file
    rm(file, force=true)

    ds = NCDataset(file, "c", attrib = _global_attrib_dict(metadata))

    # Dimensions
    # Could concatenate multiple volumes here
    #numswps = length(swpstart)
    xdim = size(radar_grid,4)
    ydim = size(radar_grid,3)
    zdim = size(radar_grid,2)
    ds.dim["time"] = Inf   # unlimited (record) dimension, so files can be concatenated
    ds.dim["X"] = xdim
    ds.dim["Y"] = ydim
    ds.dim["Z"] = zdim

    # Declare variables

    nctime = defVar(ds,"time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "time",
        "long_name"                 => "Data time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "axis"                      => "T",
        "comment"                   => "",
    ))

    ncstarttime = defVar(ds,"start_time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "start_time",
        "long_name"                 => "Data start time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "comment"                   => "",
    ))

    ncstoptime = defVar(ds,"stop_time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "stop_time",
        "long_name"                 => "Data stop time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "comment"                   => "",
    ))

    ncx = defVar(ds,"X", Float32, ("X",), attrib = OrderedDict(
        "standard_name"             => "projection_x_coordinate",
        "units"                     => "m",
        "axis"                      => "X",
    ))

    ncy = defVar(ds,"Y", Float32, ("Y",), attrib = OrderedDict(
        "standard_name"             => "projection_y_coordinate",
        "units"                     => "m",
        "axis"                      => "Y",
    ))

    ncz = defVar(ds,"Z", Float32, ("Z",), attrib = OrderedDict(
        "standard_name"             => "altitude",
        "long_name"                 => "constant altitude levels",
        "units"                     => "m",
        "positive"                  => "up",
        "axis"                      => "Z",
    ))

    nclat = defVar(ds,"latitude", Float32, ("X", "Y", "time"), attrib = OrderedDict(
        "standard_name"             => "latitude",
        "units"                     => "degrees_north",
    ))

    nclon = defVar(ds,"longitude", Float32, ("X", "Y", "time"), attrib = OrderedDict(
        "standard_name"             => "longitude",
        "units"                     => "degrees_east",
    ))

    ncgrid_mapping = defVar(ds,"grid_mapping", Int32, (), attrib = OrderedDict(
        "grid_mapping_name"         => "transverse_mercator",
        "scale_factor_at_central_meridian" => 1.0,
        "longitude_of_central_meridian" => reference_longitude,
        "latitude_of_projection_origin" => reference_latitude,
        "reference_ellipsoid_name"  => "GRS80",
        "false_easting"             => 0.0,
        "false_northing"            => 0.0,
    ))

    ncheading = defVar(ds, "heading", Float32, ("time",), attrib = OrderedDict(
        "standard_name"             => "heading",
        "units"                     => "degrees",
    ))

    # Using start time for now, but eventually need to use some reference time
    # time-dimensioned variables use explicit index 1 to grow the unlimited dim
    nctime[1] = datetime2unix(index_time)
    ncstarttime[1] = datetime2unix(start_time)
    ncstoptime[1] = datetime2unix(stop_time)
    ncz[:] = gridpoints[:,1,1,1]
    ncy[:] = gridpoints[1,:,1,2]
    ncx[:] = gridpoints[1,1,:,3]
    nclat[:,:,1] = latlon_grid[:,:,1]'
    nclon[:,:,1] = latlon_grid[:,:,2]'
    ncgrid_mapping[:] = -32768.0
    ncheading[1] = mean_heading

    # Define variables
    perm = (1, 4, 3, 2)
    # moment, z, y, x -> moment, x, y, z
    ncgrid = permutedims(radar_grid,perm)

    # Loop through the moments
    for key in keys(moment_dict)
        if haskey(variable_attrib_dict,key)
            var_attrib = merge(common_attrib, variable_attrib_dict[key])
        else
            var_attrib = merge(common_attrib, variable_attrib_dict["UNKNOWN"])
        end
        var_attrib = _with_io_sentinels(var_attrib, fill_value, undetect)
        ncvar = defVar(ds, key, Float32, ("X", "Y", "Z", "time"), attrib = var_attrib)
        ncvar[:,:,:,1] = ncgrid[moment_dict[key],:,:,:]
    end

    close(ds)
end

"""
    write_gridded_radar_rhi(file, index_time, radar_volume, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude, reference_longitude, metadata=MetadataParameters())

Write a gridded 2D RHI (range-height) radar scan to a CF-1.12 compliant NetCDF file.

Creates a NetCDF file with R (range) and Z (altitude) dimensions and time, writes grid coordinates,
latitude/longitude along the RHI azimuth, a Transverse Mercator grid mapping variable, the RHI
azimuth angle, and all radar moment variables. Any pre-existing file at the output path is deleted first.

# Arguments
- `file`: Output file path for the NetCDF file.
- `index_time`: Reference time for the time variable.
- `radar_volume`: Radar volume data structure (used to extract start/stop times and azimuth).
- `gridpoints`: 3D grid coordinate array from `initialize_regular_grid` (2D version).
- `radar_grid`: 3D array `(n_moments, zdim, rdim)` of gridded moment values.
- `latlon_grid`: 2D array `(rdim, 2)` of latitude/longitude along the RHI.
- `moment_dict`: Dictionary mapping moment names to integer indices.
- `reference_latitude::AbstractFloat`: Latitude of the projection origin (degrees).
- `reference_longitude::AbstractFloat`: Longitude of the projection origin (degrees).
- `metadata::MetadataParameters`: CF-1.12 global attributes. Defaults to `MetadataParameters()`.
"""
function write_gridded_radar_rhi(file, index_time, radar_volume, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, metadata::MetadataParameters=MetadataParameters(); fill_value::Real=-32768.0, undetect::Real=-9999.0)

    # Delete any pre-existing file
    rm(file, force=true)

    start_time = radar_volume.time[1]
    stop_time = radar_volume.time[end]
    azimuth = radar_volume.azimuth[1]

    ds = NCDataset(file, "c", attrib = _global_attrib_dict(metadata))

    # Dimensions
    # Could concatenate multiple volumes here
    #numswps = length(swpstart)
    rdim = size(radar_grid,3)
    zdim = size(radar_grid,2)
    ds.dim["time"] = Inf   # unlimited (record) dimension, so files can be concatenated
    ds.dim["R"] = rdim
    ds.dim["Z"] = zdim

    # Declare variables

    nctime = defVar(ds,"time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "time",
        "long_name"                 => "Data time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "axis"                      => "T",
        "comment"                   => "",
    ))

    ncstarttime = defVar(ds,"start_time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "start_time",
        "long_name"                 => "Data start time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "comment"                   => "",
    ))

    ncstoptime = defVar(ds,"stop_time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "stop_time",
        "long_name"                 => "Data stop time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "comment"                   => "",
    ))

    ncr = defVar(ds,"R", Float32, ("R",), attrib = OrderedDict(
        "standard_name"             => "projection_range_coordinate",
        "units"                     => "m",
        "axis"                      => "R",
    ))

    ncz = defVar(ds,"Z", Float32, ("Z",), attrib = OrderedDict(
        "standard_name"             => "altitude",
        "long_name"                 => "constant altitude levels",
        "units"                     => "m",
        "positive"                  => "up",
        "axis"                      => "Z",
    ))

    nclat = defVar(ds,"latitude", Float32, ("R", "time"), attrib = OrderedDict(
        "standard_name"             => "latitude",
        "units"                     => "degrees_north",
    ))

    nclon = defVar(ds,"longitude", Float32, ("R", "time"), attrib = OrderedDict(
        "standard_name"             => "longitude",
        "units"                     => "degrees_east",
    ))

    ncgrid_mapping = defVar(ds,"grid_mapping", Int32, (), attrib = OrderedDict(
        "grid_mapping_name"         => "transverse_mercator",
        "scale_factor_at_central_meridian" => 1.0,
        "longitude_of_central_meridian" => reference_longitude,
        "latitude_of_projection_origin" => reference_latitude,
        "reference_ellipsoid_name"  => "GRS80",
        "false_easting"             => 0.0,
        "false_northing"            => 0.0,
    ))

    ncazimuth = defVar(ds,"azimuth", Float32, ("time",), attrib = OrderedDict(
        "long_name"                 => "ray_azimuth_angle",
        "units"                     => "degrees",
    ))

    # Using start time for now, but eventually need to use some reference time
    # time-dimensioned variables use explicit index 1 to grow the unlimited dim
    nctime[1] = datetime2unix(index_time)
    ncstarttime[1] = datetime2unix(start_time)
    ncstoptime[1] = datetime2unix(stop_time)
    ncazimuth[1] = azimuth
    ncz[:] = gridpoints[:,1,1]
    ncr[:] = gridpoints[1,:,2]
    nclat[:,1] = latlon_grid[:,1]
    nclon[:,1] = latlon_grid[:,2]
    ncgrid_mapping[:] = -32768.0

    # Define variables
    perm = (1, 3, 2)
    # moment, z, r -> moment, r, z
    ncgrid = permutedims(radar_grid,perm)

    # Loop through the moments
    for key in keys(moment_dict)
        var_attrib = common_attrib
        if haskey(variable_attrib_dict,key)
            var_attrib = merge(common_attrib, variable_attrib_dict[key])
        else
            var_attrib = merge(common_attrib, variable_attrib_dict["UNKNOWN"])
        end
        var_attrib = _with_io_sentinels(var_attrib, fill_value, undetect)
        ncvar = defVar(ds, key, Float32, ("R", "Z", "time"), attrib = var_attrib)
        ncvar[:,:,1] = ncgrid[moment_dict[key],:,:]
    end

    close(ds)
end

"""
    write_gridded_radar_ppi(file, index_time, radar_volume, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude, reference_longitude, mean_heading, metadata=MetadataParameters())

Write a gridded 2D PPI (plan position indicator) radar scan to a CF-1.12 compliant NetCDF file.

Creates a NetCDF file with X and Y dimensions and time, writes grid coordinates, latitude/longitude
fields, a Transverse Mercator grid mapping variable, heading, scan name, and all radar moment variables.
Also used for writing composite grids. Any pre-existing file at the output path is deleted first.

# Arguments
- `file`: Output file path for the NetCDF file.
- `index_time`: Reference time for the time variable.
- `radar_volume`: Radar volume data structure (used to extract start/stop times and scan name).
- `gridpoints`: 3D grid coordinate array from `initialize_regular_grid` (2D version).
- `radar_grid`: 3D array `(n_moments, ydim, xdim)` of gridded moment values.
- `latlon_grid`: 3D array `(ydim, xdim, 2)` of latitude/longitude values.
- `moment_dict`: Dictionary mapping moment names to integer indices.
- `reference_latitude::AbstractFloat`: Latitude of the projection origin (degrees).
- `reference_longitude::AbstractFloat`: Longitude of the projection origin (degrees).
- `mean_heading::AbstractFloat`: Mean platform heading in degrees.
- `metadata::MetadataParameters`: CF-1.12 global attributes. The PPI writer also injects `scan_name` from `radar_volume`. Defaults to `MetadataParameters()`.

# Keywords
- `fixed_angle`: the sweep's fixed elevation angle in degrees, written as both a global
  attribute and a scalar `fixed_angle` variable. Pass `nothing` (the default) when the
  angle is not meaningful — a composite spans every elevation, and so does a multi-sweep
  PPI grid. Downstream products that select by tilt (see [`build_hybrid_scan`](@ref))
  read this back, so a single-sweep PPI grid should always set it.
"""
function write_gridded_radar_ppi(file, index_time, radar_volume, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, mean_heading::AbstractFloat, metadata::MetadataParameters=MetadataParameters(); fill_value::Real=-32768.0, undetect::Real=-9999.0, fixed_angle::Union{Real,Nothing}=nothing)

    # Delete any pre-existing file
    rm(file, force=true)

    start_time = radar_volume.time[1]
    stop_time = radar_volume.time[end]
    scan_name = radar_volume.scan_name

    extra = Pair{String,Any}["scan_name" => scan_name]
    fixed_angle === nothing || push!(extra, "fixed_angle" => Float32(fixed_angle))
    ds = NCDataset(file, "c", attrib = _global_attrib_dict(metadata; extra=extra))

    # Dimensions
    # Could concatenate multiple volumes here
    #numswps = length(swpstart)
    xdim = size(radar_grid,3)
    ydim = size(radar_grid,2)
    ds.dim["time"] = Inf   # unlimited (record) dimension, so files can be concatenated
    ds.dim["X"] = xdim
    ds.dim["Y"] = ydim

    # Declare variables

    nctime = defVar(ds,"time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "time",
        "long_name"                 => "Data time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "axis"                      => "T",
        "comment"                   => "",
    ))

    ncstarttime = defVar(ds,"start_time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "start_time",
        "long_name"                 => "Data start time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "comment"                   => "",
    ))

    ncstoptime = defVar(ds,"stop_time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "stop_time",
        "long_name"                 => "Data stop time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "comment"                   => "",
    ))

    ncx = defVar(ds,"X", Float32, ("X",), attrib = OrderedDict(
        "standard_name"             => "projection_x_coordinate",
        "units"                     => "m",
        "axis"                      => "X",
    ))

    ncy = defVar(ds,"Y", Float32, ("Y",), attrib = OrderedDict(
        "standard_name"             => "projection_y_coordinate",
        "units"                     => "m",
        "axis"                      => "Y",
    ))


    nclat = defVar(ds,"latitude", Float32, ("X", "Y", "time"), attrib = OrderedDict(
        "standard_name"             => "latitude",
        "units"                     => "degrees_north",
    ))

    nclon = defVar(ds,"longitude", Float32, ("X", "Y", "time"), attrib = OrderedDict(
        "standard_name"             => "longitude",
        "units"                     => "degrees_east",
    ))

    ncgrid_mapping = defVar(ds,"grid_mapping", Int32, (), attrib = OrderedDict(
        "grid_mapping_name"         => "transverse_mercator",
        "scale_factor_at_central_meridian" => 1.0,
        "longitude_of_central_meridian" => reference_longitude,
        "latitude_of_projection_origin" => reference_latitude,
        "reference_ellipsoid_name"  => "GRS80",
        "false_easting"             => 0.0,
        "false_northing"            => 0.0,
    ))

    ncheading = defVar(ds, "heading", Float32, ("time",), attrib = OrderedDict(
        "standard_name"             => "heading",
        "units"                     => "degrees",
    ))

    # Scalar sweep geometry, so a single-sweep PPI grid is self-describing about the
    # tilt it came from (the global attribute above carries the same value).
    if fixed_angle !== nothing
        ncfixed = defVar(ds, "fixed_angle", Float32, (), attrib = OrderedDict(
            "standard_name"         => "fixed_angle",
            "long_name"             => "fixed_angle_for_sweep",
            "units"                 => "degrees",
        ))
        ncfixed[:] = Float32(fixed_angle)
    end

    # Using start time for now, but eventually need to use some reference time
    # time-dimensioned variables use explicit index 1 to grow the unlimited dim
    nctime[1] = datetime2unix(index_time)
    ncstarttime[1] = datetime2unix(start_time)
    ncstoptime[1] = datetime2unix(stop_time)
    ncy[:] = gridpoints[:,1,1]
    ncx[:] = gridpoints[1,:,2]
    nclat[:,:,1] = latlon_grid[:,:,1]'
    nclon[:,:,1] = latlon_grid[:,:,2]'
    ncgrid_mapping[:] = -32768.0
    ncheading[1] = mean_heading

    # Define variables
    perm = (1, 3, 2)
    # moment, y, x -> moment, x, y
    ncgrid = permutedims(radar_grid,perm)

    # Loop through the moments
    for key in keys(moment_dict)
        var_attrib = common_attrib
        if haskey(variable_attrib_dict,key)
            var_attrib = merge(common_attrib, variable_attrib_dict[key])
        else
            var_attrib = merge(common_attrib, variable_attrib_dict["UNKNOWN"])
        end
        var_attrib = _with_io_sentinels(var_attrib, fill_value, undetect)
        ncvar = defVar(ds, key, Float32, ("X", "Y", "time"), attrib = var_attrib)
        ncvar[:,:,1] = ncgrid[moment_dict[key],:,:]
    end

    close(ds)
end

# Coordinate scaffolding shared by every 2-D X/Y gridded product, as
# (variable name, dimension names). Copied verbatim from a template file by
# `write_gridded_fields_2d`. `fixed_angle` is deliberately absent: a derived
# product spans more than one sweep and must not claim a single tilt.
const _GRID2D_SCAFFOLD = (
    ("time",         ("time",)),
    ("start_time",   ("time",)),
    ("stop_time",    ("time",)),
    ("X",            ("X",)),
    ("Y",            ("Y",)),
    ("latitude",     ("X", "Y", "time")),
    ("longitude",    ("X", "Y", "time")),
    ("grid_mapping", ()),
    ("heading",      ("time",)),
)

"""
    write_gridded_fields_2d(output_file, template_file, fields, metadata=MetadataParameters();
                            index_time=nothing, fill_value=-32768.0, undetect=-9999.0,
                            extra_attrib=Pair{String,Any}[])

Write a derived 2-D X/Y product to a CF-1.12 NetCDF file, taking its coordinate
scaffolding — `X`, `Y`, `latitude`, `longitude`, `time`, `start_time`, `stop_time`,
`grid_mapping` and `heading` — verbatim from an existing gridded PPI/composite file
`template_file`, and writing each entry of `fields` (name → `(xdim, ydim)` array) as
an `(X, Y, time)` `Float32` variable.

Variable attributes come from `variable_attrib_dict` (falling back to `"UNKNOWN"`)
with the `[io]` sentinels applied, exactly as the gridding writers do. The result is
layout-identical to a gridded PPI, so [`read_gridded_ppi`](@ref) and the plotting
steps consume it unchanged.

`index_time` overrides the template's `time` value; `nothing` keeps the template's.
Any pre-existing `output_file` is deleted first.

Used by [`build_hybrid_scan`](@ref); reusable by any future post-gridding 2-D product.
"""
function write_gridded_fields_2d(output_file::AbstractString, template_file::AbstractString,
        fields::AbstractDict, metadata::MetadataParameters=MetadataParameters();
        index_time=nothing, fill_value::Real=-32768.0, undetect::Real=-9999.0,
        extra_attrib=Pair{String,Any}[])

    rm(output_file, force=true)

    NCDataset(template_file) do src
        for name in ("X", "Y")
            haskey(src.dim, name) || throw(ArgumentError(
                "write_gridded_fields_2d: template $template_file has no `$name` " *
                "dimension; it must be a 2-D gridded PPI/composite file."))
        end
        xdim = src.dim["X"]
        ydim = src.dim["Y"]
        for (name, arr) in fields
            size(arr) == (xdim, ydim) || throw(ArgumentError(
                "write_gridded_fields_2d: field \"$name\" is $(size(arr)) but the " *
                "template grid is ($xdim, $ydim)."))
        end

        ds = NCDataset(output_file, "c",
            attrib = _global_attrib_dict(metadata; extra=extra_attrib))
        try
            ds.dim["time"] = Inf   # unlimited, so outputs stay concatenable
            ds.dim["X"] = xdim
            ds.dim["Y"] = ydim

            for (name, dims) in _GRID2D_SCAFFOLD
                haskey(src, name) || continue
                sv = src[name].var
                v = defVar(ds, name, eltype(sv), dims;
                           attrib = OrderedDict(src[name].attrib))
                if isempty(dims)
                    v[:] = sv[]
                else
                    raw = Array(sv)
                    v[ntuple(k -> 1:size(raw, k), ndims(raw))...] = raw
                end
            end
            index_time === nothing || (ds["time"][1] = datetime2unix(index_time))

            for name in sort!(collect(keys(fields)))
                attrib = haskey(variable_attrib_dict, name) ?
                    merge(common_attrib, variable_attrib_dict[name]) :
                    merge(common_attrib, variable_attrib_dict["UNKNOWN"])
                attrib = _with_io_sentinels(attrib, fill_value, undetect)
                v = defVar(ds, name, Float32, ("X", "Y", "time"), attrib = attrib)
                v[:, :, 1] = Float32.(fields[name])
            end
        finally
            close(ds)
        end
    end
    return output_file
end

"""
    write_gridded_radar_column(file, index_time, start_time, stop_time, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude, reference_longitude, metadata=MetadataParameters())

Write a gridded 1D vertical column profile to a CF-1.12 compliant NetCDF file.

Creates a NetCDF file with Z (altitude) dimension and time, writes the vertical grid coordinates,
latitude/longitude of the column location, a Transverse Mercator grid mapping variable, and all
radar moment variables. Any pre-existing file at the output path is deleted first.

# Arguments
- `file`: Output file path for the NetCDF file.
- `index_time`: Reference time for the time variable.
- `start_time`: Start time of the radar volume scan.
- `stop_time`: Stop time of the radar volume scan.
- `gridpoints`: 1D grid coordinate array from `initialize_regular_grid` (1D version).
- `radar_grid`: 2D array `(n_moments, zdim)` of gridded moment values.
- `latlon_grid`: 2-element array `[latitude, longitude]` of the column location.
- `moment_dict`: Dictionary mapping moment names to integer indices.
- `reference_latitude::AbstractFloat`: Latitude of the column location (degrees).
- `reference_longitude::AbstractFloat`: Longitude of the column location (degrees).
- `metadata::MetadataParameters`: CF-1.12 global attributes. Defaults to `MetadataParameters()`.
"""
function write_gridded_radar_column(file, index_time, start_time, stop_time, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, metadata::MetadataParameters=MetadataParameters(); fill_value::Real=-32768.0, undetect::Real=-9999.0)

    # Delete any pre-existing file
    rm(file, force=true)

    ds = NCDataset(file, "c", attrib = _global_attrib_dict(metadata))

    # Dimensions
    # Could concatenate multiple volumes here
    #numswps = length(swpstart)
    zdim = size(radar_grid,2)
    ds.dim["time"] = Inf   # unlimited (record) dimension, so files can be concatenated
    ds.dim["Z"] = zdim

    # Declare variables

    nctime = defVar(ds,"time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "time",
        "long_name"                 => "Data time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "axis"                      => "T",
        "comment"                   => "",
    ))

    ncstarttime = defVar(ds,"start_time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "start_time",
        "long_name"                 => "Data start time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "comment"                   => "",
    ))

    ncstoptime = defVar(ds,"stop_time", Float64, ("time",), attrib = OrderedDict(
        "standard_name"             => "stop_time",
        "long_name"                 => "Data stop time",
        "units"                     => "seconds since 1970-01-01T00:00:00Z",
        "comment"                   => "",
    ))

    ncz = defVar(ds,"Z", Float32, ("Z",), attrib = OrderedDict(
        "standard_name"             => "altitude",
        "long_name"                 => "constant altitude levels",
        "units"                     => "m",
        "positive"                  => "up",
        "axis"                      => "Z",
    ))

    nclat = defVar(ds,"latitude", Float32, ("time",), attrib = OrderedDict(
        "standard_name"             => "latitude",
        "units"                     => "degrees_north",
    ))

    nclon = defVar(ds,"longitude", Float32, ("time",), attrib = OrderedDict(
        "standard_name"             => "longitude",
        "units"                     => "degrees_east",
    ))

    ncgrid_mapping = defVar(ds,"grid_mapping", Int32, (), attrib = OrderedDict(
        "grid_mapping_name"         => "tranverse_mercator",
        "scale_factor_at_central_meridian" => 1.0,
        "longitude_of_central_meridian" => reference_longitude,
        "latitude_of_projection_origin" => reference_latitude,
        "reference_ellipsoid_name"  => "GRS80",
        "false_easting"             => 0.0,
        "false_northing"            => 0.0,
    ))

    # Using start time for now, but eventually need to use some reference time
    # time-dimensioned variables use explicit index 1 to grow the unlimited dim
    nctime[1] = datetime2unix(index_time)
    ncstarttime[1] = datetime2unix(start_time)
    ncstoptime[1] = datetime2unix(stop_time)
    ncz[:] = gridpoints[:]
    nclat[1] = latlon_grid[1]
    nclon[1] = latlon_grid[2]
    ncgrid_mapping[:] = -32768.0

    # Loop through the moments
    for key in keys(moment_dict)
        var_attrib = common_attrib
        if haskey(variable_attrib_dict,key)
            var_attrib = merge(common_attrib, variable_attrib_dict[key])
        else
            var_attrib = merge(common_attrib, variable_attrib_dict["UNKNOWN"])
        end
        var_attrib = _with_io_sentinels(var_attrib, fill_value, undetect)
        ncvar = defVar(ds, key, Float32, ("Z", "time"), attrib = var_attrib)
        ncvar[:,1] = radar_grid[moment_dict[key],:]
    end

    close(ds)
end

"""
    read_latlon_gridded_radar(file, moment_dict) -> Tuple

Read a gridded radar dataset from a NetCDF file with legacy lat/lon grid format.

Opens the specified NetCDF file and reads the `x0`, `y0`, `z0` coordinate arrays, time bounds,
and all radar moment data specified in `moment_dict`.

# Arguments
- `file`: Path to the input NetCDF file.
- `moment_dict`: Dictionary mapping moment names (e.g., `"DBZ"`) to integer indices.

# Returns
A tuple `(x0, y0, z0, start_time, stop_time, radardata)` where:
- `x0`, `y0`, `z0`: Coordinate arrays from the file.
- `start_time`, `stop_time`: Time bounds of the data.
- `radardata`: A `(n_moments, n_points)` array of `Union{Missing, Float32}` moment values.
"""
function read_latlon_gridded_radar(file, moment_dict)

    inputds = Dataset(file);

    x0 = inputds["x0"]
    y0 = inputds["y0"]
    z0 = inputds["z0"]
    start_time = inputds["start_time"]
    stop_time = inputds["stop_time"]

    # Store radar data
    n_moments = length(moment_dict)
    n_points = length(x0)*length(y0)*length(z0)
    radardata = Array{Union{Missing, Float32}}(undef,n_moments,n_points)
    for key in keys(moment_dict)
        radardata[moment_dict[key],:] = inputds[key][:]
    end

    return x0, y0, z0, start_time, stop_time, radardata
end

"""
    read_cartesian_gridded_radar(file, moment_dict) -> Tuple

Read a gridded radar dataset from a NetCDF file with legacy Cartesian grid format.

Opens the specified NetCDF file and reads the `x0`, `y0`, `z0` coordinate arrays, `lat0`/`lon0`
geographic coordinates, time bounds, and all radar moment data specified in `moment_dict`.

# Arguments
- `file`: Path to the input NetCDF file.
- `moment_dict`: Dictionary mapping moment names (e.g., `"DBZ"`) to integer indices.

# Returns
A tuple `(x0, y0, z0, lat0, lon0, start_time, stop_time, radardata)` where:
- `x0`, `y0`, `z0`: Cartesian coordinate arrays from the file.
- `lat0`, `lon0`: Geographic coordinate arrays from the file.
- `start_time`, `stop_time`: Time bounds of the data.
- `radardata`: A `(n_moments, n_points)` array of `Union{Missing, Float32}` moment values.
"""
function read_cartesian_gridded_radar(file, moment_dict)

    inputds = Dataset(file);

    x0 = inputds["x0"]
    y0 = inputds["y0"]
    z0 = inputds["z0"]
    lon0 = inputds["lon0"]
    lat0 = inputds["lat0"]
    start_time = inputds["start_time"]
    stop_time = inputds["stop_time"]

    # Store radar data
    n_moments = length(moment_dict)
    n_points = length(x0)*length(y0)*length(z0)
    radardata = Array{Union{Missing, Float32}}(undef,n_moments,n_points)
    for key in keys(moment_dict)
        radardata[moment_dict[key],:] = inputds[key][:]
    end

    return x0, y0, z0, lat0, lon0, start_time, stop_time, radardata
end

# Core 3D-volume reader (no deprecation warning). Returns coordinates and an
# index-keyed `(n_moments, n_points)` radardata array. Shared by the deprecated
# moment_dict method and the Fields-API `DaishoParameters` method below.
function _read_gridded_radar(file, moment_dict)

    inputds = Dataset(file);

    x = inputds["X"]
    y = inputds["Y"]
    z = inputds["Z"]
    lon = inputds["longitude"]
    lat = inputds["latitude"]
    start_time = inputds["start_time"]
    stop_time = inputds["stop_time"]

    # Store radar data
    n_moments = length(moment_dict)
    n_points = length(x)*length(y)*length(z)
    radardata = Array{Union{Missing, Float32}}(undef,n_moments,n_points)
    for key in keys(moment_dict)
        radardata[moment_dict[key],:] = inputds[key][:]
    end

    return x, y, z, lat, lon, start_time, stop_time, radardata
end

"""
    read_gridded_radar(file, moment_dict::AbstractDict) -> Tuple

!!! warning "Deprecated"
    The `moment_dict` (name→index) reader is the legacy v0.1 API. Prefer
    `read_gridded_radar(file, p::DaishoParameters)`, which returns fields keyed
    by name and resolves I/O sentinels from `p.io`.

Read a gridded 3D radar volume from a NetCDF file with X, Y, Z coordinates.

# Returns
A tuple `(x, y, z, lat, lon, start_time, stop_time, radardata)` where `radardata`
is a `(n_moments, n_points)` array of `Union{Missing, Float32}` moment values
indexed by `moment_dict`.
"""
function read_gridded_radar(file, moment_dict::AbstractDict)
    Base.depwarn("read_gridded_radar(file, moment_dict) is deprecated; pass a " *
                 "DaishoParameters instead.", :read_gridded_radar)
    return _read_gridded_radar(file, moment_dict)
end

"""
    read_gridded_radar(file, p::DaishoParameters) -> NamedTuple

Fields-API reader for a gridded 3D radar volume. Returns
`(; X, Y, Z, latitude, longitude, start_time, stop_time, fields, io)` where
`fields::Dict{String,Array{Float32,3}}` is keyed by field name (every field in
`p`), each shaped `(length(X), length(Y), length(Z))`. `io` is `p.io` so callers
resolve `fill_value`/`undetect` without re-reading config. Sentinels are
preserved in the data (not collapsed) — display masking is the caller's choice
(see [`mask_sentinels`](@ref)).
"""
function read_gridded_radar(file, p::DaishoParameters)
    md = field_index_dict(p)
    x, y, z, lat, lon, t0, t1, radardata = _read_gridded_radar(file, md)
    fields = Dict(name => reshape(Float32.(coalesce.(radardata[i, :], Float32(p.io.fill_value))),
                                  length(x), length(y), length(z)) for (name, i) in md)
    _merge_echo_fields!(fields, file, p, (length(x), length(y), length(z)))
    return (; X=collect(x), Y=collect(y), Z=collect(z),
            latitude=collect(lat), longitude=collect(lon),
            start_time=collect(t0), stop_time=collect(t1), fields, io=p.io)
end

# Core 2D-PPI/composite reader (no deprecation warning). Shared by the
# deprecated moment_dict method and the Fields-API method below.
function _read_gridded_ppi(file, moment_dict)

    inputds = Dataset(file);

    x = inputds["X"]
    y = inputds["Y"]
    lon = inputds["longitude"]
    lat = inputds["latitude"]
    start_time = inputds["start_time"]
    stop_time = inputds["stop_time"]

    # Store radar data
    n_moments = length(moment_dict)
    n_points = length(x)*length(y)
    radardata = Array{Union{Missing, Float32}}(undef,n_moments,n_points)
    for key in keys(moment_dict)
        radardata[moment_dict[key],:] = inputds[key][:]
    end

    return x, y, lat, lon, start_time, stop_time, radardata
end

"""
    read_gridded_ppi(file, moment_dict::AbstractDict) -> Tuple

!!! warning "Deprecated"
    The `moment_dict` reader is the legacy v0.1 API. Prefer
    `read_gridded_ppi(file, p::DaishoParameters)`.

Read a gridded 2D PPI/composite radar scan from a NetCDF file with X, Y
coordinates. Returns `(x, y, lat, lon, start_time, stop_time, radardata)` where
`radardata` is a `(n_moments, n_points)` array indexed by `moment_dict`.
"""
function read_gridded_ppi(file, moment_dict::AbstractDict)
    Base.depwarn("read_gridded_ppi(file, moment_dict) is deprecated; pass a " *
                 "DaishoParameters instead.", :read_gridded_ppi)
    return _read_gridded_ppi(file, moment_dict)
end

"""
    read_gridded_ppi(file, p::DaishoParameters) -> NamedTuple

Fields-API reader for a gridded 2D PPI/composite scan. Returns
`(; X, Y, latitude, longitude, start_time, stop_time, fields, io)` where
`fields::Dict{String,Matrix{Float32}}` is keyed by field name, each shaped
`(length(X), length(Y))`. Sentinels are preserved; see [`mask_sentinels`](@ref).
"""
function read_gridded_ppi(file, p::DaishoParameters)
    md = field_index_dict(p)
    x, y, lat, lon, t0, t1, radardata = _read_gridded_ppi(file, md)
    fields = Dict(name => reshape(Float32.(coalesce.(radardata[i, :], Float32(p.io.fill_value))),
                                  length(x), length(y)) for (name, i) in md)
    _merge_echo_fields!(fields, file, p, (length(x), length(y)))
    return (; X=collect(x), Y=collect(y),
            latitude=collect(lat), longitude=collect(lon),
            start_time=collect(t0), stop_time=collect(t1), fields, io=p.io)
end

# Core 2D-RHI reader (no deprecation warning). Shared by the deprecated
# moment_dict method and the Fields-API method below.
function _read_gridded_rhi(file, moment_dict)

    inputds = Dataset(file);

    R = inputds["R"]
    Z = inputds["Z"]
    lon = inputds["longitude"]
    lat = inputds["latitude"]
    start_time = inputds["start_time"]
    stop_time = inputds["stop_time"]

    # Store radar data
    n_moments = length(moment_dict)
    n_points = length(R)*length(Z)
    radardata = Array{Union{Missing, Float32}}(undef,n_moments,n_points)
    for key in keys(moment_dict)
        radardata[moment_dict[key],:] = inputds[key][:]
    end

    return R, Z, lat, lon, start_time, stop_time, radardata
end

"""
    read_gridded_rhi(file, moment_dict::AbstractDict) -> Tuple

!!! warning "Deprecated"
    The `moment_dict` reader is the legacy v0.1 API. Prefer
    `read_gridded_rhi(file, p::DaishoParameters)`.

Read a gridded 2D RHI radar scan from a NetCDF file with R (range) and Z
(altitude) coordinates. Returns `(R, Z, lat, lon, start_time, stop_time,
radardata)` where `radardata` is a `(n_moments, n_points)` array indexed by
`moment_dict`.
"""
function read_gridded_rhi(file, moment_dict::AbstractDict)
    Base.depwarn("read_gridded_rhi(file, moment_dict) is deprecated; pass a " *
                 "DaishoParameters instead.", :read_gridded_rhi)
    return _read_gridded_rhi(file, moment_dict)
end

"""
    read_gridded_rhi(file, p::DaishoParameters) -> NamedTuple

Fields-API reader for a gridded 2D RHI scan. Returns
`(; R, Z, latitude, longitude, start_time, stop_time, fields, io)` where
`fields::Dict{String,Matrix{Float32}}` is keyed by field name, each shaped
`(length(R), length(Z))`. Sentinels are preserved; see [`mask_sentinels`](@ref).
"""
function read_gridded_rhi(file, p::DaishoParameters)
    md = field_index_dict(p)
    R, Z, lat, lon, t0, t1, radardata = _read_gridded_rhi(file, md)
    fields = Dict(name => reshape(Float32.(coalesce.(radardata[i, :], Float32(p.io.fill_value))),
                                  length(R), length(Z)) for (name, i) in md)
    _merge_echo_fields!(fields, file, p, (length(R), length(Z)))
    return (; R=collect(R), Z=collect(Z),
            latitude=collect(lat), longitude=collect(lon),
            start_time=collect(t0), stop_time=collect(t1), fields, io=p.io)
end

"""
    mask_sentinels(a::AbstractArray, io::IOParameters) -> Array{Float32}

Return a `Float32` copy of `a` with both the true-missing (`io.fill_value`) and
undetect (`io.undetect`) sentinels replaced by `NaN`, for display. The gridded
Fields-API readers keep the raw sentinels so the missing-vs-undetect distinction
is preserved in the data; call this only when rendering.
"""
function mask_sentinels(a::AbstractArray, io::IOParameters)
    b = Array{Float32}(a)
    replace!(b, Float32(io.fill_value) => NaN32, Float32(io.undetect) => NaN32)
    return b
end

# ── Parameter-struct overloads ──────────────────────────────────────────────
# Convenience methods that pull configuration out of a `DaishoParameters` and
# delegate to the long-positional driver above. Adding a new knob means
# extending the struct + `defaults.toml` and updating the delegating call —
# callers using `p::DaishoParameters` do not change.

"""
    grid_radar_volume(radar_volume, output_file, index_time, p::DaishoParameters; heading=-9999.0)

Parameter-struct overload reading from `p.grid.volume`, `p.gridding`, and
`p.moments`. `p.grid.volume` falls back to `p.grid.cartesian` when no
`[grid.volume]` table is configured.
"""
# Post-write hook: when `[echo]` is enabled, append the hydrometeor-ID and
# rain-rate products to the just-written grid file. Implemented as a post-write
# append so every geometry (volume/latlon/rhi/ppi) and the accumulator path share
# the single `add_echo_products!` implementation; the resulting NetCDF is
# identical to computing the products before the write.
function _maybe_add_echo_products(output_file::AbstractString, p::DaishoParameters)
    (:echo in p.provided && p.echo.enabled) || return nothing
    add_echo_products!(output_file, p)
    return nothing
end

function grid_radar_volume(radar_volume::radar, output_file::AbstractString,
                            index_time, p::DaishoParameters; heading::Real=-9999.0)
    g = require_section(p, :grid, :volume; for_op="grid_radar_volume")
    gd = p.gridding
    mk = field_with_tag(p, :define_scanned;   for_op="grid_radar_volume")
    dk = field_with_tag(p, :define_detection; for_op="grid_radar_volume")
    grid_radar_volume(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.xmin, g.xincr, g.xdim, g.ymin, g.yincr, g.ydim, g.zmin, g.zincr, g.zdim,
        gd.power_threshold, mk, dk, heading,
        p.grid.metadata)
    _maybe_add_echo_products(output_file, p)
end

"""
    grid_radar_latlon_volume(radar_volume, output_file, index_time, p::DaishoParameters; heading=-9999.0)

Parameter-struct overload reading from `p.grid.latlon`, `p.gridding`, and
`p.moments`.
"""
function grid_radar_latlon_volume(radar_volume::radar, output_file::AbstractString,
                                   index_time, p::DaishoParameters; heading::Real=-9999.0)
    g = require_section(p, :grid, :latlon; for_op="grid_radar_latlon_volume")
    gd = p.gridding
    mk = field_with_tag(p, :define_scanned;   for_op="grid_radar_latlon_volume")
    dk = field_with_tag(p, :define_detection; for_op="grid_radar_latlon_volume")
    grid_radar_latlon_volume(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.lonmin, g.londim, g.latmin, g.latdim, g.degincr,
        g.zmin, g.zincr, g.zdim,
        gd.power_threshold, mk, dk, heading,
        p.grid.metadata)
    _maybe_add_echo_products(output_file, p)
end

"""
    grid_radar_rhi(radar_volume, output_file, index_time, p::DaishoParameters)

Parameter-struct overload reading from `p.grid.rhi`, `p.gridding`, and `p.moments`.
"""
function grid_radar_rhi(radar_volume::radar, output_file::AbstractString,
                         index_time, p::DaishoParameters)
    g = require_section(p, :grid, :rhi; for_op="grid_radar_rhi")
    gd = p.gridding
    mk = field_with_tag(p, :define_scanned;   for_op="grid_radar_rhi")
    dk = field_with_tag(p, :define_detection; for_op="grid_radar_rhi")
    grid_radar_rhi(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.rmin, g.rincr, g.rdim, g.zmin, g.zincr, g.zdim,
        gd.power_threshold, mk, dk,
        p.grid.metadata)
    _maybe_add_echo_products(output_file, p)
end

"""
    grid_radar_ppi(radar_volume, output_file, index_time, p::DaishoParameters; heading=-9999.0)

Parameter-struct overload using the (xmin/xincr/xdim, ymin/yincr/ydim) fields
of `p.grid.ppi` plus `p.gridding` and `p.moments`. The Cartesian z-axis
fields are ignored. `p.grid.ppi` falls back to `p.grid.cartesian` when no
`[grid.ppi]` table is configured.
"""
function grid_radar_ppi(radar_volume::radar, output_file::AbstractString,
                         index_time, p::DaishoParameters; heading::Real=-9999.0)
    g = require_section(p, :grid, :ppi; for_op="grid_radar_ppi")
    gd = p.gridding
    mk = field_with_tag(p, :define_scanned;   for_op="grid_radar_ppi")
    dk = field_with_tag(p, :define_detection; for_op="grid_radar_ppi")
    grid_radar_ppi(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.xmin, g.xincr, g.xdim, g.ymin, g.yincr, g.ydim,
        gd.power_threshold, mk, dk, heading,
        p.grid.metadata)
    _maybe_add_echo_products(output_file, p)
end

"""
    grid_radar_composite(radar_volume, output_file, index_time, p::DaishoParameters; mean_heading=-9999.0)

Parameter-struct overload using the (xmin/xincr/xdim, ymin/yincr/ydim) fields
of `p.grid.composite` plus `p.gridding` and `p.moments`. `power_threshold` is
not applicable to composite gridding. `p.grid.composite` falls back to
`p.grid.cartesian` when no `[grid.composite]` table is configured.
"""
function grid_radar_composite(radar_volume::radar, output_file::AbstractString,
                               index_time, p::DaishoParameters; mean_heading::Real=-9999.0)
    g = require_section(p, :grid, :composite; for_op="grid_radar_composite")
    gd = p.gridding
    mk = field_with_tag(p, :define_scanned;   for_op="grid_radar_composite")
    dk = field_with_tag(p, :define_detection; for_op="grid_radar_composite")
    grid_radar_composite(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.xmin, g.xincr, g.xdim, g.ymin, g.yincr, g.ydim,
        mk, dk, mean_heading,
        p.grid.metadata)
end

"""
    grid_radar_column(radar_volume, output_file, index_time, p::DaishoParameters)

Parameter-struct overload using `p.grid.column.z*` as the column axis,
plus `p.gridding` and `p.moments`. `p.grid.column` falls back to
`p.grid.cartesian` when no `[grid.column]` table is configured.
"""
function grid_radar_column(radar_volume::radar, output_file::AbstractString,
                            index_time, p::DaishoParameters)
    g = require_section(p, :grid, :column; for_op="grid_radar_column")
    gd = p.gridding
    mk = field_with_tag(p, :define_scanned;   for_op="grid_radar_column")
    dk = field_with_tag(p, :define_detection; for_op="grid_radar_column")
    grid_radar_column(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.zmin, g.zincr, g.zdim,
        gd.power_threshold, mk, dk,
        p.grid.metadata)
end

# ── Volume-aware overloads (accumulator path) ────────────────────────────────
#
# These build a GridAccumulator, fold every sweep into it via `grid_sweep!`,
# call `finalize_grid`, and pass the result to the existing writer. They
# return the accumulator so callers can introspect or save it.

function grid_radar_volume(volume::Volume, output_file::AbstractString,
                            index_time, p::DaishoParameters; heading::Real=-9999.0)
    require_section(p, :grid, :volume; for_op="grid_radar_volume")
    spec = build_grid_spec(:volume_3d, volume, p)
    accum = GridAccumulator(spec, p)
    for i in eachindex(volume.sweeps)
        grid_sweep!(accum, volume, i, p; heading = heading)
    end
    radar_grid = finalize_grid(accum)
    latlon_grid = _compute_latlon_grid(spec)
    gridpoints = _gridpoints_volume_array(spec)
    write_gridded_radar_volume(output_file, index_time,
        volume.time_coverage_start, volume.time_coverage_end,
        gridpoints, radar_grid, latlon_grid, field_index_dict(p),
        spec.reference_latitude, spec.reference_longitude, Float64(heading),
        p.grid.metadata; fill_value=p.io.fill_value, undetect=p.io.undetect)
    _maybe_add_echo_products(output_file, p)
    return accum
end

function grid_radar_latlon_volume(volume::Volume, output_file::AbstractString,
                                   index_time, p::DaishoParameters; heading::Real=-9999.0)
    require_section(p, :grid, :latlon; for_op="grid_radar_latlon_volume")
    spec = build_grid_spec(:latlon_3d, volume, p)
    accum = GridAccumulator(spec, p)
    for i in eachindex(volume.sweeps)
        grid_sweep!(accum, volume, i, p; heading = heading)
    end
    radar_grid = finalize_grid(accum)
    latlon_grid = _compute_latlon_grid(spec)
    gridpoints = _gridpoints_latlon_array(spec)
    write_gridded_radar_volume(output_file, index_time,
        volume.time_coverage_start, volume.time_coverage_end,
        gridpoints, radar_grid, latlon_grid, field_index_dict(p),
        spec.reference_latitude, spec.reference_longitude, Float64(heading),
        p.grid.metadata; fill_value=p.io.fill_value, undetect=p.io.undetect)
    _maybe_add_echo_products(output_file, p)
    return accum
end

function grid_radar_rhi(volume::Volume, output_file::AbstractString,
                        index_time, p::DaishoParameters)
    require_section(p, :grid, :rhi; for_op="grid_radar_rhi")
    spec = build_grid_spec(:rhi_2d, volume, p)
    accum = GridAccumulator(spec, p)
    for i in eachindex(volume.sweeps)
        grid_sweep!(accum, volume, i, p)
    end
    radar_grid = finalize_grid(accum)
    latlon_grid = _compute_latlon_grid(spec)
    gridpoints = _gridpoints_rhi_array(spec)
    write_gridded_radar_rhi(output_file, index_time, _writer_radar_stub(volume),
        gridpoints, radar_grid, latlon_grid, field_index_dict(p),
        spec.reference_latitude, spec.reference_longitude, p.grid.metadata;
        fill_value=p.io.fill_value, undetect=p.io.undetect)
    _maybe_add_echo_products(output_file, p)
    return accum
end

function grid_radar_ppi(volume::Volume, output_file::AbstractString,
                        index_time, p::DaishoParameters; heading::Real=-9999.0)
    require_section(p, :grid, :ppi; for_op="grid_radar_ppi")
    spec = build_grid_spec(:ppi_2d, volume, p)
    accum = GridAccumulator(spec, p)
    for i in eachindex(volume.sweeps)
        grid_sweep!(accum, volume, i, p; heading = heading)
    end
    radar_grid = finalize_grid(accum)
    latlon_grid = _compute_latlon_grid(spec)
    gridpoints = _gridpoints_ppi_array(spec)
    # A PPI grid built from exactly one sweep can name its tilt; a multi-sweep PPI
    # spans several and must not claim one.
    fixed_angle = length(volume.sweeps) == 1 ? volume.sweeps[1].fixed_angle : nothing
    write_gridded_radar_ppi(output_file, index_time, _writer_radar_stub(volume),
        gridpoints, radar_grid, latlon_grid, field_index_dict(p),
        spec.reference_latitude, spec.reference_longitude, Float64(heading),
        p.grid.metadata; fill_value=p.io.fill_value, undetect=p.io.undetect,
        fixed_angle=fixed_angle)
    _maybe_add_echo_products(output_file, p)
    return accum
end

function grid_radar_composite(volume::Volume, output_file::AbstractString,
                              index_time, p::DaishoParameters; mean_heading::Real=-9999.0)
    require_section(p, :grid, :composite; for_op="grid_radar_composite")
    spec = build_grid_spec(:composite_2d, volume, p)
    accum = GridAccumulator(spec, p)
    for i in eachindex(volume.sweeps)
        grid_sweep!(accum, volume, i, p)
    end
    radar_grid = finalize_grid(accum)
    latlon_grid = _compute_latlon_grid(spec)
    gridpoints = _gridpoints_ppi_array(spec)
    write_gridded_radar_ppi(output_file, index_time, _writer_radar_stub(volume),
        gridpoints, radar_grid, latlon_grid, field_index_dict(p),
        spec.reference_latitude, spec.reference_longitude, Float64(mean_heading),
        p.grid.metadata; fill_value=p.io.fill_value, undetect=p.io.undetect)
    return accum
end

function grid_radar_column(volume::Volume, output_file::AbstractString,
                           index_time, p::DaishoParameters)
    require_section(p, :grid, :column; for_op="grid_radar_column")
    spec = build_grid_spec(:column_1d, volume, p)
    accum = GridAccumulator(spec, p)
    for i in eachindex(volume.sweeps)
        grid_sweep!(accum, volume, i, p)
    end
    radar_grid = finalize_grid(accum)
    latlon_grid = _compute_latlon_grid(spec)
    gridpoints = _gridpoints_column_array(spec)
    write_gridded_radar_column(output_file, index_time,
        volume.time_coverage_start, volume.time_coverage_end,
        gridpoints, radar_grid, latlon_grid, field_index_dict(p),
        spec.reference_latitude, spec.reference_longitude, p.grid.metadata;
        fill_value=p.io.fill_value, undetect=p.io.undetect)
    return accum
end

# ── Unified single-pass product writer ───────────────────────────────────────

# Write the finalized scalar grid of a `ScalarGridAccumulator` to a CF-1.12 3D
# NetCDF (the `write_gridded_radar_volume` layout), creating the file. Shared by
# both `write_grid_products` methods: the wind method then appends its wind
# variables onto the same file. Stage-1 supports the 3D shapes only.
function _write_scalar_grid_file(file::AbstractString, scalar::ScalarGridAccumulator,
                                 p::DaishoParameters, index_time::DateTime,
                                 start_time::DateTime, stop_time::DateTime)
    g = scalar.grid_spec
    if g.shape === :volume_3d
        gridpoints = _gridpoints_volume_array(g)
    elseif g.shape === :latlon_3d
        gridpoints = _gridpoints_latlon_array(g)
    else
        throw(ArgumentError("write_grid_products: the unified writer supports " *
            ":volume_3d and :latlon_3d only, got $(g.shape). Use " *
            "finalize_accumulator_file for the 2D/1D shapes."))
    end
    radar_grid  = finalize_grid(scalar)
    latlon_grid = _compute_latlon_grid(g)
    moment_dict = Dict{String,Int}(name => i for (i, name) in enumerate(scalar.fields))
    write_gridded_radar_volume(file, index_time, start_time, stop_time,
        gridpoints, radar_grid, latlon_grid, moment_dict,
        g.reference_latitude, g.reference_longitude, -9999.0, p.grid.metadata;
        fill_value = scalar.fill_value, undetect = scalar.undetect)
    return file
end

"""
    write_grid_products(file, acc::FieldAccumulator, p::DaishoParameters;
                        index_time, start_time=index_time, stop_time=index_time,
                        frame=CartesianFrame()) -> file

Write the gridded products of `acc` to **one** CF-1.12 NetCDF file, dispatched on
the accumulator type. The shared coordinates (`X`/`Y`/`Z`/`time`, projected and
geographic), the Transverse Mercator `grid_mapping`, and per-point
latitude/longitude are written once.

- A [`ScalarGridAccumulator`](@ref) writes every configured field
  ([`finalize_grid`](@ref)).
- A [`WindGridAccumulator`](@ref) writes the embedded scalar fields **and** the
  dual-Doppler product ([`finalize_wind`](@ref)): the two frame components, their
  `*STD` uncertainties, plus `DET`, `NGATES`, `QFLAG`. For the stage-1
  `CartesianFrame` these are `U, V, USTD, VSTD, DET, NGATES, QFLAG`. No wind
  variables are present when the accumulator is scalar.

Stage-1 supports the 3D shapes (`:volume_3d`, `:latlon_3d`); a pre-existing file
is deleted first. `frame` selects the wind output frame (ignored for scalar).
`mask_quality` (default `true`) writes the QC'd wind product — components and σ
blanked where `quality_flag != 0`, keeping only points that passed the σ
thresholds (the accumulator retains the full non-destructive field for stage 2);
`DET`/`NGATES`/`QFLAG` stay unmasked. It has no effect on a scalar accumulator.
"""
function write_grid_products(file::AbstractString, acc::ScalarGridAccumulator,
                             p::DaishoParameters; index_time::DateTime,
                             start_time::DateTime = index_time,
                             stop_time::DateTime = index_time,
                             frame::SynthesisFrame = CartesianFrame(),
                             mask_quality::Bool = true)
    _write_scalar_grid_file(file, acc, p, index_time, start_time, stop_time)
    return file
end

function write_grid_products(file::AbstractString, acc::WindGridAccumulator,
                             p::DaishoParameters; index_time::DateTime,
                             start_time::DateTime = index_time,
                             stop_time::DateTime = index_time,
                             frame::SynthesisFrame = CartesianFrame(),
                             mask_quality::Bool = true)
    # Scalar fields + shared coordinates first (creates the file)…
    _write_scalar_grid_file(file, acc.scalar, p, index_time, start_time, stop_time)
    # …then append the wind product onto the same grid (reusing the coords). With
    # mask_quality, only σ-passing points are written; the accumulator keeps the
    # full non-destructive field for stage 2.
    out = finalize_wind(acc, p; frame = frame)
    mask_quality && (out = _quality_masked(out))
    ds = NCDataset(file, "a")
    try
        _write_wind_data_vars!(ds, out)
    finally
        close(ds)
    end
    return file
end
