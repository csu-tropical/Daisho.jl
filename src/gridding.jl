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
    grid_radar_volume(radar_volume, moment_dict, grid_type_dict, output_file, index_time, xmin, xincr, xdim, ymin, yincr, ydim, zmin, zincr, zdim, beam_inflation, power_threshold, missing_key="SQI", valid_key="DBZ", heading=-9999.0)

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
- `beam_inflation`: Factor for inflating the radius of influence with distance from the radar. Set to 0.0 to disable.
- `power_threshold`: Minimum beam power weight below which a gate is excluded.
- `missing_key`: Moment name used to determine if a gate has valid signal (default `"SQI"`).
- `valid_key`: Moment name used for valid-data gating (default `"DBZ"`).
- `heading`: Mean heading of the platform in degrees (default `-9999.0` for missing).
"""
function grid_radar_volume(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
        xmin, xincr, xdim, ymin, yincr, ydim, zmin, zincr, zdim, beam_inflation, power_threshold,
        missing_key="SQI", valid_key="DBZ", heading=-9999.0)

    # Set the reference to the first location in the volume, but could be a parameter
    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    # Initialize the gridpoints
    # This array is slightly different than Springsteel spectral arrays, need to reconcile later
    gridpoints = initialize_regular_grid(xmin, xincr, xdim, ymin, yincr, ydim, zmin, zincr, zdim)

    h_roi = xincr * 0.75
    v_roi = zincr * 0.75

    radar_grid, latlon_grid = grid_volume(reference_latitude, reference_longitude, gridpoints,
        radar_volume, moment_dict, grid_type_dict, h_roi, v_roi, beam_inflation, power_threshold,
        missing_key, valid_key)

    write_gridded_radar_volume(output_file, index_time, radar_volume.time[1],
        radar_volume.time[end], gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude, heading)

end

"""
    grid_radar_latlon_volume(radar_volume, moment_dict, grid_type_dict, output_file, index_time, lonmin, londim, latmin, latdim, degincr, zmin, zincr, zdim, beam_inflation, power_threshold, missing_key="SQI", valid_key="DBZ", heading=-9999.0)

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
- `beam_inflation`: Factor for inflating the radius of influence with distance from the radar.
- `power_threshold`: Minimum beam power weight below which a gate is excluded.
- `missing_key`: Moment name used for signal quality gating (default `"SQI"`).
- `valid_key`: Moment name used for valid-data gating (default `"DBZ"`).
- `heading`: Mean heading of the platform in degrees (default `-9999.0` for missing).
"""
function grid_radar_latlon_volume(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
    lonmin, londim, latmin, latdim, degincr, zmin, zincr, zdim, beam_inflation, power_threshold,
    missing_key="SQI", valid_key="DBZ", heading=-9999.0)

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
        radar_volume, moment_dict, grid_type_dict, h_roi, v_roi, beam_inflation, power_threshold,
        missing_key, valid_key)

    write_gridded_radar_volume(output_file, index_time, radar_volume.time[1],
        radar_volume.time[end], gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude, heading)

end

"""
    grid_radar_rhi(radar_volume, moment_dict, grid_type_dict, output_file, index_time, rmin, rincr, rdim, rhi_zmin, rhi_zincr, rhi_zdim, beam_inflation, power_threshold, missing_key="SQI", valid_key="DBZ")

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
- `beam_inflation`: Factor for inflating the radius of influence with distance from the radar.
- `power_threshold`: Minimum beam power weight below which a gate is excluded.
- `missing_key::String`: Moment name used for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name used for valid-data gating (default `"DBZ"`).
"""
function grid_radar_rhi(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
        rmin, rincr, rdim, rhi_zmin, rhi_zincr, rhi_zdim, beam_inflation, power_threshold,
        missing_key::String="SQI", valid_key::String="DBZ")

    # Set the reference to the first location in the volume, but could be a parameter
    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    # Initialize the gridpoints
    # This array is slightly different than Springsteel spectral arrays, need to reconcile later
    gridpoints = initialize_regular_grid(rmin, rincr, rdim, rhi_zmin, rhi_zincr, rhi_zdim)

    h_roi = rincr * 0.75
    v_roi = rhi_zincr * 0.75

    radar_grid, latlon_grid = grid_rhi(reference_latitude, reference_longitude, gridpoints,
        radar_volume, moment_dict, grid_type_dict, h_roi, v_roi, beam_inflation, power_threshold, missing_key, valid_key)

    write_gridded_radar_rhi(output_file, index_time, radar_volume,
        gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude)

end

"""
    grid_radar_ppi(radar_volume, moment_dict, grid_type_dict, output_file, index_time, xmin, xincr, xdim, ymin, yincr, ydim, beam_inflation, power_threshold, missing_key="SQI", valid_key="DBZ", heading=-9999.0)

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
- `beam_inflation`: Factor for inflating the radius of influence with distance from the radar.
- `power_threshold`: Minimum beam power weight below which a gate is excluded.
- `missing_key`: Moment name used for signal quality gating (default `"SQI"`).
- `valid_key`: Moment name used for valid-data gating (default `"DBZ"`).
- `heading`: Mean heading of the platform in degrees (default `-9999.0` for missing).
"""
function grid_radar_ppi(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
        xmin, xincr, xdim, ymin, yincr, ydim, beam_inflation, power_threshold,
        missing_key="SQI", valid_key="DBZ", heading=-9999.0)

    # Set the reference to the first location in the volume, but could be a parameter
    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    # Initialize the gridpoints
    # This array is slightly different than Springsteel spectral arrays, need to reconcile later
    gridpoints = initialize_regular_grid(xmin, xincr, xdim, ymin, yincr, ydim)

    h_roi = xincr * 0.75

    radar_grid, latlon_grid = grid_ppi(reference_latitude, reference_longitude, gridpoints,
        radar_volume, moment_dict, grid_type_dict, h_roi, beam_inflation, power_threshold, missing_key, valid_key)

    write_gridded_radar_ppi(output_file, index_time, radar_volume,
        gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude, heading)

end

"""
    grid_radar_composite(radar_volume, moment_dict, grid_type_dict, output_file, index_time, xmin, xincr, xdim, ymin, yincr, ydim, beam_inflation, missing_key="SQI", valid_key="DBZ", mean_heading=-9999.0)

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
- `beam_inflation`: Factor for inflating the radius of influence with distance from the radar.
- `missing_key`: Moment name used for signal quality gating (default `"SQI"`).
- `valid_key`: Moment name used for valid-data gating (default `"DBZ"`).
- `mean_heading`: Mean heading of the platform in degrees (default `-9999.0` for missing).
"""
function grid_radar_composite(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
        xmin, xincr, xdim, ymin, yincr, ydim, beam_inflation,
        missing_key="SQI", valid_key="DBZ", mean_heading=-9999.0)

    # Set the reference to the first location in the volume, but could be a parameter
    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    # Initialize the gridpoints
    # This array is slightly different than Springsteel spectral arrays, need to reconcile later
    gridpoints = initialize_regular_grid(xmin, xincr, xdim, ymin, yincr, ydim)

    h_roi = xincr * 0.75

    radar_grid, latlon_grid = grid_composite(reference_latitude, reference_longitude, gridpoints,
        radar_volume, moment_dict, grid_type_dict, h_roi, beam_inflation, missing_key, valid_key)

    write_gridded_radar_ppi(output_file, index_time, radar_volume,
        gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude, mean_heading)

end

"""
    grid_radar_column(radar_volume, moment_dict, grid_type_dict, output_file, index_time, column_zmin, column_zincr, column_zdim, beam_inflation, power_threshold, missing_key="SQI", valid_key="DBZ")

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
- `beam_inflation`: Factor for inflating the radius of influence with distance from the radar.
- `power_threshold`: Minimum beam power weight below which a gate is excluded.
- `missing_key::String`: Moment name used for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name used for valid-data gating (default `"DBZ"`).
"""
function grid_radar_column(radar_volume, moment_dict, grid_type_dict, output_file, index_time,
    column_zmin, column_zincr, column_zdim, beam_inflation, power_threshold,
    missing_key::String="SQI", valid_key::String="DBZ")

    # Set the reference to the first location in the volume, but could be a parameter
    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    # Initialize the gridpoints
    # This array is slightly different than Springsteel spectral arrays, need to reconcile later
    gridpoints = initialize_regular_grid(column_zmin, column_zincr, column_zdim)

    v_roi = column_zincr * 0.75

    radar_grid, latlon_grid = grid_column(reference_latitude, reference_longitude, gridpoints,
        radar_volume, moment_dict, grid_type_dict, v_roi, beam_inflation, power_threshold, missing_key, valid_key)

    write_gridded_radar_column(output_file, index_time, radar_volume.time[1],
        radar_volume.time[end], gridpoints, radar_grid, latlon_grid, moment_dict,
        reference_latitude, reference_longitude)

end

"""
    grid_volume(reference_latitude, reference_longitude, gridpoints, radar_volume, moment_dict, grid_type_dict, horizontal_roi, vertical_roi, beam_inflation, power_threshold, missing_key="SQI", valid_key="DBZ") -> Tuple{Array{Float64}, Array{Float64}}

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
- `beam_inflation::Float64`: Factor to inflate the radius of influence with distance from the radar.
- `power_threshold::Float64`: Minimum beam power weight for a gate to contribute.
- `missing_key::String`: Moment name for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name for valid-data gating (default `"DBZ"`).

# Returns
A tuple `(radar_grid, latlon_grid)` where:
- `radar_grid`: A `(n_moments, zdim, ydim, xdim)` array of gridded moment values. Fill value is `-32768.0` (no data), `-9999.0` (in range but QC'd out).
- `latlon_grid`: A `(ydim, xdim, 2)` array of `[latitude, longitude]` at each horizontal grid point.
"""
function grid_volume(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, gridpoints::AbstractArray,
        radar_volume::radar, moment_dict::Dict, grid_type_dict::Dict, horizontal_roi::Float64, vertical_roi::Float64, beam_inflation::Float64,
        power_threshold::Float64, missing_key::String="SQI", valid_key::String="DBZ")

    # Convert the relevant radar information to arrays
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)
    grid_origin, radar_zyx, beams = radar_arrays(reference_latitude, reference_longitude, radar_volume, TM)

    # Create a balltree that has horizontal locations of every gate in Y, X dimension
    balltree = radar_balltree_yx(radar_volume, radar_zyx, beams)

    # Allocate the grid for the radar moments and the weights for each radar gate
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

        # Find the points within the radius of influence of the horizontal gridpoint
        eff_h_radius_influence = horizontal_roi
        eff_v_radius_influence = vertical_roi
        if beam_inflation > 0.0
            origin_dist = euclidean(yx_point, [0.0, 0.0])
            eff_h_radius_influence = max(beam_inflation * origin_dist, horizontal_roi)
            eff_v_radius_influence = max(beam_inflation * origin_dist, vertical_roi)
        end
        gates = inrange(balltree, yx_point, eff_h_radius_influence)

        if !isempty(gates)

            # Found some gates that are within range horizontally
            for j in 1:size(gridpoints,1)

                # First check to see whether at least one gate is in range vertically
                grid_z = gridpoints[j,i,1]
                min_dist, min_idx = findmin(z -> abs(z - grid_z), beams[gates,4])
                if min_dist > eff_v_radius_influence
                    # There is no gate in range of this grid box
                    continue
                else
                    if !ismissing(radar_volume.moments[gates[min_idx], moment_dict[missing_key]])
                        # There is at least one gate in range so set the flags to -9999
                        for m in 1:n_moments
                            radar_grid[m,j,i] = -9999.0
                        end
                    end
                end

                # Loops through the nearby gates with valid data
                valid_gates = collect(keys(skipmissing(radar_volume.moments[gates,moment_dict[valid_key]])))
                for gate in gates[valid_gates]

                    # Calculate the beam intercept to the gridpoint
                    dz = gridpoints[j,i,1] - radar_zyx[gate][1]
                    r = beams[gate,3]

                    # Calculate the effective elevation of the gridpoint taking into account
                    # earth curvature and standard refraction
                    # Sine of the height angle using equation from Doviak and Zrnic (1993)
                    sine_h = ((dz + Reff)^2 - r^2 - Reff^2) / (2*r*Reff)

                    gridpt_el = missing
                    if abs(sine_h) < 1.0
                        gridpt_el = asin(sine_h)
                    else
                        # Point is not accessible by the radar?
                        # I think this is failing because range exceeds height at origin
                        # A better check is needed, but skip for now
                        continue
                    end

                    # Calculate the effective azimuth
                    dx = yx_point[2] - radar_zyx[gate][3]
                    dy = yx_point[1] - radar_zyx[gate][2]
                    gridpt_az = (pi/2.0) - atan(dy, dx)
                    if gridpt_az < 0
                        gridpt_az += 2*pi
                    end

                    # Calculate the spherical angle difference using Haversine formula
                    angle_diff = spherical_angle([beams[gate,1], beams[gate,2]],
                        [gridpt_az, gridpt_el])

                    # Use the half-power beamwidth to define the beam
                    # 79.43 = ln(0.5) / (0.5 deg * pi / 180.0)
                    # Center of beam = 1.0 angle_weight
                    angle_weight = exp(-angle_diff * 79.43)
                    if angle_weight < power_threshold
                        angle_weight = 0
                    end

                    # Range weighting based on center of gridbox = 1.0 range_weight
                    gridpt_r = sin(sqrt(dx^2 + dy^2)/Reff) * (Reff + dz) / cos(gridpt_el)
                    range_weight = gridpt_r / r

		            if abs(gridpt_r - r) > horizontal_roi || abs(gridpt_r - r) > vertical_roi
                        # If the range is too far away from the grid center, set range_weight to 0
                        range_weight = 0.0
                    end

                    # Multiply weights so that center of beam is 1.0
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
    grid_rhi(reference_latitude, reference_longitude, gridpoints, radar_volume, moment_dict, grid_type_dict, horizontal_roi, vertical_roi, beam_inflation, power_threshold, missing_key="SQI", valid_key="DBZ") -> Tuple{Array{Float64}, Array{Float64}}

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
- `beam_inflation::Float64`: Factor to inflate the vertical radius of influence with distance.
- `power_threshold::Float64`: Minimum beam power weight for a gate to contribute.
- `missing_key::String`: Moment name for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name for valid-data gating (default `"DBZ"`).

# Returns
A tuple `(radar_grid, latlon_grid)` where:
- `radar_grid`: A `(n_moments, zdim, rdim)` array of gridded moment values.
- `latlon_grid`: An `(rdim, 2)` array of `[latitude, longitude]` along the RHI azimuth.
"""
function grid_rhi(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, gridpoints::AbstractArray,
        radar_volume::radar, moment_dict::Dict, grid_type_dict::Dict, horizontal_roi::Float64, vertical_roi::Float64, beam_inflation::Float64,
        power_threshold::Float64, missing_key::String="SQI", valid_key::String="DBZ")

    # Convert the relevant radar information to arrays
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)
    grid_origin, radar_zyx, beams = radar_arrays(reference_latitude, reference_longitude, radar_volume, TM)

    # Create a balltree that has horizontal locations of every gate in Z, X dimension
    balltree = radar_balltree_r(radar_volume, radar_zyx, beams)

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

        # Find the points within the radius of influence of the horizontal gridpoint
        eff_h_radius_influence = horizontal_roi
        eff_v_radius_influence = vertical_roi
        origin_dist = euclidean(r_point, [0.0])
        if beam_inflation > 0.0
            # No beam inflation in horizontal in this case since we are gridding in range
            eff_v_radius_influence = max(beam_inflation * origin_dist, vertical_roi)
        end

        gates = inrange(balltree, [ origin_dist ], eff_h_radius_influence)
        if !isempty(gates)

            # Found some gates that are within range horizontally
            for j in 1:size(gridpoints,1)

                # First check to see whether at least one gate is in range vertically
                grid_z = gridpoints[j,i,1]
                min_dist, min_idx = findmin(z -> abs(z - grid_z), beams[gates,4])
                if min_dist > eff_v_radius_influence
                    # There is no gate in range of this grid box
                    continue
                else
                    if !ismissing(radar_volume.moments[gates[min_idx], moment_dict[missing_key]])
                        # There is at least one gate in range so set the flags to -9999
                        for m in 1:n_moments
                            radar_grid[m,j,i] = -9999.0
                        end
                    end
                end

                # Loops through the nearby gates with valid data
                valid_gates = collect(keys(skipmissing(radar_volume.moments[gates,moment_dict[valid_key]])))
                for gate in gates[valid_gates]

                    # Calculate the beam intercept to the gridpoint
                    dz = gridpoints[j,i,1] - radar_zyx[gate][1]
                    r = beams[gate,3]

                    # Calculate the effective elevation of the gridpoint taking into account
                    # earth curvature and standard refraction
                    # Sine of the height angle using full equation
                    sine_h = ((dz + Reff)^2 - r^2 - Reff^2) / (2*r*Reff)

                    # Use approximation from Gao et al. 2006
                    #sine_h = (z/r) - (r/(2*Reff))

                    gridpt_el = missing
                    if abs(sine_h) < 1.0
                        gridpt_el = asin(sine_h)
                    else
                        # Point is not accessible by the radar?
                        # I think this is failing because range exceeds height at origin
                        # A better check is needed, but skip for now
                        continue
                    end

                    # Don't need azimuth, just dx and dy to get the surface range
                    dx = x_point - radar_zyx[gate][3]
                    dy = y_point - radar_zyx[gate][2]

                    # Calculate the spherical angle difference using Haversine formula
                    angle_diff = spherical_angle([beams[gate,1], beams[gate,2]],
                        [beams[gate,1], gridpt_el])

                    # Use the half-power beamwidth to define the beam
                    # 79.43 = ln(0.5) / (0.5 deg * pi / 180.0)
                    # Center of beam = 1.0 angle_weight
                    angle_weight = exp(-angle_diff * 79.43)
                    if angle_weight < power_threshold
                        angle_weight = 0
                    end

                    # Range weighting based on center of gridbox = 1.0 range_weight
                    gridpt_r = sin(sqrt(dx^2 + dy^2)/Reff) * (Reff + dz) / cos(gridpt_el)
                    range_weight = gridpt_r / r

		            if abs(gridpt_r - r) > horizontal_roi || abs(gridpt_r - r) > vertical_roi
                        # If the range is too far away from the grid center, set range_weight to 0
                        range_weight = 0.0
                    end

                    # Multiply weights so that center of beam is 1.0
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
    grid_ppi(reference_latitude, reference_longitude, gridpoints, radar_volume, moment_dict, grid_type_dict, horizontal_roi, beam_inflation, power_threshold, missing_key="SQI", valid_key="DBZ") -> Tuple{Array{Float64}, Array{Float64}}

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
- `beam_inflation::Float64`: Factor to inflate the radius of influence with distance from the radar.
- `power_threshold::Float64`: Minimum beam power weight for a gate to contribute.
- `missing_key::String`: Moment name for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name for valid-data gating (default `"DBZ"`).

# Returns
A tuple `(radar_grid, latlon_grid)` where:
- `radar_grid`: A `(n_moments, ydim, xdim)` array of gridded moment values.
- `latlon_grid`: A `(ydim, xdim, 2)` array of `[latitude, longitude]` at each grid point.
"""
function grid_ppi(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, gridpoints::AbstractArray,
        radar_volume::radar, moment_dict::Dict, grid_type_dict::Dict, horizontal_roi::Float64, beam_inflation::Float64,
        power_threshold::Float64, missing_key::String="SQI", valid_key::String="DBZ")

    # Convert the relevant radar information to arrays
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)
    grid_origin, radar_zyx, beams = radar_arrays(reference_latitude, reference_longitude, radar_volume, TM)

    # Create a balltree that has horizontal locations of every gate in Y, X dimension
    balltree = radar_balltree_yx(radar_volume, radar_zyx, beams)

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

        # Find the points within the radius of influence of the horizontal gridpoint
        eff_h_radius_influence = horizontal_roi
        if beam_inflation > 0.0
            origin_dist = euclidean(yx_point, [0.0, 0.0])
            eff_h_radius_influence = max(beam_inflation * origin_dist, horizontal_roi)
        end
        gates = inrange(balltree, yx_point, eff_h_radius_influence)

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
            valid_gates = collect(keys(skipmissing(radar_volume.moments[gates,moment_dict[valid_key]])))
            for gate in gates[valid_gates]

                # Calculate the effective azimuth
                dx = yx_point[2] - radar_zyx[gate][3]
                dy = yx_point[1] - radar_zyx[gate][2]
                gridpt_az = (pi/2.0) - atan(dy, dx)
                if gridpt_az < 0
                    gridpt_az += 2*pi
                end

                # Calculate the spherical angle difference using Haversine formula
                angle_diff = spherical_angle([beams[gate,1], beams[gate,2]],
                    [gridpt_az, beams[gate,2]])

                # Use the half-power beamwidth to define the beam
                # 79.43 = ln(0.5) / (0.5 deg * pi / 180.0)
                # Center of beam = 1.0 angle_weight
                angle_weight = exp(-angle_diff * 79.43)
                if angle_weight < power_threshold
                    angle_weight = 0
                end

                # Range weighting based on center of gridbox = 1.0 range_weight
                r = beams[gate,3]
                gridpt_r = sqrt(dx^2 + dy^2)
                range_weight = gridpt_r / r

                if abs(gridpt_r - r) > horizontal_roi
                    # If the range is too far away from the grid center, set range_weight to 0
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
    grid_composite(reference_latitude, reference_longitude, gridpoints, radar_volume, moment_dict, grid_type_dict, horizontal_roi, beam_inflation, missing_key="SQI", valid_key="DBZ") -> Tuple{Array{Float64}, Array{Float64}}

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
- `beam_inflation::Float64`: Factor to inflate the radius of influence with distance from the radar.
- `missing_key::String`: Moment name for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name used to find the maximum value gate (default `"DBZ"`).

# Returns
A tuple `(radar_grid, latlon_grid)` where:
- `radar_grid`: A `(n_moments, ydim, xdim)` array of composite moment values.
- `latlon_grid`: A `(ydim, xdim, 2)` array of `[latitude, longitude]` at each grid point.
"""
function grid_composite(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, gridpoints::AbstractArray,
        radar_volume::radar, moment_dict::Dict, grid_type_dict::Dict, horizontal_roi::Float64, beam_inflation::Float64,
        missing_key::String="SQI", valid_key::String="DBZ")

    # Convert the relevant radar information to arrays
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)
    grid_origin, radar_zyx, beams = radar_arrays(reference_latitude, reference_longitude, radar_volume, TM)

    # Create a balltree that has horizontal locations of every gate in Y, X dimension
    balltree = radar_balltree_yx(radar_volume, radar_zyx, beams)

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

        # Find the points within the radius of influence of the horizontal gridpoint
        eff_h_radius_influence = horizontal_roi
        if beam_inflation > 0.0
            origin_dist = euclidean(yx_point, [0.0, 0.0])
            eff_h_radius_influence = max(beam_inflation * origin_dist, horizontal_roi) #0.01745 = 1 deg beam
        end
        gates = inrange(balltree, yx_point, eff_h_radius_influence)

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
            valid_gates = collect(keys(skipmissing(radar_volume.moments[gates,moment_dict[valid_key]])))
            if !isempty(valid_gates)
                dbzmax, max_idx = findmax(gate -> radar_volume.moments[gate,moment_dict[valid_key]], gates[valid_gates])

                # Divide by the total weight for that gridbox
                for m in 1:n_moments
                    if !ismissing(radar_volume.moments[gates[valid_gates][max_idx],m])
                        radar_grid[m,i] = radar_volume.moments[gates[valid_gates][max_idx],m]
                    end
                end # End of moment loop
            end # End of valid gates test
        end # End of gate empty test
    end # End of horizontal loop

    # Return the gridded radar array
    return radar_grid, latlon_grid
end

"""
    grid_column(reference_latitude, reference_longitude, gridpoints, radar_volume, moment_dict, grid_type_dict, vertical_roi, beam_inflation, power_threshold, missing_key="SQI", valid_key="DBZ") -> Tuple{Array{Float64}, Array{Float64}}

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
- `beam_inflation::Float64`: Factor to inflate the vertical radius of influence with altitude.
- `power_threshold::Float64`: Minimum beam power weight for a gate to contribute.
- `missing_key::String`: Moment name for signal quality gating (default `"SQI"`).
- `valid_key::String`: Moment name for valid-data gating (default `"DBZ"`).

# Returns
A tuple `(radar_grid, latlon_grid)` where:
- `radar_grid`: A `(n_moments, zdim)` array of gridded moment values.
- `latlon_grid`: A 2-element array `[latitude, longitude]` of the column location.
"""
function grid_column(reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, gridpoints::AbstractArray,
    radar_volume::radar, moment_dict::Dict, grid_type_dict::Dict, vertical_roi::Float64, beam_inflation::Float64,
    power_threshold::Float64, missing_key::String="SQI", valid_key::String="DBZ")

    # Convert the relevant radar information to arrays
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)
    grid_origin, radar_zyx, beams = radar_arrays(reference_latitude, reference_longitude, radar_volume, TM)

    # Create a balltree that has horizontal locations of every gate in Y, X dimension
    balltree = radar_balltree_yx(radar_volume, radar_zyx, beams)

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

        # Find the points within the radius of influence of the vertical gridpoint
        grid_z = gridpoints[i]
        eff_v_radius_influence = vertical_roi
        origin_dist = euclidean(grid_z, [0.0])
        if beam_inflation > 0.0
            # No beam inflation in horizontal in this case since we are gridding in Z
            eff_v_radius_influence = max(beam_inflation * origin_dist, vertical_roi)
        end

        # First check to see whether at least one gate is in range vertically
        min_dist, min_idx = findmin(z -> abs(z - grid_z), beams[:,4])
        if min_dist > eff_v_radius_influence
            # There is no gate in range of this grid box
            continue
        else
            if !ismissing(radar_volume.moments[min_idx, moment_dict[missing_key]])
                # There is at least one gate in range so set the flags to -9999
                for m in 1:n_moments
                    radar_grid[m,i] = -9999.0
                end
            end
        end

        # Loops through the nearby gates with valid data
        valid_gates = collect(keys(skipmissing(radar_volume.moments[:,moment_dict[valid_key]])))
        for gate in valid_gates

            # Calculate the beam intercept to the gridpoint
            dz = gridpoints[i] - radar_zyx[gate][1]
            r = beams[gate,3]

            # Calculate the effective elevation of the gridpoint taking into account
            # earth curvature and standard refraction
            # Sine of the height angle using full equation
            sine_h = ((dz + Reff)^2 - r^2 - Reff^2) / (2*r*Reff)

            # Use approximation from Gao et al. 2006
            #sine_h = (z/r) - (r/(2*Reff))

            gridpt_el = missing
            if abs(sine_h) < 1.0
                gridpt_el = asin(sine_h)
            else
                # Point is not accessible by the radar?
                # I think this is failing because range exceeds height at origin
                # A better check is needed, but skip for now
                continue
            end

            # Calculate the spherical angle difference using Haversine formula
            angle_diff = spherical_angle([beams[gate,1], beams[gate,2]],
                [beams[gate,1], gridpt_el])

            # Use the half-power beamwidth to define the beam
            # 79.43 = ln(0.5) / (0.5 deg * pi / 180.0)
            # Center of beam = 1.0 angle_weight
            angle_weight = exp(-angle_diff * 79.43)
            if angle_weight < power_threshold
                angle_weight = 0
            end

            # Range weighting based on center of gridbox = 1.0 range_weight
            gridpt_r = grid_z / cos(beams[gate,2])
            range_weight = gridpt_r / r
            #range_weight = 1.0 - abs(gridpt_r - r) / 200.0
            #if range_weight < 0.0
            #    range_weight = 0.0
            #end

	    if abs(gridpt_r - r) > vertical_roi
             	# If the range is too far away from the grid center, set range_weight to 0
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

"""
    write_gridded_radar_volume(file, index_time, start_time, stop_time, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude, reference_longitude, mean_heading)

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
"""
function write_gridded_radar_volume(file, index_time, start_time, stop_time, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, mean_heading::AbstractFloat)

    # Delete any pre-existing file
    rm(file, force=true)

    ds = NCDataset(file,"c", attrib = OrderedDict(
        "Conventions"               => "CF-1.12",
        "history"                   => "v1.0",
        "institution"               => "Colorado State University",
        "source"                    => "CSU SEA-POL radar",
        "instrument"                => "SEA-POL",
        "title"                     => "Level 4 Gridded SEA-POL Radar Data",
        "summary"                   => "Level 4 Gridded SEA-POL Radar Data",
        "creator_name"              => "Michael M. Bell",
        "creator_email"             => "mmbell@colostate.edu",
        "creator_id"                => "https://orcid.org/0000-0002-0496-331X",
        "project"                   => "PICCOLO, BOWTIE, ORCESTRA",
        "platform"                  => "RV METEOR",
        #"references"                => "Comma-separated list of URL/DOI to extended information",
        "keywords"                  => "radar, precipitation, sea-pol",
        "processing_level"          => "Level 4",
        "license"                   => "CC-BY-4.0",
    ))

    # Dimensions
    # Could concatenate multiple volumes here
    #numswps = length(swpstart)
    xdim = size(radar_grid,4)
    ydim = size(radar_grid,3)
    zdim = size(radar_grid,2)
    ds.dim["time"] = 1
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
    nctime[:] = datetime2unix.(index_time)
    ncstarttime[:] = datetime2unix.(start_time)
    ncstoptime[:] = datetime2unix.(stop_time)
    ncz[:] = gridpoints[:,1,1,1]
    ncy[:] = gridpoints[1,:,1,2]
    ncx[:] = gridpoints[1,1,:,3]
    nclat[:] = latlon_grid[:,:,1]'
    nclon[:] = latlon_grid[:,:,2]'
    ncgrid_mapping[:] = -32768.0
    ncheading[:] = mean_heading

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
        ncvar = defVar(ds, key, Float32, ("X", "Y", "Z", "time"), attrib = var_attrib)
        ncvar[:] = ncgrid[moment_dict[key],:,:,:]
    end

    close(ds)
end

"""
    write_gridded_radar_rhi(file, index_time, radar_volume, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude, reference_longitude)

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
"""
function write_gridded_radar_rhi(file, index_time, radar_volume, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude::AbstractFloat, reference_longitude::AbstractFloat)

    # Delete any pre-existing file
    rm(file, force=true)

    start_time = radar_volume.time[1]
    stop_time = radar_volume.time[end]
    azimuth = radar_volume.azimuth[1]

    ds = NCDataset(file,"c", attrib = OrderedDict(
        "Conventions"               => "CF-1.12",
        "history"                   => "v1.0",
        "institution"               => "Colorado State University",
        "source"                    => "CSU SEA-POL radar",
        "instrument"                => "SEA-POL",
        "title"                     => "Level 4 Gridded SEA-POL Radar Data",
        "summary"                   => "Level 4 Gridded SEA-POL Radar Data",
        "creator_name"              => "Michael M. Bell",
        "creator_email"             => "mmbell@colostate.edu",
        "creator_id"                => "https://orcid.org/0000-0002-0496-331X",
        "project"                   => "PICCOLO, BOWTIE, ORCESTRA",
        "platform"                  => "RV METEOR",
        #"references"                => "Comma-separated list of URL/DOI to extended information",
        "keywords"                  => "radar, precipitation, sea-pol",
        "processing_level"          => "Level 4",
        "license"                   => "CC-BY-4.0",
    ))

    # Dimensions
    # Could concatenate multiple volumes here
    #numswps = length(swpstart)
    rdim = size(radar_grid,3)
    zdim = size(radar_grid,2)
    ds.dim["time"] = 1
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
    nctime[:] = datetime2unix.(index_time)
    ncstarttime[:] = datetime2unix.(start_time)
    ncstoptime[:] = datetime2unix.(stop_time)
    ncazimuth[:] = azimuth
    ncz[:] = gridpoints[:,1,1]
    ncr[:] = gridpoints[1,:,2]
    nclat[:] = latlon_grid[:,1]'
    nclon[:] = latlon_grid[:,2]'
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
        ncvar = defVar(ds, key, Float32, ("R", "Z", "time"), attrib = var_attrib)
        ncvar[:] = ncgrid[moment_dict[key],:,:]
    end

    close(ds)
end

"""
    write_gridded_radar_ppi(file, index_time, radar_volume, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude, reference_longitude, mean_heading)

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
"""
function write_gridded_radar_ppi(file, index_time, radar_volume, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude::AbstractFloat, reference_longitude::AbstractFloat, mean_heading::AbstractFloat)

    # Delete any pre-existing file
    rm(file, force=true)

    start_time = radar_volume.time[1]
    stop_time = radar_volume.time[end]
    scan_name = radar_volume.scan_name

    ds = NCDataset(file,"c", attrib = OrderedDict(
        "Conventions"               => "CF-1.12",
        "history"                   => "v1.0",
        "institution"               => "Colorado State University",
        "source"                    => "CSU SEA-POL radar",
        "instrument"                => "SEA-POL",
        "title"                     => "Level 4 Gridded SEA-POL Radar Data",
        "summary"                   => "Level 4 Gridded SEA-POL Radar Data",
        "creator_name"              => "Michael M. Bell",
        "creator_email"             => "mmbell@colostate.edu",
        "creator_id"                => "https://orcid.org/0000-0002-0496-331X",
        "project"                   => "PICCOLO, BOWTIE, ORCESTRA",
        "platform"                  => "RV METEOR",
        #"references"                => "Comma-separated list of URL/DOI to extended information",
        "keywords"                  => "radar, precipitation, sea-pol",
        "processing_level"          => "Level 4",
        "license"                   => "CC-BY-4.0",
	"scan_name"                 => scan_name
    ))

    # Dimensions
    # Could concatenate multiple volumes here
    #numswps = length(swpstart)
    xdim = size(radar_grid,3)
    ydim = size(radar_grid,2)
    ds.dim["time"] = 1
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

    # Using start time for now, but eventually need to use some reference time
    nctime[:] = datetime2unix.(index_time)
    ncstarttime[:] = datetime2unix.(start_time)
    ncstoptime[:] = datetime2unix.(stop_time)
    ncy[:] = gridpoints[:,1,1]
    ncx[:] = gridpoints[1,:,2]
    nclat[:] = latlon_grid[:,:,1]'
    nclon[:] = latlon_grid[:,:,2]'
    ncgrid_mapping[:] = -32768.0
    ncheading[:] = mean_heading

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
        ncvar = defVar(ds, key, Float32, ("X", "Y", "time"), attrib = var_attrib)
        ncvar[:] = ncgrid[moment_dict[key],:,:]
    end

    close(ds)
end

"""
    write_gridded_radar_column(file, index_time, start_time, stop_time, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude, reference_longitude)

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
"""
function write_gridded_radar_column(file, index_time, start_time, stop_time, gridpoints, radar_grid, latlon_grid, moment_dict, reference_latitude::AbstractFloat, reference_longitude::AbstractFloat)

    # Delete any pre-existing file
    rm(file, force=true)

    ds = NCDataset(file,"c", attrib = OrderedDict(
        "Conventions"               => "CF-1.12",
        "history"                   => "v1.0",
        "institution"               => "Colorado State University",
        "source"                    => "CSU SEA-POL radar",
        "instrument"                => "SEA-POL",
        "title"                     => "Level 4 Gridded SEA-POL Radar Data",
        "summary"                   => "Level 4 Gridded SEA-POL Radar Data",
        "creator_name"              => "Michael M. Bell",
        "creator_email"             => "mmbell@colostate.edu",
        "creator_id"                => "https://orcid.org/0000-0002-0496-331X",
        "project"                   => "PICCOLO, BOWTIE, ORCESTRA",
        "platform"                  => "RV METEOR",
        #"references"                => "Comma-separated list of URL/DOI to extended information",
        "keywords"                  => "radar, precipitation, sea-pol",
        "processing_level"          => "Level 4",
        "license"                   => "CC-BY-4.0",
    ))

    # Dimensions
    # Could concatenate multiple volumes here
    #numswps = length(swpstart)
    zdim = size(radar_grid,2)
    ds.dim["time"] = 1 # To make this unlimited use 'Inf' here, but then all the time related variables are missing for some reason
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
    nctime[:] = datetime2unix.(index_time)
    ncstarttime[:] = datetime2unix.(start_time)
    ncstoptime[:] = datetime2unix.(stop_time)
    ncz[:] = gridpoints[:]
    nclat[:] = latlon_grid[1]'
    nclon[:] = latlon_grid[2]'
    ncgrid_mapping[:] = -32768.0

    # Loop through the moments
    for key in keys(moment_dict)
        var_attrib = common_attrib
        if haskey(variable_attrib_dict,key)
            var_attrib = merge(common_attrib, variable_attrib_dict[key])
        else
            var_attrib = merge(common_attrib, variable_attrib_dict["UNKNOWN"])
        end
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

"""
    read_gridded_radar(file, moment_dict) -> Tuple

Read a gridded 3D radar volume from a NetCDF file with X, Y, Z coordinates.

Opens the specified NetCDF file and reads the `X`, `Y`, `Z` coordinate arrays, `latitude`/`longitude`
fields, time bounds, and all radar moment data specified in `moment_dict`.

# Arguments
- `file`: Path to the input NetCDF file.
- `moment_dict`: Dictionary mapping moment names (e.g., `"DBZ"`) to integer indices.

# Returns
A tuple `(x, y, z, lat, lon, start_time, stop_time, radardata)` where:
- `x`, `y`, `z`: Coordinate arrays from the file (meters).
- `lat`, `lon`: 2D latitude and longitude arrays.
- `start_time`, `stop_time`: Time bounds of the data.
- `radardata`: A `(n_moments, n_points)` array of `Union{Missing, Float32}` moment values.
"""
function read_gridded_radar(file, moment_dict)

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
    read_gridded_ppi(file, moment_dict) -> Tuple

Read a gridded 2D PPI radar scan from a NetCDF file with X, Y coordinates.

Opens the specified NetCDF file and reads the `X`, `Y` coordinate arrays, `latitude`/`longitude`
fields, time bounds, and all radar moment data specified in `moment_dict`.

# Arguments
- `file`: Path to the input NetCDF file.
- `moment_dict`: Dictionary mapping moment names (e.g., `"DBZ"`) to integer indices.

# Returns
A tuple `(x, y, lat, lon, start_time, stop_time, radardata)` where:
- `x`, `y`: Coordinate arrays from the file (meters).
- `lat`, `lon`: 2D latitude and longitude arrays.
- `start_time`, `stop_time`: Time bounds of the data.
- `radardata`: A `(n_moments, n_points)` array of `Union{Missing, Float32}` moment values.
"""
function read_gridded_ppi(file, moment_dict)

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
    read_gridded_rhi(file, moment_dict) -> Tuple

Read a gridded 2D RHI radar scan from a NetCDF file with R (range) and Z (altitude) coordinates.

Opens the specified NetCDF file and reads the `R`, `Z` coordinate arrays, `latitude`/`longitude`
fields along the RHI azimuth, time bounds, and all radar moment data specified in `moment_dict`.

# Arguments
- `file`: Path to the input NetCDF file.
- `moment_dict`: Dictionary mapping moment names (e.g., `"DBZ"`) to integer indices.

# Returns
A tuple `(R, Z, lat, lon, start_time, stop_time, radardata)` where:
- `R`: Range coordinate array (meters).
- `Z`: Altitude coordinate array (meters).
- `lat`, `lon`: Latitude and longitude arrays along the RHI azimuth.
- `start_time`, `stop_time`: Time bounds of the data.
- `radardata`: A `(n_moments, n_points)` array of `Union{Missing, Float32}` moment values.
"""
function read_gridded_rhi(file, moment_dict)

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

# ── Parameter-struct overloads ──────────────────────────────────────────────
# Convenience methods that pull configuration out of a `DaishoParameters` and
# delegate to the long-positional driver above. Adding a new knob means
# extending the struct + `defaults.toml` and updating the delegating call —
# callers using `p::DaishoParameters` do not change.

"""
    grid_radar_volume(radar_volume, output_file, index_time, p::DaishoParameters; heading=-9999.0)

Parameter-struct overload reading from `p.grid.cartesian`, `p.gridding`, and
`p.moments`.
"""
function grid_radar_volume(radar_volume::radar, output_file::AbstractString,
                            index_time, p::DaishoParameters; heading::Real=-9999.0)
    g, gd = p.grid.cartesian, p.gridding
    grid_radar_volume(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.xmin, g.xincr, g.xdim, g.ymin, g.yincr, g.ydim, g.zmin, g.zincr, g.zdim,
        gd.beam_inflation, gd.power_threshold, gd.missing_key, gd.valid_key, heading)
end

"""
    grid_radar_latlon_volume(radar_volume, output_file, index_time, p::DaishoParameters; heading=-9999.0)

Parameter-struct overload reading from `p.grid.latlon`, `p.gridding`, and
`p.moments`.
"""
function grid_radar_latlon_volume(radar_volume::radar, output_file::AbstractString,
                                   index_time, p::DaishoParameters; heading::Real=-9999.0)
    g, gd = p.grid.latlon, p.gridding
    grid_radar_latlon_volume(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.lonmin, g.londim, g.latmin, g.latdim, g.degincr,
        g.zmin, g.zincr, g.zdim,
        gd.beam_inflation, gd.power_threshold, gd.missing_key, gd.valid_key, heading)
end

"""
    grid_radar_rhi(radar_volume, output_file, index_time, p::DaishoParameters)

Parameter-struct overload reading from `p.grid.rhi`, `p.gridding`, and `p.moments`.
"""
function grid_radar_rhi(radar_volume::radar, output_file::AbstractString,
                         index_time, p::DaishoParameters)
    g, gd = p.grid.rhi, p.gridding
    grid_radar_rhi(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.rmin, g.rincr, g.rdim, g.zmin, g.zincr, g.zdim,
        gd.beam_inflation, gd.power_threshold, gd.missing_key, gd.valid_key)
end

"""
    grid_radar_ppi(radar_volume, output_file, index_time, p::DaishoParameters; heading=-9999.0)

Parameter-struct overload using the (xmin/xincr/xdim, ymin/yincr/ydim) fields
of `p.grid.cartesian` plus `p.gridding` and `p.moments`. The Cartesian z-axis
fields are ignored.
"""
function grid_radar_ppi(radar_volume::radar, output_file::AbstractString,
                         index_time, p::DaishoParameters; heading::Real=-9999.0)
    g, gd = p.grid.cartesian, p.gridding
    grid_radar_ppi(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.xmin, g.xincr, g.xdim, g.ymin, g.yincr, g.ydim,
        gd.beam_inflation, gd.power_threshold, gd.missing_key, gd.valid_key, heading)
end

"""
    grid_radar_composite(radar_volume, output_file, index_time, p::DaishoParameters; mean_heading=-9999.0)

Parameter-struct overload using the (xmin/xincr/xdim, ymin/yincr/ydim) fields
of `p.grid.cartesian` plus `p.gridding` and `p.moments`. `power_threshold` is
not applicable to composite gridding.
"""
function grid_radar_composite(radar_volume::radar, output_file::AbstractString,
                               index_time, p::DaishoParameters; mean_heading::Real=-9999.0)
    g, gd = p.grid.cartesian, p.gridding
    grid_radar_composite(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.xmin, g.xincr, g.xdim, g.ymin, g.yincr, g.ydim,
        gd.beam_inflation, gd.missing_key, gd.valid_key, mean_heading)
end

"""
    grid_radar_column(radar_volume, output_file, index_time, p::DaishoParameters)

Parameter-struct overload using `p.grid.cartesian.z*` as the column axis,
plus `p.gridding` and `p.moments`.
"""
function grid_radar_column(radar_volume::radar, output_file::AbstractString,
                            index_time, p::DaishoParameters)
    g, gd = p.grid.cartesian, p.gridding
    grid_radar_column(radar_volume, field_index_dict(p), grid_type_index_dict(p),
        output_file, index_time,
        g.zmin, g.zincr, g.zdim,
        gd.beam_inflation, gd.power_threshold, gd.missing_key, gd.valid_key)
end

# ── Volume-aware overloads (Phase C bridge) ──────────────────────────────────
#
# These delegate to the legacy `radar` drivers via `as_legacy_radar`. The
# bridge layer is removed in Phase D once the legacy drivers are gone.

function grid_radar_volume(volume::Volume, output_file::AbstractString,
                            index_time, p::DaishoParameters; heading::Real=-9999.0)
    legacy, _names = as_legacy_radar(volume)
    grid_radar_volume(legacy, output_file, index_time, p; heading=heading)
end

function grid_radar_latlon_volume(volume::Volume, output_file::AbstractString,
                                   index_time, p::DaishoParameters; heading::Real=-9999.0)
    legacy, _names = as_legacy_radar(volume)
    grid_radar_latlon_volume(legacy, output_file, index_time, p; heading=heading)
end

function grid_radar_rhi(volume::Volume, output_file::AbstractString,
                        index_time, p::DaishoParameters)
    legacy, _names = as_legacy_radar(volume)
    grid_radar_rhi(legacy, output_file, index_time, p)
end

function grid_radar_ppi(volume::Volume, output_file::AbstractString,
                        index_time, p::DaishoParameters; heading::Real=-9999.0)
    legacy, _names = as_legacy_radar(volume)
    grid_radar_ppi(legacy, output_file, index_time, p; heading=heading)
end

function grid_radar_composite(volume::Volume, output_file::AbstractString,
                              index_time, p::DaishoParameters; mean_heading::Real=-9999.0)
    legacy, _names = as_legacy_radar(volume)
    grid_radar_composite(legacy, output_file, index_time, p; mean_heading=mean_heading)
end

function grid_radar_column(volume::Volume, output_file::AbstractString,
                           index_time, p::DaishoParameters)
    legacy, _names = as_legacy_radar(volume)
    grid_radar_column(legacy, output_file, index_time, p)
end
