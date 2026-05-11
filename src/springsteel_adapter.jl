# Springsteel.jl integration for spectral gridding in Daisho.jl
#
# This module bridges Daisho's radar gridding engine with Springsteel's
# spectral analysis framework, enabling B-spline transforms, analytic
# derivatives, and CF-compliant NetCDF output on spectral grids.

"""
    daisho_cells_from_gridpoints(n_gridpoints, mubar=3)

Compute the number of Springsteel B-spline cells needed to produce at least
`n_gridpoints` physical grid points, given `mubar` quadrature points per cell.

# Arguments
- `n_gridpoints`: Desired minimum number of physical grid points.
- `mubar`: Quadrature points per cell (default: 3).

# Returns
Number of B-spline cells (integer).

See also: [`create_radar_grid`](@ref)
"""
function daisho_cells_from_gridpoints(n_gridpoints::Integer, mubar::Integer=3)
    return max(1, Int(ceil(n_gridpoints / mubar)))
end

"""
    create_radar_grid(geometry, moment_dict;
        xmin=0.0, xmax=0.0, xdim=0,
        ymin=0.0, ymax=0.0, ydim=0,
        zmin=0.0, zmax=0.0, zdim=0,
        mubar=3, quadrature=:gauss,
        BCL=NaturalBC(), BCR=NaturalBC(),
        BCU=NaturalBC(), BCD=NaturalBC(),
        BCB=NaturalBC(), BCT=NaturalBC())

Create a Springsteel spectral grid configured for radar data analysis.

Maps Daisho's radar moment dictionary and grid specification to a
`SpringsteelGrid`. The `xdim`, `ydim`, `zdim` parameters specify the
number of B-spline cells in each dimension; the actual number of physical
grid points is `cells × mubar`.

Dimension mapping: i = X (easting), j = Y (northing), k = Z (altitude).

# Arguments
- `geometry::String`: Grid geometry type (`"R"`, `"RR"`, or `"RRR"`).
- `moment_dict::Dict`: Dictionary mapping moment names to indices (e.g., `Dict("DBZ" => 1, "VEL" => 2)`).
- `xmin`, `xmax`: X-dimension domain bounds (meters).
- `xdim`: Number of B-spline cells in X.
- `ymin`, `ymax`: Y-dimension domain bounds (meters, for RR/RRR).
- `ydim`: Number of B-spline cells in Y.
- `zmin`, `zmax`: Z-dimension domain bounds (meters, for RRR).
- `zdim`: Number of B-spline cells in Z.
- `mubar`: Quadrature points per cell (default: 3).
- `quadrature`: Quadrature rule (default: `:gauss`).
- `BCL`, `BCR`: Left/right boundary conditions for X. Accepts a Springsteel
  `BoundaryConditions` (e.g. `NaturalBC()`, `DirichletBC()`, `NeumannBC()`)
  or a legacy module-qualified Dict (e.g. `CubicBSpline.R0`, `CubicBSpline.R1T1`).
  Default: `NaturalBC()`.
- `BCU`, `BCD`: Upper/lower boundary conditions for Y. Same accepted forms as `BCL`.
- `BCB`, `BCT`: Bottom/top boundary conditions for Z. Same accepted forms as `BCL`.

# Returns
A typed `SpringsteelGrid` (`R_Grid`, `RR_Grid`, or `RRR_Grid`).

# Example
```julia
moment_dict = Dict("DBZ" => 1, "VEL" => 2)
sgrid = create_radar_grid("RRR", moment_dict;
    xmin=-50000.0, xmax=50000.0, xdim=10,
    ymin=-50000.0, ymax=50000.0, ydim=10,
    zmin=0.0, zmax=15000.0, zdim=5)
```

See also: [`get_springsteel_gridpoints_zyx`](@ref), [`populate_physical!`](@ref)
"""
function create_radar_grid(geometry::String, moment_dict::Dict;
        xmin::Real=0.0, xmax::Real=0.0, xdim::Integer=0,
        ymin::Real=0.0, ymax::Real=0.0, ydim::Integer=0,
        zmin::Real=0.0, zmax::Real=0.0, zdim::Integer=0,
        mubar::Integer=3, quadrature::Symbol=:gauss,
        BCL::Union{Dict, BoundaryConditions}=NaturalBC(),
        BCR::Union{Dict, BoundaryConditions}=NaturalBC(),
        BCU::Union{Dict, BoundaryConditions}=NaturalBC(),
        BCD::Union{Dict, BoundaryConditions}=NaturalBC(),
        BCB::Union{Dict, BoundaryConditions}=NaturalBC(),
        BCT::Union{Dict, BoundaryConditions}=NaturalBC())

    # Build vars dict from moment_dict (name => index)
    vars = Dict{String, Int}(k => v for (k, v) in moment_dict)

    # Build per-variable BC dicts
    bc_l = Dict(k => BCL for k in keys(vars))
    bc_r = Dict(k => BCR for k in keys(vars))
    bc_u = Dict(k => BCU for k in keys(vars))
    bc_d = Dict(k => BCD for k in keys(vars))
    bc_b = Dict(k => BCB for k in keys(vars))
    bc_t = Dict(k => BCT for k in keys(vars))

    geom = uppercase(geometry)

    if geom == "R"
        gp = SpringsteelGridParameters(
            geometry   = "R",
            iMin       = Float64(xmin),
            iMax       = Float64(xmax),
            num_cells  = Int(xdim),
            mubar      = Int(mubar),
            quadrature = quadrature,
            BCL        = bc_l,
            BCR        = bc_r,
            vars       = vars,
        )
    elseif geom == "RR"
        gp = SpringsteelGridParameters(
            geometry   = "RR",
            iMin       = Float64(xmin),
            iMax       = Float64(xmax),
            num_cells  = Int(xdim),
            mubar      = Int(mubar),
            quadrature = quadrature,
            jMin       = Float64(ymin),
            jMax       = Float64(ymax),
            jDim       = Int(ydim) * Int(mubar),
            BCL        = bc_l,
            BCR        = bc_r,
            BCU        = bc_u,
            BCD        = bc_d,
            vars       = vars,
        )
    elseif geom == "RRR"
        gp = SpringsteelGridParameters(
            geometry   = "RRR",
            iMin       = Float64(xmin),
            iMax       = Float64(xmax),
            num_cells  = Int(xdim),
            mubar      = Int(mubar),
            quadrature = quadrature,
            jMin       = Float64(ymin),
            jMax       = Float64(ymax),
            jDim       = Int(ydim) * Int(mubar),
            kMin       = Float64(zmin),
            kMax       = Float64(zmax),
            kDim       = Int(zdim) * Int(mubar),
            BCL        = bc_l,
            BCR        = bc_r,
            BCU        = bc_u,
            BCD        = bc_d,
            BCB        = bc_b,
            BCT        = bc_t,
            vars       = vars,
        )
    else
        error("Unsupported geometry: $geometry. Use \"R\", \"RR\", or \"RRR\".")
    end

    gp = compute_derived_params(gp)
    return createGrid(gp)
end

"""
    daisho_to_springsteel_index(x, y, z, jDim, kDim)

Convert Daisho (x, y, z) 1-based grid indices to a Springsteel flat physical index.

Springsteel ordering: k varies fastest, then j, then i.

# Arguments
- `x`, `y`, `z`: 1-based Daisho grid indices (x=easting/i, y=northing/j, z=altitude/k).
- `jDim`: Number of physical grid points in the j (Y) dimension.
- `kDim`: Number of physical grid points in the k (Z) dimension.

# Returns
1-based flat index into Springsteel's physical array.

See also: [`daisho_to_springsteel_index_2d`](@ref)
"""
function daisho_to_springsteel_index(x::Integer, y::Integer, z::Integer,
                                      jDim::Integer, kDim::Integer)
    return (x - 1) * jDim * kDim + (y - 1) * kDim + z
end

"""
    daisho_to_springsteel_index_2d(x, y, jDim)

Convert Daisho (x, y) 1-based grid indices to a Springsteel flat physical index for 2D grids.

# Arguments
- `x`, `y`: 1-based Daisho grid indices.
- `jDim`: Number of physical grid points in the j (Y) dimension.

# Returns
1-based flat index into Springsteel's physical array.
"""
function daisho_to_springsteel_index_2d(x::Integer, y::Integer, jDim::Integer)
    return (x - 1) * jDim + y
end

"""
    get_springsteel_gridpoints_zyx(sgrid::SpringsteelGrid)

Extract Gaussian quadrature coordinates from a Springsteel grid and reshape
them into Daisho's array format with `[z, y, x]` ordering.

# Arguments
- `sgrid`: A `SpringsteelGrid` (R_Grid, RR_Grid, or RRR_Grid).

# Returns
- For RRR (3D): `Array{Float64, 4}` of shape `(kDim, jDim, iDim, 3)` with `[:, :, :, 1]` = z,
  `[:, :, :, 2]` = y, `[:, :, :, 3]` = x.
- For RR (2D): `Array{Float64, 3}` of shape `(jDim, iDim, 2)` with `[:, :, 1]` = y,
  `[:, :, 2]` = x.
- For R (1D): `Vector{Float64}` of shape `(iDim,)` with x-coordinates.

See also: [`create_radar_grid`](@ref), [`populate_physical!`](@ref)
"""
function get_springsteel_gridpoints_zyx end

# 3D: RRR_Grid
function get_springsteel_gridpoints_zyx(sgrid::RRR_Grid)
    pts = getGridpoints(sgrid)  # (iDim*jDim*kDim, 3) with [x, y, z] columns
    iDim = sgrid.params.iDim
    jDim = sgrid.params.jDim
    kDim = sgrid.params.kDim

    gridpoints = Array{Float64}(undef, kDim, jDim, iDim, 3)
    g = 1
    for i in 1:iDim
        for j in 1:jDim
            for k in 1:kDim
                gridpoints[k, j, i, 3] = pts[g, 1]  # x (i-coord)
                gridpoints[k, j, i, 2] = pts[g, 2]  # y (j-coord)
                gridpoints[k, j, i, 1] = pts[g, 3]  # z (k-coord)
                g += 1
            end
        end
    end
    return gridpoints
end

# 2D: RR_Grid
function get_springsteel_gridpoints_zyx(sgrid::RR_Grid)
    pts = getGridpoints(sgrid)  # (iDim*jDim, 2) with [x, y] columns
    iDim = sgrid.params.iDim
    jDim = sgrid.params.jDim

    gridpoints = Array{Float64}(undef, jDim, iDim, 2)
    g = 1
    for i in 1:iDim
        for j in 1:jDim
            gridpoints[j, i, 2] = pts[g, 1]  # x (i-coord)
            gridpoints[j, i, 1] = pts[g, 2]  # y (j-coord)
            g += 1
        end
    end
    return gridpoints
end

# 1D: R_Grid
function get_springsteel_gridpoints_zyx(sgrid::R_Grid)
    return getGridpoints(sgrid)  # Vector of x-coordinates
end

"""
    populate_physical!(sgrid, radar_grid, moment_dict)

Copy Daisho's gridded radar data into a Springsteel grid's physical array (slot 1).

Handles fill value translation:
- `-32768.0` (true missing — no radar coverage) → `NaN`
- `-9999.0` (clear air — scanned, no echo) → preserved as `-9999.0`
- Valid values → copied as-is

# Arguments
- `sgrid::SpringsteelGrid`: Target spectral grid.
- `radar_grid::Array`: Daisho radar grid with shape `(n_moments, zdim, ydim, xdim)` for 3D,
  `(n_moments, ydim, xdim)` for 2D, or `(n_moments, xdim)` for 1D.
- `moment_dict::Dict`: Mapping of moment names to column indices.

See also: [`create_radar_grid`](@ref), [`get_springsteel_gridpoints_zyx`](@ref)
"""
function populate_physical! end

# 3D: RRR_Grid
function populate_physical!(sgrid::RRR_Grid,
        radar_grid::AbstractArray, moment_dict::Dict)
    iDim = sgrid.params.iDim
    jDim = sgrid.params.jDim
    kDim = sgrid.params.kDim

    for (name, var_idx) in moment_dict
        g = 1
        for i in 1:iDim
            for j in 1:jDim
                for k in 1:kDim
                    val = radar_grid[var_idx, k, j, i]
                    if val == -32768.0
                        sgrid.physical[g, var_idx, 1] = NaN
                    else
                        sgrid.physical[g, var_idx, 1] = val
                    end
                    g += 1
                end
            end
        end
    end
    return nothing
end

# 2D: RR_Grid
function populate_physical!(sgrid::RR_Grid,
        radar_grid::AbstractArray, moment_dict::Dict)
    iDim = sgrid.params.iDim
    jDim = sgrid.params.jDim

    for (name, var_idx) in moment_dict
        g = 1
        for i in 1:iDim
            for j in 1:jDim
                val = radar_grid[var_idx, j, i]
                if val == -32768.0
                    sgrid.physical[g, var_idx, 1] = NaN
                else
                    sgrid.physical[g, var_idx, 1] = val
                end
                g += 1
            end
        end
    end
    return nothing
end

# 1D: R_Grid
function populate_physical!(sgrid::R_Grid,
        radar_grid::AbstractArray, moment_dict::Dict)
    iDim = sgrid.params.iDim

    for (name, var_idx) in moment_dict
        for i in 1:iDim
            val = radar_grid[var_idx, i]
            if val == -32768.0
                sgrid.physical[i, var_idx, 1] = NaN
            else
                sgrid.physical[i, var_idx, 1] = val
            end
        end
    end
    return nothing
end

"""
    compute_roi(sgrid::SpringsteelGrid)

Compute horizontal and vertical radius-of-influence from a Springsteel grid's
quadrature point spacing.

Uses 0.75 × average spacing, matching Daisho convention.

# Arguments
- `sgrid`: A `SpringsteelGrid`.

# Returns
- For 3D grids: `(h_roi, v_roi)` tuple.
- For 2D grids: `(h_roi,)` tuple (single horizontal ROI).
- For 1D grids: `(roi,)` tuple.

See also: [`create_radar_grid`](@ref)
"""
function compute_roi end

# 3D: RRR_Grid
function compute_roi(sgrid::RRR_Grid)
    pts = getGridpoints(sgrid)
    iDim = sgrid.params.iDim
    jDim = sgrid.params.jDim
    kDim = sgrid.params.kDim

    # Average spacing in i (x) direction
    x_pts = [pts[(i-1)*jDim*kDim + 1, 1] for i in 1:iDim]
    h_spacing = mean(diff(x_pts))

    # Average spacing in k (z) direction
    z_pts = [pts[k, 3] for k in 1:kDim]
    v_spacing = mean(diff(z_pts))

    return (0.75 * abs(h_spacing), 0.75 * abs(v_spacing))
end

# 2D: RR_Grid
function compute_roi(sgrid::RR_Grid)
    pts = getGridpoints(sgrid)
    iDim = sgrid.params.iDim
    jDim = sgrid.params.jDim

    x_pts = [pts[(i-1)*jDim + 1, 1] for i in 1:iDim]
    h_spacing = mean(diff(x_pts))

    return (0.75 * abs(h_spacing),)
end

# 1D: R_Grid
function compute_roi(sgrid::R_Grid)
    pts = getGridpoints(sgrid)
    spacing = mean(diff(pts))
    return (0.75 * abs(spacing),)
end

"""
    radar_global_attributes(radar_volume, ref_lat, ref_lon;
        institution="", source="", heading=-9999.0)

Build a CF-compliant global attribute dictionary for radar NetCDF output.

# Arguments
- `radar_volume`: Daisho radar volume data structure.
- `ref_lat`: Reference latitude for the Transverse Mercator projection (degrees).
- `ref_lon`: Reference longitude for the Transverse Mercator projection (degrees).
- `institution`: Institution name string.
- `source`: Source instrument description.
- `heading`: Platform heading in degrees (default: -9999.0 for missing).

# Returns
`Dict{String,Any}` of CF-compliant global attributes.

See also: [`write_radar_netcdf`](@ref)
"""
function radar_global_attributes(radar_volume, ref_lat::Real, ref_lon::Real;
        institution::String="", source::String="", heading::Real=-9999.0)
    attrs = Dict{String,Any}(
        "Conventions"        => "CF-1.12",
        "title"              => "Daisho.jl spectral radar gridded data",
        "institution"        => institution,
        "source"             => source,
        "history"            => "Created " * Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS") * " by Daisho.jl",
        "references"         => "Daisho.jl radar analysis package",
        "reference_latitude" => Float64(ref_lat),
        "reference_longitude"=> Float64(ref_lon),
        "time_coverage_start"=> string(radar_volume.time[1]),
        "time_coverage_end"  => string(radar_volume.time[end]),
    )
    if heading != -9999.0
        attrs["platform_heading"] = Float64(heading)
    end
    return attrs
end

"""
    write_radar_netcdf(filename, sgrid, radar_volume, moment_dict;
        ref_lat=0.0, ref_lon=0.0, institution="", source="", heading=-9999.0,
        include_derivatives=false)

Write a Springsteel spectral grid with radar data to a CF-compliant NetCDF file.

Calls `Springsteel.write_netcdf` with radar-specific global attributes, then
patches in radar variable attributes and grid mapping via NCDatasets.

# Arguments
- `filename`: Output NetCDF file path.
- `sgrid`: A `SpringsteelGrid` with populated physical and spectral arrays.
- `radar_volume`: Daisho radar volume for metadata extraction.
- `moment_dict`: Dictionary mapping moment names to indices.
- `ref_lat`, `ref_lon`: Reference latitude/longitude for projection.
- `institution`, `source`: Metadata strings.
- `heading`: Platform heading in degrees.
- `include_derivatives`: Whether to include derivative fields in output.

See also: [`radar_global_attributes`](@ref), [`grid_radar_volume_spectral`](@ref)
"""
function write_radar_netcdf(filename::String, sgrid::SpringsteelGrid,
        radar_volume, moment_dict::Dict;
        ref_lat::Real=0.0, ref_lon::Real=0.0,
        institution::String="", source::String="", heading::Real=-9999.0,
        include_derivatives::Bool=false)

    global_attrs = radar_global_attributes(radar_volume, ref_lat, ref_lon;
        institution=institution, source=source, heading=heading)

    # Build variable attributes dict for radar moments
    var_attrs = Dict{String, Dict{String, Any}}()
    moment_units = Dict(
        "DBZ" => ("dBZ", "equivalent reflectivity factor"),
        "DZ"  => ("dBZ", "equivalent reflectivity factor"),
        "VEL" => ("m/s", "radial velocity"),
        "VR"  => ("m/s", "radial velocity"),
        "WIDTH" => ("m/s", "spectrum width"),
        "SW"  => ("m/s", "spectrum width"),
        "ZDR" => ("dB", "differential reflectivity"),
        "RHOHV" => ("1", "cross-correlation coefficient"),
        "PHIDP" => ("degrees", "differential phase"),
        "KDP" => ("degrees/km", "specific differential phase"),
    )
    for (name, idx) in moment_dict
        if haskey(moment_units, name)
            units, long_name = moment_units[name]
            var_attrs[name] = Dict{String, Any}("units" => units, "long_name" => long_name)
        else
            var_attrs[name] = Dict{String, Any}("units" => "1", "long_name" => name)
        end
    end

    # Build coordinate attributes with meters units for Cartesian grids
    coord_attrs = Dict{String, Dict{String, Any}}(
        "x" => Dict{String, Any}("units" => "m", "long_name" => "easting"),
        "y" => Dict{String, Any}("units" => "m", "long_name" => "northing"),
        "z" => Dict{String, Any}("units" => "m", "long_name" => "altitude"),
    )

    # Build grid mapping dict for Transverse Mercator projection
    gm = Dict{String, Any}(
        "grid_mapping_name" => "transverse_mercator",
        "latitude_of_projection_origin" => Float64(ref_lat),
        "longitude_of_central_meridian" => Float64(ref_lon),
        "scale_factor_at_central_meridian" => 1.0,
        "false_easting" => 0.0,
        "false_northing" => 0.0,
    )

    Springsteel.write_netcdf(filename, sgrid;
        include_derivatives=include_derivatives,
        global_attributes=global_attrs,
        coordinate_attributes=coord_attrs,
        variable_attributes=var_attrs,
        grid_mapping=gm)

    return nothing
end

"""
    mask_fill_for_transform!(sgrid)

Replace NaN and -9999.0 values with 0.0 in the physical array before spectral
transform. Returns a mask matrix tracking original fill states.

Mask values: 0 = valid, 1 = was NaN (true missing), 2 = was -9999.0 (clear air).

See also: [`restore_fill_from_mask!`](@ref)
"""
function mask_fill_for_transform!(sgrid::SpringsteelGrid)
    phys = sgrid.physical
    npts = size(phys, 1)
    nvars = size(phys, 2)
    mask = zeros(Int8, npts, nvars)

    @inbounds for v in 1:nvars
        for p in 1:npts
            val = phys[p, v, 1]
            if isnan(val)
                mask[p, v] = Int8(1)
                phys[p, v, 1] = 0.0
            elseif val == -9999.0
                mask[p, v] = Int8(2)
                phys[p, v, 1] = 0.0
            end
        end
    end
    return mask
end

"""
    restore_fill_from_mask!(sgrid, mask)

Restore NaN and -9999.0 fill values after spectral transform, using the mask
created by `mask_fill_for_transform!`.

See also: [`mask_fill_for_transform!`](@ref)
"""
function restore_fill_from_mask!(sgrid::SpringsteelGrid, mask::AbstractArray)
    phys = sgrid.physical
    npts = size(phys, 1)
    nvars = size(phys, 2)
    nslots = size(phys, 3)

    @inbounds for v in 1:nvars
        for p in 1:npts
            m = mask[p, v]
            if m == Int8(1)
                for s in 1:nslots
                    phys[p, v, s] = NaN
                end
            elseif m == Int8(2)
                for s in 1:nslots
                    phys[p, v, s] = -9999.0
                end
            end
        end
    end
    return nothing
end

"""
    grid_radar_volume_spectral(radar_volume, moment_dict, grid_type_dict,
        output_file, index_time, sgrid, beam_inflation, power_threshold;
        missing_key="SQI", valid_key="DBZ", heading=-9999.0,
        institution="", source="", include_derivatives=false)

Grid a radar volume onto a Springsteel spectral grid and perform spectral analysis.

This is the high-level spectral gridding workflow:
1. Extract quadrature coordinates from `sgrid` in Daisho format
2. Compute ROI from grid spacing
3. Call `grid_volume` (reuses entire existing gridding engine)
4. Populate the Springsteel physical array
5. Mask fill values, perform spectral + grid transforms, restore fills
6. Write CF-compliant NetCDF output
7. Return the populated `sgrid` for further analysis

# Arguments
- `radar_volume`: Radar volume data structure.
- `moment_dict`: Dictionary mapping moment names to integer indices.
- `grid_type_dict`: Dictionary mapping moment indices to interpolation type symbols.
- `output_file`: Path to the output NetCDF file.
- `index_time`: Reference time for the output dataset.
- `sgrid`: Pre-configured `SpringsteelGrid` (from `create_radar_grid`).
- `beam_inflation`: Factor for inflating ROI with distance from radar.
- `power_threshold`: Minimum beam power weight.
- `missing_key`, `valid_key`: Moment names for QC gating.
- `heading`: Platform heading in degrees.
- `institution`, `source`: Metadata strings for NetCDF output.
- `include_derivatives`: Whether to include derivative fields in output.

# Returns
The populated `SpringsteelGrid` with spectral coefficients and physical values.

See also: [`create_radar_grid`](@ref), [`write_radar_netcdf`](@ref)
"""
function grid_radar_volume_spectral(radar_volume, moment_dict::Dict, grid_type_dict::Dict,
        output_file::String, index_time, sgrid::SpringsteelGrid,
        beam_inflation::Float64, power_threshold::Float64;
        missing_key::String="SQI", valid_key::String="DBZ", heading::Real=-9999.0,
        institution::String="", source::String="",
        include_derivatives::Bool=false)

    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    # 1. Get gridpoints in Daisho format
    gridpoints = get_springsteel_gridpoints_zyx(sgrid)

    # 2. Compute ROI
    roi = compute_roi(sgrid)
    h_roi = roi[1]
    v_roi = length(roi) > 1 ? roi[2] : roi[1]

    # 3. Grid using existing engine
    radar_grid, latlon_grid = grid_volume(reference_latitude, reference_longitude,
        gridpoints, radar_volume, moment_dict, grid_type_dict,
        h_roi, v_roi, beam_inflation, power_threshold, missing_key, valid_key)

    # 4. Populate Springsteel physical array
    populate_physical!(sgrid, radar_grid, moment_dict)

    # 5. Spectral analysis with fill value masking
    mask = mask_fill_for_transform!(sgrid)
    spectralTransform!(sgrid)
    gridTransform!(sgrid)
    restore_fill_from_mask!(sgrid, mask)

    # 6. Write output
    write_radar_netcdf(output_file, sgrid, radar_volume, moment_dict;
        ref_lat=reference_latitude, ref_lon=reference_longitude,
        institution=institution, source=source, heading=heading,
        include_derivatives=include_derivatives)

    return sgrid
end

"""
    grid_radar_ppi_spectral(radar_volume, moment_dict, grid_type_dict,
        output_file, index_time, sgrid, beam_inflation, power_threshold;
        missing_key="SQI", valid_key="DBZ", heading=-9999.0,
        institution="", source="", include_derivatives=false)

Grid a radar PPI scan onto a 2D Springsteel spectral grid and perform spectral analysis.

See [`grid_radar_volume_spectral`](@ref) for the full workflow description.
"""
function grid_radar_ppi_spectral(radar_volume, moment_dict::Dict, grid_type_dict::Dict,
        output_file::String, index_time, sgrid::SpringsteelGrid,
        beam_inflation::Float64, power_threshold::Float64;
        missing_key::String="SQI", valid_key::String="DBZ", heading::Real=-9999.0,
        institution::String="", source::String="",
        include_derivatives::Bool=false)

    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    gridpoints = get_springsteel_gridpoints_zyx(sgrid)
    roi = compute_roi(sgrid)
    h_roi = roi[1]

    radar_grid, latlon_grid = grid_ppi(reference_latitude, reference_longitude,
        gridpoints, radar_volume, moment_dict, grid_type_dict,
        h_roi, beam_inflation, power_threshold, missing_key, valid_key)

    populate_physical!(sgrid, radar_grid, moment_dict)

    mask = mask_fill_for_transform!(sgrid)
    spectralTransform!(sgrid)
    gridTransform!(sgrid)
    restore_fill_from_mask!(sgrid, mask)

    write_radar_netcdf(output_file, sgrid, radar_volume, moment_dict;
        ref_lat=reference_latitude, ref_lon=reference_longitude,
        institution=institution, source=source, heading=heading,
        include_derivatives=include_derivatives)

    return sgrid
end

"""
    grid_radar_column_spectral(radar_volume, moment_dict, grid_type_dict,
        output_file, index_time, sgrid, beam_inflation, power_threshold;
        missing_key="SQI", valid_key="DBZ",
        institution="", source="", include_derivatives=false)

Grid a radar column onto a 1D Springsteel spectral grid and perform spectral analysis.

See [`grid_radar_volume_spectral`](@ref) for the full workflow description.
"""
function grid_radar_column_spectral(radar_volume, moment_dict::Dict, grid_type_dict::Dict,
        output_file::String, index_time, sgrid::SpringsteelGrid,
        beam_inflation::Float64, power_threshold::Float64;
        missing_key::String="SQI", valid_key::String="DBZ",
        institution::String="", source::String="",
        include_derivatives::Bool=false)

    reference_latitude = radar_volume.latitude[1]
    reference_longitude = radar_volume.longitude[1]

    gridpoints = get_springsteel_gridpoints_zyx(sgrid)
    roi = compute_roi(sgrid)
    v_roi = roi[1]

    radar_grid, latlon_grid = grid_column(reference_latitude, reference_longitude,
        gridpoints, radar_volume, moment_dict, grid_type_dict,
        v_roi, beam_inflation, power_threshold, missing_key, valid_key)

    populate_physical!(sgrid, radar_grid, moment_dict)

    mask = mask_fill_for_transform!(sgrid)
    spectralTransform!(sgrid)
    gridTransform!(sgrid)
    restore_fill_from_mask!(sgrid, mask)

    write_radar_netcdf(output_file, sgrid, radar_volume, moment_dict;
        ref_lat=reference_latitude, ref_lon=reference_longitude,
        institution=institution, source=source,
        include_derivatives=include_derivatives)

    return sgrid
end

# ── Parameter-struct overloads ──────────────────────────────────────────────
# Read spectral-grid configuration directly from a `DaishoParameters`. Per-axis
# BCs come from `p.grid.spectral.bc`; geometry/cell counts/bounds from
# `p.grid.spectral`.

"""
    create_radar_grid(p::DaishoParameters)

Parameter-struct overload that builds a Springsteel spectral grid from
`p.grid.spectral` and `p.moments`. The Springsteel variable map is built from
`p.moments.fields` (1-based indices in source order).
"""
function create_radar_grid(p::DaishoParameters)
    s = p.grid.spectral
    moment_dict = field_index_dict(p)
    return create_radar_grid(s.geometry, moment_dict;
        xmin=s.xmin, xmax=s.xmax, xdim=s.xdim,
        ymin=s.ymin, ymax=s.ymax, ydim=s.ydim,
        zmin=s.zmin, zmax=s.zmax, zdim=s.zdim,
        mubar=s.mubar, quadrature=s.quadrature,
        BCL=s.bc.xL, BCR=s.bc.xR,
        BCU=s.bc.yL, BCD=s.bc.yR,
        BCB=s.bc.zL, BCT=s.bc.zR)
end

"""
    grid_radar_volume_spectral(radar_volume, output_file, index_time, sgrid, p::DaishoParameters; heading=-9999.0, institution="", source="", include_derivatives=false)

Parameter-struct overload reading gridding config from `p.gridding` and the
moment map from `p.moments`.
"""
function grid_radar_volume_spectral(radar_volume, output_file::AbstractString,
        index_time, sgrid::SpringsteelGrid, p::DaishoParameters;
        heading::Real=-9999.0, institution::String="", source::String="",
        include_derivatives::Bool=false)
    moment_dict = field_index_dict(p)
    grid_radar_volume_spectral(radar_volume, moment_dict, grid_type_index_dict(p),
        output_file, index_time, sgrid,
        p.gridding.beam_inflation, p.gridding.power_threshold;
        missing_key=p.gridding.missing_key, valid_key=p.gridding.valid_key,
        heading=heading, institution=institution, source=source,
        include_derivatives=include_derivatives)
end

"""
    grid_radar_ppi_spectral(radar_volume, output_file, index_time, sgrid, p::DaishoParameters; heading=-9999.0, institution="", source="", include_derivatives=false)

Parameter-struct overload reading gridding config from `p.gridding` and the
moment map from `p.moments`.
"""
function grid_radar_ppi_spectral(radar_volume, output_file::AbstractString,
        index_time, sgrid::SpringsteelGrid, p::DaishoParameters;
        heading::Real=-9999.0, institution::String="", source::String="",
        include_derivatives::Bool=false)
    moment_dict = field_index_dict(p)
    grid_radar_ppi_spectral(radar_volume, moment_dict, grid_type_index_dict(p),
        output_file, index_time, sgrid,
        p.gridding.beam_inflation, p.gridding.power_threshold;
        missing_key=p.gridding.missing_key, valid_key=p.gridding.valid_key,
        heading=heading, institution=institution, source=source,
        include_derivatives=include_derivatives)
end

"""
    grid_radar_column_spectral(radar_volume, output_file, index_time, sgrid, p::DaishoParameters; institution="", source="", include_derivatives=false)

Parameter-struct overload reading gridding config from `p.gridding` and the
moment map from `p.moments`.
"""
function grid_radar_column_spectral(radar_volume, output_file::AbstractString,
        index_time, sgrid::SpringsteelGrid, p::DaishoParameters;
        institution::String="", source::String="", include_derivatives::Bool=false)
    moment_dict = field_index_dict(p)
    grid_radar_column_spectral(radar_volume, moment_dict, grid_type_index_dict(p),
        output_file, index_time, sgrid,
        p.gridding.beam_inflation, p.gridding.power_threshold;
        missing_key=p.gridding.missing_key, valid_key=p.gridding.valid_key,
        institution=institution, source=source,
        include_derivatives=include_derivatives)
end

# ── Volume-aware overloads (Phase C bridge) ──────────────────────────────────

function grid_radar_volume_spectral(volume::Volume, output_file::AbstractString,
        index_time, sgrid::SpringsteelGrid, p::DaishoParameters;
        heading::Real=-9999.0, institution::String="", source::String="",
        include_derivatives::Bool=false)
    legacy, _names = as_legacy_radar(volume; field_names=p.moments.fields)
    grid_radar_volume_spectral(legacy, output_file, index_time, sgrid, p;
        heading=heading, institution=institution, source=source,
        include_derivatives=include_derivatives)
end

function grid_radar_ppi_spectral(volume::Volume, output_file::AbstractString,
        index_time, sgrid::SpringsteelGrid, p::DaishoParameters;
        heading::Real=-9999.0, institution::String="", source::String="",
        include_derivatives::Bool=false)
    legacy, _names = as_legacy_radar(volume; field_names=p.moments.fields)
    grid_radar_ppi_spectral(legacy, output_file, index_time, sgrid, p;
        heading=heading, institution=institution, source=source,
        include_derivatives=include_derivatives)
end

function grid_radar_column_spectral(volume::Volume, output_file::AbstractString,
        index_time, sgrid::SpringsteelGrid, p::DaishoParameters;
        institution::String="", source::String="", include_derivatives::Bool=false)
    legacy, _names = as_legacy_radar(volume; field_names=p.moments.fields)
    grid_radar_column_spectral(legacy, output_file, index_time, sgrid, p;
        institution=institution, source=source,
        include_derivatives=include_derivatives)
end
