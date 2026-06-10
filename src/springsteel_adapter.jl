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

    sgrid = createGrid(gp)
    # Geometry-aware output sizing: Cartesian spline axes default to cells + 1
    # regularly spaced output points (cell-edge spacing). Springsteel's own
    # j default is the cylindrical outermost-ring rule and its k default is
    # mish + 1, both wrong for Cartesian grids (csu-tropical/Springsteel.jl#6).
    jout = geom in ("RR", "RRR") ? Int(ydim) + 1 : sgrid.params.j_regular_out
    kout = geom == "RRR" ? Int(zdim) + 1 : sgrid.params.k_regular_out
    return _with_regular_out(sgrid, Int(xdim) + 1, jout, kout)
end

# Rebuild a grid with explicit i/j/k_regular_out output sizing. Springsteel's
# `compute_derived_params` (run inside `createGrid`) reconstructs the params
# without carrying user-set `*_regular_out` values, so they can only be
# injected after construction. The basis/spectral/physical arrays are shared,
# not copied.
function _with_regular_out(sgrid::SpringsteelGrid,
        iout::Integer, jout::Integer, kout::Integer)
    gp = sgrid.params
    if gp.i_regular_out == iout && gp.j_regular_out == jout &&
       gp.k_regular_out == kout
        return sgrid
    end
    d = Dict{Symbol,Any}(f => getfield(gp, f)
                         for f in fieldnames(SpringsteelGridParameters))
    d[:i_regular_out] = Int(iout)
    d[:j_regular_out] = Int(jout)
    d[:k_regular_out] = Int(kout)
    gp2 = SpringsteelGridParameters(; d...)
    return typeof(sgrid)(gp2, sgrid.ibasis, sgrid.jbasis, sgrid.kbasis,
                         sgrid.spectral, sgrid.physical)
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
- **true missing** (gate not measured, the io `fill_value`) → `NaN`
- **undetect** (scanned, no echo, the io `undetect`) → preserved as-is
- **valid** values → copied as-is

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
        radar_grid::AbstractArray, moment_dict::Dict;
        fill_value::Real = -32768.0)
    iDim = sgrid.params.iDim
    jDim = sgrid.params.jDim
    kDim = sgrid.params.kDim

    # `fill_value` must match the true-missing sentinel of whatever produced
    # `radar_grid`. The legacy `grid_*` workers feeding the spectral path emit
    # -32768.0; the accumulator/finalize_grid path emits `[io] fill_value`.
    for (name, var_idx) in moment_dict
        g = 1
        for i in 1:iDim
            for j in 1:jDim
                for k in 1:kDim
                    val = radar_grid[var_idx, k, j, i]
                    if val == fill_value
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
        radar_grid::AbstractArray, moment_dict::Dict;
        fill_value::Real = -32768.0)
    iDim = sgrid.params.iDim
    jDim = sgrid.params.jDim

    for (name, var_idx) in moment_dict
        g = 1
        for i in 1:iDim
            for j in 1:jDim
                val = radar_grid[var_idx, j, i]
                if val == fill_value
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
        radar_grid::AbstractArray, moment_dict::Dict;
        fill_value::Real = -32768.0)
    iDim = sgrid.params.iDim

    for (name, var_idx) in moment_dict
        for i in 1:iDim
            val = radar_grid[var_idx, i]
            if val == fill_value
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

Uses `h_factor`/`v_factor` × average quadrature-point spacing (default 0.75,
the legacy Daisho convention). The accumulator-path Springsteel provider passes
`p.gridding.horizontal_roi_factor` / `vertical_roi_factor`.

# Arguments
- `sgrid`: A `SpringsteelGrid`.
- `h_factor`: Horizontal ROI fraction of the mean horizontal node spacing.
- `v_factor`: Vertical ROI fraction of the mean vertical node spacing (3D only).

# Returns
- For 3D grids: `(h_roi, v_roi)` tuple.
- For 2D grids: `(h_roi,)` tuple (single horizontal ROI).
- For 1D grids: `(roi,)` tuple.

See also: [`create_radar_grid`](@ref)
"""
function compute_roi end

# 3D: RRR_Grid
function compute_roi(sgrid::RRR_Grid; h_factor::Real = 0.75, v_factor::Real = 0.75)
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

    return (h_factor * abs(h_spacing), v_factor * abs(v_spacing))
end

# 2D: RR_Grid
function compute_roi(sgrid::RR_Grid; h_factor::Real = 0.75, v_factor::Real = 0.75)
    pts = getGridpoints(sgrid)
    iDim = sgrid.params.iDim
    jDim = sgrid.params.jDim

    x_pts = [pts[(i-1)*jDim + 1, 1] for i in 1:iDim]
    h_spacing = mean(diff(x_pts))

    return (h_factor * abs(h_spacing),)
end

# 1D: R_Grid
function compute_roi(sgrid::R_Grid; h_factor::Real = 0.75, v_factor::Real = 0.75)
    pts = getGridpoints(sgrid)
    spacing = mean(diff(pts))
    return (v_factor * abs(spacing),)
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

# `Volume`-typed overload for the accumulator spectral path. A `Volume` carries
# `time_coverage_start`/`time_coverage_end` rather than the legacy radar's
# per-ray `time` vector, so the coverage window is read from those fields.
function radar_global_attributes(volume::Volume, ref_lat::Real, ref_lon::Real;
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
        "time_coverage_start"=> string(volume.time_coverage_start),
        "time_coverage_end"  => string(volume.time_coverage_end),
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
        output_file, index_time, sgrid, power_threshold;
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
        power_threshold::Float64;
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
        h_roi, v_roi, power_threshold, missing_key, valid_key;
        beam_width = radar_volume.beam_width)

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
        output_file, index_time, sgrid, power_threshold;
        missing_key="SQI", valid_key="DBZ", heading=-9999.0,
        institution="", source="", include_derivatives=false)

Grid a radar PPI scan onto a 2D Springsteel spectral grid and perform spectral analysis.

See [`grid_radar_volume_spectral`](@ref) for the full workflow description.
"""
function grid_radar_ppi_spectral(radar_volume, moment_dict::Dict, grid_type_dict::Dict,
        output_file::String, index_time, sgrid::SpringsteelGrid,
        power_threshold::Float64;
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
        h_roi, power_threshold, missing_key, valid_key;
        beam_width = radar_volume.beam_width)

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
        output_file, index_time, sgrid, power_threshold;
        missing_key="SQI", valid_key="DBZ",
        institution="", source="", include_derivatives=false)

Grid a radar column onto a 1D Springsteel spectral grid and perform spectral analysis.

See [`grid_radar_volume_spectral`](@ref) for the full workflow description.
"""
function grid_radar_column_spectral(radar_volume, moment_dict::Dict, grid_type_dict::Dict,
        output_file::String, index_time, sgrid::SpringsteelGrid,
        power_threshold::Float64;
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
        v_roi, power_threshold, missing_key, valid_key;
        beam_width = radar_volume.beam_width)

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
# Build the Springsteel grid straight from the geometry-general
# `[grid.springsteel]` config — no intermediate x/y/z vocabulary.

"""
    radar_vars(p::DaishoParameters) -> Dict{String,Int}

The Springsteel variable map for hand-built grids: field name → 1-based
column index in field-name-sorted order, identical to
[`field_index_dict`](@ref). Pass it as `vars` when constructing a
`Springsteel.SpringsteelGridParameters` yourself — the escape hatch for grids
the `[grid.springsteel]` schema cannot express (per-variable spectral
filters, Fourier/Chebyshev-primary geometries). The gridding entry points
validate the grid's `vars` against the configured `[fields]` and refuse
mismatches, since the gridded-array column order must line up with the
spectral-grid variable order.
"""
radar_vars(p::DaishoParameters) = field_index_dict(p)

"""
    create_radar_grid(p::DaishoParameters)
    create_radar_grid(cfg::SpringsteelGridConfig, vars::Dict{String,Int})

Build a Springsteel grid from the geometry-general `[grid.springsteel]`
config (any supported spline-`i` geometry: R/RZ/RR/RRR, RL/RLZ/RLR,
SL/SLZ/SLR). The Springsteel variable map is built via
[`field_index_dict`](@ref) (1-based indices in field-name-sorted order,
order-independent of the source TOML).

Axis mapping into `SpringsteelGridParameters`: per-axis `min`/`max` →
`iMin`/`iMax` etc.; spline `cells` → `num_cells` (i) or `jDim`/`kDim` =
`cells × mubar`; Chebyshev `points` → `kDim`. Boundary conditions map the
axis' **min side** to Springsteel's `BCL`/`BCD`/`BCB` and **max side** to
`BCR`/`BCU`/`BCT`. Output resampling counts (`regular_out`) are injected
after construction (see `_with_regular_out`), defaulting to `cells + 1` on
spline axes.

Note the gridding engine itself currently traverses only the Cartesian
R/RR/RRR grids; the other geometries can be built (and transformed/written)
but not yet gridded onto — [`build_springsteel_grid_spec`](@ref) raises a
clear error for them.
"""
function create_radar_grid(p::DaishoParameters)
    s = require_section(p, :grid, :springsteel; for_op="create_radar_grid")
    return create_radar_grid(s, field_index_dict(p))
end

function create_radar_grid(cfg::SpringsteelGridConfig, vars::Dict{String,Int})
    bases = SPRINGSTEEL_AXIS_BASES[cfg.geometry]
    _validate_bc_fields(cfg, vars)
    kw = Dict{Symbol,Any}(
        :geometry   => cfg.geometry,
        :vars       => Dict{String,Int}(vars),
        :mubar      => cfg.mubar,
        :quadrature => cfg.quadrature,
        :iMin       => cfg.i.min,
        :iMax       => cfg.i.max,
        :num_cells  => cfg.i.cells,
        :BCL        => cfg.i.bc_min,
        :BCR        => cfg.i.bc_max,
    )
    jb, kb = bases[2], bases[3]
    if jb === :spline
        kw[:jMin] = cfg.j.min
        kw[:jMax] = cfg.j.max
        kw[:jDim] = cfg.j.cells * cfg.mubar
        # Springsteel reads the j-spline's jMin-side BC from BCD and the
        # jMax-side from BCU (factory.jl _create_cartesian_2d_rr/_3d_rrr).
        kw[:BCD] = cfg.j.bc_min
        kw[:BCU] = cfg.j.bc_max
    elseif jb === :fourier && cfg.j.max_wavenumber >= 0
        kw[:max_wavenumber] = Dict{String,Any}("default" => cfg.j.max_wavenumber)
    end
    if kb !== :none
        kw[:kMin] = cfg.k.min
        kw[:kMax] = cfg.k.max
        kw[:kDim] = kb === :spline ? cfg.k.cells * cfg.mubar : cfg.k.points
        kw[:BCB] = cfg.k.bc_min
        kw[:BCT] = cfg.k.bc_max
    end
    sgrid = createGrid(SpringsteelGridParameters(; kw...))
    gp = sgrid.params
    iout = cfg.i.regular_out > 0 ? cfg.i.regular_out : cfg.i.cells + 1
    jout = cfg.j.regular_out > 0 ? cfg.j.regular_out :
           (jb === :spline ? cfg.j.cells + 1 : gp.j_regular_out)
    kout = cfg.k.regular_out > 0 ? cfg.k.regular_out :
           (kb === :spline ? cfg.k.cells + 1 : gp.k_regular_out)
    return _with_regular_out(sgrid, iout, jout, kout)
end

# Per-field BC override keys must name configured fields (typo guard). The
# TOML parser cannot check this — it runs before the field list is fixed — so
# it happens here, at grid-construction time.
function _validate_bc_fields(cfg::SpringsteelGridConfig, vars::Dict{String,Int})
    for (axis, name) in ((cfg.i, "i"), (cfg.j, "j"), (cfg.k, "k"))
        for (sidedict, side) in ((axis.bc_min, "min"), (axis.bc_max, "max"))
            for key in keys(sidedict)
                key == "default" && continue
                haskey(vars, key) || throw(ArgumentError(
                    "[grid.springsteel.$(name).bc]: per-field override " *
                    "`$(key)` ($(side) side) does not match any configured " *
                    "field. Fields: $(join(sort(collect(keys(vars))), ", "))"))
            end
        end
    end
    return nothing
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
        p.gridding.power_threshold;
        missing_key=field_with_tag(p, :define_scanned;   for_op="grid_radar_volume_spectral"),
        valid_key=field_with_tag(p, :define_detection; for_op="grid_radar_volume_spectral"),
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
        p.gridding.power_threshold;
        missing_key=field_with_tag(p, :define_scanned;   for_op="grid_radar_ppi_spectral"),
        valid_key=field_with_tag(p, :define_detection; for_op="grid_radar_ppi_spectral"),
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
        p.gridding.power_threshold;
        missing_key=field_with_tag(p, :define_scanned;   for_op="grid_radar_column_spectral"),
        valid_key=field_with_tag(p, :define_detection; for_op="grid_radar_column_spectral"),
        institution=institution, source=source,
        include_derivatives=include_derivatives)
end

# ── Springsteel → GridSpec builder ───────────────────────────────────────────

"""
    build_springsteel_grid_spec(sgrid; reference_latitude, reference_longitude) -> GridSpec
    build_springsteel_grid_spec(sgrid, volume::Volume) -> GridSpec

Build a Daisho [`GridSpec`](@ref) whose per-axis coordinate sets reproduce the
Springsteel grid's quadrature lattice, centered on the given reference
position (the `volume` form uses the volume's reference position). The
unified accumulator engine then grids onto exactly the Springsteel node
layout.

**Cartesian tensor-product assumption.** Springsteel Cartesian grids factorize
as `node(i,j,k) = (X[i], Y[j], Z[k])`, so the per-axis coordinate sets fully
describe the lattice. Non-Cartesian (cylindrical/spherical) grids raise an
`ArgumentError` — that is where future per-node gridpoints + metric-ROI work
plugs in.
"""
function build_springsteel_grid_spec end

build_springsteel_grid_spec(sgrid::SpringsteelGrid, volume::Volume) =
    build_springsteel_grid_spec(sgrid;
        reference_latitude  = Float64(volume.latitude),
        reference_longitude = Float64(volume.longitude))

function build_springsteel_grid_spec(sgrid::RRR_Grid;
        reference_latitude::Real, reference_longitude::Real)
    gp = get_springsteel_gridpoints_zyx(sgrid)   # (kDim, jDim, iDim, 3): [z,y,x]
    Z = collect(Float64, gp[:, 1, 1, 1])
    Y = collect(Float64, gp[1, :, 1, 2])
    X = collect(Float64, gp[1, 1, :, 3])
    return GridSpec(shape = :volume_3d,
        reference_latitude  = Float64(reference_latitude),
        reference_longitude = Float64(reference_longitude),
        x_axis = X, y_axis = Y, z_axis = Z)
end

function build_springsteel_grid_spec(sgrid::RR_Grid;
        reference_latitude::Real, reference_longitude::Real)
    gp = get_springsteel_gridpoints_zyx(sgrid)   # (jDim, iDim, 2): [y,x]
    X = collect(Float64, gp[1, :, 2])
    Y = collect(Float64, gp[:, 1, 1])
    return GridSpec(shape = :ppi_2d,
        reference_latitude  = Float64(reference_latitude),
        reference_longitude = Float64(reference_longitude),
        x_axis = X, y_axis = Y, z_axis = [0.0])
end

function build_springsteel_grid_spec(sgrid::R_Grid;
        reference_latitude::Real, reference_longitude::Real)
    # The single R axis is altitude for a column (mirrors grid_radar_column_spectral).
    Z = collect(Float64, get_springsteel_gridpoints_zyx(sgrid))
    return GridSpec(shape = :column_1d,
        reference_latitude  = Float64(reference_latitude),
        reference_longitude = Float64(reference_longitude),
        x_axis = [0.0], y_axis = [0.0], z_axis = Z)
end

function build_springsteel_grid_spec(sgrid::SpringsteelGrid;
        reference_latitude::Real, reference_longitude::Real)
    throw(ArgumentError(
        "build_springsteel_grid_spec: only the Cartesian R/RR/RRR Springsteel " *
        "grids can be gridded by the accumulator engine today (got geometry " *
        "`$(sgrid.params.geometry)`). Cylindrical/spherical gridding " *
        "(per-node gridpoints + metric ROI) is a planned follow-on; such " *
        "grids can still be built, transformed, and written."))
end

# ── Volume-aware spectral entries (unified accumulator path) ──────────────────
# These grid radar *scalar* moments onto a Springsteel grid using the same
# edge-referenced, beamwidth-correct accumulator engine as the regular Cartesian
# path, then output through the Springsteel spectral API. The lower-level
# `radar`-typed spectral functions remain for back-compat.

function grid_radar_volume_spectral(volume::Volume, output_file::AbstractString,
        index_time, sgrid::SpringsteelGrid, p::DaishoParameters;
        heading::Real=-9999.0, institution::String="", source::String="",
        include_derivatives::Bool=false)
    spec = build_springsteel_grid_spec(sgrid, volume)
    acc  = ScalarGridAccumulator(spec, p)                       # scalar-only
    roi  = compute_roi(sgrid; h_factor = p.gridding.horizontal_roi_factor,
                              v_factor = p.gridding.vertical_roi_factor)
    roi3 = (roi[1], length(roi) > 1 ? roi[2] : roi[1])          # (h, v)
    for s in eachindex(volume.sweeps)
        grid_sweep!(acc, volume, s, p; heading = heading, roi = roi3)
    end
    radar_grid = finalize_grid(acc)                            # (n_fields,k,j,i)
    populate_physical!(sgrid, radar_grid, field_index_dict(p);
                       fill_value = p.io.fill_value)
    mask = mask_fill_for_transform!(sgrid)
    spectralTransform!(sgrid)
    gridTransform!(sgrid)
    restore_fill_from_mask!(sgrid, mask)
    write_radar_netcdf(output_file, sgrid, volume, field_index_dict(p);
        ref_lat = spec.reference_latitude, ref_lon = spec.reference_longitude,
        institution = institution, source = source, heading = heading,
        include_derivatives = include_derivatives)
    return sgrid
end

function grid_radar_ppi_spectral(volume::Volume, output_file::AbstractString,
        index_time, sgrid::SpringsteelGrid, p::DaishoParameters;
        heading::Real=-9999.0, institution::String="", source::String="",
        include_derivatives::Bool=false)
    spec = build_springsteel_grid_spec(sgrid, volume)
    acc  = ScalarGridAccumulator(spec, p)
    roi  = compute_roi(sgrid; h_factor = p.gridding.horizontal_roi_factor,
                              v_factor = p.gridding.vertical_roi_factor)
    roi3 = (roi[1], length(roi) > 1 ? roi[2] : roi[1])          # PPI uses h only
    for s in eachindex(volume.sweeps)
        grid_sweep!(acc, volume, s, p; heading = heading, roi = roi3)
    end
    radar_grid = finalize_grid(acc)                            # (n_fields,j,i)
    populate_physical!(sgrid, radar_grid, field_index_dict(p);
                       fill_value = p.io.fill_value)
    mask = mask_fill_for_transform!(sgrid)
    spectralTransform!(sgrid)
    gridTransform!(sgrid)
    restore_fill_from_mask!(sgrid, mask)
    write_radar_netcdf(output_file, sgrid, volume, field_index_dict(p);
        ref_lat = spec.reference_latitude, ref_lon = spec.reference_longitude,
        institution = institution, source = source, heading = heading,
        include_derivatives = include_derivatives)
    return sgrid
end

function grid_radar_column_spectral(volume::Volume, output_file::AbstractString,
        index_time, sgrid::SpringsteelGrid, p::DaishoParameters;
        institution::String="", source::String="", include_derivatives::Bool=false)
    spec = build_springsteel_grid_spec(sgrid, volume)
    acc  = ScalarGridAccumulator(spec, p)
    roi  = compute_roi(sgrid; h_factor = p.gridding.horizontal_roi_factor,
                              v_factor = p.gridding.vertical_roi_factor)
    roi3 = (roi[1], length(roi) > 1 ? roi[2] : roi[1])          # column uses v only
    for s in eachindex(volume.sweeps)
        grid_sweep!(acc, volume, s, p; roi = roi3)
    end
    radar_grid = finalize_grid(acc)                            # (n_fields,k)
    populate_physical!(sgrid, radar_grid, field_index_dict(p);
                       fill_value = p.io.fill_value)
    mask = mask_fill_for_transform!(sgrid)
    spectralTransform!(sgrid)
    gridTransform!(sgrid)
    restore_fill_from_mask!(sgrid, mask)
    write_radar_netcdf(output_file, sgrid, volume, field_index_dict(p);
        ref_lat = spec.reference_latitude, ref_lon = spec.reference_longitude,
        institution = institution, source = source,
        include_derivatives = include_derivatives)
    return sgrid
end
