# GridAccumulator — pre-normalized weighted sums on a chosen grid.
#
# Lets the caller grid one sweep at a time, persist intermediate state to JLD2,
# and combine sweeps from many files later. Replaces the legacy "all sweeps in
# one volume → one gridded NetCDF" path for workflows like airborne P3 and
# multi-Doppler retrieval, while preserving the existing weighting math.

# v2: GridAccumulator carries the io fill_value/undetect sentinels. v1 files
# (no such fields) are intentionally incompatible — regenerate them.
const GRID_ACCUMULATOR_SCHEMA_VERSION = 2

"""
    GridSpec

Geometry-agnostic description of the grid an accumulator lives on. The `shape`
tag determines the dimensionality of the arrays inside the accumulator:

- `:volume_3d`, `:latlon_3d`  → `(n_fields, n_z, n_y, n_x)`
- `:rhi_2d`                   → `(n_fields, n_z, n_r)`
- `:ppi_2d`, `:composite_2d`  → `(n_fields, n_y, n_x)`
- `:column_1d`                → `(n_fields, n_z)`

Axes carry the projected coordinates from the gridding-parameters block.
Unused axes are length-1 placeholders.
"""
Base.@kwdef struct GridSpec
    shape::Symbol
    reference_latitude::Float64
    reference_longitude::Float64
    x_axis::Vector{Float64}
    y_axis::Vector{Float64}
    z_axis::Vector{Float64}
    lat_axis::Union{Vector{Float64},Nothing} = nothing
    lon_axis::Union{Vector{Float64},Nothing} = nothing
    rhi_azimuth::Union{Float64,Nothing}      = nothing
end

"""
    SweepProvenance

Per-sweep tagging metadata recorded each time a sweep is gridded into an
accumulator. Used at finalize/retrieval time to reconstruct line-of-sight and
to audit which inputs contributed to an output grid.
"""
Base.@kwdef struct SweepProvenance
    instrument_name::String              = ""
    scan_name::String                    = ""
    sweep_number::Int                    = -1
    sweep_mode::String                   = ""
    fixed_angle::Float64                 = NaN
    time_start::Union{DateTime,Nothing}  = nothing
    time_end::Union{DateTime,Nothing}    = nothing
    source_file::String                  = ""
    ref_latitude::Float64                = NaN
    ref_longitude::Float64               = NaN
    ref_altitude::Float64                = NaN
end

"""
    FieldAccumulator

Abstract supertype for the streamed gridding accumulators that share the
single-pass 3D traversal ([`_grid_sweep_products_3d!`](@ref)). A concrete
accumulator provides `accumulate_cell!`, `contributing_fields`, and an
`_acc_grid_spec` accessor; the traversal computes the per-gate geometry once
and dispatches the per-cell accumulation on the concrete type.

Concrete subtypes: [`ScalarGridAccumulator`](@ref) (all scalar fields) and
[`WindGridAccumulator`](@ref) (embeds a scalar accumulator plus the
dual-Doppler normal system).
"""
abstract type FieldAccumulator end

"""
    ScalarGridAccumulator

Pre-normalized weighted sums on a chosen grid. Each `grid_sweep!` call adds one
sweep's contribution. `finalize_grid` divides by weights, converts linear→dBZ
where appropriate, and applies the **undetect** / **true missing** sentinels.

Layout: `weighted_sum[field_idx, axis_dims…]`. The trailing axis dims depend
on `grid_spec.shape`.

`GridAccumulator` is a deprecation alias kept for downstream compatibility.
"""
Base.@kwdef struct ScalarGridAccumulator <: FieldAccumulator
    grid_spec::GridSpec
    fields::Vector{String}
    grid_type::Dict{String,Symbol}
    field_folds::Vector{Bool}
    weighted_sum::Array{Float64}
    weight_total::Array{Float64}
    coverage::Array{Int8}
    sweeps::Vector{SweepProvenance}
    schema_version::Int
    fill_value::Float64 = -32768.0   # CF _FillValue  — true missing
    undetect::Float64   =  -9999.0   # ODIM _Undetect — undetect (clear air)
end

# Deprecation alias — a true type alias, so dispatch and `isa` work for both
# names while downstream code (e.g. Sparrow.jl) migrates.
const GridAccumulator = ScalarGridAccumulator

# ── Array shape helpers ──────────────────────────────────────────────────────

"""
    accumulator_dims(grid_spec, n_fields)

Tuple of array dimensions used by `weighted_sum`, `weight_total`, and
`coverage` for a given `grid_spec.shape`. Helps the constructor allocate, and
helps `_check_shape` validate.
"""
function accumulator_dims(g::GridSpec, n_fields::Int)
    nx = length(g.x_axis)
    ny = length(g.y_axis)
    nz = length(g.z_axis)
    if g.shape === :volume_3d || g.shape === :latlon_3d
        return (n_fields, nz, ny, nx)
    elseif g.shape === :rhi_2d
        # x_axis carries range bins; y_axis is unused (length 1).
        return (n_fields, nz, nx)
    elseif g.shape === :ppi_2d || g.shape === :composite_2d
        return (n_fields, ny, nx)
    elseif g.shape === :column_1d
        return (n_fields, nz)
    else
        throw(ArgumentError("accumulator_dims: unsupported shape $(g.shape)"))
    end
end

# ── Constructors ─────────────────────────────────────────────────────────────

"""
    ScalarGridAccumulator(grid_spec, fields, grid_type; field_folds)

Allocate an empty accumulator on the given grid for an explicit list of fields.
`field_folds` defaults to all-false; supply `true` per-field for radial
velocities and other folding quantities to make `merge_accumulators!` refuse to
combine them across distinct sweeps. `fill_value` / `undetect` are the
true-missing / undetect output sentinels (defaults match the `[io]` defaults);
`finalize_grid` emits exactly these.
"""
function ScalarGridAccumulator(grid_spec::GridSpec,
                          fields::Vector{String},
                          grid_type::Dict{String,Symbol};
                          field_folds::Vector{Bool} = fill(false, length(fields)),
                          fill_value::Float64 = -32768.0,
                          undetect::Float64 = -9999.0)
    length(field_folds) == length(fields) ||
        throw(ArgumentError("ScalarGridAccumulator: field_folds must have one entry per field"))
    for f in fields
        haskey(grid_type, f) ||
            throw(ArgumentError("ScalarGridAccumulator: grid_type missing entry for $f"))
    end
    dims = accumulator_dims(grid_spec, length(fields))
    return ScalarGridAccumulator(
        grid_spec       = grid_spec,
        fields          = copy(fields),
        grid_type       = copy(grid_type),
        field_folds     = copy(field_folds),
        weighted_sum    = zeros(Float64, dims),
        weight_total    = zeros(Float64, dims),
        coverage        = zeros(Int8, dims),
        sweeps          = SweepProvenance[],
        schema_version  = GRID_ACCUMULATOR_SCHEMA_VERSION,
        fill_value      = fill_value,
        undetect        = undetect,
    )
end

"""
    ScalarGridAccumulator(grid_spec, p::DaishoParameters)

Allocate an accumulator from a `DaishoParameters`. Column order is the field
names sorted by name (order-independent of the source TOML, matching
[`field_index_dict`](@ref) so the accumulator columns and the writer's index
map agree); interpolation is derived from each field's tags via
[`interp_of`](@ref). Field-folds defaults to all false; the caller can override
with the explicit constructor when gridding a field-folds quantity (e.g.
`VEL`).
"""
function ScalarGridAccumulator(grid_spec::GridSpec, p::DaishoParameters)
    ordered = _ordered_fields(p)
    return ScalarGridAccumulator(grid_spec,
        [fs.name for fs in ordered],
        Dict{String,Symbol}(fs.name => interp_of(fs) for fs in ordered);
        fill_value = p.io.fill_value, undetect = p.io.undetect)
end

# ── JLD2 IO ──────────────────────────────────────────────────────────────────

"""
    save_accumulator(path, accum)

Persist a `GridAccumulator` to JLD2. The full struct is written; reload with
`load_accumulator`.
"""
function save_accumulator(path::AbstractString, accum::GridAccumulator)
    JLD2.jldopen(path, "w") do f
        f["accumulator"] = accum
    end
    return path
end

"""
    load_accumulator(path) -> GridAccumulator

Read a `GridAccumulator` saved by `save_accumulator`. Raises if the schema
version on disk does not match `GRID_ACCUMULATOR_SCHEMA_VERSION`.
"""
function load_accumulator(path::AbstractString)
    accum = JLD2.jldopen(path, "r") do f
        f["accumulator"]
    end
    accum.schema_version == GRID_ACCUMULATOR_SCHEMA_VERSION ||
        throw(ArgumentError("load_accumulator: schema version $(accum.schema_version) on $(path); expected $(GRID_ACCUMULATOR_SCHEMA_VERSION)"))
    return accum
end

# ── Merge ────────────────────────────────────────────────────────────────────

function _grid_spec_equal(a::GridSpec, b::GridSpec)
    a.shape === b.shape || return false
    a.reference_latitude == b.reference_latitude || return false
    a.reference_longitude == b.reference_longitude || return false
    a.x_axis == b.x_axis || return false
    a.y_axis == b.y_axis || return false
    a.z_axis == b.z_axis || return false
    a.lat_axis == b.lat_axis || return false
    a.lon_axis == b.lon_axis || return false
    a.rhi_azimuth == b.rhi_azimuth || return false
    return true
end

"""
    merge_accumulators!(dst, src) -> dst

Combine `src` into `dst` in place. Strict compatibility is required: identical
`grid_spec`, identical `fields`, identical `grid_type`, identical
`field_folds`. No silent coercion. For scalar (`field_folds == false`) fields
the merge sums weighted sums and weight totals; for `:nearest`-mode fields the
per-cell entry with the larger `weight_total` is kept. For `field_folds ==
true` fields the merge raises — these must be processed per-sweep by the wind
retrieval downstream, not summed at the grid level.
"""
function merge_accumulators!(dst::GridAccumulator, src::GridAccumulator)
    _grid_spec_equal(dst.grid_spec, src.grid_spec) ||
        throw(ArgumentError("merge_accumulators!: grid_spec mismatch"))
    dst.fields == src.fields ||
        throw(ArgumentError("merge_accumulators!: fields mismatch"))
    dst.grid_type == src.grid_type ||
        throw(ArgumentError("merge_accumulators!: grid_type mismatch"))
    dst.field_folds == src.field_folds ||
        throw(ArgumentError("merge_accumulators!: field_folds mismatch"))
    dst.fill_value == src.fill_value ||
        throw(ArgumentError("merge_accumulators!: fill_value mismatch " *
            "($(dst.fill_value) vs $(src.fill_value)) — accumulators built " *
            "under different [io] sentinel conventions cannot be merged"))
    dst.undetect == src.undetect ||
        throw(ArgumentError("merge_accumulators!: undetect mismatch " *
            "($(dst.undetect) vs $(src.undetect)) — accumulators built " *
            "under different [io] sentinel conventions cannot be merged"))

    n_fields = length(dst.fields)
    trailing = ntuple(d -> Colon(), ndims(dst.weighted_sum) - 1)
    for m in 1:n_fields
        if dst.field_folds[m]
            name = dst.fields[m]
            throw(ArgumentError(
                "merge_accumulators!: cannot merge field-folds field $(name) across distinct sweeps; per-sweep accumulators must be processed by the wind retrieval directly"))
        end
        mode = dst.grid_type[dst.fields[m]]
        if mode === :nearest
            # Per-cell: keep entry with larger weight_total.
            dst_w = view(dst.weight_total, m, trailing...)
            src_w = view(src.weight_total, m, trailing...)
            dst_s = view(dst.weighted_sum, m, trailing...)
            src_s = view(src.weighted_sum, m, trailing...)
            @inbounds for i in eachindex(dst_w)
                if src_w[i] > dst_w[i]
                    dst_w[i] = src_w[i]
                    dst_s[i] = src_s[i]
                end
            end
        else
            @views dst.weighted_sum[m, trailing...] .+= src.weighted_sum[m, trailing...]
            @views dst.weight_total[m, trailing...] .+= src.weight_total[m, trailing...]
        end
        @views dst.coverage[m, trailing...] .= max.(
            dst.coverage[m, trailing...], src.coverage[m, trailing...])
    end

    append!(dst.sweeps, src.sweeps)
    return dst
end

# ── Per-sweep geometry helpers ───────────────────────────────────────────────

# Resolve the per-ray (lat, lon, alt) for a sweep. Mobile platforms have a
# Georeference; stationary platforms broadcast the volume-level reference.
function _sweep_ray_positions(sweep::SweepGroup,
                              ref_latitude::Float64,
                              ref_longitude::Float64,
                              ref_altitude::Float64)
    n = n_rays(sweep)
    if sweep.georeference !== nothing && length(sweep.georeference.latitude) == n
        return (collect(Float64, sweep.georeference.latitude),
                collect(Float64, sweep.georeference.longitude),
                collect(Float64, sweep.georeference.altitude))
    else
        return (fill(ref_latitude, n), fill(ref_longitude, n), fill(ref_altitude, n))
    end
end

# Per-sweep analog of `radar_arrays`. Returns `(grid_origin, radar_zyx, beams)`
# where `radar_zyx[idx]` and `beams[idx, :]` use the flat
# `idx = (ray - 1) * n_gates + gate` layout (column-major, matching legacy).
function _sweep_zyx_and_beams(sweep::SweepGroup,
                               ref_latitude::Float64,
                               ref_longitude::Float64,
                               ref_altitude::Float64,
                               projection)
    n_gates_s = length(sweep.range)
    n_rays_s  = n_rays(sweep)

    lats, lons, alts = _sweep_ray_positions(sweep, ref_latitude, ref_longitude, ref_altitude)
    radar_loc = convert.(projection, LatLon.(lats, lons))
    beam_origin = [ [alts[i],
                     Float64(ustrip(radar_loc[i].y)),
                     Float64(ustrip(radar_loc[i].x))]  for i in eachindex(radar_loc) ]
    # Layout: radar_zyx[gate, ray] = beam_origin[ray]. Column-major flat index
    # `(gate, ray) → idx = (ray-1)*n_gates + gate` matches the legacy bridge.
    radar_zyx = [ zyx for _ in sweep.range, zyx in beam_origin ]

    beams = [ (deg2rad(sweep.azimuth[j]),
               deg2rad(sweep.elevation[j]),
               sweep.range[i],
               beam_height(sweep.range[i], sweep.elevation[j], alts[j]))
              for i in eachindex(sweep.range), j in eachindex(sweep.elevation) ]
    beams = [ beams[i][k] for i in eachindex(beams), k in 1:4 ]

    grid_origin = convert(projection, LatLon(ref_latitude, ref_longitude))
    return grid_origin, radar_zyx, beams, n_gates_s, n_rays_s
end

# Horizontal-plane BallTree on (y, x) surface positions of every gate.
function _sweep_balltree_yx(radar_zyx::AbstractArray, beams::AbstractArray)
    n = size(beams, 1)
    gate_yx = zeros(Float64, 2, n)
    for i in 1:n
        surface_range = Reff * asin(beams[i, 3] * cos(beams[i, 2]) / (Reff + beams[i, 4]))
        gate_yx[1, i] = radar_zyx[i][2] + surface_range * cos(beams[i, 1])
        gate_yx[2, i] = radar_zyx[i][3] + surface_range * sin(beams[i, 1])
    end
    return BallTree(gate_yx)
end

# 1D radial BallTree on surface distance from the reference origin.
function _sweep_balltree_r(radar_zyx::AbstractArray, beams::AbstractArray)
    n = size(beams, 1)
    gate_r = zeros(Float64, 1, n)
    for i in 1:n
        surface_range = Reff * asin(beams[i, 3] * cos(beams[i, 2]) / (Reff + beams[i, 4]))
        y = radar_zyx[i][2] + surface_range * cos(beams[i, 1])
        x = radar_zyx[i][3] + surface_range * sin(beams[i, 1])
        gate_r[i] = sqrt(x^2 + y^2)
    end
    return BallTree(gate_r)
end

# Return the gate value at (ray, gate) for a named field, with NaN treated as
# missing. Returns `missing` if the sweep doesn't carry the field.
@inline function _gate_value(sweep::SweepGroup, field_name::String,
                              ray::Int, gate::Int)
    haskey(sweep.fields, field_name) || return missing
    v = sweep.fields[field_name].data[ray, gate]
    if ismissing(v)
        return missing
    elseif isa(v, AbstractFloat) && isnan(v)
        return missing
    else
        return Float64(v)
    end
end

# Flat-gate-index → (ray, gate_in_ray). Matches the layout produced by
# `_sweep_zyx_and_beams`.
@inline _ray_of(flat::Int, n_gates::Int)        = ((flat - 1) ÷ n_gates) + 1
@inline _gate_in_ray(flat::Int, n_gates::Int)   = ((flat - 1) % n_gates) + 1

# Per-gate geometry + interpolation weight at one grid point. Computed once per
# contributing gate in the shared traversal (`_grid_sweep_products_3d!`) and
# reused by every consumer (scalar accumulation and the wind normal system) so
# all products see identical effective angles and weights.
#
# `gridpt_az` is the effective azimuth (clockwise from +y / true north) and
# `gridpt_el` the refraction-corrected elevation from the gate's radar origin to
# the grid point — the line-of-sight direction where the wind is estimated.
# `total_weight = range_weight · angle_weight` is the existing Gaussian
# beam-pattern × range interpolation weight. A return of `total_weight ≤ 0`
# (including the `|sine_h| ≥ 1` non-physical case) signals the gate does not
# contribute and should be skipped.
@inline function _gate_grid_geometry(grid_z::Float64, yx_point,
                                     radar_zyx, beams, g_flat::Int,
                                     horizontal_roi::Float64, vertical_roi::Float64,
                                     power_threshold::Float64)
    # Refraction-corrected height angle.
    dz = grid_z - radar_zyx[g_flat][1]
    r  = beams[g_flat, 3]
    sine_h = ((dz + Reff)^2 - r^2 - Reff^2) / (2 * r * Reff)
    abs(sine_h) < 1.0 || return (NaN, NaN, 0.0)
    gridpt_el = asin(sine_h)

    # Effective azimuth from gate origin to gridpoint.
    dx = yx_point[2] - radar_zyx[g_flat][3]
    dy = yx_point[1] - radar_zyx[g_flat][2]
    gridpt_az = (pi / 2.0) - atan(dy, dx)
    gridpt_az < 0 && (gridpt_az += 2 * pi)

    angle_diff = spherical_angle([beams[g_flat, 1], beams[g_flat, 2]],
                                  [gridpt_az, gridpt_el])
    angle_weight = exp(-angle_diff * 79.43)
    angle_weight < power_threshold && (angle_weight = 0.0)

    gridpt_r = sin(sqrt(dx^2 + dy^2) / Reff) * (Reff + dz) / cos(gridpt_el)
    range_weight = gridpt_r / r
    if abs(gridpt_r - r) > horizontal_roi || abs(gridpt_r - r) > vertical_roi
        range_weight = 0.0
    end
    total_weight = range_weight * angle_weight
    return (gridpt_az, gridpt_el, total_weight)
end

# ── grid_sweep! dispatcher ───────────────────────────────────────────────────

"""
    grid_sweep!(accum, sweep::SweepGroup, p::DaishoParameters;
                ref_latitude, ref_longitude, ref_altitude,
                source_file="", heading=-9999.0,
                instrument_name="", scan_name="")

Add one sweep's contribution to `accum`. The sweep is gridded with the
configured ROI and the existing weighting math (Gaussian beam-pattern,
range, refraction-corrected height angle). Linear-mode fields accumulate in
linear units; `finalize_grid` converts back to dBZ.

`ref_latitude` / `ref_longitude` / `ref_altitude` is the reference position
the sweep was scanned from. For stationary radars supply the volume's
lat/lon/alt; for mobile radars supply the first ray's georeference (or any
representative position). Per-ray georeference, when present on the
`SweepGroup`, takes precedence.

A `SweepProvenance` entry is appended to `accum.sweeps`.
"""
function grid_sweep!(accum::FieldAccumulator, sweep::SweepGroup,
                     p::DaishoParameters;
                     ref_latitude::Float64,
                     ref_longitude::Float64,
                     ref_altitude::Float64,
                     source_file::AbstractString = "",
                     heading::Real = -9999.0,
                     instrument_name::AbstractString = "",
                     scan_name::AbstractString = "")
    shape = _acc_grid_spec(accum).shape
    if shape === :volume_3d || shape === :latlon_3d
        _grid_sweep_products_3d!(accum, sweep, p, ref_latitude, ref_longitude, ref_altitude)
    elseif shape === :rhi_2d
        _grid_sweep_rhi_2d!(accum, sweep, p, ref_latitude, ref_longitude, ref_altitude)
    elseif shape === :ppi_2d
        _grid_sweep_ppi_2d!(accum, sweep, p, ref_latitude, ref_longitude, ref_altitude)
    elseif shape === :composite_2d
        _grid_sweep_composite_2d!(accum, sweep, p, ref_latitude, ref_longitude, ref_altitude)
    elseif shape === :column_1d
        _grid_sweep_column_1d!(accum, sweep, p, ref_latitude, ref_longitude, ref_altitude)
    else
        throw(ArgumentError("grid_sweep!: unsupported shape $shape"))
    end
    push!(_acc_sweeps(accum), SweepProvenance(
        instrument_name = String(instrument_name),
        scan_name = String(scan_name),
        sweep_number = sweep.sweep_number,
        sweep_mode = sweep.sweep_mode,
        fixed_angle = sweep.fixed_angle,
        time_start = isempty(sweep.time) ? nothing : sweep.time[1],
        time_end   = isempty(sweep.time) ? nothing : sweep.time[end],
        source_file = String(source_file),
        ref_latitude = ref_latitude,
        ref_longitude = ref_longitude,
        ref_altitude = ref_altitude,
    ))
    return accum
end

"""
    grid_sweep!(accum, volume::Volume, sweep_index::Int, p; heading=-9999.0,
                source_file="")

Volume convenience overload. Resolves the per-sweep reference position from
the sweep's georeference when present (mobile), else from the volume's
stationary `latitude`/`longitude`/`altitude`.
"""
function grid_sweep!(accum::FieldAccumulator, volume::Volume, sweep_index::Int,
                     p::DaishoParameters; heading::Real = -9999.0,
                     source_file::AbstractString = "")
    sweep = volume.sweeps[sweep_index]
    if sweep.georeference !== nothing && !isempty(sweep.georeference.latitude)
        ref_lat = sweep.georeference.latitude[1]
        ref_lon = sweep.georeference.longitude[1]
        ref_alt = sweep.georeference.altitude[1]
    else
        ref_lat = volume.latitude
        ref_lon = volume.longitude
        ref_alt = volume.altitude
    end
    return grid_sweep!(accum, sweep, p;
        ref_latitude = Float64(ref_lat),
        ref_longitude = Float64(ref_lon),
        ref_altitude = Float64(ref_alt),
        source_file = source_file,
        heading = heading,
        instrument_name = volume.instrument_name,
        scan_name = volume.scan_name)
end

# ── 3D volume / latlon worker ────────────────────────────────────────────────

# Materialize a `(zdim, ydim, xdim, 3)` gridpoints array from the GridSpec.
# For volume_3d this is the cartesian product of the axes. For latlon_3d,
# x/y meters are computed from lat/lon via the Transverse Mercator projection.
function _materialize_gridpoints_3d(g::GridSpec, projection, grid_origin)
    nx = length(g.x_axis)
    ny = length(g.y_axis)
    nz = length(g.z_axis)
    gridpoints = Array{Float64}(undef, nz, ny, nx, 3)
    if g.shape === :latlon_3d
        # lat_axis / lon_axis are the geographic axes; project per (lat, lon).
        lat_axis = g.lat_axis === nothing ? error("latlon_3d GridSpec missing lat_axis") : g.lat_axis
        lon_axis = g.lon_axis === nothing ? error("latlon_3d GridSpec missing lon_axis") : g.lon_axis
        for j in 1:ny, i in 1:nx
            cartTM = convert(projection, LatLon(lat_axis[j], lon_axis[i]))
            ycoord = ustrip(cartTM.y) - ustrip(grid_origin.y)
            xcoord = ustrip(cartTM.x) - ustrip(grid_origin.x)
            for k in 1:nz
                gridpoints[k, j, i, 1] = g.z_axis[k]
                gridpoints[k, j, i, 2] = ycoord
                gridpoints[k, j, i, 3] = xcoord
            end
        end
    else  # :volume_3d
        for k in 1:nz, j in 1:ny, i in 1:nx
            gridpoints[k, j, i, 1] = g.z_axis[k]
            gridpoints[k, j, i, 2] = g.y_axis[j]
            gridpoints[k, j, i, 3] = g.x_axis[i]
        end
    end
    return gridpoints
end

# ── Unified single-pass product accumulator interface ────────────────────────
#
# The shared 3D traversal (`_grid_sweep_products_3d!`) computes the per-gate
# geometry+weight **once** per cell and hands it to `accumulate_cell!`, which is
# dispatched on the concrete accumulator type. This lets the scalar grid and the
# dual-Doppler wind solve be driven by one geometry pass instead of two.

# Resolved per-sweep role fields (define_scanned / define_detection), looked up
# once per sweep rather than once per cell.
struct SweepKeys
    missing_key::String   # :define_scanned — presence proves the gate was scanned
    valid_key::String     # :define_detection — presence proves a detectable echo
end

# Per-gate precomputed contribution: the geometry+weight at one grid cell. The
# geometry is field-independent, so a single `GateContribution` serves every
# consumer (scalar accumulation and the wind normal system). `beam_z` is the
# gate's beam height, carried so a consumer can re-apply a vertical-range filter
# without recomputing geometry. All fields are isbits ⇒ no per-cell heap churn.
struct GateContribution
    g_flat::Int
    ray::Int
    gate::Int
    az::Float64
    el::Float64
    w::Float64
    beam_z::Float64
end

# Opaque-ish Cartesian cell handle passed to `accumulate_cell!`. `(k, j, i)`
# index the trailing accumulator dims; `grid_z`/`eff_v_roi` carry the scalars a
# consumer needs for vertical filtering. A future Springsteel node provider would
# supply its own cell handle (linear node index) with its own `accumulate_cell!`.
struct CartesianCell
    k::Int
    j::Int
    i::Int
    grid_z::Float64
    eff_v_roi::Float64
    n_gates::Int
end

# ── FieldAccumulator interface (ScalarGridAccumulator) ─────────────────────

"""
    contributing_fields(acc) -> Vector{String}

The scalar fields whose gate values the accumulator consumes. Drives nothing in
the traversal directly (geometry is field-independent) but documents which
fields a consumer reads; a `WindGridAccumulator` exposes its embedded scalar's
fields.
"""
contributing_fields(acc::ScalarGridAccumulator) = acc.fields

# GridSpec / provenance accessors — type-generic so the traversal/writer work for
# both the scalar and (embedding) wind accumulators. A `WindGridAccumulator`
# forwards both to its embedded scalar (its single source of truth).
_acc_grid_spec(acc::ScalarGridAccumulator) = acc.grid_spec
_acc_sweeps(acc::ScalarGridAccumulator)    = acc.sweeps

# Fields whose presence at a gate triggers a geometry computation (and thus a
# `GateContribution` slot). Returning exactly the scalar `valid_key` reproduces
# the legacy worker, which computed geometry only for define_detection gates —
# no perf regression. (Wind adds its velocity field; see the WindGridAccumulator
# method.)
_geometry_trigger_fields(acc::ScalarGridAccumulator, keys::SweepKeys) =
    (keys.valid_key,)

# ── Grid-provider seam (developer note: Springsteel quadrature-node grids) ───
#
# The unified traversal is factored so the grid-point *provider* (where the
# analysis points live and how the accumulator planes are indexed) is separable
# from the per-cell accumulation math. Three pieces span the seam:
#
#   1. `GridSpec.shape` selects the provider. `grid_sweep!` dispatches on it;
#      `:volume_3d`/`:latlon_3d` route to the Cartesian provider+traversal below.
#   2. A per-cell handle (`CartesianCell` here) carries the opaque cell index plus
#      whatever scalars a consumer needs (here `grid_z`/`eff_v_roi`). It is passed
#      to `accumulate_cell!` and is the *only* place the grid topology appears.
#   3. `accumulate_cell!(acc, cell, sweep, scanned_gates, contribs, keys, p)` is
#      dispatched on both the accumulator type *and* the cell-handle type, so a
#      new provider's cells select new methods without touching the existing ones.
#
# The accumulator structs are already topology-agnostic: `weighted_sum` /
# `weight_total` / `coverage` (and the wind planes) are plain `Array{Float64}`,
# shaped by `accumulator_dims` / `wind_accumulator_dims` from the `GridSpec`.
# Nothing here hardcodes the `(nz, ny, nx)` lattice except the `CartesianCell`
# methods and the Cartesian 3D writer — both of which a node provider simply adds
# alongside.
#
# A Springsteel quadrature-node provider (deferred; its own follow-on plan) plugs
# in as:
#   • a node `GridSpec.shape` (e.g. `:springsteel_nodes`) and `accumulator_dims`/
#     `wind_accumulator_dims` returning `(n_fields, n_nodes)` / `(n_nodes,)`;
#   • a node traversal `_grid_sweep_products_nodes!` that materializes the
#     quadrature-node coordinates from the GridSpec, queries the BallTree per node,
#     and builds a `NodeCell{node_index, …}` + per-node `GateContribution`s
#     (reusing `_gate_grid_geometry`, `SweepKeys`, `_geometry_trigger_fields`);
#   • `accumulate_cell!(acc, ::NodeCell, …)` methods (the per-cell coverage /
#     weighted-accumulation / rank-1 wind math is identical — only the plane
#     indexing changes from `[m, k, j, i]` to `[m, n]`);
#   • a `grid_sweep!` shape branch routing to the node traversal, and a node-aware
#     `write_grid_products` path.
# No part of the current Cartesian path needs to change to add it.

function _grid_sweep_products_3d!(acc::FieldAccumulator, sweep::SweepGroup,
                          p::DaishoParameters,
                          ref_latitude::Float64, ref_longitude::Float64,
                          ref_altitude::Float64)
    g  = _acc_grid_spec(acc)
    gd = p.gridding
    keys = SweepKeys(
        field_with_tag(p, :define_scanned;   for_op="grid_sweep! (accumulator path)"),
        field_with_tag(p, :define_detection; for_op="grid_sweep! (accumulator path)"))
    triggers = _geometry_trigger_fields(acc, keys)

    TM = CoordRefSystems.shift(TransverseMercator{1.0, g.reference_latitude, WGS84Latest},
                                lonₒ = g.reference_longitude)
    grid_origin, radar_zyx, beams, n_gates_s, n_rays_s =
        _sweep_zyx_and_beams(sweep, ref_latitude, ref_longitude, ref_altitude, TM)
    balltree = _sweep_balltree_yx(radar_zyx, beams)
    gridpoints = _materialize_gridpoints_3d(g, TM, grid_origin)

    nx = length(g.x_axis)
    ny = length(g.y_axis)
    nz = length(g.z_axis)

    # ROI mirrors the legacy per-driver derivations (xincr * 0.75 etc.). For
    # the accumulator path we read these from the grid axes directly.
    horizontal_roi = if g.shape === :latlon_3d
        # Convert degincr → meters using the SAMURAI approximation centered at
        # the reference latitude. Matches the legacy `grid_radar_latlon_volume`
        # ROI formula.
        latrad = g.reference_latitude * pi / 180.0
        fac_lat = 111.13209 - 0.56605 * cos(2.0 * latrad)
        fac_lon = 111.41513 * cos(latrad)
        deg_km = sqrt(fac_lat^2 + fac_lon^2)
        degincr = ny >= 2 ? (g.lat_axis[2] - g.lat_axis[1]) :
                  (nx >= 2 ? (g.lon_axis[2] - g.lon_axis[1]) : 0.01)
        deg_km * 1000.0 * degincr * 0.75
    else
        xincr = nx >= 2 ? (g.x_axis[2] - g.x_axis[1]) : 0.0
        xincr * 0.75
    end
    zincr = nz >= 2 ? (g.z_axis[2] - g.z_axis[1]) : 0.0
    vertical_roi = zincr * 0.75

    beam_inflation  = gd.beam_inflation
    power_threshold = gd.power_threshold

    Threads.@threads for ii in CartesianIndices((ny, nx))
        j_y, i_x = ii.I
        yx_point = [gridpoints[1, j_y, i_x, 2], gridpoints[1, j_y, i_x, 3]]

        eff_h_roi = horizontal_roi
        eff_v_roi = vertical_roi
        if beam_inflation > 0.0
            origin_dist = euclidean(yx_point, [0.0, 0.0])
            eff_h_roi = max(beam_inflation * origin_dist, horizontal_roi)
            eff_v_roi = max(beam_inflation * origin_dist, vertical_roi)
        end
        gates = inrange(balltree, yx_point, eff_h_roi)
        isempty(gates) && continue

        # Per-column scratch reused across z (each thread owns distinct columns).
        scanned_gates = Int[]               # vertically-filtered, for coverage=1
        contribs = GateContribution[]       # geometry computed once per gate

        for k_z in 1:nz
            grid_z = gridpoints[k_z, j_y, i_x, 1]
            empty!(scanned_gates)
            empty!(contribs)

            for g_flat in gates
                # Vertically-filtered set drives the scalar "any scanned gate"
                # coverage presence check (no geometry needed).
                if abs(beams[g_flat, 4] - grid_z) <= eff_v_roi
                    push!(scanned_gates, g_flat)
                end
                ray  = _ray_of(g_flat, n_gates_s)
                gate = _gate_in_ray(g_flat, n_gates_s)
                # Compute geometry only for gates a consumer needs (legacy parity:
                # scalar ⇒ define_detection-present gates).
                triggered = false
                for tf in triggers
                    if !ismissing(_gate_value(sweep, tf, ray, gate))
                        triggered = true
                        break
                    end
                end
                triggered || continue
                az, el, w = _gate_grid_geometry(grid_z, yx_point, radar_zyx, beams,
                    g_flat, horizontal_roi, vertical_roi, power_threshold)
                w > 0.0 || continue
                push!(contribs, GateContribution(g_flat, ray, gate, az, el, w,
                    beams[g_flat, 4]))
            end

            cell = CartesianCell(k_z, j_y, i_x, grid_z, eff_v_roi, n_gates_s)
            accumulate_cell!(acc, cell, sweep, scanned_gates, contribs, keys, p)
        end
    end
    return acc
end

"""
    accumulate_cell!(acc::ScalarGridAccumulator, cell, sweep, scanned_gates,
                     contribs, keys, p) -> acc

Per-cell scalar accumulation, lifted from the legacy `_grid_sweep_3d!` body.
`scanned_gates` are the vertically-filtered in-range gates (the coverage=1
presence set); `contribs` carry the precomputed geometry+weight for the
define_detection-present gates. Coverage semantics are preserved exactly:
coverage=1 where any scanned gate is `define_scanned`-present, coverage=2 where a
`define_detection` gate with positive weight accumulates a field.
"""
function accumulate_cell!(acc::ScalarGridAccumulator, cell::CartesianCell,
                          sweep::SweepGroup, scanned_gates::Vector{Int},
                          contribs::Vector{GateContribution}, keys::SweepKeys,
                          p::DaishoParameters)
    k, j, i = cell.k, cell.j, cell.i
    n_fields = length(acc.fields)

    # coverage=1: any in-range (vertically-filtered) gate that was scanned.
    any_scanned = false
    for g_flat in scanned_gates
        ray  = _ray_of(g_flat, cell.n_gates)
        gate = _gate_in_ray(g_flat, cell.n_gates)
        if !ismissing(_gate_value(sweep, keys.missing_key, ray, gate))
            any_scanned = true
            break
        end
    end
    any_scanned || return acc
    @inbounds for m in 1:n_fields
        if acc.coverage[m, k, j, i] == Int8(0)
            acc.coverage[m, k, j, i] = Int8(1)
        end
    end

    # coverage=2: per-field weighted accumulation at define_detection gates.
    for c in contribs
        ismissing(_gate_value(sweep, keys.valid_key, c.ray, c.gate)) && continue
        total_weight = c.w
        @inbounds for m in 1:n_fields
            fname = acc.fields[m]
            v = _gate_value(sweep, fname, c.ray, c.gate)
            ismissing(v) && continue
            mode = acc.grid_type[fname]
            acc.coverage[m, k, j, i] = Int8(2)
            if mode === :linear
                linear_z = 10.0 ^ (v / 10.0)
                acc.weighted_sum[m, k, j, i] += total_weight * linear_z
                acc.weight_total[m, k, j, i] += total_weight
            elseif mode === :nearest
                if total_weight > acc.weight_total[m, k, j, i]
                    acc.weighted_sum[m, k, j, i] = v
                    acc.weight_total[m, k, j, i] = total_weight
                end
            else
                acc.weighted_sum[m, k, j, i] += total_weight * v
                acc.weight_total[m, k, j, i] += total_weight
            end
        end
    end
    return acc
end

# ── RHI worker (2D, range × z) ───────────────────────────────────────────────

function _grid_sweep_rhi_2d!(accum::GridAccumulator, sweep::SweepGroup,
                              p::DaishoParameters,
                              ref_latitude::Float64, ref_longitude::Float64,
                              ref_altitude::Float64)
    g  = accum.grid_spec
    gd = p.gridding
    missing_key = field_with_tag(p, :define_scanned;   for_op="grid_sweep! (accumulator path)")
    valid_key   = field_with_tag(p, :define_detection; for_op="grid_sweep! (accumulator path)")

    TM = CoordRefSystems.shift(TransverseMercator{1.0, g.reference_latitude, WGS84Latest},
                                lonₒ = g.reference_longitude)
    grid_origin, radar_zyx, beams, n_gates_s, _ =
        _sweep_zyx_and_beams(sweep, ref_latitude, ref_longitude, ref_altitude, TM)
    balltree = _sweep_balltree_r(radar_zyx, beams)

    nz = length(g.z_axis)
    nr = length(g.x_axis)   # range bins stored in x_axis for the RHI shape
    n_fields = length(accum.fields)

    rincr = nr >= 2 ? (g.x_axis[2] - g.x_axis[1]) : 0.0
    zincr = nz >= 2 ? (g.z_axis[2] - g.z_axis[1]) : 0.0
    horizontal_roi = rincr * 0.75
    vertical_roi   = zincr * 0.75

    beam_inflation  = gd.beam_inflation
    power_threshold = gd.power_threshold

    # Use rhi_azimuth if explicitly supplied, else fall back to sweep.azimuth[1].
    az_rhi = if g.rhi_azimuth !== nothing
        deg2rad(g.rhi_azimuth)
    else
        deg2rad(sweep.azimuth[1])
    end

    Threads.@threads for i_r in 1:nr
        r_point = g.x_axis[i_r]
        y_point = r_point * cos(az_rhi)
        x_point = r_point * sin(az_rhi)

        eff_h_roi = horizontal_roi
        eff_v_roi = vertical_roi
        origin_dist = euclidean(r_point, [0.0])
        if beam_inflation > 0.0
            eff_v_roi = max(beam_inflation * origin_dist, vertical_roi)
        end
        gates = inrange(balltree, [origin_dist], eff_h_roi)
        isempty(gates) && continue

        for k_z in 1:nz
            grid_z = g.z_axis[k_z]

            any_scanned = false
            for g_flat in gates
                if abs(beams[g_flat, 4] - grid_z) > eff_v_roi
                    continue
                end
                ray  = _ray_of(g_flat, n_gates_s)
                gate = _gate_in_ray(g_flat, n_gates_s)
                if !ismissing(_gate_value(sweep, missing_key, ray, gate))
                    any_scanned = true
                    break
                end
            end
            if any_scanned
                @inbounds for m in 1:n_fields
                    if accum.coverage[m, k_z, i_r] == Int8(0)
                        accum.coverage[m, k_z, i_r] = Int8(1)
                    end
                end
            else
                continue
            end

            for g_flat in gates
                ray  = _ray_of(g_flat, n_gates_s)
                gate_in = _gate_in_ray(g_flat, n_gates_s)
                vk = _gate_value(sweep, valid_key, ray, gate_in)
                ismissing(vk) && continue

                dz = grid_z - radar_zyx[g_flat][1]
                r  = beams[g_flat, 3]
                sine_h = ((dz + Reff)^2 - r^2 - Reff^2) / (2 * r * Reff)
                abs(sine_h) < 1.0 || continue
                gridpt_el = asin(sine_h)

                dx = x_point - radar_zyx[g_flat][3]
                dy = y_point - radar_zyx[g_flat][2]
                angle_diff = spherical_angle([beams[g_flat, 1], beams[g_flat, 2]],
                                              [beams[g_flat, 1], gridpt_el])
                angle_weight = exp(-angle_diff * 79.43)
                angle_weight < power_threshold && (angle_weight = 0.0)

                gridpt_r = sin(sqrt(dx^2 + dy^2) / Reff) * (Reff + dz) / cos(gridpt_el)
                range_weight = gridpt_r / r
                if abs(gridpt_r - r) > horizontal_roi || abs(gridpt_r - r) > vertical_roi
                    range_weight = 0.0
                end
                total_weight = range_weight * angle_weight
                total_weight > 0.0 || continue

                @inbounds for m in 1:n_fields
                    fname = accum.fields[m]
                    v = _gate_value(sweep, fname, ray, gate_in)
                    ismissing(v) && continue
                    mode = accum.grid_type[fname]
                    accum.coverage[m, k_z, i_r] = Int8(2)
                    if mode === :linear
                        linear_z = 10.0 ^ (v / 10.0)
                        accum.weighted_sum[m, k_z, i_r] += total_weight * linear_z
                        accum.weight_total[m, k_z, i_r] += total_weight
                    elseif mode === :nearest
                        if total_weight > accum.weight_total[m, k_z, i_r]
                            accum.weighted_sum[m, k_z, i_r] = v
                            accum.weight_total[m, k_z, i_r] = total_weight
                        end
                    else
                        accum.weighted_sum[m, k_z, i_r] += total_weight * v
                        accum.weight_total[m, k_z, i_r] += total_weight
                    end
                end
            end
        end
    end
    return accum
end

# ── PPI worker (2D, y × x) ───────────────────────────────────────────────────

function _grid_sweep_ppi_2d!(accum::GridAccumulator, sweep::SweepGroup,
                              p::DaishoParameters,
                              ref_latitude::Float64, ref_longitude::Float64,
                              ref_altitude::Float64)
    g  = accum.grid_spec
    gd = p.gridding
    missing_key = field_with_tag(p, :define_scanned;   for_op="grid_sweep! (accumulator path)")
    valid_key   = field_with_tag(p, :define_detection; for_op="grid_sweep! (accumulator path)")

    TM = CoordRefSystems.shift(TransverseMercator{1.0, g.reference_latitude, WGS84Latest},
                                lonₒ = g.reference_longitude)
    _, radar_zyx, beams, n_gates_s, _ =
        _sweep_zyx_and_beams(sweep, ref_latitude, ref_longitude, ref_altitude, TM)
    balltree = _sweep_balltree_yx(radar_zyx, beams)

    nx = length(g.x_axis)
    ny = length(g.y_axis)
    n_fields = length(accum.fields)

    xincr = nx >= 2 ? (g.x_axis[2] - g.x_axis[1]) : 0.0
    horizontal_roi  = xincr * 0.75
    beam_inflation  = gd.beam_inflation
    power_threshold = gd.power_threshold

    Threads.@threads for ii in CartesianIndices((ny, nx))
        j_y, i_x = ii.I
        yx_point = [g.y_axis[j_y], g.x_axis[i_x]]

        eff_h_roi = horizontal_roi
        if beam_inflation > 0.0
            origin_dist = euclidean(yx_point, [0.0, 0.0])
            eff_h_roi = max(beam_inflation * origin_dist, horizontal_roi)
        end
        gates = inrange(balltree, yx_point, eff_h_roi)
        isempty(gates) && continue

        # Any in-range gate with non-missing missing_key → coverage 1.
        any_scanned = false
        for g_flat in gates
            ray  = _ray_of(g_flat, n_gates_s)
            gate = _gate_in_ray(g_flat, n_gates_s)
            if !ismissing(_gate_value(sweep, missing_key, ray, gate))
                any_scanned = true
                break
            end
        end
        if any_scanned
            @inbounds for m in 1:n_fields
                if accum.coverage[m, j_y, i_x] == Int8(0)
                    accum.coverage[m, j_y, i_x] = Int8(1)
                end
            end
        else
            continue
        end

        for g_flat in gates
            ray  = _ray_of(g_flat, n_gates_s)
            gate_in = _gate_in_ray(g_flat, n_gates_s)
            vk = _gate_value(sweep, valid_key, ray, gate_in)
            ismissing(vk) && continue

            dx = g.x_axis[i_x] - radar_zyx[g_flat][3]
            dy = g.y_axis[j_y] - radar_zyx[g_flat][2]
            gridpt_az = (pi / 2.0) - atan(dy, dx)
            gridpt_az < 0 && (gridpt_az += 2 * pi)

            angle_diff = spherical_angle([beams[g_flat, 1], beams[g_flat, 2]],
                                          [gridpt_az, beams[g_flat, 2]])
            angle_weight = exp(-angle_diff * 79.43)
            angle_weight < power_threshold && (angle_weight = 0.0)

            r = beams[g_flat, 3]
            gridpt_r = sqrt(dx^2 + dy^2)
            range_weight = gridpt_r / r
            if abs(gridpt_r - r) > horizontal_roi
                range_weight = 0.0
            end
            total_weight = range_weight * angle_weight
            total_weight > 0.0 || continue

            @inbounds for m in 1:n_fields
                fname = accum.fields[m]
                v = _gate_value(sweep, fname, ray, gate_in)
                ismissing(v) && continue
                mode = accum.grid_type[fname]
                accum.coverage[m, j_y, i_x] = Int8(2)
                if mode === :linear
                    linear_z = 10.0 ^ (v / 10.0)
                    accum.weighted_sum[m, j_y, i_x] += total_weight * linear_z
                    accum.weight_total[m, j_y, i_x] += total_weight
                elseif mode === :nearest
                    if total_weight > accum.weight_total[m, j_y, i_x]
                        accum.weighted_sum[m, j_y, i_x] = v
                        accum.weight_total[m, j_y, i_x] = total_weight
                    end
                else
                    accum.weighted_sum[m, j_y, i_x] += total_weight * v
                    accum.weight_total[m, j_y, i_x] += total_weight
                end
            end
        end
    end
    return accum
end

# ── Composite worker (2D, column-maximum) ────────────────────────────────────

function _grid_sweep_composite_2d!(accum::GridAccumulator, sweep::SweepGroup,
                                    p::DaishoParameters,
                                    ref_latitude::Float64, ref_longitude::Float64,
                                    ref_altitude::Float64)
    g  = accum.grid_spec
    gd = p.gridding
    missing_key = field_with_tag(p, :define_scanned;   for_op="grid_sweep! (accumulator path)")
    valid_key   = field_with_tag(p, :define_detection; for_op="grid_sweep! (accumulator path)")

    TM = CoordRefSystems.shift(TransverseMercator{1.0, g.reference_latitude, WGS84Latest},
                                lonₒ = g.reference_longitude)
    _, radar_zyx, beams, n_gates_s, _ =
        _sweep_zyx_and_beams(sweep, ref_latitude, ref_longitude, ref_altitude, TM)
    balltree = _sweep_balltree_yx(radar_zyx, beams)

    nx = length(g.x_axis)
    ny = length(g.y_axis)
    n_fields = length(accum.fields)

    xincr = nx >= 2 ? (g.x_axis[2] - g.x_axis[1]) : 0.0
    horizontal_roi  = xincr * 0.75
    beam_inflation  = gd.beam_inflation

    # Find the valid_key column index for the max selection. If the valid_key
    # field isn't in the accumulator, composite produces no contribution.
    valid_idx = findfirst(==(valid_key), accum.fields)

    Threads.@threads for ii in CartesianIndices((ny, nx))
        j_y, i_x = ii.I
        yx_point = [g.y_axis[j_y], g.x_axis[i_x]]

        eff_h_roi = horizontal_roi
        if beam_inflation > 0.0
            origin_dist = euclidean(yx_point, [0.0, 0.0])
            eff_h_roi = max(beam_inflation * origin_dist, horizontal_roi)
        end
        gates = inrange(balltree, yx_point, eff_h_roi)
        isempty(gates) && continue

        any_scanned = false
        for g_flat in gates
            ray  = _ray_of(g_flat, n_gates_s)
            gate = _gate_in_ray(g_flat, n_gates_s)
            if !ismissing(_gate_value(sweep, missing_key, ray, gate))
                any_scanned = true
                break
            end
        end
        if any_scanned
            @inbounds for m in 1:n_fields
                if accum.coverage[m, j_y, i_x] == Int8(0)
                    accum.coverage[m, j_y, i_x] = Int8(1)
                end
            end
        else
            continue
        end

        valid_idx === nothing && continue

        # Scan in-range gates, pick the one with the largest valid_key value.
        best_val = -Inf
        best_flat = 0
        for g_flat in gates
            ray  = _ray_of(g_flat, n_gates_s)
            gate_in = _gate_in_ray(g_flat, n_gates_s)
            vk = _gate_value(sweep, valid_key, ray, gate_in)
            ismissing(vk) && continue
            if vk > best_val
                best_val = vk
                best_flat = g_flat
            end
        end
        best_flat == 0 && continue

        ray  = _ray_of(best_flat, n_gates_s)
        gate_in = _gate_in_ray(best_flat, n_gates_s)
        # Composite carries that gate's value directly; weight_total is a
        # filled-by-this-sweep flag (we use 1.0). The :nearest finalize path
        # returns weighted_sum unchanged.
        @inbounds for m in 1:n_fields
            fname = accum.fields[m]
            v = _gate_value(sweep, fname, ray, gate_in)
            ismissing(v) && continue
            # Overwrite only when this sweep's best > prior; merge_accumulators!
            # for :nearest already picks the higher weight_total, so use the
            # composite value itself as the "weight" so cross-sweep merges
            # also pick the column max.
            cur_w = accum.weight_total[m, j_y, i_x]
            if m == valid_idx
                if best_val > cur_w - 1.0  # tie-aware
                    accum.weighted_sum[m, j_y, i_x] = v
                    accum.weight_total[m, j_y, i_x] = best_val + 1.0
                    accum.coverage[m, j_y, i_x] = Int8(2)
                end
            else
                # Companion field: tag along with whichever gate was selected.
                # Use the same convention so merges stay consistent.
                accum.weighted_sum[m, j_y, i_x] = v
                accum.weight_total[m, j_y, i_x] = best_val + 1.0
                accum.coverage[m, j_y, i_x] = Int8(2)
            end
        end
    end
    return accum
end

# ── Column worker (1D, z) ────────────────────────────────────────────────────

function _grid_sweep_column_1d!(accum::GridAccumulator, sweep::SweepGroup,
                                 p::DaishoParameters,
                                 ref_latitude::Float64, ref_longitude::Float64,
                                 ref_altitude::Float64)
    g  = accum.grid_spec
    gd = p.gridding
    missing_key = field_with_tag(p, :define_scanned;   for_op="grid_sweep! (accumulator path)")
    valid_key   = field_with_tag(p, :define_detection; for_op="grid_sweep! (accumulator path)")

    TM = CoordRefSystems.shift(TransverseMercator{1.0, g.reference_latitude, WGS84Latest},
                                lonₒ = g.reference_longitude)
    _, radar_zyx, beams, n_gates_s, n_rays_s =
        _sweep_zyx_and_beams(sweep, ref_latitude, ref_longitude, ref_altitude, TM)

    nz = length(g.z_axis)
    n_fields = length(accum.fields)
    n_gate_total = size(beams, 1)

    zincr = nz >= 2 ? (g.z_axis[2] - g.z_axis[1]) : 0.0
    vertical_roi    = zincr * 0.75
    beam_inflation  = gd.beam_inflation
    power_threshold = gd.power_threshold

    Threads.@threads for k_z in 1:nz
        grid_z = g.z_axis[k_z]
        eff_v_roi = vertical_roi
        origin_dist = euclidean(grid_z, [0.0])
        if beam_inflation > 0.0
            eff_v_roi = max(beam_inflation * origin_dist, vertical_roi)
        end

        # Coverage: any gate in vertical range with non-missing missing_key.
        any_scanned = false
        for g_flat in 1:n_gate_total
            abs(beams[g_flat, 4] - grid_z) > eff_v_roi && continue
            ray  = _ray_of(g_flat, n_gates_s)
            gate = _gate_in_ray(g_flat, n_gates_s)
            if !ismissing(_gate_value(sweep, missing_key, ray, gate))
                any_scanned = true
                break
            end
        end
        if any_scanned
            @inbounds for m in 1:n_fields
                if accum.coverage[m, k_z] == Int8(0)
                    accum.coverage[m, k_z] = Int8(1)
                end
            end
        else
            continue
        end

        for g_flat in 1:n_gate_total
            ray  = _ray_of(g_flat, n_gates_s)
            gate_in = _gate_in_ray(g_flat, n_gates_s)
            vk = _gate_value(sweep, valid_key, ray, gate_in)
            ismissing(vk) && continue

            dz = grid_z - radar_zyx[g_flat][1]
            r  = beams[g_flat, 3]
            sine_h = ((dz + Reff)^2 - r^2 - Reff^2) / (2 * r * Reff)
            abs(sine_h) < 1.0 || continue
            gridpt_el = asin(sine_h)

            angle_diff = spherical_angle([beams[g_flat, 1], beams[g_flat, 2]],
                                          [beams[g_flat, 1], gridpt_el])
            angle_weight = exp(-angle_diff * 79.43)
            angle_weight < power_threshold && (angle_weight = 0.0)

            gridpt_r = grid_z / cos(beams[g_flat, 2])
            range_weight = gridpt_r / r
            if abs(gridpt_r - r) > vertical_roi
                range_weight = 0.0
            end
            total_weight = range_weight * angle_weight
            total_weight > 0.0 || continue

            @inbounds for m in 1:n_fields
                fname = accum.fields[m]
                v = _gate_value(sweep, fname, ray, gate_in)
                ismissing(v) && continue
                mode = accum.grid_type[fname]
                accum.coverage[m, k_z] = Int8(2)
                if mode === :linear
                    linear_z = 10.0 ^ (v / 10.0)
                    accum.weighted_sum[m, k_z] += total_weight * linear_z
                    accum.weight_total[m, k_z] += total_weight
                elseif mode === :nearest
                    if total_weight > accum.weight_total[m, k_z]
                        accum.weighted_sum[m, k_z] = v
                        accum.weight_total[m, k_z] = total_weight
                    end
                else
                    accum.weighted_sum[m, k_z] += total_weight * v
                    accum.weight_total[m, k_z] += total_weight
                end
            end
        end
    end
    return accum
end

# ── Volume-level helpers (build_grid_spec, writers' gridpoints, latlon) ──────

"""
    build_grid_spec(shape, volume, p; rhi_azimuth=nothing) -> GridSpec

Helper for the Volume drivers. Pulls the right grid sub-spec from `p` based on
`shape` and pairs it with the volume's reference position to build a
`GridSpec`.

For mobile-platform sweeps the reference position falls back to the first
ray's georeference when the volume's stationary lat/lon/alt is the zero
default; callers can override by constructing the `GridSpec` directly.
"""
function build_grid_spec(shape::Symbol, volume::Volume, p::DaishoParameters;
                          rhi_azimuth::Union{Nothing,Real} = nothing)
    ref_lat = volume.latitude
    ref_lon = volume.longitude
    ref_alt = volume.altitude

    # If the volume's stationary lat/lon look like the zero default but the
    # first sweep has a georeference, fall back to that — typical mobile
    # platform path.
    if (ref_lat == 0.0 && ref_lon == 0.0) && !isempty(volume.sweeps)
        s1 = volume.sweeps[1]
        if s1.georeference !== nothing && !isempty(s1.georeference.latitude)
            ref_lat = s1.georeference.latitude[1]
            ref_lon = s1.georeference.longitude[1]
            ref_alt = s1.georeference.altitude[1]
        end
    end

    if shape === :volume_3d
        g = p.grid.cartesian
        return GridSpec(
            shape = :volume_3d,
            reference_latitude = ref_lat,
            reference_longitude = ref_lon,
            x_axis = collect(Float64, g.xmin .+ (0:(g.xdim-1)) .* g.xincr),
            y_axis = collect(Float64, g.ymin .+ (0:(g.ydim-1)) .* g.yincr),
            z_axis = collect(Float64, g.zmin .+ (0:(g.zdim-1)) .* g.zincr),
        )
    elseif shape === :latlon_3d
        g = p.grid.latlon
        # Snap the projection origin to a degincr boundary so the geographic
        # grid lines up across runs (matches legacy `grid_radar_latlon_volume`).
        snapped_lat = ref_lat - rem(ref_lat, g.degincr)
        snapped_lon = ref_lon - rem(ref_lon, g.degincr)
        lat_axis = collect(Float64,
            round(snapped_lat + g.latmin, digits = 6) .+
            (0:(g.latdim-1)) .* g.degincr)
        lon_axis = collect(Float64,
            round(snapped_lon + g.lonmin, digits = 6) .+
            (0:(g.londim-1)) .* g.degincr)
        return GridSpec(
            shape = :latlon_3d,
            reference_latitude = snapped_lat,
            reference_longitude = snapped_lon,
            # Placeholder Cartesian axes for shape sizing; latlon worker uses
            # lat_axis/lon_axis for the geographic axes.
            x_axis = collect(Float64, 1:g.londim),
            y_axis = collect(Float64, 1:g.latdim),
            z_axis = collect(Float64, g.zmin .+ (0:(g.zdim-1)) .* g.zincr),
            lat_axis = lat_axis,
            lon_axis = lon_axis,
        )
    elseif shape === :rhi_2d
        g = p.grid.rhi
        # Resolve the RHI azimuth: explicit override, else the first sweep's
        # first ray's azimuth.
        az = if rhi_azimuth !== nothing
            Float64(rhi_azimuth)
        elseif !isempty(volume.sweeps) && !isempty(volume.sweeps[1].azimuth)
            Float64(volume.sweeps[1].azimuth[1])
        else
            0.0
        end
        return GridSpec(
            shape = :rhi_2d,
            reference_latitude = ref_lat,
            reference_longitude = ref_lon,
            x_axis = collect(Float64, g.rmin .+ (0:(g.rdim-1)) .* g.rincr),
            y_axis = [0.0],  # unused
            z_axis = collect(Float64, g.zmin .+ (0:(g.zdim-1)) .* g.zincr),
            rhi_azimuth = az,
        )
    elseif shape === :ppi_2d || shape === :composite_2d
        g = p.grid.cartesian
        return GridSpec(
            shape = shape,
            reference_latitude = ref_lat,
            reference_longitude = ref_lon,
            x_axis = collect(Float64, g.xmin .+ (0:(g.xdim-1)) .* g.xincr),
            y_axis = collect(Float64, g.ymin .+ (0:(g.ydim-1)) .* g.yincr),
            z_axis = [0.0],
        )
    elseif shape === :column_1d
        g = p.grid.cartesian
        return GridSpec(
            shape = :column_1d,
            reference_latitude = ref_lat,
            reference_longitude = ref_lon,
            x_axis = [0.0],
            y_axis = [0.0],
            z_axis = collect(Float64, g.zmin .+ (0:(g.zdim-1)) .* g.zincr),
        )
    else
        throw(ArgumentError("build_grid_spec: unsupported shape $shape"))
    end
end

# Return a (zdim, ydim, xdim, 3) gridpoints array shaped like
# `initialize_regular_grid` so the existing writers can consume it.
function _gridpoints_volume_array(g::GridSpec)
    nx = length(g.x_axis); ny = length(g.y_axis); nz = length(g.z_axis)
    out = Array{Float64}(undef, nz, ny, nx, 3)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        out[k, j, i, 1] = g.z_axis[k]
        out[k, j, i, 2] = g.y_axis[j]
        out[k, j, i, 3] = g.x_axis[i]
    end
    return out
end

function _gridpoints_latlon_array(g::GridSpec)
    nx = length(g.lon_axis); ny = length(g.lat_axis); nz = length(g.z_axis)
    TM = CoordRefSystems.shift(TransverseMercator{1.0, g.reference_latitude, WGS84Latest},
                                lonₒ = g.reference_longitude)
    grid_origin = convert(TM, LatLon(g.reference_latitude, g.reference_longitude))
    out = Array{Float64}(undef, nz, ny, nx, 3)
    @inbounds for j in 1:ny, i in 1:nx
        cartTM = convert(TM, LatLon(g.lat_axis[j], g.lon_axis[i]))
        y = ustrip(cartTM.y) - ustrip(grid_origin.y)
        x = ustrip(cartTM.x) - ustrip(grid_origin.x)
        for k in 1:nz
            out[k, j, i, 1] = g.z_axis[k]
            out[k, j, i, 2] = y
            out[k, j, i, 3] = x
        end
    end
    return out
end

function _gridpoints_rhi_array(g::GridSpec)
    nr = length(g.x_axis); nz = length(g.z_axis)
    out = Array{Float64}(undef, nz, nr, 2)
    @inbounds for j in 1:nz, i in 1:nr
        out[j, i, 1] = g.z_axis[j]
        out[j, i, 2] = g.x_axis[i]
    end
    return out
end

function _gridpoints_ppi_array(g::GridSpec)
    nx = length(g.x_axis); ny = length(g.y_axis)
    out = Array{Float64}(undef, ny, nx, 2)
    @inbounds for j in 1:ny, i in 1:nx
        out[j, i, 1] = g.y_axis[j]
        out[j, i, 2] = g.x_axis[i]
    end
    return out
end

function _gridpoints_column_array(g::GridSpec)
    return collect(Float64, g.z_axis)
end

# Compute the lat/lon grid the existing writers consume. Shape depends on
# the driver: 3D writers want (ydim, xdim, 2), RHI wants (rdim, 2), column
# wants a 2-element [lat, lon].
function _compute_latlon_grid(g::GridSpec)
    TM = CoordRefSystems.shift(TransverseMercator{1.0, g.reference_latitude, WGS84Latest},
                                lonₒ = g.reference_longitude)
    grid_origin = convert(TM, LatLon(g.reference_latitude, g.reference_longitude))
    if g.shape === :volume_3d
        nx = length(g.x_axis); ny = length(g.y_axis)
        out = Array{Float64}(undef, ny, nx, 2)
        @inbounds for j in 1:ny, i in 1:nx
            cartTM = convert(TM, Cartesian{WGS84Latest}(
                grid_origin.x + g.x_axis[i] * u"m",
                grid_origin.y + g.y_axis[j] * u"m"))
            latlon = convert(LatLon, cartTM)
            out[j, i, 1] = ustrip(latlon.lat)
            out[j, i, 2] = ustrip(latlon.lon)
        end
        return out
    elseif g.shape === :latlon_3d
        nx = length(g.lon_axis); ny = length(g.lat_axis)
        out = Array{Float64}(undef, ny, nx, 2)
        @inbounds for j in 1:ny, i in 1:nx
            out[j, i, 1] = g.lat_axis[j]
            out[j, i, 2] = g.lon_axis[i]
        end
        return out
    elseif g.shape === :rhi_2d
        nr = length(g.x_axis)
        az = g.rhi_azimuth === nothing ? 0.0 : deg2rad(g.rhi_azimuth)
        out = Array{Float64}(undef, nr, 2)
        @inbounds for i in 1:nr
            r_pt = g.x_axis[i]
            y_pt = r_pt * cos(az); x_pt = r_pt * sin(az)
            cartTM = convert(TM, Cartesian{WGS84Latest}(
                grid_origin.x + x_pt * u"m", grid_origin.y + y_pt * u"m"))
            latlon = convert(LatLon, cartTM)
            out[i, 1] = ustrip(latlon.lat)
            out[i, 2] = ustrip(latlon.lon)
        end
        return out
    elseif g.shape === :ppi_2d || g.shape === :composite_2d
        nx = length(g.x_axis); ny = length(g.y_axis)
        out = Array{Float64}(undef, ny, nx, 2)
        @inbounds for j in 1:ny, i in 1:nx
            cartTM = convert(TM, Cartesian{WGS84Latest}(
                grid_origin.x + g.x_axis[i] * u"m",
                grid_origin.y + g.y_axis[j] * u"m"))
            latlon = convert(LatLon, cartTM)
            out[j, i, 1] = ustrip(latlon.lat)
            out[j, i, 2] = ustrip(latlon.lon)
        end
        return out
    elseif g.shape === :column_1d
        return [g.reference_latitude, g.reference_longitude]
    else
        throw(ArgumentError("_compute_latlon_grid: unsupported shape $(g.shape)"))
    end
end

# Build a minimal legacy `radar` adapter just for writers that consume
# `.time`, `.azimuth`, `.scan_name`. No moments matrix is allocated.
function _writer_radar_stub(volume::Volume)
    n_total_rays = sum(n_rays(s) for s in volume.sweeps; init = 0)
    azimuth = Vector{Union{Missing,Float32}}(undef, n_total_rays)
    elevation = Vector{Union{Missing,Float32}}(undef, n_total_rays)
    times = Vector{DateTime}(undef, n_total_rays)
    cursor = 0
    for s in volume.sweeps
        n = n_rays(s)
        rng = (cursor + 1):(cursor + n)
        azimuth[rng] .= Float32.(s.azimuth)
        elevation[rng] .= Float32.(s.elevation)
        times[rng] .= s.time
        cursor += n
    end
    # Empty other fields the writers don't read.
    empty32 = Vector{Union{Missing,Float32}}(undef, n_total_rays)
    empty32 .= missing
    fa32 = Vector{Union{Missing,Float32}}(undef, length(volume.sweeps))
    fa32 .= missing
    return radar(
        scan_name = volume.scan_name,
        azimuth = azimuth,
        elevation = elevation,
        ew_platform = zeros(Float32, n_total_rays),
        ns_platform = zeros(Float32, n_total_rays),
        w_platform  = zeros(Float32, n_total_rays),
        nyquist_velocity = empty32,
        range = Float32.(isempty(volume.sweeps) ? Float64[] : volume.sweeps[1].range),
        time = times,
        latitude  = empty32,
        longitude = empty32,
        altitude  = empty32,
        fixed_angles = fa32,
        swpstart = fa32,
        swpend   = fa32,
        moments  = Array{Union{Missing,Float64}}(undef, 0, 0),
    )
end

# ── Multi-file / per-sweep workflow ──────────────────────────────────────────

"""
    grid_sweep_to_file(volume, sweep_index, accumulator_file, p;
                       grid_spec=nothing, heading=-9999.0,
                       merge_into_existing=true) -> path

Grid one sweep of `volume` into a JLD2 accumulator file. Useful for the
airborne / multi-file workflow where each CfRadial file is one sweep along a
flight track.

If `accumulator_file` already exists and `merge_into_existing=true`, the file
is loaded, the new sweep folded in via `grid_sweep!`, and the result saved
back. If `merge_into_existing=false` an `ArgumentError` is raised — refusing
to overwrite is the safer default.

If `accumulator_file` does not yet exist, `grid_spec` must be supplied so the
fresh accumulator can be allocated.

Returns the file path.
"""
function grid_sweep_to_file(volume::Volume, sweep_index::Int,
                             accumulator_file::AbstractString,
                             p::DaishoParameters;
                             grid_spec::Union{Nothing,GridSpec} = nothing,
                             heading::Real = -9999.0,
                             merge_into_existing::Bool = true)
    if isfile(accumulator_file)
        merge_into_existing || throw(ArgumentError(
            "grid_sweep_to_file: $(accumulator_file) exists; pass merge_into_existing=true to merge"))
        acc = load_accumulator(accumulator_file)
        if grid_spec !== nothing
            _grid_spec_equal(acc.grid_spec, grid_spec) || throw(ArgumentError(
                "grid_sweep_to_file: supplied grid_spec disagrees with the one stored in $(accumulator_file)"))
        end
    else
        grid_spec === nothing && throw(ArgumentError(
            "grid_sweep_to_file: $(accumulator_file) does not exist; supply grid_spec to create"))
        acc = GridAccumulator(grid_spec, p)
    end
    grid_sweep!(acc, volume, sweep_index, p; heading = heading,
                source_file = String(accumulator_file))
    save_accumulator(accumulator_file, acc)
    return accumulator_file
end

"""
    finalize_accumulator_file(input_file, output_file, p; index_time) -> output_file

Load an accumulator from JLD2, call `finalize_grid`, and dispatch to the
appropriate `write_gridded_radar_*` writer based on `accumulator.grid_spec.shape`.
Returns the output file path.
"""
function finalize_accumulator_file(input_file::AbstractString,
                                    output_file::AbstractString,
                                    p::DaishoParameters;
                                    index_time)
    acc = load_accumulator(input_file)
    radar_grid = finalize_grid(acc)
    latlon_grid = _compute_latlon_grid(acc.grid_spec)
    moment_dict = Dict{String,Int}(name => i for (i, name) in enumerate(acc.fields))

    # Reconstruct start/stop time from the recorded SweepProvenance entries.
    start_time = nothing
    stop_time = nothing
    for sp in acc.sweeps
        if sp.time_start !== nothing
            start_time = start_time === nothing ? sp.time_start :
                         min(start_time, sp.time_start)
        end
        if sp.time_end !== nothing
            stop_time = stop_time === nothing ? sp.time_end :
                        max(stop_time, sp.time_end)
        end
    end
    start_time === nothing && (start_time = index_time)
    stop_time === nothing && (stop_time = index_time)

    shape = acc.grid_spec.shape
    if shape === :volume_3d
        gridpoints = _gridpoints_volume_array(acc.grid_spec)
        write_gridded_radar_volume(output_file, index_time, start_time, stop_time,
            gridpoints, radar_grid, latlon_grid, moment_dict,
            acc.grid_spec.reference_latitude,
            acc.grid_spec.reference_longitude, -9999.0)
    elseif shape === :latlon_3d
        gridpoints = _gridpoints_latlon_array(acc.grid_spec)
        write_gridded_radar_volume(output_file, index_time, start_time, stop_time,
            gridpoints, radar_grid, latlon_grid, moment_dict,
            acc.grid_spec.reference_latitude,
            acc.grid_spec.reference_longitude, -9999.0)
    elseif shape === :rhi_2d
        gridpoints = _gridpoints_rhi_array(acc.grid_spec)
        stub = _writer_stub_from_provenance(acc)
        write_gridded_radar_rhi(output_file, index_time, stub,
            gridpoints, radar_grid, latlon_grid, moment_dict,
            acc.grid_spec.reference_latitude,
            acc.grid_spec.reference_longitude)
    elseif shape === :ppi_2d || shape === :composite_2d
        gridpoints = _gridpoints_ppi_array(acc.grid_spec)
        stub = _writer_stub_from_provenance(acc)
        write_gridded_radar_ppi(output_file, index_time, stub,
            gridpoints, radar_grid, latlon_grid, moment_dict,
            acc.grid_spec.reference_latitude,
            acc.grid_spec.reference_longitude, -9999.0)
    elseif shape === :column_1d
        gridpoints = _gridpoints_column_array(acc.grid_spec)
        write_gridded_radar_column(output_file, index_time, start_time, stop_time,
            gridpoints, radar_grid, latlon_grid, moment_dict,
            acc.grid_spec.reference_latitude,
            acc.grid_spec.reference_longitude)
    else
        throw(ArgumentError("finalize_accumulator_file: unsupported shape $shape"))
    end
    return output_file
end

# Minimal radar-shaped stub built from accumulator provenance — gives the
# writers `time`, `azimuth`, `scan_name` without requiring the original
# Volume. Used when finalizing an accumulator from disk where the input
# volumes are no longer in memory.
function _writer_stub_from_provenance(acc::GridAccumulator)
    times = DateTime[]
    azimuths = Float32[]
    scan_name = ""
    if !isempty(acc.sweeps)
        scan_name = acc.sweeps[1].scan_name
        for sp in acc.sweeps
            if sp.time_start !== nothing
                push!(times, sp.time_start)
            end
            if sp.time_end !== nothing
                push!(times, sp.time_end)
            end
            # Use the sweep's fixed angle direction proxy as the RHI azimuth.
            if !isnan(sp.fixed_angle)
                push!(azimuths, Float32(sp.fixed_angle))
            end
        end
    end
    isempty(times)    && push!(times, DateTime(2000, 1, 1))
    isempty(azimuths) && push!(azimuths, Float32(0.0))

    n = length(azimuths)
    return radar(
        scan_name = scan_name,
        azimuth = Vector{Union{Missing,Float32}}(azimuths),
        elevation = Vector{Union{Missing,Float32}}(fill(Float32(0.0), n)),
        ew_platform = zeros(Float32, n),
        ns_platform = zeros(Float32, n),
        w_platform  = zeros(Float32, n),
        nyquist_velocity = Vector{Union{Missing,Float32}}(fill(Float32(0.0), n)),
        range = Float32[0.0],
        time  = times,
        latitude  = Vector{Union{Missing,Float32}}(fill(Float32(0.0), n)),
        longitude = Vector{Union{Missing,Float32}}(fill(Float32(0.0), n)),
        altitude  = Vector{Union{Missing,Float32}}(fill(Float32(0.0), n)),
        fixed_angles = Vector{Union{Missing,Float32}}(azimuths),
        swpstart = Vector{Union{Missing,Float32}}(Float32[0.0]),
        swpend   = Vector{Union{Missing,Float32}}(Float32[Float32(n - 1)]),
        moments  = Array{Union{Missing,Float64}}(undef, 0, 0),
    )
end

"""
    combine_accumulator_files(input_files, output_file) -> output_file

Load all listed accumulator files (all must share a `grid_spec`) and merge
them into a single accumulator written to `output_file`. Useful for combining
per-sweep airborne accumulators along a leg into one accumulator before
finalization.

Fields with `field_folds=true` cannot be merged across distinct sweeps; the
underlying `merge_accumulators!` will raise an `ArgumentError` and the caller
must process the per-sweep files directly through the wind retrieval instead.
"""
function combine_accumulator_files(input_files::Vector{<:AbstractString},
                                    output_file::AbstractString)
    isempty(input_files) && throw(ArgumentError(
        "combine_accumulator_files: input_files is empty"))
    dst = load_accumulator(input_files[1])
    for path in input_files[2:end]
        src = load_accumulator(path)
        merge_accumulators!(dst, src)
    end
    save_accumulator(output_file, dst)
    return output_file
end

"""
    finalize_grid(accum) -> Array{Float64}

Normalize an accumulator into the grid shape the existing writers consume.
`:weighted` fields divide by `weight_total`; `:linear` fields divide and then
convert back to dBZ; `:nearest` carries through unchanged. Cells with no
weight but with `coverage == 1` are flagged **undetect** (scanned, no echo).
Cells with `coverage == 0` are flagged **true missing** (gate not measured).
The actual sentinel values come from the accumulator's `undetect` /
`fill_value` (sourced from `[io]`).
"""
function finalize_grid(accum::GridAccumulator)
    fill_value = accum.fill_value
    undetect   = accum.undetect
    out = fill(fill_value, size(accum.weighted_sum))
    n_fields = length(accum.fields)
    trailing_size = size(accum.weighted_sum)[2:end]
    @inbounds for m in 1:n_fields
        name = accum.fields[m]
        mode = accum.grid_type[name]
        for ix in CartesianIndices(trailing_size)
            cov = accum.coverage[m, ix]
            w   = accum.weight_total[m, ix]
            s   = accum.weighted_sum[m, ix]
            if cov == Int8(2) && w > 0
                if mode === :nearest
                    out[m, ix] = s
                elseif mode === :linear
                    avg = s / w
                    out[m, ix] = avg > 0.0 ? 10.0 * log10(avg) : undetect
                else
                    out[m, ix] = s / w
                end
            elseif cov == Int8(1)
                out[m, ix] = undetect
            else
                out[m, ix] = fill_value
            end
        end
    end
    return out
end
