# GridAccumulator — pre-normalized weighted sums on a chosen grid.
#
# Lets the caller grid one sweep at a time, persist intermediate state to JLD2,
# and combine sweeps from many files later. Replaces the legacy "all sweeps in
# one volume → one gridded NetCDF" path for workflows like airborne P3 and
# multi-Doppler retrieval, while preserving the existing weighting math.

const GRID_ACCUMULATOR_SCHEMA_VERSION = 1

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
    GridAccumulator

Pre-normalized weighted sums on a chosen grid. Each `grid_sweep!` call adds one
sweep's contribution. `finalize_grid` divides by weights, converts linear→dBZ
where appropriate, and applies the `-9999`/`-32768` sentinels.

Layout: `weighted_sum[field_idx, axis_dims…]`. The trailing axis dims depend
on `grid_spec.shape`.
"""
Base.@kwdef struct GridAccumulator
    grid_spec::GridSpec
    fields::Vector{String}
    grid_type::Dict{String,Symbol}
    field_folds::Vector{Bool}
    weighted_sum::Array{Float64}
    weight_total::Array{Float64}
    coverage::Array{Int8}
    sweeps::Vector{SweepProvenance}
    schema_version::Int
end

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
    GridAccumulator(grid_spec, fields, grid_type; field_folds)

Allocate an empty accumulator on the given grid for an explicit list of fields.
`field_folds` defaults to all-false; supply `true` per-field for radial
velocities and other folding quantities to make `merge_accumulators!` refuse to
combine them across distinct sweeps.
"""
function GridAccumulator(grid_spec::GridSpec,
                          fields::Vector{String},
                          grid_type::Dict{String,Symbol};
                          field_folds::Vector{Bool} = fill(false, length(fields)))
    length(field_folds) == length(fields) ||
        throw(ArgumentError("GridAccumulator: field_folds must have one entry per field"))
    for f in fields
        haskey(grid_type, f) ||
            throw(ArgumentError("GridAccumulator: grid_type missing entry for $f"))
    end
    dims = accumulator_dims(grid_spec, length(fields))
    return GridAccumulator(
        grid_spec       = grid_spec,
        fields          = copy(fields),
        grid_type       = copy(grid_type),
        field_folds     = copy(field_folds),
        weighted_sum    = zeros(Float64, dims),
        weight_total    = zeros(Float64, dims),
        coverage        = zeros(Int8, dims),
        sweeps          = SweepProvenance[],
        schema_version  = GRID_ACCUMULATOR_SCHEMA_VERSION,
    )
end

"""
    GridAccumulator(grid_spec, p::DaishoParameters)

Allocate an accumulator using `p.moments.fields` for the column order and
`p.moments.grid_type` for the interpolation hints. Field-folds defaults to all
false; the caller can override with the explicit constructor when gridding a
field-folds quantity (e.g. `VEL`).
"""
function GridAccumulator(grid_spec::GridSpec, p::DaishoParameters)
    return GridAccumulator(grid_spec, p.moments.fields, p.moments.grid_type)
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
