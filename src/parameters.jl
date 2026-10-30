# Daisho.jl runtime parameter system.
#
# Loads TOML configuration into immutable typed structs. Each section of the
# TOML maps 1:1 to a section struct; the top-level `DaishoParameters` bundles
# them. `config/defaults.toml` ships as the canonical template: users obtain a
# copy via `print_config("mygrid.toml")` and edit it. Loading is strict —
# `DaishoParameters(path)` reads only the user file, never silently falling
# back to bundled defaults.

using TOML

const DEFAULTS_TOML_PATH = joinpath(@__DIR__, "..", "config", "defaults.toml")

# ── Field tag vocabulary ─────────────────────────────────────────────────────
# Each field in `[fields]` maps to a flat array of tags drawn from this
# allowlist. No tag is required. New per-field capabilities are added as new
# tags + a consumer, never a schema change.

const FIELD_TAG_VOCAB = (
    :linear_interp, :weighted_interp, :nearest_interp,  # interpolation
    :define_detection, :define_scanned,                 # role tags (consumed)
    :velocity,                                          # multi-Doppler radial velocity
    :beam_height,                                       # value sourced from gate geometry
    :reflectivity,                                      # gate dBZ (fall-speed correction)
    :spectrum_width,                                    # gate SW (per-gate velocity error)
)
const FIELD_INTERP_TAGS   = (:linear_interp, :weighted_interp, :nearest_interp)
const FIELD_SINGULAR_TAGS = (:define_detection, :define_scanned, :velocity, :beam_height,
                             :reflectivity, :spectrum_width)

# ── Section structs ──────────────────────────────────────────────────────────

"""
    FieldSpec

One canonical field and the flat set of vocabulary tags it carries. Tags drive
interpolation ([`interp_of`](@ref)) and gridding roles ([`field_with_tag`](@ref));
order is insignificant.
"""
struct FieldSpec
    name::String
    tags::Set{Symbol}
end

"""
    MomentParameters

Canonical fields of interest, each a [`FieldSpec`](@ref) carrying a flat tag
list. The CfRadial 2.1 refactor dropped the historical raw/qc distinction: a
field is just a field, and its CfRadial relationships (e.g. `is_quality_field`,
`qualified_variables`) are recorded in `Field.metadata`.

# Fields
- `fields::Vector{FieldSpec}`: canonical fields of interest. Field order is
  insignificant — internal column order is derived deterministically by sorted
  field name.
"""
struct MomentParameters
    fields::Vector{FieldSpec}
end

"""
    has_tag(fs::FieldSpec, t::Symbol) -> Bool

Whether field `fs` carries tag `t`.
"""
has_tag(fs::FieldSpec, t::Symbol) = t in fs.tags

"""
    interp_of(fs::FieldSpec) -> Symbol

Interpolation mode derived from a field's tags: `:linear` (`linear_interp`),
`:nearest` (`nearest_interp`), or `:weighted` (the default when no
interpolation tag is present). A `:beam_height` field is always `:weighted` (a
height must never be dB-averaged), regardless of any interpolation tag.
"""
function interp_of(fs::FieldSpec)
    :beam_height    in fs.tags && return :weighted
    :linear_interp  in fs.tags && return :linear
    :nearest_interp in fs.tags && return :nearest
    return :weighted
end

# Deterministic, TOML-order-independent column order.
_ordered_fields(p) = sort(p.moments.fields; by = fs -> fs.name)

"""
    field_index_dict(p::DaishoParameters) -> Dict{String,Int}

Build a name→column-index dict over the fields sorted by name. Used by drivers
that need a moment_dict for the legacy gridding workers. Column order is
order-independent of the source TOML.
"""
field_index_dict(p) = Dict{String,Int}(fs.name => i
    for (i, fs) in enumerate(_ordered_fields(p)))

"""
    grid_type_index_dict(p::DaishoParameters) -> Dict{Int,Symbol}

Build a column-index→interpolation-mode dict over the fields sorted by name
(matching [`field_index_dict`](@ref)). Used by drivers that need a
grid_type_dict for the legacy gridding workers.
"""
grid_type_index_dict(p) = Dict{Int,Symbol}(i => interp_of(fs)
    for (i, fs) in enumerate(_ordered_fields(p)))

"""
    field_with_tag(p::DaishoParameters, tag::Symbol; for_op::String="") -> String

Resolve the single field carrying a singular role tag (e.g. `:define_scanned`,
`:define_detection`). Point-of-use replacement for the old
`missing_key`/`valid_key`: a non-gridding workflow need not declare these, so
the error is raised here (naming the operation) rather than at load.
"""
function field_with_tag(p, tag::Symbol; for_op::String="")
    m = [fs.name for fs in p.moments.fields if tag in fs.tags]
    length(m) == 1 && return m[1]
    if isempty(m)
        throw(ArgumentError("[fields]: no field is tagged `$tag`" *
            (isempty(for_op) ? "" : " (required by $for_op)") *
            ". Add \"$tag\" to exactly one field's tag list."))
    end
    throw(ArgumentError("[fields]: tag `$tag` is on multiple fields " *
        "$m; it must be on exactly one."))
end

"""
    field_with_tag_or(p, tag::Symbol, default="") -> String

Like [`field_with_tag`](@ref) but for an *optional* singular tag: returns the
single field carrying `tag`, or `default` when no field carries it (still errors
if more than one does). Used for `:beam_height`, which is optional.
"""
function field_with_tag_or(p, tag::Symbol, default::AbstractString="")
    m = [fs.name for fs in p.moments.fields if tag in fs.tags]
    isempty(m) && return default
    length(m) == 1 && return m[1]
    throw(ArgumentError("[fields]: tag `$tag` is on multiple fields " *
        "$m; it must be on at most one."))
end

"""
    QCParameters

Quality-control thresholds applied by [`threshold_qc!`](@ref).

# Fields
- `sqi_threshold::Float64`: minimum signal quality index.
- `snr_threshold::Float64`: minimum signal-to-noise ratio (dB).
- `spectrum_width_max::Float64`: maximum spectrum width (m/s).
- `rhohv_threshold::Float64`: minimum cross-correlation coefficient.
"""
Base.@kwdef struct QCParameters
    sqi_threshold::Float64      = 0.35
    snr_threshold::Float64      = 6.0
    spectrum_width_max::Float64 = 8.0
    rhohv_threshold::Float64    = 0.7
end

"""
    GriddingParameters

Engine-level gridding knobs shared across all gridding drivers.

# Fields
- `power_threshold::Float64`: the beam power level that defines the beam edge. The
  edge-referenced gate inclusion sets the beam half-angle to
  `beam_cutoff = ln(1/power_threshold) / beam_coef`, so **lower `power_threshold`
  ⇒ wider beam** (more of the exponential tail is counted as "the beam"). At the
  default `0.5` this is the half-power half-beamwidth, reproducing the legacy
  behaviour.
- `horizontal_roi_factor::Float64`: multiplier on the horizontal grid increment
  setting the box half-width for gate inclusion (default `0.75`).
- `vertical_roi_factor::Float64`: multiplier on the vertical grid increment
  setting the box half-height for gate inclusion (default `0.75`).
- `range_guard_min::Float64`: lower clamp (metres) on the gate slant range used as
  the `range_weight` divisor, guarding the near-radar singularity (default `1.0`).
  This is a *weighting* guard, not a gate filter — see `range_minimum` below.
- `range_weight_max::Float64`: upper clamp on the (unitless) `range_weight` ratio,
  bounding the near-radar singularity (default `10.0`).
- `range_minimum::Float64` / `range_maximum::Float64`: slant-range
  **gate-inclusion** bounds (metres). A gate is gridded only when its slant range
  `r` satisfies `range_minimum ≤ r ≤ range_maximum`; out-of-range gates are
  excluded entirely (they contribute no coverage and no value), so a grid cell
  reachable only by excluded gates is left as true-missing. Defaults `0.0` / `Inf`
  apply no filtering. Use them to homogenise a volume whose low sweep ranges
  farther than the others, or to cap a product at a higher-quality range. Distinct
  from the `range_guard_min` / `range_weight_max` weighting guards above.

The `*_factor`/`range_*` knobs are **optional** in a config file (they fall back
to the defaults above when omitted) so existing configs still load. The former
`beam_inflation` knob is **deprecated and removed** — per-range beam reach is now
derived from the radar beamwidth and the edge-referenced footprint. A leftover
`beam_inflation` key in a config is accepted and ignored.

The two gate-role moments are no longer configured here: the field whose
presence proves a gate was scanned (formerly `missing_key`) and the field
whose presence proves a detectable echo (formerly `valid_key`) are now
declared as the `define_scanned` / `define_detection` tags in `[fields]` and
resolved via [`field_with_tag`](@ref).
"""
Base.@kwdef struct GriddingParameters
    power_threshold::Float64       = 0.5
    horizontal_roi_factor::Float64 = 0.75
    vertical_roi_factor::Float64   = 0.75
    range_guard_min::Float64       = 1.0
    range_weight_max::Float64      = 10.0
    range_minimum::Float64         = 0.0
    range_maximum::Float64         = Inf
end

"""
    CartesianGridParameters

Regular Cartesian grid specification (3D Easting × Northing × Altitude, meters).

`range_minimum`/`range_maximum` are optional per-product overrides of the
`[gridding]` slant-range gate-inclusion bounds: when set (non-`nothing`) on a
product's table (e.g. `[grid.volume]`) they take precedence over the `[gridding]`
global for that product only; `nothing` (default) inherits the global. This lets,
say, the volume grid cap range at 120 km while composite/PPI stay full-range from
one config.
"""
Base.@kwdef struct CartesianGridParameters
    xmin::Float64  = -125000.0
    xincr::Float64 = 500.0
    xdim::Int      = 501
    ymin::Float64  = -125000.0
    yincr::Float64 = 500.0
    ydim::Int      = 501
    zmin::Float64  = 0.0
    zincr::Float64 = 500.0
    zdim::Int      = 37
    range_minimum::Union{Float64,Nothing} = nothing
    range_maximum::Union{Float64,Nothing} = nothing
end

"""
    LatLonGridParameters

Geographic regular grid specification used by `grid_radar_latlon_volume`.
Spacing is `degincr` for both lon and lat directions; vertical is in meters.
"""
Base.@kwdef struct LatLonGridParameters
    lonmin::Float64  = 0.0
    londim::Int      = 0
    latmin::Float64  = 0.0
    latdim::Int      = 0
    degincr::Float64 = 0.01
    zmin::Float64    = 0.0
    zincr::Float64   = 500.0
    zdim::Int        = 37
    # Optional per-product slant-range overrides of the `[gridding]` global
    # (`nothing` inherits); see `CartesianGridParameters`.
    range_minimum::Union{Float64,Nothing} = nothing
    range_maximum::Union{Float64,Nothing} = nothing
end

"""
    RhiGridParameters

Range-Height-Indicator grid specification (range × altitude, meters).
"""
Base.@kwdef struct RhiGridParameters
    rmin::Float64  = 0.0
    rincr::Float64 = 250.0
    rdim::Int      = 481
    zmin::Float64  = 0.0
    zincr::Float64 = 250.0
    zdim::Int      = 41
    # Optional per-product slant-range overrides of the `[gridding]` global
    # (`nothing` inherits); see `CartesianGridParameters`.
    range_minimum::Union{Float64,Nothing} = nothing
    range_maximum::Union{Float64,Nothing} = nothing
end

"""
    SpringsteelAxisConfig

One axis of a `[grid.springsteel]` section, in Springsteel's native i/j/k
vocabulary. Which keys are meaningful depends on the basis the geometry puts
on the axis (see [`SpringsteelGridConfig`](@ref) and
[`SPRINGSTEEL_AXIS_BASES`](@ref)):

- **spline** axes use `min`/`max`/`cells`. `cells` counts B-spline **cells**
  (the Springsteel `num_cells`), not gridpoints; physical quadrature points
  per axis = `cells × mubar`.
- **Chebyshev** axes (the `k` of `RZ`/`RLZ`/`SLZ`) use `min`/`max`/`points`
  (physical gridpoints, the Springsteel `kDim`).
- **Fourier** axes (the `j` of cylindrical/spherical geometries) are
  auto-sized by Springsteel from the ring formula; only `max_wavenumber`
  applies (`-1` = uncapped).

`regular_out` is the number of regularly spaced output points
`write_radar_netcdf` resamples onto along this axis; `0` selects the
geometry-aware default (`cells + 1` for spline axes, Springsteel's default
otherwise). `bc_min`/`bc_max` hold the boundary conditions for the axis'
min/max sides as per-variable dicts keyed by field name with a `"default"`
fallback, mirroring Springsteel's own convention.
"""
Base.@kwdef struct SpringsteelAxisConfig
    min::Float64        = 0.0
    max::Float64        = 0.0
    cells::Int          = 0
    points::Int         = 0
    regular_out::Int    = 0
    max_wavenumber::Int = -1
    bc_min::Dict{String,BoundaryConditions} =
        Dict{String,BoundaryConditions}("default" => NaturalBC())
    bc_max::Dict{String,BoundaryConditions} =
        Dict{String,BoundaryConditions}("default" => NaturalBC())
end

"""
    SPRINGSTEEL_AXIS_BASES

Basis placed on each (i, j, k) axis by every Springsteel geometry the
`[grid.springsteel]` schema supports: `:spline`, `:chebyshev`, `:fourier`
(auto-sized azimuthal), or `:none` (the axis does not exist). Drives both the
TOML validation (which keys an axis table must/may carry) and the
`SpringsteelGridParameters` construction in `create_radar_grid`.

The Fourier/Chebyshev-primary geometries (`L*`, `Z*`) need explicit spectral
dimensions and are out of the schema's scope — build those in Julia via
`Springsteel.SpringsteelGridParameters`/`createGrid` and pass the grid to the
gridding entry points directly.
"""
const SPRINGSTEEL_AXIS_BASES = Dict{String,NTuple{3,Symbol}}(
    "R"   => (:spline, :none,    :none),
    "RZ"  => (:spline, :none,    :chebyshev),
    "RR"  => (:spline, :spline,  :none),
    "RRR" => (:spline, :spline,  :spline),
    "RL"  => (:spline, :fourier, :none),
    "RLZ" => (:spline, :fourier, :chebyshev),
    "RLR" => (:spline, :fourier, :spline),
    "SL"  => (:spline, :fourier, :none),
    "SLZ" => (:spline, :fourier, :chebyshev),
    "SLR" => (:spline, :fourier, :spline),
)

"""
    SpringsteelGridConfig

Geometry-general Springsteel grid specification loaded from
`[grid.springsteel]`. Expresses a `Springsteel.SpringsteelGridParameters`
directly — geometry, quadrature, per-axis bounds/sizes, per-variable boundary
conditions, and output resampling counts — with no intermediate x/y/z
vocabulary (x/y/z only makes sense for the Cartesian geometries; i/j/k spans
them all: i = easting or radius, j = northing or azimuth, k = altitude).

Supported geometries (the spline-`i` families): Cartesian `R`/`RZ`/`RR`/`RRR`,
cylindrical `RL`/`RLZ`/`RLR`, spherical `SL`/`SLZ`/`SLR`. The Springsteel
variable map is **not** configured here — it is always derived from `[fields]`
via [`field_index_dict`](@ref) so the gridded-array column order and the
spectral-grid variable order can never disagree. Anything beyond this schema
(per-variable spectral filters, exotic setups) should be built in Julia with
`Springsteel.SpringsteelGridParameters(vars = radar_vars(p), …)` +
`createGrid` and passed to the gridding entry points directly.

# Fields
- `geometry::String`: Springsteel geometry code (see above).
- `mubar::Int`: quadrature points per spline cell.
- `quadrature::Symbol`: `:gauss` or `:regular`.
- `i`, `j`, `k`: per-axis [`SpringsteelAxisConfig`](@ref).
"""
Base.@kwdef struct SpringsteelGridConfig
    geometry::String   = "RRR"
    mubar::Int         = 3
    quadrature::Symbol = :gauss
    i::SpringsteelAxisConfig = SpringsteelAxisConfig(min = -125000.0, max = 125000.0, cells = 50)
    j::SpringsteelAxisConfig = SpringsteelAxisConfig(min = -125000.0, max = 125000.0, cells = 50)
    k::SpringsteelAxisConfig = SpringsteelAxisConfig(min = 0.0,       max = 15000.0,  cells = 10)
end

"""
    MetadataParameters

CF-1.12 global attributes written to every gridded NetCDF output. Every field
must be set in the user TOML under `[grid.metadata]`; the bundled template
ships generic placeholders (e.g. `institution = "Your Institution"`) so an
unedited config produces files that are obviously templates rather than
masquerading as someone else's data.

# Fields
- `Conventions::String`: CF metadata conventions version.
- `history::String`: provenance string written into the file.
- `institution::String`: producing institution.
- `source::String`: producing instrument / platform identifier.
- `instrument::String`: short instrument name.
- `title::String`: dataset title.
- `summary::String`: longer-form summary (often equal to `title`).
- `creator_name::String`: data creator's name.
- `creator_email::String`: data creator's email address.
- `creator_id::String`: data creator's identifier (e.g., ORCID URL).
- `project::String`: associated field projects, comma-separated.
- `platform::String`: hosting platform (vessel, aircraft, fixed site).
- `keywords::String`: comma-separated CF/ACDD keywords.
- `processing_level::String`: ACDD processing level (e.g., `"Level 4"`).
- `license::String`: data license (e.g., `"CC-BY-4.0"`).
- `references::String`: optional URL/DOI list. May be set to `""`; the empty
  string suppresses the corresponding NetCDF global attribute.

# Examples
```toml
[grid.metadata]
Conventions      = "CF-1.12"
history          = "v1.0"
institution      = "Example University"
source           = "Example C-band radar"
instrument       = "EXAMPLE"
title            = "Gridded Radar Data"
summary          = "Gridded Radar Data"
creator_name     = "Data Creator"
creator_email    = "creator@example.org"
creator_id       = "https://orcid.org/0000-0000-0000-0000"
project          = "EXAMPLE-2026"
platform         = "Field site"
keywords         = "radar, precipitation"
processing_level = "Level 4"
license          = "CC-BY-4.0"
references       = "https://doi.org/10.0000/example"
```
"""
Base.@kwdef struct MetadataParameters
    Conventions::String       = "CF-1.12"
    history::String           = "v1.0"
    institution::String       = "Your Institution"
    source::String            = "Your radar"
    instrument::String        = "Radar"
    title::String             = "Gridded Radar Data"
    summary::String           = "Gridded Radar Data"
    creator_name::String      = "Data Creator"
    creator_email::String     = ""
    creator_id::String        = ""
    project::String           = ""
    platform::String          = ""
    keywords::String          = "radar, precipitation"
    processing_level::String  = "Level 4"
    license::String           = "CC-BY-4.0"
    references::String        = ""
end

"""
    OriginParameters

Optional override of the grid origin and a moving (storm-relative) frame,
from `[grid.origin]`.

- `latitude_deg`, `longitude_deg`: the frame origin at `reference_time`.
  `nothing` (both absent) keeps the default origin: the radar position for a
  single volume, the centroid of the radar positions for multi-radar runs.
- `reference_time`: UTC time at which the frame is at the origin. Required
  when the motion is nonzero.
- `motion_east_mps`, `motion_north_mps`: constant frame motion (u, v). An
  observation at time `t` is placed at its projected position minus
  `(u, v)·(t − reference_time)`, so a feature moving with the frame stays put.

See also: [`resolve_grid_origin`](@ref), [`has_frame_motion`](@ref)
"""
Base.@kwdef struct OriginParameters
    latitude_deg::Union{Float64,Nothing}    = nothing
    longitude_deg::Union{Float64,Nothing}   = nothing
    reference_time::Union{DateTime,Nothing} = nothing
    motion_east_mps::Float64                = 0.0
    motion_north_mps::Float64               = 0.0
end

"""
    has_frame_motion(o::OriginParameters) -> Bool

True when `[grid.origin]` sets a nonzero frame motion.
"""
has_frame_motion(o::OriginParameters) = o.motion_east_mps != 0.0 || o.motion_north_mps != 0.0

# The moving-frame keywords of `GridSpec` / `_sweep_zyx_and_beams`.
_frame_kw(o::OriginParameters) = (reference_time = o.reference_time,
    motion_east_mps = o.motion_east_mps, motion_north_mps = o.motion_north_mps)

function _check_origin(o::OriginParameters)
    (o.latitude_deg === nothing) == (o.longitude_deg === nothing) || throw(ArgumentError(
        "[grid.origin]: set both `latitude_deg` and `longitude_deg`, or neither " *
        "(neither keeps the default origin)"))
    if o.latitude_deg !== nothing
        isfinite(o.latitude_deg) && isfinite(o.longitude_deg) || throw(ArgumentError(
            "[grid.origin]: `latitude_deg`/`longitude_deg` must be finite"))
        -90.0 <= o.latitude_deg <= 90.0 || throw(ArgumentError(
            "[grid.origin]: `latitude_deg` = $(o.latitude_deg) is outside [-90, 90]"))
    end
    isfinite(o.motion_east_mps) && isfinite(o.motion_north_mps) || throw(ArgumentError(
        "[grid.origin]: `motion_east_mps`/`motion_north_mps` must be finite"))
    if has_frame_motion(o) && o.reference_time === nothing
        throw(ArgumentError(
            "[grid.origin]: a nonzero motion needs `reference_time`, the UTC time at " *
            "which the frame is at the origin (e.g. reference_time = \"2008-09-19T04:00:00Z\")"))
    end
    return o
end

# Parse `reference_time`: a TOML datetime, or an ISO-8601 UTC string (optional
# trailing `Z`); "" means unset. Other UTC offsets are rejected.
function _origin_time(v)
    v isa DateTime && return v
    v isa AbstractString || throw(ArgumentError(
        "[grid.origin]: `reference_time` must be an ISO-8601 UTC time string"))
    isempty(v) && return nothing
    occursin(r"[+-]\d{2}:?\d{2}$", v) && throw(ArgumentError(
        "[grid.origin]: `reference_time` = \"$v\" has a UTC offset; give the time in UTC"))
    try
        return _parse_iso8601(v)
    catch
        throw(ArgumentError(
            "[grid.origin]: cannot parse `reference_time` = \"$v\" as an ISO-8601 UTC time"))
    end
end

function _origin_from_dict(d::AbstractDict)
    allowed = Set(string.(fieldnames(OriginParameters)))
    unknown = setdiff(Set(keys(d)), allowed)
    isempty(unknown) || throw(ArgumentError(
        "Unknown key(s) $(_fmt_keys(unknown)) in section `[grid.origin]`. " *
        "Allowed keys: $(join(fieldnames(OriginParameters), ", "))"))
    num(k) = (v = d[k]; v isa Real ? Float64(v) : throw(ArgumentError(
        "[grid.origin]: `$k` must be a number")))
    return OriginParameters(
        latitude_deg     = haskey(d, "latitude_deg")     ? num("latitude_deg")  : nothing,
        longitude_deg    = haskey(d, "longitude_deg")    ? num("longitude_deg") : nothing,
        reference_time   = haskey(d, "reference_time")   ? _origin_time(d["reference_time"]) : nothing,
        motion_east_mps  = haskey(d, "motion_east_mps")  ? num("motion_east_mps")  : 0.0,
        motion_north_mps = haskey(d, "motion_north_mps") ? num("motion_north_mps") : 0.0)
end

"""
    GridParameters

Aggregates the spatial-grid sub-specs plus the NetCDF metadata block. Pull
out the one matching the gridding driver you are calling.

The Cartesian-family products each get their own section so a single config can
drive different geometries for each: `volume` (`grid_radar_volume`), `composite`
(`grid_radar_composite`), `ppi` (`grid_radar_ppi`), and `column`
(`grid_radar_column`, used for QVPs). Each defaults to `cartesian` when its
own `[grid.<product>]` table is absent, so `[grid.cartesian]` acts as the
shared base geometry and the per-product tables are overrides.
"""
struct GridParameters
    cartesian::CartesianGridParameters
    volume::CartesianGridParameters
    composite::CartesianGridParameters
    ppi::CartesianGridParameters
    column::CartesianGridParameters
    latlon::LatLonGridParameters
    rhi::RhiGridParameters
    springsteel::SpringsteelGridConfig
    metadata::MetadataParameters
    origin::OriginParameters
end

# Keyword constructor centralizing the per-product fallback: a Cartesian-family
# product passed as `nothing` (or omitted) inherits `cartesian`. This keeps the
# inheritance identical whether a `GridParameters` is built from a TOML (see
# `_grid_from_dict`) or constructed directly in Julia — `p.grid.volume` is
# always a concrete `CartesianGridParameters`, never a sentinel.
function GridParameters(;
        cartesian::CartesianGridParameters = CartesianGridParameters(),
        volume::Union{Nothing,CartesianGridParameters}    = nothing,
        composite::Union{Nothing,CartesianGridParameters} = nothing,
        ppi::Union{Nothing,CartesianGridParameters}       = nothing,
        column::Union{Nothing,CartesianGridParameters}    = nothing,
        latlon::LatLonGridParameters       = LatLonGridParameters(),
        rhi::RhiGridParameters             = RhiGridParameters(),
        springsteel::SpringsteelGridConfig = SpringsteelGridConfig(),
        metadata::MetadataParameters       = MetadataParameters(),
        origin::OriginParameters           = OriginParameters())
    GridParameters(cartesian,
        volume    === nothing ? cartesian : volume,
        composite === nothing ? cartesian : composite,
        ppi       === nothing ? cartesian : ppi,
        column    === nothing ? cartesian : column,
        latlon, rhi, springsteel, metadata, _check_origin(origin))
end

"""
    IOParameters

I/O sentinels, named to mirror [`FieldMetadata`](@ref) and the CfRadial 2.1 /
ODIM data model exactly. Daisho preserves the distinction between **true
missing** (`fill_value`, CF `_FillValue` — gate not measured) and **undetect**
(`undetect`, ODIM `_Undetect` — gate scanned, no detectable signal). These two
values are authoritative across the gridding/finalize/NetCDF-write path; the
defaults are not a universal convention and users will legitimately differ.
"""
Base.@kwdef struct IOParameters
    fill_value::Float64 = -32768.0   # CF _FillValue  — true missing
    undetect::Float64   =  -9999.0   # ODIM _Undetect — undetect (clear air)
end

"""
    FallSpeedParameters

Hydrometeor terminal fall-speed correction for the radial velocities entering
the wind normal system, loaded from the optional `[synthesis.fall_speed]`
sub-table (a present table means "enabled"; every key is optional and defaults
as below). Mirrors samurai (`VarDriver3D.cpp`), with heights in metres and the
density correction taken from the configured `[reference_state]` profile:

    Z    = 10^(dBZ/10)                                  (linear reflectivity, mm⁶ m⁻³)
    DCOR = (ρ̄(0)/ρ̄(z))^density_exponent                (Beard 1985 density correction)
    V_R  = −rain_coefficient · Z^rain_exponent · DCOR   (Joss & Waldvogel 1971)
    V_S  = −snow_coefficient · Z^snow_exponent · DCOR   (Atlas et al. 1973)

Fall speeds are in m/s and **negative (downward)**; the corrected gate value is
`ṽ = v_r − w_t·sin(el)`. See `terminal_fall_speed` for the dBZ and
melting-layer blending. Requires a `[fields]` entry tagged `reflectivity`.

# Fields
- `rain_coefficient::Float64`: rain relation coefficient (m/s per Z^rain_exponent), default `2.6`.
- `rain_exponent::Float64`: rain relation exponent on linear Z (dimensionless), default `0.107`.
- `snow_coefficient::Float64`: snow relation coefficient (m/s per Z^snow_exponent), default `0.817`.
- `snow_exponent::Float64`: snow relation exponent on linear Z (dimensionless), default `0.063`.
- `density_exponent::Float64`: exponent on `ρ̄(0)/ρ̄(z)` (dimensionless), default `0.45`.
- `mixed_phase_dbz::Float64`: reflectivity (dBZ) at or below which the
  above-melting-layer fall speed is pure snow; default `20.0`.
- `rain_dbz::Float64`: reflectivity (dBZ) above which the above-melting-layer
  fall speed is pure rain; linear blend in dBZ between the two. Must exceed
  `mixed_phase_dbz`; default `30.0`.
- `melting_layer_bottom_m::Float64`: height (m, grid z convention) of the
  melting-layer bottom. `NaN` (default) ⇒ derived from the reference profile's
  freezing level ([`freezing_level_m`](@ref)); an error is raised at use if
  that profile carries no temperature.
- `melting_layer_depth_m::Float64`: melting-layer depth (m) over which the fall
  speed blends linearly in height from rain (bottom) to the above-layer value
  (top); default `1000.0` (samurai `melting_zone_width` = 1 km).
"""
Base.@kwdef struct FallSpeedParameters
    rain_coefficient::Float64       = 2.6
    rain_exponent::Float64          = 0.107
    snow_coefficient::Float64       = 0.817
    snow_exponent::Float64          = 0.063
    density_exponent::Float64       = 0.45
    mixed_phase_dbz::Float64        = 20.0
    rain_dbz::Float64               = 30.0
    melting_layer_bottom_m::Float64 = NaN
    melting_layer_depth_m::Float64  = 1000.0
end

"""
    GateErrorParameters

Per-gate radial-velocity error model, loaded from the optional
`[synthesis.gate_error]` sub-table (a present table means "enabled"; every key
is optional). Each gate's error standard deviation (m/s) is

    σ_m = max(error_guard_min_m_s, |sin(el)|·fall_speed_error_m_s + SW·spectrum_width_error_factor)

(samurai `radar_fallspeed_error` / `radar_sw_error` / `radar_min_error`); the SW
term is omitted when no `spectrum_width`-tagged field is configured or the gate
has no SW value. It is folded into the gridding weight as `w′ = w·σ²_vr/σ_m²`
(`σ²_vr = velocity_variance`), which keeps the stage-1 planes and sandwich
covariance formula valid and reduces to the unscaled weight when absent.

# Fields
- `fall_speed_error_m_s::Float64`: assumed fall-speed error (m/s), scaled by
  `|sin(el)|`; default `2.0`.
- `spectrum_width_error_factor::Float64`: multiplier (dimensionless) on the gate
  spectrum width (m/s); default `1.0`.
- `error_guard_min_m_s::Float64`: lower clamp (m/s) on σ_m, a stability guard
  that keeps near-horizontal gates from receiving unbounded weight; default `1.0`.
"""
Base.@kwdef struct GateErrorParameters
    fall_speed_error_m_s::Float64        = 2.0
    spectrum_width_error_factor::Float64 = 1.0
    error_guard_min_m_s::Float64         = 1.0
end

"""
    SynthesisParameters

Stage-1 dual-Doppler wind-retrieval QC, loaded from the optional `[synthesis]`
block. Quality is judged **only** by the per-component normalized uncertainty
(CEDRIC DTEST style), evaluated **in the output frame** — there is no radar or
gate count threshold, since geometry adequacy already shows up in σ. Masking is
non-destructive: where the system is solvable, the components and their σ are
always written; the quality flag records which σ threshold(s) failed.

# Fields
- `velocity_variance::Float64`: assumed radial-velocity variance `σ²_vr` (m/s)²;
  `1.0` ⇒ normalized (CEDRIC USTD/VSTD) σ units. It only rescales the
  covariance to physical units and never enters the gridding weights.
- `max_sigma::Vector{Float64}`: per-output-component maximum normalized σ,
  indexed by frame component. The thresholds are **independent** (one component
  may be well-determined while the other is not). For the stage-1
  `CartesianFrame`, `max_sigma[1]`/`max_sigma[2]` are σ_u / σ_v (TOML keys
  `max_sigma_1` / `max_sigma_2`); for future plane/polar frames they bind to
  in-plane/cross-plane or tangential/radial. Storing a vector means adding a
  named frame needs no schema change.
- `max_elevation::Float64`: maximum line-of-sight elevation (degrees) a gate may
  have to enter the **2-unknown** `(u, v)` normal system. The dual-Doppler
  approximation drops the vertical term `w·sin(el)`, so steep beams (airborne
  tail radars reach ±70°) contaminate the horizontal solve; gates above this
  angle still grid every scalar field but are excluded from the wind solve.
  Optional (TOML key `max_elevation`); defaults to `45.0`. Applied **only**
  when `include_vertical_velocity = false`: in the 3-unknown system steep beams
  observe `w` and are kept.
- `include_vertical_velocity::Bool`: accumulate the `(u, v, w)` 3×3 normal
  system (`g = [sin az cos el, cos az cos el, sin el]`) instead of the 2×2
  `(u, v)` one. Required by the variational stage (its super-obs need the `w`
  row). The stage-1 [`finalize_wind`](@ref) product is unchanged in meaning: it
  solves the `(u, v)` block (the dual-Doppler approximation applied post hoc).
  Optional; defaults to `false`.
- `fall_speed::Union{Nothing,FallSpeedParameters}`: hydrometeor fall-speed
  correction (`[synthesis.fall_speed]`); `nothing` (absent) ⇒ no correction.
- `gate_error::Union{Nothing,GateErrorParameters}`: per-gate error model
  (`[synthesis.gate_error]`); `nothing` (absent) ⇒ uniform `σ²_vr` per gate.
"""
Base.@kwdef struct SynthesisParameters
    velocity_variance::Float64 = 1.0
    max_sigma::Vector{Float64} = [2.0, 2.0]
    max_elevation::Float64     = 45.0
    include_vertical_velocity::Bool = false
    fall_speed::Union{Nothing,FallSpeedParameters} = nothing
    gate_error::Union{Nothing,GateErrorParameters} = nothing
end

"""
    VARIATIONAL_ROLE_VOCAB

Role vocabulary for `[variational.roles]` overrides and (in a later phase)
automatic state-variable role resolution: `"wind_u"`, `"wind_v"`, `"wind_w"`,
`"reflectivity"`, `"scalar"`.
"""
const VARIATIONAL_ROLE_VOCAB = ("wind_u", "wind_v", "wind_w", "reflectivity", "scalar")

"""
    VARIATIONAL_BACKGROUND_VOCAB

Allowed values of `[variational] background` (a flat vocabulary):
`"zero"`, `"synthesis"`, `"synthesis_uniform_sigma"`, `"synthesis_profile"`.
See [`VariationalParameters`](@ref) and the `Background` constructors.
"""
const VARIATIONAL_BACKGROUND_VOCAB = ("zero", "synthesis", "synthesis_uniform_sigma",
                                      "synthesis_profile")

"""
    VARIATIONAL_CONTINUITY_LATTICE_VOCAB

Allowed values of `[variational] mass_continuity_lattice`: `"nodes"` (the
B-spline nodes `x = xmin + n·DX`, `n = 0:cells`, per axis — the default
regular output grid; samurai's placement) or `"mish"` (every quadrature
point, `mubar` per cell per axis).
"""
const VARIATIONAL_CONTINUITY_LATTICE_VOCAB = ("nodes", "mish")

"""
    VARIATIONAL_QUALITY_FLAG_MAP_VOCAB

Allowed values of `[variational] quality_flag_map`, how the stage-1 `QFLAG`
on the mish is resampled to the regular output grid: `"nearest"` (the
nearest mish point on each axis), `"worst"` (bitwise OR over the mish
points within half a cell of the output point) or `"best"` (bitwise AND
over the same neighbourhood).
"""
const VARIATIONAL_QUALITY_FLAG_MAP_VOCAB = ("nearest", "worst", "best")

"""
    BackgroundErrorEntry

One state variable's background-error specification for the variational
control-variable filter chain `C = S∘D∘F` (`F` = mish-space filter, `D` =
diagonal σ_b scaling, `S` = spline projection), loaded from
`[variational.background_error]`.

# Fields
- `sigma::Float64`: background-error standard deviation σ_b for this state
  variable (m/s for wind components; native units for a `[fields]`-derived
  scalar state variable), applied as the diagonal scaling `D`. Must be
  positive.
- `filter_length_m::NTuple{3,Float64}`: mish-space filter length (metres)
  along each `(i, j, k)` axis, applied by `F` before `D`. `0.0` on an axis
  disables filtering on that axis; every entry must be finite and
  non-negative.
"""
Base.@kwdef struct BackgroundErrorEntry
    sigma::Float64
    filter_length_m::NTuple{3,Float64}
end

"""
    RadialVelocityObsParameters

Radial-velocity super-observation parameters, loaded from the optional
`[variational.radial_velocity]` sub-table (a present table narrows the
defaults below; every key is optional).

# Fields
- `error_inflation::Float64`: multiplier on `σ²_vr` (the stage-1
  `[synthesis] velocity_variance`) applied only to the super-observation
  inverse variance; default `1.0`. Must be positive.
- `horizontal_approximation::Bool`: drop the `w·sin(el)` term from the
  super-obs geometry vector (samurai `horizontal_radar_appx`), so the
  super-obs reduce to the dual-Doppler `(u, v)` approximation and
  `[synthesis] include_vertical_velocity = false` is permitted; default
  `false`.
- `eigenvalue_guard_relative::Float64`: eigendirections of the per-point
  stage-1 normal system below this fraction of its largest eigenvalue emit
  no super-observation row; default `1.0e-10`. Must lie in `[0, 1)`.
"""
Base.@kwdef struct RadialVelocityObsParameters
    error_inflation::Float64           = 1.0
    horizontal_approximation::Bool     = false
    eigenvalue_guard_relative::Float64 = 1.0e-10
end

"""
    ScalarObsEntry

One `[fields]`-derived scalar state variable's value-observation
parameters, loaded from `[variational.scalar_obs.<name>]`. Required for
every non-wind (`"U"`/`"V"`/`"W"`) entry in `[variational] state`.

# Fields
- `sigma::Float64`: observation-error standard deviation (native field
  units) for treating the gridded mish value as a direct observation of the
  scalar state variable. Must be positive.
- `undetect_value::Float64`: value substituted for a gate at `[io] undetect`
  (scanned, no detectable echo) before it enters the observation set, e.g.
  a reflectivity floor. A true-missing (`[io] fill_value`) gate contributes
  no observation.
"""
Base.@kwdef struct ScalarObsEntry
    sigma::Float64
    undetect_value::Float64
end

"""
    VariationalParameters

Configuration for the incremental 3DVAR wind solver, loaded from the
optional `[variational]` block. The whole block (including its sub-tables)
validates strictly whenever present, whether or not `enabled` is `true` —
disabling the analysis does not relax the schema.

# Fields
- `enabled::Bool`: master switch; default `false`.
- `state::Vector{String}`: state-vector variable names, in control-variable
  order. Each entry is `"U"`, `"V"`, `"W"`, or a `[fields]` name (appended
  as a value-observed scalar state variable). Must be non-empty, unique,
  and contain both `"U"` and `"V"`. Default `["U", "V", "W"]`.
- `background::String`: one of [`VARIATIONAL_BACKGROUND_VOCAB`](@ref).
  Default `"zero"`.
  - `"zero"`: background 0 everywhere with the configured
    `background_error` σ (samurai-like).
  - `"synthesis"`: `U`/`V` background = the smooth spline fit to the
    stage-1 `(u, v)` at unflagged points (one auxiliary variational solve),
    with σ_b a smoothed stage-1 σ map (clamped to `[σ/10, σ]` of the
    configured σ).
  - `"synthesis_uniform_sigma"`: the same smooth fit, with the configured
    (uniform) σ.
  - `"synthesis_profile"`: `U`/`V` background = the level-mean stage-1
    profile (constant on each mish level), with the configured σ.
  `W` and scalar state variables always have a zero background.
- `background_error_inflation::Float64`: multiplies σ_b of every variable
  in the three `synthesis*` modes, guarding against double-counting the
  radar data (the stage-1 background and the super-obs come from the same
  gates). Ignored for `"zero"`. Must be positive. Default `1.0`.
- `mass_continuity::Bool`: enforce the anelastic mass-continuity pseudo-
  observation (`∂u/∂x + ∂v/∂y + ∂w/∂z + w·dlnρ̄/dz = 0`) at every point of
  the `mass_continuity_lattice`. Requires `"W"` in `state`. Default `true`.
- `mass_continuity_lattice::String`: where the continuity pseudo-observations
  sit, one of [`VARIATIONAL_CONTINUITY_LATTICE_VOCAB`](@ref): `"nodes"`
  (the B-spline nodes, `cells + 1` per axis — the default regular output
  grid and samurai's placement) or `"mish"` (every quadrature point,
  `mubar³` constraints per cell). The per-point σ is the same in both;
  `"mish"` therefore carries `mubar³` times the constraint weight per cell.
  Default `"nodes"`.
- `mass_continuity_sigma_per_s::Float64`: standard deviation σ of the
  anelastic divergence residual (s⁻¹) at each constraint point; inverse
  variance is `1/σ²`. Must be positive. Default `1.0e-4`.
- `max_outer_iterations::Int`: outer Gauss–Newton iterations; only matters
  with nonlinear observations. Must be `≥ 1`. Default `1`.
- `max_inner_iterations::Int`: inner conjugate-gradient iterations per
  outer loop. Must be `≥ 1`. Default `200`.
- `inner_tolerance::Float64`: relative CG residual stopping tolerance; must
  lie in `(0, 1)`. Default `1.0e-4`.
- `outer_tolerance::Float64`: relative increment-norm stopping tolerance
  between outer loops; must lie in `(0, 1)`. Default `1.0e-3`.
- `cg_progress_interval::Int`: print one solver progress line (CG
  iteration, residual norm, relative residual, cost) every this many inner
  CG iterations, plus a header and a per-outer-iteration summary; `0`
  silences the solver. Must be `≥ 0`. Default `10`.
- `output_background::Bool`: write the `*_BACKGROUND` fields. Default
  `false`.
- `output_increment::Bool`: write the `*_INCREMENT` fields. Default
  `false`.
- `output_mish::Bool`: also write mish-lattice fields and stage-1/super-obs
  diagnostics. Default `false`.
- `quality_flag_map::String`: how the stage-1 `QFLAG` (on the mish) is
  resampled onto the regular output grid, one of
  [`VARIATIONAL_QUALITY_FLAG_MAP_VOCAB`](@ref): `"nearest"`, `"worst"`
  (bitwise OR within half a cell) or `"best"` (bitwise AND). Default
  `"nearest"`.
- `mask_quality_flags::Vector{String}`: flag meanings whose bits blank
  (`NaN`) the regular-grid wind outputs (`U`, `V`, `W`, their derivative,
  background and increment fields, `VORT`, `DIV`, `MCRESIDUAL`) where the
  resampled `QFLAG` carries any of them. Entries are the CF `flag_meanings`
  of `QFLAG`: `"sigma1_above_threshold"`, `"sigma2_above_threshold"`,
  `"singular_geometry"`, `"no_data"`. Empty (the default) masks nothing;
  `QFLAG` itself is always written.
- `filter_type::String`: mish-space background-error filter kernel,
  `"gaussian"` or `"lanczos"` (Gaussian has no negative lobes; Lanczos
  does, documented in the theory guide). Default `"gaussian"`.
- `background_error::Dict{String,BackgroundErrorEntry}`: per-variable
  [`BackgroundErrorEntry`](@ref), keyed by state name or `"default"`.
  Every state variable must resolve an entry (own or `"default"`) — see
  [`background_error_for`](@ref). Default a single `"default"` entry
  (`σ = 10.0`, `filter_length_m = (2000.0, 2000.0, 500.0)`).
- `radial_velocity::RadialVelocityObsParameters`: super-observation
  parameters ([`RadialVelocityObsParameters`](@ref)).
- `scalar_obs::Dict{String,ScalarObsEntry}`: one [`ScalarObsEntry`](@ref)
  per `[fields]` name appearing in `state`.
- `roles::Dict{String,Symbol}`: optional explicit state-variable role
  override (e.g. `W = "wind_w"`); keys must be in `state` and values drawn
  from [`VARIATIONAL_ROLE_VOCAB`](@ref).
"""
Base.@kwdef struct VariationalParameters
    enabled::Bool                          = false
    state::Vector{String}                  = ["U", "V", "W"]
    background::String                     = "zero"
    background_error_inflation::Float64    = 1.0
    mass_continuity::Bool                  = true
    mass_continuity_lattice::String        = "nodes"
    mass_continuity_sigma_per_s::Float64   = 1.0e-4
    max_outer_iterations::Int              = 1
    max_inner_iterations::Int              = 200
    inner_tolerance::Float64               = 1.0e-4
    outer_tolerance::Float64               = 1.0e-3
    cg_progress_interval::Int              = 10
    output_background::Bool                = false
    output_increment::Bool                 = false
    output_mish::Bool                      = false
    quality_flag_map::String               = "nearest"
    mask_quality_flags::Vector{String}     = String[]
    filter_type::String                    = "gaussian"
    background_error::Dict{String,BackgroundErrorEntry} =
        Dict("default" => BackgroundErrorEntry(10.0, (2000.0, 2000.0, 500.0)))
    radial_velocity::RadialVelocityObsParameters = RadialVelocityObsParameters()
    scalar_obs::Dict{String,ScalarObsEntry} = Dict{String,ScalarObsEntry}()
    roles::Dict{String,Symbol}              = Dict{String,Symbol}()
end

"""
    background_error_for(vp::VariationalParameters, name::String) -> BackgroundErrorEntry

Resolve the [`BackgroundErrorEntry`](@ref) for state variable `name`: its own
`[variational.background_error.<name>]` entry if present, else the
`"default"` entry, else an `ArgumentError`.
"""
function background_error_for(vp::VariationalParameters, name::String)
    haskey(vp.background_error, name) && return vp.background_error[name]
    haskey(vp.background_error, "default") && return vp.background_error["default"]
    throw(ArgumentError(
        "[variational.background_error]: no entry for `$(name)` and no " *
        "`default` entry to fall back to."))
end

"""
    ReferenceStateParameters

Hydrostatic reference-density profile ρ̄(z), loaded from the optional
`[reference_state]` block. Shared across consumers that need a base-state
density or temperature profile: variational-stage mass continuity
(`∂u/∂x + ∂v/∂y + ∂w/∂z + w·dlnρ̄/dz = 0`), radar fall-speed correction, and
(in a future phase) FHC temperature. See [`build_reference_profile`](@ref),
[`ExponentialReferenceProfile`](@ref), and [`SoundingReferenceProfile`](@ref).

# Fields
- `source::String`: `"exponential"` (default) for a simple analytic profile, or
  `"sounding"` to build a full hydrostatic reference state from a sounding file
  via Springsteel (`Springsteel.calculate_reference_state`).
- `sounding_file::String`: path to the sounding file; required (and must
  exist) when `source = "sounding"`. Format: a surface line `p_sfc theta q_v`
  (pressure in hPa, potential temperature in K, water-vapor mixing ratio in
  g/kg), followed by one `z theta q_v` line per level (`z` in metres, `q_v` in
  g/kg), matching the samurai/Scythe sounding format.
- `moisture::Bool`: sounding source only; when `true` (default) the reference
  density includes the vapor partial density (`ρ̄ = ρ̄_d + ρ̄_v`); when `false`
  the state is built dry (`ρ̄ = ρ̄_d`).
- `density_surface_kg_m3::Float64`: exponential source only; surface density
  ρ₀ in kg m⁻³ (default `1.2`).
- `density_scale_height_m::Float64`: exponential source only; scale height H
  in metres (default `8500.0`); ρ̄(z) = ρ₀·exp(−z/H).
- `temperature_surface_k::Float64`: exponential source only; optional surface
  temperature T₀ in Kelvin. `NaN` (the default) means no temperature profile
  is available — [`reference_temperature_kelvin`](@ref) and
  [`freezing_level_m`](@ref) then return `nothing`, and
  [`temperature_celsius`](@ref) throws.
- `lapse_rate_k_m::Float64`: exponential source only; lapse rate Γ in K/m
  (default `0.0065`, the US Standard Atmosphere troposphere value), used only
  when `temperature_surface_k` is finite: T(z) = T₀ − Γ·z.
"""
Base.@kwdef struct ReferenceStateParameters
    source::String                = "exponential"
    sounding_file::String         = ""
    moisture::Bool                = true
    density_surface_kg_m3::Float64 = 1.2
    density_scale_height_m::Float64 = 8500.0
    temperature_surface_k::Float64 = NaN
    lapse_rate_k_m::Float64       = 0.0065
end

"""
    EchoProductsParameters

Configuration for the post-gridding echo products — fuzzy hydrometeor
identification (HID/FHC) and polarimetric rain rate — applied to the *gridded*
radar variables (the beam-power-weighted averages), loaded from the optional
`[echo]` block. Both algorithms are CSU_RadarTools ports
([`csu_fhc_summer`](@ref), [`calc_blended_rain_tropical`](@ref)).

# Fields
- `enabled::Bool`: master switch; when `false` the gridding drivers skip echo
  products entirely.
- `band::String`: radar frequency band (`"S"`, `"C"`, `"X"`) selecting both the
  FHC membership functions and the rain-rate coefficients.
- `compute_fhc::Bool`: write the hydrometeor classification field.
- `compute_blended_rain::Bool`: write the blended rain-rate field.
- `rain_components::Vector{String}`: which individual rain components to also
  write, any subset of `RATE_Z`, `RATE_Z_CONV`, `RATE_Z_STRAT`, `RATE_KDP`,
  `RATE_Z_ZDR`, `RATE_KDP_ZDR`. These are computed directly from the gridded
  variables and are available even when `compute_blended_rain` is `false` — useful
  for visualization and as fallbacks when blended inputs (e.g. Kdp) are
  missing/noisy.
- `dbz_field`/`zdr_field`/`kdp_field`/`rhohv_field`/`ldr_field::String`: input
  field names to read from the grid; `ldr_field=""` disables LDR.
- `fhc_method::Symbol`: `:hybrid` (default) or `:linear`.
- `use_temp::Bool`: include temperature in the FHC.
- `temp_source::Symbol`: where temperature comes from — `:profile` (the
  `temperature` T(z) profile, default), `:field` (a gridded temperature field,
  `temp_field`), or `:reference_state` (future Springsteel reference state;
  currently errors at runtime).
- `temp_field::String`: gridded temperature field name used when
  `temp_source = :field` (default `"TEMP_FOR_PID"`).
- `temp_field_units::String`: units of `temp_field`, `"C"` (default) or `"K"`
  (converted to °C internally; the FHC works in °C).
- `height_field::String`: name of a gridded beam-height field (e.g. a field
  tagged `beam_height`). When set and present, the `:profile` temperature source
  samples the T(z) profile at this per-cell height — enabling temperature on 2-D
  PPI/RHI grids that have no Z axis. Empty (default) falls back to the grid's
  z-axis heights.
- `temp_factor::Float64`: broadens the temperature membership functions when > 1.
- `correct_ice_method::Bool`: label rain method code `2` only on hail cells
  (default `true`); `false` replicates the Python ice-method bug exactly.
- `fhc_output`/`rain_output`/`rain_method_output::String`: output variable names.
  `rain_method_output=""` suppresses the method field.
- `weights::Dict{Symbol,Float64}`: per-variable FHC weights (keys `:DZ, :DR, :KD,
  :RH, :LD, :T`).
- `temperature::Union{TemperatureProfile,Nothing}`: vertical T(z) profile (°C),
  required when `use_temp` is `true` and `temp_source = :profile`. A temporary
  stand-in for a future shared hydrostatic reference state (see
  `temperature_profile.jl`).
"""
Base.@kwdef struct EchoProductsParameters
    enabled::Bool                 = false
    band::String                  = "S"
    compute_fhc::Bool             = true
    compute_blended_rain::Bool    = true
    rain_components::Vector{String} = String[]
    dbz_field::String             = "DBZ"
    zdr_field::String             = "ZDR"
    kdp_field::String             = "KDP"
    rhohv_field::String           = "RHOHV"
    ldr_field::String             = ""
    fhc_method::Symbol            = :hybrid
    use_temp::Bool                = true
    temp_source::Symbol           = :profile
    temp_field::String            = "TEMP_FOR_PID"
    temp_field_units::String      = "C"
    height_field::String          = ""
    temp_factor::Float64          = 1.0
    correct_ice_method::Bool      = true
    fhc_output::String            = "HID_CSU"
    rain_output::String           = "RATE_CSU_BLENDED"
    rain_method_output::String    = ""
    weights::Dict{Symbol,Float64} = Dict(:DZ => 1.5, :DR => 0.8, :KD => 1.0,
                                          :RH => 0.8, :LD => 0.5, :T => 0.4)
    temperature::Union{TemperatureProfile,Nothing} = nothing
end

"""
    HybridScanParameters

Configuration for the hybrid-scan product, loaded from the optional
`[hybrid_scan]` block. A hybrid scan collapses a set of same-time gridded PPI
tilts into one near-surface 2-D field set: it starts from a base (lowest) tilt
and, wherever that tilt has no measurement, looks upward through the higher tilts
and takes the first one whose beam is still below `beam_height_maximum` at that
cell. See [`apply_hybrid_scan`](@ref) and [`build_hybrid_scan`](@ref).

It runs *after* the `[echo]` products, since the fields it usually carries
(blended rain rate, hydrometeor ID) are echo outputs.

# Fields
- `enabled::Bool`: master switch; when `false` the drivers skip the product.
- `fields::Vector{String}`: which gridded fields to carry through the selection.
  Empty (the default) means *every* field present in the tilt files, which keeps
  the output a drop-in gridded PPI that the existing readers and plotters accept.
- `select_field::String`: the field whose validity decides the selection at each
  cell — the hybrid-scan analogue of the `define_detection` tag.
- `base_angle::Union{Float64,Nothing}`: preferred base tilt in degrees. When the
  configured angle is not among the available tilts (or is `nothing`), the lowest
  available tilt is used instead.
- `base_angle_tolerance::Float64`: how close (degrees) an available tilt must be
  to `base_angle` to count as it.
- `beam_height_maximum::Float64`: inclusion limit in metres. A tilt above the base
  may only supply a cell where its beam height there is at or below this. The base
  tilt itself is never height-limited — it is the best available look by definition.
- `radar_altitude::Float64`: antenna height in metres, used by the geometric
  beam-height fallback.
- `require_detection::Bool`: what counts as a tilt having *answered* a cell, applied
  identically at the base tilt and at the tilts above it. `false` (default): any
  non-`fill_value` reading answers, so a tilt reporting `undetect` has observed clear
  air and ends the search — the product is the lowest measurement of any kind. `true`:
  only a real detection answers, so the search climbs past clear air looking for echo,
  reproducing the original PICCOLO script.
- `height_field::String`: name of a gridded beam-height field (e.g. one tagged
  `beam_height`). When set and present in a tilt, it supplies that tilt's per-cell
  heights in preference to the geometric fallback.
- `elevation_output::String`: output variable recording which tilt supplied each
  cell. Empty suppresses it.
- `height_output::String`: output variable recording the selected beam height.
  Empty (default) suppresses it.
- `angle_pattern::String`: fallback regular expression, with one capture group, for
  recovering a tilt angle from a gridded PPI *filename* when the file predates the
  `fixed_angle` variable Daisho now writes. Empty (default) means attribute only.
"""
Base.@kwdef struct HybridScanParameters
    enabled::Bool                        = false
    fields::Vector{String}               = String[]
    select_field::String                 = "DBZ"
    base_angle::Union{Float64,Nothing}   = nothing
    base_angle_tolerance::Float64        = 0.05
    beam_height_maximum::Float64         = 1000.0
    radar_altitude::Float64              = 0.0
    require_detection::Bool              = false
    height_field::String                 = ""
    elevation_output::String             = "elevation_angle"
    height_output::String                = ""
    angle_pattern::String                = ""
end

"""
    DaishoParameters

Top-level immutable runtime configuration loaded from a TOML file. Construct
via `DaishoParameters()` for the bundled template or `DaishoParameters(path)`
for a user file.

`[fields]` and `[io]` are mandatory (no sensible default: every driver needs
fields, and the two `[io]` sentinels distinguish true-missing vs undetect on
every NetCDF write). `[qc]`, `[gridding]`, and the `[grid.*]` sub-tables are
optional and default-construct when absent; an operation that actually needs a
missing block raises a clear point-of-use error (see [`require_section`](@ref))
rather than silently gridding with template numbers. Validation *within* any
present block stays strict.

# Fields
- `moments::MomentParameters`
- `qc::QCParameters`
- `gridding::GriddingParameters`
- `grid::GridParameters`
- `io::IOParameters`
- `synthesis::SynthesisParameters`
- `echo::EchoProductsParameters`
- `hybrid_scan::HybridScanParameters`
- `reference_state::ReferenceStateParameters`
- `variational::VariationalParameters`
- `provided::Set{Symbol}`: which optional top-level sections (`:qc`,
  `:gridding`, `:grid`, `:synthesis`, `:echo`, `:hybrid_scan`,
  `:reference_state`, `:variational`) were actually present in the loaded
  TOML. Absence is tracked explicitly — it is *not* inferred from default
  values.

# Examples
```julia
print_config("mygrid.toml")            # write the template
# edit mygrid.toml for your radar, grid, and CF metadata
p = DaishoParameters("mygrid.toml")    # strict load
threshold_qc!(sweep, p)
grid_radar_volume(volume, "out.nc", time, p)
```
"""
struct DaishoParameters
    moments::MomentParameters
    qc::QCParameters
    gridding::GriddingParameters
    grid::GridParameters
    io::IOParameters
    synthesis::SynthesisParameters
    echo::EchoProductsParameters
    hybrid_scan::HybridScanParameters
    reference_state::ReferenceStateParameters
    variational::VariationalParameters
    provided::Set{Symbol}
end

# Back-compat convenience: callers that built a `DaishoParameters` before the
# `[synthesis]` block existed pass six positional args (…, io, provided). Inject
# default synthesis and echo blocks so those call sites keep working.
DaishoParameters(moments::MomentParameters, qc::QCParameters,
                 gridding::GriddingParameters, grid::GridParameters,
                 io::IOParameters, provided::Set{Symbol}) =
    DaishoParameters(moments, qc, gridding, grid, io, SynthesisParameters(),
                     EchoProductsParameters(), HybridScanParameters(), provided)

# Back-compat convenience for callers built before the `[echo]` block existed
# (seven positional args ending …, synthesis, provided). Inject a default
# echo block.
DaishoParameters(moments::MomentParameters, qc::QCParameters,
                 gridding::GriddingParameters, grid::GridParameters,
                 io::IOParameters, synthesis::SynthesisParameters,
                 provided::Set{Symbol}) =
    DaishoParameters(moments, qc, gridding, grid, io, synthesis,
                     EchoProductsParameters(), HybridScanParameters(), provided)

# Back-compat convenience for callers built before the `[hybrid_scan]` block
# existed (eight positional args ending …, echo, provided).
DaishoParameters(moments::MomentParameters, qc::QCParameters,
                 gridding::GriddingParameters, grid::GridParameters,
                 io::IOParameters, synthesis::SynthesisParameters,
                 echo::EchoProductsParameters, provided::Set{Symbol}) =
    DaishoParameters(moments, qc, gridding, grid, io, synthesis, echo,
                     HybridScanParameters(), provided)

# Back-compat convenience for callers built before the `[reference_state]`
# block existed (nine positional args ending …, hybrid_scan, provided).
# Inject a default reference_state block.
DaishoParameters(moments::MomentParameters, qc::QCParameters,
                 gridding::GriddingParameters, grid::GridParameters,
                 io::IOParameters, synthesis::SynthesisParameters,
                 echo::EchoProductsParameters, hybrid_scan::HybridScanParameters,
                 provided::Set{Symbol}) =
    DaishoParameters(moments, qc, gridding, grid, io, synthesis, echo,
                     hybrid_scan, ReferenceStateParameters(), provided)

# Back-compat convenience for callers built before the `[variational]` block
# existed (ten positional args ending …, reference_state, provided). Inject a
# default variational block.
DaishoParameters(moments::MomentParameters, qc::QCParameters,
                 gridding::GriddingParameters, grid::GridParameters,
                 io::IOParameters, synthesis::SynthesisParameters,
                 echo::EchoProductsParameters, hybrid_scan::HybridScanParameters,
                 reference_state::ReferenceStateParameters,
                 provided::Set{Symbol}) =
    DaishoParameters(moments, qc, gridding, grid, io, synthesis, echo,
                     hybrid_scan, reference_state, VariationalParameters(), provided)

# ── Loader ───────────────────────────────────────────────────────────────────

"""
    DaishoParameters() -> DaishoParameters
    DaishoParameters(path::AbstractString) -> DaishoParameters

Construct a `DaishoParameters` instance. With no arguments, loads the bundled
template from `config/defaults.toml`. With a path, loads only the user's TOML
— there is no silent fallback to bundled defaults. `[fields]` and `[io]` must
be present; `[qc]`, `[gridding]`, and `[grid.*]` are optional and
default-construct when absent. Any block that *is* present is validated
strictly (unknown or missing keys raise an `ArgumentError` naming the item).

Use [`print_config`](@ref) to write a complete starter template to a file you
can edit.
"""
DaishoParameters() = DaishoParameters(_load_toml(DEFAULTS_TOML_PATH))
DaishoParameters(path::AbstractString) = DaishoParameters(_load_toml(path))

# Internal: build from an already-parsed Dict (used by ctors and tests).
# `[fields]`/`[io]` mandatory; `[qc]`/`[gridding]`/`[grid.*]` optional. The
# `provided` set records which optional sections were actually present — never
# inferred from default values, since a defaulted block still has usable
# numbers and must not silently drive a grid.
function DaishoParameters(d::AbstractDict)
    provided = Set{Symbol}()
    for s in (:qc, :gridding, :grid, :synthesis, :echo, :hybrid_scan, :reference_state, :variational)
        haskey(d, string(s)) && push!(provided, s)
    end

    moments   = _fields_from_dict(_section(d, "fields"))
    qc        = haskey(d, "qc")        ? _struct_from_dict(QCParameters, d["qc"]; section="qc") : QCParameters()
    gridding  = haskey(d, "gridding")  ? _gridding_from_dict(d["gridding"])                     : GriddingParameters()
    grid      = haskey(d, "grid")      ? _grid_from_dict(d["grid"])                             : GridParameters()
    io        = _io_from_dict(_section(d, "io"))
    synthesis = haskey(d, "synthesis") ? _synthesis_from_dict(d["synthesis"])                   : SynthesisParameters()
    echo      = haskey(d, "echo")      ? _echo_from_dict(d["echo"])                             : EchoProductsParameters()
    hybrid    = haskey(d, "hybrid_scan") ? _hybrid_scan_from_dict(d["hybrid_scan"])             : HybridScanParameters()
    reference_state = haskey(d, "reference_state") ? _reference_state_from_dict(d["reference_state"]) : ReferenceStateParameters()
    variational = haskey(d, "variational") ? _variational_from_dict(d["variational"], moments) : VariationalParameters()
    return DaishoParameters(moments, qc, gridding, grid, io, synthesis, echo, hybrid,
                            reference_state, variational, provided)
end

"""
    require_section(p, root::Symbol, sub::Symbol...; for_op::String) -> Any

Point-of-use accessor for an optional section. Raises a clear `ArgumentError`
naming the operation and the missing block if `root` was absent from the
loaded TOML; otherwise returns `p.root` drilled through any `sub` fields.
"""
function require_section(p, root::Symbol, sub::Symbol...; for_op::String)
    root in p.provided || throw(ArgumentError(
        "Operation `$for_op` needs the `[$root]` section, but it was " *
        "absent from the loaded TOML. Add it (run " *
        "`print_config(\"template.toml\")` for the full template)."))
    obj = getfield(p, root)
    for s in sub
        obj = getfield(obj, s)
    end
    return obj
end

"""
    print_config([io::IO = stdout])
    print_config(path::AbstractString; force::Bool=false) -> String

Write the bundled Daisho parameter template (currently `config/defaults.toml`)
to `io`, or copy it to a file at `path`. The file form refuses to overwrite an
existing file unless `force=true`, and returns the destination path.

`DaishoParameters(path)` requires every documented key to be present in the
file you pass it, so the typical workflow is:

```julia
using Daisho
print_config("mygrid.toml")          # write template
# edit mygrid.toml for your radar, grid, and CF metadata
p = DaishoParameters("mygrid.toml")  # strict load
```
"""
function print_config(io::IO=stdout)
    write(io, read(DEFAULTS_TOML_PATH))
    return nothing
end

function print_config(path::AbstractString; force::Bool=false)
    if isfile(path) && !force
        throw(ArgumentError(
            "print_config: refusing to overwrite existing file `$path` " *
            "(pass `force=true` to overwrite)"))
    end
    cp(DEFAULTS_TOML_PATH, path; force=true)
    return path
end

function _load_toml(path::AbstractString)
    isfile(path) || throw(ArgumentError("Daisho parameter file not found: $path"))
    return TOML.parsefile(path)
end

# Look up a required sub-table. Missing sections are user errors, not
# something to paper over with empty dicts.
function _section(d::AbstractDict, key::String; parent::String="")
    if !haskey(d, key)
        full = isempty(parent) ? key : "$(parent).$(key)"
        throw(ArgumentError(
            "Missing required TOML section `[$(full)]`. " *
            "Run `print_config(\"template.toml\")` to write a complete template."))
    end
    return d[key]
end

# Build a struct by pulling matching keys out of `d`. Unknown keys are
# reported (so users notice typos); missing keys are reported (so silent
# fallback to defaults can't happen).
function _struct_from_dict(::Type{T}, d::AbstractDict; section::String="",
        optional::Set{Symbol}=Set{Symbol}()) where {T}
    fields = fieldnames(T)
    field_set = Set(fields)
    keysyms = Set(Symbol.(keys(d)))

    unknown = setdiff(keysyms, field_set)
    if !isempty(unknown)
        throw(ArgumentError(
            "Unknown key(s) $(_fmt_keys(unknown)) in section " *
            "`[$(section)]` for $(T). " *
            "Allowed keys: $(join(fields, ", "))"))
    end

    # `optional` fields default (via the struct's @kwdef) when absent.
    missing_keys = setdiff(field_set, keysyms, optional)
    if !isempty(missing_keys)
        throw(ArgumentError(
            "Missing required key(s) $(_fmt_keys(missing_keys)) in section " *
            "`[$(section)]` for $(T). " *
            "Run `print_config(\"template.toml\")` for a complete template."))
    end

    kw = Dict{Symbol,Any}()
    for (k, v) in d
        sym = Symbol(k)
        kw[sym] = _coerce_field(T, sym, v)
    end
    return T(; kw...)
end

_fmt_keys(keys) = "`" * join(sort(string.(collect(keys))), "`, `") * "`"

# Coerce loaded values to the field's declared type. Catches the common
# Int/Float64 mismatch (TOML's `0.0` is Float64, `0` is Int).
function _coerce_field(::Type{T}, sym::Symbol, v) where {T}
    F = fieldtype(T, sym)
    if F === Symbol && v isa AbstractString
        return Symbol(v)
    elseif F === Float64 && v isa Real
        return Float64(v)
    elseif F === Union{Float64,Nothing} && v isa Real
        return Float64(v)
    elseif F === Int && v isa Integer
        return Int(v)
    else
        return v
    end
end

# ── Section-specific loaders ────────────────────────────────────────────────

function _fields_from_dict(d::AbstractDict)
    # Migration diagnostic: the old shape used a `names` array plus a
    # `[fields.grid_type]` sub-table. Diagnose, don't parse it.
    if haskey(d, "names") || haskey(d, "grid_type")
        throw(ArgumentError(
            "[fields] now maps each field to a flat array of tags, e.g.\n" *
            "    DBZ = [\"linear_interp\", \"define_detection\"]\n" *
            "    SQI = [\"weighted_interp\", \"define_scanned\"]\n" *
            "The old `names = [...]` array and `[fields.grid_type]` sub-table " *
            "are gone; interpolation is the `*_interp` tag. Also remove " *
            "`missing_key`/`valid_key` from `[gridding]` and express them as " *
            "the `define_scanned` / `define_detection` field tags. " *
            "Run `print_config(\"template.toml\")` for the new template."))
    end

    isempty(d) && throw(ArgumentError(
        "[fields]: at least one field must be declared."))

    vocab = Set(FIELD_TAG_VOCAB)
    interp = Set(FIELD_INTERP_TAGS)
    specs = FieldSpec[]
    for (name, raw) in d
        raw isa AbstractArray || throw(ArgumentError(
            "[fields]: field `$(name)` must map to an array of tags " *
            "(a TOML array `[ ... ]`), got $(typeof(raw)). " *
            "Example: $(name) = [\"weighted_interp\"]"))
        tags = Set{Symbol}()
        for x in raw
            t = Symbol(String(x))
            t in vocab || throw(ArgumentError(
                "[fields]: unknown tag `$t` on field `$(name)`. " *
                "Allowed tags: $(join(FIELD_TAG_VOCAB, ", "))"))
            push!(tags, t)
        end
        if length(intersect(tags, interp)) > 1
            throw(ArgumentError(
                "[fields]: field `$(name)` has more than one interpolation " *
                "tag $(sort(collect(intersect(tags, interp)))); at most one of " *
                "$(join(FIELD_INTERP_TAGS, ", ")) is allowed."))
        end
        push!(specs, FieldSpec(String(name), tags))
    end

    # Each singular role tag may appear on at most one field. Presence is NOT
    # required here — `field_with_tag` enforces presence at point-of-use.
    for tag in FIELD_SINGULAR_TAGS
        carriers = [fs.name for fs in specs if tag in fs.tags]
        length(carriers) > 1 && throw(ArgumentError(
            "[fields]: tag `$tag` is on multiple fields $(sort(carriers)); " *
            "it must be on at most one."))
    end

    return MomentParameters(specs)
end

# Build GriddingParameters, with a targeted migration diagnostic for the old
# missing_key/valid_key keys (clearer than _struct_from_dict's generic
# "unknown key" message).
function _gridding_from_dict(d::AbstractDict)
    if haskey(d, "missing_key") || haskey(d, "valid_key")
        throw(ArgumentError(
            "`[gridding]` no longer takes `missing_key`/`valid_key`. The " *
            "gate-role moments are now declared as field tags: tag the " *
            "scanned-indicator field with `define_scanned` (was " *
            "`missing_key`) and the detection field with `define_detection` " *
            "(was `valid_key`) in `[fields]`. " *
            "Run `print_config(\"template.toml\")` for the new template."))
    end
    if haskey(d, "range_floor")
        throw(ArgumentError(
            "`[gridding]` `range_floor` has been renamed to `range_guard_min` " *
            "(the near-radar minimum on the range-weight divisor, in metres). " *
            "Rename the key; note it is distinct from the new gate-inclusion " *
            "`range_minimum`/`range_maximum` bounds. " *
            "Run `print_config(\"template.toml\")` for the new template."))
    end
    # The ROI factors and numerical guards are optional (default when absent) so
    # configs predating them still load; `power_threshold` stays required. The
    # deprecated `beam_inflation` key is accepted and silently ignored (short
    # deprecation path; per-range reach is now beamwidth/footprint-derived).
    required   = (:power_threshold,)
    optional   = (:horizontal_roi_factor, :vertical_roi_factor, :range_guard_min,
                  :range_weight_max, :range_minimum, :range_maximum)
    deprecated = (:beam_inflation,)
    allowed = (required..., optional..., deprecated...)
    keysyms = Set(Symbol.(keys(d)))
    unknown = setdiff(keysyms, Set(allowed))
    isempty(unknown) || throw(ArgumentError(
        "Unknown key(s) $(_fmt_keys(unknown)) in section `[gridding]`. " *
        "Allowed keys: $(join((required..., optional...), ", "))"))
    missing_keys = setdiff(Set(required), keysyms)
    isempty(missing_keys) || throw(ArgumentError(
        "Missing required key(s) $(_fmt_keys(missing_keys)) in section " *
        "`[gridding]`. Run `print_config(\"template.toml\")` for a complete " *
        "template."))
    kw = Dict{Symbol,Any}()
    for (k, v) in d
        sym = Symbol(k)
        sym in deprecated && continue   # accept and ignore
        kw[sym] = _coerce_field(GriddingParameters, sym, v)
    end
    gp = GriddingParameters(; kw...)
    gp.range_minimum >= 0.0 || throw(ArgumentError(
        "`[gridding]` `range_minimum` must be ≥ 0 (got $(gp.range_minimum))."))
    gp.range_maximum >= gp.range_minimum || throw(ArgumentError(
        "`[gridding]` `range_maximum` ($(gp.range_maximum)) must be ≥ " *
        "`range_minimum` ($(gp.range_minimum))."))
    return gp
end

# Build IOParameters (mandatory). Targeted migration diagnostic for the old
# fill_value_missing/fill_value_clear keys (clearer than the generic message).
function _io_from_dict(d::AbstractDict)
    if haskey(d, "fill_value_missing") || haskey(d, "fill_value_clear")
        throw(ArgumentError(
            "`[io]` keys were renamed to `fill_value` / `undetect` to match " *
            "the CfRadial 2.1 / ODIM data model: `fill_value_missing` → " *
            "`fill_value` (CF _FillValue, true missing), `fill_value_clear` " *
            "→ `undetect` (ODIM _Undetect, scanned no echo). " *
            "Run `print_config(\"template.toml\")` for the new template."))
    end
    return _struct_from_dict(IOParameters, d; section="io")
end

# Samurai configuration keys that have a Daisho counterpart under `[synthesis]`
# (or one of its sub-tables); a migration diagnostic names the Daisho key.
const _SAMURAI_SYNTHESIS_KEYS = Dict(
    "radar_fallspeed_error" => "`fall_speed_error_m_s` in `[synthesis.gate_error]`",
    "radar_sw_error"        => "`spectrum_width_error_factor` in `[synthesis.gate_error]`",
    "radar_min_error"       => "`error_guard_min_m_s` in `[synthesis.gate_error]`",
    "horizontal_radar_appx" => "`include_vertical_velocity` in `[synthesis]` " *
                               "(samurai `horizontal_radar_appx = true` ≈ " *
                               "`include_vertical_velocity = false`)",
    "max_radar_elevation"   => "`max_elevation` in `[synthesis]` (degrees)",
    "melting_zone_width"    => "`melting_layer_depth_m` in `[synthesis.fall_speed]` " *
                               "(metres; samurai used km)",
)

function _check_samurai_synthesis_keys(d::AbstractDict, section::String)
    for k in keys(d)
        ks = String(k)
        if haskey(_SAMURAI_SYNTHESIS_KEYS, ks)
            throw(ArgumentError(
                "`[$section]` key `$ks` is a samurai configuration key. Its " *
                "Daisho equivalent is $(_SAMURAI_SYNTHESIS_KEYS[ks]). " *
                "Run `print_config(\"template.toml\")` for the documented template."))
        end
    end
    return nothing
end

# Build an all-optional `@kwdef` sub-table struct strictly (unknown keys raise,
# naming the section).
_optional_subtable(::Type{T}, d::AbstractDict, section::String) where {T} =
    _struct_from_dict(T, d; section = section, optional = Set(fieldnames(T)))

function _fall_speed_from_dict(d::AbstractDict)
    _check_samurai_synthesis_keys(d, "synthesis.fall_speed")
    fs = _optional_subtable(FallSpeedParameters, d, "synthesis.fall_speed")
    for f in (:rain_coefficient, :rain_exponent, :snow_coefficient,
              :snow_exponent, :density_exponent)
        v = getfield(fs, f)
        (isfinite(v) && v > 0.0) || throw(ArgumentError(
            "`[synthesis.fall_speed]` `$f` must be positive (got $v)."))
    end
    fs.mixed_phase_dbz < fs.rain_dbz || throw(ArgumentError(
        "`[synthesis.fall_speed]` `mixed_phase_dbz` ($(fs.mixed_phase_dbz)) must " *
        "be less than `rain_dbz` ($(fs.rain_dbz))."))
    (isfinite(fs.melting_layer_depth_m) && fs.melting_layer_depth_m > 0.0) ||
        throw(ArgumentError("`[synthesis.fall_speed]` `melting_layer_depth_m` " *
            "must be positive (got $(fs.melting_layer_depth_m))."))
    return fs
end

function _gate_error_from_dict(d::AbstractDict)
    _check_samurai_synthesis_keys(d, "synthesis.gate_error")
    ge = _optional_subtable(GateErrorParameters, d, "synthesis.gate_error")
    (isfinite(ge.fall_speed_error_m_s) && ge.fall_speed_error_m_s >= 0.0) ||
        throw(ArgumentError("`[synthesis.gate_error]` `fall_speed_error_m_s` " *
            "must be ≥ 0 (got $(ge.fall_speed_error_m_s))."))
    (isfinite(ge.spectrum_width_error_factor) && ge.spectrum_width_error_factor >= 0.0) ||
        throw(ArgumentError("`[synthesis.gate_error]` `spectrum_width_error_factor` " *
            "must be ≥ 0 (got $(ge.spectrum_width_error_factor))."))
    (isfinite(ge.error_guard_min_m_s) && ge.error_guard_min_m_s > 0.0) ||
        throw(ArgumentError("`[synthesis.gate_error]` `error_guard_min_m_s` " *
            "must be positive (got $(ge.error_guard_min_m_s))."))
    return ge
end

# Build SynthesisParameters from the flat `[synthesis]` keys. The per-component
# σ thresholds live under the scalar TOML keys `max_sigma_1` / `max_sigma_2`
# (more legible than a TOML array) but are stored as the `max_sigma` vector.
# Strict: unknown or missing keys raise, matching the rest of the loader. The
# optional `fall_speed` / `gate_error` sub-tables are each parsed strictly; a
# present sub-table means "enabled".
function _synthesis_from_dict(d::AbstractDict)
    _check_samurai_synthesis_keys(d, "synthesis")
    required = ("velocity_variance", "max_sigma_1", "max_sigma_2")
    optional = ("max_elevation", "include_vertical_velocity")
    subtables = ("fall_speed", "gate_error")
    allowed = (required..., optional..., subtables...)
    unknown = setdiff(Set(Symbol.(keys(d))), Set(Symbol.(allowed)))
    isempty(unknown) || throw(ArgumentError(
        "Unknown key(s) $(_fmt_keys(unknown)) in section `[synthesis]`. " *
        "Allowed keys: $(join(allowed, ", "))"))
    missing_keys = setdiff(Set(Symbol.(required)), Set(Symbol.(keys(d))))
    isempty(missing_keys) || throw(ArgumentError(
        "Missing required key(s) $(_fmt_keys(missing_keys)) in section " *
        "`[synthesis]`. Run `print_config(\"template.toml\")` for a complete template."))
    for t in subtables
        haskey(d, t) && !(d[t] isa AbstractDict) && throw(ArgumentError(
            "`[synthesis]` `$t` must be a sub-table `[synthesis.$t]`, got " *
            "$(typeof(d[t]))."))
    end
    ivv = get(d, "include_vertical_velocity", false)
    ivv isa Bool || throw(ArgumentError(
        "`[synthesis]` `include_vertical_velocity` must be a boolean (true/false), " *
        "got $(repr(ivv))."))
    return SynthesisParameters(
        velocity_variance = Float64(d["velocity_variance"]),
        max_sigma = [Float64(d["max_sigma_1"]), Float64(d["max_sigma_2"])],
        max_elevation = haskey(d, "max_elevation") ? Float64(d["max_elevation"]) :
                        SynthesisParameters().max_elevation,
        include_vertical_velocity = ivv,
        fall_speed = haskey(d, "fall_speed") ? _fall_speed_from_dict(d["fall_speed"]) : nothing,
        gate_error = haskey(d, "gate_error") ? _gate_error_from_dict(d["gate_error"]) : nothing,
    )
end

# Build a TemperatureProfile from a `[echo.temperature]` sub-table. Accepts a
# 2-column `file`, paired `heights`/`temperatures` arrays, or a `profile` list of
# `[height_m, temperature_C]` pairs.
function _temperature_from_dict(d::AbstractDict)
    if haskey(d, "file")
        return read_temperature_profile(String(d["file"]))
    elseif haskey(d, "heights") && haskey(d, "temperatures")
        return TemperatureProfile(Float64.(d["heights"]), Float64.(d["temperatures"]))
    elseif haskey(d, "profile")
        return TemperatureProfile([(Float64(p[1]), Float64(p[2])) for p in d["profile"]])
    end
    throw(ArgumentError("`[echo.temperature]` needs either `file`, or " *
        "`heights` + `temperatures`, or a `profile` list of [height, temp] pairs."))
end

# Build EchoProductsParameters from the optional `[echo]` block. Scalar keys
# default when absent (a large convenience block) but unknown keys raise to catch
# typos, matching the rest of the loader. `weights` and `temperature` are
# sub-tables.
function _echo_from_dict(d::AbstractDict)
    scalar_fields = (:enabled, :band, :compute_fhc, :compute_blended_rain,
        :rain_components, :dbz_field, :zdr_field, :kdp_field, :rhohv_field,
        :ldr_field, :fhc_method, :use_temp, :temp_source, :temp_field,
        :temp_field_units, :height_field, :temp_factor, :correct_ice_method,
        :fhc_output, :rain_output, :rain_method_output)
    subtables = (:weights, :temperature)
    allowed = Set((scalar_fields..., subtables...))
    unknown = setdiff(Set(Symbol.(keys(d))), allowed)
    isempty(unknown) || throw(ArgumentError(
        "Unknown key(s) $(_fmt_keys(unknown)) in section `[echo]`. " *
        "Allowed keys: $(join((scalar_fields..., subtables...), ", "))"))

    valid_components = ("RATE_Z", "RATE_Z_CONV", "RATE_Z_STRAT", "RATE_KDP",
                        "RATE_Z_ZDR", "RATE_KDP_ZDR")
    kw = Dict{Symbol,Any}()
    for f in scalar_fields
        sk = String(f)
        haskey(d, sk) || continue
        if f === :rain_components
            comps = String.(d[sk])
            for c in comps
                c in valid_components || throw(ArgumentError(
                    "`[echo]` rain_components entry \"$c\" is not valid. " *
                    "Allowed: $(join(valid_components, ", "))"))
            end
            kw[f] = comps
        else
            kw[f] = _coerce_field(EchoProductsParameters, f, d[sk])
        end
    end

    if haskey(d, "weights")
        w = Dict{Symbol,Float64}(:DZ => 1.5, :DR => 0.8, :KD => 1.0,
                                 :RH => 0.8, :LD => 0.5, :T => 0.4)
        for (k, v) in d["weights"]
            ks = Symbol(k)
            ks in keys(w) || throw(ArgumentError(
                "Unknown weight key `$k` in `[echo.weights]`; allowed: " *
                "DZ, DR, KD, RH, LD, T"))
            w[ks] = Float64(v)
        end
        kw[:weights] = w
    end

    if haskey(d, "temperature")
        kw[:temperature] = _temperature_from_dict(d["temperature"])
    end

    dp = EchoProductsParameters(; kw...)

    dp.temp_source in (:profile, :field, :reference_state) || throw(ArgumentError(
        "`[echo]` temp_source = \"$(dp.temp_source)\" is not valid; allowed: " *
        "\"profile\", \"field\", \"reference_state\"."))
    dp.temp_field_units in ("C", "K") || throw(ArgumentError(
        "`[echo]` temp_field_units = \"$(dp.temp_field_units)\" is not valid; " *
        "allowed: \"C\", \"K\"."))
    if dp.use_temp
        if dp.temp_source === :profile && dp.temperature === nothing
            throw(ArgumentError(
                "`[echo]` has `use_temp = true` with `temp_source = \"profile\"` " *
                "but no `[echo.temperature]` profile. Add a temperature profile, " *
                "switch `temp_source`, or set `use_temp = false`."))
        elseif dp.temp_source === :field && isempty(dp.temp_field)
            throw(ArgumentError(
                "`[echo]` has `temp_source = \"field\"` but `temp_field` is empty; " *
                "set it to the gridded temperature field name."))
        end
    end
    return dp
end

# Build HybridScanParameters from the optional `[hybrid_scan]` block. Every key
# defaults when absent; unknown keys raise to catch typos, matching the rest of
# the loader.
function _hybrid_scan_from_dict(d::AbstractDict)
    allowed_fields = (:enabled, :fields, :select_field, :base_angle,
        :base_angle_tolerance, :beam_height_maximum, :radar_altitude,
        :require_detection, :height_field, :elevation_output, :height_output,
        :angle_pattern)
    unknown = setdiff(Set(Symbol.(keys(d))), Set(allowed_fields))
    isempty(unknown) || throw(ArgumentError(
        "Unknown key(s) $(_fmt_keys(unknown)) in section `[hybrid_scan]`. " *
        "Allowed keys: $(join(allowed_fields, ", "))"))

    kw = Dict{Symbol,Any}()
    for f in allowed_fields
        sk = String(f)
        haskey(d, sk) || continue
        kw[f] = f === :fields ? String.(d[sk]) : _coerce_field(HybridScanParameters, f, d[sk])
    end

    hp = HybridScanParameters(; kw...)

    isempty(hp.select_field) && throw(ArgumentError(
        "`[hybrid_scan]` select_field must name the field whose validity drives " *
        "the tilt selection (e.g. \"DBZ\"); it cannot be empty."))
    hp.beam_height_maximum > 0.0 || throw(ArgumentError(
        "`[hybrid_scan]` beam_height_maximum = $(hp.beam_height_maximum) must be " *
        "positive (metres above the radar)."))
    hp.base_angle_tolerance >= 0.0 || throw(ArgumentError(
        "`[hybrid_scan]` base_angle_tolerance = $(hp.base_angle_tolerance) must be " *
        "non-negative (degrees)."))
    if !isempty(hp.angle_pattern)
        try
            Regex(hp.angle_pattern)
        catch e
            throw(ArgumentError(
                "`[hybrid_scan]` angle_pattern is not a valid regular expression: " *
                sprint(showerror, e)))
        end
    end
    if !isempty(hp.fields) && !(hp.select_field in hp.fields)
        throw(ArgumentError(
            "`[hybrid_scan]` select_field \"$(hp.select_field)\" must be one of " *
            "the carried `fields` ($(join(hp.fields, ", "))), otherwise the " *
            "selected cells would not appear in the output."))
    end
    return hp
end

# Build ReferenceStateParameters from the optional `[reference_state]` block.
# Every key defaults when absent; unknown keys raise to catch typos, matching
# the rest of the loader. `source` gates which keys are actually meaningful,
# but every key stays legal regardless of `source` (e.g. the exponential
# defaults may sit alongside a sounding source, unused but not rejected).
function _reference_state_from_dict(d::AbstractDict)
    allowed = (:source, :sounding_file, :moisture, :density_surface_kg_m3,
        :density_scale_height_m, :temperature_surface_k, :lapse_rate_k_m)
    unknown = setdiff(Set(Symbol.(keys(d))), Set(allowed))
    isempty(unknown) || throw(ArgumentError(
        "Unknown key(s) $(_fmt_keys(unknown)) in section `[reference_state]`. " *
        "Allowed keys: $(join(allowed, ", "))"))

    kw = Dict{Symbol,Any}()
    for f in allowed
        sk = String(f)
        haskey(d, sk) || continue
        kw[f] = _coerce_field(ReferenceStateParameters, f, d[sk])
    end
    rp = ReferenceStateParameters(; kw...)

    rp.source in ("exponential", "sounding") || throw(ArgumentError(
        "`[reference_state]` source = \"$(rp.source)\" is not valid; allowed: " *
        "\"exponential\", \"sounding\"."))
    if rp.source == "sounding"
        isempty(rp.sounding_file) && throw(ArgumentError(
            "`[reference_state]` source = \"sounding\" requires `sounding_file` " *
            "to be set."))
        isfile(rp.sounding_file) || throw(ArgumentError(
            "`[reference_state]` sounding_file \"$(rp.sounding_file)\" does not " *
            "exist."))
    end
    rp.density_surface_kg_m3 > 0.0 || throw(ArgumentError(
        "`[reference_state]` density_surface_kg_m3 must be positive (got " *
        "$(rp.density_surface_kg_m3))."))
    rp.density_scale_height_m > 0.0 || throw(ArgumentError(
        "`[reference_state]` density_scale_height_m must be positive (got " *
        "$(rp.density_scale_height_m))."))
    return rp
end

# Samurai configuration keys that have a Daisho counterpart under
# `[variational]` (or one of its sub-tables); a migration diagnostic names the
# Daisho key. Checked before strict parsing at every level the old flat key
# could have appeared at (top-level, background_error, radial_velocity).
const _SAMURAI_VARIATIONAL_KEYS = Dict(
    "mc_weight"      => "`mass_continuity_sigma_per_s` in `[variational]`",
    "bg_rhou_error"  => "a `U` entry in `[variational.background_error]`",
    "bg_rhov_error"  => "a `V` entry in `[variational.background_error]`",
    "bg_rhow_error"  => "a `W` entry in `[variational.background_error]`",
    "i_filter_length" => "`filter_length_m` (metres, per axis, i first) in " *
                         "`[variational.background_error.<name>]`",
    "j_filter_length" => "`filter_length_m` (metres, per axis, j second) in " *
                         "`[variational.background_error.<name>]`",
    "k_filter_length" => "`filter_length_m` (metres, per axis, k third) in " *
                         "`[variational.background_error.<name>]`",
    "horizontal_radar_appx" => "`horizontal_approximation` in " *
                               "`[variational.radial_velocity]`",
    "num_iterations" => "`max_outer_iterations` in `[variational]`",
)

function _check_samurai_variational_keys(d::AbstractDict, section::String)
    for k in keys(d)
        ks = String(k)
        if haskey(_SAMURAI_VARIATIONAL_KEYS, ks)
            throw(ArgumentError(
                "`[$section]` key `$ks` is a samurai configuration key. Its " *
                "Daisho equivalent is $(_SAMURAI_VARIATIONAL_KEYS[ks]). " *
                "Run `print_config(\"template.toml\")` for the documented template."))
        end
    end
    return nothing
end

# Build BackgroundErrorEntry per key of `[variational.background_error]`.
# Keys must be "default" or a name in `state`; each value is an inline table
# `{ sigma = ..., filter_length_m = [i, j, k] }`.
function _variational_background_error_from_dict(d::AbstractDict, state::Vector{String})
    _check_samurai_variational_keys(d, "variational.background_error")
    allowed_names = Set(vcat(["default"], state))
    out = Dict{String,BackgroundErrorEntry}()
    for (k, v) in d
        ks = String(k)
        ks in allowed_names || throw(ArgumentError(
            "Unknown key `$(ks)` in section `[variational.background_error]`; " *
            "it must be \"default\" or a name in `[variational] state`. " *
            "Allowed: $(join(sort(collect(allowed_names)), ", "))"))
        v isa AbstractDict || throw(ArgumentError(
            "`[variational.background_error]` entry `$(ks)` must be an inline " *
            "table with `sigma` and `filter_length_m` keys, got $(typeof(v))."))
        entry_section = "variational.background_error.$(ks)"
        _check_samurai_variational_keys(v, entry_section)
        unknown = setdiff(Set(keys(v)), Set(["sigma", "filter_length_m"]))
        isempty(unknown) || throw(ArgumentError(
            "Unknown key(s) $(_fmt_keys(Symbol.(collect(unknown)))) in section " *
            "`[$(entry_section)]`. Allowed keys: sigma, filter_length_m"))
        haskey(v, "sigma") || throw(ArgumentError(
            "Missing required key `sigma` in section `[$(entry_section)]`."))
        haskey(v, "filter_length_m") || throw(ArgumentError(
            "Missing required key `filter_length_m` in section `[$(entry_section)]`."))
        sigma = Float64(v["sigma"])
        sigma > 0.0 || throw(ArgumentError(
            "`[$(entry_section)]` sigma must be positive (got $(sigma))."))
        fl = v["filter_length_m"]
        (fl isa AbstractArray && length(fl) == 3) || throw(ArgumentError(
            "`[$(entry_section)]` filter_length_m must be an array of exactly " *
            "3 numbers [i, j, k] in metres, got $(repr(fl))."))
        fld = Float64.(fl)
        all(x -> isfinite(x) && x >= 0.0, fld) || throw(ArgumentError(
            "`[$(entry_section)]` filter_length_m entries must be finite and " *
            "non-negative (metres; 0 disables filtering on that axis), got " *
            "$(fld)."))
        out[ks] = BackgroundErrorEntry(sigma, (fld[1], fld[2], fld[3]))
    end
    return out
end

# Build RadialVelocityObsParameters from the optional
# `[variational.radial_velocity]` sub-table; every key defaults when absent.
function _variational_radial_velocity_from_dict(d::AbstractDict)
    _check_samurai_variational_keys(d, "variational.radial_velocity")
    allowed = (:error_inflation, :horizontal_approximation, :eigenvalue_guard_relative)
    unknown = setdiff(Set(Symbol.(keys(d))), Set(allowed))
    isempty(unknown) || throw(ArgumentError(
        "Unknown key(s) $(_fmt_keys(unknown)) in section " *
        "`[variational.radial_velocity]`. Allowed keys: $(join(allowed, ", "))"))
    kw = Dict{Symbol,Any}()
    for f in allowed
        sk = String(f)
        haskey(d, sk) || continue
        kw[f] = _coerce_field(RadialVelocityObsParameters, f, d[sk])
    end
    rv = RadialVelocityObsParameters(; kw...)
    rv.error_inflation > 0.0 || throw(ArgumentError(
        "`[variational.radial_velocity]` error_inflation must be positive " *
        "(got $(rv.error_inflation))."))
    (isfinite(rv.eigenvalue_guard_relative) && 0.0 <= rv.eigenvalue_guard_relative < 1.0) ||
        throw(ArgumentError(
            "`[variational.radial_velocity]` eigenvalue_guard_relative must " *
            "lie in [0, 1) (got $(rv.eigenvalue_guard_relative))."))
    return rv
end

# Build ScalarObsEntry per key of `[variational.scalar_obs]`. Keys must be
# `[fields]` names that appear in `state` (the non-wind entries); every such
# name must have an entry (checked by the caller).
function _variational_scalar_obs_from_dict(d::AbstractDict, scalar_state::Vector{String})
    _check_samurai_variational_keys(d, "variational.scalar_obs")
    allowed_names = Set(scalar_state)
    out = Dict{String,ScalarObsEntry}()
    for (k, v) in d
        ks = String(k)
        ks in allowed_names || throw(ArgumentError(
            "Unknown key `$(ks)` in section `[variational.scalar_obs]`; it " *
            "must be a `[fields]` name in `[variational] state`. " *
            "Allowed: $(join(sort(collect(allowed_names)), ", "))"))
        v isa AbstractDict || throw(ArgumentError(
            "`[variational.scalar_obs]` entry `$(ks)` must be an inline table " *
            "with `sigma` and `undetect_value` keys, got $(typeof(v))."))
        entry_section = "variational.scalar_obs.$(ks)"
        unknown = setdiff(Set(keys(v)), Set(["sigma", "undetect_value"]))
        isempty(unknown) || throw(ArgumentError(
            "Unknown key(s) $(_fmt_keys(Symbol.(collect(unknown)))) in section " *
            "`[$(entry_section)]`. Allowed keys: sigma, undetect_value"))
        haskey(v, "sigma") || throw(ArgumentError(
            "Missing required key `sigma` in section `[$(entry_section)]`."))
        haskey(v, "undetect_value") || throw(ArgumentError(
            "Missing required key `undetect_value` in section `[$(entry_section)]`."))
        sigma = Float64(v["sigma"])
        sigma > 0.0 || throw(ArgumentError(
            "`[$(entry_section)]` sigma must be positive (got $(sigma))."))
        out[ks] = ScalarObsEntry(sigma, Float64(v["undetect_value"]))
    end
    return out
end

# Build the `[variational.roles]` override Dict{String,Symbol}. Keys must be
# in `state`; values must be strings drawn from VARIATIONAL_ROLE_VOCAB.
function _variational_roles_from_dict(d::AbstractDict, state::Vector{String})
    _check_samurai_variational_keys(d, "variational.roles")
    allowed_names = Set(state)
    out = Dict{String,Symbol}()
    for (k, v) in d
        ks = String(k)
        ks in allowed_names || throw(ArgumentError(
            "Unknown key `$(ks)` in section `[variational.roles]`; it must be " *
            "one of `[variational] state` ($(join(state, ", ")))."))
        v isa AbstractString || throw(ArgumentError(
            "`[variational.roles]` `$(ks)` must be a string role name, got " *
            "$(typeof(v))."))
        role_str = String(v)
        role_str in VARIATIONAL_ROLE_VOCAB || throw(ArgumentError(
            "`[variational.roles]` `$(ks)` = \"$(role_str)\" is not a valid " *
            "role. Allowed: $(join(VARIATIONAL_ROLE_VOCAB, ", "))."))
        out[ks] = Symbol(role_str)
    end
    return out
end

# Build VariationalParameters from the optional `[variational]` block. Strict:
# unknown keys raise in every (sub-)table; every cross-field constraint in
# plan §4/§0 is enforced here (`state` composition, `background_error`
# resolution, `scalar_obs` coverage, `roles` domain, tolerances/iterations).
# `moments` is needed to validate `[fields]`-derived `state` entries.
function _variational_from_dict(d::AbstractDict, moments::MomentParameters)
    _check_samurai_variational_keys(d, "variational")

    scalar_fields = (:enabled, :state, :background, :background_error_inflation,
        :mass_continuity, :mass_continuity_lattice, :mass_continuity_sigma_per_s,
        :max_outer_iterations, :max_inner_iterations, :inner_tolerance, :outer_tolerance,
        :cg_progress_interval, :output_background, :output_increment, :output_mish,
        :quality_flag_map, :mask_quality_flags, :filter_type)
    subtables = (:background_error, :radial_velocity, :scalar_obs, :roles)
    allowed = Set((scalar_fields..., subtables...))
    unknown = setdiff(Set(Symbol.(keys(d))), allowed)
    isempty(unknown) || throw(ArgumentError(
        "Unknown key(s) $(_fmt_keys(unknown)) in section `[variational]`. " *
        "Allowed keys: $(join((scalar_fields..., subtables...), ", "))"))
    for t in subtables
        sk = String(t)
        haskey(d, sk) && !(d[sk] isa AbstractDict) && throw(ArgumentError(
            "`[variational]` `$(sk)` must be a sub-table `[variational.$(sk)]`, " *
            "got $(typeof(d[sk]))."))
    end

    kw = Dict{Symbol,Any}()
    for f in scalar_fields
        sk = String(f)
        haskey(d, sk) || continue
        if f === :state
            kw[f] = String.(d[sk])
        elseif f === :mask_quality_flags
            d[sk] isa AbstractVector || throw(ArgumentError(
                "`[variational]` mask_quality_flags must be an array of flag meanings, " *
                "got $(repr(d[sk]))."))
            kw[f] = String.(d[sk])
        else
            kw[f] = _coerce_field(VariationalParameters, f, d[sk])
        end
    end

    state = get(kw, :state, VariationalParameters().state)
    isempty(state) && throw(ArgumentError(
        "`[variational]` `state` must be non-empty."))
    length(state) == length(Set(state)) || throw(ArgumentError(
        "`[variational]` `state` entries must be unique (got $(state))."))
    "U" in state || throw(ArgumentError(
        "`[variational]` `state` must contain \"U\"."))
    "V" in state || throw(ArgumentError(
        "`[variational]` `state` must contain \"V\"."))
    field_names = Set(fs.name for fs in moments.fields)
    for name in state
        name in ("U", "V", "W") && continue
        name in field_names && continue
        throw(ArgumentError(
            "`[variational]` state entry `$(name)` is not \"U\", \"V\", \"W\", " *
            "or a configured `[fields]` name. Fields: " *
            "$(join(sort(collect(field_names)), ", "))"))
    end

    mc = get(kw, :mass_continuity, VariationalParameters().mass_continuity)
    mc isa Bool || throw(ArgumentError(
        "`[variational]` mass_continuity must be a boolean (true/false), got " *
        "$(repr(mc))."))
    mc && !("W" in state) && throw(ArgumentError(
        "`[variational]` mass_continuity = true requires \"W\" in `state`."))
    mcl = get(kw, :mass_continuity_lattice, VariationalParameters().mass_continuity_lattice)
    mcl in VARIATIONAL_CONTINUITY_LATTICE_VOCAB || throw(ArgumentError(
        "`[variational]` mass_continuity_lattice = \"$(mcl)\" is not valid; allowed: " *
        join(("\"$(m)\"" for m in VARIATIONAL_CONTINUITY_LATTICE_VOCAB), ", ") * "."))

    qmap = get(kw, :quality_flag_map, VariationalParameters().quality_flag_map)
    qmap in VARIATIONAL_QUALITY_FLAG_MAP_VOCAB || throw(ArgumentError(
        "`[variational]` quality_flag_map = \"$(qmap)\" is not valid; allowed: " *
        join(("\"$(m)\"" for m in VARIATIONAL_QUALITY_FLAG_MAP_VOCAB), ", ") * "."))
    mask_flags = get(kw, :mask_quality_flags, VariationalParameters().mask_quality_flags)
    meanings = first.(QFLAG_MEANINGS)
    for m in mask_flags
        m in meanings || throw(ArgumentError(
            "`[variational]` mask_quality_flags entry \"$(m)\" is not a QFLAG meaning; " *
            "allowed: " * join(("\"$(x)\"" for x in meanings), ", ") * "."))
    end
    length(mask_flags) == length(Set(mask_flags)) || throw(ArgumentError(
        "`[variational]` mask_quality_flags entries must be unique (got $(mask_flags))."))

    background = get(kw, :background, VariationalParameters().background)
    background in VARIATIONAL_BACKGROUND_VOCAB || throw(ArgumentError(
        "`[variational]` background = \"$(background)\" is not valid; allowed: " *
        join(("\"$(m)\"" for m in VARIATIONAL_BACKGROUND_VOCAB), ", ") * "."))

    filter_type = get(kw, :filter_type, VariationalParameters().filter_type)
    filter_type in ("gaussian", "lanczos") || throw(ArgumentError(
        "`[variational]` filter_type = \"$(filter_type)\" is not valid; " *
        "allowed: \"gaussian\", \"lanczos\"."))

    bei = get(kw, :background_error_inflation, VariationalParameters().background_error_inflation)
    (isfinite(bei) && bei > 0.0) || throw(ArgumentError(
        "`[variational]` background_error_inflation must be positive (got " *
        "$(bei))."))

    mcs = get(kw, :mass_continuity_sigma_per_s, VariationalParameters().mass_continuity_sigma_per_s)
    (isfinite(mcs) && mcs > 0.0) || throw(ArgumentError(
        "`[variational]` mass_continuity_sigma_per_s must be positive (got " *
        "$(mcs))."))

    max_outer = get(kw, :max_outer_iterations, VariationalParameters().max_outer_iterations)
    max_outer >= 1 || throw(ArgumentError(
        "`[variational]` max_outer_iterations must be ≥ 1 (got $(max_outer))."))
    max_inner = get(kw, :max_inner_iterations, VariationalParameters().max_inner_iterations)
    max_inner >= 1 || throw(ArgumentError(
        "`[variational]` max_inner_iterations must be ≥ 1 (got $(max_inner))."))

    inner_tol = get(kw, :inner_tolerance, VariationalParameters().inner_tolerance)
    (isfinite(inner_tol) && 0.0 < inner_tol < 1.0) || throw(ArgumentError(
        "`[variational]` inner_tolerance must lie in (0, 1) (got $(inner_tol))."))
    outer_tol = get(kw, :outer_tolerance, VariationalParameters().outer_tolerance)
    (isfinite(outer_tol) && 0.0 < outer_tol < 1.0) || throw(ArgumentError(
        "`[variational]` outer_tolerance must lie in (0, 1) (got $(outer_tol))."))
    cgpi = get(kw, :cg_progress_interval, VariationalParameters().cg_progress_interval)
    cgpi >= 0 || throw(ArgumentError(
        "`[variational]` cg_progress_interval must be ≥ 0 (0 = silent; got $(cgpi))."))

    if haskey(d, "background_error")
        kw[:background_error] = _variational_background_error_from_dict(d["background_error"], state)
    end
    background_error = get(kw, :background_error, VariationalParameters().background_error)
    for name in state
        haskey(background_error, name) || haskey(background_error, "default") ||
            throw(ArgumentError(
                "`[variational]` state variable `$(name)` has no " *
                "`[variational.background_error]` entry and there is no " *
                "`default` entry to fall back to."))
    end

    if haskey(d, "radial_velocity")
        kw[:radial_velocity] = _variational_radial_velocity_from_dict(d["radial_velocity"])
    end

    scalar_state = [n for n in state if !(n in ("U", "V", "W"))]
    if haskey(d, "scalar_obs")
        kw[:scalar_obs] = _variational_scalar_obs_from_dict(d["scalar_obs"], scalar_state)
    end
    scalar_obs = get(kw, :scalar_obs, VariationalParameters().scalar_obs)
    for name in scalar_state
        haskey(scalar_obs, name) || throw(ArgumentError(
            "`[variational]` state entry `$(name)` (a `[fields]` name) needs a " *
            "`[variational.scalar_obs.$(name)]` entry."))
    end

    if haskey(d, "roles")
        kw[:roles] = _variational_roles_from_dict(d["roles"], state)
    end

    return VariationalParameters(; kw...)
end

# Each `[grid.*]` sub-table is independently optional and defaults when absent;
# any sub-table that is present is still validated strictly.
function _grid_from_dict(d::AbstractDict)
    if haskey(d, "spectral")
        throw(ArgumentError(
            "`[grid.spectral]` was replaced by `[grid.springsteel]`, which " *
            "speaks Springsteel's native i/j/k axis vocabulary (general " *
            "across geometries) and counts B-spline CELLS unambiguously " *
            "(the old `xdim` counted cells but read like a gridpoint dim). " *
            "Map xmin/xmax/xdim → `[grid.springsteel.i]` min/max/cells " *
            "(likewise y → j, z → k) and `[grid.spectral.bc.xL/xR]` → " *
            "`[grid.springsteel.i.bc]` min/max. " *
            "Run `print_config(\"template.toml\")` for the new template."))
    end
    # `range_minimum`/`range_maximum` are optional per-product overrides of the
    # `[gridding]` global; absent ⇒ `nothing` ⇒ inherit (see `_range_bounds`).
    range_opt = Set([:range_minimum, :range_maximum])
    cartesian   = haskey(d, "cartesian")   ? _struct_from_dict(CartesianGridParameters, d["cartesian"]; section="grid.cartesian", optional=range_opt) : CartesianGridParameters()
    # The Cartesian-family products fall back to `cartesian` when their own table
    # is absent (resolved by the GridParameters constructor from these `nothing`s),
    # so `[grid.cartesian]` is the shared base geometry and the per-product tables
    # are overrides for radars that need distinct grids.
    volume      = haskey(d, "volume")      ? _struct_from_dict(CartesianGridParameters, d["volume"];    section="grid.volume",    optional=range_opt) : nothing
    composite   = haskey(d, "composite")   ? _struct_from_dict(CartesianGridParameters, d["composite"]; section="grid.composite", optional=range_opt) : nothing
    ppi         = haskey(d, "ppi")         ? _struct_from_dict(CartesianGridParameters, d["ppi"];        section="grid.ppi",       optional=range_opt) : nothing
    column      = haskey(d, "column")      ? _struct_from_dict(CartesianGridParameters, d["column"];     section="grid.column",    optional=range_opt) : nothing
    latlon      = haskey(d, "latlon")      ? _struct_from_dict(LatLonGridParameters,    d["latlon"];    section="grid.latlon",    optional=range_opt) : LatLonGridParameters()
    rhi         = haskey(d, "rhi")         ? _struct_from_dict(RhiGridParameters,       d["rhi"];       section="grid.rhi",       optional=range_opt) : RhiGridParameters()
    springsteel = haskey(d, "springsteel") ? _springsteel_from_dict(d["springsteel"])                                            : SpringsteelGridConfig()
    metadata    = haskey(d, "metadata")    ? _struct_from_dict(MetadataParameters,      d["metadata"];  section="grid.metadata")  : MetadataParameters()
    origin      = haskey(d, "origin")      ? _origin_from_dict(d["origin"])                                                  : OriginParameters()
    return GridParameters(cartesian=cartesian, volume=volume, composite=composite,
                          ppi=ppi, column=column, latlon=latlon, rhi=rhi,
                          springsteel=springsteel, metadata=metadata, origin=origin)
end

# Build a SpringsteelGridConfig from the `[grid.springsteel]` table. The
# geometry decides which axis tables must/may exist and which keys each may
# carry (see SPRINGSTEEL_AXIS_BASES); everything present is validated strictly.
function _springsteel_from_dict(d::AbstractDict)
    allowed = Set(["geometry", "mubar", "quadrature", "i", "j", "k"])
    unknown = setdiff(Set(keys(d)), allowed)
    isempty(unknown) || throw(ArgumentError(
        "Unknown key(s) $(_fmt_keys(unknown)) in section `[grid.springsteel]`. " *
        "Allowed keys: geometry, mubar, quadrature, i, j, k"))

    haskey(d, "geometry") || throw(ArgumentError(
        "Missing required key `geometry` in section `[grid.springsteel]`. " *
        "Run `print_config(\"template.toml\")` for a complete template."))
    geometry = String(d["geometry"])
    haskey(SPRINGSTEEL_AXIS_BASES, geometry) || throw(ArgumentError(
        "[grid.springsteel]: unsupported geometry `$(geometry)`. Supported: " *
        "$(join(sort(collect(keys(SPRINGSTEEL_AXIS_BASES))), ", ")). " *
        "Fourier/Chebyshev-primary geometries (L, LL, LLZ, Z, ZZ, ZZZ) need " *
        "explicit spectral dimensions; build those in Julia via " *
        "`Springsteel.SpringsteelGridParameters`/`createGrid` (with " *
        "`vars = radar_vars(p)`) and pass the grid to the gridding entry " *
        "points directly."))

    mubar = haskey(d, "mubar") ? Int(d["mubar"]) : 3
    quadrature = haskey(d, "quadrature") ? Symbol(String(d["quadrature"])) : :gauss
    quadrature in (:gauss, :regular) || throw(ArgumentError(
        "[grid.springsteel]: unknown quadrature `$(quadrature)`. " *
        "Allowed: \"gauss\", \"regular\"."))

    bases = SPRINGSTEEL_AXIS_BASES[geometry]
    axis(name, basis) = _springsteel_axis_from_dict(
        haskey(d, name) ? d[name] : nothing, name, basis)
    return SpringsteelGridConfig(
        geometry = geometry, mubar = mubar, quadrature = quadrature,
        i = axis("i", bases[1]), j = axis("j", bases[2]), k = axis("k", bases[3]))
end

# Parse one `[grid.springsteel.<axis>]` table per the basis its geometry puts
# on that axis. `d === nothing` means the table was absent.
function _springsteel_axis_from_dict(d, name::String, basis::Symbol)
    sect = "grid.springsteel.$(name)"

    if basis === :none
        d === nothing || throw(ArgumentError(
            "[$(sect)]: this geometry has no $(name) axis; remove the table."))
        return SpringsteelAxisConfig()
    end

    if basis === :fourier
        # Auto-sized azimuthal Fourier axis: Springsteel derives the dimensions
        # from the ring formula and the BCs are forced periodic; only the
        # wavenumber cap and the output resampling count are configurable.
        d === nothing && return SpringsteelAxisConfig()
        allowed = Set(["max_wavenumber", "regular_out"])
        unknown = setdiff(Set(keys(d)), allowed)
        isempty(unknown) || throw(ArgumentError(
            "Unknown key(s) $(_fmt_keys(unknown)) in section `[$(sect)]`. " *
            "This geometry's $(name) axis is an auto-sized Fourier axis " *
            "(periodic, ring-formula dimensions); allowed keys: " *
            "max_wavenumber, regular_out"))
        return SpringsteelAxisConfig(
            max_wavenumber = haskey(d, "max_wavenumber") ? Int(d["max_wavenumber"]) : -1,
            regular_out    = haskey(d, "regular_out")    ? Int(d["regular_out"])    : 0)
    end

    # Spline / Chebyshev axes need an explicit table.
    d === nothing && throw(ArgumentError(
        "Missing required TOML section `[$(sect)]` (this geometry's $(name) " *
        "axis is $(basis === :spline ? "a B-spline" : "a Chebyshev") axis). " *
        "Run `print_config(\"template.toml\")` for a complete template."))
    sizekey = basis === :spline ? "cells" : "points"
    required = ("min", "max", sizekey)
    allowed = Set([required..., "regular_out", "bc"])
    unknown = setdiff(Set(keys(d)), allowed)
    isempty(unknown) || throw(ArgumentError(
        "Unknown key(s) $(_fmt_keys(unknown)) in section `[$(sect)]`. " *
        "Allowed keys: $(join(sort(collect(allowed)), ", ")) " *
        (basis === :spline ?
            "(`cells` counts B-spline cells; physical points = cells × mubar)" :
            "(`points` counts Chebyshev gridpoints)")))
    missing_keys = setdiff(Set(required), Set(keys(d)))
    isempty(missing_keys) || throw(ArgumentError(
        "Missing required key(s) $(_fmt_keys(Symbol.(collect(missing_keys)))) " *
        "in section `[$(sect)]`. " *
        "Run `print_config(\"template.toml\")` for a complete template."))

    bc_min, bc_max = haskey(d, "bc") ?
        _springsteel_bc_from_dict(d["bc"], sect) :
        (Dict{String,BoundaryConditions}("default" => NaturalBC()),
         Dict{String,BoundaryConditions}("default" => NaturalBC()))
    return SpringsteelAxisConfig(
        min = Float64(d["min"]), max = Float64(d["max"]),
        cells  = basis === :spline    ? Int(d["cells"])  : 0,
        points = basis === :chebyshev ? Int(d["points"]) : 0,
        regular_out = haskey(d, "regular_out") ? Int(d["regular_out"]) : 0,
        bc_min = bc_min, bc_max = bc_max)
end

# Parse a `[grid.springsteel.<axis>.bc]` table into per-variable (bc_min,
# bc_max) dicts. The `min`/`max` keys set the side's "default" BC; any other
# key names a field and maps to a table with its own `min`/`max` overrides.
# Field names are validated against `[fields]` at `create_radar_grid` time
# (this parser runs before the field list is fixed).
function _springsteel_bc_from_dict(d::AbstractDict, sect::String)
    bc_min = Dict{String,BoundaryConditions}("default" => NaturalBC())
    bc_max = Dict{String,BoundaryConditions}("default" => NaturalBC())
    for (k, v) in d
        if k == "min" || k == "max"
            (k == "min" ? bc_min : bc_max)["default"] =
                _parse_bc_value(v; at = "[$(sect).bc] `$(k)`")
        else
            v isa AbstractDict || throw(ArgumentError(
                "[$(sect).bc]: per-field override `$(k)` must be a table " *
                "with `min` and/or `max` keys, e.g. " *
                "$(k) = { min = \"natural\" }; got $(typeof(v))."))
            unknown = setdiff(Set(keys(v)), Set(["min", "max"]))
            isempty(unknown) || throw(ArgumentError(
                "Unknown key(s) $(_fmt_keys(unknown)) in `[$(sect).bc]` " *
                "override for field `$(k)`. Allowed keys: min, max"))
            haskey(v, "min") && (bc_min[String(k)] =
                _parse_bc_value(v["min"]; at = "[$(sect).bc] `$(k).min`"))
            haskey(v, "max") && (bc_max[String(k)] =
                _parse_bc_value(v["max"]; at = "[$(sect).bc] `$(k).max`"))
        end
    end
    return bc_min, bc_max
end

# A BC value is either a plain string (a parameterless BC type, e.g.
# "natural") or a table with a `type` key plus that type's parameters.
_parse_bc_value(v::AbstractString; at::String) =
    _parse_bc(Dict{String,Any}("type" => String(v)); at = at)
_parse_bc_value(v::AbstractDict; at::String) = _parse_bc(v; at = at)
_parse_bc_value(v; at::String) = throw(ArgumentError(
    "$(at): a boundary condition must be a string (e.g. \"natural\") or a " *
    "table (e.g. { type = \"dirichlet\", value = 0.0 }); got $(typeof(v))."))

# Map a BC sub-table to a Springsteel BoundaryConditions.
function _parse_bc(d::AbstractDict; at::String="BC sub-table")
    where_ = at
    haskey(d, "type") || throw(ArgumentError(
        "$(where_) missing required `type` key"))
    t = lowercase(String(d["type"]))
    if t == "natural"
        return NaturalBC()
    elseif t == "dirichlet"
        return DirichletBC(Float64(get(d, "value", 0.0)))
    elseif t == "neumann"
        return NeumannBC(Float64(get(d, "value", 0.0)))
    elseif t == "second_derivative"
        return SecondDerivativeBC(Float64(get(d, "value", 0.0)))
    elseif t == "robin"
        haskey(d, "alpha") && haskey(d, "beta") ||
            throw(ArgumentError("BC type `robin` requires `alpha` and `beta` keys"))
        return RobinBC(Float64(d["alpha"]), Float64(d["beta"]),
                       Float64(get(d, "gamma", 0.0)))
    elseif t == "periodic"
        return PeriodicBC()
    elseif t == "exponential"
        haskey(d, "lambda") ||
            throw(ArgumentError("BC type `exponential` requires `lambda` key"))
        return ExponentialBC(Float64(d["lambda"]))
    elseif t == "cauchy"
        haskey(d, "u") && haskey(d, "du") ||
            throw(ArgumentError("BC type `cauchy` requires `u` and `du` keys"))
        return CauchyBC(Float64(d["u"]), Float64(d["du"]))
    elseif t == "antisymmetric"
        return AntisymmetricBC()
    elseif t == "symmetric"
        return SymmetricBC()
    elseif t == "zeros"
        return ZerosBC()
    else
        throw(ArgumentError("Unknown BC type `$(t)`. Recognized: " *
            "natural, dirichlet, neumann, second_derivative, robin, periodic, " *
            "exponential, cauchy, antisymmetric, symmetric, zeros"))
    end
end
