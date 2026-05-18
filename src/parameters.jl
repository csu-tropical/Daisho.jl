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
    :velocity,                                          # reserved
)
const FIELD_INTERP_TAGS   = (:linear_interp, :weighted_interp, :nearest_interp)
const FIELD_SINGULAR_TAGS = (:define_detection, :define_scanned, :velocity)

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
interpolation tag is present).
"""
function interp_of(fs::FieldSpec)
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
- `beam_inflation::Float64`: factor for inflating ROI with distance from radar
  (0.0 disables).
- `power_threshold::Float64`: minimum beam power weight for a gate to contribute.

The two gate-role moments are no longer configured here: the field whose
presence proves a gate was scanned (formerly `missing_key`) and the field
whose presence proves a detectable echo (formerly `valid_key`) are now
declared as the `define_scanned` / `define_detection` tags in `[fields]` and
resolved via [`field_with_tag`](@ref).
"""
Base.@kwdef struct GriddingParameters
    beam_inflation::Float64  = 0.01
    power_threshold::Float64 = 0.5
end

"""
    CartesianGridParameters

Regular Cartesian grid specification (3D Easting × Northing × Altitude, meters).
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
end

"""
    SpectralBCParameters

Per-axis-side boundary conditions for a Springsteel spectral grid. Each field
holds a Springsteel `BoundaryConditions`. Defaults are `NaturalBC()` (no
constraint).
"""
Base.@kwdef struct SpectralBCParameters
    xL::BoundaryConditions = NaturalBC()
    xR::BoundaryConditions = NaturalBC()
    yL::BoundaryConditions = NaturalBC()
    yR::BoundaryConditions = NaturalBC()
    zL::BoundaryConditions = NaturalBC()
    zR::BoundaryConditions = NaturalBC()
end

"""
    SpectralGridParameters

Spectral (B-spline) grid specification for Springsteel. `xdim`/`ydim`/`zdim`
count B-spline cells; physical points per dimension = `cells * mubar`.

# Fields
- `geometry::String`: `"R"`, `"RR"`, or `"RRR"`.
- `mubar::Int`: quadrature points per cell.
- `quadrature::Symbol`: `:gauss` or `:regular`.
- `xmin`, `xmax`, `xdim`: i-axis (X) bounds and cell count.
- `ymin`, `ymax`, `ydim`: j-axis (Y) bounds and cell count (for RR/RRR).
- `zmin`, `zmax`, `zdim`: k-axis (Z) bounds and cell count (for RRR).
- `bc::SpectralBCParameters`: per-axis-side boundary conditions.
"""
Base.@kwdef struct SpectralGridParameters
    geometry::String   = "RRR"
    mubar::Int         = 3
    quadrature::Symbol = :gauss
    xmin::Float64      = -125000.0
    xmax::Float64      =  125000.0
    xdim::Int          = 50
    ymin::Float64      = -125000.0
    ymax::Float64      =  125000.0
    ydim::Int          = 50
    zmin::Float64      = 0.0
    zmax::Float64      = 15000.0
    zdim::Int          = 10
    bc::SpectralBCParameters = SpectralBCParameters()
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
    GridParameters

Aggregates the spatial-grid sub-specs plus the NetCDF metadata block. Pull
out the one matching the gridding driver you are calling (e.g.,
`p.grid.cartesian` for `grid_radar_volume`).
"""
Base.@kwdef struct GridParameters
    cartesian::CartesianGridParameters = CartesianGridParameters()
    latlon::LatLonGridParameters       = LatLonGridParameters()
    rhi::RhiGridParameters             = RhiGridParameters()
    spectral::SpectralGridParameters   = SpectralGridParameters()
    metadata::MetadataParameters       = MetadataParameters()
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
    DaishoParameters

Top-level immutable runtime configuration loaded from a TOML file. Construct
via `DaishoParameters()` for the bundled template or `DaishoParameters(path)`
for a user file (strict — every key in the template must be present).

# Fields
- `moments::MomentParameters`
- `qc::QCParameters`
- `gridding::GriddingParameters`
- `grid::GridParameters`
- `io::IOParameters`

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
end

# ── Loader ───────────────────────────────────────────────────────────────────

"""
    DaishoParameters() -> DaishoParameters
    DaishoParameters(path::AbstractString) -> DaishoParameters

Construct a `DaishoParameters` instance. With no arguments, loads the bundled
template from `config/defaults.toml`. With a path, loads only the user's TOML
— there is no silent fallback to bundled defaults. Every section and key
documented in the template must be present, otherwise an `ArgumentError` is
raised naming the missing item.

Use [`print_config`](@ref) to write a complete starter template to a file you
can edit.
"""
DaishoParameters() = DaishoParameters(_load_toml(DEFAULTS_TOML_PATH))
DaishoParameters(path::AbstractString) = DaishoParameters(_load_toml(path))

# Internal: build from an already-parsed Dict (used by ctors and tests).
function DaishoParameters(d::AbstractDict)
    moments  = _fields_from_dict(_section(d, "fields"))
    qc       = _struct_from_dict(QCParameters,       _section(d, "qc");                  section="qc")
    gridding = _struct_from_dict(GriddingParameters, _section(d, "gridding");            section="gridding")
    grid     = _grid_from_dict(_section(d, "grid"))
    io       = _struct_from_dict(IOParameters,       _section(d, "io");                  section="io")
    return DaishoParameters(moments, qc, gridding, grid, io)
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
function _struct_from_dict(::Type{T}, d::AbstractDict; section::String="") where {T}
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

    missing_keys = setdiff(field_set, keysyms)
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
    return _struct_from_dict(GriddingParameters, d; section="gridding")
end

function _grid_from_dict(d::AbstractDict)
    cartesian = _struct_from_dict(CartesianGridParameters, _section(d, "cartesian"; parent="grid"); section="grid.cartesian")
    latlon    = _struct_from_dict(LatLonGridParameters,    _section(d, "latlon";    parent="grid"); section="grid.latlon")
    rhi       = _struct_from_dict(RhiGridParameters,       _section(d, "rhi";       parent="grid"); section="grid.rhi")
    spectral  = _spectral_from_dict(_section(d, "spectral"; parent="grid"))
    metadata  = _struct_from_dict(MetadataParameters,      _section(d, "metadata";  parent="grid"); section="grid.metadata")
    return GridParameters(cartesian=cartesian, latlon=latlon, rhi=rhi,
                          spectral=spectral, metadata=metadata)
end

function _spectral_from_dict(d::AbstractDict)
    bc_dict = _section(d, "bc"; parent="grid.spectral")
    bc = SpectralBCParameters(
        xL = _parse_bc(_section(bc_dict, "xL"; parent="grid.spectral.bc"); side="xL"),
        xR = _parse_bc(_section(bc_dict, "xR"; parent="grid.spectral.bc"); side="xR"),
        yL = _parse_bc(_section(bc_dict, "yL"; parent="grid.spectral.bc"); side="yL"),
        yR = _parse_bc(_section(bc_dict, "yR"; parent="grid.spectral.bc"); side="yR"),
        zL = _parse_bc(_section(bc_dict, "zL"; parent="grid.spectral.bc"); side="zL"),
        zR = _parse_bc(_section(bc_dict, "zR"; parent="grid.spectral.bc"); side="zR"),
    )
    # The scalar struct loader requires every field to be present; `bc`
    # comes from the per-side parsing above, not from `d` itself, so stuff a
    # placeholder in to satisfy the missing-key check and ignore it below.
    rest = Dict{String,Any}(k => v for (k, v) in d if k != "bc")
    rest["bc"] = SpectralBCParameters()
    base = _struct_from_dict(SpectralGridParameters, rest; section="grid.spectral")
    # Rebuild with parsed bc.
    return SpectralGridParameters(
        geometry   = base.geometry,
        mubar      = base.mubar,
        quadrature = base.quadrature,
        xmin = base.xmin, xmax = base.xmax, xdim = base.xdim,
        ymin = base.ymin, ymax = base.ymax, ydim = base.ydim,
        zmin = base.zmin, zmax = base.zmax, zdim = base.zdim,
        bc = bc,
    )
end

# Map a BC sub-table to a Springsteel BoundaryConditions.
function _parse_bc(d::AbstractDict; side::String="")
    where_ = isempty(side) ? "BC sub-table" : "[grid.spectral.bc.$(side)]"
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
