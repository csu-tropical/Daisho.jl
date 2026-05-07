# Daisho.jl runtime parameter system.
#
# Loads layered TOML configuration into immutable typed structs. Each section
# of the TOML maps 1:1 to a section struct; the top-level `DaishoParameters`
# bundles them. Bundled defaults live at `config/defaults.toml`; user TOML
# files are deep-merged over those defaults.

using TOML

const DEFAULTS_TOML_PATH = joinpath(@__DIR__, "..", "config", "defaults.toml")

# ── Section structs ──────────────────────────────────────────────────────────

"""
    MomentParameters

Canonical field names of interest plus a per-name gridding-interpolation hint.
The CfRadial 2.1 refactor dropped the historical raw/qc distinction: a field
is just a field, and its CfRadial relationships (e.g. `is_quality_field`,
`qualified_variables`) are recorded in `Field.metadata`.

# Fields
- `fields::Vector{String}`: canonical field names of interest, in load order.
- `grid_type::Dict{String,Symbol}`: per-name interpolation hint (`:linear`,
  `:weighted`, `:nearest`).
"""
struct MomentParameters
    fields::Vector{String}
    grid_type::Dict{String,Symbol}
end

"""
    field_index_dict(p::DaishoParameters) -> Dict{String,Int}

Build a name→column-index dict from `p.moments.fields`. Used by drivers that
need a moment_dict for the legacy gridding workers.
"""
field_index_dict(p) = Dict{String,Int}(name => i for (i, name) in enumerate(p.moments.fields))

"""
    grid_type_index_dict(p::DaishoParameters) -> Dict{Int,Symbol}

Build a column-index→grid-type dict from `p.moments.fields` and
`p.moments.grid_type`. Used by drivers that need a grid_type_dict for the
legacy gridding workers.
"""
grid_type_index_dict(p) = Dict{Int,Symbol}(
    i => p.moments.grid_type[p.moments.fields[i]] for i in eachindex(p.moments.fields))

"""
    QCParameters

Quality-control thresholds applied by [`threshold_qc`](@ref).

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
- `missing_key::String`: moment name used to detect "no signal" gates.
- `valid_key::String`: moment name used for valid-data gating.
"""
Base.@kwdef struct GriddingParameters
    beam_inflation::Float64  = 0.01
    power_threshold::Float64 = 0.5
    missing_key::String      = "SQI"
    valid_key::String        = "DBZ"
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
    GridParameters

Aggregates the four spatial-grid sub-specs. Pull out the one matching the
gridding driver you are calling (e.g., `p.grid.cartesian` for
`grid_radar_volume`).
"""
Base.@kwdef struct GridParameters
    cartesian::CartesianGridParameters = CartesianGridParameters()
    latlon::LatLonGridParameters       = LatLonGridParameters()
    rhi::RhiGridParameters             = RhiGridParameters()
    spectral::SpectralGridParameters   = SpectralGridParameters()
end

"""
    IOParameters

I/O fill values. Daisho preserves the distinction between true missing
(`fill_value_missing`, no radar coverage) and clear air
(`fill_value_clear`, scanned, no echo).
"""
Base.@kwdef struct IOParameters
    fill_value_missing::Float64 = -32768.0
    fill_value_clear::Float64   =  -9999.0
end

"""
    DaishoParameters

Top-level immutable runtime configuration loaded from a TOML file. Construct
via `DaishoParameters()` for bundled defaults or `DaishoParameters(path)` for
defaults plus user overrides.

# Fields
- `moments::MomentParameters`
- `qc::QCParameters`
- `gridding::GriddingParameters`
- `grid::GridParameters`
- `io::IOParameters`

# Examples
```julia
p = DaishoParameters()                       # bundled defaults
p = DaishoParameters("config/seapol.toml")   # defaults + user overrides
threshold_qc(raw, raw_dict, qc, qc_dict, p)
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

Construct a `DaishoParameters` instance. With no arguments, loads bundled
defaults from `config/defaults.toml`. With a path, loads the bundled defaults
and then deep-merges the user's TOML on top, so user files only need to
specify keys they wish to override.
"""
function DaishoParameters()
    return DaishoParameters(_load_toml_layered(nothing))
end

function DaishoParameters(path::AbstractString)
    return DaishoParameters(_load_toml_layered(path))
end

# Internal: build from an already-parsed Dict (used by ctors and tests).
function DaishoParameters(d::AbstractDict)
    moments  = _fields_from_dict(_section(d, "fields"))
    qc       = _struct_from_dict(QCParameters,         _section(d, "qc"))
    gridding = _struct_from_dict(GriddingParameters,   _section(d, "gridding"))
    grid     = _grid_from_dict(_section(d, "grid"))
    io       = _struct_from_dict(IOParameters,         _section(d, "io"))
    return DaishoParameters(moments, qc, gridding, grid, io)
end

function _load_toml_layered(user_path::Union{Nothing,AbstractString})
    defaults = TOML.parsefile(DEFAULTS_TOML_PATH)
    if user_path === nothing
        return defaults
    end
    isfile(user_path) || throw(ArgumentError("Daisho parameter file not found: $user_path"))
    user = TOML.parsefile(user_path)
    return _deep_merge(defaults, user)
end

# Recursively merges `b` into `a`. Sub-tables merge key-by-key; leaves take
# `b`'s value. Returns a new Dict (does not mutate).
function _deep_merge(a::AbstractDict, b::AbstractDict)
    out = Dict{String,Any}()
    for (k, v) in a
        out[String(k)] = v
    end
    for (k, v) in b
        ks = String(k)
        if v isa AbstractDict && haskey(out, ks) && out[ks] isa AbstractDict
            out[ks] = _deep_merge(out[ks], v)
        else
            out[ks] = v
        end
    end
    return out
end

_section(d::AbstractDict, key::String) =
    haskey(d, key) ? d[key] : Dict{String,Any}()

# Build any kwdef struct by pulling matching keys out of `d`. Unknown keys
# are reported (so users notice typos). Missing keys fall back to the kwdef
# default.
function _struct_from_dict(::Type{T}, d::AbstractDict) where {T}
    fields = fieldnames(T)
    kw = Dict{Symbol,Any}()
    for (k, v) in d
        sym = Symbol(k)
        if sym in fields
            kw[sym] = _coerce_field(T, sym, v)
        else
            throw(ArgumentError(
                "Unknown key `$(k)` in section for $(T). " *
                "Allowed keys: $(join(fields, ", "))"))
        end
    end
    return T(; kw...)
end

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
    names_raw = get(d, "names", String[])
    names = String.(names_raw)

    grid_type_raw = get(d, "grid_type", Dict{String,String}())
    grid_type = Dict{String,Symbol}()
    for (k, v) in grid_type_raw
        grid_type[String(k)] = Symbol(String(v))
    end

    # Every named field must have a grid_type entry.
    for name in names
        if !haskey(grid_type, name)
            known = join(sort(collect(keys(grid_type))), ", ")
            throw(ArgumentError(
                "[fields]: missing grid_type entry for field `$(name)`. " *
                "Known grid_type keys: $(known)"))
        end
    end

    return MomentParameters(names, grid_type)
end

function _grid_from_dict(d::AbstractDict)
    cartesian = _struct_from_dict(CartesianGridParameters, _section(d, "cartesian"))
    latlon    = _struct_from_dict(LatLonGridParameters,    _section(d, "latlon"))
    rhi       = _struct_from_dict(RhiGridParameters,       _section(d, "rhi"))
    spectral  = _spectral_from_dict(_section(d, "spectral"))
    return GridParameters(cartesian=cartesian, latlon=latlon, rhi=rhi, spectral=spectral)
end

function _spectral_from_dict(d::AbstractDict)
    bc_dict = _section(d, "bc")
    bc = SpectralBCParameters(
        xL = _parse_bc(_section(bc_dict, "xL")),
        xR = _parse_bc(_section(bc_dict, "xR")),
        yL = _parse_bc(_section(bc_dict, "yL")),
        yR = _parse_bc(_section(bc_dict, "yR")),
        zL = _parse_bc(_section(bc_dict, "zL")),
        zR = _parse_bc(_section(bc_dict, "zR")),
    )
    # Strip bc out before forwarding to generic struct loader.
    rest = Dict{String,Any}(k => v for (k, v) in d if k != "bc")
    base = _struct_from_dict(SpectralGridParameters, rest)
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

# Map a BC sub-table to a Springsteel BoundaryConditions. Empty table → NaturalBC().
function _parse_bc(d::AbstractDict)
    isempty(d) && return NaturalBC()
    haskey(d, "type") || throw(ArgumentError(
        "BC sub-table missing required `type` key"))
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
