# Hybrid scan: a near-surface 2-D product built from a set of same-time gridded PPI tilts.
#
# A single low tilt is the best look at the surface, but it is also the one most often
# blocked, ducted, or simply out of range. The hybrid scan starts from that base tilt and,
# wherever it has no measurement, walks upward through the higher tilts and takes the first
# one whose beam is still near the surface at that cell — bounded by
# `beam_height_maximum`, so a fill never comes from aloft. The result is one 2-D field set
# plus a record of which tilt supplied each cell.
#
# This runs *after* the `[echo]` products, since the fields it usually carries (blended
# rain rate, hydrometeor ID) are echo outputs computed on the gridded polarimetric
# variables.
#
# Two entry points, mirroring `echo_products.jl`:
#   * `apply_hybrid_scan` — pure, in-memory: takes per-tilt field dicts and returns the
#     merged field dict.
#   * `build_hybrid_scan` — standalone: reads a set of Daisho gridded PPI NetCDFs,
#     resolves each one's tilt, and writes the product as another gridded PPI.
#
# Sentinel policy. One question decides everything — "has this tilt answered this cell?"
# — and `require_detection` sets what counts as an answer, identically at the base tilt
# and at the tilts above it:
#   * `require_detection = false` (default): any non-`fill_value` reading answers, so a
#     tilt reporting `undetect` has genuinely observed clear air and ends the search.
#   * `require_detection = true`: only a real detection answers; `undetect` keeps the
#     search going. This reproduces the original PICCOLO script.
# The base tilt is *not* otherwise privileged: the only asymmetry is that it is never
# height-gated (at long range even the lowest sweep can exceed `beam_height_maximum`,
# and it is still the best available look), while a fill from above must be at or below
# that height.
#
# Cells no tilt answers keep the base tilt's (sentinel) values. The elevation/height
# outputs name the tilt each value came from, and are `fill_value` only where the base
# was never measured at all.

"""
    hybrid_beam_heights(X, Y, elevation, hp::HybridScanParameters) -> Matrix{Float64}

Per-cell beam height (metres) over an `X`/`Y` Cartesian grid for a sweep at
`elevation` degrees, via the 4/3-effective-earth [`beam_height`](@ref) model with
the antenna at `hp.radar_altitude`.

`X` and `Y` are metres relative to the radar, so the horizontal distance to a cell
stands in for the slant range. At the low tilts a hybrid scan selects from, the two
differ by well under a beam width; this is the same approximation the original
PICCOLO script made.
"""
function hybrid_beam_heights(X::AbstractVector, Y::AbstractVector, elevation::Real,
                             hp::HybridScanParameters)
    h = Array{Float64}(undef, length(X), length(Y))
    @inbounds for j in eachindex(Y), i in eachindex(X)
        r = sqrt(Float64(X[i])^2 + Float64(Y[j])^2)
        h[i, j] = beam_height(r, elevation, hp.radar_altitude)
    end
    return h
end

# Per-tilt heights: prefer the gridded beam-height field when configured and present,
# falling back to geometry cell-by-cell wherever that field carries a sentinel.
function _hybrid_heights(fields::AbstractDict, X::AbstractVector, Y::AbstractVector,
                         elevation::Real, hp::HybridScanParameters, io::IOParameters)
    geometric = hybrid_beam_heights(X, Y, elevation, hp)
    (isempty(hp.height_field) || !haskey(fields, hp.height_field)) && return geometric
    gridded = fields[hp.height_field]
    size(gridded) == size(geometric) || throw(ArgumentError(
        "hybrid scan: height field \"$(hp.height_field)\" is $(size(gridded)) but the " *
        "grid is $(size(geometric))."))
    @inbounds for i in eachindex(geometric)
        v = gridded[i]
        _dp_invalid(v, io) || (geometric[i] = Float64(v))
    end
    return geometric
end

# Index (into an ascending-by-elevation tilt vector) of the base tilt: the configured
# `base_angle` when one of the tilts is within tolerance of it, else the lowest tilt.
function _hybrid_base_index(sorted, hp::HybridScanParameters)
    if hp.base_angle !== nothing
        best, best_delta = 0, Inf
        for (k, t) in enumerate(sorted)
            delta = abs(Float64(t.elevation) - hp.base_angle)
            if delta <= hp.base_angle_tolerance && delta < best_delta
                best, best_delta = k, delta
            end
        end
        best == 0 || return best
    end
    return 1
end

# Which fields to carry through the selection. An empty `hp.fields` means every field
# present across the tilts, which keeps the output a drop-in gridded PPI.
function _hybrid_carried_fields(sorted, hp::HybridScanParameters)
    isempty(hp.fields) || return hp.fields
    names = Set{String}()
    for t in sorted
        union!(names, String.(keys(t.fields)))
    end
    return sort!(collect(names))
end

@inline _hybrid_height_at(heights::AbstractArray, i) = Float64(heights[i])
@inline _hybrid_height_at(h::Real, _) = Float64(h)

# Has this tilt answered this cell? The single science choice behind the hybrid scan,
# applied identically to the base tilt (does it still need a fill?) and to the tilts
# above it (does this one supply the fill?).
@inline _hybrid_answered(v, io, require_detection) =
    require_detection ? !_dp_invalid(v, io) : !_dp_missing(v, io)

"""
    hybrid_scan_output_names(hp::HybridScanParameters) -> Vector{String}

The names of the variables a hybrid scan writes: the carried fields plus the
configured elevation and height outputs.

When `hp.fields` is empty the carried set is "every field present in the tilts" and
is only known at runtime, so the returned list holds just the elevation/height
outputs.
"""
function hybrid_scan_output_names(hp::HybridScanParameters)
    names = copy(hp.fields)
    isempty(hp.elevation_output) || push!(names, hp.elevation_output)
    isempty(hp.height_output)    || push!(names, hp.height_output)
    return names
end

"""
    apply_hybrid_scan(tilts, hp::HybridScanParameters; io::IOParameters)
        -> Dict{String,Array{Float32}}

Collapse a set of same-time gridded tilts into one near-surface field set.

`tilts` is any iterable of objects with:
  * `elevation` — the tilt's elevation angle in degrees,
  * `fields`    — a name → array dict of that tilt's gridded fields (all the same shape),
  * `heights`   — per-cell beam heights in metres, either an array shaped like the
    fields or a scalar. Build these with [`hybrid_beam_heights`](@ref).

The base tilt (see `hp.base_angle`) seeds every output. Every cell the base tilt does
not answer then looks upward through the higher tilts in ascending order and takes the
first that both answers the cell and has a beam height there at or below
`hp.beam_height_maximum`.

"Answers" means the same thing at every tilt, and `hp.require_detection` is the single
switch: with `false` (the default) any non-`fill_value` reading answers, so a tilt
reporting `undetect` has observed clear air and ends the search; with `true` only a
real detection answers, and the search climbs past clear air. The base tilt is not
otherwise special — in particular it is never height-gated, since at long range even
the lowest sweep can exceed the limit and is still the best available look.

Returns the carried fields plus, when configured, `hp.elevation_output` (the tilt each
value came from) and `hp.height_output` (that tilt's beam height there). Cells no tilt
answers keep the base tilt's values, and their elevation/height outputs report the base
tilt — or `io.fill_value` where even the base was never measured.
"""
function apply_hybrid_scan(tilts, hp::HybridScanParameters; io::IOParameters)
    sorted = sort(collect(tilts); by = t -> Float64(t.elevation))
    isempty(sorted) && throw(ArgumentError(
        "apply_hybrid_scan: no tilts supplied; a hybrid scan needs at least one."))

    base_idx = _hybrid_base_index(sorted, hp)
    base = sorted[base_idx]
    haskey(base.fields, hp.select_field) || throw(ArgumentError(
        "apply_hybrid_scan: select_field \"$(hp.select_field)\" not found in the " *
        "base tilt at $(base.elevation)° " *
        "(have: $(join(sort(String.(collect(keys(base.fields)))), ", ")))."))

    names = _hybrid_carried_fields(sorted, hp)
    select_base = base.fields[hp.select_field]
    shape = size(select_base)
    fv = Float32(io.fill_value)

    out = Dict{String,Array{Float32}}()
    for name in names
        out[name] = haskey(base.fields, name) ?
            Float32.(base.fields[name]) : fill(fv, shape)
    end

    # Elevation/height provenance names the tilt each value came from. Seeded with the
    # base tilt, since unfilled cells keep the base tilt's values; fill_value only where
    # the base never measured at all. Overwritten below wherever a tilt fills in.
    elev = isempty(hp.elevation_output) ? nothing : Array{Float32}(undef, shape)
    hgt  = isempty(hp.height_output)    ? nothing : Array{Float32}(undef, shape)
    base_angle = Float32(base.elevation)
    @inbounds for i in eachindex(select_base)
        unmeasured = _dp_missing(select_base[i], io)
        elev === nothing || (elev[i] = unmeasured ? fv : base_angle)
        hgt === nothing || (hgt[i] = unmeasured ? fv :
            Float32(_hybrid_height_at(base.heights, i)))
    end

    @inbounds for i in eachindex(select_base)
        # The base tilt already answered this cell; nothing above can improve on it.
        _hybrid_answered(select_base[i], io, hp.require_detection) && continue
        for k in eachindex(sorted)
            k == base_idx && continue
            # Only ever look upward: a tilt at or below the base adds nothing.
            Float64(sorted[k].elevation) <= Float64(base.elevation) && continue
            h = _hybrid_height_at(sorted[k].heights, i)
            (isfinite(h) && h <= hp.beam_height_maximum) || continue
            select_k = get(sorted[k].fields, hp.select_field, nothing)
            select_k === nothing && continue
            _hybrid_answered(select_k[i], io, hp.require_detection) || continue

            for name in names
                f = get(sorted[k].fields, name, nothing)
                out[name][i] = f === nothing ? fv : Float32(f[i])
            end
            elev === nothing || (elev[i] = Float32(sorted[k].elevation))
            hgt === nothing || (hgt[i] = Float32(h))
            break
        end
    end

    elev === nothing || (out[hp.elevation_output] = elev)
    hgt === nothing || (out[hp.height_output] = hgt)
    return out
end

"""
    hybrid_scan_tilt_angle(file, hp::HybridScanParameters) -> Union{Float64,Nothing}

The elevation angle of a gridded PPI `file`: the scalar `fixed_angle` variable Daisho
writes, else the `fixed_angle` global attribute, else the first capture group of
`hp.angle_pattern` applied to the basename (for archives predating both). Returns
`nothing` when the tilt cannot be determined.
"""
function hybrid_scan_tilt_angle(file::AbstractString, hp::HybridScanParameters)
    angle = NCDataset(file) do ds
        if haskey(ds, "fixed_angle")
            return Float64(ds["fixed_angle"].var[])
        elseif haskey(ds.attrib, "fixed_angle")
            return Float64(ds.attrib["fixed_angle"])
        end
        return nothing
    end
    angle === nothing || return angle

    isempty(hp.angle_pattern) && return nothing
    m = match(Regex(hp.angle_pattern), basename(file))
    (m === nothing || m.captures[1] === nothing) && return nothing
    return tryparse(Float64, m.captures[1])
end

"""
    build_hybrid_scan(files, output_file, p::DaishoParameters; index_time=nothing)
        -> Vector{String}

Read a set of Daisho gridded PPI NetCDF `files` covering one time, collapse them into
a hybrid scan per `p.hybrid_scan`, and write the result to `output_file` in the same
gridded-PPI layout (so [`read_gridded_ppi`](@ref) and the plotting steps read it back
unchanged). Returns the variable names written.

Each file's tilt is resolved by [`hybrid_scan_tilt_angle`](@ref); files whose tilt
cannot be determined are skipped with a warning rather than silently mis-ordered. The
coordinate scaffolding and, by default, the output time are taken from the base tilt's
file.

The standalone counterpart to the Sparrow workflow step: useful for reprocessing an
archive of gridded PPIs without re-gridding.
"""
function build_hybrid_scan(files, output_file::AbstractString, p::DaishoParameters;
                           index_time = nothing)
    hp = p.hybrid_scan
    io = p.io

    tilts = NamedTuple{(:elevation, :fields, :heights, :file),
                       Tuple{Float64,<:AbstractDict,Matrix{Float64},String}}[]
    for file in files
        angle = hybrid_scan_tilt_angle(file, hp)
        if angle === nothing
            @warn "build_hybrid_scan: skipping $file — no fixed_angle in the file and " *
                  "no `angle_pattern` match on its name."
            continue
        end
        g = read_gridded_ppi(file, p)
        heights = _hybrid_heights(g.fields, g.X, g.Y, angle, hp, io)
        push!(tilts, (elevation = Float64(angle), fields = g.fields,
                      heights = heights, file = String(file)))
    end
    isempty(tilts) && throw(ArgumentError(
        "build_hybrid_scan: none of the $(length(collect(files))) input file(s) could " *
        "be resolved to an elevation angle; nothing to combine."))

    sort!(tilts; by = t -> t.elevation)
    template = tilts[_hybrid_base_index(tilts, hp)].file

    fields = apply_hybrid_scan(tilts, hp; io = io)
    write_gridded_fields_2d(output_file, template, fields, p.grid.metadata;
        index_time = index_time, fill_value = io.fill_value, undetect = io.undetect,
        extra_attrib = Pair{String,Any}["hybrid_scan_tilts" =>
            join((string(t.elevation) for t in tilts), " ")])
    return sort!(collect(keys(fields)))
end
