# Fuzzy-logic hydrometeor identification (HID / FHC)
#
# Julia port of CSU_RadarTools `csu_fhc_summer` (warm-season, 10-category fuzzy
# logic classification). Faithful to
#   ~/Development/CSU_RadarTools/csu_radartools/csu_fhc.py        (csu_fhc_summer)
#   ~/Development/CSU_RadarTools/csu_radartools/beta_functions.py (get_mbf_sets_summer)
#
# The membership beta function (MBF) is
#
#     beta(x) = 1 / (1 + (((x - m) / a)^2)^b)
#
# with per-class (m, a, b) coefficients shipped as CSV tables under
# `data/beta_function_parameters/` (copied verbatim from CSU_RadarTools for
# provenance). Each `<band>-band_<Variable>.csv` holds 10 rows (one per class)
# with columns m, a, b.
#
# Design notes for the gridded use case (differs from the Python original, which
# classified per radar gate):
#   * Operates on the *gridded* radar variables — the beam-power-weighted averages
#     produced by the Daisho accumulator — rather than raw gates.
#   * Sentinel/NaN-aware: a gate whose DBZ is non-finite or equals a fill/undetect
#     sentinel is left unclassified (`masked`, default 0). The echo-products
#     layer is responsible for mapping that to the proper fill_value/undetect
#     sentinel so the missing-vs-clear-air distinction is preserved.
#   * Missing individual polarimetric variables are dropped from the weighted sum
#     on a per-gate basis (per-gate renormalization). When every provided variable
#     is finite this is identical to the Python global weighting, so reference
#     fixtures match exactly.

"""
    FHC_SUMMER_CLASSES

Summer (warm-season) hydrometeor classes recognized by the fuzzy-logic
classifier, in argmax order so that the integer class label returned by
[`csu_fhc_summer`](@ref) indexes directly into this tuple (class `i` ==
`FHC_SUMMER_CLASSES[i]`). The ten classes are drizzle, rain, ice crystals,
aggregates, wet snow, vertical ice, low-density graupel, high-density graupel,
hail, and big drops. A class label of `0` denotes an unclassified (masked) gate.
"""
const FHC_SUMMER_CLASSES = ("Drizzle", "Rain", "Ice Crystals", "Aggregates",
    "Wet Snow", "Vertical Ice", "Low-Density Graupel", "High-Density Graupel",
    "Hail", "Big Drops")

"""
    FHC_N_TYPES

Number of hydrometeor classes in [`FHC_SUMMER_CLASSES`](@ref) (10). This is the
leading dimension of the per-class score array produced during classification.
"""
const FHC_N_TYPES = length(FHC_SUMMER_CLASSES)

"""
    DEFAULT_FHC_WEIGHTS

Default relative weights for the fuzzy hydrometeor classification variables
(matches CSU_RadarTools `DEFAULT_WEIGHTS`): reflectivity `DZ`, differential
reflectivity `DR`, specific differential phase `KD`, copolar correlation `RH`,
linear depolarization ratio `LD`, and temperature `T`. In the `:hybrid` method
only the polarimetric weights (`DR`, `KD`, `RH`, `LD`) enter the weighted sum;
`DZ` and `T` act as pure multipliers, so their weights are unused there. Pass a
`NamedTuple` with the same keys to [`csu_fhc_summer`](@ref) to override.
"""
const DEFAULT_FHC_WEIGHTS = (DZ = 1.5, DR = 0.8, KD = 1.0, RH = 0.8, LD = 0.5, T = 0.4)

# Polarimetric variable keys participating in the hybrid weighted sum.
const _FHC_POL_KEYS = (:DR, :KD, :RH, :LD)

"""
    beta_mbf(x, m, a, b)

Membership beta function `1 / (1 + (((x - m) / a)^2)^b)`. Returns a value in
`(0, 1]`. This is the elementwise kernel of the fuzzy classification.
"""
@inline beta_mbf(x, m, a, b) = 1.0 / (1.0 + (((x - m) / a)^2)^b)

# Directory holding the shipped beta-function coefficient CSVs.
_beta_param_dir() = normpath(joinpath(@__DIR__, "..", "data", "beta_function_parameters"))

# Read one `<band>-band_<Variable>.csv` into per-class (m, a, b) vectors. Mirrors
# CSU_RadarTools `get_beta_set`: every column is divided by `factor` (only the
# temperature table uses factor != 1, via temp_factor, to broaden the T MBFs).
function _read_beta_csv(path::AbstractString; factor::Real = 1.0)
    m = Float64[]
    a = Float64[]
    b = Float64[]
    open(path, "r") do io
        readline(io)  # header: ",m,a,b"
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line, ',')
            # Column 1 is the class name (no embedded commas); columns 2:4 are m,a,b.
            push!(m, parse(Float64, strip(parts[2])) / factor)
            push!(a, parse(Float64, strip(parts[3])) / factor)
            push!(b, parse(Float64, strip(parts[4])) / factor)
        end
    end
    return (m = m, a = a, b = b)
end

"""
    get_mbf_sets_summer(band="S"; use_temp=true, temp_factor=1.0)

Load the warm-season membership beta function coefficient sets for a frequency
`band` ("S", "C", or "X"). Returns a `Dict{Symbol,NamedTuple}` keyed by the
classifier variable labels (`:DZ, :DR, :KD, :LD, :RH, :T`), each a `(m, a, b)`
NamedTuple of length `FHC_N_TYPES`. The `:T` entry is `nothing` when
`use_temp=false`. Port of CSU_RadarTools `get_mbf_sets_summer`.
"""
function get_mbf_sets_summer(band::AbstractString = "S"; use_temp::Bool = true,
        temp_factor::Real = 1.0)
    dir = _beta_param_dir()
    f(name) = joinpath(dir, string(band, "-band_", name, ".csv"))
    sets = Dict{Symbol,Union{NamedTuple,Nothing}}()
    sets[:DZ] = _read_beta_csv(f("Reflectivity"))
    sets[:DR] = _read_beta_csv(f("Differential_Reflectivity"))
    sets[:KD] = _read_beta_csv(f("Specific_Differential_Phase"))
    sets[:LD] = _read_beta_csv(f("Linear_Depolarization_Ratio"))
    sets[:RH] = _read_beta_csv(f("Correlation_Coefficient"))
    sets[:T]  = use_temp ? _read_beta_csv(f("Temperature"); factor = temp_factor) : nothing
    return sets
end

# Weight lookup that accepts either a NamedTuple or a Dict (string or symbol keys).
_fhc_weight(w::NamedTuple, k::Symbol) = getfield(w, k)
_fhc_weight(w::AbstractDict, k::Symbol) = haskey(w, k) ? w[k] : w[String(k)]

# True when a datum is a usable measurement (finite and not a sentinel).
@inline function _fhc_valid(x, fill_value, undetect)
    return isfinite(x) && x != fill_value && x != undetect
end

"""
    csu_fhc_summer(; dz, zdr=nothing, kdp=nothing, rho=nothing, ldr=nothing,
                   T=nothing, weights=DEFAULT_FHC_WEIGHTS, method=:hybrid,
                   band="S", use_temp=true, temp_factor=1.0,
                   return_scores=false, fill_value=-32768.0, undetect=-9999.0,
                   masked=0)

Fuzzy-logic hydrometeor identification for warm-season precipitation. Julia port
of CSU_RadarTools `csu_fhc_summer`.

`dz` (reflectivity, dBZ) is required; `zdr` (dB), `kdp` (deg/km), `rho`
(unitless), `ldr` (dB) and `T` (temperature, °C) are optional and, when supplied,
must broadcast to the same shape as `dz`. Inputs may be scalars or arrays of any
dimensionality.

Returns an integer classification array (or scalar) with values `1:FHC_N_TYPES`
(see [`FHC_SUMMER_CLASSES`](@ref)); gates whose `dz` is non-finite or equal to a
sentinel are set to `masked`. With `return_scores=true`, returns the full score
array of shape `(FHC_N_TYPES, size(dz)...)` instead (matching the Python
`return_scores` ordering).

Methods: `:hybrid` (default) weights the polarimetric variables into a normalized
score, then multiplies by the temperature and reflectivity memberships; `:linear`
treats all present variables as a single weighted sum.
"""
function csu_fhc_summer(; dz, zdr = nothing, kdp = nothing, rho = nothing,
        ldr = nothing, T = nothing, weights = DEFAULT_FHC_WEIGHTS,
        method::Symbol = :hybrid, band::AbstractString = "S",
        use_temp::Bool = true, temp_factor::Real = 1.0,
        return_scores::Bool = false, fill_value::Real = -32768.0,
        undetect::Real = -9999.0, masked::Integer = 0)

    if T === nothing
        use_temp = false
    end
    if !(method === :hybrid || method === :linear)
        throw(ArgumentError("FHC method must be :hybrid or :linear, got :$method"))
    end

    sets = get_mbf_sets_summer(band; use_temp = use_temp, temp_factor = temp_factor)

    # Reference shape/size from dz (the required field).
    dz_arr = dz isa AbstractArray ? dz : fill(Float64(dz), 1)
    shp = size(dz_arr)
    n = length(dz_arr)

    # Provided optional fields, aligned to dz's linear indexing.
    pol_data = (DR = zdr, KD = kdp, RH = rho, LD = ldr)
    provided_pol = Tuple(k for k in _FHC_POL_KEYS if getfield(pol_data, k) !== nothing)

    _at(::Nothing, i) = nothing
    _at(v::AbstractArray, i) = v[i]
    _at(v, i) = v

    scores = Array{Float64}(undef, FHC_N_TYPES, n)
    out = fill(Int(masked), n)

    @inbounds for i in 1:n
        d = dz_arr[i]
        if !_fhc_valid(d, fill_value, undetect)
            scores[:, i] .= 0.0
            continue
        end

        Ti = use_temp ? _at(T, i) : nothing
        t_valid = use_temp && Ti !== nothing && _fhc_valid(Ti, fill_value, undetect)

        # Per-gate set of present polarimetric variables (provided and valid here).
        for c in 1:FHC_N_TYPES
            local score::Float64
            if method === :hybrid
                score = 1.0
                num = 0.0
                den = 0.0
                for k in provided_pol
                    val = _at(getfield(pol_data, k), i)
                    (val === nothing || !_fhc_valid(val, fill_value, undetect)) && continue
                    s = sets[k]
                    w = Float64(_fhc_weight(weights, k))
                    num += w * beta_mbf(val, s.m[c], s.a[c], s.b[c])
                    den += w
                end
                if den > 0.0
                    score *= num / den
                end
                if t_valid
                    s = sets[:T]
                    score *= beta_mbf(Ti, s.m[c], s.a[c], s.b[c])
                end
                s = sets[:DZ]
                score *= beta_mbf(d, s.m[c], s.a[c], s.b[c])
            else  # :linear — single weighted sum over all present variables
                num = 0.0
                den = 0.0
                for k in provided_pol
                    val = _at(getfield(pol_data, k), i)
                    (val === nothing || !_fhc_valid(val, fill_value, undetect)) && continue
                    s = sets[k]
                    w = Float64(_fhc_weight(weights, k))
                    num += w * beta_mbf(val, s.m[c], s.a[c], s.b[c])
                    den += w
                end
                if t_valid
                    s = sets[:T]
                    w = Float64(_fhc_weight(weights, :T))
                    num += w * beta_mbf(Ti, s.m[c], s.a[c], s.b[c])
                    den += w
                end
                let s = sets[:DZ], w = Float64(_fhc_weight(weights, :DZ))
                    num += w * beta_mbf(d, s.m[c], s.a[c], s.b[c])
                    den += w
                end
                score = den > 0.0 ? num / den : 0.0
            end
            scores[c, i] = score
        end

        # argmax over classes (first max wins, matching numpy argmax tie-breaking).
        best = 1
        bestval = scores[1, i]
        for c in 2:FHC_N_TYPES
            if scores[c, i] > bestval
                bestval = scores[c, i]
                best = c
            end
        end
        out[i] = best
    end

    if return_scores
        full = reshape(scores, FHC_N_TYPES, shp...)
        return dz isa AbstractArray ? full : reshape(full, FHC_N_TYPES)
    end
    return dz isa AbstractArray ? reshape(out, shp) : out[1]
end
