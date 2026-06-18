# Temporary 1-D temperature profile for hydrometeor classification
#
# The fuzzy HID (`csu_fhc_summer`) needs absolute temperature in °C as a function
# of height. Daisho does not yet carry a thermodynamic reference state, so this is
# a deliberately minimal, swappable stand-in: a piecewise-linear T(z) table.
#
# SWAP PATH: Scythe.jl builds a full hydrostatic `ReferenceState` (potential
# temperature + pressure + density) from a sounding; that utility is expected to
# move into Springsteel as a shared reference state (also needed for dual-Doppler
# mass continuity). When it lands, add a method
#   `temperature_celsius(::ReferenceState, z)`
# that derives absolute T from θ and the hydrostatic pressure. FHC depends ONLY on
# the `temperature_celsius(profile, z)` accessor below, so that swap requires no
# change to the classification code. Keep the accessor name aligned with the
# future shared utility.

"""
    TemperatureProfile(heights, temperatures)

A piecewise-linear vertical temperature profile. `heights` are in meters (MSL or
AGL, matching the grid's z-axis convention) and `temperatures` are in °C. Entries
are sorted by height on construction; values outside the range are held constant
at the nearest endpoint (no extrapolation).

This is a temporary stand-in for a full hydrostatic reference state (see the swap
note in `temperature_profile.jl`). Sample it with [`temperature_celsius`](@ref).
"""
struct TemperatureProfile
    heights::Vector{Float64}      # meters, ascending
    temperatures::Vector{Float64} # °C

    function TemperatureProfile(heights::AbstractVector, temperatures::AbstractVector)
        length(heights) == length(temperatures) ||
            throw(ArgumentError("heights and temperatures must have equal length"))
        length(heights) >= 1 ||
            throw(ArgumentError("temperature profile needs at least one level"))
        perm = sortperm(collect(Float64, heights))
        h = collect(Float64, heights)[perm]
        t = collect(Float64, temperatures)[perm]
        allunique(h) ||
            throw(ArgumentError("temperature profile heights must be unique"))
        return new(h, t)
    end
end

"""
    TemperatureProfile(pairs)

Construct from an iterable of `(height_m, temperature_C)` pairs/tuples — convenient
for inline TOML configuration, e.g. `[[0.0, 25.0], [5000.0, -8.0]]`.
"""
function TemperatureProfile(pairs)
    h = Float64[]
    t = Float64[]
    for p in pairs
        length(p) == 2 ||
            throw(ArgumentError("each temperature entry must be (height_m, temperature_C)"))
        push!(h, Float64(p[1]))
        push!(t, Float64(p[2]))
    end
    return TemperatureProfile(h, t)
end

"""
    read_temperature_profile(path) -> TemperatureProfile

Read a two-column text file of `height_m temperature_C` (whitespace- or
comma-delimited; blank lines and `#` comments ignored).
"""
function read_temperature_profile(path::AbstractString)
    h = Float64[]
    t = Float64[]
    open(path, "r") do io
        for line in eachline(io)
            s = strip(line)
            (isempty(s) || startswith(s, "#")) && continue
            parts = split(s, (',', ' ', '\t'); keepempty = false)
            length(parts) >= 2 ||
                throw(ArgumentError("malformed temperature profile line: \"$line\""))
            push!(h, parse(Float64, parts[1]))
            push!(t, parse(Float64, parts[2]))
        end
    end
    return TemperatureProfile(h, t)
end

"""
    temperature_celsius(profile, z) -> Float64

Temperature (°C) at height `z` (meters) by linear interpolation, clamped to the
profile endpoints outside its range. This is the single accessor the HID code
depends on; a future shared reference state should implement the same method.
"""
function temperature_celsius(profile::TemperatureProfile, z::Real)
    h = profile.heights
    t = profile.temperatures
    n = length(h)
    n == 1 && return t[1]
    z <= h[1] && return t[1]
    z >= h[n] && return t[n]
    # Binary search for the bracketing interval.
    hi = searchsortedfirst(h, z)   # first index with h[hi] >= z
    h[hi] == z && return t[hi]
    lo = hi - 1
    frac = (z - h[lo]) / (h[hi] - h[lo])
    return t[lo] + frac * (t[hi] - t[lo])
end

"""
    temperature_celsius(profile, z_axis::AbstractVector) -> Vector{Float64}

Sample the profile at every height in `z_axis` (meters).
"""
temperature_celsius(profile::TemperatureProfile, z_axis::AbstractVector) =
    Float64[temperature_celsius(profile, z) for z in z_axis]
