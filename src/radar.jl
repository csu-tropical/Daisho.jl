# Radar utilities.
#
# After the CfRadial 2.1 refactor (Phase D), the public API is `Volume`-based.
# The lowercase `radar` struct here is internal scaffolding for the gridding
# workers (which continue to operate on flat ray/moment arrays). The
# `as_legacy_radar` bridge in `cfradial_bridge.jl` is what builds it from a
# `Volume`. Do not export `radar` or its helpers; consumers should use
# `Volume`.

"""
    radar

Internal flat-volume scaffolding used by the gridding workers. Constructed
from a public `Volume` via `as_legacy_radar`. Holds per-ray (azimuth,
elevation, time, lat/lon/alt, …) plus a single `(n_rays * n_gates, n_fields)`
moments matrix indexed by a name→column dict.

# Fields
- `scan_name::String`
- `azimuth`, `elevation`: per-ray (degrees).
- `ew_platform`, `ns_platform`, `w_platform`: per-ray platform velocity (m/s).
- `nyquist_velocity`: per-ray (m/s).
- `range`: per-gate (m).
- `time`: per-ray.
- `latitude`, `longitude`, `altitude`: per-ray (mobile) or single-element (stationary).
- `fixed_angles`: per-sweep (degrees).
- `swpstart`, `swpend`: 0-based per-sweep ray indices.
- `moments`: `(n_rays * n_gates, n_fields)` matrix.
"""
Base.@kwdef struct radar
    scan_name::String
    azimuth::Array{Union{Missing, Float32}}
    elevation::Array{Union{Missing, Float32}}
    ew_platform::Array{Union{Missing, Float32}}
    ns_platform::Array{Union{Missing, Float32}}
    w_platform::Array{Union{Missing, Float32}}
    nyquist_velocity::Array{Union{Missing, Float32}}
    range::Array{Union{Missing, Float32}}
    time::Array{DateTime}
    latitude::Array{Union{Missing, Float32}}
    longitude::Array{Union{Missing, Float32}}
    altitude::Array{Union{Missing, Float32}}
    fixed_angles::Array{Union{Missing, Float32}}
    swpstart::Array{Union{Missing, Float32}}
    swpend::Array{Union{Missing, Float32}}
    moments::Array{Union{Missing, Float64}}
    # Half-power beamwidth (degrees) for the angular gate weighting; carried so the
    # `radar`-typed gridders derive the correct `beam_coef` instead of assuming 1°.
    # `@kwdef` default keeps existing keyword constructions working; populated by
    # `as_legacy_radar` from `radar_parameters.beam_width_h`.
    beam_width::Float64 = 1.0
end

"""
    split_sweeps(radar_volume) -> Vector{radar}

Split a multi-sweep flat `radar` into an array of per-sweep `radar` structs.
Used internally by gridders that operate sweep-by-sweep.
"""
function split_sweeps(radar_volume)

    swp_indices = [ range(Int(radar_volume.swpstart[i] + 1),Int(radar_volume.swpend[i] + 1)) for i in eachindex(radar_volume.swpstart) ]

    sweeps = Array{radar}(undef,length(radar_volume.swpstart))
    for i in eachindex(swp_indices)

        swp = swp_indices[i]
        scan_name = radar_volume.scan_name
        azimuth = view(radar_volume.azimuth, swp)
        elevation = view(radar_volume.elevation, swp)
        ew_platform = view(radar_volume.ew_platform, swp)
        ns_platform = view(radar_volume.ns_platform, swp)
        w_platform = view(radar_volume.w_platform, swp)
        nyquist_velocity = view(radar_volume.nyquist_velocity, swp)
        range = radar_volume.range
        time = view(radar_volume.time, swp)
        latitude = Array{Union{Missing, Float32}}(undef, length(swp))
        longitude = Array{Union{Missing, Float32}}(undef, length(swp))
        altitude = Array{Union{Missing, Float32}}(undef, length(swp))
        if length(radar_volume.latitude) == 1
            latitude = repeat(radar_volume.latitude, length(swp))
            longitude = repeat(radar_volume.longitude, length(swp))
            altitude = repeat(radar_volume.altitude, length(swp))
        else
            latitude = view(radar_volume.latitude, swp)
            longitude = view(radar_volume.longitude, swp)
            altitude = view(radar_volume.altitude, swp)
        end
        fixed_angles = view(radar_volume.fixed_angles, i)
        swpstart = view(radar_volume.swpstart, i)
        swpend = view(radar_volume.swpend, i)

        m_start = Int(length(radar_volume.range)*(radar_volume.swpstart[i]) + 1)
        m_end = Int(length(radar_volume.range)*(radar_volume.swpend[i] + 1))
        moments = view(radar_volume.moments, m_start:m_end, :)
        sweeps[i] = radar(scan_name, azimuth, elevation, ew_platform, ns_platform, w_platform, nyquist_velocity, range, time,
            latitude, longitude, altitude, fixed_angles, swpstart, swpend, moments, radar_volume.beam_width)
    end

    return sweeps
end

"""
    beam_height(slant_range, elevation, radar_height) -> Float64

Beam height above MSL using the 4/3 effective Earth radius standard refraction
model. `slant_range` in meters, `elevation` in degrees, `radar_height` in meters.
"""
function beam_height(slant_range, elevation, radar_height)
    Reff = 4.0 * 6371000.0 / 3.0
    elrad = deg2rad(elevation)
    h = sqrt(slant_range^2 + Reff^2 + 2*slant_range*Reff*sin(elrad)) - Reff + radar_height
    return h
end

"""
    dB_to_linear(moment) -> Array

Convert a moment array from decibels to linear units element-wise.
"""
dB_to_linear(moment) = 10.0 .^ (moment[:] ./ 10.0)

"""
    linear_to_dB(moment) -> Array

Convert a moment array from linear units to decibels element-wise.
"""
linear_to_dB(moment) = 10.0 .* log10.(moment[:])

"""
    dB_to_linear!(moment)

In-place dB → linear conversion.
"""
dB_to_linear!(moment) = (moment[:] .= 10.0 .^ (moment[:] ./ 10.0); nothing)

"""
    linear_to_dB!(moment)

In-place linear → dB conversion.
"""
linear_to_dB!(moment) = (moment[:] .= 10.0 .* log10.(moment[:]); nothing)

"""
    get_radar_orientation(file) -> Matrix{Float64}

Read heading, pitch, and roll arrays from a CfRadial NetCDF file. Returns an
`(n_rays, 3)` matrix; columns missing in the file are filled with `NaN`.
"""
function get_radar_orientation(file)
    inputds = NCDataset(file)
    num_rays = length(inputds["azimuth"])
    headingdata = fill(NaN, num_rays)
    pitchdata = fill(NaN, num_rays)
    rolldata = fill(NaN, num_rays)
    if haskey(inputds,"heading")
        headingdata = inputds["heading"][:]
    end
    if haskey(inputds,"pitch")
        pitchdata = inputds["pitch"][:]
    end
    if haskey(inputds,"roll")
        rolldata = inputds["roll"][:]
    end
    return [headingdata pitchdata rolldata]
end
