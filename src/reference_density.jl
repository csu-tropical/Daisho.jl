# Hydrostatic reference density profiles ρ̄(z).
#
# Shared base-state density (and, optionally, temperature) profile consumed by
# the variational-stage anelastic mass continuity constraint
# (`∂u/∂x + ∂v/∂y + ∂w/∂z + w·dlnρ̄/dz = 0`), the radar fall-speed correction,
# and (a future phase) FHC temperature. Configured via the `[reference_state]`
# TOML block → `ReferenceStateParameters` (see `parameters.jl`).
#
# Two sources:
#   "exponential" — a minimal analytic ρ̄(z) = ρ₀·exp(−z/H), optionally with a
#                    linear temperature lapse. No grid needed.
#   "sounding"    — a full hydrostatic reference state built by Springsteel
#                    (`Springsteel.calculate_reference_state`) from a sounding
#                    file, sampled on the grid's vertical mish.
#
# SWAP PATH: `temperature_profile.jl` documents a future
# `temperature_celsius(::ReferenceState, z)` method as the intended successor
# to the piecewise-linear `TemperatureProfile`. This file adds that method for
# both reference-profile types (FHC itself is untouched in this phase).

"""
    AbstractReferenceProfile

Supertype for hydrostatic reference-density profiles ρ̄(z). Concrete subtypes
are [`ExponentialReferenceProfile`](@ref) and [`SoundingReferenceProfile`](@ref).
Build one from a loaded `DaishoParameters` with [`build_reference_profile`](@ref).
Query it with [`reference_density`](@ref), [`dlnrho_dz`](@ref),
[`reference_temperature_kelvin`](@ref), [`freezing_level_m`](@ref), and
[`temperature_celsius`](@ref).
"""
abstract type AbstractReferenceProfile end

"""
    ExponentialReferenceProfile(rho_surface_kg_m3, scale_height_m,
                                 temperature_surface_k, lapse_rate_k_m)

Minimal analytic reference-density profile

    ρ̄(z) = ρ₀·exp(−z / H)

with surface density `ρ₀` (kg m⁻³) and scale height `H` (m); `z` is metres,
matching the grid's z-axis/vertical-mish convention (MSL or AGL). Needs no
grid — usable anywhere a profile is needed without a sounding.

Optionally carries a linear temperature lapse `T(z) = T₀ − Γ·z` when
`temperature_surface_k` is finite; `NaN` (Daisho's convention for "absent")
means no temperature profile is available, and
[`reference_temperature_kelvin`](@ref)/[`freezing_level_m`](@ref) return
`nothing` in that case ([`temperature_celsius`](@ref) throws).

Built from config via `[reference_state]` + [`build_reference_profile`](@ref).
"""
struct ExponentialReferenceProfile <: AbstractReferenceProfile
    rho_surface_kg_m3::Float64
    scale_height_m::Float64
    temperature_surface_k::Float64
    lapse_rate_k_m::Float64
end

"""
    SoundingReferenceProfile(file, sgrid; moisture=true)

Hydrostatic reference-density profile built from a sounding file via
Springsteel's reference-state machinery
(`Springsteel.calculate_reference_state`), sampled at the vertical mish points
of `sgrid`'s k-axis. `file` is the samurai/Scythe sounding format: a surface
line `p_sfc theta q_v` (pressure in hPa, potential temperature in K, vapor
mixing ratio in g/kg), then one `z theta q_v` line per level (`z` in metres).

`moisture=true` (default) includes the vapor partial density in ρ̄ (ρ̄ = ρ̄_d +
ρ̄_v, a `MoistReferenceState`); `moisture=false` builds a dry state (ρ̄ = ρ̄_d).

# Fields
- `z::Vector{Float64}`: vertical mish points (m), ascending.
- `rho::Vector{Float64}`: ρ̄ at each mish point (kg m⁻³).
- `drho_dz::Vector{Float64}`: dρ̄/dz at each mish point (kg m⁻³ m⁻¹), taken
  directly from the reference state's spectrally-consistent derivative slot —
  never finite-differenced.
- `temperature_k::Vector{Float64}`: reference temperature (K) at each mish
  point (`Springsteel.reference_temperature`).
- `state::Springsteel.AbstractReferenceState`: the underlying Springsteel
  reference state, for callers that need more than density/temperature
  (entropy, saturation ratio, sound speed, …).

Off-mish queries ([`reference_density`](@ref), [`dlnrho_dz`](@ref)) use cubic
Hermite interpolation on `(rho, drho_dz)` between the bracketing mish levels,
clamped to the end values outside `[z[1], z[end]]`.
"""
struct SoundingReferenceProfile <: AbstractReferenceProfile
    z::Vector{Float64}
    rho::Vector{Float64}
    drho_dz::Vector{Float64}
    temperature_k::Vector{Float64}
    state::Springsteel.AbstractReferenceState
end

function SoundingReferenceProfile(file::AbstractString, sgrid::Springsteel.SpringsteelGrid;
        moisture::Bool=true)
    isfile(file) || throw(ArgumentError(
        "SoundingReferenceProfile: sounding file not found: $file"))

    column = Springsteel.reference_column(sgrid, sgrid.params)
    z = Vector{Float64}(column.mishPoints)
    state = Springsteel.calculate_reference_state(file, z, column; moisture=moisture)

    rho_d = Springsteel.ref_rho_d(state)
    rho_v = Springsteel.ref_rho_v(state)
    is_moist = rho_v !== 0.0

    rho     = is_moist ? Vector{Float64}(rho_d[:, 1] .+ rho_v[:, 1]) : Vector{Float64}(rho_d[:, 1])
    drho_dz = is_moist ? Vector{Float64}(rho_d[:, 2] .+ rho_v[:, 2]) : Vector{Float64}(rho_d[:, 2])
    temperature_k = Vector{Float64}(Springsteel.reference_temperature(state))

    return SoundingReferenceProfile(z, rho, drho_dz, temperature_k, state)
end

# Cubic Hermite interpolation of (rho, drho_dz) at `z`, between the
# bracketing mish levels, clamped to the end values outside the mish range.
# Returns (value, d/dz). Never finite-differences — the derivative slot comes
# straight from the stored `drho_dz` (itself spectrally consistent by
# construction), and the Hermite basis reproduces it exactly at the nodes.
function _hermite_sample(r::SoundingReferenceProfile, z::Float64)
    zs = r.z
    n = length(zs)
    n == 1 && return r.rho[1], r.drho_dz[1]
    if z <= zs[1]
        return r.rho[1], r.drho_dz[1]
    elseif z >= zs[n]
        return r.rho[n], r.drho_dz[n]
    end
    hi = searchsortedfirst(zs, z)
    zs[hi] == z && return r.rho[hi], r.drho_dz[hi]
    lo = hi - 1
    h = zs[hi] - zs[lo]
    t = (z - zs[lo]) / h
    t2 = t * t
    t3 = t2 * t

    h00 = 2t3 - 3t2 + 1
    h10 = t3 - 2t2 + t
    h01 = -2t3 + 3t2
    h11 = t3 - t2
    val = h00 * r.rho[lo] + h10 * h * r.drho_dz[lo] +
          h01 * r.rho[hi] + h11 * h * r.drho_dz[hi]

    dh00 = (6t2 - 6t) / h
    dh10 = 3t2 - 4t + 1
    dh01 = (-6t2 + 6t) / h
    dh11 = 3t2 - 2t
    deriv = dh00 * r.rho[lo] + dh10 * r.drho_dz[lo] +
            dh01 * r.rho[hi] + dh11 * r.drho_dz[hi]

    return val, deriv
end

"""
    reference_density(r::AbstractReferenceProfile, z::Real) -> Float64

Reference density ρ̄(z) in kg m⁻³.
"""
reference_density(r::ExponentialReferenceProfile, z::Real) =
    r.rho_surface_kg_m3 * exp(-Float64(z) / r.scale_height_m)

reference_density(r::SoundingReferenceProfile, z::Real) = _hermite_sample(r, Float64(z))[1]

"""
    dlnrho_dz(r::AbstractReferenceProfile, z::Real) -> Float64

d(ln ρ̄)/dz in m⁻¹ — the term the anelastic mass-continuity pseudo-observation
needs (`w·dlnρ̄/dz`). For the exponential profile this is exactly `−1/H`
everywhere; for the sounding profile it is `drho_dz(z) / rho(z)` from the
Hermite-sampled pair.
"""
dlnrho_dz(r::ExponentialReferenceProfile, z::Real) = -1.0 / r.scale_height_m

function dlnrho_dz(r::SoundingReferenceProfile, z::Real)
    val, deriv = _hermite_sample(r, Float64(z))
    return deriv / val
end

"""
    reference_temperature_kelvin(r::AbstractReferenceProfile, z::Real) -> Union{Float64,Nothing}

Reference temperature (K) at height `z`, or `nothing` when no temperature
profile is available. For [`ExponentialReferenceProfile`](@ref), `nothing`
when `temperature_surface_k` is `NaN` (no temperature configured); otherwise
the linear lapse `T₀ − Γz`. For [`SoundingReferenceProfile`](@ref), always
available (piecewise-linear interpolation of the stored mish temperatures;
never `nothing`).
"""
function reference_temperature_kelvin(r::ExponentialReferenceProfile, z::Real)
    isfinite(r.temperature_surface_k) || return nothing
    return r.temperature_surface_k - r.lapse_rate_k_m * Float64(z)
end

function reference_temperature_kelvin(r::SoundingReferenceProfile, z::Real)
    zs = r.z
    n = length(zs)
    zf = Float64(z)
    n == 1 && return r.temperature_k[1]
    zf <= zs[1] && return r.temperature_k[1]
    zf >= zs[n] && return r.temperature_k[n]
    hi = searchsortedfirst(zs, zf)
    zs[hi] == zf && return r.temperature_k[hi]
    lo = hi - 1
    frac = (zf - zs[lo]) / (zs[hi] - zs[lo])
    return r.temperature_k[lo] + frac * (r.temperature_k[hi] - r.temperature_k[lo])
end

"""
    freezing_level_m(r::AbstractReferenceProfile) -> Union{Float64,Nothing}

Lowest height (m) at which the reference temperature crosses 273.15 K
(0 °C), or `nothing` when unavailable (no temperature profile) or when the
profile never crosses freezing.

For [`ExponentialReferenceProfile`](@ref): `(temperature_surface_k − 273.15) /
lapse_rate_k_m`, or `nothing` when no temperature is configured, the lapse
rate is zero, or the crossing would be below the surface (`z < 0`). For
[`SoundingReferenceProfile`](@ref): linear interpolation between the first
pair of adjacent mish levels whose temperatures bracket 273.15 K.
"""
function freezing_level_m(r::ExponentialReferenceProfile)
    isfinite(r.temperature_surface_k) || return nothing
    r.lapse_rate_k_m == 0.0 && return nothing
    z = (r.temperature_surface_k - 273.15) / r.lapse_rate_k_m
    return z >= 0.0 ? z : nothing
end

function freezing_level_m(r::SoundingReferenceProfile)
    T = r.temperature_k
    z = r.z
    for i in 1:(length(z) - 1)
        t1, t2 = T[i] - 273.15, T[i+1] - 273.15
        if t1 == 0.0
            return z[i]
        elseif t1 * t2 < 0.0
            frac = t1 / (t1 - t2)
            return z[i] + frac * (z[i+1] - z[i])
        end
    end
    T[end] == 273.15 && return z[end]
    return nothing
end

"""
    temperature_celsius(r::ExponentialReferenceProfile, z::Real) -> Float64
    temperature_celsius(r::SoundingReferenceProfile, z::Real) -> Float64

Reference temperature (°C) at height `z`. Additional methods of the existing
[`temperature_celsius`](@ref) accessor (see the swap-path note in
`temperature_profile.jl`): FHC keeps working unchanged against a
`TemperatureProfile`, while a reference-density profile can now answer the
same query. Throws `ArgumentError` when no temperature is available (the
exponential profile with `temperature_surface_k = NaN`).
"""
function temperature_celsius(r::ExponentialReferenceProfile, z::Real)
    Tk = reference_temperature_kelvin(r, z)
    Tk === nothing && throw(ArgumentError(
        "temperature_celsius: this ExponentialReferenceProfile has no " *
        "temperature profile (temperature_surface_k is NaN); set it in " *
        "[reference_state] to enable temperature queries."))
    return Tk - 273.15
end

function temperature_celsius(r::SoundingReferenceProfile, z::Real)
    Tk = reference_temperature_kelvin(r, z)
    Tk === nothing && throw(ArgumentError(
        "temperature_celsius: no temperature available for this " *
        "SoundingReferenceProfile."))
    return Tk - 273.15
end

"""
    build_reference_profile(p::DaishoParameters, sgrid) -> AbstractReferenceProfile

Build the reference-density profile configured by `p.reference_state`,
dispatching on `p.reference_state.source`. The exponential source ignores
`sgrid` entirely (it needs no grid); the sounding source reads
`p.reference_state.sounding_file` on `sgrid`'s vertical mish (see
[`SoundingReferenceProfile`](@ref)).
"""
function build_reference_profile(p::DaishoParameters, sgrid)
    rp = p.reference_state
    if rp.source == "exponential"
        return ExponentialReferenceProfile(rp.density_surface_kg_m3, rp.density_scale_height_m,
                                            rp.temperature_surface_k, rp.lapse_rate_k_m)
    elseif rp.source == "sounding"
        return SoundingReferenceProfile(rp.sounding_file, sgrid; moisture=rp.moisture)
    else
        throw(ArgumentError(
            "build_reference_profile: unknown [reference_state] source " *
            "\"$(rp.source)\"; allowed: \"exponential\", \"sounding\"."))
    end
end
