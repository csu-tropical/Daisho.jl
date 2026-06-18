# Polarimetric rain-rate algorithms
#
# Julia port of CSU_RadarTools tropical/oceanic rain rate code, faithful to
#   ~/Development/CSU_RadarTools/csu_radartools/common.py                  (component R relations)
#   ~/Development/CSU_RadarTools/csu_radartools/csu_blended_rain_tropical.py (blended algorithm)
#
# Designed around the tropical/oceanic equations of Thompson et al. (2016).
#
# Two layers:
#   * The component R relations — R(Z), R(Kdp), R(Kdp,Zdr), R(Z,Zdr) — are
#     exported as standalone functions so each can be computed, saved, and
#     visualized on its own, and used as a fallback when the blended algorithm's
#     inputs (e.g. Kdp) are missing or noisy.
#   * `calc_blended_rain_tropical` selects, per grid cell, the best estimate via
#     the Thompson flowchart and returns both the blended rate and a method code.
#
# numpy compatibility: a negative base raised to a fractional power yields NaN in
# numpy (with a warning) but throws in Julia, so [`_safe_pow`](@ref) reproduces
# the NaN result. This matters only for component values at unphysical inputs
# (e.g. negative Kdp); such cells are never selected by the blended flowchart.

# ---------------------------------------------------------------------------
# Band-specific coefficients (transcribed verbatim from the `predef` block of
# calc_blended_rain_tropical). The R-Z coefficients are band-independent.
# ---------------------------------------------------------------------------

# R(Z) = (10^(dBZ/10) / a)^(1/b)
const RAIN_RZ_ALL   = (a = 216.0, b = 1.39)   # all/unspecified
const RAIN_RZ_CONV  = (a = 126.0, b = 1.39)   # convective
const RAIN_RZ_STRAT = (a = 291.0, b = 1.55)   # stratiform

# Per-band (a, b, c) for R(Kdp), R(Kdp,Zdr), R(Z,Zdr). The `c` exponents are the
# Thompson values divided by 10 (so the algorithm form is 10^(c·Zdr)).
const RAIN_BAND_COEFFS = Dict(
    "S" => (kdp = (a = 59.5202, b = 0.7451),
            kdp_zdr = (a = 96.5726, b = 0.9315, c = -2.1140 / 10),
            z_zdr   = (a = 0.0085, b = 0.9237, c = -5.2389 / 10)),
    "C" => (kdp = (a = 34.5703, b = 0.7331),
            kdp_zdr = (a = 45.6976, b = 0.8763, c = -1.6718 / 10),
            z_zdr   = (a = 0.0086, b = 0.9088, c = -4.2059 / 10)),
    "X" => (kdp = (a = 21.9729, b = 0.7221),
            kdp_zdr = (a = 28.1289, b = 0.9194, c = -1.6876 / 10),
            z_zdr   = (a = 0.0085, b = 0.9294, c = -4.4580 / 10)),
)

_rain_band_coeffs(band::AbstractString) = haskey(RAIN_BAND_COEFFS, band) ?
    RAIN_BAND_COEFFS[band] :
    throw(ArgumentError("Unknown radar band \"$band\"; expected \"S\", \"C\", or \"X\""))

# numpy-compatible power: NaN for a negative base with a non-integer exponent.
@inline function _safe_pow(x::Real, p::Real)
    if x < 0 && !isinteger(p)
        return oftype(float(x), NaN)
    end
    return float(x)^p
end

"""
    rain_linearize(dbz)

Convert reflectivity from log (dBZ) to linear Z (mm⁶ m⁻³): `10^(dBZ/10)`.
"""
@inline rain_linearize(dbz) = 10.0^(dbz / 10.0)

# ---------------------------------------------------------------------------
# Component rain-rate relations (scalar kernels + broadcasting public methods)
# ---------------------------------------------------------------------------

_rain_zr(dz, a, b) = (rain_linearize(dz) / a)^(1.0 / b)
_rain_kdp(kdp, a, b) = a * _safe_pow(kdp, b)
_rain_kdp_zdr(kdp, zdr, a, b, c) = a * _safe_pow(kdp, b) * 10.0^(c * zdr)
_rain_z_zdr(dz, zdr, a, b, c) = a * rain_linearize(dz)^b * 10.0^(c * zdr)

"""
    calc_rain_zr(dz; a=216.0, b=1.39)

R(Z) rain rate (mm h⁻¹) from reflectivity `dz` (dBZ): `R = (10^(dBZ/10)/a)^(1/b)`.
Use the convective ([`RAIN_RZ_CONV`](@ref)) or stratiform ([`RAIN_RZ_STRAT`](@ref))
coefficients for those regimes. Accepts a scalar or array.
"""
calc_rain_zr(dz; a = RAIN_RZ_ALL.a, b = RAIN_RZ_ALL.b) = _rain_zr.(dz, a, b)

"""
    calc_rain_kdp(kdp; a, b)

R(Kdp) rain rate (mm h⁻¹): `R = a·Kdp^b`. Pass band coefficients from
[`RAIN_BAND_COEFFS`](@ref) or use [`calc_rain_kdp`](@ref)`(kdp, band)`.
Accepts a scalar or array.
"""
calc_rain_kdp(kdp; a, b) = _rain_kdp.(kdp, a, b)

"""
    calc_rain_kdp_zdr(kdp, zdr; a, b, c)

R(Kdp,Zdr) rain rate (mm h⁻¹): `R = a·Kdp^b·10^(c·Zdr)`. Accepts scalars or arrays.
"""
calc_rain_kdp_zdr(kdp, zdr; a, b, c) = _rain_kdp_zdr.(kdp, zdr, a, b, c)

"""
    calc_rain_z_zdr(dz, zdr; a, b, c)

R(Z,Zdr) rain rate (mm h⁻¹): `R = a·(10^(dBZ/10))^b·10^(c·Zdr)`. Accepts scalars or arrays.
"""
calc_rain_z_zdr(dz, zdr; a, b, c) = _rain_z_zdr.(dz, zdr, a, b, c)

# Band convenience methods so a component can be called with just data + band.
calc_rain_kdp(kdp, band::AbstractString) =
    (c = _rain_band_coeffs(band).kdp; calc_rain_kdp(kdp; a = c.a, b = c.b))
calc_rain_kdp_zdr(kdp, zdr, band::AbstractString) =
    (c = _rain_band_coeffs(band).kdp_zdr; calc_rain_kdp_zdr(kdp, zdr; a = c.a, b = c.b, c = c.c))
calc_rain_z_zdr(dz, zdr, band::AbstractString) =
    (c = _rain_band_coeffs(band).z_zdr; calc_rain_z_zdr(dz, zdr; a = c.a, b = c.b, c = c.c))

# ---------------------------------------------------------------------------
# Blended tropical rain rate
# ---------------------------------------------------------------------------

# Method codes returned by calc_blended_rain_tropical.
#   1 = R(Kdp, Zdr)   2 = R(Kdp)    3 = R(Z, Zdr)
#   4 = R(Z_all)      5 = R(Z_conv) 6 = R(Z_strat)   -1 = invalid / excluded
const RAIN_METHOD_KDP_ZDR = 1
const RAIN_METHOD_KDP     = 2
const RAIN_METHOD_Z_ZDR   = 3
const RAIN_METHOD_Z_ALL   = 4
const RAIN_METHOD_Z_CONV  = 5
const RAIN_METHOD_Z_STRAT = 6
const RAIN_METHOD_INVALID = -1

@inline _rain_valid(x, fill_value, undetect) =
    isfinite(x) && x != fill_value && x != undetect

"""
    calc_blended_rain_tropical(; dz, zdr, kdp, fhc=nothing, cs=nothing, band="S",
        thresh_dz=38.0, thresh_zdr=0.25, thresh_kdp=0.3,
        correct_ice_method=true, fill_value=-32768.0, undetect=-9999.0)

Blended tropical/oceanic rain rate (Thompson et al. 2016). Julia port of
CSU_RadarTools `calc_blended_rain_tropical`.

Per grid cell, selects the best rain-rate estimate via the Thompson flowchart from
reflectivity `dz` (dBZ), differential reflectivity `zdr` (dB) and specific
differential phase `kdp` (deg/km). Returns `(rain, method)` arrays (mm h⁻¹ and
method codes; see `RAIN_METHOD_*`).

Selection priority:
  1. `kdp ≥ thresh_kdp ∧ dz ≥ thresh_dz ∧ zdr ≥ thresh_zdr` → R(Kdp,Zdr)
  2. `kdp ≥ thresh_kdp ∧ dz ≥ thresh_dz ∧ zdr < thresh_zdr` → R(Kdp)
  3. `zdr ≥ thresh_zdr ∧ ¬(kdp ≥ thresh_kdp ∧ dz ≥ thresh_dz)` → R(Z,Zdr)
  4. otherwise → R(Z_all), or R(Z_conv)/R(Z_strat) when a convective/stratiform
     map `cs` is supplied (`cs`: 1=stratiform, 2=convective, 3=mixed, 0=unknown).

Optional `fhc` (hydrometeor class array, e.g. from [`csu_fhc_summer`](@ref)) masks
ice: classes `2 < fhc < 10` get rain 0, and hail (`fhc == 9`) with the Kdp/Z
condition met falls back to R(Kdp).

`correct_ice_method` (default `true`) labels the method code `2` only on hail
cells. The Python original sets it on *all* ice cells (an apparent bug at
csu_blended_rain_tropical.py:221); set `false` to replicate that exactly. Rain
*values* are identical either way.
"""
function calc_blended_rain_tropical(; dz, zdr, kdp, fhc = nothing, cs = nothing,
        band::AbstractString = "S", thresh_dz::Real = 38.0,
        thresh_zdr::Real = 0.25, thresh_kdp::Real = 0.3,
        correct_ice_method::Bool = true, fill_value::Real = -32768.0,
        undetect::Real = -9999.0)

    coeffs = _rain_band_coeffs(band)
    ck = coeffs.kdp
    ckz = coeffs.kdp_zdr
    czz = coeffs.z_zdr

    dz_arr = dz isa AbstractArray ? dz : fill(Float64(dz), 1)
    shp = size(dz_arr)
    n = length(dz_arr)

    _at(::Nothing, i) = nothing
    _at(v::AbstractArray, i) = v[i]
    _at(v, i) = v

    rain = zeros(Float64, n)
    meth = fill(RAIN_METHOD_INVALID, n)

    @inbounds for i in 1:n
        d = dz_arr[i]
        zd = _at(zdr, i)
        kd = _at(kdp, i)

        # Invalid reflectivity → leave as 0 rain / invalid method.
        if !_rain_valid(d, fill_value, undetect)
            continue
        end

        cond_kdp = kd !== nothing && _rain_valid(kd, fill_value, undetect) && kd >= thresh_kdp
        cond_dbz = d >= thresh_dz
        cond_kdpz = cond_kdp && cond_dbz
        cond_zdr = zd !== nothing && _rain_valid(zd, fill_value, undetect) && zd >= thresh_zdr

        if cond_kdpz && cond_zdr
            meth[i] = RAIN_METHOD_KDP_ZDR
            rain[i] = _rain_kdp_zdr(kd, zd, ckz.a, ckz.b, ckz.c)
        elseif cond_kdpz && !cond_zdr
            meth[i] = RAIN_METHOD_KDP
            rain[i] = _rain_kdp(kd, ck.a, ck.b)
        elseif cond_zdr && !cond_kdpz
            meth[i] = RAIN_METHOD_Z_ZDR
            rain[i] = _rain_z_zdr(d, zd, czz.a, czz.b, czz.c)
        else  # cond_meth_4: ¬cond_kdpz ∧ ¬cond_zdr
            csv = _at(cs, i)
            if cs === nothing
                meth[i] = RAIN_METHOD_Z_ALL
                rain[i] = _rain_zr(d, RAIN_RZ_ALL.a, RAIN_RZ_ALL.b)
            elseif csv == 2
                meth[i] = RAIN_METHOD_Z_CONV
                rain[i] = _rain_zr(d, RAIN_RZ_CONV.a, RAIN_RZ_CONV.b)
            elseif csv == 1
                meth[i] = RAIN_METHOD_Z_STRAT
                rain[i] = _rain_zr(d, RAIN_RZ_STRAT.a, RAIN_RZ_STRAT.b)
            elseif csv == 3
                meth[i] = RAIN_METHOD_Z_ALL
                rain[i] = _rain_zr(d, RAIN_RZ_ALL.a, RAIN_RZ_ALL.b)
            end
            # csv == 0 / other: stays rain 0, method -1 (matches Python init).
        end

        # Ice / hail handling from the hydrometeor class.
        if fhc !== nothing
            fv = _at(fhc, i)
            if fv !== nothing && fv > 2 && fv < 10
                cond_hail = fv == 9 && cond_kdpz
                if correct_ice_method
                    rain[i] = 0.0
                    meth[i] = RAIN_METHOD_INVALID
                    if cond_hail
                        rain[i] = _rain_kdp(kd, ck.a, ck.b)
                        meth[i] = RAIN_METHOD_KDP
                    end
                else
                    # Replicate Python exactly: rain 0 (hail→R(Kdp)), method 2 for
                    # all ice cells (csu_blended_rain_tropical.py:217-221).
                    rain[i] = cond_hail ? _rain_kdp(kd, ck.a, ck.b) : 0.0
                    meth[i] = RAIN_METHOD_KDP
                end
            end
        end

        # Out-of-range reflectivity → no rain.
        if d < -10
            rain[i] = 0.0
            meth[i] = RAIN_METHOD_INVALID
        end
    end

    if dz isa AbstractArray
        return reshape(rain, shp), reshape(meth, shp)
    else
        return rain[1], meth[1]
    end
end
