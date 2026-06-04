# Multi-Doppler wind synthesis — stage 1.
#
# A gridpoint-independent, overdetermined, weighted least-squares dual-Doppler
# horizontal wind retrieval `(u, v)`, streamed through an accumulator that
# mirrors the `GridAccumulator` model. Each contributing gate (from any
# radar/sweep) adds a rank-1 update to a per-gridpoint normal system
# `AᵀWA x = AᵀWb`; no per-radar pre-averaging and no retention of individual
# gate rows. Multi-radar synthesis is simply continuing to grid more sweeps into
# the same accumulator. A preserved per-component uncertainty (the equal-variance
# "sandwich" covariance) drives a non-destructive quality flag.
#
# Reference: CEDRIC appendix F (Miller & Anderson, 1991), §1 "Cartesian
# components of motion from radial velocities". The observation equation, with
# the vertical term dropped (W neglected; the dual-Doppler approximation), is
#
#     v_r ≈ u·a + v·b,   a = sin(az)·cos(el),   b = cos(az)·cos(el)
#
# where (az, el) are the *effective* line-of-sight angles from the gate's radar
# origin to the grid point — exactly the `gridpt_az`/`gridpt_el` the shared
# `_gate_grid_geometry` kernel already computes for the scalar gridding path.
#
# The whole accumulation is performed in the **Cartesian** basis and is
# frame-agnostic: the output frame is a per-gridpoint orthonormal rotation `R`
# that factors out of every accumulated sum, so it is applied only at finalize.
# Stage 1 ships only the identity `CartesianFrame`.

# v1: initial wind accumulator schema. Bumps on any layout change.
const WIND_ACCUMULATOR_SCHEMA_VERSION = 1

# ── Packed symmetric storage ─────────────────────────────────────────────────
# The per-gridpoint `AᵀWA` and `M2` are symmetric N×N matrices stored packed as
# the row-major upper triangle, length `N(N+1)/2`. For N=2 the order is
# `[aa, ab, bb]`; for N=3 `[aa, ab, ac, bb, bc, cc]`. `_packed_index` maps an
# (i, j) entry (either order) to its slot so the assembly/solve are general-N.

@inline _n_packed(N::Int) = N * (N + 1) ÷ 2

@inline function _packed_index(i::Int, j::Int, N::Int)
    a, b = i <= j ? (i, j) : (j, i)
    before = (a - 1) * N - ((a - 1) * (a - 2)) ÷ 2   # entries in rows < a
    return before + (b - a) + 1
end

# ── WindGridAccumulator ──────────────────────────────────────────────────────

"""
    WindGridAccumulator

Streamed per-gridpoint dual-Doppler normal system. Every plane is accumulated in
the **Cartesian** basis; the struct carries **no frame** (the output frame is
supplied to [`finalize_wind`](@ref), so one accumulator can be finalized in
several frames without re-gridding).

# Fields
- `grid_spec::GridSpec`: grid geometry (stage 1: `:volume_3d` / `:latlon_3d`).
- `velocity_field::String`: the `:velocity`-tagged field projected per gate.
- `n_unknowns::Int`: `2` for stage 1 (`u, v`); `3` reserved (`u, v, w`, deferred).
- `AtWA::Array{Float64}`: `(n_packed, trailing…)` packed `Σ w·ggᵀ`.
- `AtWb::Array{Float64}`: `(n_unknowns, trailing…)` `Σ w·v_r·g`.
- `M2::Array{Float64}`: `(n_packed, trailing…)` packed `Σ w²·ggᵀ` (sandwich cov).
- `weight_total::Array{Float64}`: `(trailing…)` `Σ w`.
- `gate_count::Array{Int32}`: `(trailing…)` contributing-gate count — diagnostic
  only (**not** a QC threshold; used for the intrinsic "did any data land here"
  check and as an informational output field). No radar identity is tracked.
- `sweeps::Vector{SweepProvenance}`, `schema_version::Int`,
  `fill_value`/`undetect`: as in [`GridAccumulator`](@ref).

The trailing dims come from [`wind_accumulator_dims`](@ref) (stage 1: `(nz, ny,
nx)`). **This struct is the stage-2 handoff artifact:** its
`AtWA`/`AtWb`/`M2`/`weight_total`/counts are exactly what a variational stage
needs to recover masked baseline points. [`finalize_wind`](@ref) produces the
single-stage *product*; the accumulator is the richer *input* for stage 2.
"""
Base.@kwdef struct WindGridAccumulator
    grid_spec::GridSpec
    velocity_field::String
    n_unknowns::Int
    AtWA::Array{Float64}
    AtWb::Array{Float64}
    M2::Array{Float64}
    weight_total::Array{Float64}
    gate_count::Array{Int32}
    sweeps::Vector{SweepProvenance}
    schema_version::Int
    fill_value::Float64 = -32768.0
    undetect::Float64   =  -9999.0
end

"""
    wind_accumulator_dims(grid_spec) -> NTuple

Trailing array dimensions for the wind accumulator planes (without the leading
packed/component axis). Stage 1 supports the 3D shapes only: `:volume_3d` and
`:latlon_3d` → `(nz, ny, nx)`.
"""
function wind_accumulator_dims(g::GridSpec)
    if g.shape === :volume_3d || g.shape === :latlon_3d
        return (length(g.z_axis), length(g.y_axis), length(g.x_axis))
    else
        throw(ArgumentError(
            "wind_accumulator_dims: stage-1 wind synthesis supports :volume_3d " *
            "and :latlon_3d only, got $(g.shape)"))
    end
end

"""
    WindGridAccumulator(grid_spec, velocity_field; n_unknowns=2,
                        fill_value=-32768.0, undetect=-9999.0)

Allocate an empty wind accumulator on `grid_spec` for the named velocity field.
`n_unknowns` is `2` (stage 1) or `3` (reserved; storage is allocated but
[`finalize_wind`](@ref) does not yet solve it). `fill_value`/`undetect` are the
true-missing / no-coverage output sentinels.
"""
function WindGridAccumulator(grid_spec::GridSpec, velocity_field::AbstractString;
                             n_unknowns::Int = 2,
                             fill_value::Float64 = -32768.0,
                             undetect::Float64 = -9999.0)
    (n_unknowns == 2 || n_unknowns == 3) || throw(ArgumentError(
        "WindGridAccumulator: n_unknowns must be 2 (stage 1) or 3 (reserved), " *
        "got $n_unknowns"))
    trailing = wind_accumulator_dims(grid_spec)
    np = _n_packed(n_unknowns)
    return WindGridAccumulator(
        grid_spec      = grid_spec,
        velocity_field = String(velocity_field),
        n_unknowns     = n_unknowns,
        AtWA           = zeros(Float64, np, trailing...),
        AtWb           = zeros(Float64, n_unknowns, trailing...),
        M2             = zeros(Float64, np, trailing...),
        weight_total   = zeros(Float64, trailing...),
        gate_count     = zeros(Int32, trailing...),
        sweeps         = SweepProvenance[],
        schema_version = WIND_ACCUMULATOR_SCHEMA_VERSION,
        fill_value     = fill_value,
        undetect       = undetect,
    )
end

"""
    WindGridAccumulator(grid_spec, p::DaishoParameters; n_unknowns=2)

Allocate a wind accumulator from a `DaishoParameters`. The velocity field is the
one carrying the `:velocity` tag (via [`field_with_tag`](@ref)); the sentinels
come from `[io]`.
"""
function WindGridAccumulator(grid_spec::GridSpec, p::DaishoParameters;
                             n_unknowns::Int = 2)
    vel = field_with_tag(p, :velocity; for_op="wind synthesis (WindGridAccumulator)")
    return WindGridAccumulator(grid_spec, vel; n_unknowns = n_unknowns,
        fill_value = p.io.fill_value, undetect = p.io.undetect)
end

# ── Output frame ─────────────────────────────────────────────────────────────

"""
    SynthesisFrame

Abstraction for the orthonormal *output frame* the wind is expressed in. Because
the rotation `R = rotation_at(frame, x, y, z)` depends only on the grid point
(not the gate), it factors out of every accumulated sum, so accumulation stays
Cartesian and the frame is applied only at finalize. A concrete frame provides:

- [`component_names`](@ref)`(frame) -> (String, String)`
- [`rotation_at`](@ref)`(frame, x, y, z) -> R` (N×N orthonormal)

Stage 1 ships only [`CartesianFrame`](@ref). Plane/polar/coplane frames are pure
additions later (no accumulator change).
"""
abstract type SynthesisFrame end

"""
    CartesianFrame()

The identity output frame: components are `(U, V)` (zonal, meridional) and the
rotation is the identity, so the finalize-time view equals the accumulated
Cartesian solution.
"""
struct CartesianFrame <: SynthesisFrame end

"""
    component_names(frame) -> (String, String)

The two in-frame component names. `CartesianFrame` ⇒ `("U", "V")`.
"""
component_names(::CartesianFrame) = ("U", "V")

"""
    rotation_at(frame, x, y, z) -> Matrix{Float64}

The N×N orthonormal rotation taking the Cartesian solution into the output
frame at grid point `(x, y, z)`. `CartesianFrame` ⇒ the 2×2 identity.
"""
rotation_at(::CartesianFrame, x::Real, y::Real, z::Real) = [1.0 0.0; 0.0 1.0]

# ── Synthesis output ─────────────────────────────────────────────────────────

"""
    SynthesisOutput

Single-stage retrieval product returned by [`finalize_wind`](@ref). The grids
are named frame-relatively: for `CartesianFrame`, `comp1`/`comp2` carry `U`/`V`
and `sigma1`/`sigma2` carry σ_u/σ_v. `n_gates` is informational (not a flag).

`quality_flag` (Int8) bit encoding — non-destructive: where the system is
solvable, `comp1, comp2, sigma1, sigma2` are always written regardless of bits:
- `0` = solvable and both in-frame σ within thresholds
- bit `1` (`0x01`) = σ₁ above `max_sigma[1]`
- bit `2` (`0x02`) = σ₂ above `max_sigma[2]`
- bit `3` (`0x04`) = **singular**: ≥1 gate but `D ≤ 0` / σ non-finite → components `fill_value`
- bit `4` (`0x08`) = **no data**: no weighted gate reached the point → components `fill_value`
"""
Base.@kwdef struct SynthesisOutput
    grid_spec::GridSpec
    component_names::Tuple{String,String}
    comp1::Array{Float64}
    comp2::Array{Float64}
    sigma1::Array{Float64}
    sigma2::Array{Float64}
    det::Array{Float64}
    n_gates::Array{Int32}
    quality_flag::Array{Int8}
    fill_value::Float64
    undetect::Float64
end

# Quality-flag bit masks (see SynthesisOutput).
const QFLAG_SIGMA1   = Int8(0x01)
const QFLAG_SIGMA2   = Int8(0x02)
const QFLAG_SINGULAR = Int8(0x04)
const QFLAG_NODATA   = Int8(0x08)

# ── Streaming accumulation ───────────────────────────────────────────────────

# Fill the per-gate Cartesian coefficient vector `g` from effective angles.
# N=2: g = [sin(az)·cos(el), cos(az)·cos(el)] (coefficients of u, v).
# N=3 (deferred): restores g[3] = sin(el) (coefficient of w).
@inline function _wind_coeffs!(g::AbstractVector{Float64}, az::Float64,
                               el::Float64, N::Int)
    cose = cos(el)
    g[1] = sin(az) * cose
    g[2] = cos(az) * cose
    N >= 3 && (g[3] = sin(el))
    return g
end

"""
    grid_sweep_wind!(acc::WindGridAccumulator, sweep::SweepGroup, p; ref_latitude,
                     ref_longitude, ref_altitude, source_file="",
                     instrument_name="", scan_name="") -> acc

Add one sweep's dual-Doppler contribution to `acc`. For every gate whose
`:velocity` value is present and whose interpolation weight is positive, the
effective line-of-sight angles `(az, el)` (from the shared
[`_gate_grid_geometry`](@ref) kernel) form the Cartesian coefficient vector
`g = [sin az·cos el, cos az·cos el]`, and the gate applies the rank-1 updates

    AtWA += w·ggᵀ,   AtWb += w·v_r·g,   M2 += w²·ggᵀ

to the per-gridpoint normal system, plus `weight_total += w` and
`gate_count += 1`. There is **no `radar_index`** — nothing about the synthesis
depends on which radar a gate came from. A gate's own velocity validity gates
its participation, independent of the `:define_detection`/`:define_scanned`
roles; the `field_folds` merge-guard does not apply (velocity is projected,
never scalar-averaged).

A [`SweepProvenance`](@ref) entry is appended to `acc.sweeps`.
"""
function grid_sweep_wind!(acc::WindGridAccumulator, sweep::SweepGroup,
                          p::DaishoParameters;
                          ref_latitude::Float64,
                          ref_longitude::Float64,
                          ref_altitude::Float64,
                          source_file::AbstractString = "",
                          instrument_name::AbstractString = "",
                          scan_name::AbstractString = "")
    shape = acc.grid_spec.shape
    (shape === :volume_3d || shape === :latlon_3d) || throw(ArgumentError(
        "grid_sweep_wind!: stage-1 wind synthesis supports :volume_3d and " *
        ":latlon_3d only, got $shape"))
    _grid_sweep_wind_3d!(acc, sweep, p, ref_latitude, ref_longitude, ref_altitude)
    push!(acc.sweeps, SweepProvenance(
        instrument_name = String(instrument_name),
        scan_name = String(scan_name),
        sweep_number = sweep.sweep_number,
        sweep_mode = sweep.sweep_mode,
        fixed_angle = sweep.fixed_angle,
        time_start = isempty(sweep.time) ? nothing : sweep.time[1],
        time_end   = isempty(sweep.time) ? nothing : sweep.time[end],
        source_file = String(source_file),
        ref_latitude = ref_latitude,
        ref_longitude = ref_longitude,
        ref_altitude = ref_altitude,
    ))
    return acc
end

"""
    grid_sweep_wind!(acc, volume::Volume, sweep_index::Int, p; source_file="") -> acc

Volume convenience overload. Resolves the per-sweep reference position from the
sweep's georeference when present (mobile), else from the volume's stationary
`latitude`/`longitude`/`altitude`. Mirrors [`grid_sweep!`](@ref) but takes no
`heading`/`radar_index`.
"""
function grid_sweep_wind!(acc::WindGridAccumulator, volume::Volume,
                          sweep_index::Int, p::DaishoParameters;
                          source_file::AbstractString = "")
    sweep = volume.sweeps[sweep_index]
    if sweep.georeference !== nothing && !isempty(sweep.georeference.latitude)
        ref_lat = sweep.georeference.latitude[1]
        ref_lon = sweep.georeference.longitude[1]
        ref_alt = sweep.georeference.altitude[1]
    else
        ref_lat = volume.latitude
        ref_lon = volume.longitude
        ref_alt = volume.altitude
    end
    return grid_sweep_wind!(acc, sweep, p;
        ref_latitude = Float64(ref_lat),
        ref_longitude = Float64(ref_lon),
        ref_altitude = Float64(ref_alt),
        source_file = source_file,
        instrument_name = volume.instrument_name,
        scan_name = volume.scan_name)
end

# 3D worker. Mirrors `_grid_sweep_3d!`'s geometry setup and ROI derivation, but
# projects the velocity into the per-gridpoint Cartesian normal system instead
# of accumulating scalar weighted sums.
function _grid_sweep_wind_3d!(acc::WindGridAccumulator, sweep::SweepGroup,
                              p::DaishoParameters,
                              ref_latitude::Float64, ref_longitude::Float64,
                              ref_altitude::Float64)
    g  = acc.grid_spec
    gd = p.gridding
    velkey = acc.velocity_field
    N = acc.n_unknowns

    TM = CoordRefSystems.shift(TransverseMercator{1.0, g.reference_latitude, WGS84Latest},
                                lonₒ = g.reference_longitude)
    grid_origin, radar_zyx, beams, n_gates_s, _ =
        _sweep_zyx_and_beams(sweep, ref_latitude, ref_longitude, ref_altitude, TM)
    balltree = _sweep_balltree_yx(radar_zyx, beams)
    gridpoints = _materialize_gridpoints_3d(g, TM, grid_origin)

    nx = length(g.x_axis)
    ny = length(g.y_axis)
    nz = length(g.z_axis)

    # ROI mirrors `_grid_sweep_3d!`.
    horizontal_roi = if g.shape === :latlon_3d
        latrad = g.reference_latitude * pi / 180.0
        fac_lat = 111.13209 - 0.56605 * cos(2.0 * latrad)
        fac_lon = 111.41513 * cos(latrad)
        deg_km = sqrt(fac_lat^2 + fac_lon^2)
        degincr = ny >= 2 ? (g.lat_axis[2] - g.lat_axis[1]) :
                  (nx >= 2 ? (g.lon_axis[2] - g.lon_axis[1]) : 0.01)
        deg_km * 1000.0 * degincr * 0.75
    else
        xincr = nx >= 2 ? (g.x_axis[2] - g.x_axis[1]) : 0.0
        xincr * 0.75
    end
    zincr = nz >= 2 ? (g.z_axis[2] - g.z_axis[1]) : 0.0
    vertical_roi = zincr * 0.75

    beam_inflation  = gd.beam_inflation
    power_threshold = gd.power_threshold

    Threads.@threads for ii in CartesianIndices((ny, nx))
        j_y, i_x = ii.I
        yx_point = [gridpoints[1, j_y, i_x, 2], gridpoints[1, j_y, i_x, 3]]

        eff_h_roi = horizontal_roi
        eff_v_roi = vertical_roi
        if beam_inflation > 0.0
            origin_dist = euclidean(yx_point, [0.0, 0.0])
            eff_h_roi = max(beam_inflation * origin_dist, horizontal_roi)
            eff_v_roi = max(beam_inflation * origin_dist, vertical_roi)
        end
        gates = inrange(balltree, yx_point, eff_h_roi)
        isempty(gates) && continue

        gcoef = Vector{Float64}(undef, N)   # per-column scratch for `g`
        for k_z in 1:nz
            grid_z = gridpoints[k_z, j_y, i_x, 1]
            for g_flat in gates
                # Vertical-range filter: only consider gates near this z.
                abs(beams[g_flat, 4] - grid_z) > eff_v_roi && continue

                ray  = _ray_of(g_flat, n_gates_s)
                gate = _gate_in_ray(g_flat, n_gates_s)
                vr = _gate_value(sweep, velkey, ray, gate)
                ismissing(vr) && continue

                gridpt_az, gridpt_el, w = _gate_grid_geometry(grid_z, yx_point,
                    radar_zyx, beams, g_flat,
                    horizontal_roi, vertical_roi, power_threshold)
                w > 0.0 || continue

                _wind_coeffs!(gcoef, gridpt_az, gridpt_el, N)
                @inbounds begin
                    for a in 1:N, b in a:N
                        pk = _packed_index(a, b, N)
                        gg = gcoef[a] * gcoef[b]
                        acc.AtWA[pk, k_z, j_y, i_x] += w * gg
                        acc.M2[pk,  k_z, j_y, i_x] += w * w * gg
                    end
                    for a in 1:N
                        acc.AtWb[a, k_z, j_y, i_x] += w * vr * gcoef[a]
                    end
                    acc.weight_total[k_z, j_y, i_x] += w
                    acc.gate_count[k_z, j_y, i_x]   += Int32(1)
                end
            end
        end
    end
    return acc
end
