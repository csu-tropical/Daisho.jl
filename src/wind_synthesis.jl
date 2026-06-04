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
`_gate_grid_geometry` kernel) form the Cartesian coefficient vector
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

# ── Finalize: solve, sandwich error, frame rotation, non-destructive QC ───────

"""
    finalize_wind(acc::WindGridAccumulator, p::DaishoParameters;
                  frame::SynthesisFrame = CartesianFrame()) -> SynthesisOutput

Solve the per-gridpoint dual-Doppler normal system and produce the single-stage
retrieval product. For each grid point (§2.3–2.4 of the stage-1 plan):

1. Point estimate `[u; v] = (AᵀWA)⁻¹ · AᵀWb` from the packed 2×2 `AᵀWA`, with
   `D = S_aa·S_bb − S_ab²` the determinant (→0 along the baseline).
2. Equal-variance "sandwich" covariance `cov_cart = σ²_vr · Ci · M2 · Ci`
   (`σ²_vr = velocity_variance`; at equal gate weights `M2 = AᵀWA` and this
   collapses to `σ²_vr·Ci`, the CEDRIC Eq-13 `Σg²` form).
3. Rotation into the active output frame: `[c1; c2] = R·[u; v]`,
   `cov_frame = R·cov_cart·Rᵀ`, with `R = rotation_at(frame, x, y, z)`; the
   in-frame σ are `sqrt` of the covariance diagonal.

Masking is **non-destructive**: where the system is solvable, `comp1, comp2,
sigma1, sigma2` are always written, and `quality_flag` records, independently
per component, which `max_sigma` threshold was exceeded. Two intrinsic
(non-tunable) states blank the components to `fill_value`: **no data** (no
weighted gate reached the point) and **singular** (`D ≤ 0` / non-finite cov,
e.g. a single look direction or a baseline point). Stage 1 solves `n_unknowns ==
2` only.
"""
function finalize_wind(acc::WindGridAccumulator, p::DaishoParameters;
                       frame::SynthesisFrame = CartesianFrame())
    acc.n_unknowns == 2 || throw(ArgumentError(
        "finalize_wind: stage-1 solve supports n_unknowns=2 only; got " *
        "$(acc.n_unknowns) (the 3-unknown path is scaffolded but not shipped)."))
    g = acc.grid_spec
    σ2 = p.synthesis.velocity_variance
    length(p.synthesis.max_sigma) >= 2 || throw(ArgumentError(
        "finalize_wind: [synthesis] max_sigma needs ≥2 components for the " *
        "2-unknown solve, got $(length(p.synthesis.max_sigma))."))
    max_s1 = p.synthesis.max_sigma[1]
    max_s2 = p.synthesis.max_sigma[2]
    fv = acc.fill_value

    nz, ny, nx = wind_accumulator_dims(g)
    comp1  = fill(fv, nz, ny, nx)
    comp2  = fill(fv, nz, ny, nx)
    sigma1 = fill(fv, nz, ny, nx)
    sigma2 = fill(fv, nz, ny, nx)
    det    = zeros(Float64, nz, ny, nx)
    qflag  = zeros(Int8, nz, ny, nx)
    ngat   = copy(acc.gate_count)

    xax, yax, zax = g.x_axis, g.y_axis, g.z_axis

    @inbounds for i in 1:nx, j in 1:ny, k in 1:nz
        if acc.gate_count[k, j, i] == Int32(0) || acc.weight_total[k, j, i] <= 0.0
            qflag[k, j, i] = QFLAG_NODATA          # no weighted gate reached here
            continue
        end
        Saa = acc.AtWA[1, k, j, i]; Sab = acc.AtWA[2, k, j, i]; Sbb = acc.AtWA[3, k, j, i]
        Sav = acc.AtWb[1, k, j, i]; Sbv = acc.AtWb[2, k, j, i]
        D = Saa * Sbb - Sab * Sab
        det[k, j, i] = D
        # Singular guard. A single look direction makes AᵀWA rank-1, so D = S_aa·
        # S_bb − S_ab² collapses to numerical roundoff (~eps·S_aa·S_bb), not a
        # clean 0; left unchecked it produces a falsely-confident (0,0) with
        # σ=0. The scale-relative test catches that intrinsic rank deficiency,
        # while a *physically* near-baseline pair (small but real crossing
        # angle) keeps a positive D ≫ the floor and survives — its instability
        # surfaces as a large σ caught by the per-component thresholds, not here.
        if !isfinite(D) || D <= 1e-12 * Saa * Sbb
            qflag[k, j, i] = QFLAG_SINGULAR        # degenerate geometry / baseline
            continue
        end
        invD = 1.0 / D
        # Ci = (AᵀWA)⁻¹ (symmetric): [c11 c12; c12 c22].
        c11 =  Sbb * invD; c12 = -Sab * invD; c22 = Saa * invD
        u = c11 * Sav + c12 * Sbv
        v = c12 * Sav + c22 * Sbv
        # Sandwich covariance cov_cart = σ²·Ci·M2·Ci.
        Taa = acc.M2[1, k, j, i]; Tab = acc.M2[2, k, j, i]; Tbb = acc.M2[3, k, j, i]
        p11 = c11 * Taa + c12 * Tab; p12 = c11 * Tab + c12 * Tbb
        p21 = c12 * Taa + c22 * Tab; p22 = c12 * Tab + c22 * Tbb
        cov11 = σ2 * (p11 * c11 + p12 * c12)
        cov12 = σ2 * (p11 * c12 + p12 * c22)
        cov21 = σ2 * (p21 * c11 + p22 * c12)
        cov22 = σ2 * (p21 * c12 + p22 * c22)
        # Rotate into the output frame: x_frame = R·x, cov_frame = R·cov·Rᵀ.
        R = rotation_at(frame, xax[i], yax[j], zax[k])
        r11 = R[1, 1]; r12 = R[1, 2]; r21 = R[2, 1]; r22 = R[2, 2]
        c1 = r11 * u + r12 * v
        c2 = r21 * u + r22 * v
        rc11 = r11 * cov11 + r12 * cov21; rc12 = r11 * cov12 + r12 * cov22
        rc21 = r21 * cov11 + r22 * cov21; rc22 = r21 * cov12 + r22 * cov22
        cf11 = rc11 * r11 + rc12 * r12
        cf22 = rc21 * r21 + rc22 * r22
        if !isfinite(cf11) || !isfinite(cf22) || cf11 < 0.0 || cf22 < 0.0
            qflag[k, j, i] = QFLAG_SINGULAR
            continue
        end
        s1 = sqrt(cf11); s2 = sqrt(cf22)
        comp1[k, j, i] = c1; comp2[k, j, i] = c2
        sigma1[k, j, i] = s1; sigma2[k, j, i] = s2
        f = Int8(0)
        s1 > max_s1 && (f |= QFLAG_SIGMA1)          # in-frame, independent
        s2 > max_s2 && (f |= QFLAG_SIGMA2)
        qflag[k, j, i] = f
    end

    return SynthesisOutput(
        grid_spec = g,
        component_names = component_names(frame),
        comp1 = comp1, comp2 = comp2, sigma1 = sigma1, sigma2 = sigma2,
        det = det, n_gates = ngat, quality_flag = qflag,
        fill_value = fv, undetect = acc.undetect)
end

# ── Persistence + multi-file merge ───────────────────────────────────────────

"""
    save_wind_accumulator(path, accum::WindGridAccumulator) -> path

Persist a `WindGridAccumulator` to JLD2. Reload with [`load_wind_accumulator`](@ref).
Mirrors [`save_accumulator`](@ref).
"""
function save_wind_accumulator(path::AbstractString, accum::WindGridAccumulator)
    JLD2.jldopen(path, "w") do f
        f["wind_accumulator"] = accum
    end
    return path
end

"""
    load_wind_accumulator(path) -> WindGridAccumulator

Read a `WindGridAccumulator` saved by [`save_wind_accumulator`](@ref). Raises if
the on-disk `schema_version` does not match `WIND_ACCUMULATOR_SCHEMA_VERSION`.
"""
function load_wind_accumulator(path::AbstractString)
    accum = JLD2.jldopen(path, "r") do f
        f["wind_accumulator"]
    end
    accum.schema_version == WIND_ACCUMULATOR_SCHEMA_VERSION || throw(ArgumentError(
        "load_wind_accumulator: schema version $(accum.schema_version) on " *
        "$(path); expected $(WIND_ACCUMULATOR_SCHEMA_VERSION)"))
    return accum
end

"""
    merge_wind_accumulators!(dst, src) -> dst

Combine `src` into `dst` in place. Because the accumulation is **linear** —
every plane is a simple sum of per-gate rank-1 updates — the merge is an
elementwise add of `AtWA`/`AtWb`/`M2`/`weight_total`/`gate_count` and a concat
of `sweeps`. This is what makes the streaming model valid across files and
radars: gridding sweeps A and B into one accumulator equals gridding them
separately and merging. Strict compatibility is required (identical `grid_spec`,
`velocity_field`, `n_unknowns`, and `[io]` sentinels); no silent coercion.
"""
function merge_wind_accumulators!(dst::WindGridAccumulator, src::WindGridAccumulator)
    _grid_spec_equal(dst.grid_spec, src.grid_spec) ||
        throw(ArgumentError("merge_wind_accumulators!: grid_spec mismatch"))
    dst.velocity_field == src.velocity_field ||
        throw(ArgumentError("merge_wind_accumulators!: velocity_field mismatch " *
            "($(dst.velocity_field) vs $(src.velocity_field))"))
    dst.n_unknowns == src.n_unknowns ||
        throw(ArgumentError("merge_wind_accumulators!: n_unknowns mismatch " *
            "($(dst.n_unknowns) vs $(src.n_unknowns))"))
    dst.fill_value == src.fill_value ||
        throw(ArgumentError("merge_wind_accumulators!: fill_value mismatch " *
            "($(dst.fill_value) vs $(src.fill_value)) — accumulators built under " *
            "different [io] sentinel conventions cannot be merged"))
    dst.undetect == src.undetect ||
        throw(ArgumentError("merge_wind_accumulators!: undetect mismatch " *
            "($(dst.undetect) vs $(src.undetect)) — accumulators built under " *
            "different [io] sentinel conventions cannot be merged"))

    dst.AtWA         .+= src.AtWA
    dst.AtWb         .+= src.AtWb
    dst.M2           .+= src.M2
    dst.weight_total .+= src.weight_total
    dst.gate_count   .+= src.gate_count
    append!(dst.sweeps, src.sweeps)
    return dst
end

# ── NetCDF output ────────────────────────────────────────────────────────────

# CF standard_name / long_name for a wind component, by output-frame name.
# Known Cartesian names map to CF standard names; anything else (future
# plane/polar frames, e.g. VT/VR) falls back to a generic long_name.
function _cf_component_attrs(name::AbstractString)
    if name == "U"
        return ("eastward_wind", "Eastward wind component")
    elseif name == "V"
        return ("northward_wind", "Northward wind component")
    elseif name == "W"
        return ("upward_air_velocity", "Upward wind component")
    else
        return (nothing, "$(name) wind component")
    end
end

"""
    write_wind_synthesis(file, out::SynthesisOutput, p::DaishoParameters;
                         index_time, start_time=index_time, stop_time=index_time) -> file

Write a [`SynthesisOutput`](@ref) to a CF-1.12 gridded NetCDF file, mirroring the
`write_gridded_radar_volume` layout (X/Y/Z + time dims, projected coordinates, a
Transverse Mercator `grid_mapping`, and per-point latitude/longitude). Field
names follow the active output frame's [`component_names`](@ref): the two
components, their `*STD` uncertainties, plus `DET`, `NGATES`, and `QFLAG`. For
the stage-1 `CartesianFrame` these are `U, V, USTD, VSTD, DET, NGATES, QFLAG`
(with CF `standard_name` `eastward_wind`/`northward_wind`); a polar/plane frame
later emits its own names (e.g. `VT, VR`) through the same adapter. CF global
attributes come from `[grid.metadata]`. Any pre-existing file is deleted first.
"""
function write_wind_synthesis(file::AbstractString, out::SynthesisOutput,
                              p::DaishoParameters;
                              index_time::DateTime,
                              start_time::DateTime = index_time,
                              stop_time::DateTime = index_time)
    g = out.grid_spec
    (g.shape === :volume_3d || g.shape === :latlon_3d) || throw(ArgumentError(
        "write_wind_synthesis: stage-1 output supports :volume_3d and " *
        ":latlon_3d only, got $(g.shape)"))
    nz, ny, nx = size(out.comp1)

    rm(file, force = true)
    ds = NCDataset(file, "c", attrib = _global_attrib_dict(p.grid.metadata))
    try
        ds.dim["time"] = 1
        ds.dim["X"] = nx
        ds.dim["Y"] = ny
        ds.dim["Z"] = nz

        defVar(ds, "time", Float64, ("time",); attrib = OrderedDict(
            "standard_name" => "time", "long_name" => "Data time",
            "units" => "seconds since 1970-01-01T00:00:00Z", "axis" => "T"))[:] =
            datetime2unix(index_time)
        defVar(ds, "start_time", Float64, ("time",); attrib = OrderedDict(
            "standard_name" => "start_time", "long_name" => "Data start time",
            "units" => "seconds since 1970-01-01T00:00:00Z"))[:] =
            datetime2unix(start_time)
        defVar(ds, "stop_time", Float64, ("time",); attrib = OrderedDict(
            "standard_name" => "stop_time", "long_name" => "Data stop time",
            "units" => "seconds since 1970-01-01T00:00:00Z"))[:] =
            datetime2unix(stop_time)

        defVar(ds, "X", Float32, ("X",); attrib = OrderedDict(
            "standard_name" => "projection_x_coordinate", "units" => "m",
            "axis" => "X"))[:] = Float32.(g.x_axis)
        defVar(ds, "Y", Float32, ("Y",); attrib = OrderedDict(
            "standard_name" => "projection_y_coordinate", "units" => "m",
            "axis" => "Y"))[:] = Float32.(g.y_axis)
        defVar(ds, "Z", Float32, ("Z",); attrib = OrderedDict(
            "standard_name" => "altitude", "long_name" => "constant altitude levels",
            "units" => "m", "positive" => "up", "axis" => "Z"))[:] = Float32.(g.z_axis)

        latlon = _compute_latlon_grid(g)   # (ny, nx, 2)
        defVar(ds, "latitude", Float32, ("X", "Y", "time"); attrib = OrderedDict(
            "standard_name" => "latitude", "units" => "degrees_north"))[:, :, 1] =
            Float32.(latlon[:, :, 1]')
        defVar(ds, "longitude", Float32, ("X", "Y", "time"); attrib = OrderedDict(
            "standard_name" => "longitude", "units" => "degrees_east"))[:, :, 1] =
            Float32.(latlon[:, :, 2]')

        defVar(ds, "grid_mapping", Int32, (); attrib = OrderedDict(
            "grid_mapping_name" => "transverse_mercator",
            "scale_factor_at_central_meridian" => 1.0,
            "longitude_of_central_meridian" => g.reference_longitude,
            "latitude_of_projection_origin" => g.reference_latitude,
            "reference_ellipsoid_name" => "GRS80",
            "false_easting" => 0.0, "false_northing" => 0.0))[:] = Int32(-32768)

        # (nz, ny, nx) → (nx, ny, nz) for the X,Y,Z,time variables.
        _xyz(A) = permutedims(A, (3, 2, 1))
        c1, c2 = out.component_names
        s1, s2 = c1 * "STD", c2 * "STD"
        fv = Float32(out.fill_value)

        function _wind_field!(name, data, long_name; std_name = nothing,
                              units = "m s-1")
            attrs = OrderedDict{String,Any}("long_name" => long_name,
                "units" => units, "grid_mapping" => "grid_mapping",
                "coordinates" => "longitude latitude",
                "_FillValue" => fv)
            std_name === nothing || (attrs["standard_name"] = std_name)
            defVar(ds, name, Float32, ("X", "Y", "Z", "time"); attrib = attrs)[:, :, :, 1] =
                Float32.(_xyz(data))
        end

        sn1, ln1 = _cf_component_attrs(c1)
        sn2, ln2 = _cf_component_attrs(c2)
        _wind_field!(c1, out.comp1, ln1; std_name = sn1)
        _wind_field!(c2, out.comp2, ln2; std_name = sn2)
        _wind_field!(s1, out.sigma1, "$(c1) normalized uncertainty (standard deviation)")
        _wind_field!(s2, out.sigma2, "$(c2) normalized uncertainty (standard deviation)")
        _wind_field!("DET", out.det, "Normal-equation determinant (baseline conditioning)";
                     units = "1")

        defVar(ds, "NGATES", Int32, ("X", "Y", "Z", "time"); attrib = OrderedDict(
            "long_name" => "Number of contributing gates",
            "grid_mapping" => "grid_mapping",
            "coordinates" => "longitude latitude"))[:, :, :, 1] = _xyz(out.n_gates)

        defVar(ds, "QFLAG", Int8, ("X", "Y", "Z", "time"); attrib = OrderedDict(
            "long_name" => "Quality flag (bitwise)",
            "flag_masks" => Int8[QFLAG_SIGMA1, QFLAG_SIGMA2, QFLAG_SINGULAR, QFLAG_NODATA],
            "flag_meanings" => "sigma1_above_threshold sigma2_above_threshold " *
                               "singular_geometry no_data",
            "grid_mapping" => "grid_mapping",
            "coordinates" => "longitude latitude"))[:, :, :, 1] = _xyz(out.quality_flag)
    finally
        close(ds)
    end
    return file
end
