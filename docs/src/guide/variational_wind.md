# Variational Wind Solver

The variational wind solver (stage 2) turns a stage-1
[dual-Doppler retrieval](wind_synthesis.md) into a coupled `(u, v, w)`
analysis that satisfies anelastic mass continuity, on Springsteel's
quadrature grid. See the [theory page](../theory/variational_wind.md) for the
underlying mathematics; this guide covers configuration and workflow.

## Workflow

1. **Stage-1 accumulation** — grid every sweep of every radar volume into a
   [`WindGridAccumulator`](@ref) on the Springsteel quadrature grid, exactly
   as for [spectral gridding](spectral_gridding.md), but with
   `[synthesis] include_vertical_velocity = true` (unless you plan to use
   `horizontal_approximation`) so the per-point normal system carries a `w`
   row.
2. **Variational analysis** — the accumulated per-point normal systems
   become radial-velocity super-observations, combined with anelastic mass
   continuity and (optionally) direct scalar observations, and solved by
   the incremental Gauss–Newton/CG loop.
3. **NetCDF output** — the analysis is exported through Springsteel's
   regular-grid transform, alongside the usual gridded scalar fields.

### High-level entry point

```julia
using Daisho, Dates

p = DaishoParameters("mygrid.toml")   # [variational] enabled = true
sgrid = Daisho.create_radar_grid(p)   # from [grid.springsteel] + [fields]

result = analyze_radar_volumes_variational(
    volumes, "analysis.nc", DateTime(2026, 9, 23, 12), sgrid, p;
    heading = -9999.0, institution = "", source = "",
    include_derivatives = false, frame = CartesianFrame())
```

[`analyze_radar_volumes_variational`](@ref) takes one `Volume` or a vector
of them (multi-radar, exactly as for stage 1 — there is no radar-index
argument); the grid is centred on the centroid of the radars' reference
positions. The call grids every sweep into a [`WindGridAccumulator`](@ref)
(so one `[fields]` entry must carry the `velocity` tag), runs the scalar
spectral products exactly as
[`grid_radar_volume_spectral`](@ref Daisho.grid_radar_volume_spectral)
does, builds the reference-density profile, the state vector, the
background and the observation set, solves the incremental cost function,
and writes the file with [`write_variational_netcdf`](@ref). The third
argument (`index_time`, a `DateTime`) becomes the file's `time` coordinate.

It returns the [`VariationalResult`](@ref). Pass `return_problem = true` to
get `(result, problem)` instead — the [`VariationalProblem`](@ref) holds the
stage-1 product (`problem.stage1`), the observation set, and the state grid
with the analysis loaded — and `write_output = false` to skip the file.

A ready-to-use script for this path ships at
`docs/examples/variational_demo.jl`: it builds the `[variational]`
configuration from a base TOML, runs the analysis, prints the solver stats,
cost-by-group, mass-continuity residual RMS, and `NOBS_VR` coverage, and (with
CairoMakie installed) plots a horizontal `U`/`V` + `DBZ` cross-section and a
vertical `W` cross-section read back from the written NetCDF.

```
julia --project=. docs/examples/variational_demo.jl radarA.nc radarB.nc [out_basename [config.toml]]
```

### In-memory API

For tests, interactive work, or an orchestrator (Sparrow) that wants the
analysis without going through a file, build and solve a problem directly
on the Springsteel quadrature grid:

```julia
spec = Daisho.build_springsteel_grid_spec(sgrid;
    reference_latitude = lat, reference_longitude = lon)
acc = build_accumulator(spec, p)                   # WindGridAccumulator
h_roi, v_roi = Daisho.compute_roi(sgrid; h_factor = p.gridding.horizontal_roi_factor,
                                         v_factor = p.gridding.vertical_roi_factor)
for vol in volumes, s in eachindex(vol.sweeps)
    grid_sweep!(acc, vol, s, p; roi = (h_roi, v_roi))
end
radar_grid = finalize_grid(acc.scalar)             # gridded scalars on the mish

prob = VariationalProblem(acc, radar_grid, sgrid, p)
result = solve_variational(prob)                   # ::VariationalResult

analysis_field(result, "U")                        # U on the mish, (nz, ny, nx)
mass_continuity_residual(prob, result)             # ∂u/∂x + ∂v/∂y + ∂w/∂z + w·dlnρ̄/dz
evaluate_analysis!(prob.state_grid, result)        # load the analysis into the state grid
```

`radar_grid` supplies value observations for any `[fields]`-derived scalar
in the state (§ "Adding a scalar to the state" below); pass `nothing` when
the state has no such variable. [`analysis_field`](@ref),
[`background_field`](@ref) and [`increment_field`](@ref) return mish arrays;
[`horizontal_divergence`](@ref), [`vertical_vorticity`](@ref) and
[`mass_continuity_residual`](@ref) the mish diagnostics. For custom
observation sets and backgrounds see [`build_observation_set`](@ref),
[`Background`](@ref) and the in-memory
`VariationalProblem(state, state_grid, observations, background, reference, p)`
constructor.

## Configuration

A complete `[variational]` configuration, with the state variables it needs
from `[synthesis]` and `[reference_state]`. These are the real blocks from
`config/defaults.toml`; every key not shown here defaults as documented in
[`SynthesisParameters`](@ref), [`ReferenceStateParameters`](@ref), and
[`VariationalParameters`](@ref).

```toml
# Hydrostatic reference-density profile rho-bar(z) — shared by mass continuity,
# the fall-speed correction, and (future) FHC temperature.
[reference_state]
source                 = "exponential"   # "exponential" | "sounding"
density_surface_kg_m3  = 1.2
density_scale_height_m = 8500.0

[synthesis]
velocity_variance = 1.0
max_sigma_1       = 2.0
max_sigma_2       = 2.0
max_elevation     = 45.0
# Accumulate the (u,v,w) 3x3 normal system instead of (u,v) 2x2 — required by
# the variational stage unless [variational.radial_velocity]
# horizontal_approximation = true.
include_vertical_velocity = true

# Hydrometeor fall-speed correction (samurai relations), OPTIONAL. A present
# table means enabled. Needs a [fields] entry tagged `reflectivity`.
[synthesis.fall_speed]
rain_coefficient       = 2.6
rain_exponent          = 0.107
snow_coefficient       = 0.817
snow_exponent          = 0.063
density_exponent       = 0.45
mixed_phase_dbz        = 20.0
rain_dbz               = 30.0
# melting_layer_bottom_m = 4500.0   # absent => [reference_state] freezing level
melting_layer_depth_m  = 1000.0

# Per-gate radial-velocity error model, OPTIONAL. A present table means
# enabled. The spectrum-width term needs a [fields] entry tagged
# `spectrum_width` (silently omitted otherwise).
[synthesis.gate_error]
fall_speed_error_m_s        = 2.0
spectrum_width_error_factor = 1.0
error_guard_min_m_s         = 1.0

# Incremental 3DVAR wind solver.
[variational]
enabled = true
state = ["U", "V", "W"]              # append a [fields] name to analyse a scalar
background = "zero"                  # "zero" | "synthesis" | "synthesis_uniform_sigma" | "synthesis_profile"
background_error_inflation = 1.0
mass_continuity = true
mass_continuity_sigma_per_s = 1.0e-4
max_outer_iterations = 1
max_inner_iterations = 200
inner_tolerance = 1.0e-4
outer_tolerance = 1.0e-3
output_background = false
output_increment  = false
output_mish       = false
filter_type = "gaussian"             # "gaussian" | "lanczos"

[variational.background_error]
default = { sigma = 10.0, filter_length_m = [2000.0, 2000.0, 500.0] }
W       = { sigma = 2.0,  filter_length_m = [2000.0, 2000.0, 500.0] }
# DBZ   = { sigma = 10.0, filter_length_m = [1000.0, 1000.0, 250.0] }

[variational.radial_velocity]
error_inflation = 1.0
horizontal_approximation = false
eigenvalue_guard_relative = 1.0e-10

# One [variational.scalar_obs.<name>] table is REQUIRED for every [fields]
# name appearing in `state`:
# [variational.scalar_obs.DBZ]
# sigma = 5.0
# undetect_value = -10.0

# Optional explicit role override:
# [variational.roles]
# W = "wind_w"

# State variables reuse the [grid.springsteel.k.bc] per-field override table
# by name — e.g. w = 0 at the surface:
[grid.springsteel.k.bc]
min = "natural"
max = "natural"
W = { min = "dirichlet" }
```

### `[variational]` key reference

| Key | Units | Default | Meaning |
|:----|:------|:--------|:--------|
| `enabled` | — | `false` | Master switch. The whole block (and sub-tables) validates strictly whether or not this is `true`. |
| `state` | — | `["U","V","W"]` | State-vector names, in order. Each is `"U"`/`"V"`/`"W"` or a `[fields]` name (appended as a value-observed scalar). Must be non-empty, unique, and contain both `"U"` and `"V"`. |
| `background` | — | `"zero"` | `"zero"`, `"synthesis"`, `"synthesis_uniform_sigma"`, or `"synthesis_profile"`. See "Choosing a background" below. |
| `background_error_inflation` | — | `1.0` | Multiplies σ_b of every state variable in the three `synthesis*` modes (ignored for `"zero"`). Must be positive. |
| `mass_continuity` | — | `true` | Enforce the anelastic continuity pseudo-observation at every mish point. Requires `"W"` in `state`. |
| `mass_continuity_sigma_per_s` | s⁻¹ | `1.0e-4` | σ of the divergence residual; inverse-variance weight `1/σ²`. Must be positive. |
| `max_outer_iterations` | — | `1` | Gauss–Newton outer iterations. `1` is exact when every observation is linear; only matters with a nonlinear observation type. |
| `max_inner_iterations` | — | `200` | CG iteration cap per outer iteration. |
| `inner_tolerance` | — | `1.0e-4` | CG relative residual tolerance, in `(0, 1)`. |
| `outer_tolerance` | — | `1.0e-3` | Relative increment-norm stopping tolerance between outer iterations, in `(0, 1)`. |
| `output_background` | — | `false` | Write `*_BACKGROUND` fields. |
| `output_increment` | — | `false` | Write `*_INCREMENT` fields. |
| `output_mish` | — | `false` | Also write mish-lattice fields and stage-1/super-obs diagnostics (§ "Output variables"). |
| `filter_type` | — | `"gaussian"` | `"gaussian"` or `"lanczos"` mish-space background-error filter kernel. Gaussian has no negative lobes; Lanczos does, near boundaries. |
| `[variational.background_error.<name>].sigma` | native (m/s for winds) | required | Background-error standard deviation σ_b of state variable `<name>` (or `"default"`) at every mish point: the mish filter is variance-normalised (`diag(F Fᵀ) = 1`), so filtered white noise scaled by σ_b has exactly this pointwise standard deviation, whatever the filter length. Must be positive. |
| `[variational.background_error.<name>].filter_length_m` | m | required | Mish-space filter length along `[i, j, k]`. `0.0` on an axis disables filtering there. |
| `[variational.radial_velocity].error_inflation` | — | `1.0` | Multiplies `σ²_vr` for the super-observation inverse variance only. |
| `[variational.radial_velocity].horizontal_approximation` | — | `false` | Drop the `w·sin(el)` term from the super-obs geometry (samurai `horizontal_radar_appx`); allows `[synthesis] include_vertical_velocity = false`. |
| `[variational.radial_velocity].eigenvalue_guard_relative` | — | `1.0e-10` | Eigendirections below this fraction of `λ_max` emit no super-obs row. In `[0, 1)`. |
| `[variational.scalar_obs.<name>].sigma` | native field units | required | Observation-error σ for scalar state variable `<name>`. |
| `[variational.scalar_obs.<name>].undetect_value` | native field units | required | Value substituted at an `[io] undetect` gate before it becomes an observation (§ below). |
| `[variational.roles].<name>` | — | (auto) | Explicit role override (`"wind_u"`, `"wind_v"`, `"wind_w"`, `"reflectivity"`, `"scalar"`); without it, roles resolve automatically. |

### New `[synthesis]` keys

| Key | Units | Default | Meaning |
|:----|:------|:--------|:--------|
| `include_vertical_velocity` | — | `false` | Accumulate the `(u, v, w)` 3×3 normal system instead of the `(u, v)` 2×2 one. Required by the variational stage unless `horizontal_approximation = true`. The stage-1 `(u, v)` product is unchanged in meaning. |
| `[synthesis.fall_speed].rain_coefficient` | m/s | `2.6` | Joss & Waldvogel (1971) rain-relation coefficient. |
| `[synthesis.fall_speed].rain_exponent` | — | `0.107` | Rain-relation exponent on linear Z. |
| `[synthesis.fall_speed].snow_coefficient` | m/s | `0.817` | Atlas et al. (1973) snow-relation coefficient. |
| `[synthesis.fall_speed].snow_exponent` | — | `0.063` | Snow-relation exponent on linear Z. |
| `[synthesis.fall_speed].density_exponent` | — | `0.45` | Exponent on `ρ̄(0)/ρ̄(z)` (Beard 1985). |
| `[synthesis.fall_speed].mixed_phase_dbz` | dBZ | `20.0` | At/below this, the above-melting-layer speed is pure snow. |
| `[synthesis.fall_speed].rain_dbz` | dBZ | `30.0` | Above this, the above-melting-layer speed is pure rain; linear blend between. |
| `[synthesis.fall_speed].melting_layer_bottom_m` | m | `NaN` | Melting-layer bottom height. `NaN` ⇒ the `[reference_state]` freezing level (error if that profile has no temperature). |
| `[synthesis.fall_speed].melting_layer_depth_m` | m | `1000.0` | Height over which the fall speed blends from rain to the above-layer value. |
| `[synthesis.gate_error].fall_speed_error_m_s` | m/s | `2.0` | Assumed fall-speed error, scaled by `|sin(el)|`. |
| `[synthesis.gate_error].spectrum_width_error_factor` | — | `1.0` | Multiplier on the gate spectrum width (needs a `spectrum_width`-tagged field). |
| `[synthesis.gate_error].error_guard_min_m_s` | m/s | `1.0` | Lower clamp on the per-gate σ. |

`[reference_state]` (`source`, `density_surface_kg_m3`, `density_scale_height_m`,
`sounding_file`, `moisture`, `temperature_surface_k`, `lapse_rate_k_m`) is
documented in full on the [Reference State guide](reference_state.md); it is
shared unchanged by mass continuity and the fall-speed correction above.

## Choosing a background

`W` (and any scalar state variable) always starts from 0 with its
configured σ_b; the modes differ in the `U`/`V` first guess and σ_b. The
three `synthesis*` modes need the stage-1 product (the high-level
`VariationalProblem(acc, …)` supplies it) and multiply every σ_b by
`background_error_inflation`.

- **`"zero"`** (default): the background is 0 everywhere, with σ_b from
  `[variational.background_error]`. This is the samurai-like choice, right
  when no first guess deserves more trust than the current volume's own
  data. At the edges of the data the analysis relaxes toward 0 over the
  filter length.
- **`"synthesis"`**: `U`/`V` background = a **smooth fit** to the stage-1
  `(u, v)` at unflagged points. The fit is one auxiliary variational solve
  (one extra CG solve): configured filter and σ, one value observation per
  good point weighted by its stage-1 σ, and no continuity or radar
  super-obs. It relaxes smoothly to 0 over the filter length in data gaps.
  σ_b is the stage-1 σ map with its variance smoothed by the same filter,
  clamped to `[σ/10, σ]` of the configured σ. The background is tightest
  where stage 1 is well determined and falls back to the configured σ in
  gaps. Use it to *adjust* a good stage-1 analysis, for example to add `w`
  and enforce continuity without moving well-observed dual-Doppler winds far,
  or to steady the analysis near a baseline.
- **`"synthesis_uniform_sigma"`**: the same smooth fit, with the configured
  (uniform) σ_b. Use it when stage-1 coverage is spotty or its σ is
  unreliable. The first guess helps, but no point is trusted more than
  another.
- **`"synthesis_profile"`**: `U`/`V` background = the **level-mean** stage-1
  profile (the plain mean of the good points on each mish level, constant
  horizontally; levels with none are interpolated linearly in height),
  with the configured σ_b. It is the least local of the synthesis modes.
  Use it for spotty coverage, or as an environmental first guess that
  relaxes gaps toward the mean flow instead of 0.

**Double counting.** Stage 1 and the radial-velocity super-observations are
built from the *same* gates, so every `synthesis*` background counts that
data twice: once in the prior mean, once in the likelihood. `"synthesis"`
double counts most, because it trusts the stage-1 σ; `"synthesis_profile"`
least. Increase `background_error_inflation` to weaken the background
relative to the always-present radar term. It trades off the double count;
it does not eliminate it. It also raises the Hessian's condition number
`∝ σ_b²`, so large inflations need a larger `max_inner_iterations`. In the
test case, inflation 10 took ≈ 4–5k CG iterations and inflation 100 took
≈ 20–30k.

## Adding a scalar to the state

Append a `[fields]` name to `state` to analyse it alongside the wind — its
gridded mish values become direct observations of a new state variable
(e.g. a smoothed, mass-continuity-consistent reflectivity field):

```toml
[variational]
state = ["U", "V", "W", "DBZ"]

[variational.scalar_obs.DBZ]
sigma = 5.0             # observation-error sigma, native field units
undetect_value = -10.0  # value substituted for an [io] undetect gate

[variational.background_error]
default = { sigma = 10.0, filter_length_m = [2000.0, 2000.0, 500.0] }
DBZ     = { sigma = 10.0, filter_length_m = [1000.0, 1000.0, 250.0] }
```

At each mish point, the gridded value of `DBZ` is inspected against the
`[io]` sentinels **before** it becomes an observation — this is the same
true-missing/undetect distinction used everywhere else in Daisho and it must
not be collapsed:

- `[io] fill_value` (default `-32768.0`, true missing — never scanned) or
  `NaN`: **no observation** at that point.
- `[io] undetect` (default `-9999.0`, scanned but no detectable echo): an
  observation of `undetect_value`, e.g. a reflectivity floor well below the
  detection threshold — the point *is* informative (it says "no echo here"),
  it just cannot use the raw sentinel as a value.
- Any other value: an observation of that value directly.

## Boundary conditions for state variables

State variables resolve boundary conditions from the same
`[grid.springsteel.<axis>.bc]` per-field override tables the obs/scalar grid
uses, keyed by state name:

```toml
[grid.springsteel.k.bc]
min = "natural"
max = "natural"
W = { min = "dirichlet" }   # w = 0 at the surface
```

Every axis without its own override for a state variable falls back to that
axis's `"default"` (`min`/`max`) BC. **Inhomogeneous** boundary conditions
(`R3X`/`R1T1X`, whose boundary data live in Springsteel's `ahat` vector) are
refused for state variables — the variational increment is homogeneous by
construction, so that data would be silently unusable; use a homogeneous
type (`natural`, `dirichlet` at 0, `neumann` at 0, …) instead.

## Output variables

[`write_variational_netcdf`](@ref) writes one file. The scalar `[fields]`
come first, exactly as [`grid_radar_volume_spectral`](@ref Daisho.grid_radar_volume_spectral)
writes them (Springsteel's regular output grid, dims `x, y, z`, plus
`time`); every regular-grid variable is `Float64` with a `NaN` fill on the
same dims.

On the **regular output grid**:

| Variable | Units | Meaning |
|:---------|:------|:--------|
| `U`, `V`, `W` | m s⁻¹ | The analysis (a `[fields]`-derived state variable such as `DBZ` is written as `DBZ_ANALYSIS`). |
| `VORT` | s⁻¹ | `∂v/∂x − ∂u/∂y` |
| `DIV` | s⁻¹ | `∂u/∂x + ∂v/∂y` |
| `MCRESIDUAL` | s⁻¹ | `DIV + ∂w/∂z + w·dlnρ̄/dz` — the anelastic continuity residual, a direct check of solver convergence (only when `W` is in the state). |
| `U_x`, `U_xx`, `U_y`, … | | With `include_derivatives = true`: the derivative fields of every analysis variable, with the same suffixes as the scalars. |
| `*_BACKGROUND` | | With `output_background`: the background, per state variable. |
| `*_INCREMENT` | | With `output_increment`: analysis − background, so `BACKGROUND + INCREMENT = analysis`. |

With `output_mish = true`, also on the **mish (quadrature) lattice**
(`x_mish`, `y_mish`, `z_mish` dimensions): `U_MISH`, `V_MISH`, `W_MISH`
(the analysis itself at the quadrature points, `Float64`), the stage-1
per-point product `U_STAGE1`, `V_STAGE1`, `USTD`, `VSTD`, `DET`, `NGATES`,
`QFLAG` (never resampled — these are properties of the accumulator, not the
spline; written as the QC'd product with the `[io]` fill, as
[`write_wind_synthesis`](@ref) does), the super-observation row count
`NOBS_VR` (§ theory, §4), and `RHOBAR(z_mish)` (the reference-density
profile actually used).

Global attributes add `variational_background`,
`variational_outer_iterations`, `variational_cg_iterations` (total CG
iterations) and `variational_converged` (`1`/`0`) to the usual radar set.

## Troubleshooting

- **`NOBS_VR` is 0 or 1 at many points.** The stage-1 geometry there is
  degenerate (single look or near-baseline) — expected at range-fold edges
  or near a radar baseline; the mass-continuity term and the background
  still constrain those points, just more weakly. Widespread low
  `NOBS_VR` across the whole domain usually means too few radars are
  contributing, or `[synthesis] max_elevation`/ROI settings are excluding
  more gates than intended.
- **High `MCRESIDUAL`.** Either the solver has not converged (check
  `result.converged` and `result.stats.outer_iterations` /
  `result.stats.inner_iterations` ([`VarSolverStats`](@ref)), or the
  `variational_*` global attributes of the file —
  raise `max_inner_iterations` or loosen `inner_tolerance` if CG is hitting
  the cap) or `mass_continuity_sigma_per_s` is too loose relative to the
  radial-velocity term (tighten it, or check `[reference_state]` for a
  ρ̄(z) that does not match the analysis domain).
- **CG iteration counts climbing or not converging.** Conditioning is driven
  by `(σ_b/σ_o)² Σw` — very large background σ or very dense, low-error
  observation coverage steepens the Hessian; check
  `[variational.background_error]` σ values are physically reasonable, and
  that `[variational.radial_velocity] error_inflation` /
  `mass_continuity_sigma_per_s` are not making the observation term
  overwhelmingly dominant relative to the `I` term.
- **Checking which observation term dominates the cost.** The result
  carries a per-kind cost breakdown (`result.cost_by_group`; per backend:
  `result.stats.cost_by_backend`) — compare `:radial_velocity`,
  `:mass_continuity`, and `:scalar_value` (if present) to see which term the
  optimizer is spending its effort on.

## Migrating from samurai

Old samurai keys are rejected with a diagnostic naming the Daisho
equivalent (from `[variational]`, its sub-tables, and `[synthesis]`):

| samurai key | Daisho equivalent |
|:------------|:-------------------|
| `mc_weight` | `mass_continuity_sigma_per_s` in `[variational]` |
| `bg_rhou_error` | a `U` entry in `[variational.background_error]` |
| `bg_rhov_error` | a `V` entry in `[variational.background_error]` |
| `bg_rhow_error` | a `W` entry in `[variational.background_error]` |
| `i_filter_length` | `filter_length_m` (metres, per axis, i first) in `[variational.background_error.<name>]` |
| `j_filter_length` | `filter_length_m` (metres, per axis, j second) in `[variational.background_error.<name>]` |
| `k_filter_length` | `filter_length_m` (metres, per axis, k third) in `[variational.background_error.<name>]` |
| `horizontal_radar_appx` (in `[variational.radial_velocity]` context) | `horizontal_approximation` in `[variational.radial_velocity]` |
| `num_iterations` | `max_outer_iterations` in `[variational]` |
| `radar_fallspeed_error` | `fall_speed_error_m_s` in `[synthesis.gate_error]` |
| `radar_sw_error` | `spectrum_width_error_factor` in `[synthesis.gate_error]` |
| `radar_min_error` | `error_guard_min_m_s` in `[synthesis.gate_error]` |
| `horizontal_radar_appx` (in `[synthesis]` context) | `include_vertical_velocity` in `[synthesis]` (samurai `horizontal_radar_appx = true` ≈ `include_vertical_velocity = false`) |
| `max_radar_elevation` | `max_elevation` in `[synthesis]` (degrees) |
| `melting_zone_width` | `melting_layer_depth_m` in `[synthesis.fall_speed]` (metres; samurai used km) |

Also remember the two semantic differences that are *not* caught by a key
rename (theory page, §10): observation weights are `1/σ²` (not samurai's
`1/σ`), and filter lengths are configured in **metres**, with an *effective*
`FF^\top` correlation length of `√2` times the configured (cell) length —
re-derive, don't copy, ported σ and filter-length values.
