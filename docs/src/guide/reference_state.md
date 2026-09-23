# Reference State

A *reference-density profile* ρ̄(z) is the shared hydrostatic base state that
several Daisho consumers need: the variational-stage anelastic mass-continuity
constraint (`∂u/∂x + ∂v/∂y + ∂w/∂z + w·dlnρ̄/dz = 0`), the radar fall-speed
correction, and — in a future phase — FHC temperature. Rather than each
consumer building its own profile, Daisho builds one
[`AbstractReferenceProfile`](@ref Daisho.AbstractReferenceProfile) from the
optional `[reference_state]` block and passes it around.

Configuration surfaces as
[`ReferenceStateParameters`](@ref Daisho.ReferenceStateParameters) on the
loaded [`DaishoParameters`](@ref); build the profile itself with
[`build_reference_profile`](@ref Daisho.build_reference_profile):

```julia
p = DaishoParameters("mygrid.toml")
ρ̄ = build_reference_profile(p, sgrid)     # sgrid only matters for source = "sounding"
reference_density(ρ̄, 3000.0)              # kg/m^3 at z = 3000 m
dlnrho_dz(ρ̄, 3000.0)                      # m^-1
```

The block is absent by default, so every existing configuration keeps loading
unchanged (`source = "exponential"` with the shipped defaults).

## Two sources

### `"exponential"` — no grid needed

```toml
[reference_state]
source                 = "exponential"
density_surface_kg_m3  = 1.2
density_scale_height_m = 8500.0
```

`ρ̄(z) = density_surface_kg_m3 · exp(−z / density_scale_height_m)`, so
`dlnrho_dz` is exactly `−1/H` everywhere — a cheap, always-available profile
for testing and for radars without a sounding.

Uncommenting `temperature_surface_k` (and, optionally, `lapse_rate_k_m`, which
defaults to the US Standard Atmosphere tropospheric value `0.0065` K/m) adds a
linear temperature lapse `T(z) = T₀ − Γz`, enabling
[`reference_temperature_kelvin`](@ref Daisho.reference_temperature_kelvin),
[`freezing_level_m`](@ref Daisho.freezing_level_m), and
[`temperature_celsius`](@ref Daisho.temperature_celsius). Leaving
`temperature_surface_k` unset (`NaN`) means no temperature profile is
available: the first two return `nothing` and `temperature_celsius` throws.

### `"sounding"` — full hydrostatic state from a sounding

```toml
[reference_state]
source        = "sounding"
sounding_file = "sounding.txt"
moisture      = true
```

This builds a full hydrostatic reference state via Springsteel's reference-
state machinery (`Springsteel.calculate_reference_state`), sampled on the
vertical mish points of the grid's k-axis — the same spectral basis the grid
itself uses, so the density and its vertical derivative are mutually
consistent to machine precision (the derivative is read directly from the
reference state's spectral derivative slot, never finite-differenced).

`moisture = true` (default) includes the vapor partial density in ρ̄
(`ρ̄ = ρ̄_d + ρ̄_v`); `moisture = false` builds a dry state (`ρ̄ = ρ̄_d`).

Sampling at a height between two mish levels uses cubic Hermite interpolation
on the `(ρ̄, dρ̄/dz)` pairs, and is clamped to the end values outside the mish
range — a query never extrapolates past the profile.

#### Sounding file format

The samurai/Scythe format: a surface line, then one line per level.

```text
p_sfc theta_sfc q_v_sfc
z1 theta1 q_v1
z2 theta2 q_v2
...
```

- `p_sfc`: surface pressure (hPa).
- `theta`: potential temperature (K).
- `q_v`: water-vapor mixing ratio (g/kg).
- `z`: height (m) — must cover at least the grid's vertical extent; a model
  level outside the sounding's altitude range is an error.

## Choosing a profile for variational continuity

The exponential profile is a reasonable default for quick-look or synthetic
work; for a real case, a sounding close to the analysis time and location
gives mass continuity a base state that actually matches the observed
buoyancy structure, which matters more as the vertical domain grows (a fixed
scale height over-simplifies a real troposphere-to-stratosphere sounding).
