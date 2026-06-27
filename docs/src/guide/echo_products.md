# Echo Products

*Echo products* are quantities derived from gridded polarimetric fields:
**fuzzy-logic hydrometeor identification** (HID / FHC) and **polarimetric
rain-rate** estimates. Daisho currently computes them on the *gridded* radar variables —
the beam-power-weighted averages produced by the gridding step — rather than on
raw gates. Gridding the smooth polarimetric moments first and classifying the
result avoids the ill-posed problem of regridding integer hydrometeor
categories. The methods are ported from the [CSU Radar Tools](https://github.com/CSU-Radarmet/CSU_RadarTools) Python package.

All echo products are configured through the `[echo]` block and surface as
[`EchoProductsParameters`](@ref Daisho.EchoProductsParameters) on the loaded
[`DaishoParameters`](@ref). They are written *after* gridding, so they are not
part of `[fields]`; the gridded readers know to surface them via
[`echo_output_names`](@ref Daisho.echo_output_names).

## Three ways to run it

**1. Inline, as part of gridding (recommended).** Add an `[echo]` block with
`enabled = true` and grid as usual — the products are appended to the output
NetCDF automatically for every geometry (volume, lat/lon, PPI, RHI) and for the
accumulator path:

```julia
p = DaishoParameters("mygrid.toml")          # [echo] enabled = true
volume = read_cfradial("radar_file.nc")
grid_radar_volume(volume, "out.nc", volume.time_coverage_start, p)
# out.nc now also contains HID_CSU and RATE_CSU_BLENDED
```

**2. Standalone, reprocessing existing grids.** Append products in place to one
or more already-written Daisho grids (single- or multi-time; volume / PPI / RHI)
with [`add_echo_products!`](@ref Daisho.add_echo_products!) — no regridding
required:

```julia
p = DaishoParameters("mygrid.toml")
written = add_echo_products!("leg_gridded.nc", p)   # modifies the file in place
```

A ready-to-use script for this path ships at
`docs/examples/add_echo_products_demo.jl`:

```
julia --project=. docs/examples/add_echo_products_demo.jl config.toml grid.nc [more.nc ...]
```

**3. In-memory, on a field dictionary.** For custom pipelines,
[`apply_echo_products`](@ref Daisho.apply_echo_products) is pure: it takes a
`Dict` of gridded fields and returns a `Dict` of new product fields, leaving I/O
to the caller.

## Configuration

The full annotated `[echo]` block lives in the bundled template (write it with
`print_config("mygrid.toml")`). The essentials:

```toml
[echo]
enabled              = true         # master switch (off by default)
band                 = "S"          # radar band: "S", "C", or "X"
compute_fhc          = true         # write hydrometeor classification
compute_blended_rain = true         # write the blended rain rate
# Individual rain components to also write (any subset):
#   RATE_Z, RATE_Z_CONV, RATE_Z_STRAT, RATE_KDP, RATE_Z_ZDR, RATE_KDP_ZDR
rain_components      = []

# Input field names read from the grid (ldr_field = "" disables LDR):
dbz_field            = "DBZ"
zdr_field            = "ZDR"
kdp_field            = "KDP"
rhohv_field          = "RHOHV"
ldr_field            = ""

# FHC options:
fhc_method           = "hybrid"     # "hybrid" or "linear"
use_temp             = true         # include temperature in the FHC

# Output variable names:
fhc_output           = "HID_CSU"
rain_output          = "RATE_CSU_BLENDED"
rain_method_output   = ""           # "" suppresses the per-cell method field
```

The input names default to `DBZ`/`ZDR`/`KDP`/`RHOHV` — set them to match the
fields actually present in your grid (e.g. the QC'd `*_QC` copies).

### Temperature

The fuzzy classifier uses temperature to separate ice and liquid categories.
Three sources are selectable with `temp_source`:

```toml
temp_source          = "profile"    # "profile", "field", or "reference_state"
temp_field           = "TEMP_FOR_PID"  # gridded field, used when temp_source = "field"
temp_field_units     = "C"          # "C" or "K" (K is converted to °C)
height_field         = ""           # see below
temp_factor          = 1.0          # > 1 broadens the temperature memberships

# Piecewise-linear T(z): height (m) -> temperature (°C), linearly interpolated.
[echo.temperature]
heights      = [0.0,  1000.0, 3000.0, 5000.0, 10000.0, 15000.0]
temperatures = [25.0, 18.0,    5.0,   -8.0,   -40.0,   -70.0]
```

- **`"profile"`** samples the `[echo.temperature]` `T(z)` profile at each cell's
  height. On a 3-D grid the grid z-axis is used directly. On a **2-D PPI/RHI**
  grid (no Z axis), set `height_field` to a gridded beam-height field (e.g.
  `"HEIGHT"`) so the profile can be sampled per cell. See
  [`TemperatureProfile`](@ref Daisho.TemperatureProfile) and
  [`read_temperature_profile`](@ref Daisho.read_temperature_profile).
- **`"field"`** uses a gridded temperature field directly (per cell, may be 3-D),
  converting K → °C when `temp_field_units = "K"`.
- **`"reference_state"`** is reserved for a future Springsteel reference state
  (not yet implemented).

Set `use_temp = false` to classify without temperature.

## Outputs and the sentinel policy

The variables written are exactly those reported by
[`echo_output_names`](@ref Daisho.echo_output_names): the HID field
(`fhc_output`, default `HID_CSU`), the blended rain field (`rain_output`,
default `RATE_CSU_BLENDED`), the optional method field, and any requested
`rain_components`. Re-running overwrites these rather than duplicating them.

The integer HID labels index into [`FHC_SUMMER_CLASSES`](@ref
Daisho.FHC_SUMMER_CLASSES) (class `i` == `FHC_SUMMER_CLASSES[i]`); a label of `0`
marks an unclassified cell.

Echo products preserve Daisho's **true-missing vs clear-air** distinction (see
the [`[io]` sentinels](radar_io.md)):

| Input reflectivity cell | HID | Rain |
|:------------------------|:----|:-----|
| true-missing (`io.fill_value`) | `fill_value` | `fill_value` |
| undetect / clear air (`io.undetect`) | `undetect` | `undetect` (not `0`) |
| valid, but a required input invalid | `fill_value` | `fill_value` |

Writing the undetect sentinel through to the rain fields (rather than `0`) keeps
downstream masking able to tell "scanned, no echo" from "rain rate of zero".

## See also

- [Echo Products API reference](../api/echo_products.md) — full docstrings for
  every function and constant, including the standalone CSU_RadarTools ports
  ([`csu_fhc_summer`](@ref Daisho.csu_fhc_summer),
  [`calc_blended_rain_tropical`](@ref Daisho.calc_blended_rain_tropical), and the
  component rain relations).
- [Gridding](gridding.md) — producing the gridded polarimetric fields that echo
  products are computed from.
