# Daisho.jl changelog

## Unreleased

## v0.2.0 — Flat per-field tag vocabulary, optional blocks, gate-state terminology

*First registered release. Changes below are relative to the unregistered v0.1
tag.*

**Breaking TOML / config change.** There is no transitional reader (consistent
with the strict-no-fallback philosophy); three targeted migration diagnostics
guide the rewrite.

### `[fields]` is now a flat per-field tag array

Each field maps to an array of tags drawn from a documented allowlist instead
of a `names` array plus a `[fields.grid_type]` sub-table:

```toml
[fields]
DBZ = ["linear_interp", "define_detection"]
SQI = ["weighted_interp", "define_scanned"]
```

- Interpolation tags `linear_interp` / `weighted_interp` / `nearest_interp`
  (at most one per field; `weighted_interp` is the default when none given)
  replace the per-name `grid_type` map.
- `define_detection` / `define_scanned` replace the `[gridding]`
  `valid_key` / `missing_key` keys (now removed) — the gate-role moments are
  properties *of a field*, declared as field tags and resolved at point of
  use. Zero declared role tags is allowed for non-gridding workflows; the
  error is raised only when a gridding driver actually needs one.
- `velocity` is reserved (validated, consumed by nothing yet) for forthcoming
  multi-Doppler radial-velocity selection.
- Field and tag order are insignificant; internal column order is derived
  deterministically by sorted field name.

### Optional blocks

`[fields]` and `[io]` are mandatory. `[qc]`, `[gridding]`, and the `[grid.*]`
sub-tables are optional and default-construct when absent. An operation that
needs a missing block raises a clear point-of-use error naming the operation
and the section. Validation *within* any present block stays strict.
`DaishoParameters` gained a `provided::Set{Symbol}` field recording which
optional top-level sections were present.

### Gate-state terminology reconciled (CfRadial 2.1 / ODIM)

One spec-anchored vocabulary is used verbatim across code, docstrings, and
configs:

- **true missing** — gate not measured — CF `_FillValue` — `[io] fill_value`
- **undetect** — scanned, no detectable signal — ODIM `_Undetect` —
  `[io] undetect`
- **valid** — scanned, signal detected — the measured value

`[io]` keys were renamed `fill_value_missing` → `fill_value` and
`fill_value_clear` → `undetect` to mirror `FieldMetadata` exactly. `[io]` is
now authoritative: `p.io.fill_value` / `p.io.undetect` thread through the
accumulator, `finalize_grid`, and gridded NetCDF writers (which now also emit
`_Undetect`). Behavior is preserved at the default sentinel values.

### Hybrid scan near-surface product

A post-gridding product that collapses the PPI tilts of one time into a single
2-D field set staying as close to the surface as the radar allows, configured
by the `[hybrid_scan]` block (`HybridScanParameters`).

- `apply_hybrid_scan` is the pure kernel over per-tilt field dicts;
  `build_hybrid_scan` reads gridded PPI files, resolves each tilt from the
  recorded `fixed_angle` (with an `angle_pattern` filename fallback for older
  archives), and writes the product as another gridded PPI.
- `require_detection` is the one science switch, applied identically at every
  tilt: `false` takes the lowest measurement of any kind, so a tilt reporting
  `undetect` has observed clear air and ends the search; `true` takes the
  lowest rain, climbing past clear air to look for echo.
- The base tilt is never height-gated — at long range even the lowest sweep
  exceeds `beam_height_maximum` and is still the best available look — while
  fills from above it are. The elevation output names the tilt each value came
  from, and is the fill value only where no tilt measured the cell at all.

### Sweep fixed angle and a generic 2-D product writer

`write_gridded_radar_ppi` gains an optional `fixed_angle`, written as both a
global attribute and a scalar variable, so a single-sweep PPI grid is
self-describing about the tilt it came from instead of relying on the filename.
The `Volume` driver supplies it when the volume holds exactly one sweep and the
legacy `radar` driver uses the mean ray elevation; a composite spans every
elevation and must not claim one, so it stays unset there.

`write_gridded_fields_2d` writes a named field dict as an `(X, Y, time)`
NetCDF, taking its coordinate scaffolding verbatim from an existing gridded
PPI. The result is layout-identical to a gridded PPI, so `read_gridded_ppi`
consumes derived products unchanged.

### Echo products on 1-D column (QVP) grids

`grid_radar_column` now runs the `[echo]` products (hydrometeor ID, rain rate)
like every other geometry — both the `Volume`/accumulator and the legacy `radar`
overloads previously wrote the column grid and silently skipped the echo hook, so
a quasi-vertical profile came out with no rain rate and no error.

`add_echo_products!` correspondingly accepts the 1-D column layout (a `Z`
dimension with no `X`/`Y`/`R`), alongside the existing `X,Y,Z`, `R,Z` and `X,Y`
layouts. Because a column's only axis is its z-axis, `temp_source = "profile"`
samples the `[echo.temperature]` profile directly with no `height_field` needed.

### Fields-API gridded readers

`read_gridded_rhi` / `read_gridded_ppi` / `read_gridded_radar` gain a
`DaishoParameters` method (`read_gridded_*(file, p::DaishoParameters)`) that
returns a `NamedTuple` of coordinates plus a `fields::Dict{String,Array}` keyed
by field name and pre-shaped to the grid, with `io = p.io` carried along so
callers resolve `fill_value`/`undetect` without re-reading config. Raw sentinels
are preserved in the data; `mask_sentinels(a, io)` collapses both to `NaN` for
display. This is the standard reading path going forward — symmetric with the
`DaishoParameters` grid writers.

The legacy `read_gridded_*(file, moment_dict)` (name→index) methods are
**deprecated** (they emit `Base.depwarn`) but still work.

### Springsteel spectral gridding

Radar data grids onto Springsteel cubic B-spline spectral grids through the
same edge-referenced, beamwidth-correct unified accumulator as the Cartesian
paths; only the node layout differs. The grid is configured by
`[grid.springsteel]` plus per-axis `[grid.springsteel.i]` / `.j` / `.k`
sub-tables in Springsteel's native i/j/k vocabulary, and grid variables are
validated against `[fields]` up front. Multiple volumes accumulate onto one
shared grid. Requires the registered Springsteel ≥ 1.0.

### CfRadial reader fixes

- Scalars that carry a `_FillValue` and are never written — `volume_number`,
  `sweep_number`, `platform_type` and friends in LROSE aircraft files — are now
  treated as absent and take the documented default. Previously they either
  threw `MethodError: no method matching Int64(::Missing)` or quietly read back
  as the literal string `"missing"`.
- `range` / `frequency` variables carrying a `_FillValue` (typed
  `Union{Missing,Float32}` by NCDatasets) no longer crash `read_cfradial` with
  "Cannot convert Missing to Float64"; they are routed through the
  missing→`NaN` vector helper (GitHub Issue #3).
- CfRadial 2.1 per-sweep `georeference` groups were silently dropped on every
  read — the guard tested for a variable literally named `:group` — losing
  airborne/mobile platform geometry. They are now read correctly.

### Migration

- `config/defaults.toml`, `config/seapol.toml` rewritten here.
- Regenerate scratch configs via `print_config("template.toml")`.
- Old `*.jld2` per-sweep accumulator files are incompatible
  (`GRID_ACCUMULATOR_SCHEMA_VERSION` 1 → 2); `load_accumulator` raises a
  clear version-mismatch error. Regenerate them.
- Sparrow.jl passes a path string to `DaishoParameters` — no Sparrow code
  change, but its Daisho `.toml` files must be migrated.
