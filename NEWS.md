# Daisho.jl release notes

## v0.2.0 — Flat per-field tag vocabulary, optional blocks, gate-state terminology

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

### Migration

- `config/defaults.toml`, `config/seapol.toml` rewritten here.
- Regenerate scratch configs via `print_config("template.toml")`.
- Old `*.jld2` per-sweep accumulator files are incompatible
  (`GRID_ACCUMULATOR_SCHEMA_VERSION` 1 → 2); `load_accumulator` raises a
  clear version-mismatch error. Regenerate them.
- Sparrow.jl passes a path string to `DaishoParameters` — no Sparrow code
  change, but its Daisho `.toml` files must be migrated.
