# Hybrid Scan

A *hybrid scan* collapses the several PPI tilts of one volume into a single 2-D
field set that stays as close to the surface as the radar allows.

The lowest sweep is the best look at the surface, but it is also the one most
often blocked, ducted, or simply out of range — and at long range its beam has
climbed well above the ground anyway. The hybrid scan starts from that base tilt
and, wherever it has no answer for a cell, walks upward through the higher tilts
and takes the first one that does answer, provided that tilt's beam is still near
the surface there.

The result is a gridded PPI like any other, plus an `elevation_angle` field
recording which tilt supplied each cell. Because the layout is identical to a
gridded PPI, [`read_gridded_ppi`](@ref Daisho.read_gridded_ppi) and the plotting
tools read it back unchanged.

Configuration lives in the `[hybrid_scan]` block and surfaces as
[`HybridScanParameters`](@ref Daisho.HybridScanParameters) on the loaded
[`DaishoParameters`](@ref). The product runs *after* gridding, and after
[echo products](echo_products.md) when the fields it carries (rain rate,
hydrometeor ID) are echo outputs.

## What counts as an answer

The single science choice is `require_detection`, and it is applied identically
at the base tilt and at every tilt above it:

| `require_detection` | A tilt answers a cell when… | You get |
|---|---|---|
| `false` (default) | its reading is not the `fill_value` sentinel | the lowest **measurement** of any kind — a tilt reporting `undetect` has scanned the cell and observed clear air, and that ends the search |
| `true` | its reading is a real detection | the lowest **rain** — the search climbs past clear air looking for echo |

This is why the two sentinels must stay distinct through gridding: `fill_value`
means *never measured* (blocked, out of range, not scanned) and is always worth
looking upward from, while `undetect` means *scanned, nothing there* and is a
genuine observation. See [`IOParameters`](@ref Daisho.IOParameters).

## The base tilt

`base_angle` names the preferred base tilt; when no available tilt is within
`base_angle_tolerance` of it — a volume that skipped the lowest sweep, say — the
lowest tilt present is used instead. Leave `base_angle` unset to always take the
lowest available.

The base tilt is not otherwise privileged. In particular it is **never
height-gated**: at long range even the lowest sweep exceeds
`beam_height_maximum`, and it is still the best available look at that cell. The
height limit applies only to fills from above, so a gap is never patched with an
echo from aloft.

## Beam heights

Each tilt needs a per-cell beam height to test against `beam_height_maximum`.
Daisho prefers a gridded beam-height field when one is configured and present —
set `height_field` to a field tagged `beam_height` in `[fields]` — and otherwise
computes it from the tilt angle and the cell's distance from the radar with
[`hybrid_beam_heights`](@ref Daisho.hybrid_beam_heights), which uses the 4/3
effective-earth [`beam_height`](@ref Daisho.beam_height) model at
`radar_altitude`.

## Configuration

```toml
[hybrid_scan]
enabled              = true
fields               = []          # [] = every field present in the tilts
select_field         = "DBZ"       # the field whose readings drive the selection
base_angle           = 0.5
base_angle_tolerance = 0.05
beam_height_maximum  = 1000.0      # metres; limit on fills from above the base
radar_altitude       = 20.0        # metres; antenna height
require_detection    = false
height_field         = ""          # gridded beam-height field, when available
elevation_output     = "elevation_angle"
height_output        = ""          # "" suppresses
angle_pattern        = ""          # legacy filename fallback, see below
```

Leaving `fields` empty carries every field the tilts have, which is what keeps
the output a drop-in gridded PPI. Name fields explicitly only when you want a
slimmer product; `select_field` must be among them.

## Running it

Point [`build_hybrid_scan`](@ref Daisho.build_hybrid_scan) at the gridded PPI
files of one time:

```julia
p = DaishoParameters("mygrid.toml")            # [hybrid_scan] enabled = true
files = filter(f -> occursin("gridded_ppi_20240903_150007", f),
               readdir("gridded/ppi/20240903"; join=true))
build_hybrid_scan(files, "hybrid_20240903_1500.nc", p)
```

Each file's tilt comes from the `fixed_angle` that
[`write_gridded_radar_ppi`](@ref Daisho.write_gridded_radar_ppi) records in
single-sweep PPI grids. For archives written before that existed, set
`angle_pattern` to a regular expression with one capture group matching the angle
in the filename, e.g. `"_([0-9]+\\.[0-9]+)\\.nc$"` for
`gridded_ppi_20240903_150007_0.5.nc`.

To work in memory instead — over tilts you have already read, or as part of a
larger pipeline — call [`apply_hybrid_scan`](@ref Daisho.apply_hybrid_scan) with
per-tilt field dicts and heights.

Sparrow's `HybridScanStep` wraps `build_hybrid_scan` as a workflow step, running
it over each time chunk's PPI grids.

## Reading the output

`elevation_angle` names the tilt each value came from, which makes the product
self-diagnosing: a map of it shows the low-tilt coverage hole being filled and
how high the fills reach. It is the `fill_value` sentinel only where no tilt
measured the cell at all; cells no tilt *answered* keep the base tilt's values
and report the base angle, since that is where the value came from.
