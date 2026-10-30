# Gridding

Daisho.jl implements a beam-aware radar gridding algorithm that accounts for Earth curvature, standard atmospheric refraction, and the radar beam pattern.

## Grid Types

Several gridding modes are available:

| Mode | Function | Dimensions | Use Case |
|------|----------|------------|----------|
| Volume | `grid_radar_volume` | 3D (X, Y, Z) | Full volumetric analysis |
| Lat/Lon Volume | `grid_radar_latlon_volume` | 3D (lon, lat, Z) | Geographic coordinate grids |
| RHI | `grid_radar_rhi` | 2D (R, Z) | Range-height cross sections |
| PPI | `grid_radar_ppi` | 2D (X, Y) | Plan position indicator |
| Composite | `grid_radar_composite` | 2D (X, Y) | Maximum value composite |
| Column | `grid_radar_column` | 1D (Z) | Vertical profile above radar |

## Basic Usage

All grid geometry, field selection, interpolation, weighting, and CF metadata
come from a [`DaishoParameters`](@ref) loaded from a TOML config (see
*Configuration files* below). The drivers take the `Volume`, an output path, an
index time, and the parameters:

### Volume Gridding

```julia
p = DaishoParameters("mygrid.toml")
volume = read_cfradial("scan.nc")
Daisho.grid_radar_volume(volume, "output.nc", volume.time_coverage_start, p)
```

### PPI Gridding

```julia
Daisho.grid_radar_ppi(volume, "output_ppi.nc", volume.time_coverage_start, p)
```

The `[grid.*]` sub-tables of the config select each product's geometry
(`grid_radar_volume` uses `[grid.volume]`, falling back to `[grid.cartesian]`;
`grid_radar_rhi` uses `[grid.rhi]`; and so on).

## Gridding Algorithm

The algorithm uses a BallTree spatial index for efficient nearest-neighbor queries:

1. Build a BallTree of all radar gate locations in projected coordinates
2. For each grid point, find nearby radar gates within the radius of influence
3. Compute beam-pattern weights (Gaussian based on angular separation)
4. Compute range weights
5. Apply weighted interpolation (linear for dBZ, weighted average for velocity, etc.)

## Key Parameters

- **`power_threshold`**: The beam power level that defines the beam edge. The edge-referenced gate inclusion sets the beam half-angle to `beam_cutoff = ln(1/power_threshold) / beam_coef`, so a **lower** `power_threshold` makes the beam **wider** (more of the exponential tail is counted as the beam). At the default `0.5` this is the half-power half-beamwidth. (The former `beam_inflation` knob is removed — per-range beam reach is now derived from the radar beamwidth and footprint; a leftover `beam_inflation` key is accepted and ignored.)
- **`horizontal_roi_factor`** / **`vertical_roi_factor`**: Multipliers on the horizontal/vertical grid increment setting the grid-cell half-width/half-height for gate inclusion. Optional; default `0.75` each.
- **`range_guard_min`** / **`range_weight_max`**: Numerical guards on the near-radar `range_weight` singularity — a lower clamp (metres) on the slant-range divisor and an upper clamp on the resulting (unitless) weight. Optional; defaults `1.0` and `10.0`. These are *weighting* guards, not gate filters.
- **`range_minimum`** / **`range_maximum`**: Slant-range **gate-inclusion** bounds (metres). A gate is gridded only when its slant range `r` satisfies `range_minimum ≤ r ≤ range_maximum`; out-of-range gates are excluded from every product (no coverage, no value), so a cell reachable only by excluded gates is left true-missing. Optional; defaults `0.0` / `Inf` (no filtering). Use them to homogenise a volume whose low sweep ranges farther than the others, or to cap a product at a higher-quality range.
- **Field roles (`define_detection` / `define_scanned`)**: The gate-role moments are not gridding parameters — they are declared as field *tags* in `[fields]`. Tag the detection field (its presence proves a detectable echo; formerly `valid_key`, e.g. `DBZ`) with `define_detection`, and the scanned-indicator field (its presence proves the gate was scanned; formerly `missing_key`, e.g. `SQI` or `PID`) with `define_scanned`. They are resolved at gridding time via [`field_with_tag`](@ref Daisho.field_with_tag).

## Configuration files

The high-level [`DaishoParameters`](@ref) struct bundles QC thresholds, gridding knobs, grid geometry (Cartesian / lat-lon / RHI / spectral), CF-1.12 metadata, and I/O fill values. It is loaded from a TOML file. To start a configuration, write the bundled template to a path you control and edit it for your deployment:

```julia
using Daisho
print_config("mygrid.toml")            # write the template
# edit mygrid.toml: radar, grid geometry, [grid.metadata] CF attributes …
p = DaishoParameters("mygrid.toml")    # strict load
```

`DaishoParameters(path)` is strict: every key documented in the template must be present in your file. There is no silent fallback to bundled defaults — a missing or mis-typed key raises an `ArgumentError` at load time, naming the offending section. This avoids subtle bugs where a forgotten or fat-fingered parameter would silently take a default value.

## Coordinate Systems

Daisho uses Transverse Mercator projections (via CoordRefSystems.jl) centered on the radar location. The approximate inverse projection function converts projected coordinates back to lat/lon:

```julia
lat, lon = Daisho.appx_inverse_projection(ref_lat, ref_lon, [y_meters, x_meters])
```

### Grid origin and moving frame

By default, the projection origin is the radar position: the volume's
lat/lon, or the first georeference fix for a mobile platform. Multi-radar
Springsteel and variational runs use the centroid of the radar positions
instead. The optional `[grid.origin]` table overrides this for every grid kind,
so the grid doesn't depend on which files are in a folder and can match
another analysis:

```toml
[grid.origin]
latitude_deg   = 18.0
longitude_deg  = 150.0
```

Adding a constant motion turns the grid into a moving, storm-relative frame.
This is useful for fast-moving squall lines, or hurricane flights that span
several hours:

```toml
[grid.origin]
latitude_deg     = 18.0                    # where the origin is at reference_time
longitude_deg    = 150.0
reference_time   = "2008-09-19T04:00:00Z"  # UTC; required with a nonzero motion
motion_east_mps  = -4.0                    # u
motion_north_mps =  0.0                    # v
```

The origin is at `(latitude_deg, longitude_deg)` at `reference_time` and moves
at `(u, v)`. Each ray observed at time `t` is placed at its projected position
minus `(u, v)·(t − reference_time)`. A feature moving with the frame therefore
grids to one place however long the analysis window is.
- The motion can be combined with the default origin: leave the lat/lon unset
  and the frame passes through the radar position at `reference_time`.
- All of these keys are optional. Absent keys keep the default origin and a
  fixed earth frame.
- An origin equal to the radar position gives exactly the default result.

Notes:
- Positions use the same Transverse Mercator projection as the rest of the
  gridding. This replaces SAMURAI's frame-of-reference (`.cen`) file and its
  generator script, whose flat-earth degree-to-km factors give slightly
  different positions far from the origin.
- Winds remain earth-relative: radial velocities are earth-relative, and
  only positions move with the frame.
- Gridded lat/lon coordinates are where the grid sits at `reference_time`.
  Files written with a moving frame carry the global attributes
  `origin_reference_time`, `frame_motion_east_mps` and `frame_motion_north_mps`.
- RHI grids accept an origin override but reject a nonzero motion. Column
  (QVP) grids use height only, so the motion has no effect on them. The legacy
  `radar`-struct drivers accept the origin but reject a nonzero motion; use
  the `Volume` API (`read_cfradial`) for a moving frame.

## Output Format

Gridded data is written as CF-compliant NetCDF files with:
- Coordinate variables (X, Y, Z or lat, lon)
- Grid mapping metadata (Transverse Mercator)
- All gridded moments as 3D or 2D variables
- Time, start_time, stop_time
- Latitude/longitude grids for each horizontal point

## Per-sweep gridding and accumulator-based combination

The Volume drivers above build a [`ScalarGridAccumulator`](@ref Daisho.ScalarGridAccumulator)
internally, fold every sweep of the volume into it, and finalize once. The
accumulator is also exposed as a first-class object so callers can grid one
sweep at a time, persist intermediate state to JLD2, and combine sweeps from
many files later. (To grid the scalar fields *and* the dual-Doppler wind in one
pass, use [`build_accumulator`](@ref Daisho.build_accumulator) — see the
[wind synthesis guide](wind_synthesis.md).) This is the natural workflow for airborne radars where each
CfRadial file is one sweep along a flight track, and for multi-Doppler retrieval
inputs where sweeps from different radars need to land on a common grid.

The flow uses three verbs:

- [`grid_sweep_to_file`](@ref Daisho.grid_sweep_to_file) — grid one sweep
  of a volume into a JLD2 accumulator file. Either create a fresh accumulator
  with a `grid_spec` argument, or merge into an existing rolling file.
- [`combine_accumulator_files`](@ref Daisho.combine_accumulator_files) — merge
  many per-sweep accumulator files into one, provided they share a grid spec.
- [`finalize_accumulator_file`](@ref Daisho.finalize_accumulator_file) — divide
  by weights, apply linear→dBZ conversion, and write a normalized NetCDF using
  the appropriate `write_gridded_radar_*` writer.

Example (one CfRadial file per sweep along a flight leg):

```julia
p = DaishoParameters("mygrid.toml")
first_volume = read_cfradial(first(p3_files))
grid_spec = build_grid_spec(:volume_3d, first_volume, p)

for file in p3_files
    volume = read_cfradial(file)
    threshold_qc!(volume.sweeps[1], p)
    grid_sweep_to_file(volume, 1, "leg.accum.jld2", p;
                       grid_spec = grid_spec,
                       merge_into_existing = true)
end

finalize_accumulator_file("leg.accum.jld2", "leg_gridded.nc", p;
                          index_time = first_volume.time_coverage_start)
```

### Field-folds quantities

Radial velocity (`VEL`) and other folding quantities (anything where
`Field.metadata.field_folds == true`) cannot be physically meaningfully merged
across distinct look directions. The merge step
([`combine_accumulator_files`](@ref Daisho.combine_accumulator_files) and
[`merge_accumulators!`](@ref Daisho.merge_accumulators!)) refuses to combine
field-folds fields across distinct sweeps and points the user at the
wind-retrieval workflow, which reads the per-sweep accumulators directly.
