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

### Volume Gridding

```julia
Daisho.grid_radar_volume(
    volume, qc_dict, grid_type_dict, "output.nc", index_time,
    -50000.0, 500.0, 201,    # X: min, increment, dimension
    -50000.0, 500.0, 201,    # Y: min, increment, dimension
    0.0, 500.0, 37,          # Z: min, increment, dimension
    0.01,                     # beam_inflation
    0.5                       # power_threshold
)
```

### PPI Gridding

```julia
Daisho.grid_radar_ppi(
    volume, qc_dict, grid_type_dict, "output_ppi.nc", index_time,
    -125000.0, 500.0, 501,   # X
    -125000.0, 500.0, 501,   # Y
    0.01, 0.5                # beam_inflation, power_threshold
)
```

## Gridding Algorithm

The algorithm uses a BallTree spatial index for efficient nearest-neighbor queries:

1. Build a BallTree of all radar gate locations in projected coordinates
2. For each grid point, find nearby radar gates within the radius of influence
3. Compute beam-pattern weights (Gaussian based on angular separation)
4. Compute range weights
5. Apply weighted interpolation (linear for dBZ, weighted average for velocity, etc.)

## Key Parameters

- **`beam_inflation`**: Expands the radius of influence proportionally to distance from the radar. Accounts for beam broadening with range. Used by the 2D/legacy drivers; the 3D edge-referenced path derives its per-range reach from the beam footprint instead. Typical value: 0.01.
- **`power_threshold`**: The beam power level that defines the beam edge. In the 3D edge-referenced path the beam half-angle is `beam_cutoff = ln(1/power_threshold) / beam_coef`, so a **lower** `power_threshold` makes the beam **wider** (more of the exponential tail is counted as the beam). At the default `0.5` this is the half-power half-beamwidth.
- **`horizontal_roi_factor`** / **`vertical_roi_factor`**: Multipliers on the horizontal/vertical grid increment setting the grid-cell half-width/half-height for gate inclusion. Optional; default `0.75` each.
- **`range_floor`** / **`range_weight_max`**: Numerical guards on the near-radar `range_weight` singularity — a lower clamp (metres) on the slant-range divisor and an upper clamp on the resulting weight. Optional; defaults `1.0` and `10.0`.
- **`missing_key`**: The moment used to determine if a gate has valid data (e.g., "SQI").
- **`valid_key`**: The moment used to check for non-missing data (e.g., "DBZ").

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
