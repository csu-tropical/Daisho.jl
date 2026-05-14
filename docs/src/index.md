# Daisho.jl

**Daisho.jl** is a Julia package for radar meteorology data processing, quality control, and gridding. It provides tools for working with weather radar data in CfRadial format, performing quality control operations, and gridding radar observations onto regular Cartesian or latitude-longitude grids.

## Features

- **CfRadial I/O**: Read and write CfRadial NetCDF radar data files
- **Quality Control**: Threshold-based QC, despeckling, platform motion removal, terrain masking
- **Gridding**: Beam-aware interpolation onto 3D Cartesian, lat/lon, RHI, PPI, composite, and column grids
- **Moving Platform Support**: Full support for airborne and ship-based radars with platform motion correction
- **SRTM Integration**: Digital elevation model integration for terrain-aware quality control
- **Coordinate Transforms**: Transverse Mercator projections and approximate inverse projections

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/csu-tropical/Daisho.jl")
```

## Quick Start

Daisho is driven by a single TOML parameter file loaded into a
[`DaishoParameters`](@ref) struct. The struct bundles
the canonical field list, QC thresholds, gridding knobs, grid geometry
(Cartesian / lat-lon / RHI / spectral), CF-1.12 output metadata, and I/O
fill values. To start a configuration, write the bundled template to a path
you control and edit it for your deployment:

```julia
using Daisho

print_config("mygrid.toml")            # write the template
# edit mygrid.toml for your radar, grid, and CF metadata
p = DaishoParameters("mygrid.toml")    # strict load
```

The `[fields]` block in the TOML replaces the old
`initialize_moment_dictionaries` helper. It lists the canonical field names
of interest plus a per-name interpolation hint used by the gridder:

```toml
[fields]
names = ["DBZ", "ZDR", "KDP", "RHOHV", "VEL", "WIDTH", "PHIDP", "SQI"]

[fields.grid_type]
DBZ   = "linear"
ZDR   = "linear"
KDP   = "weighted"
RHOHV = "weighted"
VEL   = "weighted"
WIDTH = "weighted"
PHIDP = "weighted"
SQI   = "weighted"
```

Grid types control how each field is interpolated during gridding:
`"linear"` (convert to linear units before averaging, e.g., reflectivity),
`"weighted"` (weighted average in native units), or `"nearest"`
(nearest-neighbor). Loading is strict: every key documented in the template
must be present in your file. A missing or mis-typed key raises an
`ArgumentError` at load time naming the offending section — there is no
silent fallback to defaults.

With `p` in hand, the read → QC → grid pipeline is:

```julia
# Read a CfRadial file (auto-detects CfRadial 1.4 vs 2.1)
volume = read_cfradial("radar_file.nc")

# Apply per-sweep threshold QC using p.qc thresholds.
# Each input field FOO gains a QC'd copy FOO_QC in sweep.fields.
for sweep in volume.sweeps
    threshold_qc!(sweep, p)
end

# Grid the data using p.grid.cartesian, p.gridding, p.moments, p.grid.metadata
grid_radar_volume(volume, "output.nc", volume.time_coverage_start, p)
```

See the [Gridding](guide/gridding.md) guide for the lat-lon, RHI, PPI,
composite, column, and spectral variants, and the multi-sweep accumulator
workflow for building rolling grids across flight legs.

## Package Structure

- [`Volume`](@ref Daisho.Volume): Core data structure for radar volumes
- [Radar I/O](guide/radar_io.md): Reading and writing CfRadial data
- [Quality Control](guide/quality_control.md): QC workflow and functions
- [Gridding](guide/gridding.md): Gridding algorithms and options
- [SRTM](guide/srtm.md): Terrain data integration
