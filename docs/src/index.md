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
Pkg.add(url="https://github.com/mmbell/Daisho.jl")
```

## Quick Start

```julia
using Daisho

# Define moment names and create dictionaries
raw_names = ["DBZ", "VEL", "WIDTH", "SQI", "SNR", "RHOHV"]
qc_names = ["DBZ", "VEL", "WIDTH"]
grid_types = [:linear, :weighted, :weighted]
raw_dict, qc_dict, grid_dict = Daisho.initialize_moment_dictionaries(raw_names, qc_names, grid_types)

# Read a CfRadial file
volume = Daisho.read_cfradial("radar_file.nc", raw_dict)

# Initialize QC fields
qc_moments = Daisho.initialize_qc_fields(volume, raw_dict, qc_dict)

# Apply quality control
qc_moments = Daisho.threshold_qc(volume.moments, raw_dict, qc_moments, qc_dict,
    "SQI", 0.3, "SQI", true)

# Grid the data
Daisho.grid_radar_volume(volume, qc_dict, grid_dict, "output.nc", volume.time[1],
    -50000.0, 500.0, 201, -50000.0, 500.0, 201, 0.0, 500.0, 37,
    0.01, 0.5)
```

## Package Structure

- [`Volume`](@ref Daisho.Volume): Core data structure for radar volumes
- [Radar I/O](guide/radar_io.md): Reading and writing CfRadial data
- [Quality Control](guide/quality_control.md): QC workflow and functions
- [Gridding](guide/gridding.md): Gridding algorithms and options
- [SRTM](guide/srtm.md): Terrain data integration
