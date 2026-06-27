# Daisho.jl

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Stable: [![Stable Build Status](https://github.com/csu-tropical/Daisho.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/csu-tropical/Daisho.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://csu-tropical.github.io/Daisho.jl/stable/)

Development: [![Development Build Status](https://github.com/csu-tropical/Daisho.jl/actions/workflows/CI.yml/badge.svg?branch=development)](https://github.com/csu-tropical/Daisho.jl/actions/workflows/CI.yml?query=branch%3Adevelopment) [![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://csu-tropical.github.io/Daisho.jl/dev/)

[![codecov](https://codecov.io/gh/csu-tropical/Daisho.jl/graph/badge.svg)](https://codecov.io/gh/csu-tropical/Daisho.jl)

Daisho is a data analysis and assimilation package written in Julia for atmospheric science applications, with an emphasis on weather radar. It is being developed to provide functionality similar to the FRACTL and SAMURAI C++ software, with additional polarimetric radar processing and beam-aware gridding for moving platforms. The name comes from the Japanese term meaning "large and small" and is the name of a matched pair of traditionally made Japanese swords (*nihontō*) worn by the samurai in feudal Japan. Daisho provides the radar data-analysis and assimilation front end that pairs with the [Springsteel.jl](https://github.com/csu-tropical/Springsteel.jl) semi-spectral grid engine. The software is in active development and is still in pre-release.

## What's in the box

- **CfRadial I/O** — read and write CfRadial NetCDF radar volumes, auto-detecting CfRadial 1.4 vs 2.1, with robust handling of fill values, packed fields, and per-sweep metadata.
- **Quality control** — threshold-based QC, despeckling, platform-motion removal, and SRTM terrain masking, configured per field through a documented tag vocabulary.
- **Beam-aware gridding** — interpolate radar observations onto 3-D Cartesian, lat/lon, RHI, PPI, composite, and column grids, with edge-referenced radius-of-influence that accounts for beam geometry on moving (airborne / ship-based) platforms.
- **Spectral gridding** — grid directly onto [Springsteel.jl](https://github.com/csu-tropical/Springsteel.jl) semi-spectral grids for downstream variational analysis.
- **Echo products** — post-gridding hydrometeor identification (FHC) and rain-rate retrievals under a configurable `[echo]` namespace.
- **Multi-Doppler wind synthesis** — stage-1 (u, v) variational synthesis (in development).
- **TOML-driven configuration** — the entire read → QC → grid pipeline is driven by a single, strictly-validated `DaishoParameters` TOML file.

## Installation

Daisho is a registered Julia package. To install, open the Julia REPL, enter Package mode by pressing `]`, and run:

```
pkg> add Daisho
```

Or equivalently from Julia code:

```julia
using Pkg
Pkg.add("Daisho")
```

For development work, clone the repo and `pkg> dev /path/to/Daisho.jl` to track local changes.

## Quick start

Daisho is driven by a single TOML parameter file loaded into a `DaishoParameters` struct, which bundles the field list, QC thresholds, gridding knobs, grid geometry, CF output metadata, and I/O fill values. Write the bundled template, edit it for your deployment, then run the read → QC → grid pipeline:

```julia
using Daisho

print_config("mygrid.toml")              # write the template, then edit it
p = DaishoParameters("mygrid.toml")      # strict load

volume = read_cfradial("radar_file.nc")  # auto-detects CfRadial 1.4 vs 2.1

for sweep in volume.sweeps               # per-sweep threshold QC → FOO_QC fields
    threshold_qc!(sweep, p)
end

grid_radar_volume(volume, "output.nc", volume.time_coverage_start, p)
```

The `[fields]` block maps each field to a flat array of tags from a documented vocabulary (interpolation style, detection/scanned roles, …). For the lat/lon, RHI, PPI, composite, column, and spectral variants, plus the multi-sweep accumulator workflow, see the [Gridding guide](https://csu-tropical.github.io/Daisho.jl/dev/guide/gridding/).

## Documentation

Full documentation lives at **[csu-tropical.github.io/Daisho.jl/dev/](https://csu-tropical.github.io/Daisho.jl/dev/)**. Start here:

| Page | What you'll find |
|:-----|:-----------------|
| [Reading & Writing Radar Data](https://csu-tropical.github.io/Daisho.jl/dev/guide/radar_io/) | CfRadial 1.4 / 2.1 I/O, the `Volume` / sweep / field types, fill-value handling |
| [Quality Control](https://csu-tropical.github.io/Daisho.jl/dev/guide/quality_control/) | Threshold QC, despeckling, platform-motion removal, terrain masking |
| [Gridding](https://csu-tropical.github.io/Daisho.jl/dev/guide/gridding/) | Cartesian / lat-lon / RHI / PPI / composite / column grids and the accumulator workflow |
| [Spectral Gridding](https://csu-tropical.github.io/Daisho.jl/dev/guide/spectral_gridding/) | Gridding onto Springsteel semi-spectral grids |
| [Multi-Doppler Wind Synthesis](https://csu-tropical.github.io/Daisho.jl/dev/guide/wind_synthesis/) | Stage-1 (u, v) variational synthesis |
| [SRTM Terrain Integration](https://csu-tropical.github.io/Daisho.jl/dev/guide/srtm/) | Digital elevation models for terrain-aware QC |
| [Theory](https://csu-tropical.github.io/Daisho.jl/dev/theory/beam_geometry/) | Beam geometry, the gridding algorithm, and wind-synthesis derivations |
| [API Reference](https://csu-tropical.github.io/Daisho.jl/dev/api/parameters/) | Full docstring reference for parameters, types, I/O, QC, gridding, and echo products |

## Contributing

Issues and pull requests are welcome. Daisho is in active pre-release development; multi-Doppler and variational analysis are the current focus, alongside deeper integration with the [Springsteel.jl](https://github.com/csu-tropical/Springsteel.jl) framework. Please open an issue to discuss substantial changes before submitting a PR.
