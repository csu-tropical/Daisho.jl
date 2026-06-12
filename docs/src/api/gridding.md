# Gridding

## Grid Initialization

```@docs
Daisho.initialize_regular_grid
Daisho.appx_inverse_projection
```

## High-Level Gridding (Volume-typed)

```@docs
Daisho.grid_radar_volume
Daisho.grid_radar_latlon_volume
Daisho.grid_radar_rhi
Daisho.grid_radar_ppi
Daisho.grid_radar_composite
Daisho.grid_radar_column
```

## Core Gridding Algorithms

```@docs
Daisho.grid_volume
Daisho.grid_rhi
Daisho.grid_ppi
Daisho.grid_composite
Daisho.grid_column
```

## Multi-Sweep Accumulator

```@docs
Daisho.GridSpec
Daisho.SweepProvenance
Daisho.ScalarGridAccumulator
Daisho.build_accumulator
Daisho.build_grid_spec
Daisho.grid_sweep!
Daisho.finalize_grid
Daisho.merge_accumulators!
Daisho.save_accumulator
Daisho.load_accumulator
Daisho.grid_sweep_to_file
Daisho.combine_accumulator_files
Daisho.finalize_accumulator_file
```

## Gridded Data I/O

```@docs
Daisho.write_grid_products
Daisho.write_gridded_radar_volume
Daisho.write_gridded_radar_rhi
Daisho.write_gridded_radar_ppi
Daisho.write_gridded_radar_column
Daisho.read_gridded_radar
Daisho.read_gridded_ppi
Daisho.read_gridded_rhi
Daisho.mask_sentinels
```
