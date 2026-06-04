# API Reference

## Parameters

```@docs
Daisho.DaishoParameters
Daisho.print_config
Daisho.require_section
Daisho.MomentParameters
Daisho.FieldSpec
Daisho.has_tag
Daisho.interp_of
Daisho.field_with_tag
Daisho.field_index_dict
Daisho.grid_type_index_dict
Daisho.QCParameters
Daisho.GriddingParameters
Daisho.GridParameters
Daisho.CartesianGridParameters
Daisho.LatLonGridParameters
Daisho.RhiGridParameters
Daisho.SpectralGridParameters
Daisho.SpectralBCParameters
Daisho.MetadataParameters
Daisho.IOParameters
```

## Core Types (CfRadial 2.1)

```@docs
Daisho.Volume
Daisho.SweepGroup
Daisho.Field
Daisho.FieldMetadata
Daisho.Georeference
Daisho.RadarMonitoring
Daisho.RadarParameters
Daisho.RadarCalibration
Daisho.RadarCalibrationEntry
Daisho.GeoreferenceCorrection
Daisho.SpectrumGroup
Daisho.LidarMonitoring
Daisho.LidarParameters
Daisho.LidarCalibration
```

## Volume Helpers

```@docs
Daisho.add_field!
Daisho.remove_field!
Daisho.n_rays
Daisho.field_names
Daisho.has_field
```

## Radar I/O

```@docs
Daisho.read_cfradial
Daisho.write_cfradial
Daisho.update_cfradial
Daisho.validate_spec
Daisho.ValidationReport
Daisho.get_radar_orientation
```

## Radar Utilities

```@docs
Daisho.beam_height
Daisho.dB_to_linear
Daisho.linear_to_dB
Daisho.dB_to_linear!
Daisho.linear_to_dB!
```

## Quality Control

```@docs
Daisho.threshold_qc!
Daisho.fix_SEAPOL_RHOHV!
Daisho.despeckle
Daisho.despeckle_azimuthal
Daisho.stddev_phidp_threshold
Daisho.remove_platform_motion!
Daisho.threshold_dbz
Daisho.threshold_height
Daisho.threshold_terrain_height
Daisho.add_azimuthal_offset
Daisho.mask_sector
Daisho.smooth_sqi
```

## Gridding

### Grid Initialization

```@docs
Daisho.initialize_regular_grid
Daisho.appx_inverse_projection
```

### High-Level Gridding (Volume-typed)

```@docs
Daisho.grid_radar_volume
Daisho.grid_radar_latlon_volume
Daisho.grid_radar_rhi
Daisho.grid_radar_ppi
Daisho.grid_radar_composite
Daisho.grid_radar_column
```

### Spectral Gridding (Springsteel)

```@docs
Daisho.grid_radar_volume_spectral
Daisho.grid_radar_ppi_spectral
Daisho.grid_radar_column_spectral
Daisho.create_radar_grid
```

### Spectral Helpers

```@docs
Daisho.get_springsteel_gridpoints_zyx
Daisho.populate_physical!
Daisho.write_radar_netcdf
Daisho.radar_global_attributes
```

### Core Gridding Algorithms

```@docs
Daisho.grid_volume
Daisho.grid_rhi
Daisho.grid_ppi
Daisho.grid_composite
Daisho.grid_column
```

### Multi-Sweep Accumulator

```@docs
Daisho.GridSpec
Daisho.SweepProvenance
Daisho.GridAccumulator
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

### Multi-Doppler Wind Synthesis (Stage 1)

```@docs
Daisho.WindGridAccumulator
Daisho.wind_accumulator_dims
Daisho.grid_sweep_wind!
Daisho.finalize_wind
Daisho.SynthesisOutput
Daisho.SynthesisFrame
Daisho.CartesianFrame
Daisho.component_names
Daisho.rotation_at
Daisho.SynthesisParameters
Daisho.save_wind_accumulator
Daisho.load_wind_accumulator
Daisho.merge_wind_accumulators!
Daisho.write_wind_synthesis
```

### Gridded Data I/O

```@docs
Daisho.write_gridded_radar_volume
Daisho.write_gridded_radar_rhi
Daisho.write_gridded_radar_ppi
Daisho.write_gridded_radar_column
Daisho.read_gridded_radar
Daisho.read_gridded_ppi
Daisho.read_gridded_rhi
```

## SRTM / Terrain

```@docs
Daisho.terrain_height
Daisho.read_srtm_elevation_rasters
Daisho.read_srtm_elevation_multi
Daisho.read_srtm_elevation_dict
```
