# Core Types and Volume Helpers

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

## Legacy `radar` Compatibility

Helpers to convert between the CfRadial 2.1 [`Volume`](@ref Daisho.Volume) type
and the legacy `radar` struct.

```@docs
Daisho.as_volume
Daisho.as_legacy_radar
```
