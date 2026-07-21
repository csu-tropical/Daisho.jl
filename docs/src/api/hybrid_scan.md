# Hybrid Scan

A near-surface 2-D product built by collapsing a set of same-time gridded PPI
tilts into one field set, driven by
[`HybridScanParameters`](@ref Daisho.HybridScanParameters) from the
`[hybrid_scan]` configuration block. See the
[Hybrid Scan guide](../guide/hybrid_scan.md) for the algorithm and worked
configuration.

## Hybrid Scan Driver

```@docs
Daisho.HybridScanParameters
Daisho.apply_hybrid_scan
Daisho.build_hybrid_scan
Daisho.hybrid_scan_output_names
```

## Tilt Geometry

```@docs
Daisho.hybrid_beam_heights
Daisho.hybrid_scan_tilt_angle
```
