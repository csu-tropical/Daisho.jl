# Spectral Gridding with Springsteel

Daisho integrates with [Springsteel.jl](https://github.com/csu-tropical/Springsteel.jl) to provide spectral analysis of gridded radar data using cubic B-spline basis functions. This enables analytic derivatives, spectral filtering, and (future) variational wind synthesis.

## Quick Start

```julia
using Daisho
using Springsteel

# Define moments and grid
moment_dict = Dict("DBZ" => 1, "VEL" => 2, "WIDTH" => 3)
grid_type_dict = Dict(1 => :linear, 2 => :weighted, 3 => :weighted)

# Create a 3D spectral grid (RRR = splines in all three dimensions)
sgrid = Daisho.create_radar_grid("RRR", moment_dict;
    xmin=-100000.0, xmax=100000.0, xdim=10,   # 10 cells in X (easting)
    ymin=-100000.0, ymax=100000.0, ydim=10,   # 10 cells in Y (northing)
    zmin=0.0, zmax=15000.0, zdim=5)            # 5 cells in Z (altitude)

# Grid and analyze
sgrid = Daisho.grid_radar_volume_spectral(
    radar_volume, moment_dict, grid_type_dict,
    "output.nc", index_time, sgrid,
    0.0,    # beam_inflation
    0.01)   # power_threshold
```

## Dimension Conventions

Daisho maps Springsteel's abstract dimensions to physical radar coordinates:

| Springsteel | Daisho | Physical meaning |
|:-----------:|:------:|:----------------|
| i           | X      | Easting (meters) |
| j           | Y      | Northing (meters) |
| k           | Z      | Altitude (meters) |

The `xdim`, `ydim`, `zdim` parameters in `create_radar_grid` specify the number of B-spline **cells**, not grid points. Each cell contains `mubar` (default: 3) Gaussian quadrature points, so the actual number of physical grid points is `cells * mubar`.

## Grid Types

| Type  | Geometry | Use case |
|:-----:|:--------:|:---------|
| `"R"` | 1D       | Vertical profiles, single columns |
| `"RR"`| 2D       | PPI scans, horizontal cross-sections |
| `"RRR"`| 3D      | Full volume analysis |

## Workflow

The spectral gridding pipeline follows these steps:

1. **Create grid**: `create_radar_grid()` builds a Springsteel grid with the right geometry and variables
2. **Extract coordinates**: `get_springsteel_gridpoints_zyx()` converts quadrature points to Daisho's `[z,y,x]` array format
3. **Compute ROI**: `compute_roi()` determines radius-of-influence from grid spacing
4. **Grid data**: The existing `grid_volume`/`grid_ppi` engine interpolates radar gates onto the grid
5. **Populate spectral grid**: `populate_physical!()` maps gridded values into the Springsteel physical array
6. **Spectral transform**: `spectralTransform!()` computes B-spline coefficients
7. **Grid transform**: `gridTransform!()` evaluates values and analytic derivatives
8. **Write output**: `write_radar_netcdf()` produces CF-1.12 compliant NetCDF

The high-level functions (`grid_radar_volume_spectral`, `grid_radar_ppi_spectral`, `grid_radar_column_spectral`) execute this entire pipeline in one call.

## Boundary Conditions

Boundary conditions control the spline behavior at domain edges:

| BC | Name | Effect |
|:---|:-----|:-------|
| `CubicBSpline.R0` | Rank 0 | No constraint (default) |
| `CubicBSpline.R1T0` | Zero value | Field goes to zero at boundary |
| `CubicBSpline.R1T1` | Zero derivative | Flat boundary (Neumann) |
| `CubicBSpline.R1T2` | Zero curvature | Natural spline boundary |

For radar data, `R0` (unconstrained) is typically appropriate since the domain edges are arbitrary analysis boundaries.

## Fill Value Handling

Radar data distinguishes three mutually exclusive gate states (CfRadial 2.1 /
ODIM vocabulary), two of which are non-valid and handled differently:

- **true missing**: the gate was not measured. Sentinel `[io] fill_value`
  (CF `_FillValue`, default `-32768.0`). Converted to `NaN` in the spectral grid.
- **undetect**: the gate was scanned but no signal was detected. Sentinel
  `[io] undetect` (ODIM `_Undetect`, default `-9999.0`). Preserved in the
  output — this is physically meaningful information.
- **valid**: the gate was scanned and a signal was detected (a real value).

Before spectral transforms, both fill states are temporarily replaced with `0.0` to prevent spectral corruption, then restored afterward using a mask. This is a known approximation -- future variational analysis will handle data gaps properly via the cost function.

## Accessing Derivatives

After `gridTransform!`, the physical array contains analytic derivatives:

```julia
# For a 3D (RRR) grid:
# Slot 1: f(x,y,z)     — field values
# Slot 2: df/dx         — x-derivative
# Slot 3: d²f/dx²       — x second derivative
# Slot 4: df/dy         — y-derivative
# Slot 5: d²f/dy²       — y second derivative
# Slot 6: df/dz         — z-derivative
# Slot 7: d²f/dz²       — z second derivative

# Example: get vertical gradient of reflectivity
dbz_idx = moment_dict["DBZ"]
dDBZ_dz = sgrid.physical[:, dbz_idx, 6]
```

## NetCDF Output

Output files are CF-1.12 compliant with:
- Coordinate variables with units in meters
- Transverse Mercator grid mapping
- Standard radar variable attributes (units, long\_name)
- Optional derivative fields via `include_derivatives=true`
