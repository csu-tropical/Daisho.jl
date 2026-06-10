# Spectral Gridding with Springsteel

Daisho integrates with [Springsteel.jl](https://github.com/csu-tropical/Springsteel.jl) to provide spectral analysis of gridded radar data using cubic B-spline basis functions. This enables analytic derivatives, spectral filtering, and (future) variational wind synthesis.

## Quick Start

Configure the grid in your TOML under `[grid.springsteel]`, then everything
runs off the parameter struct:

```julia
using Daisho
using Springsteel

p = DaishoParameters("myconfig.toml")
volume = read_cfradial("cfrad.20140703_..._SUR.nc")

sgrid = Daisho.create_radar_grid(p)    # from [grid.springsteel] + [fields]
sgrid = Daisho.grid_radar_volume_spectral(volume, "output.nc", nothing, sgrid, p)
```

Multiple radars can be accumulated onto one shared grid centered on the
centroid of the radar positions:

```julia
vols = read_cfradial.(files)
Daisho.grid_radar_volume_spectral(vols, "output.nc", nothing, sgrid, p)
```

The grids are produced by the same edge-referenced, beamwidth-correct unified
accumulator engine as the regular Cartesian path; only the node layout (the
Gauss quadrature lattice) differs.

## Configuration: `[grid.springsteel]`

The section speaks Springsteel's native **i/j/k** axis vocabulary, because
x/y/z only makes sense for Cartesian geometries — i is easting for Cartesian
grids but radius for cylindrical ones:

```toml
[grid.springsteel]
geometry   = "RRR"     # R | RZ | RR | RRR | RL | RLZ | RLR | SL | SLZ | SLR
mubar      = 3         # quadrature points per B-spline cell
quadrature = "gauss"   # "gauss" | "regular"

[grid.springsteel.i]   # easting (Cartesian) / radius (cylindrical, spherical)
min   = -120000.0
max   =  120000.0
cells = 240            # B-spline CELLS, not gridpoints (points = cells × mubar)
# regular_out = 241    # optional; output resampling count (default cells + 1)

[grid.springsteel.i.bc]
min = "natural"
max = { type = "dirichlet", value = 0.0 }
# DBZ = { min = "neumann" }   # per-field override

[grid.springsteel.j]   # northing (Cartesian); auto-sized Fourier azimuth for RL*/SL*
min   = -120000.0
max   =  120000.0
cells = 240

[grid.springsteel.k]   # altitude
min   = 0.0
max   = 18000.0
cells = 19
```

Per-axis keys depend on the basis the geometry puts on the axis: B-spline axes
take `min`/`max`/`cells`, Chebyshev axes (the `k` of `RZ`/`RLZ`/`SLZ`) take
`min`/`max`/`points`, and the Fourier azimuth of cylindrical/spherical
geometries is auto-sized (only an optional `max_wavenumber`). The Springsteel
variable map is always derived from `[fields]` — it is not configured here.

!!! note "Gridding scope"
    Any supported geometry can be **built**, transformed, and written, but the
    accumulator engine currently **grids radar data onto the Cartesian
    R/RR/RRR grids only**; cylindrical/spherical gridding (per-node
    gridpoints + metric ROI) is a planned follow-on.

### Escape hatch: hand-built grids

Some Springsteel configuration cannot round-trip through TOML (per-variable
spectral filters, the Fourier/Chebyshev-primary `L*`/`Z*` geometries,
computed values). For those, build the grid in Julia and pass it to the same
entry points — `p` still supplies `[fields]`, `[io]`, `[gridding]`, and QC:

```julia
gp = Springsteel.SpringsteelGridParameters(
    geometry = "RRR", iMin = -120000.0, iMax = 120000.0, num_cells = 240,
    spline_filter = Dict("DBZ" => ...),    # things TOML can't express
    vars = radar_vars(p))                  # REQUIRED: column-order contract
sgrid = Springsteel.createGrid(gp)
Daisho.grid_radar_volume_spectral(volume, "out.nc", nothing, sgrid, p)
```

The entry points validate the grid's `vars` against the configured `[fields]`
(via [`radar_vars`](@ref)) and refuse mismatches, since the gridded-array
column order must line up with the spectral-grid variable order.

## Dimension Conventions

Daisho maps Springsteel's abstract dimensions to physical radar coordinates
(Cartesian geometries):

| Springsteel | Daisho | Physical meaning |
|:-----------:|:------:|:----------------|
| i           | X      | Easting (meters) |
| j           | Y      | Northing (meters) |
| k           | Z      | Altitude (meters) |

Cell counts are B-spline **cells**, not grid points. Each cell contains
`mubar` (default: 3) Gaussian quadrature points, so the actual number of
physical grid points is `cells * mubar` — unlike `[grid.cartesian]`, whose
dims count gridpoints.

## Grid Types

| Type  | Geometry | Use case |
|:-----:|:--------:|:---------|
| `"R"` | 1D       | Vertical profiles, single columns |
| `"RR"`| 2D       | PPI scans, horizontal cross-sections |
| `"RRR"`| 3D      | Full volume analysis |
| `"RZ"`, `"RL"`, `"RLZ"`, `"RLR"`, `"SL"`, `"SLZ"`, `"SLR"` | 2D/3D | Constructible/transformable today; gridding support planned |

## Workflow

The spectral gridding pipeline follows these steps:

1. **Create grid**: `create_radar_grid(p)` builds a Springsteel grid from `[grid.springsteel]` and `[fields]`
2. **Express the lattice**: `build_springsteel_grid_spec()` reproduces the quadrature lattice as a Daisho `GridSpec`
3. **Compute ROI**: `compute_roi()` derives a representative radius-of-influence from the mean (non-uniform) node spacing
4. **Grid data**: the unified accumulator (`ScalarGridAccumulator` + `grid_sweep!` with the ROI override) interpolates radar gates onto the lattice
5. **Populate spectral grid**: `populate_physical!()` maps gridded values into the Springsteel physical array
6. **Spectral transform**: `spectralTransform!()` computes B-spline coefficients
7. **Grid transform**: `gridTransform!()` evaluates values and analytic derivatives
8. **Write output**: `write_radar_netcdf()` produces CF-1.12 compliant NetCDF resampled onto a regular output grid (`regular_out` points per axis)

The high-level functions (`grid_radar_volume_spectral`, `grid_radar_ppi_spectral`, `grid_radar_column_spectral`) execute this entire pipeline in one call.

## Boundary Conditions

Boundary conditions control the spline behavior at domain edges. Each axis'
`bc` table sets the `min`/`max` sides, with optional per-field overrides:

| TOML type | Effect |
|:---|:-------|
| `"natural"` | No constraint (default) |
| `"dirichlet"` | Fixed value at boundary (`value`, default 0) |
| `"neumann"` | Fixed first derivative (`value`, default 0) |
| `"second_derivative"` | Fixed second derivative (`value`, default 0) |
| `"robin"` | `αu + βu′ = γ` (`alpha`, `beta`, `gamma`) |
| `"periodic"`, `"symmetric"`, `"antisymmetric"`, `"zeros"`, `"exponential"`, `"cauchy"` | See Springsteel docs |

For radar data, `"natural"` (unconstrained) is typically appropriate since the
domain edges are arbitrary analysis boundaries.

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
