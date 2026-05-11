# Reading and Writing Radar Data

Daisho.jl works with radar data in CfRadial (CF/Radial) NetCDF format, a standard for weather radar data.

## Reading CfRadial Files

The primary function for reading radar data is [`Daisho.read_cfradial`](@ref):

```julia
# Create a moment dictionary mapping variable names to column indices
moment_dict = Dict("DBZ" => 1, "VEL" => 2, "WIDTH" => 3)

# Read the CfRadial file
volume = Daisho.read_cfradial("path/to/cfradial.nc", moment_dict)
```

The returned `Daisho.radar` struct contains all the data from the file, including:
- Azimuth, elevation, and range arrays
- Platform motion vectors (for moving platforms)
- Nyquist velocity
- Geographic coordinates (latitude, longitude, altitude)
- Sweep information
- All requested radar moments in a 2D array

## The `radar` Struct

The `Daisho.radar` struct is the core data container. Moments are stored as a 2D array where rows correspond to range gates across all rays (flattened) and columns correspond to different radar variables.

## Moment Dictionaries

Daisho uses dictionaries to map moment names to array column indices:

```julia
raw_names = ["DBZ", "ZDR", "KDP", "RHOHV", "VEL", "WIDTH", "PHIDP", "SQI", "SNR"]
qc_names = ["DBZ", "ZDR", "KDP", "RHOHV", "VEL", "WIDTH", "PHIDP", "SQI"]
grid_types = [:linear, :linear, :weighted, :weighted, :weighted, :weighted, :weighted, :weighted]

raw_dict, qc_dict, grid_dict = Daisho.initialize_moment_dictionaries(raw_names, qc_names, grid_types)
```

Grid types determine how moments are interpolated during gridding:
- `:linear` - Convert to linear units before averaging (e.g., reflectivity)
- `:weighted` - Weighted average in native units (e.g., velocity)
- `:nearest` - Nearest-neighbor interpolation

## Writing QC'd CfRadial Files

After quality control, write the QC'd data back to CfRadial format with [`Daisho.write_cfradial`](@ref). The CfRadial 2.1 refactor unified the per-radar variants (SIGMET, single-/dual-pol, P-3) into this single `Volume`-based writer.

## Splitting Sweeps

To work with individual sweeps from a volume scan:

```julia
sweeps = Daisho.split_sweeps(volume)
for sweep in sweeps
    # Process each sweep independently
end
```

## Platform Orientation

For moving platforms (ships, aircraft), retrieve heading, pitch, and roll:

```julia
orientation = Daisho.get_radar_orientation("path/to/cfradial.nc")
# Returns [heading pitch roll] matrix with one row per ray
```
