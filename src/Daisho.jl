__precompile__()
module Daisho

using Dates, Statistics
using NetCDF, HDF5, NCDatasets
using DataStructures
using NearestNeighbors, Distances
using CoordRefSystems, Unitful
using Springsteel
using JLD2

# Constants
const Reff = 4.0 * 6371000.0 / 3.0

include("netcdf_parameters.jl")
include("parameters.jl")
include("radar.jl")
include("cfradial.jl")
include("cfradial_reader.jl")
include("cfradial_writer.jl")
include("cfradial_validate.jl")
include("cfradial_bridge.jl")
include("SRTM.jl")
include("qualitycontrol.jl")
include("grid_accumulator.jl")
include("gridding.jl")
include("springsteel_adapter.jl")

# Runtime parameter system exports
export DaishoParameters
export MomentParameters, QCParameters, GriddingParameters
export GridParameters, CartesianGridParameters, LatLonGridParameters
export RhiGridParameters, SpectralGridParameters, SpectralBCParameters
export MetadataParameters
export IOParameters

# CfRadial 2.1 data model
export Volume, SweepGroup, Field, FieldMetadata
export Georeference, RadarMonitoring, LidarMonitoring, SpectrumGroup
export RadarParameters, LidarParameters
export RadarCalibration, RadarCalibrationEntry, LidarCalibration
export GeoreferenceCorrection
export add_field!, remove_field!, n_rays, field_names, has_field
export read_cfradial, write_cfradial, update_cfradial
export validate_spec, ValidationReport
export as_legacy_radar, as_volume
export threshold_qc!

# Grid accumulator (multi-sweep / multi-file gridding workflow)
export GridSpec, SweepProvenance, GridAccumulator
export save_accumulator, load_accumulator, merge_accumulators!
export grid_sweep!, finalize_grid, build_grid_spec
export grid_sweep_to_file, finalize_accumulator_file, combine_accumulator_files

end # module Daisho
