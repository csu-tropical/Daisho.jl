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
include("temperature_profile.jl")
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
include("wind_synthesis.jl")
include("gridding.jl")
include("fhc.jl")
include("rainrate.jl")
include("echo_products.jl")
include("springsteel_adapter.jl")

# Runtime parameter system exports
export DaishoParameters, print_config
export MomentParameters, QCParameters, GriddingParameters
export GridParameters, CartesianGridParameters, LatLonGridParameters
export RhiGridParameters, SpringsteelGridConfig, SpringsteelAxisConfig
export MetadataParameters
export radar_vars
export IOParameters
export SynthesisParameters

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
export GridSpec, SweepProvenance, ScalarGridAccumulator, GridAccumulator
export save_accumulator, load_accumulator, merge_accumulators!
export grid_sweep!, finalize_grid, build_grid_spec
export grid_sweep_to_file, finalize_accumulator_file, combine_accumulator_files

# Multi-Doppler wind synthesis (stage 1)
export WindGridAccumulator, wind_accumulator_dims, SynthesisOutput
export SynthesisFrame, CartesianFrame, component_names, rotation_at
export finalize_wind, build_accumulator
export save_wind_accumulator, load_wind_accumulator, merge_wind_accumulators!
export write_wind_synthesis, write_grid_products

# Hydrometeor identification (FHC) and rain-rate echo products
export csu_fhc_summer, get_mbf_sets_summer, beta_mbf
export FHC_SUMMER_CLASSES, FHC_N_TYPES, DEFAULT_FHC_WEIGHTS
export calc_blended_rain_tropical
export calc_rain_zr, calc_rain_kdp, calc_rain_kdp_zdr, calc_rain_z_zdr
export TemperatureProfile, read_temperature_profile, temperature_celsius
export EchoProductsParameters
export apply_echo_products, add_echo_products!, echo_output_names

end # module Daisho
