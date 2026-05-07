__precompile__()
module Daisho

using Dates, Statistics
using NetCDF, HDF5, NCDatasets
using DataStructures
using NearestNeighbors, Distances
using CoordRefSystems, Unitful
using Springsteel

# Constants
const Reff = 4.0 * 6371000.0 / 3.0

include("netcdf_parameters.jl")
include("parameters.jl")
include("radar.jl")
include("SRTM.jl")
include("qualitycontrol.jl")
include("gridding.jl")
include("springsteel_adapter.jl")

# Runtime parameter system exports
export DaishoParameters
export MomentParameters, QCParameters, GriddingParameters
export GridParameters, CartesianGridParameters, LatLonGridParameters
export RhiGridParameters, SpectralGridParameters, SpectralBCParameters
export IOParameters

end # module Daisho
