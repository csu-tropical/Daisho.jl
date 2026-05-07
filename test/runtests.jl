using Test
using Daisho
using Dates
using NCDatasets
using DataStructures
using NearestNeighbors
using CoordRefSystems
using Unitful
using Rasters
using Springsteel

# Load test helpers
include("test_helpers.jl")

@testset "Daisho.jl" begin
    include("test_parameters.jl")
    include("test_runtime_parameters.jl")
    include("test_radar.jl")
    include("test_qualitycontrol.jl")
    include("test_srtm.jl")
    include("test_gridding.jl")
    include("test_springsteel.jl")
end
