using Test
using Daisho
using Dates
using NCDatasets
using DataStructures
using NearestNeighbors
using CoordRefSystems
using Rasters
using Springsteel

# Load test helpers
include("test_helpers.jl")

@testset "Daisho.jl" begin
    include("test_parameters.jl")
    include("test_runtime_parameters.jl")
    include("test_reference_density.jl")
    include("test_variational_config.jl")
    include("test_variational_state.jl")
    include("test_radar.jl")
    include("test_qualitycontrol.jl")
    include("test_srtm.jl")
    include("test_gridding.jl")
    include("test_gridded_readers.jl")
    include("test_composite_gridding.jl")
    include("test_range_filter.jl")
    include("test_springsteel.jl")
    include("test_cfradial_types.jl")
    include("test_cfradial_reader.jl")
    include("test_cfradial_writer.jl")
    include("test_cfradial_validate.jl")
    include("test_cfradial_bridge.jl")
    include("test_grid_accumulator.jl")
    include("test_grid_accumulator_workflow.jl")
    include("test_wind_synthesis.jl")
    include("test_variational_operators.jl")
    include("test_variational_filter.jl")
    include("test_variational_solver.jl")
    include("test_variational_observations.jl")
    include("test_variational_solve.jl")
    include("test_variational_pipeline.jl")
    include("test_fhc.jl")
    include("test_rainrate.jl")
    include("test_echo_products.jl")
    include("test_hybrid_scan.jl")
end
