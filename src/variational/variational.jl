# Variational (3DVAR) wind solver — operator layer and, in later phases, the
# observation operators, cost function, solver, state vector, and pipeline.
# Nothing is exported yet; the public API is decided in a later phase.

include("linear_operator.jl")
include("pencil.jl")
include("spline_operator.jl")
include("filter_operator.jl")
include("obs_operator.jl")
include("cost_function.jl")
include("solver.jl")
include("state.jl")
include("observations.jl")
include("superobs.jl")

# Later phases: problem.jl, pipeline.jl
