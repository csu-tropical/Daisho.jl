# Variational (3DVAR) wind solver — operator layer and, in later phases, the
# observation operators, cost function, solver, state vector, and pipeline.
# Nothing is exported yet; the public API is decided in a later phase.

include("linear_operator.jl")
include("pencil.jl")
include("spline_operator.jl")
include("filter_operator.jl")

# Later phases: obs_operator.jl, cost_function.jl,
# solver.jl, state.jl, observations.jl, superobs.jl, problem.jl, pipeline.jl
