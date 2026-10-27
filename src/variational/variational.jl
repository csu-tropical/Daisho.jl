# Variational (3DVAR) wind solver: operator layer, observation operators, cost
# function, solver, state vector, observations, problem, and the file-writing
# pipeline. The public API is exported from `src/Daisho.jl`; the operator layer
# stays unexported (qualified `Daisho.` names, documented in the API reference).

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
include("problem.jl")
include("pipeline.jl")
