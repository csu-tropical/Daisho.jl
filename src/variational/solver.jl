# Gauss–Newton outer loop with a Krylov CG inner solve (plan §2):
#
#   ξ_total = 0
#   for k = 1:max_outer
#       a_k = a_g + C ξ_total;  d = y − h(a_k);  linearise nonlinear obs at a_k
#       rhs = Cᵀ H_kᵀ R d − ξ_total                  (= −∇J(ξ_total))
#       CG:  (I + Cᵀ H_kᵀ R H_k C) δξ = rhs           (x0 = 0)
#       ξ_total += δξ
#       stop if ‖δξ‖ / max(‖ξ_total‖, eps) < outer_tolerance
#   end

"""
    VarSolverOptions(; max_outer = 1, max_inner = 200, inner_tolerance = 1e-4,
                     outer_tolerance = 1e-3, preconditioner = nothing,
                     history = true, verbose = false)

Options of [`solve_incremental!`](@ref).

- `max_outer`: Gauss–Newton (outer) iterations; 1 is exact for linear obs.
- `max_inner`: CG iteration cap per outer iteration.
- `inner_tolerance`: CG relative residual tolerance (`rtol`, relative to
  `‖rhs‖`; `atol = 0`).
- `outer_tolerance`: stop when `‖δξ‖ / max(‖ξ_total‖, eps) < outer_tolerance`.
- `preconditioner`: passed to `Krylov.cg!` as `M`. Krylov's convention: `M`
  approximates the INVERSE Hessian and is applied with `mul!`, except that a
  `LinearAlgebra.Factorization` (an approximation of the Hessian itself) is
  applied with `ldiv!` (`ldiv = true`). `nothing` ⇒ no preconditioner.
  Untested hook.
- `history`: keep the CG residual norms of every inner solve.
- `verbose`: log one line per outer iteration.
"""
Base.@kwdef struct VarSolverOptions
    max_outer::Int = 1
    max_inner::Int = 200
    inner_tolerance::Float64 = 1e-4
    outer_tolerance::Float64 = 1e-3
    preconditioner::Any = nothing
    history::Bool = true
    verbose::Bool = false
end

"""
    VarSolverStats

Result diagnostics of [`solve_incremental!`](@ref):

- `outer_iterations`: outer iterations performed;
- `inner_iterations[k]`: CG iterations of outer iteration `k`;
- `inner_converged[k]`: whether that CG solve met its tolerance;
- `inner_residuals[k]`: CG residual norms (empty unless `history`);
- `cost_history`: nonlinear cost `J(ξ_total)`; entry 1 is the initial cost at
  `ξ = 0`, entry `k + 1` the cost after outer iteration `k`;
- `cost_by_backend`: per-backend observation cost at the final state;
- `increment_norms[k]`: `‖δξ‖` of outer iteration `k`;
- `converged`: the outer tolerance was met, or every observation is linear
  and the last CG solve converged (one Gauss–Newton step is then exact).
"""
struct VarSolverStats
    outer_iterations::Int
    inner_iterations::Vector{Int}
    inner_converged::Vector{Bool}
    inner_residuals::Vector{Vector{Float64}}
    cost_history::Vector{Float64}
    cost_by_backend::Vector{Float64}
    increment_norms::Vector{Float64}
    converged::Bool
end

"""
    solve_incremental!(cf::IncrementalCostFunction, opts = VarSolverOptions()) -> VarSolverStats

Minimise the incremental cost by Gauss–Newton outer iterations, each solving
the linear SPD system `(I + CᵀH_kᵀRH_kC) δξ = rhs` with `Krylov.cg!` (one
reused `CgWorkspace`, `x0 = 0`). Resets and then accumulates `cf.ξ_total`;
on return `cf.a_k`, `cf.d`, `cf.rhs` refer to the LAST linearisation (not
the final state). Use [`analysis_coefficients!`](@ref) for `a_g + C ξ_total`.
"""
function solve_incremental!(cf::IncrementalCostFunction, opts::VarSolverOptions = VarSolverOptions())
    opts.max_outer >= 1 || throw(ArgumentError("VarSolverOptions: max_outer must be ≥ 1"))
    opts.max_inner >= 1 || throw(ArgumentError("VarSolverOptions: max_inner must be ≥ 1"))
    opts.inner_tolerance > 0 || throw(ArgumentError("VarSolverOptions: inner_tolerance must be > 0"))
    opts.outer_tolerance > 0 || throw(ArgumentError("VarSolverOptions: outer_tolerance must be > 0"))
    n = length(cf.ξ_total)
    fill!(cf.ξ_total, 0.0)
    ws = Krylov.CgWorkspace(n, n, Vector{Float64})
    Hop = HessianOperator(cf)
    M = opts.preconditioner === nothing ? I : opts.preconditioner
    ldiv = M isa Factorization
    inner_its = Int[]; inner_ok = Bool[]; inner_res = Vector{Float64}[]
    costs = [cost_nonlinear(cf)]
    incs = Float64[]
    converged = false
    k = 0
    for outer in 1:opts.max_outer
        k = outer
        relinearize!(cf)
        Krylov.cg!(ws, Hop, cf.rhs; M = M, ldiv = ldiv, rtol = opts.inner_tolerance,
                   atol = 0.0, itmax = opts.max_inner, history = opts.history)
        δ = ws.x
        cf.ξ_total .+= δ
        push!(inner_its, ws.stats.niter)
        push!(inner_ok, ws.stats.solved)
        push!(inner_res, opts.history ? copy(ws.stats.residuals) : Float64[])
        δn = norm(δ)
        push!(incs, δn)
        push!(costs, cost_nonlinear(cf))
        ratio = δn / max(norm(cf.ξ_total), eps())
        opts.verbose && @info "solve_incremental! outer $(outer)" cg_iterations = ws.stats.niter cg_status = ws.stats.status increment = δn relative_increment = ratio cost = costs[end]
        if ratio < opts.outer_tolerance
            converged = true
            break
        end
    end
    if !converged && is_linear(cf.obs) && inner_ok[end]
        converged = true
    end
    jb = cost_nonlinear_by_backend(cf)
    return VarSolverStats(k, inner_its, inner_ok, inner_res, costs, jb, incs, converged)
end
