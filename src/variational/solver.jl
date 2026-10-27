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
                     history = true, report_interval = 0, io = stdout)

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
- `report_interval`: print solver progress to `io` — a header (control size,
  observation counts, initial cost split into `J_b` and per-backend `J_o`),
  one line per outer iteration (`‖∇J‖`, `J_k`), a CG line at iteration 1 and
  every `report_interval` CG iterations (`‖r‖`, `‖r‖/‖r₀‖`, model cost `J`,
  `ΔJ`, elapsed time; see `Daisho.cg_model_cost`) and a summary after each
  CG solve. `0` (default) is silent. Reporting never changes the result.
- `io`: destination of the progress lines.
"""
Base.@kwdef struct VarSolverOptions
    max_outer::Int = 1
    max_inner::Int = 200
    inner_tolerance::Float64 = 1e-4
    outer_tolerance::Float64 = 1e-3
    preconditioner::Any = nothing
    history::Bool = true
    report_interval::Int = 0
    io::IO = stdout
end

"""
    cg_model_cost(J_k, rhs, x, r) -> Float64

Cost along the CG path of one outer iteration without extra Hessian products.
For the SPD system `H x = b` (`b = rhs = −∇J(ξ_k)`) with residual
`r = b − H x`, the quadratic model `q(x) = ½ xᵀHx − bᵀx` equals
`−½ xᵀ(b + r)`, so the model cost is `J_k + q(x) = J_k − ½ xᵀ(b + r)`, where
`J_k` is the nonlinear cost at the linearisation point. For linear
observations this is exactly `J(ξ_k + x)`.
"""
cg_model_cost(J_k::Real, rhs::AbstractVector, x::AbstractVector, r::AbstractVector) =
    J_k - 0.5 * (dot(x, rhs) + dot(x, r))

# Total cost and per-backend observation cost at cf.ξ_total (one evaluation).
function _cost_split(cf::IncrementalCostFunction)
    jo = cost_nonlinear_by_backend(cf)
    jb = 0.5 * dot(cf.ξ_total, cf.ξ_total)
    return jb + sum(jo; init = 0.0), jb, jo
end

function _print_cost_split(io::IO, J, jb, jo, backends)
    @printf(io, "J=%.10e  J_b=%.6e", J, jb)
    for (b, j) in zip(backends, jo)
        @printf(io, "  J_o[%s]=%.6e", b.kind, j)
    end
    println(io)
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
    opts.report_interval >= 0 || throw(ArgumentError("VarSolverOptions: report_interval must be ≥ 0"))
    report = opts.report_interval > 0
    io = opts.io
    n = length(cf.ξ_total)
    fill!(cf.ξ_total, 0.0)
    ws = Krylov.CgWorkspace(n, n, Vector{Float64})
    Hop = HessianOperator(cf)
    M = opts.preconditioner === nothing ? I : opts.preconditioner
    ldiv = M isa Factorization
    inner_its = Int[]; inner_ok = Bool[]; inner_res = Vector{Float64}[]
    J0, jb0, jo = _cost_split(cf)
    costs = [J0]
    incs = Float64[]
    converged = false
    k = 0
    backends = cf.obs.backends
    if report
        @printf(io, "solve_incremental!: n=%d controls, nobs=%d", n, nobs(cf.obs))
        isempty(backends) || print(io, " (", join(("$(b.kind) $(nobs(b))" for b in backends), ", "), ")")
        println(io)
        print(io, "  initial  ")
        _print_cost_split(io, J0, jb0, jo, backends)
    end
    for outer in 1:opts.max_outer
        k = outer
        t0 = time()
        relinearize!(cf)
        if report
            J_k = costs[end]
            r0 = norm(cf.rhs)
            @printf(io, "outer %d  ‖∇J‖=%.3e  J=%.6e\n", outer, r0, J_k)
            it = Ref(0); Jprev = Ref(J_k)
            cb = function (w)
                it[] += 1
                if it[] == 1 || it[] % opts.report_interval == 0
                    rn = norm(w.r)
                    Jm = cg_model_cost(J_k, cf.rhs, w.x, w.r)
                    @printf(io, "  cg %5d  ‖r‖=%.3e  ‖r‖/‖r₀‖=%.3e  J=%.6e  ΔJ=%+.3e  t=%.1fs\n",
                            it[], rn, rn / max(r0, floatmin(Float64)), Jm, Jm - Jprev[], time() - t0)
                    Jprev[] = Jm
                end
                return false
            end
            Krylov.cg!(ws, Hop, cf.rhs; M = M, ldiv = ldiv, rtol = opts.inner_tolerance,
                       atol = 0.0, itmax = opts.max_inner, history = opts.history, callback = cb)
        else
            Krylov.cg!(ws, Hop, cf.rhs; M = M, ldiv = ldiv, rtol = opts.inner_tolerance,
                       atol = 0.0, itmax = opts.max_inner, history = opts.history)
        end
        δ = ws.x
        cf.ξ_total .+= δ
        push!(inner_its, ws.stats.niter)
        push!(inner_ok, ws.stats.solved)
        push!(inner_res, opts.history ? copy(ws.stats.residuals) : Float64[])
        δn = norm(δ)
        push!(incs, δn)
        Jn, jbn, jo = _cost_split(cf)
        push!(costs, Jn)
        ratio = δn / max(norm(cf.ξ_total), eps())
        if report
            @printf(io, "  outer %d done: cg=%d  status=\"%s\"  ‖δξ‖=%.3e  ‖δξ‖/‖ξ‖=%.3e  t=%.1fs\n",
                    outer, ws.stats.niter, ws.stats.status, δn, ratio, time() - t0)
            print(io, "  result   ")
            _print_cost_split(io, Jn, jbn, jo, backends)
        end
        if ratio < opts.outer_tolerance
            converged = true
            break
        end
    end
    if !converged && is_linear(cf.obs) && inner_ok[end]
        converged = true
    end
    return VarSolverStats(k, inner_its, inner_ok, inner_res, costs, jo, incs, converged)
end
