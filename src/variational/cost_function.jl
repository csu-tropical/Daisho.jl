# Incremental variational cost function (plan §2):
#
#   J(ξ)  = ½‖ξ‖² + ½ Σ_m r_m ([H_k C ξ]_m − d_m)²,   r_m = 1/σ_m²
#   d     = y − h(a_k),   a_k = a_g + C ξ_total        (current linearisation)
#   ∇J(ξ) = ξ + Cᵀ H_kᵀ R (H_k C ξ − d)
#   Hess  = I + Cᵀ H_kᵀ R H_k C                         (SPD, eigenvalues ≥ 1)
#
# ξ is the control variable on the mish (C = S∘D∘F in production). The
# Gauss–Newton outer loop (solver.jl) solves Hess·δξ = rhs with
# rhs = Cᵀ H_kᵀ R d − ξ_total = −∇J_nonlinear(ξ_total).

"""
    IncrementalCostFunction(C::AbstractLinearOp, obs::ObsSet, a_g::AbstractVector)

Incremental cost function for the control-to-coefficient operator `C`
(size `(ncoef_total, n)`, `ChainOp(S, D, F)` in production), the observations
`obs` (acting on coefficients, `size(obs, 2) == size(C, 1)`), and the
background coefficients `a_g` (e.g. `apply!(a_g, S, x_g)`).

Fields: `C`, `obs`, `a_g`, `ξ_total` (accumulated control increment, starts
at 0), `a_k = a_g + C ξ_total` at the current linearisation, `d = y − h(a_k)`
(innovation), `rhs = Cᵀ H_kᵀ R d − ξ_total`, plus scratch vectors. The
constructor linearises at `ξ_total = 0` ([`relinearize!`](@ref)).

See also: [`hessian_mul!`](@ref), [`cost`](@ref), [`cost_nonlinear`](@ref),
[`gradient!`](@ref), [`HessianOperator`](@ref), [`solve_incremental!`](@ref)
"""
mutable struct IncrementalCostFunction{TC<:AbstractLinearOp}
    C::TC
    obs::ObsSet
    a_g::Vector{Float64}
    ξ_total::Vector{Float64}
    a_k::Vector{Float64}
    d::Vector{Float64}
    rhs::Vector{Float64}
    ctmp::Vector{Float64}
    ytmp1::Vector{Float64}
    ytmp2::Vector{Float64}
    xtmp::Vector{Float64}
end

function IncrementalCostFunction(C::AbstractLinearOp, obs::ObsSet, a_g::AbstractVector{<:Real})
    nc, n = size(C)
    size(obs, 2) == nc || throw(DimensionMismatch(
        "IncrementalCostFunction: C maps to $(nc) coefficients but the ObsSet acts on $(size(obs, 2))"))
    length(a_g) == nc || throw(DimensionMismatch(
        "IncrementalCostFunction: background has $(length(a_g)) coefficients, expected $(nc)"))
    all(isfinite, a_g) || throw(ArgumentError("IncrementalCostFunction: background coefficients must be finite"))
    m = nobs(obs)
    cf = IncrementalCostFunction(C, obs, Vector{Float64}(a_g), zeros(n), zeros(nc), zeros(m),
                                 zeros(n), zeros(nc), zeros(m), zeros(m), zeros(n))
    relinearize!(cf)
    return cf
end

"""
    relinearize!(cf::IncrementalCostFunction) -> cf

Linearise at the current `ξ_total`: `a_k = a_g + C ξ_total`,
`d = y − h(a_k)` (nonlinear backends also update their tangent weights at
`a_k`), `rhs = Cᵀ H_kᵀ R d − ξ_total` (the negative gradient of the
nonlinear cost at `ξ_total`).
"""
function relinearize!(cf::IncrementalCostFunction)
    apply!(cf.ctmp, cf.C, cf.ξ_total)
    cf.a_k .= cf.a_g .+ cf.ctmp
    evaluate!(cf.ytmp1, cf.obs, cf.a_k; linearize = true)
    cf.d .= cf.obs.y .- cf.ytmp1
    cf.ytmp2 .= cf.obs.rinv .* cf.d
    apply_Ht!(cf.ctmp, cf.obs, cf.ytmp2)
    apply_adjoint!(cf.rhs, cf.C, cf.ctmp)
    cf.rhs .-= cf.ξ_total
    return cf
end

"""
    hessian_mul!(y, cf::IncrementalCostFunction, p) -> y

`y = p + Cᵀ H_kᵀ R H_k C p` (the Gauss–Newton Hessian at the current
linearisation). `y` and `p` must not alias. Allocation-free after warm-up
when `C` and the `ObsSet` were built with `nchunks = 1`.
"""
function hessian_mul!(y::AbstractVector{Float64}, cf::IncrementalCostFunction,
                      p::AbstractVector{Float64})
    apply!(cf.ctmp, cf.C, p)
    apply_H!(cf.ytmp1, cf.obs, cf.ctmp)
    r = cf.obs.rinv
    @inbounds @simd for m in eachindex(r)
        cf.ytmp1[m] *= r[m]
    end
    apply_Ht!(cf.ctmp, cf.obs, cf.ytmp1)
    apply_adjoint!(y, cf.C, cf.ctmp)
    @inbounds @simd for i in eachindex(y)
        y[i] += p[i]
    end
    return y
end

# ytmp1 = H_k C ξ − d
function _model_residual!(cf::IncrementalCostFunction, ξ::AbstractVector{Float64})
    length(ξ) == length(cf.ξ_total) || throw(DimensionMismatch(
        "control vector has length $(length(ξ)), expected $(length(cf.ξ_total))"))
    apply!(cf.ctmp, cf.C, ξ)
    apply_H!(cf.ytmp1, cf.obs, cf.ctmp)
    cf.ytmp1 .-= cf.d
    return cf.ytmp1
end

"""
    cost(cf::IncrementalCostFunction, ξ) -> Float64

Quadratic model of the cost at the current linearisation:
`J(ξ) = ½‖ξ‖² + ½ Σ r (H_k C ξ − d)²` (ξ is an increment relative to
`ξ_total`).
"""
function cost(cf::IncrementalCostFunction, ξ::AbstractVector{Float64})
    res = _model_residual!(cf, ξ)
    r = cf.obs.rinv
    Jo = 0.0
    @inbounds for m in eachindex(r)
        Jo += r[m] * res[m]^2
    end
    return 0.5 * dot(ξ, ξ) + 0.5 * Jo
end

"""
    gradient!(g, cf::IncrementalCostFunction, ξ) -> g

Gradient of the quadratic model [`cost`](@ref): `g = ξ + Cᵀ H_kᵀ R (H_k C ξ − d)`.
`g` and `ξ` must not alias.
"""
function gradient!(g::AbstractVector{Float64}, cf::IncrementalCostFunction,
                   ξ::AbstractVector{Float64})
    res = _model_residual!(cf, ξ)
    res .*= cf.obs.rinv
    apply_Ht!(cf.ctmp, cf.obs, res)
    apply_adjoint!(g, cf.C, cf.ctmp)
    g .+= ξ
    return g
end

# ytmp1 = y − h(a_g + C ξ)   (true, possibly nonlinear h; no linearisation)
function _nonlinear_residual!(cf::IncrementalCostFunction, ξ::AbstractVector{Float64})
    length(ξ) == length(cf.ξ_total) || throw(DimensionMismatch(
        "control vector has length $(length(ξ)), expected $(length(cf.ξ_total))"))
    apply!(cf.ctmp, cf.C, ξ)
    cf.ctmp .+= cf.a_g
    evaluate!(cf.ytmp1, cf.obs, cf.ctmp)
    cf.ytmp1 .= cf.obs.y .- cf.ytmp1
    return cf.ytmp1
end

"""
    cost_nonlinear(cf::IncrementalCostFunction, ξ = cf.ξ_total) -> Float64

Cost with the true (possibly nonlinear) observation operator at the total
control `ξ`: `½‖ξ‖² + ½ Σ r (y − h(a_g + C ξ))²`. Does not change the
linearisation.
"""
function cost_nonlinear(cf::IncrementalCostFunction, ξ::AbstractVector{Float64} = cf.ξ_total)
    res = _nonlinear_residual!(cf, ξ)
    r = cf.obs.rinv
    Jo = 0.0
    @inbounds for m in eachindex(r)
        Jo += r[m] * res[m]^2
    end
    return 0.5 * dot(ξ, ξ) + 0.5 * Jo
end

"""
    cost_nonlinear_by_backend(cf::IncrementalCostFunction, ξ = cf.ξ_total) -> Vector{Float64}

Per-backend observation cost `½ Σ r (y − h(a_g + C ξ))²` (see
[`cost_by_backend`](@ref)).
"""
cost_nonlinear_by_backend(cf::IncrementalCostFunction, ξ::AbstractVector{Float64} = cf.ξ_total) =
    cost_by_backend(cf.obs, _nonlinear_residual!(cf, ξ))

"""
    control_increment(cf::IncrementalCostFunction) -> Vector{Float64}

The accumulated control increment `ξ_total` (the field itself, not a copy).
"""
control_increment(cf::IncrementalCostFunction) = cf.ξ_total

"""
    analysis_coefficients!(a, cf::IncrementalCostFunction) -> a

Analysis coefficients `a = a_g + C ξ_total`.
"""
function analysis_coefficients!(a::AbstractVector{Float64}, cf::IncrementalCostFunction)
    apply!(a, cf.C, cf.ξ_total)
    a .+= cf.a_g
    return a
end

"""
    HessianOperator(cf::IncrementalCostFunction)

Matrix-free `n × n` Gauss–Newton Hessian `I + Cᵀ H_kᵀ R H_k C` for Krylov.jl:
implements `Base.size`, `Base.eltype` (`Float64`) and
`LinearAlgebra.mul!(y, H, x)` ([`hessian_mul!`](@ref)).
"""
struct HessianOperator{TC}
    cf::IncrementalCostFunction{TC}
end

Base.size(H::HessianOperator) = (n = length(H.cf.ξ_total); (n, n))
Base.size(H::HessianOperator, d::Integer) = size(H)[d]
Base.eltype(::HessianOperator) = Float64
LinearAlgebra.mul!(y::AbstractVector{Float64}, H::HessianOperator, x::AbstractVector{Float64}) =
    hessian_mul!(y, H.cf, x)
