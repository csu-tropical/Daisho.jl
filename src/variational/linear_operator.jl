# Abstract linear-operator layer for the variational solver.
#
# Every operator maps a flat input vector to a flat output vector in place and
# provides its exact transpose. Composition (`ChainOp`) and diagonal scaling
# (`DiagonalScaling`) are the only generic building blocks; the spline, filter,
# and observation operators live in their own files.

"""
    AbstractLinearOp

Supertype of the variational solver's linear operators. A concrete operator
`op` of size `(m, n)` implements

- `apply!(y, op, x)`: `y = op·x` (`length(x) == n`, `length(y) == m`),
- `apply_adjoint!(x, op, y)`: `x = opᵀ·y`,
- `Base.size(op)::Tuple{Int,Int}`.

Both applications overwrite their output and must not alias input and output.
"""
abstract type AbstractLinearOp end

"""
    apply!(y, op::AbstractLinearOp, x) -> y

Overwrite `y` with `op·x`.
"""
function apply! end

"""
    apply_adjoint!(x, op::AbstractLinearOp, y) -> x

Overwrite `x` with `opᵀ·y` (the exact transpose of [`apply!`](@ref)).
"""
function apply_adjoint! end

Base.size(op::AbstractLinearOp, d::Integer) = size(op)[d]

# ─────────────────────────────────────────────────────────────────────────────
# ChainOp
# ─────────────────────────────────────────────────────────────────────────────

"""
    ChainOp(ops::AbstractLinearOp...)

Product `ops[1] · ops[2] · … · ops[end]`, applied right to left like matrix
multiplication: `ChainOp(S, D, F)` applies `F`, then `D`, then `S`. The
adjoint applies `Sᵀ`, then `Dᵀ`, then `Fᵀ`. Intermediate buffers are allocated
once at construction: `bufs[k]` (`k = 1:length(ops)-1`) holds the vector
between `ops[k+1]` and `ops[k]`, so `length(bufs[k]) == size(ops[k], 2)`.
Adjacent sizes are checked at construction (`DimensionMismatch`).
"""
struct ChainOp{T<:Tuple} <: AbstractLinearOp
    ops::T
    bufs::Vector{Vector{Float64}}
end

function ChainOp(ops::AbstractLinearOp...)
    isempty(ops) && throw(ArgumentError("ChainOp needs at least one operator"))
    for k in 1:length(ops)-1
        size(ops[k], 2) == size(ops[k+1], 1) || throw(DimensionMismatch(
            "ChainOp: operator $(k) has input size $(size(ops[k], 2)) but operator " *
            "$(k+1) has output size $(size(ops[k+1], 1))"))
    end
    bufs = [zeros(size(ops[k], 2)) for k in 1:length(ops)-1]
    return ChainOp{typeof(ops)}(ops, bufs)
end

Base.size(C::ChainOp) = (size(first(C.ops), 1), size(last(C.ops), 2))

function apply!(y::AbstractVector{Float64}, C::ChainOp, x::AbstractVector{Float64})
    _chain_fwd!(y, C.ops, C.bufs, x)
    return y
end

function apply_adjoint!(x::AbstractVector{Float64}, C::ChainOp, y::AbstractVector{Float64})
    _chain_adj!(x, C.ops, C.bufs, y, 1)
    return x
end

# Tuple recursion keeps every call statically dispatched (no allocation).
_chain_fwd!(y, ops::Tuple{Any}, bufs, x) = apply!(y, ops[1], x)
function _chain_fwd!(y, ops::Tuple, bufs, x)
    t = bufs[length(ops) - 1]
    apply!(t, last(ops), x)
    return _chain_fwd!(y, Base.front(ops), bufs, t)
end

_chain_adj!(x, ops::Tuple{Any}, bufs, y, k) = apply_adjoint!(x, ops[1], y)
function _chain_adj!(x, ops::Tuple, bufs, y, k)
    t = bufs[k]
    apply_adjoint!(t, first(ops), y)
    return _chain_adj!(x, Base.tail(ops), bufs, t, k + 1)
end

# ─────────────────────────────────────────────────────────────────────────────
# DiagonalScaling
# ─────────────────────────────────────────────────────────────────────────────

"""
    DiagonalScaling(sigma::AbstractVector{<:Real}, npts::Integer)
    DiagonalScaling(sigma::AbstractVector, npts::Integer)

Square diagonal operator on a variable-major vector of `length(sigma)`
variables with `npts` points each. Entry `sigma[v]` is either a scalar (the
same scale at every point of variable `v`) or a length-`npts` vector (a
per-point map). All scales must be finite and strictly positive. The operator
is symmetric, so the adjoint equals the forward application.
"""
struct DiagonalScaling <: AbstractLinearOp
    npts::Int
    scalars::Vector{Float64}          # per variable; used when maps[v] is empty
    maps::Vector{Vector{Float64}}     # per variable; empty ⇒ uniform scalar
end

function DiagonalScaling(sigma::AbstractVector, npts::Integer)
    npts >= 1 || throw(ArgumentError("DiagonalScaling: npts must be ≥ 1, got $(npts)"))
    nv = length(sigma)
    nv >= 1 || throw(ArgumentError("DiagonalScaling: at least one variable required"))
    scalars = zeros(nv)
    maps = [Float64[] for _ in 1:nv]
    for (v, s) in enumerate(sigma)
        if s isa Real
            (isfinite(s) && s > 0) || throw(ArgumentError(
                "DiagonalScaling: scale for variable $(v) must be finite and > 0, got $(s)"))
            scalars[v] = Float64(s)
        elseif s isa AbstractVector{<:Real}
            length(s) == npts || throw(ArgumentError(
                "DiagonalScaling: map for variable $(v) has length $(length(s)), expected $(npts)"))
            all(z -> isfinite(z) && z > 0, s) || throw(ArgumentError(
                "DiagonalScaling: map for variable $(v) must be finite and > 0 everywhere"))
            maps[v] = Vector{Float64}(s)
            scalars[v] = NaN
        else
            throw(ArgumentError(
                "DiagonalScaling: entry $(v) must be a scalar or a vector, got $(typeof(s))"))
        end
    end
    return DiagonalScaling(Int(npts), scalars, maps)
end

Base.size(D::DiagonalScaling) = (n = length(D.scalars) * D.npts; (n, n))

function apply!(y::AbstractVector{Float64}, D::DiagonalScaling, x::AbstractVector{Float64})
    n = D.npts
    (length(x) == length(y) == size(D, 1)) || throw(DimensionMismatch(
        "DiagonalScaling: size $(size(D)), x $(length(x)), y $(length(y))"))
    @inbounds for v in eachindex(D.scalars)
        off = (v - 1) * n
        m = D.maps[v]
        if isempty(m)
            s = D.scalars[v]
            @simd for p in 1:n
                y[off + p] = s * x[off + p]
            end
        else
            @simd for p in 1:n
                y[off + p] = m[p] * x[off + p]
            end
        end
    end
    return y
end

apply_adjoint!(x::AbstractVector{Float64}, D::DiagonalScaling, y::AbstractVector{Float64}) =
    apply!(x, D, y)

# ─────────────────────────────────────────────────────────────────────────────
# Dot-product (adjoint) test helper
# ─────────────────────────────────────────────────────────────────────────────

"""
    adjoint_relative_error(op::AbstractLinearOp; rng = nothing) -> Float64

Dot-product test: for random `x`, `y` (standard normal, drawn from `rng` when
given) return `|⟨op x, y⟩ − ⟨x, opᵀ y⟩| / (‖op x‖·‖y‖)`. An exact adjoint gives
a value at round-off level (≲ 1e-14).
"""
function adjoint_relative_error(op::AbstractLinearOp; rng = nothing)
    m, n = size(op)
    x = rng === nothing ? randn(n) : randn(rng, n)
    y = rng === nothing ? randn(m) : randn(rng, m)
    Ax = zeros(m)
    Aty = zeros(n)
    apply!(Ax, op, x)
    apply_adjoint!(Aty, op, y)
    denom = norm(Ax) * norm(y)
    denom == 0 && return abs(dot(x, Aty))
    return abs(dot(Ax, y) - dot(x, Aty)) / denom
end
