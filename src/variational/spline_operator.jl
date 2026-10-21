# Spline projection (S = SA∘SB per axis) and evaluation (tensor SI) operators
# with exact adjoints, built on Springsteel's 1-D cubic B-spline primitives but
# owning all scratch so they are allocation-free and thread-safe (Springsteel's
# SAtransform!/SBtransform!/gridTransform! mutate buffers shared across
# pencils and must never be called from here).
#
# Per axis:  SB = E0ᵀ·diag(w)      (w_i = DX·quadweights[mu(i)], Springsteel _sb_matrix)
#            SA = Γᵀ(Γ(P+Q)Γᵀ)⁻¹Γ   (symmetric ⇒ SAᵀ = SA)
#            SI = E0 (value) / E1 (first derivative); coefficients after SA
#                 already satisfy the BCs, so SI needs no Γ.

const _SC = Springsteel.CubicBSpline

"""
    AxisSpline(spline::Springsteel.CubicBSpline.Spline1D; points = nothing)

Read-only, allocation-free 1-D cubic B-spline tables for one axis of one
variable, built from a template `Spline1D` of a Springsteel grid. Holds, per
evaluation point `i`, the first of the four coefficient indices with support
(`first_coef[i]`), the basis values `Φ0[s, i]` and first derivatives
`Φ1[s, i]` (`s = 1:4`), the SB quadrature weights `w`, and shared references
to the spline's Γ (`γ`), `(Γ(P+Q)Γᵀ)` factor, and full `(P+Q)` matrix (`pq`).

By default the evaluation points are the spline's mish (quadrature) points.
`points` (any coordinates inside `[xmin, xmax]`, e.g.
[`node_coordinates`](@ref)) builds the tables at those locations instead;
such an axis supports [`eval!`](@ref)/[`scatter!`](@ref) (evaluation and its
adjoint) but not the quadrature transforms (`w` is zero).

Throws `ArgumentError` for inhomogeneous boundary conditions (R3X, R1T1X),
whose boundary data live in `ahat` and are not supported by the variational
operators.

See also: [`eval!`](@ref), [`scatter!`](@ref), [`sb!`](@ref), [`sbt!`](@ref), [`sa!`](@ref)
"""
struct AxisSpline
    sp::_SC.SplineParameters
    Dim::Int
    bDim::Int
    Minterior::Int
    first_coef::Vector{Int32}
    Φ0::Matrix{Float64}
    Φ1::Matrix{Float64}
    w::Vector{Float64}
    γ::_SC.GammaBC
    factor::Union{_SC.BandedCholesky3,_SC.DenseSplineFactor}
    pq::Symmetric{Float64,Matrix{Float64}}
    mishPoints::Vector{Float64}
end

function AxisSpline(spline::_SC.Spline1D;
                    points::Union{Nothing,AbstractVector{<:Real}} = nothing)
    sp = spline.params
    _SC._has_ahat(sp) && throw(ArgumentError(
        "AxisSpline: inhomogeneous boundary conditions (R3X / R1T1X, BCL=$(sp.BCL), " *
        "BCR=$(sp.BCR)) carry boundary data in `ahat` and are not supported by the " *
        "variational spline operators"))
    bDim = sp.bDim
    if points === nothing
        Dim = sp.mishDim
        mubar = sp.mubar
        length(spline.mishPoints) == Dim || throw(DimensionMismatch(
            "AxisSpline: spline has $(length(spline.mishPoints)) mish points, expected $(Dim)"))
        pts = Vector{Float64}(spline.mishPoints)
    else
        Dim = length(points)
        Dim >= 1 || throw(ArgumentError("AxisSpline: `points` must be non-empty"))
        mubar = 0
        pts = Vector{Float64}(points)
    end
    first_coef = Vector{Int32}(undef, Dim)
    Φ0 = zeros(4, Dim)
    Φ1 = zeros(4, Dim)
    w = zeros(Dim)
    for i in 1:Dim
        x = pts[i]
        if points === nothing
            mc = (i - 1) ÷ mubar          # 0-based cell
            mu = (i - 1) % mubar + 1      # quadrature point within the cell
            w[i] = sp.DX * spline.quadweights[mu]
        else
            (sp.xmin <= x <= sp.xmax) || throw(DomainError(x,
                "AxisSpline: point $(i) lies outside the spline domain [$(sp.xmin), $(sp.xmax)]"))
            # The last node (x == xmax) belongs to the final cell, so its four
            # coefficients stay inside 1:bDim.
            mc = min(floor(Int, (x - sp.xmin) * sp.DXrecip), sp.num_cells - 1)
        end
        first_coef[i] = mc + 1        # coefficient index of node m = mc - 1
        for s in 1:4
            m = mc + s - 2
            Φ0[s, i] = _SC.basis(sp, m, x, 0)
            Φ1[s, i] = _SC.basis(sp, m, x, 1)
        end
    end
    γ = spline.gammaBC
    return AxisSpline(sp, Dim, bDim, size(γ, 1), first_coef, Φ0, Φ1, w, γ,
                      spline.pqFactor, spline.pq, pts)
end

"""
    node_coordinates(spline::Spline1D) -> Vector{Float64}

The B-spline node (cell-edge) coordinates of a Springsteel axis,
`LinRange(xmin, xmax, num_cells + 1)` — the `:nodes` evaluation lattice of
[`SplineEvaluation`](@ref) and the default regular output grid.
"""
node_coordinates(spline::_SC.Spline1D) =
    collect(LinRange(spline.params.xmin, spline.params.xmax, spline.params.num_cells + 1))

"""
    eval!(u, ax::AxisSpline, a, Φ) -> u

Evaluate the spline with coefficients `a` (length `bDim`) at the mish points:
`u[i] = Σ_{s=1:4} Φ[s,i]·a[first_coef[i]+s-1]`. Pass `ax.Φ0` for values or
`ax.Φ1` for first derivatives. No Γ: BCs already live in `a`.
"""
function eval!(u::AbstractVector{Float64}, ax::AxisSpline,
               a::AbstractVector{Float64}, Φ::Matrix{Float64})
    length(u) == ax.Dim && length(a) == ax.bDim || throw(DimensionMismatch(
        "eval!: length(u)=$(length(u)) (Dim $(ax.Dim)), length(a)=$(length(a)) (bDim $(ax.bDim))"))
    fc = ax.first_coef
    @inbounds for i in 1:ax.Dim
        c = Int(fc[i])
        u[i] = Φ[1, i] * a[c] + Φ[2, i] * a[c+1] + Φ[3, i] * a[c+2] + Φ[4, i] * a[c+3]
    end
    return u
end

"""
    scatter!(a, ax::AxisSpline, u, Φ) -> a

Exact transpose of [`eval!`](@ref): `a .= 0; a[first_coef[i]+s-1] += Φ[s,i]·u[i]`.
"""
function scatter!(a::AbstractVector{Float64}, ax::AxisSpline,
                  u::AbstractVector{Float64}, Φ::Matrix{Float64})
    length(u) == ax.Dim && length(a) == ax.bDim || throw(DimensionMismatch(
        "scatter!: length(u)=$(length(u)) (Dim $(ax.Dim)), length(a)=$(length(a)) (bDim $(ax.bDim))"))
    fill!(a, 0.0)
    fc = ax.first_coef
    @inbounds for i in 1:ax.Dim
        c = Int(fc[i])
        ui = u[i]
        a[c]   += Φ[1, i] * ui
        a[c+1] += Φ[2, i] * ui
        a[c+2] += Φ[3, i] * ui
        a[c+3] += Φ[4, i] * ui
    end
    return a
end

"""
    sb!(b, ax::AxisSpline, u, scratch) -> b

SB transform `b = E0ᵀ·(w .* u)` (equal to Springsteel's `_sb_matrix * u`).
`scratch` must have length ≥ `Dim`.
"""
function sb!(b::AbstractVector{Float64}, ax::AxisSpline,
             u::AbstractVector{Float64}, scratch::AbstractVector{Float64})
    length(u) == ax.Dim && length(scratch) >= ax.Dim || throw(DimensionMismatch(
        "sb!: length(u)=$(length(u)), length(scratch)=$(length(scratch)), Dim $(ax.Dim)"))
    s = view(scratch, 1:ax.Dim)
    w = ax.w
    @inbounds @simd for i in 1:ax.Dim
        s[i] = w[i] * u[i]
    end
    return scatter!(b, ax, s, ax.Φ0)
end

"""
    sbt!(u, ax::AxisSpline, b) -> u

Exact transpose of [`sb!`](@ref): `u = w .* (E0·b)`.
"""
function sbt!(u::AbstractVector{Float64}, ax::AxisSpline, b::AbstractVector{Float64})
    eval!(u, ax, b, ax.Φ0)
    w = ax.w
    @inbounds @simd for i in 1:ax.Dim
        u[i] *= w[i]
    end
    return u
end

"""
    sa!(a, ax::AxisSpline, b, Min, Mout) -> a

SA transform `a = Γᵀ(Γ(P+Q)Γᵀ)⁻¹Γ·b` (symmetric, so it is its own adjoint).
`Min` and `Mout` are distinct scratch vectors of length ≥ `Minterior`. `a` and
`b` may alias.
"""
function sa!(a::AbstractVector{Float64}, ax::AxisSpline, b::AbstractVector{Float64},
             Min::AbstractVector{Float64}, Mout::AbstractVector{Float64})
    n = ax.Minterior
    length(Min) >= n && length(Mout) >= n || throw(DimensionMismatch(
        "sa!: interior scratch shorter than Minterior $(n)"))
    mi = view(Min, 1:n)
    mo = view(Mout, 1:n)
    mul!(mi, ax.γ, b)
    ldiv!(mo, ax.factor, mi)
    mul!(a, ax.γ', mo)
    return a
end

# ─────────────────────────────────────────────────────────────────────────────
# Pencil kernels (callable structs keep sweeps type-stable and closure-free)
# ─────────────────────────────────────────────────────────────────────────────

struct _ProjFwdKernel; ax::AxisSpline; end
function (k::_ProjFwdKernel)(out, in, sc::PencilScratch)
    b = view(sc.tmp, 1:k.ax.bDim)
    sb!(b, k.ax, in, sc.extra)
    sa!(out, k.ax, b, sc.Min, sc.Mout)
    return nothing
end

struct _ProjAdjKernel; ax::AxisSpline; end
function (k::_ProjAdjKernel)(out, in, sc::PencilScratch)
    b = view(sc.tmp, 1:k.ax.bDim)
    sa!(b, k.ax, in, sc.Min, sc.Mout)
    sbt!(out, k.ax, b)
    return nothing
end

struct _EvalKernel; ax::AxisSpline; Φ::Matrix{Float64}; end
(k::_EvalKernel)(out, in, sc::PencilScratch) = (eval!(out, k.ax, in, k.Φ); nothing)

struct _ScatterKernel; ax::AxisSpline; Φ::Matrix{Float64}; end
(k::_ScatterKernel)(out, in, sc::PencilScratch) = (scatter!(out, k.ax, in, k.Φ); nothing)

struct _PQKernel; ax::AxisSpline; end
(k::_PQKernel)(out, in, sc::PencilScratch) = (mul!(out, k.ax.pq, in); nothing)

# ─────────────────────────────────────────────────────────────────────────────
# Grid geometry shared by the 3-D operators
# ─────────────────────────────────────────────────────────────────────────────

# Dimensions of an RRR state grid plus the per-(axis, var) AxisSplines.
# axes[1, v] = k-axis, axes[2, v] = j-axis, axes[3, v] = i-axis (sweep order).
# `iDim/jDim/kDim` are the EVALUATION lattice sizes: the mish (`lattice =
# :mish`, the grid's quadrature points) or the B-spline nodes (`:nodes`,
# `cells + 1` per axis); the coefficient sizes `b_*` are the same for both.
struct _SplineGridGeometry
    iDim::Int; jDim::Int; kDim::Int
    b_iDim::Int; b_jDim::Int; b_kDim::Int
    columns::Vector{Int}              # grid.params.vars column of each variable
    axes::Matrix{AxisSpline}
end

function _spline_grid_geometry(grid, varnames::AbstractVector{<:AbstractString};
                               lattice::Symbol = :mish)
    lattice in (:mish, :nodes) || throw(ArgumentError(
        "variational spline operators: lattice must be :mish or :nodes, got :$(lattice)"))
    gp = grid.params
    gp.geometry == "RRR" || throw(ArgumentError(
        "variational spline operators need an RRR Springsteel grid, got geometry $(gp.geometry)"))
    isempty(varnames) && throw(ArgumentError("variational spline operators: no variables given"))
    allunique(varnames) || throw(ArgumentError("variational spline operators: duplicate variable names $(varnames)"))
    columns = Int[]
    for n in varnames
        haskey(gp.vars, n) || throw(ArgumentError(
            "variable `$(n)` is not on the grid; grid variables: $(join(sort(collect(keys(gp.vars))), ", "))"))
        push!(columns, gp.vars[n])
    end
    nv = length(varnames)
    axes = Matrix{AxisSpline}(undef, 3, nv)
    mk(spline) = lattice === :mish ? AxisSpline(spline) :
                                     AxisSpline(spline; points = node_coordinates(spline))
    for (v, col) in enumerate(columns)
        axes[1, v] = mk(grid.kbasis.data[1, 1, col])
        axes[2, v] = mk(grid.jbasis.data[1, 1, col])
        axes[3, v] = mk(grid.ibasis.data[1, 1, col])
        (axes[1, v].bDim, axes[2, v].bDim, axes[3, v].bDim) == (gp.b_kDim, gp.b_jDim, gp.b_iDim) &&
        (lattice === :nodes || (axes[1, v].Dim, axes[2, v].Dim, axes[3, v].Dim) == (gp.kDim, gp.jDim, gp.iDim)) &&
        (axes[1, v].Dim, axes[2, v].Dim, axes[3, v].Dim) == (axes[1, 1].Dim, axes[2, 1].Dim, axes[3, 1].Dim) ||
            throw(DimensionMismatch(
            "variational spline operators: template spline sizes for `$(varnames[v])` disagree with the grid"))
    end
    return _SplineGridGeometry(axes[3, 1].Dim, axes[2, 1].Dim, axes[1, 1].Dim,
                               gp.b_iDim, gp.b_jDim, gp.b_kDim, columns, axes)
end

_nmish(g::_SplineGridGeometry) = g.iDim * g.jDim * g.kDim   # lattice points (mish or nodes)
const _npoints = _nmish
_ncoef(g::_SplineGridGeometry) = g.b_iDim * g.b_jDim * g.b_kDim
_nvars(g::_SplineGridGeometry) = size(g.axes, 2)

function _geometry_pool(g::_SplineGridGeometry, nchunks::Integer)
    maxlen = max(g.iDim, g.jDim, g.kDim, g.b_iDim, g.b_jDim, g.b_kDim)
    maxint = maximum(ax.Minterior for ax in g.axes)
    return scratch_pool(nchunks, maxlen, maxint)
end

_varrange(v::Int, n::Int) = ((v - 1) * n + 1):(v * n)

# ─────────────────────────────────────────────────────────────────────────────
# SplineProjection  S : mish → coefficients
# ─────────────────────────────────────────────────────────────────────────────

"""
    SplineProjection(grid, varnames::Vector{String}; nchunks = Threads.nthreads())

Tensor-product spline projection `S` (per axis `SA∘SB`, applied along k, then
j, then i) from a variable-major mish vector (per variable `(kDim, jDim, iDim)`
column-major) to a variable-major coefficient vector (per variable
`(b_kDim, b_jDim, b_iDim)` column-major). `varnames` selects and orders the
variables among `grid.params.vars`. The resulting coefficients satisfy each
variable's boundary conditions exactly. `apply_adjoint!` applies the exact
transpose `Sᵀ` (i, then j, then k; per pencil `sa!` then `sbt!`).

Pencil sweeps run in `nchunks` tasks; results do not depend on `nchunks`.
"""
struct SplineProjection <: AbstractLinearOp
    geom::_SplineGridGeometry
    buf1::Vector{Float64}     # (b_kDim, jDim, iDim)
    buf2::Vector{Float64}     # (b_kDim, b_jDim, iDim)
    pool::Vector{PencilScratch}
end

function SplineProjection(grid, varnames::AbstractVector{<:AbstractString};
                          nchunks::Integer = Threads.nthreads())
    g = _spline_grid_geometry(grid, varnames)
    return SplineProjection(g, zeros(g.b_kDim * g.jDim * g.iDim),
                            zeros(g.b_kDim * g.b_jDim * g.iDim), _geometry_pool(g, nchunks))
end

Base.size(S::SplineProjection) = (_nvars(S.geom) * _ncoef(S.geom), _nvars(S.geom) * _nmish(S.geom))

function apply!(a::AbstractVector{Float64}, S::SplineProjection, x::AbstractVector{Float64})
    g = S.geom
    (length(a), length(x)) == size(S) || throw(DimensionMismatch(
        "SplineProjection apply!: size $(size(S)), a $(length(a)), x $(length(x))"))
    nm, nc = _nmish(g), _ncoef(g)
    dM  = (g.kDim,   g.jDim,   g.iDim)
    d1  = (g.b_kDim, g.jDim,   g.iDim)
    d2  = (g.b_kDim, g.b_jDim, g.iDim)
    dC  = (g.b_kDim, g.b_jDim, g.b_iDim)
    for v in 1:_nvars(g)
        xv = view(x, _varrange(v, nm))
        av = view(a, _varrange(v, nc))
        sweep_axis!(_ProjFwdKernel(g.axes[1, v]), S.buf1, xv,     1, dM, d1, S.pool)
        sweep_axis!(_ProjFwdKernel(g.axes[2, v]), S.buf2, S.buf1, 2, d1, d2, S.pool)
        sweep_axis!(_ProjFwdKernel(g.axes[3, v]), av,     S.buf2, 3, d2, dC, S.pool)
    end
    return a
end

function apply_adjoint!(x::AbstractVector{Float64}, S::SplineProjection, a::AbstractVector{Float64})
    g = S.geom
    (length(a), length(x)) == size(S) || throw(DimensionMismatch(
        "SplineProjection apply_adjoint!: size $(size(S)), a $(length(a)), x $(length(x))"))
    nm, nc = _nmish(g), _ncoef(g)
    dM  = (g.kDim,   g.jDim,   g.iDim)
    d1  = (g.b_kDim, g.jDim,   g.iDim)
    d2  = (g.b_kDim, g.b_jDim, g.iDim)
    dC  = (g.b_kDim, g.b_jDim, g.b_iDim)
    for v in 1:_nvars(g)
        xv = view(x, _varrange(v, nm))
        av = view(a, _varrange(v, nc))
        sweep_axis!(_ProjAdjKernel(g.axes[3, v]), S.buf2, av,     3, dC, d2, S.pool)
        sweep_axis!(_ProjAdjKernel(g.axes[2, v]), S.buf1, S.buf2, 2, d2, d1, S.pool)
        sweep_axis!(_ProjAdjKernel(g.axes[1, v]), xv,     S.buf1, 1, d1, dM, S.pool)
    end
    return x
end

# ─────────────────────────────────────────────────────────────────────────────
# SplineEvaluation  E : coefficients → requested (var, slot) mish slabs
# ─────────────────────────────────────────────────────────────────────────────

"""
    SplineEvaluation(grid, varnames::Vector{String}, slots::Vector{Tuple{Int,Int}};
                     lattice = :mish, nchunks = Threads.nthreads())

Tensor-product spline evaluation from a variable-major coefficient vector (as
produced by [`SplineProjection`](@ref) for the same `varnames`) to the
requested `(var_index, slot)` slabs on the evaluation `lattice`,
concatenated in the order given. `var_index` indexes `varnames`; `slot` is
`1 = value, 2 = ∂/∂i (x), 3 = ∂/∂j (y), 4 = ∂/∂k (z)`. `apply_adjoint!`
scatters the slabs back through the same branches and accumulates into the
(zeroed) coefficient vector.

`lattice = :mish` evaluates at the grid's quadrature points (each slab has
`iDim·jDim·kDim` entries in the mish layout, k fastest); `lattice = :nodes`
evaluates at the B-spline nodes `LinRange(min, max, cells + 1)` of every
axis ([`node_coordinates`](@ref); `(cells_k+1)·(cells_j+1)·(cells_i+1)`
entries per slab, same k-fastest layout) — the default regular output grid
and the mass-continuity lattice of `mass_continuity_lattice = "nodes"`.
[`lattice_dims`](@ref) gives the slab shape.

See also: [`slab_view`](@ref)
"""
struct SplineEvaluation <: AbstractLinearOp
    geom::_SplineGridGeometry
    lattice::Symbol
    slots::Vector{Tuple{Int,Int}}
    slotpos::Matrix{Int}      # (4, nvars): slab index of (slot, var), 0 if not requested
    A0::Vector{Float64}       # (b_kDim, b_jDim, iDim)  i-axis value branch
    A1::Vector{Float64}       # (b_kDim, b_jDim, iDim)  i-axis derivative branch
    B00::Vector{Float64}      # (b_kDim, jDim, iDim)    value in i and j
    B01::Vector{Float64}      # (b_kDim, jDim, iDim)    value in i, ∂j
    B10::Vector{Float64}      # (b_kDim, jDim, iDim)    ∂i, value in j
    T::Vector{Float64}        # accumulation scratch (largest intermediate)
    pool::Vector{PencilScratch}
end

function SplineEvaluation(grid, varnames::AbstractVector{<:AbstractString},
                          slots::AbstractVector{<:Tuple{Integer,Integer}};
                          lattice::Symbol = :mish, nchunks::Integer = Threads.nthreads())
    g = _spline_grid_geometry(grid, varnames; lattice = lattice)
    nv = _nvars(g)
    isempty(slots) && throw(ArgumentError("SplineEvaluation: no slots requested"))
    slotpos = zeros(Int, 4, nv)
    for (k, (v, s)) in enumerate(slots)
        1 <= v <= nv || throw(ArgumentError(
            "SplineEvaluation: variable index $(v) out of range 1:$(nv)"))
        1 <= s <= 4 || throw(ArgumentError(
            "SplineEvaluation: slot $(s) must be 1 (value), 2 (∂i), 3 (∂j) or 4 (∂k)"))
        slotpos[s, v] == 0 || throw(ArgumentError(
            "SplineEvaluation: slot ($(v), $(s)) requested more than once"))
        slotpos[s, v] = k
    end
    nA = g.b_kDim * g.b_jDim * g.iDim
    nB = g.b_kDim * g.jDim * g.iDim
    nT = max(nA, nB, _ncoef(g))
    return SplineEvaluation(g, lattice, [(Int(v), Int(s)) for (v, s) in slots], slotpos,
                            zeros(nA), zeros(nA), zeros(nB), zeros(nB), zeros(nB),
                            zeros(nT), _geometry_pool(g, nchunks))
end

Base.size(E::SplineEvaluation) = (length(E.slots) * _nmish(E.geom), _nvars(E.geom) * _ncoef(E.geom))

"""
    lattice_dims(E::SplineEvaluation) -> (nk, nj, ni)

The `(kDim, jDim, iDim)` shape of one slab of `E` (the mish dims, or
`cells .+ 1` for `lattice = :nodes`).
"""
lattice_dims(E::SplineEvaluation) = (E.geom.kDim, E.geom.jDim, E.geom.iDim)

"""
    slab_view(y, E::SplineEvaluation, k) -> AbstractArray{Float64,3}

The `k`-th requested slab of an evaluation output `y`, reshaped to
`(kDim, jDim, iDim)`.
"""
function slab_view(y::AbstractVector, E::SplineEvaluation, k::Integer)
    g = E.geom
    return reshape(view(y, _varrange(Int(k), _nmish(g))), g.kDim, g.jDim, g.iDim)
end

function apply!(y::AbstractVector{Float64}, E::SplineEvaluation, a::AbstractVector{Float64})
    g = E.geom
    (length(y), length(a)) == size(E) || throw(DimensionMismatch(
        "SplineEvaluation apply!: size $(size(E)), y $(length(y)), a $(length(a))"))
    nm, nc = _nmish(g), _ncoef(g)
    dC = (g.b_kDim, g.b_jDim, g.b_iDim)
    dA = (g.b_kDim, g.b_jDim, g.iDim)
    dB = (g.b_kDim, g.jDim,   g.iDim)
    dM = (g.kDim,   g.jDim,   g.iDim)
    pool = E.pool
    for v in 1:_nvars(g)
        p1, p2, p3, p4 = E.slotpos[1, v], E.slotpos[2, v], E.slotpos[3, v], E.slotpos[4, v]
        n1, n2, n3, n4 = p1 > 0, p2 > 0, p3 > 0, p4 > 0
        (n1 | n2 | n3 | n4) || continue
        axk, axj, axi = g.axes[1, v], g.axes[2, v], g.axes[3, v]
        av = view(a, _varrange(v, nc))
        # i axis
        (n1 | n3 | n4) && sweep_axis!(_EvalKernel(axi, axi.Φ0), E.A0, av, 3, dC, dA, pool)
        n2             && sweep_axis!(_EvalKernel(axi, axi.Φ1), E.A1, av, 3, dC, dA, pool)
        # j axis
        (n1 | n4) && sweep_axis!(_EvalKernel(axj, axj.Φ0), E.B00, E.A0, 2, dA, dB, pool)
        n3        && sweep_axis!(_EvalKernel(axj, axj.Φ1), E.B01, E.A0, 2, dA, dB, pool)
        n2        && sweep_axis!(_EvalKernel(axj, axj.Φ0), E.B10, E.A1, 2, dA, dB, pool)
        # k axis → slabs
        n1 && sweep_axis!(_EvalKernel(axk, axk.Φ0), view(y, _varrange(p1, nm)), E.B00, 1, dB, dM, pool)
        n4 && sweep_axis!(_EvalKernel(axk, axk.Φ1), view(y, _varrange(p4, nm)), E.B00, 1, dB, dM, pool)
        n3 && sweep_axis!(_EvalKernel(axk, axk.Φ0), view(y, _varrange(p3, nm)), E.B01, 1, dB, dM, pool)
        n2 && sweep_axis!(_EvalKernel(axk, axk.Φ0), view(y, _varrange(p2, nm)), E.B10, 1, dB, dM, pool)
    end
    return y
end

# Scatter sweep that either writes `dst` (first contribution) or adds into it
# through the accumulation scratch `T`.
function _scatter_accum!(dst, T, first::Bool, kernel, src, axis, dsrc, ddst, pool)
    if first
        sweep_axis!(kernel, dst, src, axis, dsrc, ddst, pool)
    else
        t = view(T, 1:length(dst))
        sweep_axis!(kernel, t, src, axis, dsrc, ddst, pool)
        dst .+= t
    end
    return dst
end

function apply_adjoint!(a::AbstractVector{Float64}, E::SplineEvaluation, y::AbstractVector{Float64})
    g = E.geom
    (length(y), length(a)) == size(E) || throw(DimensionMismatch(
        "SplineEvaluation apply_adjoint!: size $(size(E)), y $(length(y)), a $(length(a))"))
    nm, nc = _nmish(g), _ncoef(g)
    dC = (g.b_kDim, g.b_jDim, g.b_iDim)
    dA = (g.b_kDim, g.b_jDim, g.iDim)
    dB = (g.b_kDim, g.jDim,   g.iDim)
    dM = (g.kDim,   g.jDim,   g.iDim)
    pool = E.pool
    T = E.T
    for v in 1:_nvars(g)
        av = view(a, _varrange(v, nc))
        p1, p2, p3, p4 = E.slotpos[1, v], E.slotpos[2, v], E.slotpos[3, v], E.slotpos[4, v]
        n1, n2, n3, n4 = p1 > 0, p2 > 0, p3 > 0, p4 > 0
        if !(n1 | n2 | n3 | n4)
            fill!(av, 0.0)
            continue
        end
        axk, axj, axi = g.axes[1, v], g.axes[2, v], g.axes[3, v]
        # k axis: slabs → B branches
        n1 && sweep_axis!(_ScatterKernel(axk, axk.Φ0), E.B00, view(y, _varrange(p1, nm)), 1, dM, dB, pool)
        n4 && _scatter_accum!(E.B00, T, !n1, _ScatterKernel(axk, axk.Φ1),
                              view(y, _varrange(p4, nm)), 1, dM, dB, pool)
        n3 && sweep_axis!(_ScatterKernel(axk, axk.Φ0), E.B01, view(y, _varrange(p3, nm)), 1, dM, dB, pool)
        n2 && sweep_axis!(_ScatterKernel(axk, axk.Φ0), E.B10, view(y, _varrange(p2, nm)), 1, dM, dB, pool)
        # j axis: B branches → A branches
        (n1 | n4) && sweep_axis!(_ScatterKernel(axj, axj.Φ0), E.A0, E.B00, 2, dB, dA, pool)
        n3        && _scatter_accum!(E.A0, T, !(n1 | n4), _ScatterKernel(axj, axj.Φ1),
                                     E.B01, 2, dB, dA, pool)
        n2        && sweep_axis!(_ScatterKernel(axj, axj.Φ0), E.A1, E.B10, 2, dB, dA, pool)
        # i axis: A branches → coefficients
        a0 = n1 | n3 | n4
        a0 && sweep_axis!(_ScatterKernel(axi, axi.Φ0), av, E.A0, 3, dA, dC, pool)
        n2 && _scatter_accum!(av, T, !a0, _ScatterKernel(axi, axi.Φ1), E.A1, 3, dA, dC, pool)
    end
    return a
end

# ─────────────────────────────────────────────────────────────────────────────
# Export to Springsteel's spectral layout
# ─────────────────────────────────────────────────────────────────────────────

"""
    coefficients_to_springsteel_spectral!(grid, a, varnames) -> grid.spectral

Write Daisho coefficients `a` (variable-major, per variable
`(b_kDim, b_jDim, b_iDim)` k-fastest, satisfying each variable's BCs — e.g.
the output of [`SplineProjection`](@ref)) into `grid.spectral` so that
`Springsteel.gridTransform!(grid)` reproduces the same field and derivatives.

Springsteel's spectral array stores SB outputs (B-vectors), i-fastest. Since
`a = Γᵀx` along every axis, `B = (P+Q)·a` per axis satisfies
`SA(B) = Γᵀ(Γ(P+Q)Γᵀ)⁻¹Γ(P+Q)Γᵀx = a`; this function applies `(P+Q)` along all
three axes and permutes into `spectral[(z-1)·b_jDim·b_iDim + (l-1)·b_iDim + i, v]`.
Columns of variables not in `varnames` are left untouched.
"""
function coefficients_to_springsteel_spectral!(grid, a::AbstractVector{Float64},
                                               varnames::AbstractVector{<:AbstractString};
                                               nchunks::Integer = Threads.nthreads())
    g = _spline_grid_geometry(grid, varnames)
    nc = _ncoef(g)
    length(a) == _nvars(g) * nc || throw(DimensionMismatch(
        "coefficients_to_springsteel_spectral!: length(a) = $(length(a)), expected $(_nvars(g) * nc)"))
    pool = _geometry_pool(g, nchunks)
    dC = (g.b_kDim, g.b_jDim, g.b_iDim)
    t1 = zeros(nc)
    t2 = zeros(nc)
    spectral = grid.spectral
    for (v, col) in enumerate(g.columns)
        av = view(a, _varrange(v, nc))
        sweep_axis!(_PQKernel(g.axes[1, v]), t1, av, 1, dC, dC, pool)
        sweep_axis!(_PQKernel(g.axes[2, v]), t2, t1, 2, dC, dC, pool)
        sweep_axis!(_PQKernel(g.axes[3, v]), t1, t2, 3, dC, dC, pool)
        B = reshape(t1, dC)
        @inbounds for i in 1:g.b_iDim, l in 1:g.b_jDim, z in 1:g.b_kDim
            spectral[(z - 1) * g.b_jDim * g.b_iDim + (l - 1) * g.b_iDim + i, col] = B[z, l, i]
        end
    end
    return spectral
end
