# Observation operators of the variational solver.
#
# Every backend acts on the variable-major spline COEFFICIENT vector `a` (per
# variable `(b_kDim, b_jDim, b_iDim)` column-major, k fastest — the output of
# `SplineProjection`). An observation is a weighted sum of terms
# `(var, slot, weight)` with derivative slots `1 = value, 2 = ∂x (i),
# 3 = ∂y (j), 4 = ∂z (k)`:
#
#     h_m(a) = Σ_t w_t · [slot_t of var_t](location_m)        (linear obs)
#
# - `MishObsBackend`: observations located on an evaluation lattice — the
#   mish (quadrature points) or the B-spline nodes (`lattice = :nodes`). The
#   forward operator is one shared `SplineEvaluation` pass per lattice (owned by the
#   `ObsSet`) to the union of the requested (var, slot) slabs, followed by
#   per-obs weighted sums; the adjoint scatters into the slabs and applies the
#   evaluation adjoint once. Optional nonlinear `model` hook.
# - `ScatteredObsBackend`: observations at arbitrary (x, y, z) inside the
#   domain, stored as a sparse matrix of tensor-product basis stencils.
# - `ObsSet`: the union of backends, concatenated obs vectors, shared slab
#   workspace. `ObsSet <: AbstractLinearOp` (coefficients → observations), so
#   `apply!`/`apply_adjoint!`/`adjoint_relative_error` work on it directly.
#
# Inverse variances `rinv = 1/σ²` are passed explicitly (see
# `inverse_variance`); they must be finite and strictly positive.

"""
    AbstractObsBackend

Supertype of observation backends of the variational solver. A backend holds
a block of observations (values `y`, inverse variances `r = 1/σ²`) and their
operator `h` acting on the spline coefficient vector. Backends are combined
and applied through an [`ObsSet`](@ref); the concrete backends are
[`MishObsBackend`](@ref) and [`ScatteredObsBackend`](@ref).

Interface: [`nobs`](@ref), [`obs_values`](@ref), [`obs_rinv`](@ref),
[`slots_needed`](@ref), [`is_linear`](@ref).
"""
abstract type AbstractObsBackend end

"""
    nobs(b::AbstractObsBackend) -> Int
    nobs(set::ObsSet) -> Int

Number of observations.
"""
function nobs end

"""
    obs_values(b) -> Vector{Float64}

Observed values `y` (for a [`MishObsBackend`](@ref), in the backend's
point-sorted order; see its `perm` field).
"""
function obs_values end

"""
    obs_rinv(b) -> Vector{Float64}

Observation inverse variances `r = 1/σ²` (strictly positive), in the same
order as [`obs_values`](@ref).
"""
function obs_rinv end

"""
    slots_needed(b) -> Vector{Tuple{Int,Int}}

Distinct `(var, slot)` pairs a mish backend reads from the shared evaluation
slabs; empty for backends that do not use the slabs.
"""
function slots_needed end

"""
    is_linear(b) -> Bool

Whether the observation operator is linear in the coefficients.
"""
function is_linear end

"""
    inverse_variance(sigma) -> Vector{Float64}

`1 ./ sigma.^2`, with a check that every σ is finite and > 0. Convenience for
the `rinv` argument of the backend constructors.
"""
function inverse_variance(sigma::AbstractVector{<:Real})
    all(s -> isfinite(s) && s > 0, sigma) || throw(ArgumentError(
        "inverse_variance: every σ must be finite and > 0"))
    return [1.0 / Float64(s)^2 for s in sigma]
end
inverse_variance(sigma::Real) = inverse_variance([sigma])[1]

function _check_obs_vectors(what, n, y, rinv)
    length(y) == n || throw(DimensionMismatch("$(what): $(n) observations but $(length(y)) values"))
    length(rinv) == n || throw(DimensionMismatch("$(what): $(n) observations but $(length(rinv)) inverse variances"))
    all(isfinite, y) || throw(ArgumentError("$(what): observation values must be finite"))
    all(r -> isfinite(r) && r > 0, rinv) || throw(ArgumentError(
        "$(what): inverse variances must be finite and > 0"))
    return nothing
end

function _check_term(what, m, var, slot, w, nvars)
    1 <= var <= nvars || throw(ArgumentError(
        "$(what): observation $(m) has variable index $(var) outside 1:$(nvars)"))
    1 <= slot <= 4 || throw(ArgumentError(
        "$(what): observation $(m) has slot $(slot); must be 1 (value), 2 (∂x), 3 (∂y) or 4 (∂z)"))
    isfinite(w) || throw(ArgumentError("$(what): observation $(m) has a non-finite weight"))
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# MishObsBackend
# ─────────────────────────────────────────────────────────────────────────────

"""
    MishObsBackend(kind::Symbol, dims::Tuple{Integer,Integer}, points, y, rinv,
                   terms::AbstractVector{<:AbstractVector{<:Tuple{Integer,Integer,Real}}};
                   model = nothing, lattice = :mish)
    MishObsBackend(kind, grid, varnames, points, y, rinv, terms; model = nothing, lattice = :mish)

Observations located on an evaluation lattice of the state grid — the mish
(`lattice = :mish`, the quadrature points) or the B-spline nodes
(`lattice = :nodes`, see [`SplineEvaluation`](@ref)) — stored
structure-of-arrays and sorted (stably) by point. `dims = (nvars, npoints)`
gives the number of state variables and of lattice points per variable (the
second form takes them from the RRR `grid` and `varnames`, as for
[`SplineProjection`](@ref)). For observation
`m`: `points[m] ∈ 1:npoints` is the lattice flat index (k fastest), `y[m]` the
value, `rinv[m] = 1/σ²` (> 0, see [`inverse_variance`](@ref)), and `terms[m]`
a non-empty list of `(var, slot, weight)` with `var ∈ 1:nvars` and
`slot ∈ 1:4` (`1 = value, 2 = ∂x, 3 = ∂y, 4 = ∂z`). A linear observation is
`h_m(a) = Σ_t weight_t · slab(var_t, slot_t)[point_m]`.

`kind` is a free diagnostic label (e.g. `:radial_velocity`,
`:mass_continuity`, `:point`, `:wind_speed`).

**Nonlinear hook.** `model(sample, m) -> (value, weights)`: `sample[s]` is
the current value of `slots[s]` (every slot this backend uses, in the order
of the `slots` field) at observation `m`'s point; `value` is `h_m(a)` and
`weights` (any indexable of length `length(terms[m])`) the tangent-linear
weights of `m`'s terms, in the order given at construction. `m` is the
ORIGINAL (construction-order) observation index. Linearisation (through the
[`ObsSet`](@ref)) overwrites the stored weights; the initial weights in
`terms` are used until the first linearisation. `model === nothing` ⇒ linear.

Fields (sorted order): `point`, `y`, `rinv`, `term_ptr` (CSR, length
`nobs+1`), `term_slab` (index into `slots`), `term_w`, `slots`, `model`,
`kind`, `perm` (`perm[m]` = original index of sorted observation `m`),
`nvars`, `nmish` (lattice points per variable), `lattice`. Nonlinear
models are supported on the mish lattice only.
"""
struct MishObsBackend <: AbstractObsBackend
    kind::Symbol
    nvars::Int
    nmish::Int
    point::Vector{Int32}
    y::Vector{Float64}
    rinv::Vector{Float64}
    term_ptr::Vector{Int32}
    term_slab::Vector{Int32}
    term_w::Vector{Float64}
    slots::Vector{Tuple{Int,Int}}
    model::Any
    perm::Vector{Int}
    lattice::Symbol
end

function MishObsBackend(kind::Symbol, dims::Tuple{Integer,Integer},
                        points::AbstractVector{<:Integer}, y::AbstractVector{<:Real},
                        rinv::AbstractVector{<:Real},
                        terms::AbstractVector{<:AbstractVector{<:Tuple{Integer,Integer,Real}}};
                        model = nothing, lattice::Symbol = :mish)
    nvars, nmish = Int(dims[1]), Int(dims[2])
    (nvars >= 1 && nmish >= 1) || throw(ArgumentError("MishObsBackend: dims must be ≥ 1, got $(dims)"))
    what = "MishObsBackend(:$(kind))"
    lattice in (:mish, :nodes) || throw(ArgumentError(
        "$(what): lattice must be :mish or :nodes, got :$(lattice)"))
    (lattice === :mish || model === nothing) || throw(ArgumentError(
        "$(what): nonlinear observations (a model hook) are supported on the mish lattice only"))
    n = length(points)
    length(terms) == n || throw(DimensionMismatch("$(what): $(n) points but $(length(terms)) term lists"))
    _check_obs_vectors(what, n, y, rinv)
    for m in 1:n
        1 <= points[m] <= nmish || throw(ArgumentError(
            "$(what): observation $(m) has point $(points[m]) outside 1:$(nmish)"))
        isempty(terms[m]) && throw(ArgumentError("$(what): observation $(m) has no terms"))
        for (v, s, w) in terms[m]
            _check_term(what, m, v, s, w, nvars)
        end
    end
    perm = sortperm(collect(points); alg = Base.Sort.DEFAULT_STABLE)
    # Slot list: distinct (var, slot) in order of first appearance (sorted order).
    slots = Tuple{Int,Int}[]
    slotidx = Dict{Tuple{Int,Int},Int}()
    point = Vector{Int32}(undef, n)
    ys = Vector{Float64}(undef, n)
    rs = Vector{Float64}(undef, n)
    term_ptr = Vector{Int32}(undef, n + 1)
    term_slab = Int32[]
    term_w = Float64[]
    term_ptr[1] = 1
    for (m, o) in enumerate(perm)
        point[m] = points[o]
        ys[m] = y[o]
        rs[m] = rinv[o]
        for (v, s, w) in terms[o]
            key = (Int(v), Int(s))
            k = get!(slotidx, key) do
                push!(slots, key)
                length(slots)
            end
            push!(term_slab, k)
            push!(term_w, Float64(w))
        end
        term_ptr[m + 1] = length(term_w) + 1
    end
    return MishObsBackend(kind, nvars, nmish, point, ys, rs, term_ptr, term_slab, term_w,
                          slots, model, perm, lattice)
end

function MishObsBackend(kind::Symbol, grid, varnames::AbstractVector{<:AbstractString},
                        points, y, rinv, terms; model = nothing, lattice::Symbol = :mish)
    g = _spline_grid_geometry(grid, varnames; lattice = lattice)
    return MishObsBackend(kind, (_nvars(g), _nmish(g)), points, y, rinv, terms;
                          model = model, lattice = lattice)
end

nobs(b::MishObsBackend) = length(b.point)
obs_values(b::MishObsBackend) = b.y
obs_rinv(b::MishObsBackend) = b.rinv
slots_needed(b::MishObsBackend) = b.slots
is_linear(b::MishObsBackend) = b.model === nothing

# ─────────────────────────────────────────────────────────────────────────────
# ScatteredObsBackend
# ─────────────────────────────────────────────────────────────────────────────

"""
    ScatteredObsBackend(kind::Symbol, grid, varnames, xyz::AbstractMatrix, y, rinv,
                        terms::AbstractVector{<:AbstractVector{<:Tuple{Integer,Integer,Real}}})

Linear observations at arbitrary points `xyz[m, :] = (x, y, z)` (metres, the
grid's i, j, k coordinates) inside the domain of the RRR `grid`; `varnames`
selects and orders the state variables as for [`SplineProjection`](@ref).
Terms, values and `rinv = 1/σ²` are as for [`MishObsBackend`](@ref).

The operator is stored as a sparse `nobs × (nvars·ncoef)` matrix `H` (and its
transpose `Ht`): each term `(var, slot, w)` contributes the 4×4×4
tensor-product stencil `w·φ_i^{(dᵢ)}(x)·φ_j^{(dⱼ)}(y)·φ_k^{(dₖ)}(z)` (first
derivative on the slot's axis, values elsewhere) at the Daisho coefficient
indices `(var−1)·ncoef + mk + (mj−1)·b_kDim + (mi−1)·b_kDim·b_jDim`, with the
node range of Springsteel's single-point `SItransform`.

Points outside `[min, max]` on any axis throw `ArgumentError` (callers
pre-filter). Linear only.
"""
struct ScatteredObsBackend <: AbstractObsBackend
    kind::Symbol
    H::SparseMatrixCSC{Float64,Int}
    Ht::SparseMatrixCSC{Float64,Int}
    y::Vector{Float64}
    rinv::Vector{Float64}
end

# Nonzero basis values (derivative d) of the (up to) four nodes supporting x,
# as (coefficient index, value) pairs appended to `idx`/`val`.
function _axis_stencil!(idx::Vector{Int}, val::Vector{Float64}, sp, x::Float64, d::Int)
    empty!(idx); empty!(val)
    xm = ceil(Int, (x - sp.xmin - 2.0 * sp.DX) * sp.DXrecip)
    for m in xm:(xm + 3)
        (-1 <= m <= sp.num_cells + 1) || continue
        push!(idx, m + 2)
        push!(val, _SC.basis(sp, m, x, d))
    end
    return nothing
end

function ScatteredObsBackend(kind::Symbol, grid, varnames::AbstractVector{<:AbstractString},
                             xyz::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
                             rinv::AbstractVector{<:Real},
                             terms::AbstractVector{<:AbstractVector{<:Tuple{Integer,Integer,Real}}})
    g = _spline_grid_geometry(grid, varnames)
    nv, nc = _nvars(g), _ncoef(g)
    what = "ScatteredObsBackend(:$(kind))"
    n = size(xyz, 1)
    size(xyz, 2) == 3 || throw(DimensionMismatch("$(what): xyz must be nobs × 3, got $(size(xyz))"))
    length(terms) == n || throw(DimensionMismatch("$(what): $(n) points but $(length(terms)) term lists"))
    _check_obs_vectors(what, n, y, rinv)
    I = Int[]; J = Int[]; V = Float64[]
    ii = Int[]; vi = Float64[]; ij = Int[]; vj = Float64[]; ik = Int[]; vk = Float64[]
    for m in 1:n
        isempty(terms[m]) && throw(ArgumentError("$(what): observation $(m) has no terms"))
        x = (Float64(xyz[m, 1]), Float64(xyz[m, 2]), Float64(xyz[m, 3]))
        all(isfinite, x) || throw(ArgumentError("$(what): observation $(m) has a non-finite location"))
        for (v, s, w) in terms[m]
            _check_term(what, m, v, s, w, nv)
            spi = g.axes[3, v].sp; spj = g.axes[2, v].sp; spk = g.axes[1, v].sp
            for (c, sp) in ((1, spi), (2, spj), (3, spk))
                sp.xmin <= x[c] <= sp.xmax || throw(ArgumentError(
                    "$(what): observation $(m) coordinate $(c) = $(x[c]) outside the domain " *
                    "[$(sp.xmin), $(sp.xmax)]"))
            end
            _axis_stencil!(ii, vi, spi, x[1], s == 2 ? 1 : 0)
            _axis_stencil!(ij, vj, spj, x[2], s == 3 ? 1 : 0)
            _axis_stencil!(ik, vk, spk, x[3], s == 4 ? 1 : 0)
            off = (v - 1) * nc
            for (a, mi) in enumerate(ii), (b, mj) in enumerate(ij), (c, mk) in enumerate(ik)
                val = Float64(w) * vi[a] * vj[b] * vk[c]
                val == 0.0 && continue
                push!(I, m)
                push!(J, off + mk + (mj - 1) * g.b_kDim + (mi - 1) * g.b_kDim * g.b_jDim)
                push!(V, val)
            end
        end
    end
    H = sparse(I, J, V, n, nv * nc)
    return ScatteredObsBackend(kind, H, sparse(transpose(H)), Vector{Float64}(y),
                               Vector{Float64}(rinv))
end

nobs(b::ScatteredObsBackend) = size(b.H, 1)
obs_values(b::ScatteredObsBackend) = b.y
obs_rinv(b::ScatteredObsBackend) = b.rinv
slots_needed(::ScatteredObsBackend) = Tuple{Int,Int}[]
is_linear(::ScatteredObsBackend) = true

# ─────────────────────────────────────────────────────────────────────────────
# ObsSet
# ─────────────────────────────────────────────────────────────────────────────

# One evaluation lattice of an ObsSet: the shared SplineEvaluation over the
# union of the slots its backends need, and the forward/adjoint slab buffers.
struct _LatticeEval
    lattice::Symbol
    ev::SplineEvaluation
    slots::Vector{Tuple{Int,Int}}
    slabs::Vector{Float64}
    slabs_adj::Vector{Float64}
end

# A lattice backend bound to an ObsSet's shared slab workspace.
struct _MishBinding
    backend::MishObsBackend
    offset::Int                       # obs offset in the concatenated vectors
    term_idx::Vector{Int}             # per term: absolute index into `slabs`
    sample_idx::Vector{Int}           # per backend slot: slab offset (0-based) into `slabs`
    chunks::Vector{UnitRange{Int}}    # obs chunks that never split a point
    sample::Vector{Float64}           # model sample buffer
    slabs::Vector{Float64}            # the lattice's forward slabs (aliases _LatticeEval.slabs)
    slabs_adj::Vector{Float64}        # the lattice's adjoint slabs
end

"""
    ObsSet(grid, varnames, backends; nchunks = Threads.nthreads())

The union of observation backends acting on the coefficient vector of the
state variables `varnames` of the RRR `grid` (the same `grid`/`varnames` as
the [`SplineProjection`](@ref)). Observations are concatenated in backend
order (`ranges[b]` is backend `b`'s block); `y` and `rinv` hold the
concatenated values and inverse variances.

One [`SplineEvaluation`](@ref) `ev` is built for the union `ev_slots` of the
`(var, slot)` pairs used by all mish-lattice backends (`nothing` if there are
none), and a second one (`ev_nodes`, `ev_nodes_slots`) for the node-lattice
backends (`MishObsBackend` with `lattice = :nodes`); `slab_map[b]` maps
backend `b`'s slots to indices in its lattice's slot list (empty for
scattered backends).

`ObsSet <: AbstractLinearOp` of size `(nobs, nvars·ncoef)`:
`apply!(y, set, a)` = [`apply_H!`](@ref), `apply_adjoint!(a, set, y)` =
[`apply_Ht!`](@ref). Mish observations are processed in `nchunks` tasks
(chunks never split a mish point, so the adjoint scatter is race-free and the
results do not depend on `nchunks`); `nchunks = 1` is allocation-free.

See also: [`evaluate!`](@ref), [`linearize!`](@ref), [`cost_by_backend`](@ref)
"""
struct ObsSet <: AbstractLinearOp
    backends::Vector{AbstractObsBackend}
    ranges::Vector{UnitRange{Int}}
    ev::Union{Nothing,SplineEvaluation}             # mish lattice
    ev_slots::Vector{Tuple{Int,Int}}
    ev_nodes::Union{Nothing,SplineEvaluation}       # node lattice
    ev_nodes_slots::Vector{Tuple{Int,Int}}
    lattices::Vector{_LatticeEval}                  # the lattices present, mish first
    slab_map::Vector{Vector{Int}}
    slabs::Vector{Float64}                          # mish forward slabs
    slabs_adj::Vector{Float64}
    a_tmp::Vector{Float64}
    y::Vector{Float64}
    rinv::Vector{Float64}
    mish::Vector{_MishBinding}                      # every lattice backend (mish and nodes)
    scattered::Vector{Tuple{Int,ScatteredObsBackend}}   # (obs offset, backend)
    nmish::Int
    nnodes::Int
    ncoef_total::Int
end

# Split 1:n into ≤ nchunks contiguous ranges whose boundaries never fall
# between two observations at the same point.
function _point_chunks(point::Vector{Int32}, nchunks::Int)
    n = length(point)
    n == 0 && return UnitRange{Int}[]
    nc = clamp(nchunks, 1, n)
    chunks = UnitRange{Int}[]
    lo = 1
    for c in 1:nc
        lo > n && break
        hi = c == nc ? n : max(lo, div(c * n, nc))
        while hi < n && point[hi + 1] == point[hi]
            hi += 1
        end
        push!(chunks, lo:hi)
        lo = hi + 1
    end
    return chunks
end

function ObsSet(grid, varnames::AbstractVector{<:AbstractString},
                backends::AbstractVector{<:AbstractObsBackend};
                nchunks::Integer = Threads.nthreads())
    nchunks >= 1 || throw(ArgumentError("ObsSet: nchunks must be ≥ 1, got $(nchunks)"))
    g = _spline_grid_geometry(grid, varnames)
    nv, nm, nc = _nvars(g), _nmish(g), _ncoef(g)
    ncoef_total = nv * nc
    gn = _spline_grid_geometry(grid, varnames; lattice = :nodes)
    nn = _nmish(gn)
    npts = Dict(:mish => nm, :nodes => nn)
    slots_of = Dict(:mish => Tuple{Int,Int}[], :nodes => Tuple{Int,Int}[])
    for b in backends
        if b isa MishObsBackend
            (b.nvars, b.nmish) == (nv, npts[b.lattice]) || throw(DimensionMismatch(
                "ObsSet: $(b.lattice) backend :$(b.kind) has dims $((b.nvars, b.nmish)), grid has $((nv, npts[b.lattice]))"))
            sl = slots_of[b.lattice]
            for s in b.slots
                s in sl || push!(sl, s)
            end
        elseif b isa ScatteredObsBackend
            size(b.H, 2) == ncoef_total || throw(DimensionMismatch(
                "ObsSet: scattered backend :$(b.kind) has $(size(b.H, 2)) coefficient columns, expected $(ncoef_total)"))
        else
            throw(ArgumentError("ObsSet: unsupported backend type $(typeof(b))"))
        end
    end
    lattices = _LatticeEval[]
    for lat in (:mish, :nodes)
        sl = slots_of[lat]
        isempty(sl) && continue
        ev = SplineEvaluation(grid, varnames, sl; lattice = lat, nchunks = nchunks)
        ns = length(sl) * npts[lat]
        push!(lattices, _LatticeEval(lat, ev, sl, zeros(ns), zeros(ns)))
    end
    lat_index = Dict(L.lattice => k for (k, L) in enumerate(lattices))
    ranges = UnitRange{Int}[]
    slab_map = Vector{Int}[]
    mish = _MishBinding[]
    scattered = Tuple{Int,ScatteredObsBackend}[]
    off = 0
    for b in backends
        n = nobs(b)
        push!(ranges, (off + 1):(off + n))
        if b isa MishObsBackend
            if n == 0
                # No observations ⇒ no slots ⇒ possibly no lattice built; bind nothing.
                push!(slab_map, Int[])
                push!(mish, _MishBinding(b, off, Int[], Int[], UnitRange{Int}[], Float64[],
                                         Float64[], Float64[]))
                continue
            end
            L = lattices[lat_index[b.lattice]]
            np = npts[b.lattice]
            map = [findfirst(==(s), L.slots) for s in b.slots]
            push!(slab_map, map)
            term_idx = Vector{Int}(undef, length(b.term_w))
            for m in 1:n, t in b.term_ptr[m]:(b.term_ptr[m + 1] - 1)
                term_idx[t] = (map[b.term_slab[t]] - 1) * np + Int(b.point[m])
            end
            sample_idx = [(k - 1) * np for k in map]
            push!(mish, _MishBinding(b, off, term_idx, sample_idx,
                                     _point_chunks(b.point, Int(nchunks)), zeros(length(b.slots)),
                                     L.slabs, L.slabs_adj))
        else
            push!(slab_map, Int[])
            push!(scattered, (off, b))
        end
        off += n
    end
    y = zeros(off); rinv = zeros(off)
    for (b, r) in zip(backends, ranges)
        y[r] .= obs_values(b)
        rinv[r] .= obs_rinv(b)
    end
    km = get(lat_index, :mish, 0)
    kn = get(lat_index, :nodes, 0)
    ev = km == 0 ? nothing : lattices[km].ev
    ev_slots = km == 0 ? Tuple{Int,Int}[] : lattices[km].slots
    ev_nodes = kn == 0 ? nothing : lattices[kn].ev
    ev_nodes_slots = kn == 0 ? Tuple{Int,Int}[] : lattices[kn].slots
    slabs = km == 0 ? Float64[] : lattices[km].slabs
    slabs_adj = km == 0 ? Float64[] : lattices[km].slabs_adj
    return ObsSet(collect(AbstractObsBackend, backends), ranges, ev, ev_slots, ev_nodes,
                  ev_nodes_slots, lattices, slab_map, slabs, slabs_adj, zeros(ncoef_total),
                  y, rinv, mish, scattered, nm, nn, ncoef_total)
end

nobs(set::ObsSet) = length(set.y)
obs_values(set::ObsSet) = set.y
obs_rinv(set::ObsSet) = set.rinv
is_linear(set::ObsSet) = all(is_linear, set.backends)
Base.size(set::ObsSet) = (length(set.y), set.ncoef_total)

# Run f(range) over the chunks: chunk 1 on the calling task, the rest spawned.
function _run_chunks(f::F, chunks::Vector{UnitRange{Int}}) where {F}
    n = length(chunks)
    n == 0 && return nothing
    if n == 1
        f(chunks[1])
    else
        @sync begin
            for c in 2:n
                r = chunks[c]
                Threads.@spawn f(r)
            end
            f(chunks[1])
        end
    end
    return nothing
end

struct _MishHKernel
    y::Vector{Float64}
    b::_MishBinding
end
function (k::_MishHKernel)(r::UnitRange{Int})
    be = k.b.backend
    ptr = be.term_ptr; w = be.term_w; idx = k.b.term_idx
    slabs = k.b.slabs; y = k.y; off = k.b.offset
    @inbounds for m in r
        s = 0.0
        for t in ptr[m]:(ptr[m + 1] - 1)
            s += w[t] * slabs[idx[t]]
        end
        y[off + m] = s
    end
    return nothing
end

struct _MishHtKernel
    b::_MishBinding
    y::Vector{Float64}
end
function (k::_MishHtKernel)(r::UnitRange{Int})
    be = k.b.backend
    ptr = be.term_ptr; w = be.term_w; idx = k.b.term_idx
    sa = k.b.slabs_adj; y = k.y; off = k.b.offset
    @inbounds for m in r
        ym = y[off + m]
        for t in ptr[m]:(ptr[m + 1] - 1)
            sa[idx[t]] += w[t] * ym
        end
    end
    return nothing
end

function _check_obs_sizes(set::ObsSet, y, a, what)
    (length(y) == length(set.y) && length(a) == set.ncoef_total) || throw(DimensionMismatch(
        "ObsSet $(what): size $(size(set)), observations $(length(y)), coefficients $(length(a))"))
    return nothing
end

"""
    apply_H!(y, set::ObsSet, a) -> y

Tangent-linear observation operator at the last linearisation: `y = H·a`
(one evaluation pass per lattice to its shared slabs, per-obs weighted sums
for lattice backends, sparse products for scattered backends). For linear
backends this is the observation operator itself.
"""
function apply_H!(y::AbstractVector{Float64}, set::ObsSet, a::AbstractVector{Float64})
    _check_obs_sizes(set, y, a, "apply_H!")
    for L in set.lattices
        apply!(L.slabs, L.ev, a)
    end
    for b in set.mish
        _run_chunks(_MishHKernel(y, b), b.chunks)
    end
    for (off, b) in set.scattered
        nobs(b) == 0 && continue
        mul!(view(y, (off + 1):(off + nobs(b))), b.H, a)
    end
    return y
end

"""
    apply_Ht!(a, set::ObsSet, y) -> a

Exact transpose of [`apply_H!`](@ref): OVERWRITES `a` with `Hᵀ·y` (scatter
into each lattice's adjoint slabs, one evaluation-adjoint pass per lattice,
then the scattered backends accumulate `Htᵀ·y`).
"""
function apply_Ht!(a::AbstractVector{Float64}, set::ObsSet, y::AbstractVector{Float64})
    _check_obs_sizes(set, y, a, "apply_Ht!")
    for L in set.lattices
        fill!(L.slabs_adj, 0.0)
    end
    for b in set.mish
        _run_chunks(_MishHtKernel(b, y), b.chunks)
    end
    if isempty(set.lattices)
        fill!(a, 0.0)
    else
        for (k, L) in enumerate(set.lattices)
            if k == 1
                apply_adjoint!(a, L.ev, L.slabs_adj)
            else
                apply_adjoint!(set.a_tmp, L.ev, L.slabs_adj)
                a .+= set.a_tmp
            end
        end
    end
    for (off, b) in set.scattered
        nobs(b) == 0 && continue
        mul!(a, b.Ht, view(y, (off + 1):(off + nobs(b))), 1.0, 1.0)
    end
    return a
end

apply!(y::AbstractVector{Float64}, set::ObsSet, a::AbstractVector{Float64}) = apply_H!(y, set, a)
apply_adjoint!(a::AbstractVector{Float64}, set::ObsSet, y::AbstractVector{Float64}) = apply_Ht!(a, set, y)

# Nonlinear mish backend: evaluate the model at every observation (serial;
# models may allocate). Writes values into y (if `values`) and tangent
# weights into the backend (if `weights`). Function barrier on the model type.
function _mish_model_pass!(y, b::_MishBinding, values::Bool, weights::Bool)
    return _mish_model_pass!(y, b, b.backend.model, values, weights)
end
function _mish_model_pass!(y, b::_MishBinding, model::M, values::Bool, weights::Bool) where {M}
    be = b.backend
    slabs = b.slabs
    sample = b.sample
    sidx = b.sample_idx
    ptr = be.term_ptr
    for m in 1:nobs(be)
        p = Int(be.point[m])
        @inbounds for s in eachindex(sample)
            sample[s] = slabs[sidx[s] + p]
        end
        val, wts = model(sample, be.perm[m])
        nt = ptr[m + 1] - ptr[m]
        length(wts) == nt || throw(DimensionMismatch(
            "MishObsBackend(:$(be.kind)) model returned $(length(wts)) weights for observation " *
            "$(be.perm[m]) with $(nt) terms"))
        values && (y[b.offset + m] = Float64(val))
        if weights
            for (q, t) in enumerate(ptr[m]:(ptr[m + 1] - 1))
                be.term_w[t] = Float64(wts[q])
            end
        end
    end
    return nothing
end

"""
    evaluate!(y, set::ObsSet, a; linearize = false) -> y

Full (possibly nonlinear) observation operator `y = h(a)`. Linear backends
give the same result as [`apply_H!`](@ref); nonlinear mish backends call
their `model`. With `linearize = true` the nonlinear backends' tangent
weights are also updated at `a` in the same pass (see [`linearize!`](@ref)).
"""
function evaluate!(y::AbstractVector{Float64}, set::ObsSet, a::AbstractVector{Float64};
                   linearize::Bool = false)
    _check_obs_sizes(set, y, a, "evaluate!")
    for L in set.lattices
        apply!(L.slabs, L.ev, a)
    end
    for b in set.mish
        if b.backend.model === nothing
            _run_chunks(_MishHKernel(y, b), b.chunks)
        else
            _mish_model_pass!(y, b, true, linearize)
        end
    end
    for (off, b) in set.scattered
        nobs(b) == 0 && continue
        mul!(view(y, (off + 1):(off + nobs(b))), b.H, a)
    end
    return y
end

"""
    linearize!(set::ObsSet, a) -> set

Update the tangent-linear weights of every nonlinear backend at the
coefficient state `a` (no-op for linear backends).
"""
function linearize!(set::ObsSet, a::AbstractVector{Float64})
    length(a) == set.ncoef_total || throw(DimensionMismatch(
        "ObsSet linearize!: $(length(a)) coefficients, expected $(set.ncoef_total)"))
    any(b -> b.backend.model !== nothing, set.mish) || return set
    apply!(set.slabs, set.ev, a)          # nonlinear models live on the mish lattice only
    for b in set.mish
        b.backend.model === nothing || _mish_model_pass!(nothing, b, false, true)
    end
    return set
end

"""
    cost_by_backend(set::ObsSet, residual) -> Vector{Float64}

Per-backend observation cost `½ Σ_{m ∈ ranges[b]} rinv_m · residual_m²` for a
concatenated residual vector (the caller chooses the residual, e.g.
`y − h(a)` for the nonlinear cost or `H·δa − d` for the quadratic model).
"""
function cost_by_backend(set::ObsSet, residual::AbstractVector{<:Real})
    length(residual) == length(set.y) || throw(DimensionMismatch(
        "cost_by_backend: residual has length $(length(residual)), expected $(length(set.y))"))
    return [0.5 * sum((set.rinv[m] * residual[m]^2 for m in r); init = 0.0) for r in set.ranges]
end
