# Mish-space spatial filter F of the control-variable chain C = S∘D∘F, with
# its exact transpose.
#
# Per axis and variable the filter is Springsteel's spline-direction kernel
# convolution (`_convolve_axis!`), stored as a sparse matrix on the axis's
# mish points x:
#
#     F = diag(1/W)·K,    K_ij = K(x_j − x_i) for |x_j − x_i| ≤ support,
#                         W_i  = Σ_j K_ij      (row normalisation; W_i ≤ 0 ⇒ identity row).
#
# Both kernels (Gaussian, Lanczos) are even and the support window is
# symmetric, so K is symmetric; F is not: (F − Fᵀ)_ij = K_ij·(1/W_i − 1/W_j),
# which is nonzero wherever the row weights of i and j differ. That happens
# (a) within one support of a boundary, where the window is truncated, so
# Fᵀ·1 ≠ 1 for columns within 2·support of a boundary; and (b) on a
# non-uniform (Gauss) mish also in the interior, because the truncated window
# of a point then depends on its position within its cell (W_i varies at the
# size of the truncated kernel tail: Fᵀ·1 − 1 ≈ 2.5e-5 in the interior for a
# σ = 1.5-cell, 4σ Gaussian at mubar = 3; it is uniform to round-off on a
# uniform mish, or with n_sigma = 8). One
# Springsteel subtlety: its left-edge test `x_j < x_i − support` is not the
# exact mirror of the right-edge test `x_j − x_i > support` in floating point,
# so a pair at exactly the support distance can be in one row and not the
# other; the row-normalised matrix reproduces Springsteel's tests verbatim.
#
# The 3-D filter is separable: the three axis operators act on different
# indices of the (kDim, jDim, iDim) mish array and commute exactly, so the
# sweep order (k, j, i forward; i, j, k adjoint) affects only round-off.

const _SpringsteelSplineFilter = Union{Springsteel.GaussianFilter,Springsteel.LanczosFilter}

"""
    AxisFilter(filter::Springsteel.AbstractFilter, ax::AxisSpline; symmetric = false)

Sparse `Dim×Dim` matrix form of a Springsteel spline-direction filter
(`GaussianFilter` or `LanczosFilter`; `sigma`/`a` in cell widths of
`ax.sp.DX`) on the mish points of `ax`, with its materialised transpose `Ft`
so both directions are CSC products.

- `symmetric = false` (default): `F[i,j] = K(x_j − x_i)/W_i` within the kernel
  support, exactly Springsteel's `_convolve_axis!` (row-normalised, so
  `F·1 = 1`; a row whose weight sum is ≤ 0 is the identity row). Not
  symmetric: see the file header.
- `symmetric = true`: `F = W_q^{1/2}·(K/κ)·W_q^{1/2}` with the SB quadrature
  weights `W_q = diag(ax.w)` (`DX·quadweights`) and `κ = ∫K` over the
  support (so the kernel integrates to one; Springsteel's Lanczos kernel has
  `K(0) = 1`, `∫K ≈ DX`), built on the upper triangle and mirrored, so it is
  exactly symmetric. There is no row normalisation: in the interior
  `F·√w ≈ √w` to the accuracy of the mish quadrature of `∫K/κ = 1`, so
  constants are preserved (`F·1 ≈ 1`) only on a mish with uniform weights
  (regular quadrature); on a Gauss mish `F·1` deviates by O(10%) with the
  in-cell weight pattern. Near boundaries the truncated window damps the
  output.

Kernel values that are exactly zero (Lanczos zero crossings and support edge)
are not stored; the diagonal always is. `SpectralFilter` (and any other
filter type) throws `ArgumentError`.

Fields: `F`, `Ft`, `support` (kernel half-width, metres), `length_cells`
(Gaussian `sigma` or Lanczos `a`, in cells), `symmetric`.
"""
struct AxisFilter
    F::SparseMatrixCSC{Float64,Int}
    Ft::SparseMatrixCSC{Float64,Int}
    support::Float64
    length_cells::Float64
    symmetric::Bool
end

function _check_spline_filter(filter)
    filter isa Springsteel.SpectralFilter && throw(ArgumentError(
        "SpectralFilter (boxcar / notch / window cutoffs) cannot be applied along a " *
        "spline axis; use GaussianFilter or LanczosFilter"))
    filter isa _SpringsteelSplineFilter || throw(ArgumentError(
        "unsupported spline-axis filter $(typeof(filter)); use GaussianFilter or LanczosFilter"))
    if filter isa Springsteel.GaussianFilter
        (isfinite(filter.sigma) && filter.sigma > 0) || throw(ArgumentError(
            "GaussianFilter: sigma must be finite and > 0 (cells), got $(filter.sigma)"))
        filter.n_sigma >= 1 || throw(ArgumentError(
            "GaussianFilter: n_sigma must be ≥ 1, got $(filter.n_sigma)"))
    else
        filter.a >= 1 || throw(ArgumentError("LanczosFilter: a must be ≥ 1, got $(filter.a)"))
    end
    return nothing
end

_filter_length(f::Springsteel.GaussianFilter) = Float64(f.sigma)
_filter_length(f::Springsteel.LanczosFilter) = Float64(f.a)

function AxisFilter(filter::Springsteel.AbstractFilter, ax::AxisSpline; symmetric::Bool = false)
    _check_spline_filter(filter)
    K, support = Springsteel._spline_kernel(filter, ax.sp.DX)
    x = ax.mishPoints
    issorted(x) || throw(ArgumentError("AxisFilter: mish points must be non-decreasing"))
    F = symmetric ? _symmetric_filter_matrix(K, support, x, ax.w) :
                    _row_normalised_filter_matrix(K, support, x)
    Ft = symmetric ? copy(F) : sparse(transpose(F))
    return AxisFilter(F, Ft, Float64(support), _filter_length(filter), symmetric)
end

# Two-pointer sweep mirroring Springsteel._convolve_axis! exactly (same window
# tests, same `wsum > 0` rule), O(Dim·bandwidth).
function _row_normalised_filter_matrix(K, support::Real, x::AbstractVector{Float64})
    n = length(x)
    I = Int[]; J = Int[]; V = Float64[]
    j_lo = 1
    for i in 1:n
        xi = x[i]
        while j_lo <= n && x[j_lo] < xi - support
            j_lo += 1
        end
        start = length(V)
        wsum = 0.0
        j = j_lo
        while j <= n
            δ = x[j] - xi
            δ > support && break
            w = Float64(K(δ))
            wsum += w
            if w != 0.0 || j == i
                push!(I, i); push!(J, j); push!(V, w)
            end
            j += 1
        end
        if wsum > 0.0
            for t in start+1:length(V)
                V[t] /= wsum
            end
        else
            resize!(I, start); resize!(J, start); resize!(V, start)
            push!(I, i); push!(J, i); push!(V, 1.0)
        end
    end
    return sparse(I, J, V, n, n)
end

# ∫_{-support}^{support} K(δ) dδ by composite Simpson (both kernels are smooth
# inside the support; 4000 panels give ≲ 1e-13 relative error).
function _kernel_integral(K, support::Real; npanels::Int = 4000)
    h = 2support / npanels
    s = Float64(K(-support)) + Float64(K(support))
    for m in 1:npanels-1
        s += (isodd(m) ? 4.0 : 2.0) * Float64(K(-support + m * h))
    end
    return s * h / 3
end

# W_q^{1/2}·(K/κ)·W_q^{1/2} from the upper triangle, mirrored (exactly
# symmetric); κ = ∫K over the support, so the discrete integral Σ_j K_ij w_j/κ
# is ≈ 1 in the interior (Springsteel's Lanczos kernel has K(0) = 1 and
# ∫K ≈ DX, the Gaussian ∫K = erf(n_sigma/√2)).
function _symmetric_filter_matrix(K, support::Real, x::AbstractVector{Float64},
                                  wq::AbstractVector{Float64})
    n = length(x)
    κ = _kernel_integral(K, support)
    κ > 0 || throw(ArgumentError("AxisFilter: kernel integral over the support is ≤ 0"))
    sw = sqrt.(wq) ./ sqrt(κ)
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:n
        for j in i:n
            δ = x[j] - x[i]
            δ > support && break
            k = Float64(K(δ))
            (k != 0.0 || j == i) || continue
            val = sw[i] * k * sw[j]
            push!(I, i); push!(J, j); push!(V, val)
            if j != i
                push!(I, j); push!(J, i); push!(V, val)
            end
        end
    end
    return sparse(I, J, V, n, n)
end

# y = A·x for a CSC matrix, allocation-free on views.
function _csc_mul!(y::AbstractVector{Float64}, A::SparseMatrixCSC{Float64,Int},
                   x::AbstractVector{Float64})
    fill!(y, 0.0)
    colptr = A.colptr; rowval = A.rowval; nzval = A.nzval
    @inbounds for j in 1:size(A, 2)
        xj = x[j]
        for p in colptr[j]:(colptr[j+1] - 1)
            y[rowval[p]] += nzval[p] * xj
        end
    end
    return y
end

struct _SparseMulKernel; A::SparseMatrixCSC{Float64,Int}; end
(k::_SparseMulKernel)(out, in, sc::PencilScratch) = (_csc_mul!(out, k.A, in); nothing)

# ─────────────────────────────────────────────────────────────────────────────
# MishFilter
# ─────────────────────────────────────────────────────────────────────────────

"""
    MishFilter(grid, varnames, filters::AbstractMatrix{<:Union{Nothing,Springsteel.AbstractFilter}};
               symmetric = false, nchunks = Threads.nthreads())
    MishFilter(grid, varnames, lengths_cells::AbstractMatrix{<:Real};
               kind = :gaussian, n_sigma = 4, symmetric = false, nchunks = Threads.nthreads())

Separable 3-D mish-space filter `F` on a variable-major mish vector (per
variable `(kDim, jDim, iDim)` column-major; `varnames` selects and orders the
variables of the RRR `grid`, as for [`SplineProjection`](@ref)). `filters` is
`3 × nvars` with rows `1 = k`, `2 = j`, `3 = i` (the convention of the spline
operators); `nothing` leaves that axis unfiltered, and a variable with no
filter at all passes through unchanged. Each entry becomes an
[`AxisFilter`](@ref) on that variable's axis spline.

The convenience form takes filter lengths in cells: `kind = :gaussian`
(`sigma = length`, truncated at `n_sigma`·σ) or `:lanczos` (`a = length`,
must be an integer); `0` or `NaN` means no filter on that axis. Use
[`filter_lengths_cells`](@ref) to convert lengths in metres.

`apply!` sweeps k, then j, then i; `apply_adjoint!` applies the transposes in
the order i, j, k. The input is never modified; input and output must not
alias (`ArgumentError`). Results do not depend on `nchunks`.

See also: [`axis_filter`](@ref), [`filter_length_cells`](@ref), [`has_filter`](@ref)
"""
struct MishFilter <: AbstractLinearOp
    geom::_SplineGridGeometry
    axes::Matrix{Union{Nothing,AxisFilter}}   # (3, nvars): [1,v]=k, [2,v]=j, [3,v]=i
    pool::Vector{PencilScratch}
    buf::Vector{Float64}                      # one variable's mish (ping-pong scratch)
end

function MishFilter(grid, varnames::AbstractVector{<:AbstractString},
                    filters::AbstractMatrix{<:Union{Nothing,Springsteel.AbstractFilter}};
                    symmetric::Bool = false, nchunks::Integer = Threads.nthreads())
    g = _spline_grid_geometry(grid, varnames)
    nv = _nvars(g)
    size(filters) == (3, nv) || throw(DimensionMismatch(
        "MishFilter: filters must be 3 × $(nv) (rows k, j, i; one column per variable), got $(size(filters))"))
    axes = Matrix{Union{Nothing,AxisFilter}}(nothing, 3, nv)
    for v in 1:nv, a in 1:3
        f = filters[a, v]
        f === nothing && continue
        axes[a, v] = AxisFilter(f, g.axes[a, v]; symmetric = symmetric)
    end
    return MishFilter(g, axes, _geometry_pool(g, nchunks), zeros(_nmish(g)))
end

function MishFilter(grid, varnames::AbstractVector{<:AbstractString},
                    lengths_cells::AbstractMatrix{<:Real};
                    kind::Symbol = :gaussian, n_sigma::Integer = 4,
                    symmetric::Bool = false, nchunks::Integer = Threads.nthreads())
    kind in (:gaussian, :lanczos) || throw(ArgumentError(
        "MishFilter: kind must be :gaussian or :lanczos, got :$(kind)"))
    filters = Matrix{Union{Nothing,Springsteel.AbstractFilter}}(nothing, size(lengths_cells))
    for idx in eachindex(IndexCartesian(), lengths_cells)
        L = lengths_cells[idx]
        (iszero(L) || isnan(L)) && continue
        (isfinite(L) && L > 0) || throw(ArgumentError(
            "MishFilter: filter length must be > 0 cells (0 or NaN for none), got $(L) at $(Tuple(idx))"))
        if kind === :gaussian
            filters[idx] = Springsteel.GaussianFilter(sigma = Float64(L), n_sigma = Int(n_sigma))
        else
            isinteger(L) || throw(ArgumentError(
                "MishFilter: Lanczos lobe count must be an integer number of cells, got $(L) at $(Tuple(idx))"))
            filters[idx] = Springsteel.LanczosFilter(a = Int(L))
        end
    end
    return MishFilter(grid, varnames, filters; symmetric = symmetric, nchunks = nchunks)
end

Base.size(M::MishFilter) = (n = _nvars(M.geom) * _nmish(M.geom); (n, n))

_axis_index(axis::Integer) = (1 <= axis <= 3 ? Int(axis) :
    throw(ArgumentError("axis must be 1 (k), 2 (j) or 3 (i), got $(axis)")))
_axis_index(axis::Symbol) = axis === :k ? 1 : axis === :j ? 2 : axis === :i ? 3 :
    throw(ArgumentError("axis must be :i, :j or :k, got :$(axis)"))

"""
    axis_filter(M::MishFilter, axis, v) -> Union{AxisFilter,Nothing}

The [`AxisFilter`](@ref) of variable index `v` along `axis` (`1`/`:k`,
`2`/`:j`, `3`/`:i`), or `nothing` if that axis is unfiltered.
"""
axis_filter(M::MishFilter, axis::Union{Integer,Symbol}, v::Integer) = M.axes[_axis_index(axis), v]

"""
    filter_length_cells(M::MishFilter, axis, v) -> Float64

Configured filter length (Gaussian `sigma` or Lanczos `a`, in cells) of
variable index `v` along `axis`; `0.0` if that axis is unfiltered.
"""
function filter_length_cells(M::MishFilter, axis::Union{Integer,Symbol}, v::Integer)
    af = axis_filter(M, axis, v)
    return af === nothing ? 0.0 : af.length_cells
end

"""
    has_filter(M::MishFilter) -> Bool

Whether any axis of any variable is filtered (otherwise `M` is the identity).
"""
has_filter(M::MishFilter) = any(!isnothing, M.axes)

"""
    filter_lengths_cells(grid, varnames, lengths_m::AbstractMatrix{<:Real}) -> Matrix{Float64}

Convert per-axis, per-variable filter lengths in metres (`3 × nvars`, rows
`k`, `j`, `i`) to cells by dividing by each axis's cell width `DX`. `0` and
`NaN` (no filter) pass through; negative or infinite lengths throw.
"""
function filter_lengths_cells(grid, varnames::AbstractVector{<:AbstractString},
                              lengths_m::AbstractMatrix{<:Real})
    g = _spline_grid_geometry(grid, varnames)
    nv = _nvars(g)
    size(lengths_m) == (3, nv) || throw(DimensionMismatch(
        "filter_lengths_cells: lengths must be 3 × $(nv) (rows k, j, i), got $(size(lengths_m))"))
    out = zeros(3, nv)
    for v in 1:nv, a in 1:3
        L = Float64(lengths_m[a, v])
        (isnan(L) || (isfinite(L) && L >= 0)) || throw(ArgumentError(
            "filter_lengths_cells: length must be ≥ 0 metres (or NaN), got $(L) at ($(a), $(v))"))
        out[a, v] = L / g.axes[a, v].sp.DX
    end
    return out
end

function _check_filter_args(M::MishFilter, out, in, what)
    n = size(M, 1)
    (length(out) == n && length(in) == n) || throw(DimensionMismatch(
        "MishFilter $(what): size $(size(M)), output $(length(out)), input $(length(in))"))
    Base.mightalias(out, in) && throw(ArgumentError(
        "MishFilter $(what): input and output must not alias"))
    return nothing
end

# Filter one variable: sweep the non-`nothing` axes in `order`, ping-ponging
# between `yv` and `bv` so the last sweep lands in `yv`; `xv` is only read.
function _filter_variable!(yv, bv, xv, M::MishFilter, v::Int, order::NTuple{3,Int},
                           transpose::Bool)
    g = M.geom
    dM = (g.kDim, g.jDim, g.iDim)
    nsw = 0
    for a in 1:3
        M.axes[a, v] === nothing || (nsw += 1)
    end
    if nsw == 0
        copyto!(yv, xv)
        return yv
    end
    cur, nxt = isodd(nsw) ? (yv, bv) : (bv, yv)
    first = true
    for a in order
        af = M.axes[a, v]
        af === nothing && continue
        kern = _SparseMulKernel(transpose ? af.Ft : af.F)
        if first
            sweep_axis!(kern, cur, xv, a, dM, dM, M.pool)
            first = false
        else
            sweep_axis!(kern, nxt, cur, a, dM, dM, M.pool)
            cur, nxt = nxt, cur
        end
    end
    return yv
end

function apply!(y::AbstractVector{Float64}, M::MishFilter, x::AbstractVector{Float64})
    _check_filter_args(M, y, x, "apply!")
    nm = _nmish(M.geom)
    bv = view(M.buf, 1:nm)
    for v in 1:_nvars(M.geom)
        r = _varrange(v, nm)
        _filter_variable!(view(y, r), bv, view(x, r), M, v, (1, 2, 3), false)
    end
    return y
end

function apply_adjoint!(x::AbstractVector{Float64}, M::MishFilter, y::AbstractVector{Float64})
    _check_filter_args(M, x, y, "apply_adjoint!")
    nm = _nmish(M.geom)
    bv = view(M.buf, 1:nm)
    for v in 1:_nvars(M.geom)
        r = _varrange(v, nm)
        _filter_variable!(view(x, r), bv, view(y, r), M, v, (3, 2, 1), true)
    end
    return x
end
