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
#
# Variance normalisation (the solver default, decided 2026-09-23). The
# row-normalised F shrinks white noise: (FFᵀ)_ii = Σ_j F_ij² ≪ 1, so D·F·ξ
# with ξ ~ N(0, I) would have a standard deviation far below the configured
# σ_b (≈ 5% of it for 1-cell lengths on a mubar-3 mish). The solver therefore
# uses, per axis,
#
#     F̃ = N⁻¹F,    N = diag(√(Σ_j F_ij²)),
#
# so diag(F̃F̃ᵀ) = 1 exactly at every mish point. The 3-D filter is the tensor
# product of its axis filters, hence diag(F̃₃F̃₃ᵀ) = Π_axes diag(F̃_a F̃_aᵀ) = 1
# exactly too, and B = S D F̃F̃ᵀ D Sᵀ has per-point variance σ_b² before the
# spline projection. The correlation shape of F̃F̃ᵀ is unchanged (the
# √2-widened kernel); F̃'s row sums are no longer 1 (≈ √(2√π·s) in the
# interior for a Gaussian of s mish points on a uniform mish). The `:row`
# form stays available for Springsteel parity and for mean-preserving
# smoothing of fields.

const _SpringsteelSplineFilter = Union{Springsteel.GaussianFilter,Springsteel.LanczosFilter}

"""
    AxisFilter(filter::Springsteel.AbstractFilter, ax::AxisSpline;
               normalisation = :variance, symmetric = false)

Sparse `Dim×Dim` matrix form of a Springsteel spline-direction filter
(`GaussianFilter` or `LanczosFilter`; `sigma`/`a` in cell widths of
`ax.sp.DX`) on the mish points of `ax`, with its materialised transpose `Ft`
so both directions are CSC products.

- `symmetric = false` (default) builds the row-normalised convolution
  `F_row[i,j] = K(x_j − x_i)/W_i` within the kernel support, exactly
  Springsteel's `_convolve_axis!` (`F_row·1 = 1`; a row whose weight sum is
  ≤ 0 is the identity row; not symmetric: see the file header), and then
  applies `normalisation`:
  - `:variance` (default, the solver's background-error filter):
    `F = N⁻¹F_row` with `N = diag(√(Σ_j F_row[i,j]²))`, so
    `diag(F Fᵀ) = 1` exactly at every mish point (white noise keeps unit
    variance; `sigma` of `D` is then the background-error standard
    deviation). Row sums are no longer 1.
  - `:row`: `F = F_row` (Springsteel parity; mean-preserving smoothing).
- `symmetric = true` (experimental; `normalisation` is ignored and the
  stored `normalisation` is `:none` — this variant is NOT
  variance-normalised): `F = W_q^{1/2}·(K/κ)·W_q^{1/2}` with the SB quadrature
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
(Gaussian `sigma` or Lanczos `a`, in cells), `symmetric`, `normalisation`
(`:variance`, `:row`, or `:none` for the symmetric variant).
"""
struct AxisFilter
    F::SparseMatrixCSC{Float64,Int}
    Ft::SparseMatrixCSC{Float64,Int}
    support::Float64
    length_cells::Float64
    symmetric::Bool
    normalisation::Symbol
end

const _FILTER_NORMALISATIONS = (:variance, :row)

function _check_normalisation(normalisation::Symbol)
    normalisation in _FILTER_NORMALISATIONS || throw(ArgumentError(
        "filter normalisation must be :variance or :row, got :$(normalisation)"))
    return normalisation
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

function AxisFilter(filter::Springsteel.AbstractFilter, ax::AxisSpline;
                    normalisation::Symbol = :variance, symmetric::Bool = false)
    _check_normalisation(normalisation)
    _check_spline_filter(filter)
    K, support = Springsteel._spline_kernel(filter, ax.sp.DX)
    x = ax.mishPoints
    issorted(x) || throw(ArgumentError("AxisFilter: mish points must be non-decreasing"))
    if symmetric
        F = _symmetric_filter_matrix(K, support, x, ax.w)
        return AxisFilter(F, copy(F), Float64(support), _filter_length(filter), true, :none)
    end
    F = _row_normalised_filter_matrix(K, support, x)
    normalisation === :variance && _variance_normalise!(F)
    return AxisFilter(F, sparse(transpose(F)), Float64(support), _filter_length(filter),
                      false, normalisation)
end

# Rescale each row i of F by 1/√(Σ_j F_ij²) in place, so diag(F Fᵀ) = 1.
# Every row has a nonzero diagonal (identity rows included), so no row is zero.
function _variance_normalise!(F::SparseMatrixCSC{Float64,Int})
    n = size(F, 1)
    ss = zeros(n)
    rv = rowvals(F); nz = nonzeros(F)
    @inbounds for p in eachindex(nz)
        ss[rv[p]] += nz[p]^2
    end
    all(>(0.0), ss) || error("AxisFilter: a filter row has zero norm (internal invariant)")
    @inbounds for p in eachindex(nz)
        nz[p] /= sqrt(ss[rv[p]])
    end
    return F
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

# y += A·x for a CSC matrix, allocation-free on views (SparseArrays' `mul!`
# allocates on a view input on Julia 1.10).
function _csc_muladd!(y::AbstractVector{Float64}, A::SparseMatrixCSC{Float64,Int},
                      x::AbstractVector{Float64})
    colptr = A.colptr; rowval = A.rowval; nzval = A.nzval
    @inbounds for j in 1:size(A, 2)
        xj = x[j]
        for p in colptr[j]:(colptr[j+1] - 1)
            y[rowval[p]] += nzval[p] * xj
        end
    end
    return y
end

# y = A·x for a CSC matrix, allocation-free on views.
_csc_mul!(y::AbstractVector{Float64}, A::SparseMatrixCSC{Float64,Int}, x::AbstractVector{Float64}) =
    _csc_muladd!(fill!(y, 0.0), A, x)

struct _SparseMulKernel; A::SparseMatrixCSC{Float64,Int}; end
(k::_SparseMulKernel)(out, in, sc::PencilScratch) = (_csc_mul!(out, k.A, in); nothing)

# ─────────────────────────────────────────────────────────────────────────────
# MishFilter
# ─────────────────────────────────────────────────────────────────────────────

"""
    MishFilter(grid, varnames, filters::AbstractMatrix{<:Union{Nothing,Springsteel.AbstractFilter}};
               normalisation = :variance, symmetric = false, nchunks = Threads.nthreads())
    MishFilter(grid, varnames, lengths_cells::AbstractMatrix{<:Real};
               kind = :gaussian, n_sigma = 4, normalisation = :variance,
               symmetric = false, nchunks = Threads.nthreads())

Separable 3-D mish-space filter `F` on a variable-major mish vector (per
variable `(kDim, jDim, iDim)` column-major; `varnames` selects and orders the
variables of the RRR `grid`, as for [`SplineProjection`](@ref)). `filters` is
`3 × nvars` with rows `1 = k`, `2 = j`, `3 = i` (the convention of the spline
operators); `nothing` leaves that axis unfiltered, and a variable with no
filter at all passes through unchanged. Each entry becomes an
[`AxisFilter`](@ref) on that variable's axis spline with the given
`normalisation`:

- `:variance` (default; the background-error filter of the solver): every
  axis filter is variance-normalised, so the tensor product satisfies
  `diag(F Fᵀ) = Π_axes diag(F_a F_aᵀ) = 1` exactly at every mish point —
  white control noise keeps unit variance and `D`'s σ is the
  background-error standard deviation (before the spline projection).
- `:row`: Springsteel's row-normalised convolution (`F·1 = 1`), for
  Springsteel parity and mean-preserving smoothing of fields.

`symmetric = true` (experimental) ignores `normalisation` and is not
variance-normalised ([`filter_normalisation`](@ref) then returns `:none`).

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
    normalisation::Symbol                     # :variance, :row, or :none (symmetric)
end

function MishFilter(grid, varnames::AbstractVector{<:AbstractString},
                    filters::AbstractMatrix{<:Union{Nothing,Springsteel.AbstractFilter}};
                    normalisation::Symbol = :variance, symmetric::Bool = false,
                    nchunks::Integer = Threads.nthreads())
    _check_normalisation(normalisation)
    g = _spline_grid_geometry(grid, varnames)
    nv = _nvars(g)
    size(filters) == (3, nv) || throw(DimensionMismatch(
        "MishFilter: filters must be 3 × $(nv) (rows k, j, i; one column per variable), got $(size(filters))"))
    axes = Matrix{Union{Nothing,AxisFilter}}(nothing, 3, nv)
    for v in 1:nv, a in 1:3
        f = filters[a, v]
        f === nothing && continue
        axes[a, v] = AxisFilter(f, g.axes[a, v]; normalisation = normalisation,
                                symmetric = symmetric)
    end
    return MishFilter(g, axes, _geometry_pool(g, nchunks), zeros(_nmish(g)),
                      symmetric ? :none : normalisation)
end

function MishFilter(grid, varnames::AbstractVector{<:AbstractString},
                    lengths_cells::AbstractMatrix{<:Real};
                    kind::Symbol = :gaussian, n_sigma::Integer = 4,
                    normalisation::Symbol = :variance,
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
    return MishFilter(grid, varnames, filters; normalisation = normalisation,
                      symmetric = symmetric, nchunks = nchunks)
end

"""
    filter_normalisation(M::MishFilter) -> Symbol

`:variance` (every axis filter variance-normalised, `diag(F Fᵀ) = 1`),
`:row` (row-normalised, `F·1 = 1`), or `:none` (the experimental symmetric
variant).
"""
filter_normalisation(M::MishFilter) = M.normalisation

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
