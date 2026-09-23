# Pencil layouts and the threaded pencil-sweep driver.
#
# Layouts (fixed by the variational plan):
# - mish values per variable: column-major (kDim, jDim, iDim), k fastest —
#   identical to Springsteel's physical flat order
#   `(i-1)·jDim·kDim + (j-1)·kDim + k`;
# - Daisho-owned spline coefficients per variable: column-major
#   (b_kDim, b_jDim, b_iDim), k fastest (NOT Springsteel's i-fastest spectral);
# - multi-variable vectors are variable-major.
#
# A 1-D transform along one axis of a 3-D array acts on "pencils": the lines
# of the array along that axis. `sweep_axis!` gathers each pencil into a
# per-chunk scratch buffer, applies a 1-D kernel, and scatters the result into
# the destination array, whose extent along the swept axis may differ from the
# source (mish Dim ↔ coefficient bDim). Every pencil writes a disjoint part of
# the destination, so results are bit-identical for any chunk count.

"""
    PencilScratch(maxlen, maxinterior)

Per-chunk scratch for pencil sweeps: `in`/`out` hold the gathered input and
kernel output pencils, `tmp` and `extra` are spare pencil-length buffers for
multi-step kernels, and `Min`/`Mout` are interior-space (Γ-folded) buffers for
the SA solve. `maxlen` must bound every `Dim`/`bDim` and `maxinterior` every
`Minterior` of the axes the scratch is used with.
"""
struct PencilScratch
    in::Vector{Float64}
    out::Vector{Float64}
    tmp::Vector{Float64}
    extra::Vector{Float64}
    Min::Vector{Float64}
    Mout::Vector{Float64}
end

PencilScratch(maxlen::Integer, maxinterior::Integer) =
    PencilScratch(zeros(maxlen), zeros(maxlen), zeros(maxlen), zeros(maxlen),
                  zeros(maxinterior), zeros(maxinterior))

"""
    scratch_pool(nchunks, maxlen, maxinterior) -> Vector{PencilScratch}

One [`PencilScratch`](@ref) per chunk of a threaded pencil sweep.
"""
function scratch_pool(nchunks::Integer, maxlen::Integer, maxinterior::Integer)
    nchunks >= 1 || throw(ArgumentError("nchunks must be ≥ 1, got $(nchunks)"))
    return [PencilScratch(maxlen, maxinterior) for _ in 1:nchunks]
end

"""
    sweep_axis!(f, dst, src, axis, dims_src, dims_dst, pool) -> dst

Apply the 1-D kernel `f(out, in, scratch)` to every pencil along `axis`
(`1 = k`, `2 = j`, `3 = i`) of the column-major 3-D array stored flat in `src`
with dimensions `dims_src`, writing into `dst` with dimensions `dims_dst`.
`dims_src` and `dims_dst` may differ only along `axis`; `in` and `out` are
views of the chunk's scratch of exactly those lengths. Pencils are split into
`length(pool)` contiguous chunks (one `PencilScratch` each); chunks `2:n` run
as `Threads.@spawn` tasks while chunk 1 runs on the calling task, and a single
chunk runs inline with no task at all (allocation-free).
"""
function sweep_axis!(f::F, dst::AbstractVector{Float64}, src::AbstractVector{Float64},
                     axis::Int, dims_src::NTuple{3,Int}, dims_dst::NTuple{3,Int},
                     pool::Vector{PencilScratch}) where {F}
    1 <= axis <= 3 || throw(ArgumentError("sweep_axis!: axis must be 1, 2, or 3; got $(axis)"))
    for d in 1:3
        d == axis && continue
        dims_src[d] == dims_dst[d] || throw(DimensionMismatch(
            "sweep_axis!: dims_src $(dims_src) and dims_dst $(dims_dst) differ off axis $(axis)"))
    end
    length(src) == prod(dims_src) || throw(DimensionMismatch(
        "sweep_axis!: length(src) = $(length(src)) ≠ prod($(dims_src))"))
    length(dst) == prod(dims_dst) || throw(DimensionMismatch(
        "sweep_axis!: length(dst) = $(length(dst)) ≠ prod($(dims_dst))"))
    Ls = dims_src[axis]
    Ld = dims_dst[axis]
    pre = axis == 1 ? 1 : (axis == 2 ? dims_src[1] : dims_src[1] * dims_src[2])
    npencils = prod(dims_src) ÷ Ls
    nchunks = min(length(pool), npencils)
    if nchunks <= 1
        _sweep_range!(f, dst, src, pre, Ls, Ld, 1:npencils, pool[1])
    else
        # Chunks 2:nchunks run as tasks; chunk 1 runs on the calling task.
        @sync begin
            for c in 2:nchunks
                lo = div((c - 1) * npencils, nchunks) + 1
                hi = div(c * npencils, nchunks)
                sc = pool[c]
                Threads.@spawn _sweep_range!(f, dst, src, pre, Ls, Ld, lo:hi, sc)
            end
            _sweep_range!(f, dst, src, pre, Ls, Ld, 1:div(npencils, nchunks), pool[1])
        end
    end
    return dst
end

function _sweep_range!(f::F, dst, src, pre::Int, Ls::Int, Ld::Int,
                       pencils::UnitRange{Int}, sc::PencilScratch) where {F}
    pin = view(sc.in, 1:Ls)
    pout = view(sc.out, 1:Ld)
    @inbounds for p in pencils
        r, q = divrem(p - 1, pre)   # q: offset within the pre-axis block, r: post-axis index
        bs = 1 + q + pre * Ls * r
        bd = 1 + q + pre * Ld * r
        for t in 1:Ls
            pin[t] = src[bs + pre * (t - 1)]
        end
        f(pout, pin, sc)
        for t in 1:Ld
            dst[bd + pre * (t - 1)] = pout[t]
        end
    end
    return nothing
end
