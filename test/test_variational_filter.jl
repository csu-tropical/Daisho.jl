# Tests for the variational mish-space filter (Phase 3): per-axis sparse
# filter matrices built from Springsteel's spline kernels, the separable 3-D
# MishFilter with its exact adjoint, and the full control-variable chain
# C = S∘D∘F.

using Random
using LinearAlgebra
using SparseArrays
using Springsteel

const _FILT_VARS = ["U", "V", "W"]

# Small RRR state grid with a different DX on every axis (i: 1000 m,
# j: 800 m, k: 500 m) so axis/DX mix-ups show up. W has a Dirichlet bottom.
function _filt_grid(; mubar::Int = 3, quadrature::Symbol = :gauss, cells = (6, 5, 4))
    kmin = Dict{String,Springsteel.BoundaryConditions}("default" => NaturalBC(),
                                                        "W" => DirichletBC(0.0))
    cfg = Daisho.SpringsteelGridConfig(geometry = "RRR", mubar = mubar,
        quadrature = quadrature,
        i = Daisho.SpringsteelAxisConfig(min = -3000.0, max = -3000.0 + 1000.0 * cells[1], cells = cells[1]),
        j = Daisho.SpringsteelAxisConfig(min = 0.0, max = 800.0 * cells[2], cells = cells[2]),
        k = Daisho.SpringsteelAxisConfig(min = 0.0, max = 500.0 * cells[3], cells = cells[3],
                                         bc_min = kmin))
    return Daisho.create_radar_grid(cfg, Dict("U" => 1, "V" => 2, "W" => 3))
end

_frelerr(a, b) = norm(a .- b) / max(norm(b), eps())

function _falloc_after_warmup(f, args...)
    f(args...)
    return @allocated f(args...)
end

# Template spline of `grid` for axis index a (1 = k, 2 = j, 3 = i) and column c.
function _filt_template(grid, a::Int, c::Int)
    basis = a == 1 ? grid.kbasis : (a == 2 ? grid.jbasis : grid.ibasis)
    return basis.data[1, 1, c]
end

# Independent (brute-force, dense) row weights and column sums of the
# row-normalised filter matrix. The window tests are Springsteel's verbatim:
# `x_j < x_i − support` (left) and `x_j − x_i > support` (right) are not exact
# mirrors in floating point, which matters for pairs at exactly the support
# distance (e.g. a 4σ = 6-cell Gaussian on a Gauss mish).
_in_window(xj, xi, support) = !(xj < xi - support) && !(xj - xi > support)
function _brute_column_sums(K, support, x)
    n = length(x)
    W = [sum(K(x[j] - x[i]) for j in 1:n if _in_window(x[j], x[i], support)) for i in 1:n]
    return [sum(K(x[j] - x[i]) / W[i] for i in 1:n if _in_window(x[j], x[i], support)) for j in 1:n]
end

@testset "Variational mish filter (Phase 3)" begin

    for (mubar, quadrature) in ((3, :gauss), (2, :regular))
    @testset "mubar=$(mubar), quadrature=$(quadrature)" begin
        rng = MersenneTwister(2)
        grid = _filt_grid(mubar = mubar, quadrature = quadrature)
        gp = grid.params
        @test isempty(gp.spline_filter)       # state grids never self-filter
        cols = [gp.vars[n] for n in _FILT_VARS]
        nm = gp.iDim * gp.jDim * gp.kDim
        nc = gp.b_iDim * gp.b_jDim * gp.b_kDim

        # ── 1. 1-D parity with Springsteel's _convolve_axis! ─────────────────
        @testset "AxisFilter parity axis=$(a)" for a in 1:3
            for c in cols, filt in (GaussianFilter(sigma = 1.5), LanczosFilter(a = 3),
                                    GaussianFilter(sigma = 0.6, n_sigma = 3), LanczosFilter(a = 2))
                ax = Daisho.AxisSpline(_filt_template(grid, a, c))
                af = Daisho.AxisFilter(filt, ax)
                K, support = Springsteel._spline_kernel(filt, ax.sp.DX)
                @test af.support == support
                @test af.length_cells == (filt isa GaussianFilter ? filt.sigma : Float64(filt.a))
                @test size(af.F) == (ax.Dim, ax.Dim)
                @test af.Ft == sparse(transpose(af.F))
                # Zero kernel values (Lanczos zero crossings/edges) are not stored,
                # the diagonal always is.
                @test all(!iszero, nonzeros(af.F))
                @test all(af.F[i, i] != 0 for i in 1:ax.Dim)
                for _ in 1:3
                    v = randn(rng, ax.Dim)
                    ref = zeros(ax.Dim)
                    Springsteel._convolve_axis!(ref, v, ax.mishPoints, K, support)
                    @test _frelerr(af.F * v, ref) < 1e-14
                end
                # Row normalisation: F·1 = 1 to round-off.
                @test maximum(abs.(af.F * ones(ax.Dim) .- 1)) < 1e-14
            end
        end

        # ── 2. 3-D parity with Springsteel's _filter_mish! ───────────────────
        @testset "3-D parity vs _filter_mish!" begin
            sf = Dict{String,Dict{Symbol,Springsteel.AbstractFilter}}(
                "U" => Dict{Symbol,Springsteel.AbstractFilter}(
                    :i => GaussianFilter(sigma = 1.5), :j => GaussianFilter(sigma = 0.8),
                    :k => LanczosFilter(a = 2)),
                "V" => Dict{Symbol,Springsteel.AbstractFilter}(:j => LanczosFilter(a = 3)),
                "W" => Dict{Symbol,Springsteel.AbstractFilter}(
                    :k => GaussianFilter(sigma = 1.0), :i => LanczosFilter(a = 3)))
            # Rebuild the params with spline_filter set (same trick as
            # Daisho._with_regular_out); basis/physical arrays are shared.
            d = Dict{Symbol,Any}(f => getfield(gp, f) for f in fieldnames(SpringsteelGridParameters))
            d[:spline_filter] = sf
            fgrid = typeof(grid)(SpringsteelGridParameters(; d...), grid.ibasis, grid.jbasis,
                                 grid.kbasis, grid.spectral, grid.physical)
            @test !isempty(fgrid.params.spline_filter)
            filters = Matrix{Union{Nothing,Springsteel.AbstractFilter}}(nothing, 3, 3)
            for (v, n) in enumerate(_FILT_VARS), (a, dir) in enumerate((:k, :j, :i))
                filters[a, v] = get(sf[n], dir, nothing)
            end
            MF = Daisho.MishFilter(grid, _FILT_VARS, filters)
            x = randn(rng, 3nm)
            for (v, n) in enumerate(_FILT_VARS)
                fgrid.physical[:, gp.vars[n], 1] .= x[(v-1)*nm+1:v*nm]
            end
            Springsteel._filter_mish!(fgrid)
            y = zeros(3nm)
            Daisho.apply!(y, MF, x)
            for (v, n) in enumerate(_FILT_VARS)
                @test _frelerr(y[(v-1)*nm+1:v*nm], fgrid.physical[:, gp.vars[n], 1]) < 1e-13
            end
            # The input vector is never modified.
            x2 = copy(x)
            Daisho.apply!(y, MF, x)
            @test x == x2
        end

        # ── 4. Adjointness, including the full control-variable chain ───────
        @testset "Adjointness" begin
            Lg = [1.5 0.8 0.0;
                  1.2 NaN 0.0;
                  0.7 2.0 0.0]           # V: j axis off; W: identity
            MG = Daisho.MishFilter(grid, _FILT_VARS, Lg)
            @test size(MG) == (3nm, 3nm)
            @test Daisho.adjoint_relative_error(MG; rng = rng) < 1e-12
            Ll = [3.0 2.0 0.0;
                  2.0 0.0 0.0;
                  1.0 3.0 0.0]
            ML = Daisho.MishFilter(grid, _FILT_VARS, Ll; kind = :lanczos)
            @test Daisho.adjoint_relative_error(ML; rng = rng) < 1e-12
            MS = Daisho.MishFilter(grid, _FILT_VARS, Lg; symmetric = true)
            @test Daisho.adjoint_relative_error(MS; rng = rng) < 1e-12

            # W (all axes nothing) passes through unchanged; V skips its j axis.
            x = randn(rng, 3nm)
            y = zeros(3nm)
            Daisho.apply!(y, MG, x)
            @test y[2nm+1:3nm] == x[2nm+1:3nm]
            xt = zeros(3nm)
            Daisho.apply_adjoint!(xt, MG, x)
            @test xt[2nm+1:3nm] == x[2nm+1:3nm]
            @test Daisho.axis_filter(MG, 2, 2) === nothing
            @test Daisho.axis_filter(MG, :j, 2) === nothing
            @test Daisho.axis_filter(MG, :i, 1) isa Daisho.AxisFilter
            @test Daisho.filter_length_cells(MG, 3, 1) == 0.7
            @test Daisho.filter_length_cells(MG, :k, 2) == 0.8
            @test Daisho.filter_length_cells(MG, :k, 3) == 0.0
            @test Daisho.has_filter(MG)
            M0 = Daisho.MishFilter(grid, _FILT_VARS, zeros(3, 3))
            @test !Daisho.has_filter(M0)
            Daisho.apply!(y, M0, x)
            @test y == x

            # Symmetric variant: apply! and apply_adjoint! agree (sweep order
            # differs, so only to round-off).
            ys = zeros(3nm); xs = zeros(3nm)
            Daisho.apply!(ys, MS, x); Daisho.apply_adjoint!(xs, MS, x)
            @test _frelerr(xs, ys) < 1e-13

            # Full control-variable chain C = S∘D∘F.
            S = Daisho.SplineProjection(grid, _FILT_VARS)
            D = Daisho.DiagonalScaling(Union{Float64,Vector{Float64}}[rand(rng, nm) .+ 0.5, 2.0, 0.3], nm)
            C = Daisho.ChainOp(S, D, MG)
            @test size(C) == (3nc, 3nm)
            @test Daisho.adjoint_relative_error(C; rng = rng) < 1e-12
            CL = Daisho.ChainOp(S, D, ML)
            @test Daisho.adjoint_relative_error(CL; rng = rng) < 1e-12
            # ChainOp(S, D, F) applies F, then D, then S.
            a1 = zeros(3nc); Daisho.apply!(a1, C, x)
            t1 = zeros(3nm); t2 = zeros(3nm); a2 = zeros(3nc)
            Daisho.apply!(t1, MG, x); Daisho.apply!(t2, D, t1); Daisho.apply!(a2, S, t2)
            @test a1 == a2
        end

        # ── 5. Smoothing sanity ─────────────────────────────────────────────
        @testset "Smoothing reduces variance" begin
            for (L, kind) in ((fill(1.5, 3, 3), :gaussian), (fill(3.0, 3, 3), :lanczos))
                MF = Daisho.MishFilter(grid, _FILT_VARS, L; kind = kind)
                x = randn(rng, 3nm)
                for v in 1:3
                    r = (v-1)*nm+1:v*nm
                    x[r] .-= sum(x[r]) / nm
                end
                y = zeros(3nm)
                Daisho.apply!(y, MF, x)
                @test norm(y) < 0.8 * norm(x)
            end
        end

        # ── 6. Threading determinism, allocations, aliasing ─────────────────
        @testset "Threading, allocations, aliasing" begin
            Lg = [1.5 0.8 0.0; 1.2 NaN 0.0; 0.7 2.0 0.0]
            nmulti = max(Threads.nthreads(), 3)
            M1 = Daisho.MishFilter(grid, _FILT_VARS, Lg; nchunks = 1)
            MN = Daisho.MishFilter(grid, _FILT_VARS, Lg; nchunks = nmulti)
            MD = Daisho.MishFilter(grid, _FILT_VARS, Lg)
            x = randn(rng, 3nm)
            y1 = zeros(3nm); yN = zeros(3nm)
            Daisho.apply!(y1, M1, x); Daisho.apply!(yN, MN, x)
            @test y1 == yN
            Daisho.apply_adjoint!(y1, M1, x); Daisho.apply_adjoint!(yN, MN, x)
            @test y1 == yN
            y = zeros(3nm)
            n1 = _falloc_after_warmup(Daisho.apply!, y, MD, x)
            n2 = _falloc_after_warmup(Daisho.apply_adjoint!, y, MD, x)
            n3 = _falloc_after_warmup(Daisho.apply!, y, M1, x)
            n4 = _falloc_after_warmup(Daisho.apply_adjoint!, y, M1, x)
            @info "MishFilter allocations (bytes, nthreads=$(Threads.nthreads()), mubar=$(mubar))" apply = n1 adjoint = n2 apply_1chunk = n3 adjoint_1chunk = n4
            bound = 200_000 * max(1.0, Threads.nthreads() / 8)
            @test n1 < bound
            @test n2 < bound
            @test n3 == 0
            @test n4 == 0
            @test_throws ArgumentError Daisho.apply!(x, M1, x)
            @test_throws ArgumentError Daisho.apply_adjoint!(x, M1, x)
            @test_throws DimensionMismatch Daisho.apply!(zeros(3nm - 1), M1, x)
            @test_throws DimensionMismatch Daisho.apply_adjoint!(y, M1, zeros(nm))
        end

        # ── 7. Metres → cells, validation ───────────────────────────────────
        @testset "filter_lengths_cells and validation" begin
            Lm = [1000.0 250.0 0.0;       # k: DX = 500
                  1600.0 NaN   400.0;     # j: DX = 800
                  3000.0 500.0 0.0]       # i: DX = 1000
            Lc = Daisho.filter_lengths_cells(grid, _FILT_VARS, Lm)
            @test Lc isa Matrix{Float64}
            @test Lc[1, 1] == 2.0 && Lc[1, 2] == 0.5 && Lc[1, 3] == 0.0
            @test Lc[2, 1] == 2.0 && isnan(Lc[2, 2]) && Lc[2, 3] == 0.5
            @test Lc[3, 1] == 3.0 && Lc[3, 2] == 0.5 && Lc[3, 3] == 0.0
            @test_throws DimensionMismatch Daisho.filter_lengths_cells(grid, _FILT_VARS, zeros(3, 2))
            @test_throws ArgumentError Daisho.filter_lengths_cells(grid, _FILT_VARS, fill(-1.0, 3, 3))

            ax = Daisho.AxisSpline(_filt_template(grid, 3, cols[1]))
            @test_throws ArgumentError Daisho.AxisFilter(SpectralFilter(low_pass = 3), ax)
            sfm = Matrix{Union{Nothing,Springsteel.AbstractFilter}}(nothing, 3, 3)
            sfm[2, 1] = SpectralFilter(low_pass = 3)
            @test_throws ArgumentError Daisho.MishFilter(grid, _FILT_VARS, sfm)
            @test_throws ArgumentError Daisho.AxisFilter(GaussianFilter(sigma = 0.0), ax)
            @test_throws ArgumentError Daisho.MishFilter(grid, _FILT_VARS, fill(1.5, 3, 3); kind = :lanczos)
            @test_throws ArgumentError Daisho.MishFilter(grid, _FILT_VARS, fill(1.5, 3, 3); kind = :boxcar)
            @test_throws ArgumentError Daisho.MishFilter(grid, _FILT_VARS, fill(-1.0, 3, 3))
            @test_throws DimensionMismatch Daisho.MishFilter(grid, _FILT_VARS, ones(3, 2))
            @test_throws DimensionMismatch Daisho.MishFilter(grid, _FILT_VARS, ones(2, 3))
        end
    end
    end

    # ── 3/5. Boundary behaviour on a long axis (40 cells along i) ────────────
    for (mubar, quadrature) in ((3, :gauss), (2, :regular))
    @testset "Long-axis boundary behaviour mubar=$(mubar), quadrature=$(quadrature)" begin
        grid = _filt_grid(mubar = mubar, quadrature = quadrature, cells = (40, 3, 3))
        ax = Daisho.AxisSpline(_filt_template(grid, 3, grid.params.vars["U"]))
        x = ax.mishPoints
        n = ax.Dim
        xmin, xmax = ax.sp.xmin, ax.sp.xmax
        dist = min.(x .- xmin, xmax .- x)
        o = ones(n)

        for filt in (GaussianFilter(sigma = 1.5), LanczosFilter(a = 3))
            af = Daisho.AxisFilter(filt, ax)
            K, support = Springsteel._spline_kernel(filt, ax.sp.DX)
            @test maximum(abs.(af.F * o .- 1)) < 1e-14
            cs = af.Ft * o
            ref = _brute_column_sums(K, support, x)
            @test _frelerr(cs, ref) < 1e-13
            near = dist .< support
            @test abs(cs[1] - 1) > 1e-3 && abs(cs[end] - 1) > 1e-3
            @test maximum(abs.(cs[near] .- 1)) > 1e-3
            far = dist .> 2support
            @test count(far) > 10
            dev_far = maximum(abs.(cs[far] .- 1))
            @info "Fᵀ·1 − 1 beyond 2·support ($(nameof(typeof(filt))), $(quadrature))" dev_far
            if filt isa GaussianFilter
                if quadrature == :regular
                    @test dev_far < 1e-12
                else
                    @test dev_far < 1e-4   # 4σ-truncated kernel on a non-uniform mish (≈ 2.5e-5)
                end
            end
            # F − Fᵀ = K_ij(1/W_i − 1/W_j): nonzero, i.e. F is not symmetric.
            @test norm(af.F - af.Ft) > 1e-3 * norm(af.F)
        end

        # With a wide truncation (n_sigma = 8) the row weights are uniform to
        # round-off in the interior even on the Gauss mish.
        af8 = Daisho.AxisFilter(GaussianFilter(sigma = 1.5, n_sigma = 8), ax)
        far8 = dist .> 2af8.support
        count(far8) > 0 && @test maximum(abs.((af8.Ft * o)[far8] .- 1)) < 1e-12

        # Linear field unchanged away from the boundaries (Gaussian). The
        # default 4σ truncation is exact only on a uniform mish, where each
        # window is symmetric about its centre; the Gauss mish needs a wider
        # truncation (n_sigma = 8) to reach round-off.
        lin = 3.0 .+ 0.002 .* x
        for af in (Daisho.AxisFilter(GaussianFilter(sigma = 1.5, n_sigma = quadrature == :regular ? 4 : 8), ax),)
            inner = dist .> af.support
            @test maximum(abs.((af.F * lin)[inner] .- lin[inner])) < 1e-10 * maximum(abs, lin)
        end

        # Symmetric variant: exactly symmetric, no row normalisation.
        for filt in (GaussianFilter(sigma = 1.5), LanczosFilter(a = 3))
            afs = Daisho.AxisFilter(filt, ax; symmetric = true)
            @test afs.F == sparse(transpose(afs.F))
            @test issymmetric(afs.F)
            @test afs.Ft == afs.F
            inner = dist .> afs.support
            sw = sqrt.(ax.w)
            # F_sym·√w ≈ √w (quadrature of ∫K = 1) in the interior on any mish.
            dev_sqrtw = maximum(abs.((afs.F * sw)[inner] ./ sw[inner] .- 1))
            dev_one = maximum(abs.((afs.F * o)[inner] .- 1))
            @info "Symmetric filter interior deviation ($(nameof(typeof(filt))), $(quadrature))" dev_sqrtw dev_one
            @test dev_sqrtw < 5e-4                         # observed ≤ 1.3e-4
            quadrature == :regular && @test dev_one < 5e-4 # uniform weights ⇒ constants kept
            quadrature == :gauss && @test dev_one > 0.05   # Gauss weights ⇒ F_sym·1 ≠ 1 (≈ 0.15)
        end
    end
    end
end
