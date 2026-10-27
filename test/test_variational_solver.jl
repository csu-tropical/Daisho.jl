# Tests for the variational observation operators, incremental cost function,
# and Gauss–Newton/CG solver (Phase 4).

using Random
using LinearAlgebra
using SparseArrays
using Springsteel
using Springsteel: CubicBSpline

const _SOLV_VARS = ["U", "V", "W"]

# RRR state grid over [0, Lx] × [0, Ly] × [0, Lz] metres; W has a Dirichlet
# (R1T0) bottom, natural BCs elsewhere.
function _solv_grid(; cells = (4, 3, 3), extent = (4000.0, 3000.0, 3000.0), mubar::Int = 3)
    kmin = Dict{String,Springsteel.BoundaryConditions}("default" => NaturalBC(),
                                                        "W" => DirichletBC(0.0))
    cfg = Daisho.SpringsteelGridConfig(geometry = "RRR", mubar = mubar, quadrature = :gauss,
        i = Daisho.SpringsteelAxisConfig(min = 0.0, max = extent[1], cells = cells[1]),
        j = Daisho.SpringsteelAxisConfig(min = 0.0, max = extent[2], cells = cells[2]),
        k = Daisho.SpringsteelAxisConfig(min = 0.0, max = extent[3], cells = cells[3],
                                         bc_min = kmin))
    return Daisho.create_radar_grid(cfg, Dict("U" => 1, "V" => 2, "W" => 3))
end

_tiny_grid() = _solv_grid()
_large_grid() = _solv_grid(cells = (10, 10, 6), extent = (10_000.0, 10_000.0, 6000.0))

# Mish coordinates (x, y, z) of every mish flat index (k fastest).
function _mish_xyz(grid)
    g = Daisho._spline_grid_geometry(grid, _SOLV_VARS)
    xs = g.axes[3, 1].mishPoints; ys = g.axes[2, 1].mishPoints; zs = g.axes[1, 1].mishPoints
    n = length(xs) * length(ys) * length(zs)
    X = zeros(n); Y = zeros(n); Z = zeros(n)
    p = 0
    for i in eachindex(xs), j in eachindex(ys), k in eachindex(zs)
        p += 1
        X[p] = xs[i]; Y[p] = ys[j]; Z[p] = zs[k]
    end
    return X, Y, Z, g
end

# Control chain C = S∘D∘F with Gaussian filter lengths (cells) L[axis, var].
# These solver-algebra tests were tuned (σ_b, CG caps and tolerances) with the
# row-normalised filter, whose effective background σ is ≈ 5–20% of `sigma_b`;
# they pin `normalisation = :row` so the conditioning they were designed
# around is unchanged (the algebra under test holds for any SPD B). The
# solver's default variance-normalised F is covered by
# test_variational_filter.jl and test_variational_solve.jl.
function _solv_chain(grid, sigma_b, L; nchunks = Threads.nthreads(), normalisation = :row)
    nm = grid.params.iDim * grid.params.jDim * grid.params.kDim
    S = Daisho.SplineProjection(grid, _SOLV_VARS; nchunks = nchunks)
    σ = sigma_b isa AbstractVector ? sigma_b : fill(Float64(sigma_b), 3)
    D = Daisho.DiagonalScaling(Union{Float64,Vector{Float64}}[σ...], nm)
    F = Daisho.MishFilter(grid, _SOLV_VARS, L; nchunks = nchunks, normalisation = normalisation)
    return Daisho.ChainOp(S, D, F), S, D, F
end

_srelerr(a, b) = norm(a .- b) / max(norm(b), eps())

# H∘C as a linear operator (control → observations).
struct _HCOp{TC} <: Daisho.AbstractLinearOp
    set::Daisho.ObsSet
    C::TC
    buf::Vector{Float64}
end
_HCOp(set, C) = _HCOp(set, C, zeros(size(C, 1)))
Base.size(op::_HCOp) = (size(op.set, 1), size(op.C, 2))
Daisho.apply!(y::AbstractVector{Float64}, op::_HCOp, x::AbstractVector{Float64}) =
    Daisho.apply_H!(y, op.set, Daisho.apply!(op.buf, op.C, x))
Daisho.apply_adjoint!(x::AbstractVector{Float64}, op::_HCOp, y::AbstractVector{Float64}) =
    Daisho.apply_adjoint!(x, op.C, Daisho.apply_Ht!(op.buf, op.set, y))

# Random mish backend: nobs obs over random points (several repeated), random
# terms over all vars and slots.
function _rand_mish_backend(rng, nm, nobs; kind = :point)
    pts = rand(rng, 1:nm, nobs)
    pts[2] = pts[1]; pts[3] = pts[1]; pts[5] = pts[4]      # shared points
    terms = [[(rand(rng, 1:3), rand(rng, 1:4), randn(rng)) for _ in 1:rand(rng, 1:4)] for _ in 1:nobs]
    return Daisho.MishObsBackend(kind, (3, nm), pts, randn(rng, nobs), 0.5 .+ rand(rng, nobs), terms)
end

function _rand_scattered_backend(rng, grid, nobs; ext)
    xyz = hcat(ext[1] .* rand(rng, nobs), ext[2] .* rand(rng, nobs), ext[3] .* rand(rng, nobs))
    terms = [[(rand(rng, 1:3), rand(rng, 1:4), randn(rng)) for _ in 1:rand(rng, 1:3)] for _ in 1:nobs]
    return Daisho.ScatteredObsBackend(:point, grid, _SOLV_VARS, xyz, randn(rng, nobs),
                                      0.5 .+ rand(rng, nobs), terms)
end

# Wind-speed model: sample = (u, v) at the point.
function _speed_model(sample, m)
    u, v = sample[1], sample[2]
    s = max(sqrt(u^2 + v^2), 1e-6)
    return s, (u / s, v / s)
end

function _speed_backend(nm, y, rinv)
    n = length(y)
    return Daisho.MishObsBackend(:wind_speed, (3, nm), collect(1:n), y, rinv,
                                 [[(1, 1, 1.0), (2, 1, 0.0)] for _ in 1:n]; model = _speed_model)
end

# Analysed mish values (slot 1) of every variable from coefficients a.
function _mish_values(grid, a)
    E = Daisho.SplineEvaluation(grid, _SOLV_VARS, [(1, 1), (2, 1), (3, 1)])
    out = zeros(size(E, 1))
    Daisho.apply!(out, E, a)
    nm = length(out) ÷ 3
    return [out[(v-1)*nm+1:v*nm] for v in 1:3]
end

_rms(x) = sqrt(sum(abs2, x) / length(x))

@testset "Variational solver (Phase 4)" begin
    t_start = time()

    # ── 1–3. Adjointness, consistency, Hessian (tiny grid) ──────────────────
    @testset "Operators on the tiny grid" begin
        rng = MersenneTwister(3)
        grid = _tiny_grid()
        gp = grid.params
        nm = gp.iDim * gp.jDim * gp.kDim
        nc = gp.b_iDim * gp.b_jDim * gp.b_kDim
        ext = (4000.0, 3000.0, 3000.0)
        C, S, D, F = _solv_chain(grid, [2.0, 1.5, 0.7], fill(1.5, 3, 3))

        bm = _rand_mish_backend(rng, nm, 60)
        bs = _rand_scattered_backend(rng, grid, 40; ext = ext)
        @test Daisho.nobs(bm) == 60 && Daisho.nobs(bs) == 40
        @test issorted(bm.point)
        @test length(bm.term_ptr) == 61
        @test Daisho.is_linear(bm) && Daisho.is_linear(bs)
        @test Daisho.slots_needed(bs) == Tuple{Int,Int}[]
        @test allunique(Daisho.slots_needed(bm))

        @testset "Adjointness" begin
            setm = Daisho.ObsSet(grid, _SOLV_VARS, [bm])
            sets = Daisho.ObsSet(grid, _SOLV_VARS, [bs])
            setb = Daisho.ObsSet(grid, _SOLV_VARS, Daisho.AbstractObsBackend[bm, bs])
            @test size(setb) == (100, 3nc)
            @test setb.ranges == [1:60, 61:100]
            @test sets.ev === nothing
            for set in (setm, sets, setb)
                @test Daisho.adjoint_relative_error(set; rng = rng) < 1e-12
                @test Daisho.adjoint_relative_error(_HCOp(set, C); rng = rng) < 1e-12
            end
            # apply_Ht! overwrites its output.
            y = randn(rng, 100)
            a1 = zeros(3nc); a2 = fill(7.0, 3nc)
            Daisho.apply_Ht!(a1, setb, y); Daisho.apply_Ht!(a2, setb, y)
            @test a1 == a2
            a1 = zeros(3nc); a2 = fill(7.0, 3nc)
            Daisho.apply_Ht!(a1, sets, y[61:100]); Daisho.apply_Ht!(a2, sets, y[61:100])
            @test a1 == a2
            # Linear backends: evaluate! == apply_H!, linearize! is a no-op.
            a = randn(rng, 3nc)
            y1 = zeros(100); y2 = zeros(100)
            Daisho.apply_H!(y1, setb, a); Daisho.evaluate!(y2, setb, a)
            @test y1 == y2
            w0 = copy(bm.term_w)
            Daisho.linearize!(setb, a)
            @test bm.term_w == w0
            # The mish obs equal a direct slab evaluation.
            E = Daisho.SplineEvaluation(grid, _SOLV_VARS, [(v, s) for v in 1:3 for s in 1:4])
            slabs = zeros(size(E, 1)); Daisho.apply!(slabs, E, a)
            ref = [sum(bm.term_w[t] * slabs[(4 * (bm.slots[bm.term_slab[t]][1] - 1) +
                                             bm.slots[bm.term_slab[t]][2] - 1) * nm + bm.point[m]]
                       for t in bm.term_ptr[m]:bm.term_ptr[m+1]-1) for m in 1:60]
            @test _srelerr(y1[1:60], ref) < 1e-13
            # Sorted order and perm bookkeeping.
            @test Daisho.obs_values(setb)[1:60] == bm.y
            @test Daisho.obs_rinv(setb)[61:100] == bs.rinv
        end

        @testset "Scattered at a mish point == mish obs" begin
            X, Y, Z, _ = _mish_xyz(grid)
            pts = [1, 17, nm ÷ 2, nm - 3, nm]
            terms = [[(1, 1, 0.7), (2, 2, -1.3), (3, 4, 0.4), (1, 3, 2.0)] for _ in pts]
            bmp = Daisho.MishObsBackend(:point, grid, _SOLV_VARS, pts, zeros(5), ones(5), terms)
            bsp = Daisho.ScatteredObsBackend(:point, grid, _SOLV_VARS, hcat(X[pts], Y[pts], Z[pts]),
                                             zeros(5), ones(5), terms)
            sm = Daisho.ObsSet(grid, _SOLV_VARS, [bmp]); ss = Daisho.ObsSet(grid, _SOLV_VARS, [bsp])
            for _ in 1:3
                a = randn(rng, 3nc)
                ym = zeros(5); ysc = zeros(5)
                Daisho.apply_H!(ym, sm, a); Daisho.apply_H!(ysc, ss, a)
                @test _srelerr(ysc, ym) < 1e-12
            end
            # Outside the domain → ArgumentError; bad terms/points → ArgumentError.
            @test_throws ArgumentError Daisho.ScatteredObsBackend(:point, grid, _SOLV_VARS,
                [4000.1 10.0 10.0], [0.0], [1.0], [[(1, 1, 1.0)]])
            @test_throws ArgumentError Daisho.ScatteredObsBackend(:point, grid, _SOLV_VARS,
                [10.0 10.0 -1.0], [0.0], [1.0], [[(1, 1, 1.0)]])
            @test_throws ArgumentError Daisho.MishObsBackend(:point, (3, nm), [nm + 1], [0.0], [1.0], [[(1, 1, 1.0)]])
            @test_throws ArgumentError Daisho.MishObsBackend(:point, (3, nm), [1], [0.0], [1.0], [[(4, 1, 1.0)]])
            @test_throws ArgumentError Daisho.MishObsBackend(:point, (3, nm), [1], [0.0], [1.0], [[(1, 5, 1.0)]])
            @test_throws ArgumentError Daisho.MishObsBackend(:point, (3, nm), [1], [0.0], [0.0], [[(1, 1, 1.0)]])
            @test_throws ArgumentError Daisho.MishObsBackend(:point, (3, nm), [1], [0.0], [1.0], [Tuple{Int,Int,Float64}[]])
            @test_throws DimensionMismatch Daisho.MishObsBackend(:point, (3, nm), [1, 2], [0.0], [1.0, 1.0], [[(1, 1, 1.0)], [(1, 1, 1.0)]])
            @test_throws DimensionMismatch Daisho.ObsSet(grid, _SOLV_VARS, [Daisho.MishObsBackend(:p, (3, nm + 1), [1], [0.0], [1.0], [[(1, 1, 1.0)]])])
            @test Daisho.inverse_variance([2.0, 0.5]) == [0.25, 4.0]
            @test_throws ArgumentError Daisho.inverse_variance([0.0])
        end

        @testset "Hessian" begin
            nmulti = max(Threads.nthreads(), 3)
            for (nch, label) in ((1, "nchunks=1"), (nmulti, "nchunks=$(nmulti)"))
                Cc, = _solv_chain(grid, [2.0, 1.5, 0.7], fill(1.5, 3, 3); nchunks = nch)
                set = Daisho.ObsSet(grid, _SOLV_VARS, Daisho.AbstractObsBackend[bm, bs]; nchunks = nch)
                cf = Daisho.IncrementalCostFunction(Cc, set, zeros(3nc))
                n = 3nm
                Hop = Daisho.HessianOperator(cf)
                @test size(Hop) == (n, n) && eltype(Hop) == Float64
                p = randn(rng, n); q = randn(rng, n)
                Hp = zeros(n); Hq = zeros(n)
                Daisho.hessian_mul!(Hp, cf, p); mul!(Hq, Hop, q)
                @test abs(dot(Hp, q) - dot(p, Hq)) / (norm(Hp) * norm(q)) < 1e-12
                @test dot(Hp, p) >= dot(p, p) * (1 - 1e-12)
                Daisho.hessian_mul!(Hp, cf, p)
                nalloc = @allocated Daisho.hessian_mul!(Hp, cf, p)
                @info "hessian_mul! allocations ($(label), nthreads=$(Threads.nthreads()))" nalloc
                if nch == 1
                    @test nalloc == 0
                else
                    @test nalloc < 2_000_000 * max(1.0, nch / 8)   # task spawns in every pencil sweep (≈ 0.5 MB at 8 chunks)
                end
            end
        end
    end

    # ── 4. Gradient checks ──────────────────────────────────────────────────
    @testset "Mish + node lattices in one ObsSet" begin
        rng = MersenneTwister(11)
        grid = _tiny_grid()
        gm = Daisho._spline_grid_geometry(grid, _SOLV_VARS)
        gn = Daisho._spline_grid_geometry(grid, _SOLV_VARS; lattice = :nodes)
        nm, nn, nc = Daisho._nmish(gm), Daisho._nmish(gn), Daisho._ncoef(gm)
        @test nn == prod(ax.sp.num_cells + 1 for ax in (gm.axes[1, 1], gm.axes[2, 1], gm.axes[3, 1]))
        # U value obs at every mish point; continuity at every node.
        bm = Daisho.MishObsBackend(:value, (3, nm), collect(1:nm), randn(rng, nm), fill(2.0, nm),
                                   [[(1, 1, 1.0)] for _ in 1:nm])
        bn = Daisho.MishObsBackend(:mass_continuity, (3, nn), collect(1:nn), zeros(nn), fill(3.0, nn),
                                   [[(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (3, 1, -1e-4)] for _ in 1:nn];
                                   lattice = :nodes)
        @test bm.lattice == :mish && bn.lattice == :nodes
        @test_throws ArgumentError Daisho.MishObsBackend(:x, (3, nn), [1], [0.0], [1.0], [[(1, 1, 1.0)]]; lattice = :cells)
        @test_throws ArgumentError Daisho.MishObsBackend(:x, (3, nn), [1], [0.0], [1.0], [[(1, 1, 1.0)]];
                                                         lattice = :nodes, model = (s, m) -> (0.0, [1.0]))
        # A node backend sized for the mish is rejected (and vice versa).
        @test_throws DimensionMismatch Daisho.ObsSet(grid, _SOLV_VARS, [Daisho.MishObsBackend(:x, (3, nm), [1], [0.0], [1.0], [[(1, 1, 1.0)]]; lattice = :nodes)])
        for nch in (1, 3)
            set = Daisho.ObsSet(grid, _SOLV_VARS, Daisho.AbstractObsBackend[bm, bn]; nchunks = nch)
            @test Daisho.nobs(set) == nm + nn
            @test set.ev !== nothing && set.ev_nodes !== nothing
            @test length(set.lattices) == 2 && set.nnodes == nn
            @test Daisho.adjoint_relative_error(set; rng = rng) < 1e-12
            # H rows of the node backend equal a direct node-lattice evaluation.
            a = randn(rng, 3nc)
            y = zeros(nm + nn)
            Daisho.apply_H!(y, set, a)
            En = Daisho.SplineEvaluation(grid, _SOLV_VARS, [(1, 2), (2, 3), (3, 4), (3, 1)]; lattice = :nodes)
            yn = zeros(size(En, 1))
            Daisho.apply!(yn, En, a)
            direct = [yn[q] + yn[nn + q] + yn[2nn + q] - 1e-4 * yn[3nn + q] for q in 1:nn]
            @test _srelerr(y[nm + 1:end], direct) < 1e-12
            Em = Daisho.SplineEvaluation(grid, _SOLV_VARS, [(1, 1)])
            ym = zeros(nm)
            Daisho.apply!(ym, Em, a)
            @test _srelerr(y[1:nm], ym) < 1e-12
            # evaluate! (linear) == apply_H!
            y2 = zeros(nm + nn)
            Daisho.evaluate!(y2, set, a)
            @test y2 == y
        end
        # Node-only set: no mish evaluation is built.
        setn = Daisho.ObsSet(grid, _SOLV_VARS, [bn])
        @test setn.ev === nothing && setn.ev_nodes !== nothing
        @test Daisho.adjoint_relative_error(setn; rng = rng) < 1e-12
    end

    @testset "Gradient checks" begin
        rng = MersenneTwister(3)
        grid = _tiny_grid()
        gp = grid.params
        nm = gp.iDim * gp.jDim * gp.kDim
        nc = gp.b_iDim * gp.b_jDim * gp.b_kDim
        C, S, = _solv_chain(grid, [2.0, 1.5, 0.7], fill(1.5, 3, 3))
        n = 3nm

        @testset "Quadratic model" begin
            set = Daisho.ObsSet(grid, _SOLV_VARS, Daisho.AbstractObsBackend[
                _rand_mish_backend(rng, nm, 50), _rand_scattered_backend(rng, grid, 30; ext = (4000.0, 3000.0, 3000.0))])
            cf = Daisho.IncrementalCostFunction(C, set, randn(rng, 3nc))
            ξ = randn(rng, n); g = zeros(n)
            Daisho.gradient!(g, cf, ξ)
            # At ξ = 0 the gradient is −rhs.
            g0 = zeros(n); Daisho.gradient!(g0, cf, zeros(n))
            @test _srelerr(g0, -cf.rhs) < 1e-12
            for _ in 1:3
                v = randn(rng, n)
                ε = 1e-3
                fd = (Daisho.cost(cf, ξ .+ ε .* v) - Daisho.cost(cf, ξ .- ε .* v)) / (2ε)
                @test abs(fd - dot(g, v)) / abs(dot(g, v)) < 1e-6
            end
            # Taylor remainder is exactly ½ε²⟨Hv, v⟩ (quadratic).
            v = randn(rng, n)
            Hv = zeros(n); Daisho.hessian_mul!(Hv, cf, v)
            J0 = Daisho.cost(cf, ξ); gv = dot(g, v); vHv = dot(Hv, v)
            for ε in (1e-1, 1e-2, 1e-3, 1e-4)
                rem = Daisho.cost(cf, ξ .+ ε .* v) - J0 - ε * gv
                @test abs(rem / (0.5 * ε^2 * vHv) - 1) < 1e-4
            end
        end

        @testset "Nonlinear wind-speed obs" begin
            # Background: a smooth nonzero wind so speeds stay away from 0.
            X, Y, Z, _ = _mish_xyz(grid)
            xg = vcat(8 .+ sin.(X ./ 1000), -4 .+ cos.(Y ./ 900), zeros(nm))
            a_g = zeros(3nc); Daisho.apply!(a_g, S, xg)
            spd = Daisho.MishObsBackend(:wind_speed, (3, nm), collect(1:nm), 9 .+ randn(rng, nm),
                                        fill(4.0, nm), [[(1, 1, 1.0), (2, 1, 0.0)] for _ in 1:nm];
                                        model = _speed_model)
            @test !Daisho.is_linear(spd)
            pt = _rand_scattered_backend(rng, grid, 10; ext = (4000.0, 3000.0, 3000.0))
            set = Daisho.ObsSet(grid, _SOLV_VARS, Daisho.AbstractObsBackend[spd, pt])
            @test !Daisho.is_linear(set)
            cf = Daisho.IncrementalCostFunction(C, set, a_g)
            cf.ξ_total .= 0.3 .* randn(rng, n)
            Daisho.relinearize!(cf)
            ∇ = -cf.rhs
            J0 = Daisho.cost_nonlinear(cf)
            for _ in 1:3
                v = randn(rng, n)
                ε = 1e-4
                fd = (Daisho.cost_nonlinear(cf, cf.ξ_total .+ ε .* v) -
                      Daisho.cost_nonlinear(cf, cf.ξ_total .- ε .* v)) / (2ε)
                @test abs(fd - dot(∇, v)) / abs(dot(∇, v)) < 1e-5
                # Forward difference converges linearly.
                e1 = abs(Daisho.cost_nonlinear(cf, cf.ξ_total .+ 1e-3 .* v) - J0 - 1e-3 * dot(∇, v))
                e2 = abs(Daisho.cost_nonlinear(cf, cf.ξ_total .+ 1e-4 .* v) - J0 - 1e-4 * dot(∇, v))
                @test e2 < 0.02 * e1
            end
            # The tangent weights at a_k are (u/s, v/s) and h(a_k) is the speed.
            vals = _mish_values(grid, cf.a_k)
            s = sqrt.(vals[1] .^ 2 .+ vals[2] .^ 2)
            m = 7; o = spd.perm[m]
            @test spd.term_w[spd.term_ptr[m]] ≈ vals[1][o] / s[o] rtol = 1e-12
            @test spd.term_w[spd.term_ptr[m] + 1] ≈ vals[2][o] / s[o] rtol = 1e-12
            @test _srelerr(cf.d[1:nm], spd.y .- s[spd.point]) < 1e-12
        end
    end

    # ── 5. Zero obs / trivial ──────────────────────────────────────────────
    @testset "Zero observations and vanishing weight" begin
        rng = MersenneTwister(3)
        grid = _tiny_grid()
        gp = grid.params
        nm = gp.iDim * gp.jDim * gp.kDim
        nc = gp.b_iDim * gp.b_jDim * gp.b_kDim
        C, = _solv_chain(grid, 1.0, fill(1.0, 3, 3))
        empty_m = Daisho.MishObsBackend(:point, (3, nm), Int[], Float64[], Float64[],
                                        Vector{Tuple{Int,Int,Float64}}[])
        for backends in (Daisho.AbstractObsBackend[], Daisho.AbstractObsBackend[empty_m])
            set = Daisho.ObsSet(grid, _SOLV_VARS, backends)
            @test Daisho.nobs(set) == 0
            @test set.ev === nothing
            cf = Daisho.IncrementalCostFunction(C, set, randn(rng, 3nc))
            st = Daisho.solve_incremental!(cf, Daisho.VarSolverOptions())
            @test all(iszero, Daisho.control_increment(cf))
            @test st.converged
            @test st.inner_iterations == [0]
            @test st.cost_history == [0.0, 0.0]
            a = zeros(3nc); Daisho.analysis_coefficients!(a, cf)
            @test a == cf.a_g
        end
        norms = Float64[]
        for r in (1e-4, 1e-8, 1e-12)
            b = Daisho.MishObsBackend(:point, (3, nm), [nm ÷ 2], [5.0], [r], [[(1, 1, 1.0)]])
            cf = Daisho.IncrementalCostFunction(C, Daisho.ObsSet(grid, _SOLV_VARS, [b]), zeros(3nc))
            Daisho.solve_incremental!(cf, Daisho.VarSolverOptions(inner_tolerance = 1e-10))
            push!(norms, norm(Daisho.control_increment(cf)))
        end
        @info "‖ξ‖ for a single obs with rinv = 1e-4, 1e-8, 1e-12" norms
        @test norms[1] > 0
        @test norms[2] < 1.001e-4 * norms[1]
        @test norms[3] < 1.001e-8 * norms[1]
    end

    # ── 6, 9, 10. Beltrami-family recovery (larger grid) ───────────────────
    @testset "Beltrami-family recovery" begin
        rng = MersenneTwister(3)
        grid = _large_grid()
        gp = grid.params
        nm = gp.iDim * gp.jDim * gp.kDim
        nc = gp.b_iDim * gp.b_jDim * gp.b_kDim
        X, Y, Z, geom = _mish_xyz(grid)
        Lx, Ly, Lz = 10_000.0, 10_000.0, 6000.0
        k = 2π / Lx; l = 2π / Ly; m = π / (2Lz)      # w ∝ sin(mz): 0 at z = 0, max at the top
        A = 10.0; B = -6.0; Cc = -(A * k + B * l) / m  # A k + B l + Cc m = 0
        uf(x, y, z) = A * sin(k * x) * cos(l * y) * cos(m * z)
        vf(x, y, z) = B * cos(k * x) * sin(l * y) * cos(m * z)
        wf(x, y, z) = Cc * cos(k * x) * cos(l * y) * sin(m * z)
        # Analytic divergence vanishes.
        @test abs(A * k + B * l + Cc * m) < 1e-15
        truth = [uf.(X, Y, Z), vf.(X, Y, Z), wf.(X, Y, Z)]
        amp = A
        dx, dy, dz = Lx / 10, Ly / 10, Lz / 6
        interior = (X .> dx) .& (X .< Lx - dx) .& (Y .> dy) .& (Y .< Ly - dy) .& (Z .> dz) .& (Z .< Lz - dz)
        relrms(v, a) = _rms((a[v] .- truth[v])[interior]) / _rms(truth[v][interior])

        C1, = _solv_chain(grid, amp, fill(1.0, 3, 3))
        # (i) scattered value obs of u, v, w at random points.
        npt = 2000
        P = hcat(Lx .* rand(rng, npt), Ly .* rand(rng, npt), Lz .* rand(rng, npt))
        σo = 0.01 * amp
        yobs = Float64[]; terms = Vector{Tuple{Int,Int,Float64}}[]; xyz = zeros(0, 3)
        for (v, f) in enumerate((uf, vf, wf))
            append!(yobs, f.(P[:, 1], P[:, 2], P[:, 3]))
            append!(terms, [[(v, 1, 1.0)] for _ in 1:npt])
            xyz = vcat(xyz, P)
        end
        bpt = Daisho.ScatteredObsBackend(:point, grid, _SOLV_VARS, xyz, yobs, fill(1 / σo^2, 3npt), terms)
        set_i = Daisho.ObsSet(grid, _SOLV_VARS, [bpt])
        opts_i = Daisho.VarSolverOptions(max_inner = 400, inner_tolerance = 1e-6)
        cf_i = Daisho.IncrementalCostFunction(C1, set_i, zeros(3nc))
        t = @elapsed st_i = Daisho.solve_incremental!(cf_i, opts_i)
        a_i = zeros(3nc); Daisho.analysis_coefficients!(a_i, cf_i)
        ana_i = _mish_values(grid, a_i)
        err_i = [relrms(v, ana_i) for v in 1:3]
        @info "Beltrami (i) point obs" err_i = repr(err_i) cg = repr(st_i.inner_iterations) solved = st_i.inner_converged seconds = t
        @test st_i.converged
        @test all(err_i .< 1e-2)

        # (ii)/(iii) radial velocities at every mish point + mass continuity.
        radars = [(-5000.0, -5000.0, 0.0), (15_000.0, -4000.0, 0.0), (-4000.0, 15_000.0, 0.0)]
        σvr = 0.01 * amp
        # σ_div: the analytic divergence is 0; the discrete spline divergence
        # of the fitted truth is O(1e-6) s⁻¹ while individual derivative terms
        # are O(A k) ≈ 6e-3 s⁻¹, so 1e-4 s⁻¹ (≈ 1.6 % of the derivative scale,
        # comparable to σ_vr/amp = 1 %) enforces continuity without fighting
        # the spline truncation error.
        σdiv = 1e-4
        cont = Daisho.MishObsBackend(:mass_continuity, (3, nm), collect(1:nm), zeros(nm),
                                     fill(1 / σdiv^2, nm), [[(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)] for _ in 1:nm])
        function vr_backend(r)
            gs = [begin
                      d = (X[p] - r[1], Y[p] - r[2], Z[p] - r[3]); nd = sqrt(sum(abs2, d))
                      (d[1] / nd, d[2] / nd, d[3] / nd)
                  end for p in 1:nm]
            y = [g[1] * truth[1][p] + g[2] * truth[2][p] + g[3] * truth[3][p] for (p, g) in enumerate(gs)]
            return Daisho.MishObsBackend(:radial_velocity, (3, nm), collect(1:nm), y, fill(1 / σvr^2, nm),
                                         [[(1, 1, g[1]), (2, 1, g[2]), (3, 1, g[3])] for g in gs])
        end
        vrs = [vr_backend(r) for r in radars]
        opts_r = Daisho.VarSolverOptions(max_inner = 1000, inner_tolerance = 1e-6)
        errs = Dict{Int,Vector{Float64}}()
        for nrad in (3, 2)
            set = Daisho.ObsSet(grid, _SOLV_VARS, Daisho.AbstractObsBackend[vrs[1:nrad]..., cont])
            cf = Daisho.IncrementalCostFunction(C1, set, zeros(3nc))
            t = @elapsed st = Daisho.solve_incremental!(cf, opts_r)
            a = zeros(3nc); Daisho.analysis_coefficients!(a, cf)
            ana = _mish_values(grid, a)
            errs[nrad] = [relrms(v, ana) for v in 1:3]
            @info "Beltrami ($(nrad) radars + continuity)" err = repr(errs[nrad]) cg = repr(st.inner_iterations) solved = st.inner_converged Jo = repr(st.cost_by_backend) seconds = t
            @test st.converged
            @test length(st.cost_by_backend) == nrad + 1
        end
        @test all(errs[3] .< 1e-2)
        @test errs[2][1] < 1e-2 && errs[2][2] < 1e-2
        @test errs[2][3] < 5e-2

        # 9. ξ_total consistency for a linear problem.
        @testset "ξ_total consistency" begin
            cf2 = Daisho.IncrementalCostFunction(C1, set_i, zeros(3nc))
            st2 = Daisho.solve_incremental!(cf2, Daisho.VarSolverOptions(max_outer = 2, max_inner = 2000,
                                                                          inner_tolerance = 1e-10, outer_tolerance = 1e-14))
            cf1 = Daisho.IncrementalCostFunction(C1, set_i, zeros(3nc))
            st1 = Daisho.solve_incremental!(cf1, Daisho.VarSolverOptions(max_outer = 1, max_inner = 2000,
                                                                          inner_tolerance = 1e-10))
            @info "ξ_total consistency" increments = st2.increment_norms cg = st2.inner_iterations
            @test st2.outer_iterations == 2
            @test st2.increment_norms[2] < 1e-8 * st2.increment_norms[1]
            a1 = zeros(3nc); a2 = zeros(3nc)
            Daisho.analysis_coefficients!(a1, cf1); Daisho.analysis_coefficients!(a2, cf2)
            @test _srelerr(a2, a1) < 1e-8
            @test st2.cost_history[3] <= st2.cost_history[2] * (1 + 1e-12)
        end

        # 10. Determinism across chunk counts.
        @testset "Determinism (nchunks)" begin
            vals = Vector{Float64}[]
            for nch in (1, Threads.nthreads())
                Cn, = _solv_chain(grid, amp, fill(1.0, 3, 3); nchunks = nch)
                set = Daisho.ObsSet(grid, _SOLV_VARS, Daisho.AbstractObsBackend[vrs[1], vrs[2], cont, bpt]; nchunks = nch)
                cf = Daisho.IncrementalCostFunction(Cn, set, zeros(3nc))
                Daisho.solve_incremental!(cf, Daisho.VarSolverOptions(max_inner = 50, inner_tolerance = 1e-8))
                push!(vals, copy(Daisho.control_increment(cf)))
            end
            @test vals[1] == vals[2]
        end
    end

    # ── 7. BC exactness ────────────────────────────────────────────────────
    @testset "BC exactness with a wide filter" begin
        grid = _large_grid()
        gp = grid.params
        nm = gp.iDim * gp.jDim * gp.kDim
        nc = gp.b_iDim * gp.b_jDim * gp.b_kDim
        L = [1.0 1.0 3.0; 1.0 1.0 3.0; 1.0 1.0 3.0]
        C, S, D, F = _solv_chain(grid, [1.0, 1.0, 5.0], L)
        wobs = Daisho.MishObsBackend(:point, (3, nm), collect(1:nm), fill(5.0, nm), fill(1e4, nm),
                                     [[(3, 1, 1.0)] for _ in 1:nm])
        cf = Daisho.IncrementalCostFunction(C, Daisho.ObsSet(grid, _SOLV_VARS, [wobs]), zeros(3nc))
        st = Daisho.solve_incremental!(cf, Daisho.VarSolverOptions(max_inner = 300, inner_tolerance = 1e-8))
        a = zeros(3nc); Daisho.analysis_coefficients!(a, cf)
        w = _mish_values(grid, a)[3]
        wmax = maximum(abs, w)
        @test wmax > 2.0                      # the obs pulled W strongly
        ax = Daisho._spline_grid_geometry(grid, _SOLV_VARS).axes[1, 3]
        Aw = reshape(a[2nc+1:3nc], gp.b_kDim, gp.b_jDim, gp.b_iDim)
        wbot = [CubicBSpline.SItransform(ax.sp, Aw[:, mj, mi], ax.sp.xmin, 0)
                for mi in 1:3:gp.b_iDim, mj in 1:2:gp.b_jDim]
        @info "BC exactness" max_w_bottom = maximum(abs, wbot) wmax cg = st.inner_iterations
        @test maximum(abs, wbot) < 1e-12 * wmax
        # The mish-space increment D F ξ is far from zero at the lowest mish
        # level: the filter did not impose the BC, S did.
        t1 = zeros(3nm); t2 = zeros(3nm)
        Daisho.apply!(t1, F, Daisho.control_increment(cf)); Daisho.apply!(t2, D, t1)
        Winc = reshape(t2[2nm+1:3nm], gp.kDim, gp.jDim, gp.iDim)
        @test maximum(abs, Winc[1, :, :]) > 1.0
    end

    # ── 8. Nonlinear outer loop ────────────────────────────────────────────
    @testset "Nonlinear outer loop (wind speed)" begin
        rng = MersenneTwister(3)
        # A 6×6×3-cell grid over the same 10×10×6 km box: the final error is
        # set by how densely the point obs fix the wind DIRECTION (the speed
        # obs leave the tangential component to B), so the direction obs must
        # be dense relative to the B correlation length. Measured (L = 1 cell,
        # σ_b = 2): 10×10×6 cells with 50 points → rel err (1.1e-3, 3.9e-3);
        # 6×6×3 with 50 → (4.7e-4, 1.9e-3), 100 → (2.3e-4, 9.5e-4),
        # 150 → (1.6e-4, 6.8e-4).
        cells = (6, 6, 3)
        grid = _solv_grid(cells = cells, extent = (10_000.0, 10_000.0, 6000.0))
        gp = grid.params
        nm = gp.iDim * gp.jDim * gp.kDim
        nc = gp.b_iDim * gp.b_jDim * gp.b_kDim
        X, Y, Z, _ = _mish_xyz(grid)
        Lx, Ly, Lz = 10_000.0, 10_000.0, 6000.0
        C, S, = _solv_chain(grid, 2.0, fill(1.0, 3, 3))
        # Truth is a spline field (so the speed obs are exactly representable).
        xt = vcat(10 .+ 3 .* sin.(2π .* X ./ Lx) .* cos.(2π .* Y ./ Ly),
                  5 .+ 3 .* cos.(2π .* X ./ Lx) .* sin.(2π .* Y ./ Ly) .* cos.(π .* Z ./ Lz),
                  zeros(nm))
        a_t = zeros(3nc); Daisho.apply!(a_t, S, xt)
        tv = _mish_values(grid, a_t)
        spd_true = sqrt.(tv[1] .^ 2 .+ tv[2] .^ 2)
        σs = 0.01
        spd = _speed_backend(nm, spd_true, fill(1 / σs^2, nm))
        # Direction-fixing point obs of u and v at 150 random points.
        npt = 150
        P = hcat(Lx .* rand(rng, npt), Ly .* rand(rng, npt), Lz .* rand(rng, npt))
        xyz = vcat(P, P)
        pterms = vcat([[(1, 1, 1.0)] for _ in 1:npt], [[(2, 1, 1.0)] for _ in 1:npt])
        tmp = Daisho.ScatteredObsBackend(:point, grid, _SOLV_VARS, xyz, zeros(2npt), ones(2npt), pterms)
        pts = Daisho.ScatteredObsBackend(:point, grid, _SOLV_VARS, xyz, tmp.H * a_t,
                                         fill(1 / σs^2, 2npt), pterms)
        set = Daisho.ObsSet(grid, _SOLV_VARS, Daisho.AbstractObsBackend[spd, pts])
        # Background: truth + a smooth 20 % perturbation.
        xg = xt .+ vcat(2 .* cos.(π .* X ./ Lx) .* cos.(π .* Z ./ (2Lz)),
                        -1 .+ 1 .* sin.(π .* Y ./ Ly), zeros(nm))
        a_g = zeros(3nc); Daisho.apply!(a_g, S, xg)
        cf = Daisho.IncrementalCostFunction(C, set, a_g)
        t = @elapsed st = Daisho.solve_incremental!(cf, Daisho.VarSolverOptions(max_outer = 8, max_inner = 1000,
                                                                                 inner_tolerance = 1e-8,
                                                                                 outer_tolerance = 1e-6))
        a = zeros(3nc); Daisho.analysis_coefficients!(a, cf)
        av = _mish_values(grid, a)
        dx, dy, dz = Lx / cells[1], Ly / cells[2], Lz / cells[3]
        interior = (X .> dx) .& (X .< Lx - dx) .& (Y .> dy) .& (Y .< Ly - dy) .& (Z .> dz) .& (Z .< Lz - dz)
        err = [_rms((av[v] .- tv[v])[interior]) / _rms(tv[v][interior]) for v in 1:2]
        bg = _mish_values(grid, a_g)
        err_bg = [_rms((bg[v] .- tv[v])[interior]) / _rms(tv[v][interior]) for v in 1:2]
        @info "Nonlinear outer loop" outer = st.outer_iterations cg = repr(st.inner_iterations) J = repr(st.cost_history) increments = repr(st.increment_norms) err = repr(err) err_bg = repr(err_bg) seconds = t
        @test st.converged
        @test st.outer_iterations <= 6
        @test all(diff(st.cost_history[2:end]) .<= 1e-10 .* st.cost_history[2:end-1])
        @test st.cost_history[2] < st.cost_history[1]
        @test all(err .< 1e-3)
    end

    # ── CG progress reporting ──────────────────────────────────────────────
    @testset "CG progress reporting" begin
        rng = MersenneTwister(11)
        grid = _tiny_grid()
        gp = grid.params
        nm = gp.iDim * gp.jDim * gp.kDim
        nc = gp.b_iDim * gp.b_jDim * gp.b_kDim
        n = 3nm
        C, = _solv_chain(grid, [2.0, 1.5, 0.7], fill(1.5, 3, 3))
        # Many precise observations so CG needs well over 10 iterations.
        pts = rand(rng, 1:nm, 400)
        terms = [[(rand(rng, 1:3), rand(rng, 1:4), randn(rng)) for _ in 1:rand(rng, 1:4)] for _ in 1:400]
        bm = Daisho.MishObsBackend(:point, (3, nm), pts, randn(rng, 400), 50.0 .+ 50.0 .* rand(rng, 400), terms)
        backends = Daisho.AbstractObsBackend[bm,
            _rand_scattered_backend(rng, grid, 30; ext = (4000.0, 3000.0, 3000.0))]
        set = Daisho.ObsSet(grid, _SOLV_VARS, backends)
        a_g = randn(rng, 3nc)

        @testset "Model-cost identity" begin
            cf = Daisho.IncrementalCostFunction(C, set, a_g)
            J_k = Daisho.cost_nonlinear(cf)
            x = randn(rng, n)
            r = copy(cf.rhs); Hx = zeros(n)
            mul!(Hx, Daisho.HessianOperator(cf), x)
            r .-= Hx
            Jm = Daisho.cg_model_cost(J_k, cf.rhs, x, r)
            Jt = Daisho.cost_nonlinear(cf, cf.ξ_total .+ x)
            @test abs(Jm - Jt) / abs(Jt) < 1e-10
        end

        @testset "Printed lines" begin
            opts(ri, io) = Daisho.VarSolverOptions(max_inner = 400, inner_tolerance = 1e-10,
                                                   report_interval = ri, io = io)
            cf_on = Daisho.IncrementalCostFunction(C, set, a_g)
            buf = IOBuffer()
            st_on = Daisho.solve_incremental!(cf_on, opts(5, buf))
            out = String(take!(buf))
            @test st_on.inner_iterations[1] >= 10
            lines = split(out, '\n'; keepempty = false)
            @test startswith(lines[1], "solve_incremental!: n=$(n) controls, nobs=430 (point 400, point 30)")
            @test any(l -> startswith(l, "  initial  J="), lines)
            @test any(l -> startswith(l, "outer 1  ‖∇J‖="), lines)
            cglines = filter(l -> startswith(l, "  cg "), lines)
            @test length(cglines) >= 2
            @test occursin(r"^  cg +1  ", cglines[1])
            @test any(l -> startswith(l, "  outer 1 done: cg="), lines)
            res = filter(l -> startswith(l, "  result   J="), lines)
            @test length(res) == st_on.outer_iterations
            Jprint = parse(Float64, match(r"result   J=(\S+)", res[end]).captures[1])
            @test abs(Jprint - st_on.cost_history[end]) / abs(st_on.cost_history[end]) < 1e-8
            # For linear obs the last CG line's model cost approaches the final cost.
            Jcg = parse(Float64, match(r"  J=(\S+)", cglines[end]).captures[1])
            @test Jcg >= st_on.cost_history[end] * (1 - 1e-8)

            cf_off = Daisho.IncrementalCostFunction(C, set, a_g)
            buf0 = IOBuffer()
            st_off = Daisho.solve_incremental!(cf_off, opts(0, buf0))
            @test isempty(take!(buf0))
            @test Daisho.control_increment(cf_on) == Daisho.control_increment(cf_off)
            @test st_on.cost_history == st_off.cost_history
            @test st_on.inner_iterations == st_off.inner_iterations
        end

        @test_throws ArgumentError Daisho.solve_incremental!(
            Daisho.IncrementalCostFunction(C, set, a_g), Daisho.VarSolverOptions(report_interval = -1))
    end

    @info "test_variational_solver.jl runtime" seconds = time() - t_start threads = Threads.nthreads()
end
