# Tests for the variational operator layer (Phase 2): spline projection and
# evaluation operators with exact adjoints, pencil sweeps, and parity against
# Springsteel's own transforms.

using Random
using LinearAlgebra
using Springsteel
using Springsteel: CubicBSpline

const _VAR_NAMES = ["U", "V", "W"]

# Small RRR state grid: natural BCs everywhere except a Dirichlet (R1T0)
# bottom on W.
function _var_state_grid(; mubar::Int = 3, quadrature::Symbol = :gauss,
                         cells = (4, 3, 5), w_dirichlet::Bool = true)
    kmin = Dict{String,Springsteel.BoundaryConditions}("default" => NaturalBC())
    w_dirichlet && (kmin["W"] = DirichletBC(0.0))
    cfg = Daisho.SpringsteelGridConfig(geometry = "RRR", mubar = mubar,
        quadrature = quadrature,
        i = Daisho.SpringsteelAxisConfig(min = -2000.0, max = 2000.0, cells = cells[1]),
        j = Daisho.SpringsteelAxisConfig(min = 0.0, max = 3000.0, cells = cells[2]),
        k = Daisho.SpringsteelAxisConfig(min = 0.0, max = 5000.0, cells = cells[3],
                                         bc_min = kmin))
    return Daisho.create_radar_grid(cfg, Dict("U" => 1, "V" => 2, "W" => 3))
end

_relerr(a, b) = norm(a .- b) / max(norm(b), eps())

# Evaluate `f(args...)` once for warm-up, then measure its allocations.
function _alloc_after_warmup(f, args...)
    f(args...)
    return @allocated f(args...)
end

# Springsteel physical slot for each Daisho derivative slot.
const _SS_SLOT = (1, 2, 4, 6)

@testset "Variational operators (Phase 2)" begin

    for (mubar, quadrature) in ((3, :gauss), (2, :regular))
    @testset "mubar=$(mubar), quadrature=$(quadrature)" begin
        rng = MersenneTwister(1)
        grid = _var_state_grid(mubar = mubar, quadrature = quadrature)
        gp = grid.params
        @test isempty(gp.spline_filter)
        vidx = [gp.vars[n] for n in _VAR_NAMES]
        templates = Dict(
            :i => [grid.ibasis.data[1, 1, v] for v in vidx],
            :j => [grid.jbasis.data[1, 1, v] for v in vidx],
            :k => [grid.kbasis.data[1, 1, v] for v in vidx])

        # ── 1-D primitives: parity with Springsteel and exact adjoints ───────
        @testset "1-D primitives axis=$(axname)" for axname in (:i, :j, :k)
            for spline in templates[axname]
                ax = Daisho.AxisSpline(spline)
                sp = spline.params
                Dim, bDim = ax.Dim, ax.bDim
                @test Dim == sp.mishDim && bDim == sp.bDim
                @test ax.Minterior == size(spline.gammaBC, 1)

                # Basis tables vs SItransform_matrix rows.
                for (Φ, d) in ((ax.Φ0, 0), (ax.Φ1, 1))
                    M = CubicBSpline.SItransform_matrix(spline, Vector{Float64}(spline.mishPoints), d)
                    Mt = zeros(Dim, bDim)
                    for i in 1:Dim, s in 1:4
                        Mt[i, ax.first_coef[i] + s - 1] = Φ[s, i]
                    end
                    @test Mt == M
                end

                u = randn(rng, Dim)
                a = randn(rng, bDim)
                b = randn(rng, bDim)
                scratch = zeros(Dim)

                # sb! vs _sb_matrix
                bb = zeros(bDim)
                Daisho.sb!(bb, ax, u, scratch)
                ref = spline._sb_matrix * u
                @test norm(bb - ref) <= 1e-14 * norm(ref)

                # sa! vs SAtransform!
                Min = zeros(ax.Minterior); Mout = zeros(ax.Minterior)
                aa = zeros(bDim)
                Daisho.sa!(aa, ax, b, Min, Mout)
                copyto!(spline.b, b)
                CubicBSpline.SAtransform!(spline)
                @test norm(aa - spline.a) <= 1e-13 * norm(spline.a)

                # eval! vs SItransform for value and first derivative
                for (Φ, d) in ((ax.Φ0, 0), (ax.Φ1, 1))
                    uu = zeros(Dim)
                    Daisho.eval!(uu, ax, a, Φ)
                    ref = CubicBSpline.SItransform(sp, a, spline.mishPoints, zeros(Dim), d)
                    @test norm(uu - ref) <= 1e-14 * norm(ref)
                end

                # Dot-product tests
                for Φ in (ax.Φ0, ax.Φ1)
                    Ea = zeros(Dim); Daisho.eval!(Ea, ax, a, Φ)
                    Etu = zeros(bDim); Daisho.scatter!(Etu, ax, u, Φ)
                    @test abs(dot(Ea, u) - dot(a, Etu)) <= 1e-12 * norm(Ea) * norm(u)
                end
                Sbu = zeros(bDim); Daisho.sb!(Sbu, ax, u, scratch)
                Sbtb = zeros(Dim); Daisho.sbt!(Sbtb, ax, b)
                @test abs(dot(Sbu, b) - dot(u, Sbtb)) <= 1e-12 * norm(Sbu) * norm(b)
                Saa = zeros(bDim); Daisho.sa!(Saa, ax, a, Min, Mout)
                Sab = zeros(bDim); Daisho.sa!(Sab, ax, b, Min, Mout)
                @test abs(dot(Saa, b) - dot(a, Sab)) <= 1e-12 * norm(Saa) * norm(b)
            end
        end

        # ── Operators ────────────────────────────────────────────────────────
        S = Daisho.SplineProjection(grid, _VAR_NAMES)
        nm = gp.iDim * gp.jDim * gp.kDim
        nc = gp.b_iDim * gp.b_jDim * gp.b_kDim
        @test size(S) == (3nc, 3nm)
        all_slots = [(v, s) for v in 1:3 for s in 1:4]
        E = Daisho.SplineEvaluation(grid, _VAR_NAMES, all_slots)
        @test size(E) == (12nm, 3nc)

        @testset "Adjointness" begin
            @test Daisho.adjoint_relative_error(S; rng = rng) < 1e-12
            @test Daisho.adjoint_relative_error(E; rng = rng) < 1e-12
            for v in 1:3, s in 1:4
                Es = Daisho.SplineEvaluation(grid, _VAR_NAMES, [(v, s)])
                @test Daisho.adjoint_relative_error(Es; rng = rng) < 1e-12
            end
            # Partial, out-of-order slot selections.
            Ep = Daisho.SplineEvaluation(grid, _VAR_NAMES, [(3, 4), (1, 2), (3, 1), (2, 3)])
            @test size(Ep) == (4nm, 3nc)
            @test Daisho.adjoint_relative_error(Ep; rng = rng) < 1e-12

            D1 = Daisho.DiagonalScaling([2.0, 0.5, 3.0], nm)
            @test size(D1) == (3nm, 3nm)
            @test Daisho.adjoint_relative_error(D1; rng = rng) < 1e-12
            D2 = Daisho.DiagonalScaling(Union{Float64,Vector{Float64}}[1.5, rand(rng, nm) .+ 0.1, 2.0], nm)
            @test Daisho.adjoint_relative_error(D2; rng = rng) < 1e-12
            C = Daisho.ChainOp(S, D2)
            @test size(C) == (3nc, 3nm)
            @test Daisho.adjoint_relative_error(C; rng = rng) < 1e-12
            C3 = Daisho.ChainOp(E, S, D2)
            @test size(C3) == (12nm, 3nm)
            @test Daisho.adjoint_relative_error(C3; rng = rng) < 1e-12
        end

        @testset "DiagonalScaling / ChainOp semantics" begin
            x = randn(rng, 3nm)
            D2 = Daisho.DiagonalScaling(Union{Float64,Vector{Float64}}[1.5, fill(0.25, nm), 2.0], nm)
            y = similar(x); Daisho.apply!(y, D2, x)
            @test y[1:nm] == 1.5 .* x[1:nm]
            @test y[nm+1:2nm] == 0.25 .* x[nm+1:2nm]
            @test y[2nm+1:end] == 2.0 .* x[2nm+1:end]
            @test_throws ArgumentError Daisho.DiagonalScaling([1.0, 0.0], nm)
            @test_throws ArgumentError Daisho.DiagonalScaling(Union{Float64,Vector{Float64}}[1.0, fill(-1.0, nm)], nm)
            @test_throws ArgumentError Daisho.DiagonalScaling(Union{Float64,Vector{Float64}}[1.0, ones(nm - 1)], nm)
            # ChainOp(S, D) applies D first, then S.
            C = Daisho.ChainOp(S, D2)
            a1 = zeros(3nc); Daisho.apply!(a1, C, x)
            a2 = zeros(3nc); Daisho.apply!(a2, S, y)
            @test a1 == a2
            @test_throws DimensionMismatch Daisho.ChainOp(D2, S)
        end

        # ── 3-D parity with Springsteel spectralTransform!/gridTransform! ────
        x = randn(rng, 3nm)
        a = zeros(3nc)
        yE = zeros(12nm)
        @testset "3-D parity vs Springsteel" begin
            for (n, name) in enumerate(_VAR_NAMES)
                grid.physical[:, gp.vars[name], 1] .= x[(n-1)*nm+1:n*nm]
            end
            Springsteel.spectralTransform!(grid)
            Springsteel.gridTransform!(grid)
            Daisho.apply!(a, S, x)
            Daisho.apply!(yE, E, a)
            for n in 1:3, s in 1:4
                k = (n - 1) * 4 + s
                mine = vec(Daisho.slab_view(yE, E, k))
                ref = grid.physical[:, gp.vars[_VAR_NAMES[n]], _SS_SLOT[s]]
                @test _relerr(mine, ref) < 1e-12
            end
            @test size(Daisho.slab_view(yE, E, 1)) == (gp.kDim, gp.jDim, gp.iDim)
        end

        @testset "Export round trip to Springsteel spectral" begin
            ar = randn(rng, 3nc)
            # Coefficients must satisfy the BCs: project a random field.
            Daisho.apply!(ar, S, randn(rng, 3nm))
            fill!(grid.physical, 0.0)
            Daisho.coefficients_to_springsteel_spectral!(grid, ar, _VAR_NAMES)
            Springsteel.gridTransform!(grid)
            Daisho.apply!(yE, E, ar)
            for n in 1:3, s in 1:4
                k = (n - 1) * 4 + s
                mine = vec(Daisho.slab_view(yE, E, k))
                ref = grid.physical[:, gp.vars[_VAR_NAMES[n]], _SS_SLOT[s]]
                @test _relerr(ref, mine) < 1e-12
            end
        end

        @testset "Node lattice evaluation" begin
            ic = templates[:i][1].params.num_cells
            jc = templates[:j][1].params.num_cells
            kc = templates[:k][1].params.num_cells
            En = Daisho.SplineEvaluation(grid, _VAR_NAMES, all_slots; lattice = :nodes)
            nn = (kc + 1) * (jc + 1) * (ic + 1)
            @test En.lattice == :nodes && E.lattice == :mish
            @test Daisho.lattice_dims(En) == (kc + 1, jc + 1, ic + 1)
            @test Daisho.lattice_dims(E) == (gp.kDim, gp.jDim, gp.iDim)
            @test size(En) == (12nn, 3nc)
            @test Daisho.adjoint_relative_error(En; rng = rng) < 1e-12
            xn = Daisho.node_coordinates(templates[:i][1])
            yn = Daisho.node_coordinates(templates[:j][1])
            zn = Daisho.node_coordinates(templates[:k][1])
            @test xn ≈ collect(LinRange(-2000.0, 2000.0, ic + 1))
            @test zn[1] == 0.0 && zn[end] == 5000.0
            # The node axis tables: every node incl. x = xmax has its four
            # coefficients inside 1:bDim, and the tables match the basis.
            for (axname, xs) in ((:i, xn), (:j, yn), (:k, zn)), spline in templates[axname]
                ax = Daisho.AxisSpline(spline; points = xs)
                @test ax.Dim == length(xs) && all(1 .<= ax.first_coef .<= ax.bDim - 3)
                M = CubicBSpline.SItransform_matrix(spline, xs, 0)
                Mt = zeros(ax.Dim, ax.bDim)
                for i in 1:ax.Dim, s in 1:4
                    Mt[i, ax.first_coef[i] + s - 1] = ax.Φ0[s, i]
                end
                @test Mt ≈ M atol = 1e-14
                @test_throws DomainError Daisho.AxisSpline(spline; points = [xs[end] + 1.0])
            end
            # Parity with Springsteel's regularGridTransform on the node lattice.
            ar = zeros(3nc)
            Daisho.apply!(ar, S, randn(rng, 3nm))
            Daisho.coefficients_to_springsteel_spectral!(grid, ar, _VAR_NAMES)
            reg = Springsteel.regularGridTransform(grid, xn, yn, zn)
            yN = zeros(size(En, 1))
            Daisho.apply!(yN, En, ar)
            for n in 1:3, s in 1:4
                k = (n - 1) * 4 + s
                mine = vec(Daisho.slab_view(yN, En, k))
                ref = reg[:, gp.vars[_VAR_NAMES[n]], _SS_SLOT[s]]
                @test _relerr(mine, ref) < 1e-12
            end
            @test size(Daisho.slab_view(yN, En, 1)) == (kc + 1, jc + 1, ic + 1)
            # Partition of unity at every node (boundary nodes included):
            # unit coefficients give value 1 and zero derivatives.
            y1 = zeros(size(En, 1))
            Daisho.apply!(y1, En, ones(3nc))
            for n in 1:3
                @test maximum(abs.(Daisho.slab_view(y1, En, (n - 1) * 4 + 1) .- 1.0)) < 1e-12
                for s in 2:4
                    @test maximum(abs.(Daisho.slab_view(y1, En, (n - 1) * 4 + s))) < 1e-12
                end
            end
            # Threading determinism on the node lattice.
            En1 = Daisho.SplineEvaluation(grid, _VAR_NAMES, all_slots; lattice = :nodes, nchunks = 1)
            y2 = zeros(size(En, 1))
            Daisho.apply!(y2, En1, ar)
            @test y2 == yN
            a1 = zeros(3nc); a2 = zeros(3nc)
            Daisho.apply_adjoint!(a1, En, yN)
            Daisho.apply_adjoint!(a2, En1, yN)
            @test a1 == a2
            @test_throws ArgumentError Daisho.SplineEvaluation(grid, _VAR_NAMES, all_slots; lattice = :cells)
        end

        @testset "Threading determinism" begin
            nc_multi = max(Threads.nthreads(), 3)
            S1 = Daisho.SplineProjection(grid, _VAR_NAMES; nchunks = 1)
            SN = Daisho.SplineProjection(grid, _VAR_NAMES; nchunks = nc_multi)
            E1 = Daisho.SplineEvaluation(grid, _VAR_NAMES, all_slots; nchunks = 1)
            EN = Daisho.SplineEvaluation(grid, _VAR_NAMES, all_slots; nchunks = nc_multi)
            xa = randn(rng, 3nm)
            a1 = zeros(3nc); aN = zeros(3nc)
            Daisho.apply!(a1, S1, xa); Daisho.apply!(aN, SN, xa)
            @test a1 == aN
            x1 = zeros(3nm); xN = zeros(3nm)
            Daisho.apply_adjoint!(x1, S1, a1); Daisho.apply_adjoint!(xN, SN, a1)
            @test x1 == xN
            y1 = zeros(12nm); yN = zeros(12nm)
            Daisho.apply!(y1, E1, a1); Daisho.apply!(yN, EN, a1)
            @test y1 == yN
            yr = randn(rng, 12nm)
            Daisho.apply_adjoint!(a1, E1, yr); Daisho.apply_adjoint!(aN, EN, yr)
            @test a1 == aN
        end

        @testset "Allocations after warm-up" begin
            xa = randn(rng, 3nm); aa = zeros(3nc); ya = zeros(12nm)
            n1 = _alloc_after_warmup(Daisho.apply!, aa, S, xa)
            n2 = _alloc_after_warmup(Daisho.apply_adjoint!, xa, S, aa)
            n3 = _alloc_after_warmup(Daisho.apply!, ya, E, aa)
            n4 = _alloc_after_warmup(Daisho.apply_adjoint!, aa, E, ya)
            @info "Variational operator allocations (bytes, nthreads=$(Threads.nthreads()), mubar=$(mubar))" S_apply = n1 S_adjoint = n2 E_apply = n3 E_adjoint = n4
            # Only task creation allocates (~0.8 kB per spawned chunk per pencil
            # sweep, independent of grid size); the bound scales above 8 threads.
            bound = 200_000 * max(1.0, Threads.nthreads() / 8)
            @test n1 < bound
            @test n2 < bound
            @test n3 < bound
            @test n4 < bound
            S1 = Daisho.SplineProjection(grid, _VAR_NAMES; nchunks = 1)
            E1 = Daisho.SplineEvaluation(grid, _VAR_NAMES, all_slots; nchunks = 1)
            @test _alloc_after_warmup(Daisho.apply!, aa, S1, xa) == 0
            @test _alloc_after_warmup(Daisho.apply_adjoint!, xa, S1, aa) == 0
            @test _alloc_after_warmup(Daisho.apply!, ya, E1, aa) == 0
            @test _alloc_after_warmup(Daisho.apply_adjoint!, aa, E1, ya) == 0
        end
    end
    end

    # ── Quadratic reproduction and the W Dirichlet bottom ────────────────────
    @testset "Quadratic reproduction and BC enforcement" begin
        grid = _var_state_grid()
        gp = grid.params
        S = Daisho.SplineProjection(grid, _VAR_NAMES)
        E = Daisho.SplineEvaluation(grid, _VAR_NAMES, [(v, s) for v in 1:3 for s in 1:4])
        nm = gp.iDim * gp.jDim * gp.kDim
        nc = gp.b_iDim * gp.b_jDim * gp.b_kDim
        xs = Daisho.AxisSpline(grid.ibasis.data[1, 1, 1]).mishPoints
        ys = Daisho.AxisSpline(grid.jbasis.data[1, 1, 1]).mishPoints
        zs = Daisho.AxisSpline(grid.kbasis.data[1, 1, 1]).mishPoints
        Lx, Ly, Lz = 2000.0, 3000.0, 5000.0
        p(x, y, z) = (ξ = x / Lx; η = y / Ly; ζ = z / Lz;
                      1 + 2ξ + 3η - ζ + 0.5ξ^2 + ξ * η - 0.25ζ^2)
        px(x, y, z) = (2 + x / Lx + y / Ly) / Lx
        py(x, y, z) = (3 + x / Lx) / Ly
        pz(x, y, z) = (-1 - 0.5 * z / Lz) / Lz
        P = [f(xs[i], ys[j], zs[k]) for k in 1:gp.kDim, j in 1:gp.jDim, i in 1:gp.iDim, f in (p, px, py, pz)]
        x = repeat(vec(P[:, :, :, 1]), 3)
        a = zeros(3nc)
        Daisho.apply!(a, S, x)
        y = zeros(12nm)
        Daisho.apply!(y, E, a)
        for n in 1:2   # U, V carry R0 everywhere ⇒ exact reproduction
            @test maximum(abs.(Daisho.slab_view(y, E, (n-1)*4 + 1) .- P[:, :, :, 1])) < 1e-10
            for s in 2:4
                d = Daisho.slab_view(y, E, (n-1)*4 + s) .- P[:, :, :, s]
                @test maximum(abs.(d)) / maximum(abs.(P[:, :, :, s])) < 1e-9
            end
        end
        # W has a Dirichlet (R1T0) bottom: w(z = kmin) = 0 on every pencil.
        ksp = grid.kbasis.data[1, 1, gp.vars["W"]].params
        aw = reshape(view(a, 2nc+1:3nc), gp.b_kDim, gp.b_jDim, gp.b_iDim)
        wslab = Daisho.slab_view(y, E, 9)
        wmax = maximum(abs, wslab)
        @test wmax > 0.1
        # Evaluate the tensor spline on a column: collapse i and j at mish
        # points first, then evaluate the k-spline at z = kmin.
        axi = Daisho.AxisSpline(grid.ibasis.data[1, 1, gp.vars["W"]])
        axj = Daisho.AxisSpline(grid.jbasis.data[1, 1, gp.vars["W"]])
        for (ii, jj) in ((1, 1), (2, 5), (gp.iDim, gp.jDim), (7, 3))
            col = zeros(gp.b_kDim)
            for zc in 1:gp.b_kDim
                s = 0.0
                for ic in 1:4, jc in 1:4
                    s += axi.Φ0[ic, ii] * axj.Φ0[jc, jj] *
                         aw[zc, axj.first_coef[jj] + jc - 1, axi.first_coef[ii] + ic - 1]
                end
                col[zc] = s
            end
            wb = CubicBSpline.SItransform(ksp, col, [ksp.xmin], zeros(1), 0)[1]
            @test abs(wb) < 1e-12 * wmax
            # Sanity: the same column evaluated at the lowest mish point is nonzero.
            @test abs(CubicBSpline.SItransform(ksp, col, [zs[1]], zeros(1), 0)[1]) > 0
        end
    end

    @testset "Inhomogeneous (ahat) BCs are refused" begin
        for (bcl, bcr) in ((CubicBSpline.R3X, CubicBSpline.R0),
                           (CubicBSpline.R0, CubicBSpline.R3X),
                           (CubicBSpline.R1T1X, CubicBSpline.R0))
            sp = CubicBSpline.SplineParameters(xmin = 0.0, xmax = 10.0, num_cells = 6,
                                               BCL = bcl, BCR = bcr)
            @test_throws ArgumentError Daisho.AxisSpline(CubicBSpline.Spline1D(sp))
        end
    end
end
