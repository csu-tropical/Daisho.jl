#!/usr/bin/env julia
#
# Variational (3DVAR) wind-solver demo.
#
# The stage-2 counterpart to dual_doppler_demo.jl. Grids a dual-Doppler radar
# pair onto a Springsteel semi-spectral (RRR) grid through the stage-1
# `WindGridAccumulator`, then runs the incremental Gauss-Newton/CG 3DVAR
# solver (`analyze_radar_volumes_variational`) to produce a coupled
# `(u, v, w)` analysis that satisfies anelastic mass continuity — instead of
# stage 1's per-point least-squares `(u, v)` only. See the
# [Variational Wind guide](../src/guide/variational_wind.md) for the
# configuration this script exercises and the
# [theory page](../src/theory/variational_wind.md) for the mathematics.
#
# ── Input data ────────────────────────────────────────────────────────────
# Edit VOLUME_A_FILE / VOLUME_B_FILE below (or pass two paths positionally —
# see Usage) to point at a real dual-Doppler pair: two (or more) overlapping
# CfRadial volumes with
#   * a `velocity`-tagged field (dealiased, positive away from the radar) —
#     required, it selects the `WindGridAccumulator`;
#   * enough elevations to resolve the vertical structure (`w` needs beams
#     at several tilts crossing, not just the lowest);
#   * a `reflectivity`-tagged field if you want the DBZ overlay below to mean
#     something (the bundled `config/defaults.toml` already tags `DBZ`).
# There is no bundled/synthetic fallback here — real radar fixtures are
# gitignored in this repository (see `test/test_helpers.jl`'s
# `make_synthetic_dual_doppler_field` for the synthetic stand-in the test
# suite uses instead).
const VOLUME_A_FILE = joinpath(@__DIR__, "..", "..", "data", "variational_demo_radarA.nc")
const VOLUME_B_FILE = joinpath(@__DIR__, "..", "..", "data", "variational_demo_radarB.nc")
#
# Usage:
#   julia --project=. docs/examples/variational_demo.jl \
#       [radarA.nc radarB.nc [out_basename [config.toml]]]
#
#   radarA.nc, radarB.nc  CfRadial volumes (default: the constants above)
#   out_basename          NetCDF + plot basename (default: variational_demo)
#   config.toml           parameter file (default: the bundled defaults.toml)
#
# Notes:
#   * `[variational] enabled`, `[synthesis] include_vertical_velocity`, and
#     the surface `W` boundary condition are forced on regardless of what
#     `config.toml` says (see `build_variational_config`) — everything else
#     (grid extent, `[variational.background_error]` sigma/filter lengths,
#     radial-velocity/mass-continuity settings, ...) comes from that file, so
#     point it at your own copy to change the domain or tune the analysis.
#   * The bundled defaults use a 125 km / 50-cell horizontal domain (mubar 3
#     -> 150x150 mish points per level); for a quick look pass a coarser
#     config.toml, as in dual_doppler_demo.jl.
#   * Plotting needs CairoMakie; the analysis itself does not. If CairoMakie
#     is not installed, `run_variational_demo` below still runs and prints
#     every diagnostic — only the final plotting call is skipped (guarded by
#     `HAVE_CAIROMAKIE`).
#   * Troubleshooting sparse/unconverged output: see the guide's
#     "Troubleshooting" section (NOBS_VR, MCRESIDUAL, CG iteration counts).

using Daisho
using NCDatasets
using TOML
using Printf
using Dates

# CairoMakie is optional and only needed for the plotting section below; the
# analysis functions have no plotting dependency.
const HAVE_CAIROMAKIE = try
    using CairoMakie
    CairoMakie.activate!(type = "png")   # ensure the file backend is CairoMakie
    true
catch err
    @warn "CairoMakie not available; plotting will be skipped" exception = err
    false
end

"""Build the `[variational]` configuration from `config_path` (or the bundled
`config/defaults.toml` when `nothing`): force `[variational] enabled = true`,
`[synthesis] include_vertical_velocity = true` (the accumulator must carry a
`w` row for the radial-velocity super-observations), and a homogeneous
Dirichlet `W` boundary condition at the surface
(`[grid.springsteel.k.bc] W = { min = "dirichlet" }`, w = 0 at z = k_min) —
the three settings this demo needs that default to off/unset. Every other
key (grid geometry, `[variational.background_error]`, radial-velocity and
mass-continuity settings, ...) is read verbatim from the base file."""
function build_variational_config(config_path = nothing)
    base = TOML.parsefile(config_path === nothing ? Daisho.DEFAULTS_TOML_PATH : config_path)

    variational = get!(base, "variational", Dict{String,Any}())
    variational["enabled"] = true

    synthesis = get!(base, "synthesis", Dict{String,Any}())
    synthesis["include_vertical_velocity"] = true

    grid = get!(base, "grid", Dict{String,Any}())
    springsteel = get!(grid, "springsteel", Dict{String,Any}())
    kaxis = get!(springsteel, "k", Dict{String,Any}())
    bc = get!(kaxis, "bc", Dict{String,Any}())
    bc["W"] = Dict("min" => "dirichlet")

    path = tempname() * ".toml"
    open(io -> TOML.print(io, base), path, "w")
    return DaishoParameters(path)
end

"""Print the solver diagnostics this demo is built around: outer/inner
iteration counts and convergence (`result.stats`), the per-observation-kind
cost at the analysis (`result.cost_by_group`), the RMS anelastic
mass-continuity residual, and the fraction of mish points whose
radial-velocity super-observation carries at least 2 independent looks
(`NOBS_VR >= 2` — below that a point has only the background and continuity
to constrain it)."""
function print_variational_diagnostics(result, prob)
    st = result.stats
    @info "Solver stats" outer_iterations = st.outer_iterations converged = result.converged
    for (k, (n, ok)) in enumerate(zip(st.inner_iterations, st.inner_converged))
        @info @sprintf("  outer %d: %d CG iterations (inner tolerance reached: %s)", k, n, ok)
    end
    @info "Cost by observation group (½ Σ r (y - h(a))² at the analysis)" result.cost_by_group

    mc = mass_continuity_residual(prob, result)
    mc_rms = sqrt(sum(abs2, mc) / length(mc))
    @info @sprintf("Mass-continuity residual RMS = %.3e s⁻¹ (over %d mish points)", mc_rms, length(mc))

    if result.nobs_vr !== nothing
        frac = count(>=(2), result.nobs_vr) / length(result.nobs_vr)
        @info @sprintf("Mish points with NOBS_VR >= 2: %.1f%%", 100 * frac)
    end
    return nothing
end

"""Grid `volA_file`/`volB_file` onto a Springsteel grid built from `config`
(see `build_variational_config`), run the variational 3DVAR analysis, write
it to `<out_basename>.nc`, and print the diagnostics. Returns
`(result, problem, out_nc)`."""
function run_variational_demo(volA_file::AbstractString, volB_file::AbstractString;
                              out_basename::AbstractString = "variational_demo",
                              config = nothing)
    p = build_variational_config(config)
    sgrid = Daisho.create_radar_grid(p)
    @info @sprintf("Springsteel %s grid: %d x %d x %d cells (mubar %d), domain %.1f x %.1f x %.1f km",
        p.grid.springsteel.geometry, p.grid.springsteel.i.cells, p.grid.springsteel.j.cells,
        p.grid.springsteel.k.cells, p.grid.springsteel.mubar,
        (p.grid.springsteel.i.max - p.grid.springsteel.i.min) / 1000,
        (p.grid.springsteel.j.max - p.grid.springsteel.j.min) / 1000,
        (p.grid.springsteel.k.max - p.grid.springsteel.k.min) / 1000)

    @info "Reading $volA_file"
    volA = read_cfradial(volA_file)
    @info "Reading $volB_file"
    volB = read_cfradial(volB_file)
    vols = [volA, volB]

    itime = volA.time_coverage_start
    index_time = itime isa DateTime ? itime : DateTime(1970, 1, 1)

    out_nc = out_basename * ".nc"
    t_run = @elapsed (result, prob) = analyze_radar_volumes_variational(
        vols, out_nc, index_time, sgrid, p;
        institution = p.grid.metadata.institution, source = p.grid.metadata.source,
        return_problem = true)
    @info @sprintf("Solved in %.1f s", t_run)

    print_variational_diagnostics(result, prob)
    @info "Wrote $out_nc"
    return result, prob, out_nc
end

# ── Plotting (optional; needs CairoMakie) ───────────────────────────────────

# A concrete `Float64` `(nx, ny, nz)` array from a regular-grid variable,
# dropping the size-1 `time` dimension when present — the same layout
# `write_variational_netcdf` documents (dims `time, x, y, z`). NCDatasets
# masks the `_FillValue` (NaN on the regular grid) as `missing`, giving a
# `Union{Missing,Float64}` array that Makie cannot plot directly, so missing
# is folded back to `NaN`.
function _read_xyz(ds, name::AbstractString)
    v = ds[name]
    d = dimnames(v)
    a = length(d) == 4 && first(d) == "time" ? v[1, :, :, :] : v[:, :, :]
    return Float64.(coalesce.(a, NaN))
end

"""Horizontal cross-section: filled-contour `DBZ` with `(U, V)` vectors on
top, at regular-grid level `k` (default the middle level)."""
function plot_horizontal_cross_section(x, y, z_level::Real, U, V, DBZ; out_path)
    xs = x ./ 1000            # km
    ys = y ./ 1000

    fig = Figure(size = (980, 840))
    ax = Axis(fig[1, 1]; xlabel = "X east of origin (km)", ylabel = "Y north of origin (km)",
              aspect = DataAspect(),
              title = @sprintf("Variational analysis: U, V & DBZ at z = %.2f km", z_level / 1000))

    cf = contourf!(ax, xs, ys, DBZ; levels = 0:5:60, colormap = :turbo,
                   extendlow = :transparent, extendhigh = :auto)
    Colorbar(fig[1, 2], cf; label = "DBZ (dBZ)")

    target = 24
    sx = max(1, cld(length(xs), target)); sy = max(1, cld(length(ys), target))
    px = Float64[]; py = Float64[]; us = Float64[]; vs = Float64[]
    @inbounds for i in 1:sx:length(xs), j in 1:sy:length(ys)
        (isfinite(U[i, j]) && isfinite(V[i, j])) || continue
        push!(px, xs[i]); push!(py, ys[j]); push!(us, U[i, j]); push!(vs, V[i, j])
    end
    if isempty(px)
        @warn "No finite (U, V) at this level — nothing to overlay."
    else
        maxspd = maximum(hypot.(us, vs))
        cell_km = sx * (length(xs) > 1 ? (xs[2] - xs[1]) : 1.0)
        lengthscale = maxspd > 0 ? cell_km / maxspd : 1.0   # biggest arrow ≈ one cell
        arrows2d!(ax, px, py, us, vs; lengthscale = lengthscale, color = :black)
        text!(ax, xs[1], ys[end]; text = @sprintf("max |V| = %.1f m/s", maxspd),
              align = (:left, :top), offset = (6, -6), fontsize = 13,
              color = :black, font = :bold)
    end
    limits!(ax, first(xs), last(xs), first(ys), last(ys))
    save(out_path, fig)
    return out_path
end

"""Vertical (x, z) cross-section of `W` at regular-grid row `j` (default the
middle row) — the direct demonstration that the analysis has a
coupled, mass-continuity-consistent vertical velocity, not just (u, v)."""
function plot_vertical_cross_section(x, z, y_level::Real, W_xz; out_path)
    xs = x ./ 1000            # km
    zs = z ./ 1000

    fig = Figure(size = (980, 600))
    ax = Axis(fig[1, 1]; xlabel = "X east of origin (km)", ylabel = "Height (km)",
              title = @sprintf("Variational analysis: W at y = %.2f km", y_level / 1000))
    finite = filter(isfinite, vec(W_xz))
    wmax = isempty(finite) ? 1.0 : maximum(abs, finite)
    cf = contourf!(ax, xs, zs, W_xz; colormap = :balance,
                   levels = range(-wmax, wmax; length = 21))
    Colorbar(fig[1, 2], cf; label = "W (m/s)")
    limits!(ax, first(xs), last(xs), first(zs), last(zs))
    save(out_path, fig)
    return out_path
end

"""Read `U, V, W, DBZ` back from `out_nc` (`write_variational_netcdf`'s
regular-grid output, dims `time, x, y, z`) and produce the two cross-section
plots above. Returns `(horizontal_path, vertical_path)`."""
function plot_variational_demo(out_nc::AbstractString; out_basename = "variational_demo",
                               z_index = nothing, y_index = nothing)
    horiz_path = out_basename * "_horizontal.png"
    vert_path = out_basename * "_vertical.png"
    NCDataset(out_nc) do ds
        x = Float64.(ds["x"][:]); y = Float64.(ds["y"][:]); z = Float64.(ds["z"][:])
        nx, ny, nz = length(x), length(y), length(z)
        U = _read_xyz(ds, "U"); V = _read_xyz(ds, "V")
        W = _read_xyz(ds, "W"); DBZ = _read_xyz(ds, "DBZ")

        k = z_index === nothing ? cld(nz, 2) : clamp(Int(z_index), 1, nz)
        j = y_index === nothing ? cld(ny, 2) : clamp(Int(y_index), 1, ny)

        plot_horizontal_cross_section(x, y, z[k], view(U, :, :, k), view(V, :, :, k),
                                      view(DBZ, :, :, k); out_path = horiz_path)
        plot_vertical_cross_section(x, z, y[j], view(W, :, j, :); out_path = vert_path)
    end
    return horiz_path, vert_path
end

# ── CLI entry point ──────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    volA_file = length(ARGS) >= 1 ? ARGS[1] : VOLUME_A_FILE
    volB_file = length(ARGS) >= 2 ? ARGS[2] : VOLUME_B_FILE
    out_base  = length(ARGS) >= 3 ? ARGS[3] : "variational_demo"
    cfg       = length(ARGS) >= 4 ? ARGS[4] : nothing

    for f in (volA_file, volB_file)
        isfile(f) || error(
            "Not found: $f\n" *
            "Edit VOLUME_A_FILE / VOLUME_B_FILE at the top of this script, or pass two " *
            "CfRadial paths positionally:\n" *
            "  julia --project=. docs/examples/variational_demo.jl radarA.nc radarB.nc")
    end

    result, prob, out_nc = run_variational_demo(volA_file, volB_file;
        out_basename = out_base, config = cfg)

    if HAVE_CAIROMAKIE
        horiz, vert = plot_variational_demo(out_nc; out_basename = out_base)
        @info "Wrote $horiz and $vert"
    else
        @info "CairoMakie not installed; skipping plots (the NetCDF above has everything " *
              "plot_variational_demo would read)."
    end
end
