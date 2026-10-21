#!/usr/bin/env julia
#
# Variational (3DVAR) wind-solver demo.
#
# The stage-2 counterpart to dual_doppler_demo.jl. Grids a folder of radar
# files onto a Springsteel semi-spectral (RRR) grid through the stage-1
# `WindGridAccumulator`, then runs the incremental Gauss-Newton/CG 3DVAR
# solver (`analyze_radar_volumes_variational`) to produce a coupled
# `(u, v, w)` analysis that satisfies anelastic mass continuity — instead of
# stage 1's per-point least-squares `(u, v)` only. See the
# [Variational Wind guide](../src/guide/variational_wind.md) for the
# configuration this script exercises and the
# [theory page](../src/theory/variational_wind.md) for the mathematics.
#
# ── Input data ────────────────────────────────────────────────────────────
# A DIRECTORY of CfRadial files (`.nc` / `.cf`): full volumes or single
# sweeps, from one radar or several. Airborne (e.g. tail-radar) archives are
# typically stored one sweep per file, so a folder may hold dozens of files;
# they are all welcome. No per-radar grouping is needed — every sweep of
# every file accumulates into one shared accumulator. The files together need
#   * a `velocity`-tagged field (dealiased, positive away from the radar) —
#     required, it selects the `WindGridAccumulator`;
#   * enough viewing geometry to resolve the vertical structure (`w` needs
#     beams crossing at several tilts/azimuths, not just the lowest tilt);
#   * a `define_detection`-tagged field (reflectivity) if you want the overlay
#     below to mean something (the bundled `config/defaults.toml` tags `DBZ`).
# There is no bundled/synthetic fallback here — real radar fixtures are
# gitignored in this repository (see `test/test_helpers.jl`'s
# `make_synthetic_dual_doppler_field` for the synthetic stand-in the test
# suite uses instead).
#
# Usage:
#   julia --project=. docs/examples/variational_demo.jl \
#       <cfradial_dir> [out_basename [config.toml]]
#
#   cfradial_dir  directory of CfRadial files (required)
#   out_basename  NetCDF + plot basename (default: variational_demo)
#   config.toml   parameter file (default: the bundled defaults.toml)
#
# Notes:
#   * `[variational] enabled`, `[synthesis] include_vertical_velocity`, and
#     the surface `W` boundary condition are forced on regardless of what
#     `config.toml` says (see `build_variational_config`) — everything else
#     (grid extent, `[variational.background_error]` sigma/filter lengths,
#     radial-velocity/mass-continuity settings, ...) comes from that file, so
#     point it at your own copy to change the domain or tune the analysis.
#   * Because `include_vertical_velocity` is forced on, `[synthesis]
#     max_elevation` does not apply: steep (airborne) beams are kept.
#   * The analysis time (`index_time`) is the earliest `time_coverage_start`
#     found across all files.
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

"""Collect CfRadial files (.nc / .cf) in `dir`, sorted by name."""
function find_cfradial_files(dir::AbstractString)
    isdir(dir) || error("Not a directory: $dir")
    files = filter(readdir(dir; join = true)) do f
        isfile(f) && (endswith(lowercase(f), ".nc") || endswith(lowercase(f), ".cf"))
    end
    return sort(files)
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

    # Same residual at the spline nodes (keyword added separately; skip
    # silently when this Daisho version does not have it yet).
    try
        mcn = mass_continuity_residual(prob, result; lattice = :nodes)
        mcn_rms = sqrt(sum(abs2, mcn) / length(mcn))
        @info @sprintf("Mass-continuity residual RMS = %.3e s⁻¹ (over %d spline nodes)", mcn_rms, length(mcn))
    catch err
        err isa MethodError || rethrow()
    end

    if result.nobs_vr !== nothing
        frac = count(>=(2), result.nobs_vr) / length(result.nobs_vr)
        @info @sprintf("Mish points with NOBS_VR >= 2: %.1f%%", 100 * frac)
    end
    return nothing
end

"""Read every CfRadial file in `dir` (volumes or single sweeps, any number of
radars), grid them all onto a Springsteel grid built from `config` (see
`build_variational_config`), run the variational 3DVAR analysis, write it to
`<out_basename>.nc`, and print the diagnostics. Returns
`(result, problem, out_nc)`."""
function run_variational_demo(dir::AbstractString;
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

    files = find_cfradial_files(dir)
    isempty(files) && error("No CfRadial (.nc/.cf) files found in $dir")
    @info "Found $(length(files)) CfRadial file(s)"

    vols = [read_cfradial(f) for f in files]
    for (f, vol) in zip(files, vols)
        @info "Read $(basename(f)): $(vol.instrument_name), $(length(vol.sweeps)) sweep(s)"
    end

    # Earliest start time across all files (not necessarily the first file).
    starts = [v.time_coverage_start for v in vols if v.time_coverage_start isa DateTime]
    index_time = isempty(starts) ? DateTime(1970, 1, 1) : minimum(starts)

    out_nc = out_basename * ".nc"
    t_run = @elapsed (result, prob) = analyze_radar_volumes_variational(
        vols, out_nc, index_time, sgrid, p;
        institution = p.grid.metadata.institution, source = p.grid.metadata.source,
        return_problem = true)
    @info @sprintf("Solved in %.1f s", t_run)

    print_variational_diagnostics(result, prob)
    @info "Wrote $out_nc"
    print_qflag_summary(out_nc, p)
    return result, prob, out_nc
end

"""Read `QFLAG` back from `out_nc` (when present) and print the fraction of
regular-grid points with a nonzero flag and, if `[variational]
mask_quality_flags` is non-empty, the fraction those flags blank."""
function print_qflag_summary(out_nc::AbstractString, p)
    NCDataset(out_nc) do ds
        haskey(ds, "QFLAG") || return nothing
        v = ds["QFLAG"]
        d = dimnames(v)
        q = length(d) == 4 && first(d) == "time" ? v[1, :, :, :] : v[:, :, :]
        qi = Int.(coalesce.(q, 0))
        n = length(qi)
        n == 0 && return nothing
        @info @sprintf("QFLAG != 0 at %.1f%% of regular-grid points", 100 * count(!=(0), qi) / n)
        mf = p.variational.mask_quality_flags
        if !isempty(mf)
            bits = Int(Daisho.qflag_bits(mf))
            @info @sprintf("Masked by mask_quality_flags %s: %.1f%% of regular-grid points",
                           string(mf), 100 * count(x -> (x & bits) != 0, qi) / n)
        end
    end
    return nothing
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

"""Read `U, V, W` and the reflectivity field `refl_field` (the
`define_detection`-tagged field) back from `out_nc` (`write_variational_netcdf`'s
regular-grid output, dims `time, x, y, z`) and produce the two cross-section
plots above. Returns `(horizontal_path, vertical_path)`."""
function plot_variational_demo(out_nc::AbstractString; out_basename = "variational_demo",
                               z_index = nothing, y_index = nothing,
                               refl_field::AbstractString = "DBZ")
    horiz_path = out_basename * "_horizontal.png"
    vert_path = out_basename * "_vertical.png"
    NCDataset(out_nc) do ds
        x = Float64.(ds["x"][:]); y = Float64.(ds["y"][:]); z = Float64.(ds["z"][:])
        nx, ny, nz = length(x), length(y), length(z)
        U = _read_xyz(ds, "U"); V = _read_xyz(ds, "V")
        W = _read_xyz(ds, "W"); DBZ = _read_xyz(ds, refl_field)

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
    isempty(ARGS) && error(
        "usage: julia --project=. docs/examples/variational_demo.jl " *
        "<cfradial_dir> [out_basename [config.toml]]")
    dir      = ARGS[1]
    out_base = length(ARGS) >= 2 ? ARGS[2] : "variational_demo"
    cfg      = length(ARGS) >= 3 ? ARGS[3] : nothing

    result, prob, out_nc = run_variational_demo(dir; out_basename = out_base, config = cfg)

    if HAVE_CAIROMAKIE
        # Reflectivity = whatever field carries the `define_detection` tag.
        refl_field = Daisho.field_with_tag(build_variational_config(cfg), :define_detection;
                                           for_op = "variational_demo reflectivity plot")
        horiz, vert = plot_variational_demo(out_nc; out_basename = out_base,
                                            refl_field = refl_field)
        @info "Wrote $horiz and $vert"
    else
        @info "CairoMakie not installed; skipping plots (the NetCDF above has everything " *
              "plot_variational_demo would read)."
    end
end
