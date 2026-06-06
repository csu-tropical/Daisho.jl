#!/usr/bin/env julia
#
# Dual-Doppler wind-synthesis demo.
#
# Reads every CfRadial file in a directory and grids every configured field plus
# the dual-Doppler horizontal wind (u, v) in ONE single-pass accumulator
# (`build_accumulator` returns a `WindGridAccumulator` whenever a field carries
# the `velocity` tag). It writes the full 3D result — gridded scalars AND the
# wind product — to a single NetCDF with `write_grid_products`, and plots the
# reflectivity field overlaid with the (U, V) wind vectors at one altitude.
#
# The reflectivity field is whatever carries the `define_detection` tag in the
# config (not hard-coded). Alongside the PNG it writes one NetCDF (same
# basename): `*.nc` with the gridded scalar fields (reflectivity, etc.) AND
# U, V, USTD, VSTD, DET, NGATES, QFLAG on every level, so you can inspect other
# altitudes and compare coverage without re-running.
#
# Everything is built on ONE shared grid (taken from the centroid of the radar
# positions with the [grid.cartesian] spec) in a single geometry pass, so every
# radar — and the wind solve — line up. Gates are projected relative to that
# common origin, so no per-radar bookkeeping is needed here.
#
# Usage:
#   julia --project=. docs/examples/dual_doppler_demo.jl <cfradial_dir> \
#       [out.png] [z_index] [config.toml]
#
#   <cfradial_dir>  directory of CfRadial (.nc / .cf) files (≥2 radars for a
#                   real dual-Doppler solve; more radars just keep accumulating)
#   out.png         output image           (default: dual_doppler.png)
#   z_index         altitude level to plot (default: the middle level)
#   config.toml     parameter file         (default: the bundled defaults.toml)
#
# Notes:
#   * Reflectivity is gridded through the standard path, which only fills a cell
#     where the config's `define_scanned` AND `define_detection` fields are
#     present in the data. If your files have reflectivity but no separate
#     scanned-indicator field, tag the reflectivity field with both roles, e.g.
#       DZ = ["linear_interp", "define_detection", "define_scanned"]
#   * Velocity must be the `velocity`-tagged field and is assumed dealiased
#     (positive away from the radar).
#   * The bundled defaults use a 125 km / 500 m grid (501×501×37); for a quick
#     look pass a coarser config.toml.
#   * Sparse output? Check the printed coverage breakdown (or NGATES/QFLAG in the
#     NetCDF): mostly "singular" → beams aren't crossing (geometry / too few
#     elevations); mostly "no data" → grid too large/off-center or ROI too small
#     for the gate spacing; mostly "σ above threshold" → raise max_sigma_*.

using Daisho
using CairoMakie
using Printf
using Dates

CairoMakie.activate!(type = "png")   # ensure the file backend is CairoMakie

"""Collect CfRadial files (.nc / .cf) in `dir`, sorted by name."""
function find_cfradial_files(dir::AbstractString)
    isdir(dir) || error("Not a directory: $dir")
    files = filter(readdir(dir; join = true)) do f
        isfile(f) && (endswith(lowercase(f), ".nc") || endswith(lowercase(f), ".cf"))
    end
    return sort(files)
end

"""Reference (lat, lon) of a volume: the stationary position, or the first
sweep's georeference for a mobile platform reporting (0, 0)."""
function volume_refpos(vol)
    lat, lon = Float64(vol.latitude), Float64(vol.longitude)
    if lat == 0.0 && lon == 0.0 && !isempty(vol.sweeps)
        s = vol.sweeps[1]
        if s.georeference !== nothing && !isempty(s.georeference.latitude)
            return (Float64(s.georeference.latitude[1]),
                    Float64(s.georeference.longitude[1]))
        end
    end
    return (lat, lon)
end

"""A `:volume_3d` GridSpec from the `[grid.cartesian]` axes, centered on
`(ref_lat, ref_lon)`."""
function cartesian_grid_spec(p, ref_lat, ref_lon)
    g = p.grid.cartesian
    return GridSpec(shape = :volume_3d,
        reference_latitude = ref_lat, reference_longitude = ref_lon,
        x_axis = collect(Float64, g.xmin .+ (0:(g.xdim - 1)) .* g.xincr),
        y_axis = collect(Float64, g.ymin .+ (0:(g.ydim - 1)) .* g.yincr),
        z_axis = collect(Float64, g.zmin .+ (0:(g.zdim - 1)) .* g.zincr))
end

"""Print a breakdown of the per-cell quality flag and gate coverage — the first
thing to look at when the retrieval is sparse. (The same fields are in the
NetCDF as NGATES / DET / QFLAG.)"""
function print_coverage_diagnostics(wind)
    qf = wind.quality_flag
    ng = wind.n_gates
    total = length(qf)
    good     = count(==(Int8(0)), qf)
    nodata   = count(f -> (f & Daisho.QFLAG_NODATA)   != 0, qf)
    singular = count(f -> (f & Daisho.QFLAG_SINGULAR) != 0, qf)
    sigma    = total - good - nodata - singular            # solvable but σ too high
    covered  = count(>(Int32(0)), ng)
    pct(x) = 100 * x / total
    @info "Coverage diagnostics over all $total grid cells:"
    @info @sprintf("  solvable & within σ  : %9d (%.2f%%)", good, pct(good))
    @info @sprintf("  σ above threshold    : %9d (%.2f%%)", sigma, pct(sigma))
    @info @sprintf("  singular geometry    : %9d (%.2f%%)  ← one look / no beam crossing", singular, pct(singular))
    @info @sprintf("  no data              : %9d (%.2f%%)  ← no weighted gate reached cell", nodata, pct(nodata))
    @info @sprintf("  cells with ≥1 gate   : %9d (%.2f%%); max gates in a cell = %d",
                   covered, pct(covered), maximum(ng))
    return nothing
end

"""Run the dual-Doppler analysis over `dir` and write a plot to `out_path`."""
function run_dual_doppler(dir::AbstractString; out_path = "dual_doppler.png",
                          z_index = nothing, config = nothing)
    p = config === nothing ? DaishoParameters() : DaishoParameters(config)

    # CairoMakie's save() needs a recognized image extension; default to .png.
    if !(lowercase(splitext(out_path)[2]) in (".png", ".pdf", ".svg"))
        out_path *= ".png"
    end

    files = find_cfradial_files(dir)
    isempty(files) && error("No CfRadial (.nc/.cf) files found in $dir")
    @info "Found $(length(files)) CfRadial file(s)"

    vols = [read_cfradial(f) for f in files]

    # Center one shared grid on the centroid of the radar positions, so the
    # analysis domain sits over the radar overlap (where dual-Doppler resolves).
    refs = volume_refpos.(vols)
    ref_lat = sum(first, refs) / length(refs)
    ref_lon = sum(last, refs) / length(refs)
    grid_spec = cartesian_grid_spec(p, ref_lat, ref_lon)
    @info @sprintf("Grid: %d×%d×%d (x,y,z), origin (%.4f, %.4f)",
        length(grid_spec.x_axis), length(grid_spec.y_axis), length(grid_spec.z_axis),
        ref_lat, ref_lon)

    # One accumulator, one geometry pass: the config's velocity tag selects the
    # WindGridAccumulator (gridded scalars + dual-Doppler wind together).
    acc = build_accumulator(grid_spec, p)

    for (f, vol) in zip(files, vols)
        @info "Gridding $(basename(f)): $(vol.instrument_name), $(length(vol.sweeps)) sweep(s)"
        for s in eachindex(vol.sweeps)
            grid_sweep!(acc, vol, s, p)
        end
    end

    dbz_grid = finalize_grid(acc.scalar)           # (n_fields, nz, ny, nx)
    wind     = finalize_wind(acc, p)               # SynthesisOutput

    # Reflectivity = whatever field carries the `define_detection` tag (no
    # hard-coded name), so the plot tracks the user's config.
    refl_field = Daisho.field_with_tag(p, :define_detection;
                                       for_op = "dual_doppler_demo reflectivity plot")
    refl_col = Daisho.field_index_dict(p)[refl_field]

    itime = vols[1].time_coverage_start
    index_time = itime isa DateTime ? itime : DateTime(1970, 1, 1)
    base = splitext(out_path)[1]

    # One combined NetCDF: gridded scalar fields (reflectivity, etc.) AND the
    # wind product (U, V, USTD, VSTD, DET, NGATES, QFLAG) on the SAME grid, so
    # reflectivity coverage and the wind diagnostics line up cell-for-cell.
    out_nc = base * ".nc"
    write_grid_products(out_nc, acc, p; index_time = index_time)
    @info "Wrote $out_nc (" *
          join(sort(acc.scalar.fields), ", ") *
          " + U, V, USTD, VSTD, DET, NGATES, QFLAG)"

    print_coverage_diagnostics(wind)

    nz = length(grid_spec.z_axis)
    k = z_index === nothing ? cld(nz, 2) : clamp(Int(z_index), 1, nz)

    nsolved = count(==(Int8(0)), @view wind.quality_flag[k, :, :])
    @info "Level $k of $nz (z = $(grid_spec.z_axis[k]) m): " *
          "$nsolved well-conditioned wind points"

    plot_dual_doppler(grid_spec, dbz_grid, refl_col, wind, k, p;
                      refl_field = refl_field, out_path = out_path)
    @info "Wrote $out_path"
    return out_path
end

"""Plot reflectivity filled contours + (U, V) vectors at level `k`."""
function plot_dual_doppler(grid_spec, dbz_grid, refl_col, wind, k, p;
                           refl_field = "reflectivity", out_path)
    xs = grid_spec.x_axis ./ 1000          # km
    ys = grid_spec.y_axis ./ 1000
    z_km = grid_spec.z_axis[k] / 1000

    # Reflectivity slice → (nx, ny) for Makie; mask true-missing/undetect to NaN.
    dbz = Array{Float64}(undef, length(xs), length(ys))
    @inbounds for i in eachindex(xs), j in eachindex(ys)
        v = dbz_grid[refl_col, k, j, i]
        dbz[i, j] = (v == p.io.fill_value || v == p.io.undetect) ? NaN : v
    end

    fig = Figure(size = (980, 840))
    ax = Axis(fig[1, 1]; xlabel = "X east of origin (km)",
              ylabel = "Y north of origin (km)", aspect = DataAspect(),
              title = @sprintf("Dual-Doppler wind & reflectivity at z = %.1f km", z_km))

    cf = contourf!(ax, xs, ys, dbz; levels = 0:5:60, colormap = :turbo,
                   extendlow = :transparent, extendhigh = :auto)
    Colorbar(fig[1, 2], cf; label = "$refl_field (dBZ)")

    # Subsample to ~28 vectors across; only plot well-conditioned points
    # (quality_flag == 0: solvable and both σ within threshold).
    target = 28
    sx = max(1, cld(length(xs), target))
    sy = max(1, cld(length(ys), target))
    px = Float64[]; py = Float64[]; us = Float64[]; vs = Float64[]
    @inbounds for i in 1:sx:length(xs), j in 1:sy:length(ys)
        wind.quality_flag[k, j, i] == Int8(0) || continue
        push!(px, xs[i]); push!(py, ys[j])
        push!(us, wind.comp1[k, j, i]); push!(vs, wind.comp2[k, j, i])
    end

    if isempty(px)
        @warn "No well-conditioned wind vectors at level $k — nothing to overlay. " *
              "Need ≥2 radars with crossing beams covering this level."
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

# ── CLI entry point ──────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error(
        "usage: julia --project=. docs/examples/dual_doppler_demo.jl " *
        "<cfradial_dir> [out.png] [z_index] [config.toml]")
    dir    = ARGS[1]
    outpng = length(ARGS) >= 2 ? ARGS[2] : "dual_doppler.png"
    zidx   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing
    cfg    = length(ARGS) >= 4 ? ARGS[4] : nothing
    run_dual_doppler(dir; out_path = outpng, z_index = zidx, config = cfg)
end
