#!/usr/bin/env julia
#
# Dual-Doppler wind-synthesis demo.
#
# Reads every CfRadial file in a directory, grids reflectivity (the existing
# scalar GridAccumulator path) and projects each radar's radial velocity into a
# shared dual-Doppler normal system (the WindGridAccumulator path), solves for
# the horizontal wind (u, v), and plots DBZ filled contours overlaid with the
# (U, V) wind vectors at one altitude.
#
# Both products are built on ONE shared grid (taken from the first volume's
# reference position with the [grid.cartesian] spec), so every radar — and the
# wind solve — line up. The wind workers project each radar's gates relative to
# that common origin, so no per-radar bookkeeping is needed here.
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
#     where the config's `define_scanned` / `define_detection` fields are
#     present in the data. With the bundled defaults those are SQI / DBZ — if
#     your files lack SQI, tag DBZ with both roles in your config, e.g.
#       DBZ = ["linear_interp", "define_detection", "define_scanned"]
#   * Velocity must be the `velocity`-tagged field (VEL in the defaults) and is
#     assumed dealiased (positive away from the radar).
#   * The bundled defaults use a 125 km / 500 m grid (501×501×37); for a quick
#     look pass a coarser config.toml.

using Daisho
using CairoMakie
using Printf

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

"""Run the dual-Doppler analysis over `dir` and write a plot to `out_path`."""
function run_dual_doppler(dir::AbstractString; out_path = "dual_doppler.png",
                          z_index = nothing, config = nothing)
    p = config === nothing ? DaishoParameters() : DaishoParameters(config)

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

    dbz_acc  = GridAccumulator(grid_spec, p)       # scalar fields, incl. DBZ
    wind_acc = WindGridAccumulator(grid_spec, p)   # VEL → (u, v)

    for (f, vol) in zip(files, vols)
        @info "Gridding $(basename(f)): $(vol.instrument_name), $(length(vol.sweeps)) sweep(s)"
        for s in eachindex(vol.sweeps)
            grid_sweep!(dbz_acc, vol, s, p)
            grid_sweep_wind!(wind_acc, vol, s, p)
        end
    end

    dbz_grid = finalize_grid(dbz_acc)              # (n_fields, nz, ny, nx)
    wind     = finalize_wind(wind_acc, p)          # SynthesisOutput

    fidx = Daisho.field_index_dict(p)
    haskey(fidx, "DBZ") ||
        error("Config has no DBZ field (fields: $(sort(collect(keys(fidx)))))")
    dbz_col = fidx["DBZ"]

    nz = length(grid_spec.z_axis)
    k = z_index === nothing ? cld(nz, 2) : clamp(Int(z_index), 1, nz)

    nsolved = count(==(Int8(0)), @view wind.quality_flag[k, :, :])
    @info "Level $k of $nz (z = $(grid_spec.z_axis[k]) m): " *
          "$nsolved well-conditioned wind points"

    plot_dual_doppler(grid_spec, dbz_grid, dbz_col, wind, k, p; out_path = out_path)
    @info "Wrote $out_path"
    return out_path
end

"""Plot DBZ filled contours + (U, V) vectors at level `k`."""
function plot_dual_doppler(grid_spec, dbz_grid, dbz_col, wind, k, p; out_path)
    xs = grid_spec.x_axis ./ 1000          # km
    ys = grid_spec.y_axis ./ 1000
    z_km = grid_spec.z_axis[k] / 1000

    # DBZ slice → (nx, ny) for Makie; mask true-missing and undetect to NaN.
    dbz = Array{Float64}(undef, length(xs), length(ys))
    @inbounds for i in eachindex(xs), j in eachindex(ys)
        v = dbz_grid[dbz_col, k, j, i]
        dbz[i, j] = (v == p.io.fill_value || v == p.io.undetect) ? NaN : v
    end

    fig = Figure(size = (980, 840))
    ax = Axis(fig[1, 1]; xlabel = "X east of origin (km)",
              ylabel = "Y north of origin (km)", aspect = DataAspect(),
              title = @sprintf("Dual-Doppler wind & reflectivity at z = %.1f km", z_km))

    cf = contourf!(ax, xs, ys, dbz; levels = 0:5:60, colormap = :turbo,
                   extendlow = :transparent, extendhigh = :auto)
    Colorbar(fig[1, 2], cf; label = "Reflectivity (dBZ)")

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
