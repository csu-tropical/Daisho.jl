#!/usr/bin/env julia
#
# Springsteel scalar-gridding demo.
#
# The scalar-only counterpart to dual_doppler_demo.jl. Reads every CfRadial file
# in a directory and grids the configured SCALAR moments onto a Springsteel
# spectral grid (the `[grid.springsteel]` config) using the SAME edge-referenced,
# beamwidth-correct unified accumulator engine as the regular Cartesian path,
# then runs the spectral transform and writes the result through the Springsteel
# NetCDF writer.
#
# There is NO wind synthesis here — dual-Doppler on spectral grids is a separate
# follow-on. Multiple radars are accumulated onto ONE shared spectral grid
# centered on the centroid of the radar positions (same overlap geometry the
# dual-Doppler demo uses), so every radar lines up on the common origin.
#
# The Springsteel grid is non-uniform (Gauss-spaced nodes), so the accumulator is
# fed a representative radius-of-influence derived from the mean node spacing
# (`compute_roi`) instead of the axis increment — this is exactly the ROI-override
# path the regular accumulator gained for spectral grids.
#
# Usage:
#   julia --project=. docs/examples/springsteel_scalar_demo.jl <cfradial_dir> \
#       [out.png] [z_index] [config.toml]
#
#   <cfradial_dir>  directory of CfRadial (.nc / .cf) files (one or more radars)
#   out.png         output image                (default: springsteel_scalar.png)
#   z_index         altitude QUADRATURE node to plot (default: the middle node;
#                   3D only). The quadrature lattice has cells × mubar
#                   Gauss-spaced nodes per axis — it is NOT the evenly spaced
#                   NetCDF output grid (regular_out points per axis).
#   config.toml     parameter file with a
#                   [grid.springsteel] section     (default: the bundled defaults.toml)
#
# Notes:
#   * Output is one NetCDF (same basename as the image): the Springsteel spectral
#     grid with every configured scalar moment, written via write_radar_netcdf.
#   * The reflectivity field plotted is whatever carries the `define_detection`
#     tag in the config (not hard-coded). The plot shows the GRIDDED scalars
#     (the input to the spectral transform) on the spectral node axes; the
#     NetCDF carries the full spectrally-fit result.
#   * Geometry follows `[grid.springsteel].geometry` ("RRR"/"RR"/"R"); the plot is
#     produced only for 3D ("RRR") grids.

using Daisho
using Springsteel
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

"""Grid every configured scalar moment from `dir` onto a Springsteel spectral
grid and write the spectral NetCDF (+ a reflectivity plot for 3D grids).

This walks the DECOMPOSED pipeline because it also plots the gridded scalars
and prints coverage diagnostics, which need the accumulator. If you only want
the spectral NetCDF, the whole body below collapses to one call:

    Daisho.grid_radar_volume_spectral(vols, out_nc, nothing, sgrid, p)
"""
function run_springsteel_scalar(dir::AbstractString;
                                out_path = "springsteel_scalar.png",
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

    # Center the shared grid on the centroid of the radar positions, so the
    # domain sits over the radar overlap — same convention as the dual-Doppler
    # demo. The Springsteel node geometry (in meters) is fixed by the config;
    # only the reference lat/lon (where the origin lands) is set per run.
    refs = Daisho.volume_reference_position.(vols)
    ref_lat = sum(first, refs) / length(refs)
    ref_lon = sum(last, refs) / length(refs)

    # Build the Springsteel grid from [grid.springsteel], then express its
    # node lattice as a GridSpec (Cartesian tensor-product) centered on the
    # centroid so every radar projects onto the common origin.
    sgrid = Daisho.create_radar_grid(p)
    spec  = Daisho.build_springsteel_grid_spec(sgrid;
        reference_latitude = ref_lat, reference_longitude = ref_lon)
    # Two grids are in play: the accumulator grids onto the Gauss QUADRATURE
    # lattice (cells × mubar non-uniform nodes per axis — what this demo plots),
    # while write_radar_netcdf resamples the spectral fit onto a REGULAR output
    # grid (regular_out points per axis, default cells + 1).
    @info @sprintf("Springsteel %s grid: %d×%d×%d Gauss quadrature nodes (x,y,z = cells × mubar), origin (%.4f, %.4f)",
        p.grid.springsteel.geometry,
        length(spec.x_axis), length(spec.y_axis), length(spec.z_axis),
        ref_lat, ref_lon)
    @info @sprintf("NetCDF output grid: %d×%d×%d regularly spaced points (regular_out)",
        sgrid.params.i_regular_out, sgrid.params.j_regular_out,
        sgrid.params.k_regular_out)

    # Representative ROI from the mean (non-uniform) node spacing, fed to the
    # accumulator via the ROI override. `roi3 = (h, v)`; 2D/1D workers ignore the
    # component they don't use.
    roi  = Daisho.compute_roi(sgrid; h_factor = p.gridding.horizontal_roi_factor,
                                     v_factor = p.gridding.vertical_roi_factor)
    roi3 = (roi[1], length(roi) > 1 ? roi[2] : roi[1])

    # One scalar accumulator, one geometry pass per radar, all onto the shared
    # spectral node layout.
    acc = ScalarGridAccumulator(spec, p)
    for (f, vol) in zip(files, vols)
        @info "Gridding $(basename(f)): $(vol.instrument_name), $(length(vol.sweeps)) sweep(s)"
        for s in eachindex(vol.sweeps)
            grid_sweep!(acc, vol, s, p; roi = roi3)
        end
    end
    radar_grid = finalize_grid(acc)

    # Copy the gridded scalars into the Springsteel physical array (fill→NaN,
    # undetect preserved), run the spectral transform with fill masking, and
    # write the spectral NetCDF.
    moment_dict = Daisho.field_index_dict(p)
    Daisho.populate_physical!(sgrid, radar_grid, moment_dict;
                              fill_value = p.io.fill_value)
    mask = Daisho.mask_fill_for_transform!(sgrid)
    spectralTransform!(sgrid)
    gridTransform!(sgrid)
    Daisho.restore_fill_from_mask!(sgrid, mask)

    base   = splitext(out_path)[1]
    out_nc = base * ".nc"
    Daisho.write_radar_netcdf(out_nc, sgrid, vols[1], moment_dict;
        ref_lat = ref_lat, ref_lon = ref_lon,
        institution = p.grid.metadata.institution,
        source = p.grid.metadata.source)
    @info "Wrote $out_nc (" * join(sort(acc.fields), ", ") * ")"

    print_coverage_diagnostics(acc)

    # Plot a reflectivity slice (3D grids only).
    if spec.shape === :volume_3d
        refl_field = Daisho.field_with_tag(p, :define_detection;
            for_op = "springsteel_scalar_demo reflectivity plot")
        refl_col = moment_dict[refl_field]
        nz = length(spec.z_axis)
        k = z_index === nothing ? cld(nz, 2) : clamp(Int(z_index), 1, nz)
        @info "Plotting quadrature node $k of $nz (z = $(round(spec.z_axis[k]; digits = 1)) m; " *
              "Gauss-spaced — NOT the evenly spaced NetCDF output levels)"
        plot_reflectivity(spec, radar_grid, refl_col, k, p;
                          refl_field = refl_field, out_path = out_path)
        @info "Wrote $out_path"
        return out_path
    else
        @info "Geometry $(spec.shape) is not 3D — skipping the reflectivity plot."
        return out_nc
    end
end

"""Print how many grid cells received gridded data — the first thing to check
when output looks sparse."""
function print_coverage_diagnostics(acc)
    cov = acc.coverage
    total    = length(cov)
    detected = count(==(Int8(2)), cov)    # scanned AND a detection reached the cell
    scanned  = count(==(Int8(1)), cov)    # scanned, no echo (undetect)
    nodata   = total - detected - scanned
    pct(x) = 100 * x / total
    @info "Coverage over all $total grid cells (×$(length(acc.fields)) fields):"
    @info @sprintf("  detection in cell    : %9d (%.2f%%)", detected, pct(detected))
    @info @sprintf("  scanned, no echo     : %9d (%.2f%%)", scanned, pct(scanned))
    @info @sprintf("  no gate reached cell : %9d (%.2f%%)", nodata, pct(nodata))
    return nothing
end

"""Filled-contour the gridded reflectivity slice at node level `k`, on the
spectral (non-uniform) node axes."""
function plot_reflectivity(spec, radar_grid, refl_col, k, p;
                           refl_field = "reflectivity", out_path)
    xs = spec.x_axis ./ 1000          # km
    ys = spec.y_axis ./ 1000
    z_km = spec.z_axis[k] / 1000

    # Reflectivity slice → (nx, ny) for Makie; mask true-missing/undetect to NaN.
    dbz = Array{Float64}(undef, length(xs), length(ys))
    @inbounds for i in eachindex(xs), j in eachindex(ys)
        v = radar_grid[refl_col, k, j, i]
        dbz[i, j] = (v == p.io.fill_value || v == p.io.undetect) ? NaN : v
    end

    fig = Figure(size = (980, 840))
    ax = Axis(fig[1, 1]; xlabel = "X east of origin (km)",
              ylabel = "Y north of origin (km)", aspect = DataAspect(),
              title = @sprintf("Springsteel-gridded reflectivity at z = %.1f km", z_km))

    cf = contourf!(ax, xs, ys, dbz; levels = 0:5:60, colormap = :turbo,
                   extendlow = :transparent, extendhigh = :auto)
    Colorbar(fig[1, 2], cf; label = "$refl_field (dBZ)")

    limits!(ax, first(xs), last(xs), first(ys), last(ys))
    save(out_path, fig)
    return out_path
end

# ── CLI entry point ──────────────────────────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) && error(
        "usage: julia --project=. docs/examples/springsteel_scalar_demo.jl " *
        "<cfradial_dir> [out.png] [z_index] [config.toml]")
    dir    = ARGS[1]
    outpng = length(ARGS) >= 2 ? ARGS[2] : "springsteel_scalar.png"
    zidx   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : nothing
    cfg    = length(ARGS) >= 4 ? ARGS[4] : nothing
    run_springsteel_scalar(dir; out_path = outpng, z_index = zidx, config = cfg)
end
