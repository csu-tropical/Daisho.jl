#!/usr/bin/env julia
#
# Add echo products (hydrometeor ID + rain rate) to existing gridded NetCDF files.
#
# Standalone driver for `add_echo_products!`: it appends the fuzzy hydrometeor
# classification (HID/FHC) and polarimetric rain-rate fields to one or more
# already-written Daisho gridded NetCDF files, computing them from the GRIDDED
# radar variables (the beam-power-weighted averages). This is the in-place,
# reprocess-archived-grids path — no regridding required — and it handles
# multi-time concatenated files by looping over the time dimension.
#
# Usage:
#   julia --project=. docs/examples/add_echo_products_demo.jl <config.toml> <grid.nc> [more.nc ...]
#
#   <config.toml>   parameter file with an [echo] block (and [fields], [io]).
#                   The [echo] block selects the band, which products to write
#                   (compute_fhc / compute_blended_rain / rain_components), the
#                   input field names, and the T(z) profile. See the [echo]
#                   section of config/defaults.toml for the full template.
#   <grid.nc> ...   one or more gridded files (volume X/Y/Z, PPI X/Y, or RHI R/Z;
#                   single- or multi-time). Modified IN PLACE.
#
# Each file gains new variables (e.g. HID_CSU, RATE_CSU_BLENDED, and any
# individual rain components requested) with the same dimensions as the existing
# moment fields. Re-running overwrites those variables rather than duplicating.

using Daisho

function main(args)
    if length(args) < 2
        println("Usage: julia --project=. docs/examples/add_echo_products_demo.jl " *
                "<config.toml> <grid.nc> [more.nc ...]")
        return 1
    end

    config = args[1]
    files = args[2:end]

    p = DaishoParameters(config)
    e = p.echo

    # Friendly pre-flight checks (the function itself still runs regardless).
    if !(:echo in p.provided)
        @warn "No [echo] block in $config; using defaults — temperature will NOT " *
              "be used and field names default to DBZ/ZDR/KDP/RHOHV."
    end
    if e.use_temp && e.temp_source === :profile && e.temperature === nothing
        @warn "[echo] requests use_temp with temp_source=\"profile\" but no " *
              "temperature profile is configured; FHC will run WITHOUT temperature."
    elseif e.use_temp && e.temp_source === :reference_state
        @warn "[echo] temp_source=\"reference_state\" is not yet implemented; " *
              "processing will error. Use \"profile\" or \"field\"."
    end

    temp_desc = !e.use_temp ? "off" :
        e.temp_source === :field ? "field $(e.temp_field) [$(e.temp_field_units)]" :
        e.temp_source === :profile ? "profile" : String(e.temp_source)

    println("Echo products configuration:")
    println("  band                 = ", e.band)
    println("  compute_fhc          = ", e.compute_fhc, "  -> ", e.fhc_output)
    println("  compute_blended_rain = ", e.compute_blended_rain, "  -> ", e.rain_output)
    println("  rain_components      = ", isempty(e.rain_components) ? "(none)" :
            join(e.rain_components, ", "))
    println("  inputs               = ", e.dbz_field, ", ", e.zdr_field, ", ",
            e.kdp_field, ", ", e.rhohv_field,
            isempty(e.ldr_field) ? "" : ", " * e.ldr_field)
    println("  temperature          = ", temp_desc)
    println()

    for f in files
        if !isfile(f)
            @warn "Skipping missing file: $f"
            continue
        end
        print("Processing $f ... ")
        written = add_echo_products!(f, p)
        println("wrote: ", join(written, ", "))
    end
    return 0
end

# Run when invoked as a script (not when included for inspection).
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
