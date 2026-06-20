# Post-gridding echo products: hydrometeor ID + rain rate
#
# Orchestrates the CSU_RadarTools ports ([`csu_fhc_summer`](@ref),
# [`calc_blended_rain_tropical`](@ref) and the component rain relations) over a
# *gridded* radar volume — the beam-power-weighted averages produced by the
# accumulator. This is the intended workflow: grid the polarimetric variables
# (DBZ, ZDR, KDP, RHOHV) so each is a smooth weighted average, then classify and
# estimate rain rate on those gridded values rather than gridding integer HID
# categories (which only nearest-neighbor regridding could handle).
#
# Two entry points:
#   * `apply_echo_products` — pure, in-memory: takes a field dict and returns a
#     dict of new fields. Used by the gridding drivers (post-finalize) and by the
#     standalone path.
#   * `add_echo_products!` — standalone: reads a previously written Daisho
#     gridded NetCDF (volume / PPI / RHI, including multi-time concatenated files),
#     computes the products, and appends them as new variables in place.
#
# Sentinel policy (preserving the true-missing vs undetect distinction):
#   * true-missing input (`io.fill_value`)  → output `fill_value` (unknown)
#   * undetect / clear-air input (`io.undetect`) → rain 0, HID `undetect`
#   * valid input but a required field is invalid → `fill_value`

# Output field name → component rain function (closure over band coefficients).
const _RAIN_COMPONENT_BUILDERS = (
    "RATE_Z",       # R(Z) all
    "RATE_Z_CONV",  # R(Z) convective
    "RATE_Z_STRAT", # R(Z) stratiform
    "RATE_KDP",     # R(Kdp)
    "RATE_Z_ZDR",   # R(Z, Zdr)
    "RATE_KDP_ZDR", # R(Kdp, Zdr)
)

@inline _dp_missing(d, io) = (d isa AbstractFloat && isnan(d)) || d == Float32(io.fill_value)
@inline _dp_undetect(d, io) = d == Float32(io.undetect)
@inline _dp_invalid(x, io) = x === nothing || (x isa AbstractFloat && isnan(x)) ||
    x == Float32(io.fill_value) || x == Float32(io.undetect)

# Map a raw rain-rate field to Float32 with the sentinel policy applied. `dbz`
# drives the missing/clear-air decision; `required` are the additional inputs that
# must be valid for the estimate to be meaningful.
function _finalize_rain(raw::AbstractArray, dbz, required::Tuple, io::IOParameters)
    out = Array{Float32}(undef, size(raw))
    fv = Float32(io.fill_value)
    ud = Float32(io.undetect)
    @inbounds for i in eachindex(raw)
        d = dbz[i]
        if _dp_missing(d, io)
            out[i] = fv
        elseif _dp_undetect(d, io)
            # Clear air (scanned, no echo): preserve the undetect sentinel rather
            # than writing 0, so the missing/undetect/value distinction carries
            # through to the rain fields (downstream masking relies on it).
            out[i] = ud
        elseif !isfinite(raw[i]) || any(r -> _dp_invalid(r[i], io), required)
            out[i] = fv
        else
            out[i] = Float32(raw[i])
        end
    end
    return out
end

_at_field(::Nothing, i) = nothing
_at_field(v::AbstractArray, i) = v[i]

"""
    echo_output_names(dp::EchoProductsParameters) -> Vector{String}

The names of the netCDF variables this echo configuration writes: the HID field,
the blended rain field, the optional rain-method field, and any individual rain
components. Echo products are appended *after* gridding and are therefore not part
of `[fields]`; this list lets the gridded readers surface them anyway.
"""
function echo_output_names(dp::EchoProductsParameters)
    names = String[]
    dp.compute_fhc && push!(names, dp.fhc_output)
    dp.compute_blended_rain && push!(names, dp.rain_output)
    isempty(dp.rain_method_output) || push!(names, dp.rain_method_output)
    append!(names, dp.rain_components)
    return names
end

# Add any configured echo output variables that exist in `file` (and are not
# already present) to a reader's `fields` dict, reshaped to `shape`. Echo products
# are written post-grid and need not appear in `[fields]`, so the Fields-API
# readers would otherwise miss them. Read raw (`.var`) to preserve sentinels.
function _merge_echo_fields!(fields::AbstractDict, file::AbstractString,
        p::DaishoParameters, shape::Tuple)
    :echo in p.provided || return fields
    names = echo_output_names(p.echo)
    isempty(names) && return fields
    NCDataset(file) do ds
        for name in names
            (haskey(ds, name) && !haskey(fields, name)) || continue
            fields[name] = reshape(Float32.(Array(ds[name].var)), shape...)
        end
    end
    return fields
end

"""
    _echo_temperature_field(dp, fields, heights, io) -> Union{Array,Nothing}

Produce the per-cell temperature array (°C, shaped like the reflectivity grid) for
the FHC, or `nothing` when temperature is not used. Dispatches on `dp.temp_source`:

  * `:profile`  — sample the `dp.temperature` T(z) profile at each cell's height
    (needs `heights`). `heights` may be the grid z-axis (3-D) or a gridded beam-
    height field (`dp.height_field`, enabling 2-D PPI/RHI); sentinel/non-finite
    heights yield `NaN` so the FHC skips the temperature term there.
  * `:field`    — use the gridded `dp.temp_field` directly (it is already per-cell
    and may be 3-D), converting K→°C when `dp.temp_field_units == "K"`. Sentinel /
    non-finite cells become `NaN` so `csu_fhc_summer` skips the temperature term
    there (the conversion is applied only to valid cells).
  * `:reference_state` — reserved for a future Springsteel reference state.
"""
function _echo_temperature_field(dp::EchoProductsParameters, fields::AbstractDict,
        heights, io::IOParameters)
    dp.use_temp || return nothing

    if dp.temp_source === :profile
        (dp.temperature === nothing || heights === nothing) && return nothing
        # heights may be a gridded beam-height field carrying sentinels; sample the
        # profile only at valid heights and emit NaN elsewhere.
        return map(h -> _dp_invalid(h, io) ? NaN : temperature_celsius(dp.temperature, h),
                   heights)

    elseif dp.temp_source === :field
        haskey(fields, dp.temp_field) || throw(ArgumentError(
            "apply_echo_products: temp_source=:field but temperature field " *
            "\"$(dp.temp_field)\" not found in the grid " *
            "(have: $(join(sort(collect(keys(fields))), ", ")))."))
        raw = fields[dp.temp_field]
        to_celsius = dp.temp_field_units == "K"
        T = Array{Float64}(undef, size(raw))
        @inbounds for i in eachindex(raw)
            v = raw[i]
            if _dp_invalid(v, io)
                T[i] = NaN
            else
                T[i] = to_celsius ? Float64(v) - 273.15 : Float64(v)
            end
        end
        return T

    else  # :reference_state (validated at config load; reserved)
        throw(ArgumentError("apply_echo_products: temp_source=:reference_state " *
            "is not yet implemented; use :profile or :field."))
    end
end

"""
    apply_echo_products(fields, dp::EchoProductsParameters;
                           io::IOParameters, heights=nothing) -> Dict{String,Array{Float32}}

Compute the configured echo products from a dict of gridded `fields` (keyed by
name, e.g. the `fields` of [`read_gridded_radar`](@ref)). Returns a dict of new
fields keyed by their output names. `heights` is an array the same shape as the
input fields giving each cell's height in meters; when supplied with a
temperature profile it enables the temperature term of the FHC.

The hydrometeor classification, when computed, is fed into the blended rain rate
for ice/hail masking.
"""
function apply_echo_products(fields::AbstractDict, dp::EchoProductsParameters;
        io::IOParameters, heights = nothing)

    haskey(fields, dp.dbz_field) || throw(ArgumentError(
        "apply_echo_products: reflectivity field \"$(dp.dbz_field)\" not found " *
        "in the grid (have: $(join(sort(collect(keys(fields))), ", ")))."))
    dbz = fields[dp.dbz_field]
    zdr = get(fields, dp.zdr_field, nothing)
    kdp = get(fields, dp.kdp_field, nothing)
    rho = get(fields, dp.rhohv_field, nothing)
    ldr = isempty(dp.ldr_field) ? nothing : get(fields, dp.ldr_field, nothing)

    fv = io.fill_value
    ud = io.undetect

    # Prefer a gridded beam-height field for the profile source when configured and
    # present (enables temperature on 2-D grids); otherwise use the grid z-axis
    # heights passed in.
    heights_use = (!isempty(dp.height_field) && haskey(fields, dp.height_field)) ?
        fields[dp.height_field] : heights

    # Resolve the per-cell temperature array from the configured source.
    T_arr = _echo_temperature_field(dp, fields, heights_use, io)
    use_temp = T_arr !== nothing

    out = Dict{String,Array{Float32}}()

    # Hydrometeor classification (kept as Int for feeding into the rain step).
    hid_int = nothing
    if dp.compute_fhc
        hid_int = csu_fhc_summer(; dz = dbz, zdr = zdr, kdp = kdp, rho = rho,
            ldr = ldr, T = T_arr, weights = dp.weights, method = dp.fhc_method,
            band = dp.band, use_temp = use_temp, temp_factor = dp.temp_factor,
            fill_value = fv, undetect = ud, masked = 0)

        hidf = Array{Float32}(undef, size(dbz))
        @inbounds for i in eachindex(dbz)
            d = dbz[i]
            hidf[i] = _dp_missing(d, io) ? Float32(fv) :
                      _dp_undetect(d, io) ? Float32(ud) : Float32(hid_int[i])
        end
        out[dp.fhc_output] = hidf
    end

    # Blended rain rate (consumes the HID for ice/hail masking when available).
    if dp.compute_blended_rain
        rain, meth = calc_blended_rain_tropical(; dz = dbz, zdr = zdr, kdp = kdp,
            fhc = hid_int, band = dp.band,
            correct_ice_method = dp.correct_ice_method, fill_value = fv, undetect = ud)
        out[dp.rain_output] = _finalize_rain(rain, dbz, (), io)
        if !isempty(dp.rain_method_output)
            methf = Array{Float32}(undef, size(meth))
            @inbounds for i in eachindex(meth)
                d = dbz[i]
                methf[i] = _dp_missing(d, io) ? Float32(fv) : Float32(meth[i])
            end
            out[dp.rain_method_output] = methf
        end
    end

    # Individual rain components (available regardless of the blended switch).
    for name in dp.rain_components
        if name == "RATE_Z"
            raw = calc_rain_zr(dbz; a = RAIN_RZ_ALL.a, b = RAIN_RZ_ALL.b)
            out[name] = _finalize_rain(raw, dbz, (), io)
        elseif name == "RATE_Z_CONV"
            raw = calc_rain_zr(dbz; a = RAIN_RZ_CONV.a, b = RAIN_RZ_CONV.b)
            out[name] = _finalize_rain(raw, dbz, (), io)
        elseif name == "RATE_Z_STRAT"
            raw = calc_rain_zr(dbz; a = RAIN_RZ_STRAT.a, b = RAIN_RZ_STRAT.b)
            out[name] = _finalize_rain(raw, dbz, (), io)
        elseif name == "RATE_KDP"
            kdp === nothing && throw(ArgumentError(
                "RATE_KDP requested but Kdp field \"$(dp.kdp_field)\" is absent."))
            raw = calc_rain_kdp(kdp, dp.band)
            out[name] = _finalize_rain(raw, dbz, (kdp,), io)
        elseif name == "RATE_Z_ZDR"
            zdr === nothing && throw(ArgumentError(
                "RATE_Z_ZDR requested but Zdr field \"$(dp.zdr_field)\" is absent."))
            raw = calc_rain_z_zdr(dbz, zdr, dp.band)
            out[name] = _finalize_rain(raw, dbz, (zdr,), io)
        elseif name == "RATE_KDP_ZDR"
            (kdp === nothing || zdr === nothing) && throw(ArgumentError(
                "RATE_KDP_ZDR requested but Kdp or Zdr field is absent."))
            raw = calc_rain_kdp_zdr(kdp, zdr, dp.band)
            out[name] = _finalize_rain(raw, dbz, (kdp, zdr), io)
        else
            throw(ArgumentError("Unknown rain component \"$name\"."))
        end
    end

    return out
end

# Build a per-cell height array of shape `shp` whose values vary along dimension
# `zdim` according to `z_axis`.
function _heights_array(shp::Tuple, z_axis::AbstractVector, zdim::Int)
    h = Array{Float64}(undef, shp)
    for idx in CartesianIndices(h)
        h[idx] = z_axis[idx[zdim]]
    end
    return h
end

"""
    add_echo_products!(file, p::DaishoParameters) -> Vector{String}

Compute the echo products (per `p.echo`) for an already-written Daisho
gridded NetCDF `file` and append them as new variables in place. Supports 3-D
volume (`X,Y,Z`), 2-D PPI/composite (`X,Y`) and 2-D RHI (`R,Z`) layouts, and
loops over the `time` dimension for concatenated multi-time files. Returns the
names of the variables written.

The standalone counterpart to the in-grid hook: useful for reprocessing archived
grids or applying updated coefficients without regridding.
"""
function add_echo_products!(file::AbstractString, p::DaishoParameters)
    dp = p.echo
    io = p.io
    ds = NCDataset(file, "a")
    written = String[]
    try
        dimnames = keys(ds.dim)
        has(name) = name in dimnames

        # Resolve geometry, the spatial dim names, and the height axis (if any).
        if has("X") && has("Y") && has("Z")
            spatial = ("X", "Y", "Z")
            z_axis = Float64.(ds["Z"][:])
            zdim_local = 3
        elseif has("R") && has("Z")
            spatial = ("R", "Z")
            z_axis = Float64.(ds["Z"][:])
            zdim_local = 2
        elseif has("X") && has("Y")
            spatial = ("X", "Y")
            z_axis = nothing
            zdim_local = 0
        else
            throw(ArgumentError("add_echo_products!: unrecognized grid layout " *
                "(dims: $(join(collect(dimnames), ", "))); expected X/Y/Z, R/Z, or X/Y."))
        end
        ntime = has("time") ? ds.dim["time"] : 1
        spatial_dims = Tuple(ds.dim[d] for d in spatial)

        # Read the inputs once (full arrays, including the time axis). Use `.var`
        # to get the raw, shape-preserving values: plain `ds[name][:]` flattens to
        # 1-D and converts `_FillValue` to `missing`, but the sentinels must stay
        # as numeric values for the sentinel policy.
        readvar(name) = haskey(ds, name) ? Array(ds[name].var) : nothing
        dbz_all = readvar(dp.dbz_field)
        dbz_all === nothing && throw(ArgumentError(
            "add_echo_products!: reflectivity variable \"$(dp.dbz_field)\" " *
            "not present in $file."))
        zdr_all = readvar(dp.zdr_field)
        kdp_all = readvar(dp.kdp_field)
        rho_all = readvar(dp.rhohv_field)
        ldr_all = isempty(dp.ldr_field) ? nothing : readvar(dp.ldr_field)
        # Gridded temperature field, when that source is selected. Fail early with
        # a clear message if it is absent (no silent fallback).
        temp_all = nothing
        if dp.use_temp && dp.temp_source === :field
            temp_all = readvar(dp.temp_field)
            temp_all === nothing && throw(ArgumentError(
                "add_echo_products!: temp_source=:field but temperature variable " *
                "\"$(dp.temp_field)\" not present in $file."))
        end
        # Gridded beam-height field for the profile source on 2-D grids. Lenient:
        # if absent, fall back to the z-axis heights (or no temperature on 2-D).
        height_all = isempty(dp.height_field) ? nothing : readvar(dp.height_field)

        # Per-time-slice indexer into a (spatial..., time) array.
        slice(::Nothing, t) = nothing
        function slice(a::AbstractArray, t)
            if has("time") && ndims(a) == length(spatial) + 1
                return a[ntuple(_ -> Colon(), length(spatial))..., t]
            end
            return a
        end

        heights = (z_axis === nothing) ? nothing :
            _heights_array(spatial_dims, z_axis, zdim_local)

        # Allocate output arrays (spatial..., time) lazily once we know names.
        outbufs = Dict{String,Array{Float32}}()

        for t in 1:ntime
            fields = Dict{String,Any}()
            fields[dp.dbz_field] = slice(dbz_all, t)
            zdr_all !== nothing && (fields[dp.zdr_field] = slice(zdr_all, t))
            kdp_all !== nothing && (fields[dp.kdp_field] = slice(kdp_all, t))
            rho_all !== nothing && (fields[dp.rhohv_field] = slice(rho_all, t))
            ldr_all !== nothing && (fields[dp.ldr_field] = slice(ldr_all, t))
            temp_all !== nothing && (fields[dp.temp_field] = slice(temp_all, t))
            height_all !== nothing && (fields[dp.height_field] = slice(height_all, t))

            prods = apply_echo_products(fields, dp; io = io, heights = heights)
            for (name, arr) in prods
                buf = get!(outbufs, name) do
                    Array{Float32}(undef, spatial_dims..., ntime)
                end
                buf[ntuple(_ -> Colon(), length(spatial))..., t] = arr
            end
        end

        # Define (or locate) and write each output variable.
        vardims = has("time") ? (spatial..., "time") : spatial
        for (name, buf) in outbufs
            attrib = haskey(variable_attrib_dict, name) ?
                merge(common_attrib, variable_attrib_dict[name]) :
                merge(common_attrib, variable_attrib_dict["UNKNOWN"])
            attrib = _with_io_sentinels(attrib, io.fill_value, io.undetect)
            if haskey(ds, name)
                ds[name][:] = buf
            else
                v = defVar(ds, name, Float32, vardims, attrib = attrib)
                v[:] = buf
            end
            push!(written, name)
        end
    finally
        close(ds)
    end
    return written
end
