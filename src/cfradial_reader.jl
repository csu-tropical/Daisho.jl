# CfRadial reader. `read_cfradial(file)` auto-detects v1.4 vs v2.1 and returns
# a `Volume`. The internal v1 path slices flat top-level arrays into per-sweep
# `SweepGroup`s; the v2 path walks the NetCDF4 sub-group tree.
#
# Real-world LROSE divergences from spec (CFRADIAL_REFACTOR_PLAN.md §3.2.6) are
# normalized into spec shape on read. Anything outside the spec table goes to
# `extra_attrs` / `extra_vars`.

using NCDatasets
using DataStructures
using Dates

# Spec attribute keys handled explicitly by the reader (excluded from extra_attrs).
const _FIELD_SPEC_ATTRS = Set([
    "standard_name", "long_name", "units",
    "_FillValue", "_Undetect",
    "scale_factor", "add_offset",
    "coordinates", "sampling_ratio",
    "is_discrete", "field_folds",
    "fold_limit_lower", "fold_limit_upper",
    "is_quality_field", "qualified_variables", "ancillary_variables",
    "flag_values", "flag_meanings", "flag_masks",
    "thresholding_xml", "legend_xml",
])

const _CALIB_FIELD_NAMES = (
    :time, :pulse_width,
    :antenna_gain_h, :antenna_gain_v,
    :xmit_power_h, :xmit_power_v,
    :two_way_waveguide_loss_h, :two_way_waveguide_loss_v,
    :two_way_radome_loss_h, :two_way_radome_loss_v,
    :receiver_mismatch_loss, :receiver_mismatch_loss_h, :receiver_mismatch_loss_v,
    :radar_constant_h, :radar_constant_v,
    :probert_jones_correction, :dielectric_factor_used,
    :noise_hc, :noise_vc, :noise_hx, :noise_vx,
    :receiver_gain_hc, :receiver_gain_vc, :receiver_gain_hx, :receiver_gain_vx,
    :base_1km_hc, :base_1km_vc, :base_1km_hx, :base_1km_vx,
    :sun_power_hc, :sun_power_vc, :sun_power_hx, :sun_power_vx,
    :noise_source_power_h, :noise_source_power_v,
    :power_measure_loss_h, :power_measure_loss_v,
    :coupler_forward_loss_h, :coupler_forward_loss_v,
    :zdr_correction, :ldr_correction_h, :ldr_correction_v,
    :system_phidp,
    :test_power_h, :test_power_v,
    :receiver_slope_hc, :receiver_slope_vc, :receiver_slope_hx, :receiver_slope_vx,
)

const _GEOREF_FIELD_NAMES = (
    :latitude, :longitude, :altitude,
    :heading, :roll, :pitch, :drift, :rotation, :tilt,
    :eastward_velocity, :northward_velocity, :vertical_velocity,
    :eastward_wind, :northward_wind, :vertical_wind,
    :heading_rate, :roll_rate, :pitch_rate,
    :georefs_applied,
)

# ── Public entry ─────────────────────────────────────────────────────────────

"""
    read_cfradial(file::AbstractString) -> Volume

Read a CfRadial 1.4 or 2.1 NetCDF file and return a `Volume`. Format is
auto-detected from the presence of a root-level `sweep_group_name` variable
(v2) vs flat top-level field arrays (v1).

The reader is lenient about optional fields and absorbs non-spec attributes
into `extra_attrs` / `extra_vars`. Required spec fields raise `ArgumentError`
naming the file and missing field.
"""
function read_cfradial(file::AbstractString)
    isfile(file) || throw(ArgumentError("CfRadial file not found: $file"))
    NCDataset(file, "r") do ds
        if _is_cfradial2(ds)
            return _read_cfradial2(ds, file)
        else
            return _read_cfradial1(ds, file)
        end
    end
end

# ── Format detection ─────────────────────────────────────────────────────────

function _is_cfradial2(ds)
    haskey(ds, "sweep_group_name") && return true
    if haskey(ds.attrib, "Conventions")
        conv = String(ds.attrib["Conventions"])
        if occursin("Cf/Radial-2", conv) || occursin("CfRadial-2", conv)
            return true
        end
    end
    return false
end

# ── Helpers ──────────────────────────────────────────────────────────────────

# Read attribute as String, stripping NUL padding from char-array attrs.
_attr_str(ds, key, default="") = haskey(ds.attrib, key) ? _to_string(ds.attrib[key]) : default

_to_string(x::AbstractString) = String(rstrip(String(x), '\0'))
_to_string(x::AbstractArray{<:AbstractChar}) = String(rstrip(String(collect(x)), '\0'))
_to_string(x::AbstractArray{UInt8}) = String(rstrip(String(x), '\0'))
_to_string(x) = String(string(x))

# Coerce attribute value to Bool. CfRadial commonly uses "true"/"false" strings.
function _attr_bool(ds, key, default::Bool=false)
    haskey(ds.attrib, key) || return default
    v = ds.attrib[key]
    v isa Bool && return v
    s = lowercase(_to_string(v))
    return s == "true" || s == "1" || s == "yes"
end

function _attr_int(ds, key, default::Int=0)
    haskey(ds.attrib, key) || return default
    v = ds.attrib[key]
    v isa Integer && return Int(v)
    v isa Real && return Int(v)
    return parse(Int, _to_string(v))
end

# Read a NetCDF variable, stripping CF char-string padding when applicable.
function _readvar(ds, name)
    haskey(ds, name) || return nothing
    v = ds[name]
    return v[:]
end

# Read a scalar variable as a possibly-typed value. Returns nothing if missing.
# Handles 0-dim numeric scalars and 1-dim CF char-array strings (the v1 way of
# representing scalar strings).
function _readscalar(ds, name)
    haskey(ds, name) || return nothing
    var = ds[name]
    nd = ndims(var)
    if nd == 0
        return var[]
    elseif nd == 1
        et = eltype(var)
        if et <: AbstractChar || et <: UInt8
            return _to_string(var[:])
        end
        # 1-D non-string scalar: read first non-missing element.
        arr = var[:]
        if length(arr) == 1
            return arr[1]
        end
        return nothing
    end
    return nothing
end

# Strip char-array variables of a `string_length_*` dim into a Vector{String}.
function _readstring_per_sweep(ds, name, n_sweeps::Int)
    haskey(ds, name) || return nothing
    v = ds[name]
    raw = v[:]
    if raw isa AbstractMatrix
        return [_to_string(raw[:, i]) for i in 1:size(raw, 2)]
    elseif raw isa AbstractVector{<:AbstractString}
        return [_to_string(raw[i]) for i in 1:length(raw)]
    end
    return nothing
end

# Coerce ray array value to Float64 vector, treating missing/_FillValue as NaN.
function _to_f64vec(x)
    x === nothing && return nothing
    out = Vector{Float64}(undef, length(x))
    @inbounds for i in eachindex(x)
        v = x[i]
        out[i] = ismissing(v) ? NaN : Float64(v)
    end
    return out
end

function _to_int_vec(x)
    x === nothing && return nothing
    out = Vector{Int}(undef, length(x))
    @inbounds for i in eachindex(x)
        v = x[i]
        out[i] = ismissing(v) ? 0 : Int(v)
    end
    return out
end

# Coerce a possibly-missing scalar to Float64 / Int / Bool, returning a default
# (or nothing) if the value is missing. These wrappers exist because real-world
# CfRadial files routinely encode "no data" as `missing`, including for fields
# we'd otherwise want to read as plain scalars.
_f64_or(x, default) = (x === nothing || ismissing(x)) ? default : Float64(x)
_f64_or_nothing(x) = (x === nothing || ismissing(x)) ? nothing : Float64(x)
_int_or(x, default::Int=0) = (x === nothing || ismissing(x)) ? default : Int(x)

function _to_bool_vec(x)
    x === nothing && return nothing
    out = Vector{Bool}(undef, length(x))
    @inbounds for i in eachindex(x)
        v = x[i]
        out[i] = ismissing(v) ? false : !iszero(v)
    end
    return out
end

# Resolve the time-coverage start/end either from a variable (preferred) or attr.
function _read_time_coverage(ds)
    tcs = _readscalar(ds, "time_coverage_start")
    tce = _readscalar(ds, "time_coverage_end")
    if tcs === nothing && haskey(ds.attrib, "time_coverage_start")
        tcs = _to_string(ds.attrib["time_coverage_start"])
    end
    if tce === nothing && haskey(ds.attrib, "time_coverage_end")
        tce = _to_string(ds.attrib["time_coverage_end"])
    end
    return tcs, tce
end

# Parse ISO8601 (with optional Z, fractional seconds) to DateTime.
function _parse_iso8601(s)
    s isa DateTime && return s
    s = String(s)
    s = replace(s, r"Z$" => "")
    s = replace(s, " " => "T")
    # Trim fractional seconds if present.
    m = match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?", s)
    if m !== nothing
        return DateTime(m.captures[1])
    end
    return DateTime(s)
end

function _parse_iso8601_or_nothing(s)
    s === nothing && return nothing
    try
        return _parse_iso8601(s)
    catch
        return nothing
    end
end

# Apply scale_factor/add_offset and return Float32 matrix. Preserve fill-value
# positions as fill (not NaN) so that downstream code can distinguish missing
# vs clear-air via metadata.fill_value/undetect.
function _unpack_field_data(raw, scale_factor, add_offset, fill_value)
    if scale_factor === nothing && add_offset === nothing
        # Convert Missing to NaN, return Float32.
        out = Array{Float32}(undef, size(raw))
        @inbounds for i in eachindex(raw)
            v = raw[i]
            out[i] = ismissing(v) ? Float32(NaN) : Float32(v)
        end
        return out
    end
    sc = Float64(scale_factor === nothing ? 1.0 : scale_factor)
    of = Float64(add_offset === nothing ? 0.0 : add_offset)
    out = Array{Float32}(undef, size(raw))
    @inbounds for i in eachindex(raw)
        v = raw[i]
        if ismissing(v)
            out[i] = Float32(NaN)
        elseif fill_value !== nothing && v == fill_value
            out[i] = Float32(fill_value)
        else
            out[i] = Float32(Float64(v) * sc + of)
        end
    end
    return out
end

# Build FieldMetadata from a NetCDF variable's attributes.
function _read_field_metadata(var)
    a = var.attrib
    extra = Dict{String,Any}()
    standard_name = haskey(a, "standard_name") ? _to_string(a["standard_name"]) : nothing
    long_name = haskey(a, "long_name") ? _to_string(a["long_name"]) : nothing
    units = haskey(a, "units") ? _to_string(a["units"]) : nothing
    fill_value = haskey(a, "_FillValue") ? a["_FillValue"] : nothing
    undetect = haskey(a, "_Undetect") ? a["_Undetect"] : nothing
    scale_factor = haskey(a, "scale_factor") ? Float64(a["scale_factor"]) : nothing
    add_offset = haskey(a, "add_offset") ? Float64(a["add_offset"]) : nothing
    coordinates = haskey(a, "coordinates") ? _to_string(a["coordinates"]) : nothing
    sampling_ratio = haskey(a, "sampling_ratio") ? Float64(a["sampling_ratio"]) : 1.0
    is_discrete = haskey(a, "is_discrete") ? Bool(a["is_discrete"]) : false
    field_folds = haskey(a, "field_folds") ? Bool(a["field_folds"]) : false
    fold_lo = haskey(a, "fold_limit_lower") ? Float64(a["fold_limit_lower"]) : nothing
    fold_hi = haskey(a, "fold_limit_upper") ? Float64(a["fold_limit_upper"]) : nothing
    is_quality_field = haskey(a, "is_quality_field") ? Bool(a["is_quality_field"]) : false
    qualified = haskey(a, "qualified_variables") ?
        _split_var_list(_to_string(a["qualified_variables"])) : String[]
    ancillary = haskey(a, "ancillary_variables") ?
        _split_var_list(_to_string(a["ancillary_variables"])) : String[]
    flag_values = haskey(a, "flag_values") ? collect(a["flag_values"]) : nothing
    flag_meanings = haskey(a, "flag_meanings") ?
        _split_var_list(_to_string(a["flag_meanings"])) : nothing
    flag_masks = haskey(a, "flag_masks") ? collect(a["flag_masks"]) : nothing
    thresholding_xml = haskey(a, "thresholding_xml") ? _to_string(a["thresholding_xml"]) : nothing
    legend_xml = haskey(a, "legend_xml") ? _to_string(a["legend_xml"]) : nothing

    for (k, v) in a
        if !(k in _FIELD_SPEC_ATTRS)
            extra[String(k)] = v
        end
    end

    return FieldMetadata(
        standard_name = standard_name,
        long_name = long_name,
        units = units,
        fill_value = fill_value,
        undetect = undetect,
        scale_factor = scale_factor,
        add_offset = add_offset,
        coordinates = coordinates,
        sampling_ratio = sampling_ratio,
        is_discrete = is_discrete,
        field_folds = field_folds,
        fold_limit_lower = fold_lo,
        fold_limit_upper = fold_hi,
        is_quality_field = is_quality_field,
        qualified_variables = qualified,
        ancillary_variables = ancillary,
        flag_values = flag_values,
        flag_meanings = flag_meanings,
        flag_masks = flag_masks,
        thresholding_xml = thresholding_xml,
        legend_xml = legend_xml,
        extra_attrs = extra,
    )
end

_split_var_list(s::AbstractString) = filter(!isempty, split(s, r"[\s,]+"))

# Reshape a v1 field var into (n_rays, n_gates) by inspecting its dim names.
function _normalize_field_layout(var, raw, n_rays, n_gates)
    dnames = NCDatasets.dimnames(var)
    sz = size(raw)
    # Already (time, range)?
    if length(dnames) == 2
        if String(dnames[1]) == "time" && String(dnames[2]) == "range"
            return raw
        elseif String(dnames[1]) == "range" && String(dnames[2]) == "time"
            return permutedims(raw)
        end
    end
    if sz == (n_rays, n_gates)
        return raw
    elseif sz == (n_gates, n_rays)
        return permutedims(raw)
    end
    return raw
end

# Read calibration entries from a calibration group/dataset.
function _read_calibration(grp)
    haskey(grp, "calib") || haskey(grp, "r_calib") || haskey(grp, "pulse_width") ||
        haskey(grp, "r_calib_pulse_width") ||
        return nothing
    n = if haskey(grp.dim, "calib")
            grp.dim["calib"]
        elseif haskey(grp.dim, "r_calib")
            grp.dim["r_calib"]
        else
            1
        end
    entries = RadarCalibrationEntry[]
    for i in 1:n
        kw = Dict{Symbol,Any}()
        extra = Dict{String,Any}()
        for varname in keys(grp)
            stripped, mapped = _calib_canonical_name(varname)
            data = _readvar(grp, varname)
            data === nothing && continue
            value = ndims(data) == 0 || length(data) == 1 ? first(data) : data[i]
            if mapped !== nothing
                if mapped === :time
                    kw[mapped] = _parse_iso8601_or_nothing(_to_string(value))
                else
                    kw[mapped] = ismissing(value) ? nothing : Float64(value)
                end
            else
                extra[String(varname)] = value
            end
        end
        kw[:extra_vars] = extra
        push!(entries, RadarCalibrationEntry(; kw...))
    end
    return entries
end

# Map LROSE/SIGMET-style calibration var names to canonical RadarCalibrationEntry
# field names. Returns (stripped, mapped_field). If mapped_field is nothing, the
# variable should pass through as extra_vars.
function _calib_canonical_name(varname)
    s = String(varname)
    stripped = startswith(s, "r_calib_") ? s[9:end] : s
    # LROSE base_dbz_1km_* → spec base_1km_*
    canon = if stripped == "calibration_time" || stripped == "time"
        "time"
    elseif occursin(r"^base_dbz_1km_(hc|vc|hx|vx)$", stripped)
        replace(stripped, "base_dbz_1km_" => "base_1km_")
    else
        stripped
    end
    sym = Symbol(canon)
    if sym in _CALIB_FIELD_NAMES
        return stripped, sym
    end
    return stripped, nothing
end

# Read a Georeference group or per-ray vars from a flat root, scoped to indices.
function _read_georeference(grp, idx::Union{Nothing,AbstractRange}=nothing)
    # Determine whether lat/lon/alt vary per ray (georeference present) or are
    # scalar (stationary).
    haskey(grp, "latitude") || return nothing
    lat_raw = grp["latitude"][:]
    if idx !== nothing
        lat_raw = lat_raw[idx]
    end
    lat_v = _to_f64vec(lat_raw)
    lat_v === nothing && return nothing
    if length(lat_v) <= 1
        return nothing
    end
    kw = Dict{Symbol,Any}()
    extra = Dict{String,Any}()
    for fld in _GEOREF_FIELD_NAMES
        name = String(fld)
        haskey(grp, name) || continue
        raw = grp[name][:]
        if idx !== nothing && ndims(raw) == 1 && length(raw) >= last(idx)
            raw = raw[idx]
        end
        if fld === :georefs_applied
            kw[fld] = _to_bool_vec(raw)
        else
            v = _to_f64vec(raw)
            v === nothing && continue
            if length(v) == length(lat_v)
                kw[fld] = v
            end
        end
    end
    kw[:extra_vars] = extra
    return Georeference(; kw...)
end

# ── CfRadial v2 reader ───────────────────────────────────────────────────────

function _read_cfradial2(ds, file)
    # Globals
    conventions = _attr_str(ds, "Conventions", "Cf/Radial")
    version = _attr_str(ds, "version", "2.1")
    title = _attr_str(ds, "title")
    institution = _attr_str(ds, "institution")
    source = _attr_str(ds, "source")
    history = _attr_str(ds, "history")
    references = _attr_str(ds, "references")
    comment = _attr_str(ds, "comment")
    instrument_name = _attr_str(ds, "instrument_name")
    site_name = _attr_str(ds, "site_name")
    scan_name = _attr_str(ds, "scan_name")
    scan_id = _attr_int(ds, "scan_id", 0)
    platform_is_mobile = _attr_bool(ds, "platform_is_mobile", false)
    ray_times_increase = _attr_bool(ds, "ray_times_increase", true)
    simulated_data = _attr_bool(ds, "simulated_data", false)

    extra_attrs = Dict{String,Any}()
    handled_attrs = Set([
        "Conventions", "version", "title", "institution", "source", "history",
        "references", "comment", "instrument_name", "site_name",
        "scan_name", "scan_id",
        "platform_is_mobile", "ray_times_increase", "simulated_data",
    ])
    for (k, v) in ds.attrib
        if !(k in handled_attrs)
            extra_attrs[String(k)] = v
        end
    end

    # Required globals: latitude, longitude, altitude, time_coverage_start/end
    lat = _readscalar(ds, "latitude")
    lon = _readscalar(ds, "longitude")
    alt = _readscalar(ds, "altitude")
    lat === nothing && throw(ArgumentError("$file: required variable `latitude` is missing"))
    lon === nothing && throw(ArgumentError("$file: required variable `longitude` is missing"))
    alt === nothing && throw(ArgumentError("$file: required variable `altitude` is missing"))
    altitude_agl = _readscalar(ds, "altitude_agl")
    altitude_agl_v = _f64_or_nothing(altitude_agl)

    tcs, tce = _read_time_coverage(ds)
    tcs === nothing && throw(ArgumentError("$file: required `time_coverage_start` is missing"))
    tce === nothing && throw(ArgumentError("$file: required `time_coverage_end` is missing"))
    time_start = _parse_iso8601(tcs)
    time_end = _parse_iso8601(tce)

    volume_number = _readscalar(ds, "volume_number")
    volume_number_v = volume_number === nothing ? 0 : Int(volume_number)
    platform_type = _readscalar(ds, "platform_type")
    platform_type_v = platform_type === nothing ? "fixed" : _to_string(platform_type)
    instrument_type = _readscalar(ds, "instrument_type")
    instrument_type_v = instrument_type === nothing ? "radar" : _to_string(instrument_type)
    primary_axis = _readscalar(ds, "primary_axis")
    primary_axis_v = primary_axis === nothing ? "axis_z" : _to_string(primary_axis)

    status_str = _readscalar(ds, "status_str")
    if status_str === nothing
        status_str = _readscalar(ds, "status_xml")
    end
    status_str_v = status_str === nothing ? nothing : _to_string(status_str)

    # Sweep group names + fixed angles
    sweep_group_name = _readvar(ds, "sweep_group_name")
    sweep_group_name === nothing &&
        throw(ArgumentError("$file: required `sweep_group_name` variable missing in v2 file"))
    sweep_fixed_angle = _readvar(ds, "sweep_fixed_angle")

    sweeps = SweepGroup[]
    for (i, gname) in enumerate(sweep_group_name)
        gname_str = _to_string(gname)
        haskey(ds.group, gname_str) ||
            throw(ArgumentError("$file: sweep group `$(gname_str)` listed in sweep_group_name not found"))
        grp = ds.group[gname_str]
        fixed = sweep_fixed_angle === nothing ? NaN :
                _f64_or(sweep_fixed_angle[i], NaN)
        push!(sweeps, _read_sweep_v2(grp, time_start, fixed, i - 1))
    end

    # Optional root sub-groups
    radar_parameters = haskey(ds.group, "radar_parameters") ?
        _read_radar_parameters(ds.group["radar_parameters"]) : nothing
    radar_calibration = nothing
    if haskey(ds.group, "radar_calibration")
        cal_entries = _read_calibration(ds.group["radar_calibration"])
        cal_idx_v = nothing
        if cal_entries !== nothing
            radar_calibration = RadarCalibration(entries = cal_entries, calib_index = cal_idx_v)
        end
    end
    georeference_correction = haskey(ds.group, "georeference_correction") ?
        _read_georeference_correction(ds.group["georeference_correction"]) : nothing

    return Volume(
        conventions = conventions,
        version = version,
        title = title,
        institution = institution,
        source = source,
        history = history,
        references = references,
        comment = comment,
        instrument_name = instrument_name,
        site_name = site_name,
        scan_name = scan_name,
        scan_id = scan_id,
        platform_is_mobile = platform_is_mobile,
        ray_times_increase = ray_times_increase,
        simulated_data = simulated_data,
        volume_number = volume_number_v,
        platform_type = platform_type_v,
        instrument_type = instrument_type_v,
        primary_axis = primary_axis_v,
        time_coverage_start = time_start,
        time_coverage_end = time_end,
        latitude = _f64_or(lat, 0.0),
        longitude = _f64_or(lon, 0.0),
        altitude = _f64_or(alt, 0.0),
        altitude_agl = altitude_agl_v,
        sweeps = sweeps,
        radar_parameters = radar_parameters,
        radar_calibration = radar_calibration,
        georeference_correction = georeference_correction,
        status_str = status_str_v,
        extra_attrs = extra_attrs,
    )
end

function _read_sweep_v2(grp, volume_start::DateTime, fixed_angle::Float64, default_sweep_number::Int)
    haskey(grp.dim, "time") || throw(ArgumentError("v2 sweep group missing `time` dim"))
    haskey(grp.dim, "range") || throw(ArgumentError("v2 sweep group missing `range` dim"))

    sweep_number_raw = _readscalar(grp, "sweep_number")
    sweep_number = sweep_number_raw === nothing ? default_sweep_number : Int(sweep_number_raw)
    sweep_mode = _to_string(_readscalar(grp, "sweep_mode") === nothing ?
                            "azimuth_surveillance" : _readscalar(grp, "sweep_mode"))
    fixed_local = _readscalar(grp, "sweep_fixed_angle")
    fixed_v = fixed_local === nothing ? fixed_angle : _f64_or(fixed_local, fixed_angle)

    follow_mode = _to_string(_readscalar(grp, "follow_mode") === nothing ?
                             "none" : _readscalar(grp, "follow_mode"))
    prt_mode = _to_string(_readscalar(grp, "prt_mode") === nothing ?
                          "fixed" : _readscalar(grp, "prt_mode"))
    polarization_mode = _to_string(_readscalar(grp, "polarization_mode") === nothing ?
                                   "horizontal" : _readscalar(grp, "polarization_mode"))

    range_raw = _readvar(grp, "range")
    range_raw === nothing && throw(ArgumentError("v2 sweep group missing `range` variable"))
    range_vec = collect(Float64, range_raw)

    range_meters_to_first_gate = nothing
    range_meters_between_gates = nothing
    range_spacing_is_constant = true
    if haskey(grp, "range")
        ra = grp["range"].attrib
        if haskey(ra, "meters_to_center_of_first_gate")
            range_meters_to_first_gate = Float64(ra["meters_to_center_of_first_gate"])
        end
        if haskey(ra, "meters_between_gates")
            range_meters_between_gates = Float64(ra["meters_between_gates"])
        end
        if haskey(ra, "spacing_is_constant")
            v = ra["spacing_is_constant"]
            range_spacing_is_constant = v isa Bool ? v :
                lowercase(_to_string(v)) in ("true", "1")
        end
    end
    if range_meters_to_first_gate === nothing
        sr = _readscalar(grp, "start_range")
        sr === nothing || (range_meters_to_first_gate = Float64(sr))
    end
    if range_meters_between_gates === nothing
        gs = _readscalar(grp, "ray_gate_spacing")
        gs === nothing || (range_meters_between_gates = Float64(gs))
    end

    # Time: stored as seconds since volume start.
    time_raw = _readvar(grp, "time")
    time_raw === nothing && throw(ArgumentError("v2 sweep group missing `time` variable"))
    time_vec = _times_from_seconds(time_raw, grp, volume_start)

    az_raw = _readvar(grp, "azimuth")
    el_raw = _readvar(grp, "elevation")
    az_raw === nothing && throw(ArgumentError("v2 sweep missing `azimuth`"))
    el_raw === nothing && throw(ArgumentError("v2 sweep missing `elevation`"))
    azimuth = _to_f64vec(az_raw)
    elevation = _to_f64vec(el_raw)

    n_rays = length(time_vec)
    n_gates = length(range_vec)

    # Optional per-ray scan params.
    pulse_width = _to_f64vec(_readvar(grp, "pulse_width"))
    prt = _to_f64vec(_readvar(grp, "prt"))
    prt_ratio = _to_f64vec(_readvar(grp, "prt_ratio"))
    nyquist_velocity = _to_f64vec(_readvar(grp, "nyquist_velocity"))
    unambiguous_range = _to_f64vec(_readvar(grp, "unambiguous_range"))
    n_samples = _to_int_vec(_readvar(grp, "n_samples"))
    antenna_transition = _to_bool_vec(_readvar(grp, "antenna_transition"))
    scan_rate = _to_f64vec(_readvar(grp, "scan_rate"))
    rx_range_resolution = _to_f64vec(_readvar(grp, "rx_range_resolution"))

    calib_index_raw = _readvar(grp, "calib_index")
    if calib_index_raw === nothing
        calib_index_raw = _readvar(grp, "r_calib_index")
    end
    calib_index = _to_int_vec(calib_index_raw)

    target_scan_rate = _readscalar(grp, "target_scan_rate")
    target_scan_rate_v = _f64_or_nothing(target_scan_rate)
    rays_are_indexed_raw = _readscalar(grp, "rays_are_indexed")
    rays_are_indexed = if rays_are_indexed_raw === nothing
        false
    elseif rays_are_indexed_raw isa Bool
        rays_are_indexed_raw
    else
        lowercase(_to_string(rays_are_indexed_raw)) in ("true", "1")
    end
    ray_angle_resolution_raw = _readscalar(grp, "ray_angle_resolution")
    ray_angle_resolution = _f64_or_nothing(ray_angle_resolution_raw)
    qc_procedures_raw = _readscalar(grp, "qc_procedures")
    qc_procedures = qc_procedures_raw === nothing ? nothing : _to_string(qc_procedures_raw)

    # Frequency may be on this group or absent.
    freq_raw = _readvar(grp, "frequency")
    frequency = freq_raw === nothing ? Float64[] : collect(Float64, freq_raw)

    # Optional georefs_applied as ray-level var (LROSE quirk #8).
    georefs_applied_ray = _to_bool_vec(_readvar(grp, "georefs_applied"))

    # Sub-groups
    georef = nothing
    if haskey(grp, :group) && haskey(grp.group, "georeference")
        georef = _read_georeference(grp.group["georeference"])
    elseif georefs_applied_ray !== nothing
        # Synthesize a stub with only georefs_applied if no georef group.
        # Need lat/lon/alt though; fall back to nothing.
        georef = nothing
    end

    radar_monitoring = nothing
    if haskey(grp.group, "radar_monitoring")
        radar_monitoring = _read_radar_monitoring(grp.group["radar_monitoring"])
    elseif haskey(grp.group, "monitoring")
        radar_monitoring = _read_radar_monitoring(grp.group["monitoring"])
    end

    # Field variables
    fields = OrderedDict{String,Field}()
    skip_vars = Set([
        "time", "range", "azimuth", "elevation",
        "sweep_number", "sweep_fixed_angle", "sweep_mode",
        "follow_mode", "prt_mode", "polarization_mode",
        "polarization_sequence", "frequency",
        "start_range", "ray_gate_spacing",
        "pulse_width", "prt", "prt_ratio", "prt_sequence",
        "nyquist_velocity", "unambiguous_range",
        "n_samples", "antenna_transition", "scan_rate",
        "rx_range_resolution",
        "calib_index", "r_calib_index",
        "target_scan_rate", "rays_are_indexed", "ray_angle_resolution",
        "qc_procedures", "georefs_applied",
    ])
    extra_attrs = Dict{String,Any}()
    for varname in keys(grp)
        varname in skip_vars && continue
        var = grp[varname]
        dnames = NCDatasets.dimnames(var)
        # We want 2-D fields with (time, range) or (range, time).
        if length(dnames) == 2 &&
           (("time" in dnames) && ("range" in dnames))
            metadata = _read_field_metadata(var)
            raw = var[:, :]
            raw = _normalize_field_layout(var, raw, n_rays, n_gates)
            data = _unpack_field_data(raw, metadata.scale_factor,
                                      metadata.add_offset, metadata.fill_value)
            fields[String(varname)] = Field(data, metadata)
        end
    end

    return SweepGroup(
        sweep_number = sweep_number,
        sweep_mode = sweep_mode,
        fixed_angle = fixed_v,
        follow_mode = follow_mode,
        prt_mode = prt_mode,
        polarization_mode = polarization_mode,
        frequency = frequency,
        time = time_vec,
        range = range_vec,
        azimuth = azimuth,
        elevation = elevation,
        range_meters_to_first_gate = range_meters_to_first_gate,
        range_meters_between_gates = range_meters_between_gates,
        range_spacing_is_constant = range_spacing_is_constant,
        pulse_width = pulse_width,
        prt = prt,
        prt_ratio = prt_ratio,
        nyquist_velocity = nyquist_velocity,
        unambiguous_range = unambiguous_range,
        n_samples = n_samples,
        antenna_transition = antenna_transition,
        scan_rate = scan_rate,
        target_scan_rate = target_scan_rate_v,
        rx_range_resolution = rx_range_resolution,
        calib_index = calib_index,
        rays_are_indexed = rays_are_indexed,
        ray_angle_resolution = ray_angle_resolution,
        qc_procedures = qc_procedures,
        fields = fields,
        georeference = georef,
        radar_monitoring = radar_monitoring,
        extra_attrs = extra_attrs,
    )
end

# Convert a time-since-volume-start vector to absolute DateTime, handling either
# Float64 seconds or already-decoded DateTime arrays.
function _times_from_seconds(time_raw, grp, volume_start::DateTime)
    if eltype(time_raw) <: DateTime
        return Vector{DateTime}(time_raw)
    end
    units_str = haskey(grp["time"].attrib, "units") ?
        _to_string(grp["time"].attrib["units"]) : "seconds since volume start"
    base = _parse_units_base_time(units_str, volume_start)
    out = Vector{DateTime}(undef, length(time_raw))
    @inbounds for i in eachindex(time_raw)
        v = time_raw[i]
        secs = ismissing(v) ? 0.0 : Float64(v)
        out[i] = base + Millisecond(round(Int, secs * 1000))
    end
    return out
end

function _parse_units_base_time(units_str::AbstractString, fallback::DateTime)
    m = match(r"seconds since\s+([0-9T:\-Z\.\s]+)", units_str)
    if m === nothing
        return fallback
    end
    try
        return _parse_iso8601(m.captures[1])
    catch
        return fallback
    end
end

# Read RadarParameters from a v2 group or from flat root vars (v1).
function _read_radar_parameters(grp)
    extra = Dict{String,Any}()
    kw = Dict{Symbol,Any}()
    mapping = Dict(
        "radar_antenna_gain_h" => :antenna_gain_h,
        "antenna_gain_h" => :antenna_gain_h,
        "radar_antenna_gain_v" => :antenna_gain_v,
        "antenna_gain_v" => :antenna_gain_v,
        "radar_beam_width_h" => :beam_width_h,
        "beam_width_h" => :beam_width_h,
        "radar_beam_width_v" => :beam_width_v,
        "beam_width_v" => :beam_width_v,
        "radar_rx_bandwidth" => :receiver_bandwidth,
        "receiver_bandwidth" => :receiver_bandwidth,
    )
    for varname in keys(grp)
        v = _readscalar(grp, varname)
        v === nothing && continue
        if haskey(mapping, varname)
            sym = mapping[varname]
            if !haskey(kw, sym)
                kw[sym] = ismissing(v) ? nothing : Float64(v)
            end
        elseif varname == "frequency"
            extra["frequency"] = collect(grp[varname][:])
        else
            extra[String(varname)] = v
        end
    end
    kw[:extra_vars] = extra
    return RadarParameters(; kw...)
end

function _read_radar_monitoring(grp)
    kw = Dict{Symbol,Any}()
    extra = Dict{String,Any}()
    canonical = Set([
        :measured_transmit_power_h, :measured_transmit_power_v,
        :measured_sky_noise, :measured_cold_noise, :measured_hot_noise,
        :phase_difference_transmit_hv,
        :antenna_pointing_accuracy_elev, :antenna_pointing_accuracy_az,
        :calibration_offset_h, :calibration_offset_v, :zdr_offset,
    ])
    for varname in keys(grp)
        sym = Symbol(varname)
        if sym in canonical
            v = _to_f64vec(grp[varname][:])
            v === nothing || (kw[sym] = v)
        else
            extra[String(varname)] = grp[varname][:]
        end
    end
    kw[:extra_vars] = extra
    return RadarMonitoring(; kw...)
end

function _read_georeference_correction(grp)
    kw = Dict{Symbol,Any}()
    extra = Dict{String,Any}()
    canonical = Set(fieldnames(GeoreferenceCorrection))
    for varname in keys(grp)
        sym = Symbol(varname)
        if sym in canonical && sym !== :extra_vars
            v = _readscalar(grp, varname)
            v === nothing || (kw[sym] = ismissing(v) ? nothing : Float64(v))
        else
            extra[String(varname)] = _readscalar(grp, varname)
        end
    end
    kw[:extra_vars] = extra
    return GeoreferenceCorrection(; kw...)
end

# ── CfRadial v1 reader ───────────────────────────────────────────────────────

function _read_cfradial1(ds, file)
    # Globals
    conventions = _attr_str(ds, "Conventions", "CF-1.7")
    version = _attr_str(ds, "version", "CF-Radial-1.4")
    title = _attr_str(ds, "title")
    institution = _attr_str(ds, "institution")
    source = _attr_str(ds, "source")
    history = _attr_str(ds, "history")
    references = _attr_str(ds, "references")
    comment = _attr_str(ds, "comment")
    instrument_name = _attr_str(ds, "instrument_name")
    site_name = _attr_str(ds, "site_name")
    scan_name = _attr_str(ds, "scan_name")
    scan_id = _attr_int(ds, "scan_id", 0)
    platform_is_mobile = _attr_bool(ds, "platform_is_mobile", false)
    ray_times_increase = _attr_bool(ds, "ray_times_increase", true)
    simulated_data = _attr_bool(ds, "simulated_data", false)

    extra_attrs = Dict{String,Any}()
    handled_attrs = Set([
        "Conventions", "Sub_conventions", "version", "title", "institution",
        "source", "history", "references", "comment",
        "instrument_name", "site_name", "scan_name", "scan_id",
        "platform_is_mobile", "ray_times_increase", "simulated_data",
    ])
    for (k, v) in ds.attrib
        if !(k in handled_attrs)
            extra_attrs[String(k)] = v
        end
    end

    # Required vars
    haskey(ds, "latitude") || throw(ArgumentError("$file: required variable `latitude` is missing"))
    haskey(ds, "longitude") || throw(ArgumentError("$file: required variable `longitude` is missing"))
    haskey(ds, "altitude") || throw(ArgumentError("$file: required variable `altitude` is missing"))
    haskey(ds, "time") || throw(ArgumentError("$file: required variable `time` is missing"))
    haskey(ds, "range") || throw(ArgumentError("$file: required variable `range` is missing"))

    lat_arr = _readvar(ds, "latitude")
    lon_arr = _readvar(ds, "longitude")
    alt_arr = _readvar(ds, "altitude")
    lat0 = _f64_or(lat_arr isa AbstractArray ? lat_arr[1] : lat_arr, 0.0)
    lon0 = _f64_or(lon_arr isa AbstractArray ? lon_arr[1] : lon_arr, 0.0)
    alt0 = _f64_or(alt_arr isa AbstractArray ? alt_arr[1] : alt_arr, 0.0)

    altitude_agl = _readvar(ds, "altitude_agl")
    altitude_agl_v = if altitude_agl === nothing
        nothing
    elseif altitude_agl isa AbstractArray
        _f64_or_nothing(altitude_agl[1])
    else
        _f64_or_nothing(altitude_agl)
    end

    tcs, tce = _read_time_coverage(ds)
    tcs === nothing && throw(ArgumentError("$file: required `time_coverage_start` is missing"))
    tce === nothing && throw(ArgumentError("$file: required `time_coverage_end` is missing"))
    time_start = _parse_iso8601(tcs)
    time_end = _parse_iso8601(tce)

    volume_number = _readscalar(ds, "volume_number")
    volume_number_v = volume_number === nothing ? 0 : Int(volume_number)
    platform_type = _readscalar(ds, "platform_type")
    platform_type_v = platform_type === nothing ? "fixed" : _to_string(platform_type)
    instrument_type = _readscalar(ds, "instrument_type")
    instrument_type_v = instrument_type === nothing ? "radar" : _to_string(instrument_type)
    primary_axis = _readscalar(ds, "primary_axis")
    primary_axis_v = primary_axis === nothing ? "axis_z" : _to_string(primary_axis)

    status_str = _readscalar(ds, "status_str")
    if status_str === nothing
        status_str = _readscalar(ds, "status_xml")
    end
    status_str_v = status_str === nothing ? nothing : _to_string(status_str)

    # Time array
    time_raw = ds["time"]
    time_arr = time_raw[:]
    time_vec = _times_from_seconds_root(time_arr, ds, time_start)

    # Sweep boundaries
    sweep_start = _readvar(ds, "sweep_start_ray_index")
    sweep_end = _readvar(ds, "sweep_end_ray_index")
    sweep_start === nothing && throw(ArgumentError("$file: v1 file missing `sweep_start_ray_index`"))
    sweep_end === nothing && throw(ArgumentError("$file: v1 file missing `sweep_end_ray_index`"))
    n_sweeps = length(sweep_start)

    fixed_angles = _readvar(ds, "fixed_angle")
    sweep_modes = _readstring_per_sweep(ds, "sweep_mode", n_sweeps)
    follow_modes = _readstring_per_sweep(ds, "follow_mode", n_sweeps)
    prt_modes = _readstring_per_sweep(ds, "prt_mode", n_sweeps)
    pol_modes = _readstring_per_sweep(ds, "polarization_mode", n_sweeps)
    rays_indexed = _readstring_per_sweep(ds, "rays_are_indexed", n_sweeps)
    ray_angle_res = _readvar(ds, "ray_angle_res")
    target_scan_rates = _readvar(ds, "target_scan_rate")
    sweep_numbers = _readvar(ds, "sweep_number")

    # Per-ray vars
    azimuth = _to_f64vec(_readvar(ds, "azimuth"))
    elevation = _to_f64vec(_readvar(ds, "elevation"))
    pulse_width = _to_f64vec(_readvar(ds, "pulse_width"))
    prt = _to_f64vec(_readvar(ds, "prt"))
    prt_ratio = _to_f64vec(_readvar(ds, "prt_ratio"))
    nyquist_velocity = _to_f64vec(_readvar(ds, "nyquist_velocity"))
    unambiguous_range = _to_f64vec(_readvar(ds, "unambiguous_range"))
    n_samples_all = _to_int_vec(_readvar(ds, "n_samples"))
    antenna_transition_all = _to_bool_vec(_readvar(ds, "antenna_transition"))
    scan_rate_all = _to_f64vec(_readvar(ds, "scan_rate"))
    calib_index_all = _to_int_vec(_readvar(ds, "calib_index") === nothing ?
        _readvar(ds, "r_calib_index") : _readvar(ds, "calib_index"))
    rx_resolution_all = _to_f64vec(_readvar(ds, "rx_range_resolution"))

    # Range axis
    range_var = ds["range"]
    range_vec = collect(Float64, range_var[:])
    range_meters_to_first_gate = nothing
    range_meters_between_gates = nothing
    range_spacing_is_constant = true
    ra = range_var.attrib
    if haskey(ra, "meters_to_center_of_first_gate")
        range_meters_to_first_gate = Float64(ra["meters_to_center_of_first_gate"])
    end
    if haskey(ra, "meters_between_gates")
        range_meters_between_gates = Float64(ra["meters_between_gates"])
    end
    if haskey(ra, "spacing_is_constant")
        v = ra["spacing_is_constant"]
        range_spacing_is_constant = v isa Bool ? v :
            lowercase(_to_string(v)) in ("true", "1")
    end

    # Frequency (instrument-level scalar/vector)
    freq_raw = _readvar(ds, "frequency")
    frequency = freq_raw === nothing ? Float64[] : collect(Float64, freq_raw)

    # Identify field variables: 2-D vars over (time, range) or (range, time).
    n_total_rays = length(time_vec)
    n_gates = length(range_vec)
    field_vars = String[]
    for varname in keys(ds)
        var = ds[varname]
        ndims(var) == 2 || continue
        dnames = String.(NCDatasets.dimnames(var))
        if Set(dnames) == Set(["time", "range"])
            push!(field_vars, varname)
        end
    end

    # Build sweeps
    sweeps = SweepGroup[]
    for i in 1:n_sweeps
        s = _int_or(sweep_start[i], 0) + 1
        e = _int_or(sweep_end[i], 0) + 1
        idx = s:e
        n_rays = length(idx)

        sweep_number_i = sweep_numbers === nothing ? i - 1 : _int_or(sweep_numbers[i], i - 1)
        sweep_mode_i = sweep_modes === nothing ? "azimuth_surveillance" : sweep_modes[i]
        fixed_i = fixed_angles === nothing ? NaN : _f64_or(fixed_angles[i], NaN)
        follow_i = follow_modes === nothing ? "none" : follow_modes[i]
        prt_i = prt_modes === nothing ? "fixed" : prt_modes[i]
        pol_i = pol_modes === nothing ? "horizontal" : pol_modes[i]
        rays_indexed_i = rays_indexed === nothing ? false :
            lowercase(rays_indexed[i]) in ("true", "1")
        ray_angle_res_i = ray_angle_res === nothing ? nothing : _f64_or_nothing(ray_angle_res[i])
        target_rate_i = target_scan_rates === nothing ? nothing :
            _f64_or_nothing(target_scan_rates[i])

        time_i = time_vec[idx]
        azimuth_i = azimuth === nothing ? Float64[] : azimuth[idx]
        elevation_i = elevation === nothing ? Float64[] : elevation[idx]
        pulse_width_i = pulse_width === nothing ? nothing : pulse_width[idx]
        prt_i_arr = prt === nothing ? nothing : prt[idx]
        prt_ratio_i = prt_ratio === nothing ? nothing : prt_ratio[idx]
        nyquist_i = nyquist_velocity === nothing ? nothing : nyquist_velocity[idx]
        unamb_i = unambiguous_range === nothing ? nothing : unambiguous_range[idx]
        n_samp_i = n_samples_all === nothing ? nothing : n_samples_all[idx]
        ant_tr_i = antenna_transition_all === nothing ? nothing : antenna_transition_all[idx]
        scan_rate_i = scan_rate_all === nothing ? nothing : scan_rate_all[idx]
        calib_idx_i = calib_index_all === nothing ? nothing : calib_index_all[idx]
        rx_res_i = rx_resolution_all === nothing ? nothing : rx_resolution_all[idx]

        # Sub-groups: v1 stores georeference per-ray as flat arrays.
        georef_i = nothing
        if length(lat_arr) == n_total_rays
            kw = Dict{Symbol,Any}()
            kw[:latitude] = collect(Float64, lat_arr[idx])
            kw[:longitude] = collect(Float64, lon_arr[idx])
            kw[:altitude] = collect(Float64, alt_arr[idx])
            for fld in (:heading, :roll, :pitch, :drift, :rotation, :tilt,
                        :eastward_velocity, :northward_velocity, :vertical_velocity,
                        :eastward_wind, :northward_wind, :vertical_wind,
                        :heading_rate, :roll_rate, :pitch_rate)
                raw = _readvar(ds, String(fld))
                raw === nothing && continue
                if length(raw) == n_total_rays
                    kw[fld] = _to_f64vec(raw[idx])
                end
            end
            ga = _readvar(ds, "georefs_applied")
            if ga !== nothing && length(ga) == n_total_rays
                kw[:georefs_applied] = _to_bool_vec(ga[idx])
            end
            kw[:extra_vars] = Dict{String,Any}()
            georef_i = Georeference(; kw...)
        end

        # Field data
        fields_i = OrderedDict{String,Field}()
        for fname in field_vars
            var = ds[fname]
            metadata = _read_field_metadata(var)
            raw = var[:, :]
            raw = _normalize_field_layout(var, raw, n_total_rays, n_gates)
            # Slice the (time, range) matrix to this sweep.
            sliced = raw[idx, :]
            data = _unpack_field_data(sliced, metadata.scale_factor,
                                      metadata.add_offset, metadata.fill_value)
            fields_i[fname] = Field(data, metadata)
        end

        push!(sweeps, SweepGroup(
            sweep_number = sweep_number_i,
            sweep_mode = sweep_mode_i,
            fixed_angle = fixed_i,
            follow_mode = follow_i,
            prt_mode = prt_i,
            polarization_mode = pol_i,
            frequency = frequency,
            time = time_i,
            range = range_vec,
            azimuth = azimuth_i,
            elevation = elevation_i,
            range_meters_to_first_gate = range_meters_to_first_gate,
            range_meters_between_gates = range_meters_between_gates,
            range_spacing_is_constant = range_spacing_is_constant,
            pulse_width = pulse_width_i,
            prt = prt_i_arr,
            prt_ratio = prt_ratio_i,
            nyquist_velocity = nyquist_i,
            unambiguous_range = unamb_i,
            n_samples = n_samp_i,
            antenna_transition = ant_tr_i,
            scan_rate = scan_rate_i,
            target_scan_rate = target_rate_i,
            rx_range_resolution = rx_res_i,
            calib_index = calib_idx_i,
            rays_are_indexed = rays_indexed_i,
            ray_angle_resolution = ray_angle_res_i,
            fields = fields_i,
            georeference = georef_i,
        ))
    end

    # Volume-level radar_parameters / radar_calibration / georeference_correction
    radar_parameters = _read_radar_parameters_v1(ds)
    radar_calibration = _read_radar_calibration_v1(ds)
    georeference_correction = _read_georeference_correction_v1(ds)

    return Volume(
        conventions = conventions,
        version = version,
        title = title,
        institution = institution,
        source = source,
        history = history,
        references = references,
        comment = comment,
        instrument_name = instrument_name,
        site_name = site_name,
        scan_name = scan_name,
        scan_id = scan_id,
        platform_is_mobile = platform_is_mobile,
        ray_times_increase = ray_times_increase,
        simulated_data = simulated_data,
        volume_number = volume_number_v,
        platform_type = platform_type_v,
        instrument_type = instrument_type_v,
        primary_axis = primary_axis_v,
        time_coverage_start = time_start,
        time_coverage_end = time_end,
        latitude = lat0,
        longitude = lon0,
        altitude = alt0,
        altitude_agl = altitude_agl_v,
        sweeps = sweeps,
        radar_parameters = radar_parameters,
        radar_calibration = radar_calibration,
        georeference_correction = georeference_correction,
        status_str = status_str_v,
        extra_attrs = extra_attrs,
    )
end

function _times_from_seconds_root(time_arr, ds, fallback::DateTime)
    if eltype(time_arr) <: DateTime
        return Vector{DateTime}(time_arr)
    end
    units_str = haskey(ds["time"].attrib, "units") ?
        _to_string(ds["time"].attrib["units"]) : "seconds since volume start"
    base = _parse_units_base_time(units_str, fallback)
    out = Vector{DateTime}(undef, length(time_arr))
    @inbounds for i in eachindex(time_arr)
        v = time_arr[i]
        secs = ismissing(v) ? 0.0 : Float64(v)
        out[i] = base + Millisecond(round(Int, secs * 1000))
    end
    return out
end

# v1 stores radar parameters as flat vars with `meta_group="radar_parameters"`.
function _read_radar_parameters_v1(ds)
    mapping = Dict(
        "radar_antenna_gain_h" => :antenna_gain_h,
        "radar_antenna_gain_v" => :antenna_gain_v,
        "radar_beam_width_h" => :beam_width_h,
        "radar_beam_width_v" => :beam_width_v,
        "radar_rx_bandwidth" => :receiver_bandwidth,
    )
    kw = Dict{Symbol,Any}()
    extra = Dict{String,Any}()
    found = false
    for (varname, sym) in mapping
        haskey(ds, varname) || continue
        v = _readscalar(ds, varname)
        v === nothing && continue
        kw[sym] = ismissing(v) ? nothing : Float64(v)
        found = true
    end
    if haskey(ds, "frequency") && length(ds["frequency"]) > 0
        extra["frequency"] = collect(ds["frequency"][:])
        found = true
    end
    found || return nothing
    kw[:extra_vars] = extra
    return RadarParameters(; kw...)
end

function _read_radar_calibration_v1(ds)
    has_calib = false
    for k in keys(ds)
        if startswith(String(k), "r_calib_")
            has_calib = true
            break
        end
    end
    has_calib || return nothing
    n = haskey(ds.dim, "r_calib") ? ds.dim["r_calib"] :
        haskey(ds.dim, "calib") ? ds.dim["calib"] : 1
    entries = RadarCalibrationEntry[]
    for i in 1:n
        kw = Dict{Symbol,Any}()
        extra = Dict{String,Any}()
        for varname in keys(ds)
            sname = String(varname)
            startswith(sname, "r_calib_") || continue
            stripped, mapped = _calib_canonical_name(sname)
            data = ds[varname][:]
            value = ndims(data) == 0 || length(data) == 1 ? first(data) : data[i]
            if mapped !== nothing
                if mapped === :time
                    kw[mapped] = _parse_iso8601_or_nothing(_to_string(value))
                else
                    kw[mapped] = ismissing(value) ? nothing : Float64(value)
                end
            else
                extra[stripped] = value
            end
        end
        kw[:extra_vars] = extra
        push!(entries, RadarCalibrationEntry(; kw...))
    end
    cal_idx = nothing
    if haskey(ds, "r_calib_index")
        cal_idx = _to_int_vec(ds["r_calib_index"][:])
    elseif haskey(ds, "calib_index")
        cal_idx = _to_int_vec(ds["calib_index"][:])
    end
    return RadarCalibration(entries = entries, calib_index = cal_idx)
end

function _read_georeference_correction_v1(ds)
    canonical = Set(fieldnames(GeoreferenceCorrection))
    kw = Dict{Symbol,Any}()
    found = false
    for k in keys(ds)
        sym = Symbol(k)
        if sym in canonical && sym !== :extra_vars
            v = _readscalar(ds, k)
            v === nothing && continue
            kw[sym] = ismissing(v) ? nothing : Float64(v)
            found = true
        end
    end
    found || return nothing
    kw[:extra_vars] = Dict{String,Any}()
    return GeoreferenceCorrection(; kw...)
end
