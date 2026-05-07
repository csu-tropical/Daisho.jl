# CfRadial 2.1 writer.
#
# `write_cfradial(volume, file)` writes a fresh spec-canonical NetCDF4 file with
# sweep groups. `update_cfradial(in, out, volume; fields=nothing)` copies an
# existing file and overwrites/adds field variables in place, preserving the
# input's format (v1 flat or v2 grouped).

using NCDatasets
using DataStructures
using Dates
using Printf

# ── write_cfradial ───────────────────────────────────────────────────────────

"""
    write_cfradial(volume::Volume, file::AbstractString; write_extras::Bool=false)

Write `volume` to `file` as a spec-canonical CfRadial 2.1 NetCDF4 file. Drops
non-spec attributes from `extra_attrs` / `extra_vars` unless
`write_extras=true`, in which case they are emitted alongside spec content.
"""
function write_cfradial(volume::Volume, file::AbstractString; write_extras::Bool=false)
    isfile(file) && rm(file)
    NCDataset(file, "c", format=:netcdf4) do ds
        _write_volume_globals!(ds, volume; write_extras=write_extras)
        _write_volume_root_vars!(ds, volume; write_extras=write_extras)
        _write_volume_subgroups!(ds, volume; write_extras=write_extras)
        for (i, sweep) in enumerate(volume.sweeps)
            gname = @sprintf("sweep_%04d", i)
            grp = defGroup(ds, gname)
            _write_sweep!(grp, sweep, volume.time_coverage_start; write_extras=write_extras)
        end
    end
    return file
end

# Set global attributes (required + optional) and pass through extras.
function _write_volume_globals!(ds, v::Volume; write_extras::Bool)
    ds.attrib["Conventions"] = "Cf/Radial-2.1"
    ds.attrib["version"] = v.version
    ds.attrib["title"] = v.title
    ds.attrib["institution"] = v.institution
    ds.attrib["source"] = v.source
    ds.attrib["history"] = v.history
    ds.attrib["instrument_name"] = v.instrument_name
    ds.attrib["site_name"] = v.site_name
    isempty(v.references) || (ds.attrib["references"] = v.references)
    isempty(v.comment) || (ds.attrib["comment"] = v.comment)
    isempty(v.scan_name) || (ds.attrib["scan_name"] = v.scan_name)
    v.scan_id == 0 || (ds.attrib["scan_id"] = Int32(v.scan_id))
    ds.attrib["platform_is_mobile"] = v.platform_is_mobile ? "true" : "false"
    ds.attrib["ray_times_increase"] = v.ray_times_increase ? "true" : "false"
    v.simulated_data && (ds.attrib["simulated_data"] = "true")
    if write_extras
        for (k, val) in v.extra_attrs
            haskey(ds.attrib, k) && continue
            try
                ds.attrib[k] = val
            catch
            end
        end
    end
    return nothing
end

function _write_volume_root_vars!(ds, v::Volume; write_extras::Bool)
    n_sweeps = length(v.sweeps)
    defDim(ds, "sweep", n_sweeps)
    if !isempty(v.sweeps) && !isempty(v.sweeps[1].frequency)
        defDim(ds, "frequency", length(v.sweeps[1].frequency))
    end

    # time_coverage_start/end as ISO8601 string vars.
    defVar(ds, "time_coverage_start", _iso8601(v.time_coverage_start), ();
        attrib = OrderedDict("long_name" => "data_volume_start_time_utc"))
    defVar(ds, "time_coverage_end", _iso8601(v.time_coverage_end), ();
        attrib = OrderedDict("long_name" => "data_volume_end_time_utc"))

    defVar(ds, "volume_number", Int32(v.volume_number), ();
        attrib = OrderedDict("long_name" => "data_volume_index_number",
                             "_FillValue" => Int32(-9999)))
    defVar(ds, "platform_type", v.platform_type, ();
        attrib = OrderedDict("long_name" => "platform_type"))
    defVar(ds, "instrument_type", v.instrument_type, ();
        attrib = OrderedDict("long_name" => "type_of_instrument"))
    defVar(ds, "primary_axis", v.primary_axis, ();
        attrib = OrderedDict("long_name" => "primary_axis_of_rotation"))

    defVar(ds, "latitude", Float64(v.latitude), ();
        attrib = OrderedDict("long_name" => "latitude", "units" => "degrees_north",
                             "_FillValue" => -9999.0))
    defVar(ds, "longitude", Float64(v.longitude), ();
        attrib = OrderedDict("long_name" => "longitude", "units" => "degrees_east",
                             "_FillValue" => -9999.0))
    defVar(ds, "altitude", Float64(v.altitude), ();
        attrib = OrderedDict("long_name" => "altitude", "units" => "meters",
                             "_FillValue" => -9999.0, "positive" => "up"))
    if v.altitude_agl !== nothing
        defVar(ds, "altitude_agl", Float64(v.altitude_agl), ();
            attrib = OrderedDict("long_name" => "altitude_above_ground_level",
                                 "units" => "meters", "_FillValue" => -9999.0,
                                 "positive" => "up"))
    end

    if v.status_str !== nothing
        defVar(ds, "status_str", String(v.status_str), ();
            attrib = OrderedDict("long_name" => "status_of_instrument"))
    end

    sweep_names = [@sprintf("sweep_%04d", i) for i in 1:n_sweeps]
    defVar(ds, "sweep_group_name", sweep_names, ("sweep",);
        attrib = OrderedDict("long_name" => "group_name_for_sweep"))
    fixed_angles = [Float32(s.fixed_angle) for s in v.sweeps]
    defVar(ds, "sweep_fixed_angle", fixed_angles, ("sweep",);
        attrib = OrderedDict("long_name" => "fixed_angle_for_sweep",
                             "units" => "degrees",
                             "_FillValue" => Float32(-9999)))
    return nothing
end

function _write_volume_subgroups!(ds, v::Volume; write_extras::Bool)
    if v.radar_parameters !== nothing
        grp = defGroup(ds, "radar_parameters")
        _write_radar_parameters!(grp, v.radar_parameters, v.sweeps; write_extras=write_extras)
    end
    if v.radar_calibration !== nothing
        grp = defGroup(ds, "radar_calibration")
        _write_radar_calibration!(grp, v.radar_calibration; write_extras=write_extras)
    end
    if v.georeference_correction !== nothing
        grp = defGroup(ds, "georeference_correction")
        _write_georeference_correction!(grp, v.georeference_correction; write_extras=write_extras)
    end
    return nothing
end

function _write_radar_parameters!(grp, p::RadarParameters, sweeps; write_extras::Bool)
    if !isempty(sweeps) && !isempty(sweeps[1].frequency)
        defDim(grp, "frequency", length(sweeps[1].frequency))
        defVar(grp, "frequency", Float32.(sweeps[1].frequency), ("frequency",);
            attrib = OrderedDict("long_name" => "transmission_frequency",
                                 "units" => "s-1", "_FillValue" => Float32(-9999)))
    end
    if p.antenna_gain_h !== nothing
        defVar(grp, "antenna_gain_h", Float32(p.antenna_gain_h), ();
            attrib = OrderedDict("long_name" => "nominal_radar_antenna_gain_h_channel",
                                 "units" => "dB", "_FillValue" => Float32(-9999)))
    end
    if p.antenna_gain_v !== nothing
        defVar(grp, "antenna_gain_v", Float32(p.antenna_gain_v), ();
            attrib = OrderedDict("long_name" => "nominal_radar_antenna_gain_v_channel",
                                 "units" => "dB", "_FillValue" => Float32(-9999)))
    end
    if p.beam_width_h !== nothing
        defVar(grp, "beam_width_h", Float32(p.beam_width_h), ();
            attrib = OrderedDict("long_name" => "half_power_radar_beam_width_h_channel",
                                 "units" => "degrees", "_FillValue" => Float32(-9999)))
    end
    if p.beam_width_v !== nothing
        defVar(grp, "beam_width_v", Float32(p.beam_width_v), ();
            attrib = OrderedDict("long_name" => "half_power_radar_beam_width_v_channel",
                                 "units" => "degrees", "_FillValue" => Float32(-9999)))
    end
    if p.receiver_bandwidth !== nothing
        defVar(grp, "receiver_bandwidth", Float32(p.receiver_bandwidth), ();
            attrib = OrderedDict("long_name" => "radar_receiver_bandwidth",
                                 "units" => "s-1", "_FillValue" => Float32(-9999)))
    end
    return nothing
end

function _write_radar_calibration!(grp, c::RadarCalibration; write_extras::Bool)
    n = length(c.entries)
    n == 0 && return nothing
    defDim(grp, "calib", n)

    times_str = String[]
    for e in c.entries
        push!(times_str, e.time === nothing ? "" : _iso8601(e.time))
    end
    if any(!isempty, times_str)
        defVar(grp, "time", times_str, ("calib",);
            attrib = OrderedDict("long_name" => "radar_calibration_time_utc"))
    end

    flds = [
        :pulse_width, :antenna_gain_h, :antenna_gain_v,
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
    ]
    for fld in flds
        vals = [getfield(e, fld) for e in c.entries]
        any(v -> v !== nothing, vals) || continue
        v = Float32[(x === nothing ? -9999.0f0 : Float32(x)) for x in vals]
        defVar(grp, String(fld), v, ("calib",);
            attrib = OrderedDict("_FillValue" => Float32(-9999)))
    end
    return nothing
end

function _write_georeference_correction!(grp, gc::GeoreferenceCorrection; write_extras::Bool)
    for fld in fieldnames(GeoreferenceCorrection)
        fld === :extra_vars && continue
        v = getfield(gc, fld)
        v === nothing && continue
        defVar(grp, String(fld), Float64(v), ();
            attrib = OrderedDict("_FillValue" => -9999.0))
    end
    return nothing
end

function _write_sweep!(grp, s::SweepGroup, volume_start::DateTime; write_extras::Bool)
    n_rays = length(s.time)
    n_gates = length(s.range)
    defDim(grp, "time", n_rays)
    defDim(grp, "range", n_gates)

    # Per-sweep scalar attrs/vars
    defVar(grp, "sweep_number", Int32(s.sweep_number), ();
        attrib = OrderedDict("long_name" => "sweep_index_number_0_based",
                             "_FillValue" => Int32(-9999)))
    defVar(grp, "sweep_fixed_angle", Float32(s.fixed_angle), ();
        attrib = OrderedDict("long_name" => "ray_target_fixed_angle",
                             "units" => "degrees", "_FillValue" => Float32(-9999)))
    defVar(grp, "sweep_mode", s.sweep_mode, ();
        attrib = OrderedDict("long_name" => "scan_mode_for_sweep"))
    defVar(grp, "follow_mode", s.follow_mode, ();
        attrib = OrderedDict("long_name" => "follow_mode_for_scan_strategy"))
    defVar(grp, "prt_mode", s.prt_mode, ();
        attrib = OrderedDict("long_name" => "transmit_pulse_mode"))
    defVar(grp, "polarization_mode", s.polarization_mode, ();
        attrib = OrderedDict("long_name" => "polarization_mode_for_sweep"))

    # Time coordinate (seconds since volume start).
    secs = Float64[Dates.value(t - volume_start) / 1000.0 for t in s.time]
    units_str = "seconds since " * _iso8601(volume_start)
    defVar(grp, "time", secs, ("time",);
        attrib = OrderedDict("standard_name" => "time",
                             "long_name" => "time in seconds since volume start",
                             "calendar" => "gregorian",
                             "units" => units_str))

    # Range coordinate.
    range_attrs = OrderedDict{String,Any}(
        "long_name" => "Range from instrument to center of gate",
        "units" => "meters",
        "spacing_is_constant" => s.range_spacing_is_constant ? "true" : "false",
    )
    if s.range_meters_to_first_gate !== nothing
        range_attrs["meters_to_center_of_first_gate"] = Float64(s.range_meters_to_first_gate)
    end
    if s.range_meters_between_gates !== nothing
        range_attrs["meters_between_gates"] = Float64(s.range_meters_between_gates)
    end
    defVar(grp, "range", Float32.(s.range), ("range",); attrib = range_attrs)

    # Per-ray angles.
    defVar(grp, "azimuth", Float32.(s.azimuth), ("time",);
        attrib = OrderedDict("long_name" => "azimuth_angle_from_true_north",
                             "units" => "degrees", "_FillValue" => Float32(-9999)))
    defVar(grp, "elevation", Float32.(s.elevation), ("time",);
        attrib = OrderedDict("long_name" => "elevation_angle_from_horizontal_plane",
                             "units" => "degrees", "_FillValue" => Float32(-9999)))

    _write_optional_per_ray!(grp, "pulse_width", s.pulse_width, "transmitter_pulse_width", "seconds")
    _write_optional_per_ray!(grp, "prt", s.prt, "pulse_repetition_time", "seconds")
    _write_optional_per_ray!(grp, "prt_ratio", s.prt_ratio, "pulse_repetition_frequency_ratio", "")
    _write_optional_per_ray!(grp, "nyquist_velocity", s.nyquist_velocity, "unambiguous_doppler_velocity", "meters per second")
    _write_optional_per_ray!(grp, "unambiguous_range", s.unambiguous_range, "unambiguous_range", "meters")
    if s.n_samples !== nothing
        defVar(grp, "n_samples", Int32.(s.n_samples), ("time",);
            attrib = OrderedDict("long_name" => "number_of_samples_used_to_compute_moments",
                                 "_FillValue" => Int32(-9999)))
    end
    if s.antenna_transition !== nothing
        defVar(grp, "antenna_transition", Int8.(s.antenna_transition), ("time",);
            attrib = OrderedDict("long_name" => "antenna_is_in_transition_between_sweeps"))
    end
    _write_optional_per_ray!(grp, "scan_rate", s.scan_rate, "antenna_scan_rate", "degrees per second")
    if s.target_scan_rate !== nothing
        defVar(grp, "target_scan_rate", Float32(s.target_scan_rate), ();
            attrib = OrderedDict("long_name" => "target_scan_rate_for_sweep",
                                 "units" => "degrees per second",
                                 "_FillValue" => Float32(-9999)))
    end
    if s.calib_index !== nothing
        defVar(grp, "calib_index", Int32.(s.calib_index), ("time",);
            attrib = OrderedDict("long_name" => "calibration_data_array_index_per_ray",
                                 "_FillValue" => Int32(-9999)))
    end
    if s.ray_angle_resolution !== nothing
        defVar(grp, "ray_angle_resolution", Float32(s.ray_angle_resolution), ();
            attrib = OrderedDict("long_name" => "angular_resolution_between_rays",
                                 "units" => "degrees", "_FillValue" => Float32(-9999)))
    end
    if s.rays_are_indexed
        defVar(grp, "rays_are_indexed", "true", ();
            attrib = OrderedDict("long_name" => "flag_for_indexed_rays"))
    end

    # Field variables.
    for (name, fld) in s.fields
        _write_field!(grp, name, fld; write_extras=write_extras)
    end

    # Sub-groups.
    if s.georeference !== nothing
        sub = defGroup(grp, "georeference")
        _write_georeference!(sub, s.georeference; write_extras=write_extras)
    end
    if s.radar_monitoring !== nothing
        sub = defGroup(grp, "radar_monitoring")
        _write_radar_monitoring!(sub, s.radar_monitoring; write_extras=write_extras)
    end
    return nothing
end

function _write_optional_per_ray!(grp, name, vals, long_name, units)
    vals === nothing && return
    defVar(grp, name, Float32.(vals), ("time",);
        attrib = OrderedDict("long_name" => long_name,
                             "units" => units,
                             "_FillValue" => Float32(-9999)))
    return nothing
end

function _write_field!(grp, name, fld::Field; write_extras::Bool)
    md = fld.metadata
    attrs = OrderedDict{String,Any}()
    md.standard_name === nothing || (attrs["standard_name"] = md.standard_name)
    md.long_name === nothing || (attrs["long_name"] = md.long_name)
    md.units === nothing || (attrs["units"] = md.units)
    md.coordinates === nothing || (attrs["coordinates"] = md.coordinates)
    attrs["sampling_ratio"] = Float32(md.sampling_ratio)
    md.is_discrete && (attrs["is_discrete"] = "true")
    md.field_folds && (attrs["field_folds"] = "true")
    md.fold_limit_lower === nothing || (attrs["fold_limit_lower"] = Float32(md.fold_limit_lower))
    md.fold_limit_upper === nothing || (attrs["fold_limit_upper"] = Float32(md.fold_limit_upper))
    md.is_quality_field && (attrs["is_quality_field"] = "true")
    isempty(md.qualified_variables) || (attrs["qualified_variables"] = join(md.qualified_variables, " "))
    isempty(md.ancillary_variables) || (attrs["ancillary_variables"] = join(md.ancillary_variables, " "))
    md.flag_values === nothing || (attrs["flag_values"] = md.flag_values)
    md.flag_meanings === nothing || (attrs["flag_meanings"] = join(md.flag_meanings, " "))
    md.flag_masks === nothing || (attrs["flag_masks"] = md.flag_masks)
    md.thresholding_xml === nothing || (attrs["thresholding_xml"] = md.thresholding_xml)
    md.legend_xml === nothing || (attrs["legend_xml"] = md.legend_xml)
    fill_v = md.fill_value === nothing ? Float32(-32768) : Float32(md.fill_value)
    attrs["_FillValue"] = fill_v
    md.undetect === nothing || (attrs["_Undetect"] = Float32(md.undetect))
    if write_extras
        for (k, v) in md.extra_attrs
            haskey(attrs, k) || (attrs[k] = v)
        end
    end
    data32 = Array{Float32}(undef, size(fld.data))
    @inbounds for i in eachindex(fld.data)
        v = fld.data[i]
        data32[i] = (ismissing(v) || (isa(v, AbstractFloat) && isnan(v))) ? fill_v : Float32(v)
    end
    defVar(grp, name, data32, ("time", "range"); attrib = attrs)
    return nothing
end

function _write_georeference!(grp, g::Georeference; write_extras::Bool)
    n = length(g.latitude)
    defDim(grp, "time", n)
    defVar(grp, "latitude", Float64.(g.latitude), ("time",);
        attrib = OrderedDict("units" => "degrees_north", "_FillValue" => -9999.0))
    defVar(grp, "longitude", Float64.(g.longitude), ("time",);
        attrib = OrderedDict("units" => "degrees_east", "_FillValue" => -9999.0))
    defVar(grp, "altitude", Float64.(g.altitude), ("time",);
        attrib = OrderedDict("units" => "meters", "_FillValue" => -9999.0))
    for fld in (:heading, :roll, :pitch, :drift, :rotation, :tilt,
                :eastward_velocity, :northward_velocity, :vertical_velocity,
                :eastward_wind, :northward_wind, :vertical_wind,
                :heading_rate, :roll_rate, :pitch_rate)
        v = getfield(g, fld)
        v === nothing && continue
        defVar(grp, String(fld), Float32.(v), ("time",);
            attrib = OrderedDict("_FillValue" => Float32(-9999)))
    end
    if g.georefs_applied !== nothing
        defVar(grp, "georefs_applied", Int8.(g.georefs_applied), ("time",))
    end
    return nothing
end

function _write_radar_monitoring!(grp, m::RadarMonitoring; write_extras::Bool)
    n = nothing
    for fld in fieldnames(RadarMonitoring)
        fld === :extra_vars && continue
        v = getfield(m, fld)
        v === nothing && continue
        n = length(v)
        break
    end
    n === nothing && return nothing
    defDim(grp, "time", n)
    for fld in fieldnames(RadarMonitoring)
        fld === :extra_vars && continue
        v = getfield(m, fld)
        v === nothing && continue
        defVar(grp, String(fld), Float32.(v), ("time",);
            attrib = OrderedDict("_FillValue" => Float32(-9999)))
    end
    return nothing
end

_iso8601(t::DateTime) = Dates.format(t, "yyyy-mm-ddTHH:MM:SSZ")

# ── update_cfradial ──────────────────────────────────────────────────────────

"""
    update_cfradial(input_file, output_file, volume::Volume; fields=nothing)

Format-preserving in-place update. Copies `input_file` to `output_file` and
overwrites/adds field variables from `volume`. The ray layout in `volume` must
match the file's existing layout; otherwise raises an error.

`fields` selects a subset of field names to write. If `nothing`, every field
in `volume.sweeps[*].fields` is written.
"""
function update_cfradial(input_file::AbstractString, output_file::AbstractString,
                         volume::Volume; fields::Union{Nothing,AbstractVector}=nothing)
    isfile(input_file) || throw(ArgumentError("input file not found: $input_file"))
    cp(input_file, output_file; force=true)
    is_v2 = NCDataset(output_file, "r") do ds
        _is_cfradial2(ds)
    end
    field_names_to_write = fields === nothing ? _all_field_names(volume) : String.(fields)

    NCDataset(output_file, "a") do ds
        if is_v2
            _update_cfradial2!(ds, volume, field_names_to_write)
        else
            _update_cfradial1!(ds, volume, field_names_to_write)
        end
        # Append a history line.
        ts = _iso8601(now())
        old = haskey(ds.attrib, "history") ? _to_string(ds.attrib["history"]) : ""
        ds.attrib["history"] = old * "\n" * ts * ": modified by Daisho.jl update_cfradial"
    end
    return output_file
end

function _all_field_names(volume::Volume)
    names = String[]
    seen = Set{String}()
    for sweep in volume.sweeps
        for name in keys(sweep.fields)
            if !(name in seen)
                push!(names, name)
                push!(seen, name)
            end
        end
    end
    return names
end

function _update_cfradial1!(ds, volume::Volume, field_names::Vector{String})
    sweep_start = ds["sweep_start_ray_index"][:]
    sweep_end = ds["sweep_end_ray_index"][:]
    n_sweeps = length(sweep_start)
    n_sweeps == length(volume.sweeps) || throw(ErrorException(
        "update_cfradial: sweep count mismatch ($(n_sweeps) in file vs " *
        "$(length(volume.sweeps)) in volume). Use write_cfradial to restructure."))
    n_total_rays = ds.dim["time"]
    n_gates = ds.dim["range"]

    # Validate ray counts per sweep.
    for i in 1:n_sweeps
        s = Int(sweep_start[i]) + 1
        e = Int(sweep_end[i]) + 1
        expected = e - s + 1
        actual = length(volume.sweeps[i].time)
        actual == expected || throw(ErrorException(
            "update_cfradial: sweep $(i) has $(actual) rays but file expects " *
            "$(expected). Use write_cfradial to restructure."))
        length(volume.sweeps[i].range) == n_gates || throw(ErrorException(
            "update_cfradial: sweep $(i) gate count mismatch."))
    end

    for fname in field_names
        big = Array{Float32}(undef, n_total_rays, n_gates)
        big .= Float32(NaN)
        fill_v = Float32(-32768)
        any_present = false
        existing_layout = nothing
        if haskey(ds, fname)
            v = ds[fname]
            existing_layout = String.(NCDatasets.dimnames(v))
            if haskey(v.attrib, "_FillValue")
                fill_v = Float32(v.attrib["_FillValue"])
            end
        end
        for i in 1:n_sweeps
            sweep = volume.sweeps[i]
            haskey(sweep.fields, fname) || continue
            any_present = true
            s = Int(sweep_start[i]) + 1
            e = Int(sweep_end[i]) + 1
            data = sweep.fields[fname].data
            md = sweep.fields[fname].metadata
            if md.fill_value !== nothing
                fill_v = Float32(md.fill_value)
            end
            for r in 1:size(data, 1), g in 1:size(data, 2)
                v = data[r, g]
                big[s + r - 1, g] = (ismissing(v) || (isa(v, AbstractFloat) && isnan(v))) ?
                    fill_v : Float32(v)
            end
        end
        any_present || continue

        if haskey(ds, fname)
            v = ds[fname]
            if existing_layout == ["range", "time"]
                v[:, :] = permutedims(big)
            else
                v[:, :] = big
            end
        else
            # Default to (time, range) for new vars.
            attrs = _field_attrs_for_write(volume, fname; fill_value=fill_v)
            defVar(ds, fname, big, ("time", "range"); attrib = attrs)
        end
    end
    return nothing
end

function _update_cfradial2!(ds, volume::Volume, field_names::Vector{String})
    sweep_group_name = ds["sweep_group_name"][:]
    n_sweeps = length(sweep_group_name)
    n_sweeps == length(volume.sweeps) || throw(ErrorException(
        "update_cfradial: sweep count mismatch ($(n_sweeps) in file vs " *
        "$(length(volume.sweeps)) in volume). Use write_cfradial to restructure."))

    for (i, gname) in enumerate(sweep_group_name)
        gname_str = _to_string(gname)
        haskey(ds.group, gname_str) || throw(ErrorException(
            "update_cfradial: sweep group $(gname_str) missing from file."))
        grp = ds.group[gname_str]
        sweep = volume.sweeps[i]
        n_rays_file = grp.dim["time"]
        n_gates_file = grp.dim["range"]
        n_rays_file == length(sweep.time) || throw(ErrorException(
            "update_cfradial: sweep $(i) ray count mismatch ($(n_rays_file) in file " *
            "vs $(length(sweep.time)) in volume). Use write_cfradial to restructure."))
        n_gates_file == length(sweep.range) || throw(ErrorException(
            "update_cfradial: sweep $(i) gate count mismatch."))
        for fname in field_names
            haskey(sweep.fields, fname) || continue
            fld = sweep.fields[fname]
            md = fld.metadata
            fill_v = md.fill_value === nothing ? Float32(-32768) : Float32(md.fill_value)
            data32 = Array{Float32}(undef, size(fld.data))
            for k in eachindex(fld.data)
                v = fld.data[k]
                data32[k] = (ismissing(v) || (isa(v, AbstractFloat) && isnan(v))) ?
                    fill_v : Float32(v)
            end
            if haskey(grp, fname)
                v = grp[fname]
                dnames = String.(NCDatasets.dimnames(v))
                if dnames == ["range", "time"]
                    v[:, :] = permutedims(data32)
                else
                    v[:, :] = data32
                end
            else
                attrs = _field_attrs_for_write(volume, fname; fill_value=fill_v)
                defVar(grp, fname, data32, ("time", "range"); attrib = attrs)
            end
        end
    end
    return nothing
end

function _field_attrs_for_write(volume::Volume, fname::String; fill_value::Float32)
    attrs = OrderedDict{String,Any}()
    md = nothing
    for sweep in volume.sweeps
        if haskey(sweep.fields, fname)
            md = sweep.fields[fname].metadata
            break
        end
    end
    if md !== nothing
        md.standard_name === nothing || (attrs["standard_name"] = md.standard_name)
        md.long_name === nothing || (attrs["long_name"] = md.long_name)
        md.units === nothing || (attrs["units"] = md.units)
        md.coordinates === nothing || (attrs["coordinates"] = md.coordinates)
        attrs["sampling_ratio"] = Float32(md.sampling_ratio)
    end
    attrs["_FillValue"] = fill_value
    return attrs
end
