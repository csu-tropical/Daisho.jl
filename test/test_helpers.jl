# Test helper functions for creating synthetic radar data

using Dates

"""
Create a minimal synthetic radar volume for testing.
Returns a Daisho.radar struct with n_rays rays, n_gates gates, and n_moments moments.
"""
function make_synthetic_radar(;
    n_rays=10, n_gates=20, n_moments=3,
    lat=16.886, lon=-24.988, alt=50.0,
    az_start=0.0, az_step=36.0,
    el=1.0, range_start=400.0, range_step=100.0,
    scan_name="TEST_VOL",
    n_sweeps=1,
    fill_value=0.0
)
    azimuth = Float32.(collect(range(az_start, step=az_step, length=n_rays)))
    elevation = fill(Float32(el), n_rays)
    ew_platform = zeros(Float32, n_rays)
    ns_platform = zeros(Float32, n_rays)
    w_platform = zeros(Float32, n_rays)
    nyquist_velocity = fill(Float32(25.0), n_rays)
    ranges = Float32.(collect(range(range_start, step=range_step, length=n_gates)))
    time_arr = [DateTime(2024, 8, 27, 21, 40, 0) + Second(i) for i in 1:n_rays]
    latitude = fill(Float32(lat), n_rays)
    longitude = fill(Float32(lon), n_rays)
    altitude = fill(Float32(alt), n_rays)

    # For n_sweeps, evenly divide rays
    rays_per_sweep = div(n_rays, n_sweeps)
    fixed_angles = Float32.(fill(el, n_sweeps))
    swpstart = Float32.([Float32((i-1) * rays_per_sweep) for i in 1:n_sweeps])
    swpend = Float32.([Float32(i * rays_per_sweep - 1) for i in 1:n_sweeps])

    # Moments: n_gates * n_rays rows, n_moments columns
    moments = Array{Union{Missing, Float64}}(undef, n_gates * n_rays, n_moments)
    moments .= fill_value

    return Daisho.radar(
        scan_name=scan_name,
        azimuth=azimuth,
        elevation=elevation,
        ew_platform=ew_platform,
        ns_platform=ns_platform,
        w_platform=w_platform,
        nyquist_velocity=nyquist_velocity,
        range=ranges,
        time=time_arr,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        fixed_angles=fixed_angles,
        swpstart=swpstart,
        swpend=swpend,
        moments=moments
    )
end

"""
Create a synthetic moment dictionary mapping names to column indices.
"""
function make_moment_dict(names=["DBZ", "VEL", "WIDTH"])
    return Dict(name => i for (i, name) in enumerate(names))
end

"""
Create a synthetic grid type dictionary.
"""
function make_grid_type_dict(n_moments=3)
    return Dict(i => (i == 1 ? :linear : :weighted) for i in 1:n_moments)
end

"""
Create a synthetic CfRadial NetCDF file for I/O testing.
Returns the file path.
"""
function create_synthetic_cfradial(filepath;
    n_sweeps=2, n_rays=10, n_gates=20,
    moment_names=["DBZ", "VEL", "WIDTH"])

    n_moments = length(moment_names)

    ds = NCDatasets.NCDataset(filepath, "c", attrib = DataStructures.OrderedDict(
        "Conventions"       => "CF/Radial-1.3",
        "Sub_conventions"   => "ARM-1.2",
        "version"           => "1.3",
        "title"             => "Synthetic test data",
        "institution"       => "Test",
        "references"        => "None",
        "source"            => "Synthetic",
        "history"           => "Created for testing",
        "comment"           => "Test file",
        "original_format"   => "SIGMET",
        "driver"            => "NCDatasets",
        "created"           => "2024-08-27T21:40:08Z",
        "start_datetime"    => "2024-08-27T21:40:08Z",
        "time_coverage_start" => "2024-08-27T21:40:08Z",
        "start_time"        => "2024-08-27T21:40:08Z",
        "end_datetime"      => "2024-08-27T21:40:18Z",
        "time_coverage_end" => "2024-08-27T21:40:18Z",
        "end_time"          => "2024-08-27T21:40:18Z",
        "instrument_name"   => "TEST_RADAR",
        "site_name"         => "TEST_SITE",
        "scan_name"         => "TEST_VOL",
        "scan_id"           => Int32(0),
        "platform_is_mobile" => "false",
        "n_gates_vary"      => "false",
        "ray_times_increase" => "true",
    ))

    # Dimensions
    ds.dim["time"] = n_rays
    ds.dim["range"] = n_gates
    ds.dim["sweep"] = n_sweeps
    ds.dim["string_length_8"] = 8
    ds.dim["string_length_32"] = 32
    ds.dim["status_xml_length"] = 1

    # Time variable - use seconds offset from base time
    base_time = DateTime(2024, 8, 27, 21, 40, 0)
    time_offsets = Float64.(0:n_rays-1)
    nctime = defVar(ds, "time", Float64, ("time",), attrib = DataStructures.OrderedDict(
        "standard_name" => "time",
        "units"         => "seconds since 2024-08-27T21:40:00Z",
        "calendar"      => "gregorian",
    ))
    nctime[:] = time_offsets

    # Range
    range_data = Float32.(collect(range(400.0, step=100.0, length=n_gates)))
    ncrange = defVar(ds, "range", Float32, ("range",))
    ncrange[:] = range_data

    # Azimuth
    az_data = Float32.(collect(range(0.0, step=360.0/n_rays, length=n_rays)))
    ncaz = defVar(ds, "azimuth", Float32, ("time",))
    ncaz[:] = az_data

    # Elevation
    el_data = fill(Float32(1.0), n_rays)
    ncel = defVar(ds, "elevation", Float32, ("time",))
    ncel[:] = el_data

    # Nyquist velocity
    ncnyq = defVar(ds, "nyquist_velocity", Float32, ("time",))
    ncnyq[:] = fill(Float32(25.0), n_rays)

    # Latitude, longitude, altitude (scalar for stationary)
    nclat = defVar(ds, "latitude", Float64, ("time",))
    nclat[:] = fill(16.886, n_rays)
    nclon = defVar(ds, "longitude", Float64, ("time",))
    nclon[:] = fill(-24.988, n_rays)
    ncalt = defVar(ds, "altitude", Float64, ("time",))
    ncalt[:] = fill(50.0, n_rays)

    # Sweep info
    rays_per_sweep = div(n_rays, n_sweeps)
    ncfa = defVar(ds, "fixed_angle", Float32, ("sweep",))
    ncfa[:] = Float32.([1.0, 2.0][1:n_sweeps])

    ncss = defVar(ds, "sweep_start_ray_index", Int32, ("sweep",))
    ncss[:] = Int32.([((i-1) * rays_per_sweep) for i in 1:n_sweeps])
    ncse = defVar(ds, "sweep_end_ray_index", Int32, ("sweep",))
    ncse[:] = Int32.([(i * rays_per_sweep - 1) for i in 1:n_sweeps])

    # Moment data
    for (idx, name) in enumerate(moment_names)
        ncvar = defVar(ds, name, Float32, ("range", "time"))
        data = randn(Float32, n_gates, n_rays) .* 10.0
        if name == "DBZ"
            data .+= 20.0  # Typical reflectivity values
        end
        ncvar[:] = data
    end

    close(ds)
    return filepath
end

"""
Build two single-PPI `Volume`s viewing a shared domain from two stationary
radars, for dual-Doppler wind-synthesis tests. Each gate's `VEL` value is the
analytic radial projection of a uniform wind `(u, v)` onto the beam,

    VEL = u·sin(az)·cos(el) + v·cos(az)·cos(el)

(CEDRIC appendix F Eq 1 with W = 0), using the beam's own azimuth/elevation, so
gridding + solve must recover `(u, v)` at well-conditioned points. Radar A sits
south of the grid origin and radar B to the west, giving a ~90° beam-crossing
angle near the origin. Returns `(volA, volB)`.
"""
function make_synthetic_dual_doppler(; u::Real = 8.0, v::Real = -3.0,
        ref_lat::Real = 16.0, ref_lon::Real = -24.0,
        radarA_lat::Real = ref_lat - 0.20, radarA_lon::Real = ref_lon,
        radarB_lat::Real = ref_lat,        radarB_lon::Real = ref_lon - 0.20,
        azimuths = collect(0.0:2.0:358.0),
        elevations = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
        ranges = collect(2000.0:500.0:35000.0))
    function _sweep(elevation, sweep_no, t0)
        n_rays = length(azimuths); n_gates = length(ranges)
        ts = [t0 + Second(i - 1) for i in 1:n_rays]
        az = collect(Float64, azimuths); el = fill(Float64(elevation), n_rays)
        data = Array{Float32}(undef, n_rays, n_gates)
        for r in 1:n_rays
            azr = deg2rad(az[r]); elr = deg2rad(el[r])
            a = sin(azr) * cos(elr); b = cos(azr) * cos(elr)
            vr = Float32(u * a + v * b)
            for gcol in 1:n_gates
                data[r, gcol] = vr
            end
        end
        sweep = SweepGroup(
            sweep_number = sweep_no, sweep_mode = "azimuth_surveillance",
            fixed_angle = Float64(elevation), time = ts,
            range = collect(Float64, ranges), azimuth = az, elevation = el)
        add_field!(sweep, "VEL", data, FieldMetadata(units = "m/s", long_name = "VEL"))
        return sweep
    end
    function _vol(lat, lon, name)
        t0 = DateTime(2024, 1, 1, 0, 0, 0)
        sweeps = [_sweep(elevations[s], s - 1, t0 + Second((s - 1) * length(azimuths)))
                  for s in eachindex(elevations)]
        return Volume(
            instrument_name = name, site_name = name, title = "synthetic",
            institution = "Test", source = "Test", history = "Test",
            time_coverage_start = t0,
            time_coverage_end = sweeps[end].time[end],
            latitude = Float64(lat), longitude = Float64(lon), altitude = 0.0,
            sweeps = sweeps)
    end
    return (_vol(radarA_lat, radarA_lon, "RADARA"),
            _vol(radarB_lat, radarB_lon, "RADARB"))
end

"""
Build a deterministic CfRadial 2.1 `Volume` for round-trip tests, with the
moment fields named in `fields`.
"""
function synthetic_volume(; n_sweeps::Int=2, n_rays::Int=20, n_gates::Int=10,
                          fields = ["DBZ", "VEL", "SQI", "DBZ_QC"],
                          lat::Real=16.886, lon::Real=-24.988, alt::Real=50.0,
                          instrument_name::String="SYN")
    t0 = DateTime(2024, 9, 3, 15, 0, 0)
    sweeps = SweepGroup[]
    for s in 1:n_sweeps
        ts = [t0 + Second((s-1) * n_rays + r) for r in 0:(n_rays-1)]
        sweep = SweepGroup(
            sweep_number = s - 1,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = Float64(s),
            time = ts,
            range = collect(Float64, 400.0:100.0:(400.0 + 100.0 * (n_gates-1))),
            azimuth = collect(Float64, range(0.0, 360.0; length=n_rays+1))[1:n_rays],
            elevation = fill(Float64(s), n_rays),
            nyquist_velocity = fill(25.0, n_rays),
        )
        for fname in fields
            data = Float32.(reshape(1:(n_rays * n_gates), n_rays, n_gates))
            if fname == "DBZ" || fname == "DBZ_QC"
                data .= data .+ 20.0f0
            elseif fname == "SQI"
                data = fill(Float32(0.5), n_rays, n_gates)
            end
            md = FieldMetadata(units = (fname == "DBZ" || fname == "DBZ_QC") ? "dBZ" :
                                       fname == "VEL" ? "m/s" : "",
                               long_name = fname, fill_value = -32768.0)
            add_field!(sweep, fname, data, md)
        end
        push!(sweeps, sweep)
    end
    return Volume(
        instrument_name = instrument_name,
        site_name = instrument_name,
        title = "Synthetic",
        institution = "Test",
        source = "Test",
        history = "Test",
        time_coverage_start = t0,
        time_coverage_end = t0 + Second(n_sweeps * n_rays),
        latitude = Float64(lat), longitude = Float64(lon), altitude = Float64(alt),
        sweeps = sweeps,
    )
end

"""
    build_synthetic_cfradial_v1(path; n_sweeps=2, rays_per_sweep=4, n_gates=5,
                                field_names=["DBZ", "VEL"]) -> path

Write a *fuller* synthetic CfRadial **1.4** NetCDF file (flat top-level layout)
to `path`, in-memory and self-contained, with no large fixtures required. It is
deliberately exhaustive so reader tests exercise the v1 code path
(`_read_cfradial1`, `_read_radar_calibration_v1`,
`_read_georeference_correction_v1`) without the gitignored real fixtures.

The file contains:

  * global attrs (Conventions/version/title/… plus a non-spec attr that must
    land in `extra_attrs`),
  * per-ray `latitude`/`longitude`/`altitude` (mobile layout → per-sweep
    `Georeference` sub-groups are built), `altitude_agl`,
  * `time` (seconds-since with a `units` attr), `range` (with
    `meters_to_center_of_first_gate` / `meters_between_gates` /
    `spacing_is_constant` attrs), `azimuth`, `elevation`,
  * `sweep_start_ray_index` / `sweep_end_ray_index` / `fixed_angle` /
    `sweep_number` and per-sweep **2-D char** mode strings (`sweep_mode`,
    `follow_mode`, `prt_mode`, `polarization_mode`, `rays_are_indexed`),
    `ray_angle_res`, `target_scan_rate`,
  * optional per-ray vars (`pulse_width`, `prt`, `prt_ratio`,
    `nyquist_velocity`, `unambiguous_range`, `n_samples`,
    `antenna_transition`, `scan_rate`, `r_calib_index`, `rx_range_resolution`),
  * scalar string vars (`volume_number`, `platform_type`, `instrument_type`,
    `primary_axis`, `status_str`) and `frequency`,
  * radar-parameter vars (`radar_antenna_gain_h/v`, `radar_beam_width_h/v`,
    `radar_rx_bandwidth`),
  * a radar-calibration table (`r_calib_*` over an `r_calib` dim of length 2,
    including `r_calib_time` and the LROSE-style `r_calib_base_dbz_1km_hc`),
  * volume-level georeference-correction scalars (`*_correction`),
  * `field_names` fields over `(range, time)`. The first field embeds a
    `-32768` fill (→ NaN, true-missing) and a `-9999` value (clear-air; stays
    finite) so the fill/sentinel distinction is exercised.
"""
function build_synthetic_cfradial_v1(path;
        n_sweeps::Int = 2, rays_per_sweep::Int = 4, n_gates::Int = 5,
        field_names = ["DBZ", "VEL"])

    n_rays = n_sweeps * rays_per_sweep
    slen = 32

    ds = NCDatasets.NCDataset(path, "c", attrib = DataStructures.OrderedDict(
        "Conventions"         => "CF/Radial",
        "Sub_conventions"     => "CF-Radial instrument_parameters radar_parameters radar_calibration",
        "version"             => "CF-Radial-1.4",
        "title"               => "Synthetic v1 test volume",
        "institution"         => "Test",
        "source"              => "Synthetic",
        "history"             => "Created for testing",
        "references"          => "None",
        "comment"             => "Full v1 synthetic file",
        "instrument_name"     => "SYNV1",
        "site_name"           => "SYNSITE",
        "scan_name"           => "SYNSCAN",
        "scan_id"             => Int32(7),
        "platform_is_mobile"  => "true",
        "ray_times_increase"  => "true",
        "simulated_data"      => "true",
        "time_coverage_start" => "2024-08-27T21:40:00Z",
        "time_coverage_end"   => "2024-08-27T21:40:08Z",
        "non_spec_attr"       => "absorbed-into-extra_attrs",
    ))

    # Dimensions
    ds.dim["time"]      = n_rays
    ds.dim["range"]     = n_gates
    ds.dim["sweep"]     = n_sweeps
    ds.dim["r_calib"]   = 2
    ds.dim["frequency"] = 1
    ds.dim["string_length"] = slen

    # Pad a string to `slen` chars with NUL (the CF char-array convention;
    # `rpad` rejects '\0' because it has zero textwidth, so do it by hand).
    _nulpad = s -> begin
        c = collect(Char, s)
        append!(c, fill('\0', slen - length(c)))
        c
    end
    # Helper: write a per-sweep mode string as a 2-D char (string_length, sweep).
    _defmode = function (name, strings)
        v = defVar(ds, name, Char, ("string_length", "sweep"))
        for j in 1:n_sweeps
            v[:, j] = _nulpad(strings[j])
        end
        return v
    end
    # Helper: write a 1-D scalar string as a char vector (string_length,).
    _defscalarstr = function (name, s)
        v = defVar(ds, name, Char, ("string_length",))
        v[:] = _nulpad(s)
        return v
    end

    # Time (seconds since base, with units attr)
    nctime = defVar(ds, "time", Float64, ("time",), attrib = DataStructures.OrderedDict(
        "standard_name" => "time",
        "units"         => "seconds since 2024-08-27T21:40:00Z",
        "calendar"      => "gregorian",
    ))
    nctime[:] = Float64.(0:n_rays-1)

    # Range with gate-geometry attrs
    range_data = Float32.(collect(range(400.0, step = 100.0, length = n_gates)))
    ncrange = defVar(ds, "range", Float32, ("range",), attrib = DataStructures.OrderedDict(
        "meters_to_center_of_first_gate" => Float32(400.0),
        "meters_between_gates"           => Float32(100.0),
        "spacing_is_constant"            => "true",
    ))
    ncrange[:] = range_data

    # Per-ray geometry
    az = Float32.(collect(range(0.0, step = 360.0 / n_rays, length = n_rays)))
    defVar(ds, "azimuth", Float32, ("time",))[:] = az
    defVar(ds, "elevation", Float32, ("time",))[:] = fill(Float32(1.5), n_rays)

    # Per-ray platform position (mobile → builds Georeference sub-groups)
    defVar(ds, "latitude", Float64, ("time",))[:]  = fill(16.886, n_rays)
    defVar(ds, "longitude", Float64, ("time",))[:] = fill(-24.988, n_rays)
    defVar(ds, "altitude", Float64, ("time",))[:]  = fill(50.0, n_rays)
    defVar(ds, "altitude_agl", Float64, ())[:]     = 30.0
    defVar(ds, "heading", Float32, ("time",))[:]   = fill(Float32(90.0), n_rays)
    defVar(ds, "roll", Float32, ("time",))[:]      = zeros(Float32, n_rays)
    defVar(ds, "pitch", Float32, ("time",))[:]     = zeros(Float32, n_rays)

    # Optional per-ray instrument vars
    defVar(ds, "pulse_width", Float32, ("time",))[:]        = fill(Float32(1.0e-6), n_rays)
    defVar(ds, "prt", Float32, ("time",))[:]                = fill(Float32(1.0e-3), n_rays)
    defVar(ds, "prt_ratio", Float32, ("time",))[:]          = fill(Float32(1.0), n_rays)
    defVar(ds, "nyquist_velocity", Float32, ("time",))[:]   = fill(Float32(25.0), n_rays)
    defVar(ds, "unambiguous_range", Float32, ("time",))[:]  = fill(Float32(150000.0), n_rays)
    defVar(ds, "n_samples", Int32, ("time",))[:]            = fill(Int32(64), n_rays)
    defVar(ds, "antenna_transition", Int8, ("time",))[:]    = zeros(Int8, n_rays)
    defVar(ds, "scan_rate", Float32, ("time",))[:]          = fill(Float32(10.0), n_rays)
    defVar(ds, "rx_range_resolution", Float32, ("time",))[:] = fill(Float32(100.0), n_rays)
    defVar(ds, "r_calib_index", Int32, ("time",))[:]        = fill(Int32(0), n_rays)

    # Sweep boundaries / per-sweep scalars
    defVar(ds, "sweep_start_ray_index", Int32, ("sweep",))[:] =
        Int32.([(i - 1) * rays_per_sweep for i in 1:n_sweeps])
    defVar(ds, "sweep_end_ray_index", Int32, ("sweep",))[:] =
        Int32.([i * rays_per_sweep - 1 for i in 1:n_sweeps])
    defVar(ds, "fixed_angle", Float32, ("sweep",))[:] =
        Float32.([0.5 + i for i in 1:n_sweeps])
    defVar(ds, "sweep_number", Int32, ("sweep",))[:] = Int32.(0:n_sweeps-1)
    defVar(ds, "ray_angle_res", Float32, ("sweep",))[:] = fill(Float32(1.0), n_sweeps)
    defVar(ds, "target_scan_rate", Float32, ("sweep",))[:] = fill(Float32(10.0), n_sweeps)
    _defmode("sweep_mode", fill("azimuth_surveillance", n_sweeps))
    _defmode("follow_mode", fill("none", n_sweeps))
    _defmode("prt_mode", fill("fixed", n_sweeps))
    _defmode("polarization_mode", fill("horizontal", n_sweeps))
    _defmode("rays_are_indexed", fill("true", n_sweeps))

    # Scalar string / metadata vars
    defVar(ds, "volume_number", Int32, ())[:] = 42
    _defscalarstr("platform_type", "vehicle")
    _defscalarstr("instrument_type", "radar")
    _defscalarstr("primary_axis", "axis_z")
    _defscalarstr("status_str", "<status>ok</status>")
    defVar(ds, "frequency", Float32, ("frequency",))[:] = [Float32(5.6e9)]

    # Radar parameter vars
    defVar(ds, "radar_antenna_gain_h", Float32, ())[:] = 45.0
    defVar(ds, "radar_antenna_gain_v", Float32, ())[:] = 45.0
    defVar(ds, "radar_beam_width_h", Float32, ())[:]   = 1.0
    defVar(ds, "radar_beam_width_v", Float32, ())[:]   = 1.0
    defVar(ds, "radar_rx_bandwidth", Float32, ())[:]   = 1.0e6

    # Radar calibration table (r_calib dim = 2)
    defVar(ds, "r_calib_pulse_width", Float32, ("r_calib",))[:]   = Float32[1.0e-6, 2.0e-6]
    defVar(ds, "r_calib_xmit_power_h", Float32, ("r_calib",))[:]  = Float32[70.0, 70.5]
    defVar(ds, "r_calib_noise_hc", Float32, ("r_calib",))[:]      = Float32[-75.0, -74.5]
    defVar(ds, "r_calib_receiver_gain_hc", Float32, ("r_calib",))[:] = Float32[40.0, 40.2]
    defVar(ds, "r_calib_zdr_correction", Float32, ("r_calib",))[:] = Float32[0.1, 0.2]
    # LROSE-style name that canonicalizes to base_1km_hc
    defVar(ds, "r_calib_base_dbz_1km_hc", Float32, ("r_calib",))[:] = Float32[-30.0, -29.0]
    rct = defVar(ds, "r_calib_time", String, ("r_calib",))
    rct[:] = ["2024-08-27T21:40:00Z", "2024-08-27T21:40:04Z"]

    # Volume-level georeference correction scalars
    defVar(ds, "azimuth_correction", Float32, ())[:]   = 0.5
    defVar(ds, "elevation_correction", Float32, ())[:] = -0.25
    defVar(ds, "range_correction", Float32, ())[:]     = 0.0
    defVar(ds, "roll_correction", Float32, ())[:]      = 0.1

    # Field variables over (range, time). First field embeds fill/sentinels.
    for (fi, name) in enumerate(field_names)
        attrib = DataStructures.OrderedDict{String,Any}(
            "_FillValue" => Float32(-32768),
            "long_name"  => name,
            "units"      => name == "DBZ" ? "dBZ" : name == "VEL" ? "m/s" : "",
        )
        v = defVar(ds, name, Float32, ("range", "time"), attrib = attrib)
        data = Float32.(reshape(1:(n_gates * n_rays), n_gates, n_rays)) .+ Float32(fi * 0.5)
        if fi == 1
            data[1, 1] = -32768.0f0   # true-missing → NaN
            data[2, 1] = -9999.0f0    # clear-air/undetect → stays finite
        end
        v[:, :] = data
    end

    close(ds)
    return path
end

"""
    build_synthetic_cfradial_v2(path; n_sweeps=2, rays_per_sweep=4, n_gates=5,
                                field_names=["DBZ", "VEL"]) -> path

Write a *fuller* synthetic CfRadial **2.1** NetCDF file (NetCDF4 GROUP layout,
one group per sweep) to `path`, in-memory and self-contained, with no large
fixtures required. It is the v2 analog of `build_synthetic_cfradial_v1` and is
deliberately exhaustive so reader tests exercise the v2 code path
(`_read_cfradial2`, `_read_sweep_v2`, `_read_radar_parameters`,
`_read_radar_monitoring`, `_read_calibration`, `_read_georeference_correction`)
without the gitignored real fixtures.

The file contains:

  * global attrs (Conventions=`Cf/Radial-2.1`/version/title/… plus a non-spec
    attr that must land in `extra_attrs`),
  * scalar root vars (`latitude`/`longitude`/`altitude`/`altitude_agl`,
    `volume_number`, `platform_type`, `instrument_type`, `primary_axis`,
    `status_str`) and `time_coverage_start`/`_end`,
  * `sweep_group_name` / `sweep_fixed_angle` over a `sweep` dim,
  * root sub-groups `radar_parameters`, `radar_calibration` (`calib` dim = 2,
    incl. `time`), and `georeference_correction`,
  * one NetCDF GROUP per sweep, each with `time` (seconds-since, with `units`)
    and `range` dims/vars, `azimuth`/`elevation`, per-sweep `sweep_number` /
    `sweep_mode` / `sweep_fixed_angle` / `follow_mode` / `prt_mode` /
    `polarization_mode` / `rays_are_indexed` / `ray_angle_resolution` /
    `target_scan_rate`, per-sweep `start_range` / `ray_gate_spacing` (gate
    geometry), per-sweep `frequency`, optional per-ray `nyquist_velocity` /
    `pulse_width`,
  * a per-sweep `georeference` sub-group (lat/lon/alt/heading + `georefs_applied`)
    and a `radar_monitoring` sub-group (`measured_transmit_power_h`, `zdr_offset`),
  * `field_names` fields over `(time, range)`, each carrying a non-spec field
    attribute that must land in the field metadata `extra_attrs`. The first
    field on the first sweep embeds a `-32768` fill (→ NaN, true-missing) and a
    `-9999` value (clear-air; stays finite) so the fill/sentinel distinction is
    exercised.

The per-sweep `georeference` and `radar_monitoring` sub-groups are both surfaced
on read into `SweepGroup.georeference` / `SweepGroup.radar_monitoring`.
"""
function build_synthetic_cfradial_v2(path;
        n_sweeps::Int = 2, rays_per_sweep::Int = 4, n_gates::Int = 5,
        field_names = ["DBZ", "VEL"])

    sweep_names = [string("sweep_", lpad(i, 4, '0')) for i in 1:n_sweeps]

    ds = NCDatasets.NCDataset(path, "c", format = :netcdf4,
        attrib = DataStructures.OrderedDict(
            "Conventions"         => "Cf/Radial-2.1",
            "version"             => "2.1",
            "title"               => "Synthetic v2 test volume",
            "institution"         => "Test",
            "source"              => "Synthetic",
            "history"             => "Created for testing",
            "references"          => "None",
            "comment"             => "Full v2 synthetic file",
            "instrument_name"     => "SYNV2",
            "site_name"           => "SYNSITE2",
            "scan_name"           => "SYNSCAN2",
            "scan_id"             => Int32(9),
            "platform_is_mobile"  => "false",
            "ray_times_increase"  => "true",
            "simulated_data"      => "true",
            "non_spec_attr"       => "absorbed-into-extra_attrs",
        ))

    # Scalar root vars (0-dim).
    NCDatasets.defVar(ds, "time_coverage_start", "2024-09-03T15:00:00Z", ())
    NCDatasets.defVar(ds, "time_coverage_end", "2024-09-03T15:00:08Z", ())
    NCDatasets.defVar(ds, "latitude", 16.886, ())
    NCDatasets.defVar(ds, "longitude", -24.988, ())
    NCDatasets.defVar(ds, "altitude", 50.0, ())
    NCDatasets.defVar(ds, "altitude_agl", 30.0, ())
    NCDatasets.defVar(ds, "volume_number", Int32(42), ())
    NCDatasets.defVar(ds, "platform_type", "fixed", ())
    NCDatasets.defVar(ds, "instrument_type", "radar", ())
    NCDatasets.defVar(ds, "primary_axis", "axis_z", ())
    NCDatasets.defVar(ds, "status_str", "<status>ok</status>", ())

    # Sweep index vars.
    ds.dim["sweep"] = n_sweeps
    NCDatasets.defVar(ds, "sweep_group_name", sweep_names, ("sweep",))
    NCDatasets.defVar(ds, "sweep_fixed_angle",
        Float32.([0.5 + i for i in 1:n_sweeps]), ("sweep",))

    # Root sub-group: radar_parameters.
    rp = NCDatasets.defGroup(ds, "radar_parameters")
    NCDatasets.defVar(rp, "radar_antenna_gain_h", Float32(45.0), ())
    NCDatasets.defVar(rp, "radar_antenna_gain_v", Float32(45.0), ())
    NCDatasets.defVar(rp, "radar_beam_width_h", Float32(1.0), ())
    NCDatasets.defVar(rp, "radar_beam_width_v", Float32(1.0), ())
    NCDatasets.defVar(rp, "radar_rx_bandwidth", Float32(1.0e6), ())

    # Root sub-group: radar_calibration (calib dim = 2, with time).
    rc = NCDatasets.defGroup(ds, "radar_calibration")
    rc.dim["calib"] = 2
    NCDatasets.defVar(rc, "pulse_width", Float32[1.0e-6, 2.0e-6], ("calib",))
    NCDatasets.defVar(rc, "xmit_power_h", Float32[70.0, 70.5], ("calib",))
    NCDatasets.defVar(rc, "noise_hc", Float32[-75.0, -74.5], ("calib",))
    NCDatasets.defVar(rc, "receiver_gain_hc", Float32[40.0, 40.2], ("calib",))
    NCDatasets.defVar(rc, "zdr_correction", Float32[0.1, 0.2], ("calib",))
    NCDatasets.defVar(rc, "time",
        ["2024-09-03T15:00:00Z", "2024-09-03T15:00:04Z"], ("calib",))

    # Root sub-group: georeference_correction (volume-level scalars).
    gc = NCDatasets.defGroup(ds, "georeference_correction")
    NCDatasets.defVar(gc, "azimuth_correction", 0.5, ())
    NCDatasets.defVar(gc, "elevation_correction", -0.25, ())
    NCDatasets.defVar(gc, "roll_correction", 0.1, ())

    range_data = Float32.(collect(range(400.0, step = 100.0, length = n_gates)))

    for (si, gname) in enumerate(sweep_names)
        grp = NCDatasets.defGroup(ds, gname)
        grp.dim["time"] = rays_per_sweep
        grp.dim["range"] = n_gates
        grp.dim["frequency"] = 1

        NCDatasets.defVar(grp, "sweep_number", Int32(si - 1), ())
        NCDatasets.defVar(grp, "sweep_mode", "azimuth_surveillance", ())
        NCDatasets.defVar(grp, "sweep_fixed_angle", Float32(0.5 + si), ())
        NCDatasets.defVar(grp, "follow_mode", "none", ())
        NCDatasets.defVar(grp, "prt_mode", "fixed", ())
        NCDatasets.defVar(grp, "polarization_mode", "horizontal", ())
        NCDatasets.defVar(grp, "rays_are_indexed", "true", ())
        NCDatasets.defVar(grp, "ray_angle_resolution", Float32(1.0), ())
        NCDatasets.defVar(grp, "target_scan_rate", Float32(10.0), ())
        NCDatasets.defVar(grp, "frequency", Float32[5.6e9], ("frequency",))

        NCDatasets.defVar(grp, "time", Float64.(0:rays_per_sweep-1), ("time",),
            attrib = DataStructures.OrderedDict(
                "standard_name" => "time",
                "units"         => "seconds since 2024-09-03T15:00:00Z",
                "calendar"      => "gregorian",
            ))
        NCDatasets.defVar(grp, "range", range_data, ("range",))
        # Gate geometry via per-sweep scalars (start_range / ray_gate_spacing).
        NCDatasets.defVar(grp, "start_range", Float32(400.0), ())
        NCDatasets.defVar(grp, "ray_gate_spacing", Float32(100.0), ())

        NCDatasets.defVar(grp, "azimuth",
            Float32.(collect(range(0.0, step = 90.0, length = rays_per_sweep))),
            ("time",))
        NCDatasets.defVar(grp, "elevation",
            fill(Float32(0.5 + si), rays_per_sweep), ("time",))
        NCDatasets.defVar(grp, "nyquist_velocity",
            fill(Float32(25.0), rays_per_sweep), ("time",))
        NCDatasets.defVar(grp, "pulse_width",
            fill(Float32(1.0e-6), rays_per_sweep), ("time",))

        # Per-sweep georeference sub-group.
        geo = NCDatasets.defGroup(grp, "georeference")
        geo.dim["time"] = rays_per_sweep
        NCDatasets.defVar(geo, "latitude", fill(16.886, rays_per_sweep), ("time",))
        NCDatasets.defVar(geo, "longitude", fill(-24.988, rays_per_sweep), ("time",))
        NCDatasets.defVar(geo, "altitude", fill(50.0, rays_per_sweep), ("time",))
        NCDatasets.defVar(geo, "heading", fill(Float32(90.0), rays_per_sweep), ("time",))
        NCDatasets.defVar(geo, "georefs_applied",
            ones(Int8, rays_per_sweep), ("time",))

        # Per-sweep radar_monitoring sub-group.
        mon = NCDatasets.defGroup(grp, "radar_monitoring")
        mon.dim["time"] = rays_per_sweep
        NCDatasets.defVar(mon, "measured_transmit_power_h",
            fill(Float32(70.0), rays_per_sweep), ("time",))
        NCDatasets.defVar(mon, "zdr_offset",
            fill(Float32(0.1), rays_per_sweep), ("time",))

        # Field vars over (time, range). First field on sweep 1 embeds sentinels.
        for (fi, name) in enumerate(field_names)
            attrib = DataStructures.OrderedDict{String,Any}(
                "_FillValue"        => Float32(-32768),
                "long_name"         => name,
                "units"             => name == "DBZ" ? "dBZ" : name == "VEL" ? "m/s" : "",
                "custom_field_attr" => "keepme",   # non-spec → field extra_attrs
            )
            v = NCDatasets.defVar(grp, name, Float32, ("time", "range"), attrib = attrib)
            data = Float32.(reshape(1:(rays_per_sweep * n_gates),
                                    rays_per_sweep, n_gates)) .+ Float32(fi * 0.5)
            if fi == 1 && si == 1
                data[1, 1] = -32768.0f0   # true-missing → NaN
                data[1, 2] = -9999.0f0    # clear-air/undetect → stays finite
            end
            v[:, :] = data
        end
    end

    close(ds)
    return path
end
