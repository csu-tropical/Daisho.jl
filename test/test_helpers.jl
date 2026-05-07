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
