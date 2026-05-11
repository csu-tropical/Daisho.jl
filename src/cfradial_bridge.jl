# Bridge between the new `Volume` model and the legacy `radar` struct.
# Phase B scaffolding — these helpers go away in Phase D once consumers are
# fully migrated.

"""
    as_legacy_radar(volume::Volume; field_names=nothing) -> (radar, field_names)

Flatten a `Volume` into the legacy `radar` struct and return it alongside the
ordered list of canonical field names that match the moments-matrix columns.

When `field_names` is `nothing` (default), columns are laid out in alphabetical
order of the volume's fields (legacy behavior, preserved for backwards
compatibility).

When `field_names` is supplied, columns are laid out in that exact order.
Fields named but not present in any sweep produce a column of `missing`.

Mobile-platform per-ray velocities (`eastward_velocity`, `northward_velocity`,
`vertical_velocity`) are copied from `sweep.georeference` into the legacy
`ew/ns/w_platform` arrays when present; otherwise those arrays are zeros.

Assumes all sweeps share the same `range` axis.
"""
function as_legacy_radar(volume::Volume;
                          field_names::Union{Nothing,Vector{String}}=nothing)
    isempty(volume.sweeps) && throw(ArgumentError("as_legacy_radar: empty volume"))
    range_vec = volume.sweeps[1].range
    for s in volume.sweeps
        length(s.range) == length(range_vec) ||
            throw(ArgumentError("as_legacy_radar: sweeps have differing range lengths; use write_cfradial for non-uniform volumes"))
    end
    n_gates = length(range_vec)
    n_total_rays = sum(n_rays(s) for s in volume.sweeps; init=0)
    field_list = field_names === nothing ? Daisho.field_names(volume) : copy(field_names)
    n_fields = length(field_list)

    azimuth = Vector{Union{Missing,Float32}}(undef, n_total_rays)
    elevation = Vector{Union{Missing,Float32}}(undef, n_total_rays)
    ew_platform = zeros(Float32, n_total_rays)
    ns_platform = zeros(Float32, n_total_rays)
    w_platform = zeros(Float32, n_total_rays)
    nyquist = Vector{Union{Missing,Float32}}(undef, n_total_rays)
    times = Vector{DateTime}(undef, n_total_rays)
    latitudes = Vector{Union{Missing,Float32}}(undef, n_total_rays)
    longitudes = Vector{Union{Missing,Float32}}(undef, n_total_rays)
    altitudes = Vector{Union{Missing,Float32}}(undef, n_total_rays)
    fixed_angles = Vector{Union{Missing,Float32}}(undef, length(volume.sweeps))
    swpstart = Vector{Union{Missing,Float32}}(undef, length(volume.sweeps))
    swpend = Vector{Union{Missing,Float32}}(undef, length(volume.sweeps))
    moments = Array{Union{Missing,Float64}}(undef, n_total_rays * n_gates, n_fields)
    moments .= missing

    field_idx = Dict{String,Int}(name => i for (i, name) in enumerate(field_list))
    cursor = 0
    for (si, sweep) in enumerate(volume.sweeps)
        n = n_rays(sweep)
        rng = (cursor + 1):(cursor + n)
        azimuth[rng] .= Float32.(sweep.azimuth)
        elevation[rng] .= Float32.(sweep.elevation)
        if sweep.nyquist_velocity !== nothing
            nyquist[rng] .= Float32.(sweep.nyquist_velocity)
        else
            nyquist[rng] .= missing
        end
        times[rng] .= sweep.time
        # Stationary platform: copy volume-level lat/lon/alt; mobile uses sweep.georeference.
        if sweep.georeference !== nothing && length(sweep.georeference.latitude) == n
            latitudes[rng] .= Float32.(sweep.georeference.latitude)
            longitudes[rng] .= Float32.(sweep.georeference.longitude)
            altitudes[rng] .= Float32.(sweep.georeference.altitude)
        else
            latitudes[rng] .= Float32(volume.latitude)
            longitudes[rng] .= Float32(volume.longitude)
            altitudes[rng] .= Float32(volume.altitude)
        end
        if sweep.georeference !== nothing
            gr = sweep.georeference
            if gr.eastward_velocity !== nothing && length(gr.eastward_velocity) == n
                ew_platform[rng] .= Float32.(gr.eastward_velocity)
            end
            if gr.northward_velocity !== nothing && length(gr.northward_velocity) == n
                ns_platform[rng] .= Float32.(gr.northward_velocity)
            end
            if gr.vertical_velocity !== nothing && length(gr.vertical_velocity) == n
                w_platform[rng] .= Float32.(gr.vertical_velocity)
            end
        end
        fixed_angles[si] = Float32(sweep.fixed_angle)
        swpstart[si] = Float32(cursor)
        swpend[si] = Float32(cursor + n - 1)

        for (name, fld) in sweep.fields
            haskey(field_idx, name) || continue
            col = field_idx[name]
            data = fld.data
            # Convert NaN sentinel to missing for legacy consumers.
            for r in 1:n, g in 1:n_gates
                v = data[r, g]
                idx = (cursor + r - 1) * n_gates + g
                moments[idx, col] = (ismissing(v) || (isa(v, AbstractFloat) && isnan(v))) ?
                    missing : Float64(v)
            end
        end
        cursor += n
    end

    legacy = radar(
        scan_name = volume.scan_name,
        azimuth = azimuth,
        elevation = elevation,
        ew_platform = ew_platform,
        ns_platform = ns_platform,
        w_platform = w_platform,
        nyquist_velocity = nyquist,
        range = Float32.(range_vec),
        time = times,
        latitude = latitudes,
        longitude = longitudes,
        altitude = altitudes,
        fixed_angles = fixed_angles,
        swpstart = swpstart,
        swpend = swpend,
        moments = moments,
    )
    return legacy, field_list
end

"""
    as_volume(r::radar; field_names::Vector{String}, sweep_modes=nothing,
              instrument_name="", scan_name=r.scan_name,
              field_metadata=Dict{String,FieldMetadata}()) -> Volume

Reverse direction. Reconstruct a `Volume` from a legacy `radar` struct given the
canonical field names corresponding to `r.moments` columns. Sweep boundaries
come from `r.swpstart` / `r.swpend`.
"""
function as_volume(r::radar; field_names::Vector{String},
                   sweep_modes::Union{Nothing,Vector{String}}=nothing,
                   instrument_name::String = "",
                   scan_name::String = r.scan_name,
                   field_metadata::Dict{String,FieldMetadata}=Dict{String,FieldMetadata}())
    n_gates = length(r.range)
    range_vec = collect(Float64, r.range)
    n_sweeps = length(r.swpstart)
    sweeps = SweepGroup[]

    # Convert flat moments index → ray index. moments[(ray-1)*n_gates + gate, col]
    # corresponds to field[ray, gate].
    function _slice_field(col::Int, sweep_start::Int, sweep_end::Int)
        n = sweep_end - sweep_start + 1
        out = Array{Float32}(undef, n, n_gates)
        for rr in 1:n, gg in 1:n_gates
            absolute = sweep_start + rr - 1
            idx = (absolute - 1) * n_gates + gg
            v = r.moments[idx, col]
            out[rr, gg] = ismissing(v) ? Float32(NaN) : Float32(v)
        end
        return out
    end

    for i in 1:n_sweeps
        s_idx = Int(r.swpstart[i]) + 1
        e_idx = Int(r.swpend[i]) + 1
        n = e_idx - s_idx + 1
        rng = s_idx:e_idx
        sweep_mode = sweep_modes === nothing ? "azimuth_surveillance" : sweep_modes[i]
        sweep = SweepGroup(
            sweep_number = i - 1,
            sweep_mode = sweep_mode,
            fixed_angle = Float64(r.fixed_angles[i]),
            time = Vector{DateTime}(r.time[rng]),
            range = range_vec,
            azimuth = collect(Float64, coalesce.(r.azimuth[rng], 0.0)),
            elevation = collect(Float64, coalesce.(r.elevation[rng], 0.0)),
            nyquist_velocity = collect(Float64, coalesce.(r.nyquist_velocity[rng], NaN)),
        )
        for (col, name) in enumerate(field_names)
            data = _slice_field(col, s_idx, e_idx)
            md = get(field_metadata, name, FieldMetadata(units=nothing, fill_value=-32768.0))
            add_field!(sweep, name, data, md)
        end
        push!(sweeps, sweep)
    end

    # Use the first ray as the volume reference position.
    lat0 = isempty(r.latitude) ? 0.0 : Float64(coalesce(r.latitude[1], 0.0))
    lon0 = isempty(r.longitude) ? 0.0 : Float64(coalesce(r.longitude[1], 0.0))
    alt0 = isempty(r.altitude) ? 0.0 : Float64(coalesce(r.altitude[1], 0.0))

    return Volume(
        instrument_name = instrument_name,
        scan_name = scan_name,
        time_coverage_start = isempty(r.time) ? DateTime(0) : r.time[1],
        time_coverage_end = isempty(r.time) ? DateTime(0) : r.time[end],
        latitude = lat0,
        longitude = lon0,
        altitude = alt0,
        sweeps = sweeps,
    )
end
