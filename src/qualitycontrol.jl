# Quality Control functions

"""
    fix_SEAPOL_RHOHV!(volume, moment_dict)

Correct artificially high RHOHV values in SEA-POL radar data caused by noise subtraction.

The LROSE conversion adds a bias of 128 and shifts by -32768 to the RHOHV field stored in
`UNKNOWN_ID_82`. This function recovers the original RHOHV by adding 32640, then rescaling
to the range [0, 1] via `(value - 1) / 65533`.

# Arguments
- `volume`: Radar volume object containing a `moments` matrix.
- `moment_dict::Dict{String, Int}`: Dictionary mapping moment names to column indices in `volume.moments`.

# Returns
Nothing. The `volume.moments` RHOHV column is modified in place.
"""
function fix_SEAPOL_RHOHV!(volume, moment_dict)

    # SEAPOL RHOHV has artificially high RHOHV due to noise subtraction
    # have to get original RHOHV from this variable
    # per Brenda J: LROSE adds bias of 128 to field, then shifts by -32768
    # need to add 32640 to field to return to original values (10.19.22)
    # diff from Brody's original SEA-POL code (LROSE conversion difference)
    volume.moments[:,moment_dict["RHOHV"]] .= volume.moments[:,moment_dict["UNKNOWN_ID_82"]]
    volume.moments[:,moment_dict["RHOHV"]] .+= 32640
    volume.moments[:,moment_dict["RHOHV"]] .= (volume.moments[:,moment_dict["RHOHV"]].-1) ./65533.0

end

"""
    threshold_qc!(sweep::SweepGroup, p::DaishoParameters;
                  inputs=["DBZ","VEL","ZDR","KDP","RHOHV","WIDTH","PHIDP"],
                  out_suffix="_QC")

Sweep-aware quality-control. For each input field present in `sweep.fields`,
add a thresholded copy named `<input>*out_suffix`. Gates are set to `NaN`
where any of:
- `SQI < p.qc.sqi_threshold`
- `SNR < p.qc.snr_threshold`
- `RHOHV < p.qc.rhohv_threshold`
- `WIDTH > p.qc.spectrum_width_max`

The output field metadata is the input metadata plus `is_quality_field=true`,
`qualified_variables=[input_name]`, and `long_name="QC " * <input long_name>`.
The input field's `ancillary_variables` is updated to include the new output.
Returns `sweep`.
"""
function threshold_qc!(sweep::SweepGroup, p::DaishoParameters;
                       inputs::Vector{String} = ["DBZ", "VEL", "ZDR", "KDP",
                                                  "RHOHV", "WIDTH", "PHIDP"],
                       out_suffix::String = "_QC")
    # Gather threshold fields. If a threshold field is absent, skip that test.
    sqi  = haskey(sweep.fields, "SQI") ? sweep.fields["SQI"].data : nothing
    snr  = haskey(sweep.fields, "SNR") ? sweep.fields["SNR"].data : nothing
    rhohv = haskey(sweep.fields, "RHOHV") ? sweep.fields["RHOHV"].data : nothing
    width = haskey(sweep.fields, "WIDTH") ? sweep.fields["WIDTH"].data : nothing

    sqi_t  = p.qc.sqi_threshold
    snr_t  = p.qc.snr_threshold
    rh_t   = p.qc.rhohv_threshold
    sw_max = p.qc.spectrum_width_max

    for input in inputs
        haskey(sweep.fields, input) || continue
        in_field = sweep.fields[input]
        in_data = in_field.data
        out_data = similar(in_data, Float32)
        @inbounds for i in eachindex(in_data)
            v = in_data[i]
            keep = !(ismissing(v) || (isa(v, AbstractFloat) && isnan(v)))
            if keep && sqi !== nothing
                s = sqi[i]
                keep = !ismissing(s) && !(isa(s, AbstractFloat) && isnan(s)) && s >= sqi_t
            end
            if keep && snr !== nothing
                sn = snr[i]
                keep = !ismissing(sn) && !(isa(sn, AbstractFloat) && isnan(sn)) && sn >= snr_t
            end
            if keep && rhohv !== nothing
                rh = rhohv[i]
                keep = !ismissing(rh) && !(isa(rh, AbstractFloat) && isnan(rh)) && rh >= rh_t
            end
            if keep && width !== nothing
                w = width[i]
                keep = !ismissing(w) && !(isa(w, AbstractFloat) && isnan(w)) && w <= sw_max
            end
            out_data[i] = keep ? Float32(v) : Float32(NaN)
        end
        in_md = in_field.metadata
        new_long = in_md.long_name === nothing ? "QC " * input : "QC " * in_md.long_name
        out_md = FieldMetadata(
            standard_name = in_md.standard_name,
            long_name = new_long,
            units = in_md.units,
            fill_value = in_md.fill_value,
            undetect = in_md.undetect,
            scale_factor = in_md.scale_factor,
            add_offset = in_md.add_offset,
            coordinates = in_md.coordinates,
            sampling_ratio = in_md.sampling_ratio,
            is_discrete = in_md.is_discrete,
            field_folds = in_md.field_folds,
            fold_limit_lower = in_md.fold_limit_lower,
            fold_limit_upper = in_md.fold_limit_upper,
            is_quality_field = true,
            qualified_variables = [input],
            ancillary_variables = String[],
            flag_values = in_md.flag_values,
            flag_meanings = in_md.flag_meanings,
            flag_masks = in_md.flag_masks,
            thresholding_xml = in_md.thresholding_xml,
            legend_xml = in_md.legend_xml,
            extra_attrs = Dict{String,Any}(),
        )
        out_name = input * out_suffix
        sweep.fields[out_name] = Field(out_data, out_md)

        # Append the new output name to the input's ancillary_variables.
        if !(out_name in in_md.ancillary_variables)
            new_anc = push!(copy(in_md.ancillary_variables), out_name)
            sweep.fields[input] = Field(in_data, FieldMetadata(
                standard_name = in_md.standard_name,
                long_name = in_md.long_name,
                units = in_md.units,
                fill_value = in_md.fill_value,
                undetect = in_md.undetect,
                scale_factor = in_md.scale_factor,
                add_offset = in_md.add_offset,
                coordinates = in_md.coordinates,
                sampling_ratio = in_md.sampling_ratio,
                is_discrete = in_md.is_discrete,
                field_folds = in_md.field_folds,
                fold_limit_lower = in_md.fold_limit_lower,
                fold_limit_upper = in_md.fold_limit_upper,
                is_quality_field = in_md.is_quality_field,
                qualified_variables = in_md.qualified_variables,
                ancillary_variables = new_anc,
                flag_values = in_md.flag_values,
                flag_meanings = in_md.flag_meanings,
                flag_masks = in_md.flag_masks,
                thresholding_xml = in_md.thresholding_xml,
                legend_xml = in_md.legend_xml,
                extra_attrs = in_md.extra_attrs,
            ))
        end
    end
    return sweep
end

"""
    smooth_sqi(sqi)

Smooth the Signal Quality Index (SQI) field using a Gaussian kernel to help remove second-trip echo.

!!! warning
    This function is not yet implemented and will throw an error if called.

# Arguments
- `sqi`: SQI data array to be smoothed.

# Returns
Not yet implemented.
"""
function smooth_sqi(sqi)

    # Smooth the SQI with a Gaussian kernel
    # This function helps to remove second trip echo
    error("smooth_sqi is not yet implemented")
end

"""
    despeckle(speckle, moments, moment_dict, n_gates, n_rays) -> Matrix

Remove small isolated features (speckle) from radar moment data along each radial (range direction).

For each ray, contiguous non-missing segments shorter than or equal to `speckle` gates are
set to `missing` across all moment fields.

# Arguments
- `speckle::Int`: Maximum feature size (in gates) to be considered speckle and removed. Must be > 0.
- `moments::Matrix`: Radar moment data matrix (gates x moments) to be despecked.
- `moment_dict::Dict{String, Int}`: Dictionary mapping moment names to column indices.
- `n_gates::Int`: Number of range gates per ray.
- `n_rays::Int`: Number of rays (azimuths) in the sweep.

# Returns
- `Matrix`: The modified `moments` matrix with speckle features set to `missing`.
"""
function despeckle(speckle, moments, moment_dict, n_gates, n_rays)

    # Despeckle a volume of data
    0 < speckle || error("Speckle definition must be greater than 0")
    n_moments = length(moment_dict)
    for m in 1:n_moments
        moment_data = reshape(moments[:,m],n_gates,n_rays)
        for i in 1:n_rays
            raydata = moment_data[:,i]
            n = 1
            while n < n_gates
                if !ismissing(raydata[n])
                    # Found a good gate, go forward to look for missing
                    feature_size = 0
                    s = n
                    while (n < n_gates) && !ismissing(raydata[n])
                        n += 1
                        feature_size += 1
                    end

                    if feature_size > speckle
                        # Feature is larger than a speckle
                        continue
                    end

                    while (s < n_gates) && (s <= n)
                        moment_data[s,i] = missing
                        s += 1
                    end
                else
                    n += 1
                end
            end
        end
        moments[:,m] .= moment_data[:]
    end

    return moments

end

"""
    despeckle_azimuthal(speckle, moments, moment_dict, n_gates, n_rays) -> Matrix

Remove small isolated features (speckle) from radar moment data along the azimuthal direction.

For each range gate, contiguous non-missing segments shorter than or equal to `speckle` rays are
set to `missing` across all moment fields. This is the azimuthal counterpart to [`despeckle`](@ref).

# Arguments
- `speckle::Int`: Maximum feature size (in rays) to be considered speckle and removed. Must be > 0.
- `moments::Matrix`: Radar moment data matrix (gates x moments) to be despeckled.
- `moment_dict::Dict{String, Int}`: Dictionary mapping moment names to column indices.
- `n_gates::Int`: Number of range gates per ray.
- `n_rays::Int`: Number of rays (azimuths) in the sweep.

# Returns
- `Matrix`: The modified `moments` matrix with azimuthal speckle features set to `missing`.
"""
function despeckle_azimuthal(speckle, moments, moment_dict, n_gates, n_rays)

    # Despeckle a volume of data in the azimuthal direction
    0 < speckle || error("Speckle definition must be greater than 0")
    n_moments = length(moment_dict)
    for m in 1:n_moments
        moment_data = reshape(moments[:,m],n_gates,n_rays)
        for i in 1:n_gates
            ringdata = moment_data[i,:]
            n = 1
            while n < n_rays
                if !ismissing(ringdata[n])
                    # Found a good gate, go forward to look for missing
                    feature_size = 0
                    s = n
                    while (n < n_rays) && !ismissing(ringdata[n])
                        n += 1
                        feature_size += 1
                    end

                    if feature_size > speckle
                        # Feature is larger than a speckle
                        continue
                    end

                    while (s < n_rays) && (s <= n)
                        moment_data[i,s] = missing
                        s += 1
                    end
                else
                    n += 1
                end
            end
        end
        moments[:,m] .= moment_data[:]
    end

    return moments

end

"""
    stddev_phidp_threshold(moments, moment_dict, n_gates, n_rays, window=11, threshold=12) -> Matrix

Apply quality control by removing gates where the local standard deviation of differential
phase (PHIDP) exceeds a threshold, indicating noisy or non-meteorological echoes.

A sliding window of size `window` gates is used to compute the standard deviation of PHIDP
along each ray. Negative PHIDP values are wrapped by adding 360 degrees before computation.
All moment fields at gates exceeding the threshold are set to `missing`.

# Arguments
- `moments::Matrix`: Radar moment data matrix (gates x moments).
- `moment_dict::Dict{String, Int}`: Dictionary mapping moment names to column indices (must contain `"PHIDP"`).
- `n_gates::Int`: Number of range gates per ray.
- `n_rays::Int`: Number of rays (azimuths) in the sweep.
- `window::Int`: Size of the sliding window in gates (must be odd, default `11`).
- `threshold::Real`: Standard deviation threshold in degrees (default `12`).

# Returns
- `Matrix`: The modified `moments` matrix with noisy PHIDP gates set to `missing`.
"""
function stddev_phidp_threshold(moments, moment_dict, n_gates, n_rays, window = 11, threshold = 12)

    # Use a standard deviation of phidp to threshold
    0 < window || error("Window definition must be greater than 0")
    window % 2 == 1 || error("Window must be an odd number")
    window = Int((window - 1)/2)
    phidp = ifelse.(isless.(moments[:,moment_dict["PHIDP"]],0.0), moments[:,moment_dict["PHIDP"]].+360.0, moments[:,moment_dict["PHIDP"]])
    phidp = reshape(phidp,n_gates,n_rays)
    phidp_mask = falses(size(phidp))
    for i in 1:n_rays
        raydata = phidp[:,i]
        for n = 1:n_gates
            stddev = std(raydata[max(n-window,1):min(n+window,n_gates)])
            if isless(threshold, stddev)
                phidp_mask[n,i] = true
            end
        end
    end

    # Apply the mask
    for m in 1:length(moment_dict)
        moments[phidp_mask[:],m] .= missing
    end

    return moments

end

"""
    remove_platform_motion!(volume, moments, moment_dict) -> Matrix

Remove the contribution of platform motion from the Doppler velocity field for ship- or
aircraft-mounted radars.

The platform radial velocity is computed from the east-west, north-south, and vertical
platform velocity components projected onto each beam's azimuth and elevation angles,
then added back to the measured velocity. The corrected velocity is folded back into the
Nyquist interval.

# Arguments
- `volume`: Radar volume object containing `range`, `azimuth`, `elevation`, `ew_platform`, `ns_platform`, `w_platform`, and `nyquist_velocity` fields.
- `moments::Matrix`: Radar moment data matrix (gates x moments).
- `moment_dict::Dict{String, Int}`: Dictionary mapping moment names to column indices (must contain `"VEL"`).

# Returns
- `Matrix`: The modified `moments` matrix with platform-motion-corrected velocities.
"""
function remove_platform_motion!(volume, moments, moment_dict)

    # Get rid of platform motion
    n_gates = length(volume.range)
    n_rays =  length(volume.azimuth)
    vel = reshape(moments[:,moment_dict["VEL"]],n_gates,n_rays)

    # Get the platform motion
    for i in 1:n_rays
        az = volume.azimuth[i] * pi/180.0
        el = volume.elevation[i] * pi/180.0
        ew_platform = volume.ew_platform[i]
        ns_platform = volume.ns_platform[i]
        w_platform = volume.w_platform[i]
        nyquist_velocity = volume.nyquist_velocity[i]
        platform_vr = ew_platform * cos(el) * sin(az) + ns_platform * cos(el) * cos(az) + w_platform * sin(el)
        vel[:,i] .= vel[:,i] .+ platform_vr
        println("Ray: ", i, " Platform velocity: ", platform_vr)
        # Put in the Nyquist interval
        vel[:,i] = ifelse.(isless.(vel[:,i], -nyquist_velocity), vel[:,i] .+ 2.0 .* nyquist_velocity, vel[:,i])
        vel[:,i] = ifelse.(isless.(nyquist_velocity, vel[:,i]), vel[:,i] .- 2.0 .* nyquist_velocity, vel[:,i])
    end

    # Put the velocity back in the volume
    moments[:,moment_dict["VEL"]] .= vel[:]
    return moments

end

"""
    threshold_dbz(volume, raw_moment_dict, qc_moments, qc_moment_dict, dbz_threshold, vel_threshold, sw_threshold) -> Matrix

Remove likely ground clutter by identifying gates with high reflectivity, low velocity,
and narrow spectrum width below 500 m height.

Gates exceeding `dbz_threshold` that also have absolute velocity at or below `vel_threshold`
and spectrum width at or below `sw_threshold` at heights below 500 m are classified as clutter
and set to `missing`. The `"SQI_FOR_MASK"` and `"PID_FOR_QC"` fields are excluded from removal.

# Arguments
- `volume`: Radar volume object containing `moments` and metadata needed by `get_beam_info`.
- `raw_moment_dict::Dict{String, Int}`: Dictionary mapping raw moment names to column indices (must contain `"DBZ"`, `"VEL"`, `"WIDTH"`).
- `qc_moments::Matrix`: QC radar moment data matrix to be modified (gates x moments).
- `qc_moment_dict::Dict{String, Int}`: Dictionary mapping QC moment names to column indices.
- `dbz_threshold::Real`: Reflectivity threshold in dBZ above which gates are examined.
- `vel_threshold::Real`: Maximum absolute velocity (m/s) for a gate to be classified as clutter.
- `sw_threshold::Real`: Maximum spectrum width (m/s) for a gate to be classified as clutter.

# Returns
- `Matrix`: The modified `qc_moments` matrix with likely clutter gates set to `missing`.
"""
function threshold_dbz(volume, raw_moment_dict, qc_moments, qc_moment_dict,
    dbz_threshold, vel_threshold, sw_threshold)

    count = 0
    bad_count = 0
    real_count = 0
    raw_moments = volume.moments
    beam_info = get_beam_info(volume)
    # Check for dBZ values greater than the threshold
    for i in 1:size(raw_moments,1)
        if ismissing(raw_moments[i,raw_moment_dict["DBZ"]])
            continue
        end
        height = beam_info[i,4]
        count += 1
        if (raw_moments[i,raw_moment_dict["DBZ"]] >= dbz_threshold)

            if (abs(raw_moments[i,raw_moment_dict["VEL"]]) <= vel_threshold
                && raw_moments[i,raw_moment_dict["WIDTH"]] <= sw_threshold
                && height < 500.0)
            #if height < 500.0
                #raw_moments[i,raw_moment_dict["KDP"]] > 0.0 || raw_moments[i,raw_moment_dict["HID_CSU"]] < 7
                bad_count += 1
                #println("Likely clutter at $i  exceeding $(dbz_threshold) at height $(height) m")
                #println(beam_info[i,:])
                for key in keys(qc_moment_dict)
                    if key == "SQI_FOR_MASK" || key == "PID_FOR_QC"
                        # Don't QC this field
                        continue
                    end
                    #println("$key: $(raw_moments[i,raw_moment_dict[key]])")
                    qc_moments[i,:] .= missing
                end
            else
                real_count += 1
                println("Maybe real at $i  exceeding $(dbz_threshold) at height $(height) m")
                println(beam_info[i,:])
                for key in keys(raw_moment_dict)
                    println("$key: $(raw_moments[i,raw_moment_dict[key]])")
                 end
            end
        end
    end

    println("Total gates: ", count, ", Bad gates: ", bad_count, ", Real gates: ", real_count)
    if count > 0
        println(" Bad Percentage: ", round(bad_count/count * 100, digits=2), "%", ", Real Percentage: ", round(real_count/count * 100, digits=2), "%")
    end

#    for key in keys(qc_moment_dict)
#
#        if key == "SQI" || key == "PHIDP"
#            # Don't QC this field
#            continue
#        end
#
#        qc_moments[:,qc_moment_dict[key]] = ifelse.(isless.(raw_moments[:,raw_moment_dict["DBZ"]],dbz_threshold),
#            qc_moments[:,qc_moment_dict[key]], missing)
#        println("Thresholding ", key, " with dBZ threshold: ", dbz_threshold)
#
#    end

    return qc_moments

end

"""
    threshold_height(volume, raw_moment_dict, qc_moments, qc_moment_dict, height_threshold) -> Matrix

Remove radar gates located below a specified height threshold.

All QC moment fields at gates with beam height below `height_threshold` are set to `missing`.
The `"SQI_FOR_MASK"` and `"PID_FOR_QC"` fields are excluded from removal. Gates where DBZ
is already `missing` are skipped.

# Arguments
- `volume`: Radar volume object containing `moments` and metadata needed by `get_beam_info`.
- `raw_moment_dict::Dict{String, Int}`: Dictionary mapping raw moment names to column indices (must contain `"DBZ"`).
- `qc_moments::Matrix`: QC radar moment data matrix to be modified (gates x moments).
- `qc_moment_dict::Dict{String, Int}`: Dictionary mapping QC moment names to column indices.
- `height_threshold::Real`: Minimum beam height in meters; gates below this are removed.

# Returns
- `Matrix`: The modified `qc_moments` matrix with low-height gates set to `missing`.
"""
function threshold_height(volume, raw_moment_dict, qc_moments, qc_moment_dict,
    height_threshold)

    raw_moments = volume.moments
    beam_info = get_beam_info(volume)
    # Check for height values lower than the threshold
    for i in 1:size(raw_moments,1)
        if ismissing(raw_moments[i,raw_moment_dict["DBZ"]])
            continue
        end
        height = beam_info[i,4]
        if (height < height_threshold)
            for key in keys(qc_moment_dict)
                if key == "SQI_FOR_MASK" || key == "PID_FOR_QC"
                    # Don't QC this field
                    continue
                end
                qc_moments[i,qc_moment_dict[key]] = missing
            end
        end
    end
    return qc_moments

end

"""
    threshold_terrain_height(volume, raw_moment_dict, qc_moments, qc_moment_dict, terrain_threshold, tile_dir) -> Matrix

Remove radar gates located over terrain exceeding a specified elevation threshold using SRTM
digital terrain model data.

For each gate, the geographic coordinates are computed via an approximate inverse projection
from radar-relative coordinates. The terrain elevation is looked up from pre-loaded SRTM tiles.
All QC moment fields at gates over terrain above `terrain_threshold` meters are set to `missing`.
This function is multi-threaded.

# Arguments
- `volume`: Radar volume object containing position, moment, and beam geometry data.
- `raw_moment_dict::Dict{String, Int}`: Dictionary mapping raw moment names to column indices (must contain `"SQI_FOR_MASK"`, `"DBZ"`).
- `qc_moments::Matrix`: QC radar moment data matrix to be modified (gates x moments).
- `qc_moment_dict::Dict{String, Int}`: Dictionary mapping QC moment names to column indices.
- `terrain_threshold::Real`: Terrain elevation threshold in meters; gates over higher terrain are removed.
- `tile_dir::String`: Path to directory containing SRTM `.hgt` tile files.

# Returns
- `Matrix`: The modified `qc_moments` matrix with terrain-blocked gates set to `missing`.
"""
function threshold_terrain_height(volume, raw_moment_dict, qc_moments, qc_moment_dict, terrain_threshold, tile_dir)

    # Moment data
    raw_moments = volume.moments

    # Set the reference to the first location in the volume, but could be a parameter
    reference_latitude = volume.latitude[1]
    reference_longitude = volume.longitude[1]

    # Convert the relevant radar information to arrays
    TM = CoordRefSystems.shift(TransverseMercator{1.0,reference_latitude,WGS84Latest}, lonₒ= reference_longitude)
    grid_origin, radar_zyx, beams = radar_arrays(reference_latitude, reference_longitude, volume, TM)

    # Load the SRTM tiles
    tiles = read_srtm_elevation_multi(tile_dir)

    Threads.@threads for i in 1:size(beams,1)

        if ismissing(raw_moments[i,raw_moment_dict["SQI_FOR_MASK"]])
            # In the blanking sector or out of range
            continue
        end
        dbz = raw_moments[i,raw_moment_dict["DBZ"]]
        # Get the horizontal locations of every gate in Y, X dimension
        surface_range = Reff * asin(beams[i,3] * cos(beams[i,2]) / (Reff + beams[i,4]))
        y = grid_origin.y + (radar_zyx[i][2] + surface_range * cos(beams[i,1])) * 1u"m"
        x = grid_origin.x + (radar_zyx[i][3] + surface_range * sin(beams[i,1])) * 1u"m"
        #cartTM = convert(TM,Cartesian{WGS84Latest}(x, y))
        #latlon = convert(LatLon,cartTM)
        lat, lon = appx_inverse_projection(reference_latitude, reference_longitude, [ustrip.(y), ustrip.(x)])

        dtm_height = terrain_height(tiles, lat, lon)
        az = 180.0 * beams[i,1] / pi
        el = 180.0 * beams[i,2] / pi
        range = beams[i,3]/1000.0
        #if abs(az - 289.0) < 1.0 && abs(range - 17.0) < 0.5
        #    println("Ray $i Az: $az El:$el, Range: $range over water! x: $(x), y: $(y), lat $(lat), lon $(lon), height: $(dtm_height), dBZ: $(dbz)")
        #end
        if (dtm_height > terrain_threshold)
            #println("Gate $i over land at lat $(latlon.lat), lon $(latlon.lon), height $dtm_height m")
            for key in keys(qc_moment_dict)
                # Remove everything including SQI_FOR_MASK and PID_FOR_QC
                qc_moments[i,qc_moment_dict[key]] = missing
            end
        elseif !ismissing(dbz) && (dbz >= 50.0)
            #println("High dBZ gate Az: $az El:$el over water! x: $(x), y: $(y), height: $(dtm_height), dBZ: $(dbz)")
        end
    end

    return qc_moments
end

"""
    add_azimuthal_offset(volume, az_offset) -> Vector

Add a constant azimuthal offset to all rays in a radar volume and wrap to [0, 360) degrees.

This is useful for correcting systematic azimuth biases in radar pointing angles.

# Arguments
- `volume`: Radar volume object containing an `azimuth` array.
- `az_offset::Real`: Azimuthal offset in degrees to add to every ray.

# Returns
- `Vector`: The modified azimuth array, wrapped to the range [0, 360) degrees.
"""
function add_azimuthal_offset(volume, az_offset)

    # Add constant offset to azimuth
    volume.azimuth[:] .= volume.azimuth[:] .+ az_offset
    # Ensure azimuth is between 0.0 and 360.0
    volume.azimuth[:] .= mod.(volume.azimuth[:], 360.0)
    return volume.azimuth[:]

end

"""
    mask_sector(volume, raw_moment_dict, qc_moments, qc_moment_dict, heading, az_min, az_max, mask_key) -> Matrix

Mask all radar data within a heading-relative azimuthal sector.

Gates whose heading-relative azimuth (earth-relative azimuth minus heading, modulo 360)
falls between `az_min` and `az_max` are set to `missing` for all QC moment fields. Gates
where `mask_key` is already `missing` are skipped.

# Arguments
- `volume`: Radar volume object containing `range`, `azimuth`, and `elevation` fields.
- `raw_moment_dict::Dict{String, Int}`: Dictionary mapping raw moment names to column indices.
- `qc_moments::Matrix`: QC radar moment data matrix to be modified (gates x moments).
- `qc_moment_dict::Dict{String, Int}`: Dictionary mapping QC moment names to column indices.
- `heading::Vector`: Per-ray heading values in degrees.
- `az_min::Real`: Minimum heading-relative azimuth of the masked sector in degrees.
- `az_max::Real`: Maximum heading-relative azimuth of the masked sector in degrees.
- `mask_key::String`: Moment field name used to check if a gate is already masked.

# Returns
- `Matrix`: The modified `qc_moments` matrix with gates in the specified sector set to `missing`.
"""
function mask_sector(volume, raw_moment_dict, qc_moments, qc_moment_dict, heading, az_min, az_max, mask_key)

    # Mask data between two heading relative azimuths
    # Create an array with all the relevant beam info
    beams = [ (heading[j], volume.azimuth[j])
            for i in eachindex(volume.range), j in eachindex(volume.elevation) ]
    beams = [ beams[i][j] for i in eachindex(beams), j in 1:2]
    for i in 1:size(qc_moments,1)
        if ismissing(qc_moments[i,qc_moment_dict[mask_key]])
            # Already in the blanking sector or out of range
            continue
        end
        az = mod(beams[i,2] - beams[i,1], 360.0)
        if (az >= az_min) && (az <= az_max)
            #println("Gate $i in blanking sector at heading-relative $az (earth-relative $(beams[i,2]))")
            for key in keys(qc_moment_dict)
                # Remove everything including SQI_FOR_MASK and PID_FOR_QC
                qc_moments[i,qc_moment_dict[key]] = missing
            end
        end
    end

    return qc_moments
end
