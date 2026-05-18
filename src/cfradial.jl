# CfRadial 2.1 data model.
#
# Implements the CfRadial 2.1 spec data structures (CfRadialDoc-v2.1-20190901.pdf).
# `Volume` is the top-level container. It holds a vector of `SweepGroup`s, each
# with required coordinate vectors and an `OrderedDict` of `Field`s. Optional
# spec sub-groups (georeference, radar_monitoring, radar_parameters,
# radar_calibration, georeference_correction) hang off the appropriate parent.

# ── FieldMetadata (CfRadial §5.6) ────────────────────────────────────────────

"""
    FieldMetadata

Per-field metadata mirroring the CfRadial 2.1 §5.6 attribute table.

Three mutually exclusive gate states (CfRadial 2.1 / ODIM vocabulary):
- **true missing** — gate not measured. Sentinel: `fill_value` (CF `_FillValue`).
- **undetect** — gate scanned, no detectable signal. Sentinel: `undetect`
  (ODIM `_Undetect`).
- **valid** — gate scanned, signal detected (a real measured value).

Both sentinels can be present, both can be absent.

`extra_attrs` is the catch-all for non-spec attrs encountered in real files.
The reader populates it; `update_cfradial` preserves it through to disk.
`write_cfradial` drops `extra_attrs` unless `write_extras=true`.
"""
Base.@kwdef struct FieldMetadata
    standard_name::Union{String,Nothing}     = nothing
    long_name::Union{String,Nothing}         = nothing
    units::Union{String,Nothing}             = nothing
    fill_value::Union{Real,Nothing}          = nothing
    undetect::Union{Real,Nothing}            = nothing
    scale_factor::Union{Float64,Nothing}     = nothing
    add_offset::Union{Float64,Nothing}       = nothing
    coordinates::Union{String,Nothing}       = nothing
    sampling_ratio::Float64                  = 1.0
    is_discrete::Bool                        = false
    field_folds::Bool                        = false
    fold_limit_lower::Union{Float64,Nothing} = nothing
    fold_limit_upper::Union{Float64,Nothing} = nothing
    is_quality_field::Bool                   = false
    qualified_variables::Vector{String}      = String[]
    ancillary_variables::Vector{String}      = String[]
    flag_values::Union{Vector,Nothing}       = nothing
    flag_meanings::Union{Vector{String},Nothing} = nothing
    flag_masks::Union{Vector,Nothing}        = nothing
    thresholding_xml::Union{String,Nothing}  = nothing
    legend_xml::Union{String,Nothing}        = nothing
    extra_attrs::Dict{String,Any}            = Dict{String,Any}()
end

# ── Field ────────────────────────────────────────────────────────────────────

"""
    Field(data, metadata)

A single radar field on a sweep, with `data::AbstractMatrix` of shape
`(time, range)` already in physical units.
"""
struct Field
    data::AbstractMatrix
    metadata::FieldMetadata
end

Field(data::AbstractMatrix) = Field(data, FieldMetadata())

# ── Georeference (CfRadial §5.4) ─────────────────────────────────────────────

"""
    Georeference

Per-ray georeference for mobile platforms. lat/lon/alt are required; everything
else is optional. `extra_vars` passes through non-spec variables.
"""
Base.@kwdef struct Georeference
    latitude::Vector{Float64}
    longitude::Vector{Float64}
    altitude::Vector{Float64}
    heading::Union{Vector{Float64},Nothing}            = nothing
    roll::Union{Vector{Float64},Nothing}               = nothing
    pitch::Union{Vector{Float64},Nothing}              = nothing
    drift::Union{Vector{Float64},Nothing}              = nothing
    rotation::Union{Vector{Float64},Nothing}           = nothing
    tilt::Union{Vector{Float64},Nothing}               = nothing
    eastward_velocity::Union{Vector{Float64},Nothing}  = nothing
    northward_velocity::Union{Vector{Float64},Nothing} = nothing
    vertical_velocity::Union{Vector{Float64},Nothing}  = nothing
    eastward_wind::Union{Vector{Float64},Nothing}      = nothing
    northward_wind::Union{Vector{Float64},Nothing}     = nothing
    vertical_wind::Union{Vector{Float64},Nothing}      = nothing
    heading_rate::Union{Vector{Float64},Nothing}       = nothing
    roll_rate::Union{Vector{Float64},Nothing}          = nothing
    pitch_rate::Union{Vector{Float64},Nothing}         = nothing
    georefs_applied::Union{Vector{Bool},Nothing}       = nothing
    extra_vars::Dict{String,Any}                       = Dict{String,Any}()
end

# ── RadarMonitoring (CfRadial §5.5) ──────────────────────────────────────────

"""
    RadarMonitoring

Per-ray radar monitoring. All fields optional; `extra_vars` passes through
non-spec variables (e.g. measured_transmit_power_h reported under a non-canonical
name).
"""
Base.@kwdef struct RadarMonitoring
    measured_transmit_power_h::Union{Vector{Float64},Nothing}    = nothing
    measured_transmit_power_v::Union{Vector{Float64},Nothing}    = nothing
    measured_sky_noise::Union{Vector{Float64},Nothing}           = nothing
    measured_cold_noise::Union{Vector{Float64},Nothing}          = nothing
    measured_hot_noise::Union{Vector{Float64},Nothing}           = nothing
    phase_difference_transmit_hv::Union{Vector{Float64},Nothing} = nothing
    antenna_pointing_accuracy_elev::Union{Vector{Float64},Nothing} = nothing
    antenna_pointing_accuracy_az::Union{Vector{Float64},Nothing}   = nothing
    calibration_offset_h::Union{Vector{Float64},Nothing}         = nothing
    calibration_offset_v::Union{Vector{Float64},Nothing}         = nothing
    zdr_offset::Union{Vector{Float64},Nothing}                   = nothing
    extra_vars::Dict{String,Any}                                 = Dict{String,Any}()
end

# ── LidarMonitoring (stub) ───────────────────────────────────────────────────

"""
    LidarMonitoring

Type stub for CfRadial lidar monitoring. Reader/writer support is not
implemented; the type exists so that `Volume` and `SweepGroup` schemas remain
closed.
"""
Base.@kwdef struct LidarMonitoring
    extra_vars::Dict{String,Any} = Dict{String,Any}()
end

# ── SpectrumGroup (stub — CfRadial §6) ───────────────────────────────────────

"""
    SpectrumGroup

Type stub for CfRadial spectrum groups (§6). Reader/writer support is not
implemented.
"""
Base.@kwdef struct SpectrumGroup
    extra_vars::Dict{String,Any} = Dict{String,Any}()
end

# ── SweepGroup (CfRadial §5) ─────────────────────────────────────────────────

"""
    SweepGroup

A single sweep. Time axis runs over `time`, range axis over `range`. `fields`
is an `OrderedDict{String,Field}` keyed by canonical name. Optional sub-groups
(`georeference`, `radar_monitoring`, `spectra`) hang off here.
"""
Base.@kwdef struct SweepGroup
    sweep_number::Int
    sweep_mode::String
    fixed_angle::Float64
    follow_mode::String              = "none"
    prt_mode::String                 = "fixed"
    polarization_mode::String        = "horizontal"
    frequency::Vector{Float64}       = Float64[]
    polarization_sequence::Union{Vector{String},Nothing} = nothing

    time::Vector{DateTime}
    range::Vector{Float64}
    azimuth::Vector{Float64}
    elevation::Vector{Float64}

    range_meters_to_first_gate::Union{Float64,Nothing} = nothing
    range_meters_between_gates::Union{Float64,Nothing} = nothing
    range_spacing_is_constant::Bool                    = true

    pulse_width::Union{Vector{Float64},Nothing}       = nothing
    prt::Union{Vector{Float64},Nothing}               = nothing
    prt_ratio::Union{Vector{Float64},Nothing}         = nothing
    prt_sequence::Union{Matrix{Float64},Nothing}      = nothing
    nyquist_velocity::Union{Vector{Float64},Nothing}  = nothing
    unambiguous_range::Union{Vector{Float64},Nothing} = nothing
    n_samples::Union{Vector{Int},Nothing}             = nothing
    antenna_transition::Union{Vector{Bool},Nothing}   = nothing
    scan_rate::Union{Vector{Float64},Nothing}         = nothing
    target_scan_rate::Union{Float64,Nothing}          = nothing
    rx_range_resolution::Union{Vector{Float64},Nothing} = nothing
    calib_index::Union{Vector{Int},Nothing}           = nothing
    rays_are_indexed::Bool                            = false
    ray_angle_resolution::Union{Float64,Nothing}      = nothing
    qc_procedures::Union{String,Nothing}              = nothing

    fields::OrderedDict{String,Field} = OrderedDict{String,Field}()

    georeference::Union{Georeference,Nothing}        = nothing
    radar_monitoring::Union{RadarMonitoring,Nothing} = nothing
    spectra::Union{SpectrumGroup,Nothing}            = nothing

    extra_attrs::Dict{String,Any}                    = Dict{String,Any}()
end

# ── RadarParameters (CfRadial §7.1) ──────────────────────────────────────────

"""
    RadarParameters

Volume-level static radar parameters. `extra_vars` passes through anything
not enumerated here (e.g. transmit/receive sub-system tags).
"""
Base.@kwdef struct RadarParameters
    antenna_gain_h::Union{Float64,Nothing}     = nothing
    antenna_gain_v::Union{Float64,Nothing}     = nothing
    beam_width_h::Union{Float64,Nothing}       = nothing
    beam_width_v::Union{Float64,Nothing}       = nothing
    receiver_bandwidth::Union{Float64,Nothing} = nothing
    extra_vars::Dict{String,Any}               = Dict{String,Any}()
end

# ── LidarParameters (stub) ───────────────────────────────────────────────────

"""
    LidarParameters

Type stub for CfRadial lidar parameters. Reader/writer support is not
implemented.
"""
Base.@kwdef struct LidarParameters
    extra_vars::Dict{String,Any} = Dict{String,Any}()
end

# ── RadarCalibration (CfRadial §7.3) ─────────────────────────────────────────

"""
    RadarCalibrationEntry

One row of a radar_calibration group, indexed by pulse-width / mode. Fields
mirror CfRadial §7.3 table.
"""
Base.@kwdef struct RadarCalibrationEntry
    time::Union{DateTime,Nothing}              = nothing
    pulse_width::Union{Float64,Nothing}        = nothing
    antenna_gain_h::Union{Float64,Nothing}     = nothing
    antenna_gain_v::Union{Float64,Nothing}     = nothing
    xmit_power_h::Union{Float64,Nothing}       = nothing
    xmit_power_v::Union{Float64,Nothing}       = nothing
    two_way_waveguide_loss_h::Union{Float64,Nothing} = nothing
    two_way_waveguide_loss_v::Union{Float64,Nothing} = nothing
    two_way_radome_loss_h::Union{Float64,Nothing}    = nothing
    two_way_radome_loss_v::Union{Float64,Nothing}    = nothing
    receiver_mismatch_loss::Union{Float64,Nothing}   = nothing
    receiver_mismatch_loss_h::Union{Float64,Nothing} = nothing
    receiver_mismatch_loss_v::Union{Float64,Nothing} = nothing
    radar_constant_h::Union{Float64,Nothing}   = nothing
    radar_constant_v::Union{Float64,Nothing}   = nothing
    probert_jones_correction::Union{Float64,Nothing} = nothing
    dielectric_factor_used::Union{Float64,Nothing}   = nothing
    noise_hc::Union{Float64,Nothing}           = nothing
    noise_vc::Union{Float64,Nothing}           = nothing
    noise_hx::Union{Float64,Nothing}           = nothing
    noise_vx::Union{Float64,Nothing}           = nothing
    receiver_gain_hc::Union{Float64,Nothing}   = nothing
    receiver_gain_vc::Union{Float64,Nothing}   = nothing
    receiver_gain_hx::Union{Float64,Nothing}   = nothing
    receiver_gain_vx::Union{Float64,Nothing}   = nothing
    base_1km_hc::Union{Float64,Nothing}        = nothing
    base_1km_vc::Union{Float64,Nothing}        = nothing
    base_1km_hx::Union{Float64,Nothing}        = nothing
    base_1km_vx::Union{Float64,Nothing}        = nothing
    sun_power_hc::Union{Float64,Nothing}       = nothing
    sun_power_vc::Union{Float64,Nothing}       = nothing
    sun_power_hx::Union{Float64,Nothing}       = nothing
    sun_power_vx::Union{Float64,Nothing}       = nothing
    noise_source_power_h::Union{Float64,Nothing} = nothing
    noise_source_power_v::Union{Float64,Nothing} = nothing
    power_measure_loss_h::Union{Float64,Nothing} = nothing
    power_measure_loss_v::Union{Float64,Nothing} = nothing
    coupler_forward_loss_h::Union{Float64,Nothing} = nothing
    coupler_forward_loss_v::Union{Float64,Nothing} = nothing
    zdr_correction::Union{Float64,Nothing}     = nothing
    ldr_correction_h::Union{Float64,Nothing}   = nothing
    ldr_correction_v::Union{Float64,Nothing}   = nothing
    system_phidp::Union{Float64,Nothing}       = nothing
    test_power_h::Union{Float64,Nothing}       = nothing
    test_power_v::Union{Float64,Nothing}       = nothing
    receiver_slope_hc::Union{Float64,Nothing}  = nothing
    receiver_slope_vc::Union{Float64,Nothing}  = nothing
    receiver_slope_hx::Union{Float64,Nothing}  = nothing
    receiver_slope_vx::Union{Float64,Nothing}  = nothing
    extra_vars::Dict{String,Any}               = Dict{String,Any}()
end

"""
    RadarCalibration(entries, calib_index)

Volume-level calibration table. `entries` holds one `RadarCalibrationEntry` per
pulse-width / mode; `calib_index` (optional) is a per-ray index across the volume.
"""
Base.@kwdef struct RadarCalibration
    entries::Vector{RadarCalibrationEntry} = RadarCalibrationEntry[]
    calib_index::Union{Vector{Int},Nothing} = nothing
end

# ── LidarCalibration (stub) ──────────────────────────────────────────────────

"""
    LidarCalibration

Type stub for CfRadial lidar calibration. Reader/writer support is not
implemented.
"""
Base.@kwdef struct LidarCalibration
    extra_vars::Dict{String,Any} = Dict{String,Any}()
end

# ── GeoreferenceCorrection (CfRadial §7.5) ───────────────────────────────────

"""
    GeoreferenceCorrection

Volume-level static georeference offsets, applied uniformly to all rays.
"""
Base.@kwdef struct GeoreferenceCorrection
    azimuth_correction::Union{Float64,Nothing}                = nothing
    elevation_correction::Union{Float64,Nothing}              = nothing
    range_correction::Union{Float64,Nothing}                  = nothing
    longitude_correction::Union{Float64,Nothing}              = nothing
    latitude_correction::Union{Float64,Nothing}               = nothing
    pressure_altitude_correction::Union{Float64,Nothing}      = nothing
    radar_altitude_correction::Union{Float64,Nothing}         = nothing
    eastward_ground_speed_correction::Union{Float64,Nothing}  = nothing
    northward_ground_speed_correction::Union{Float64,Nothing} = nothing
    vertical_velocity_correction::Union{Float64,Nothing}      = nothing
    heading_correction::Union{Float64,Nothing}                = nothing
    roll_correction::Union{Float64,Nothing}                   = nothing
    pitch_correction::Union{Float64,Nothing}                  = nothing
    drift_correction::Union{Float64,Nothing}                  = nothing
    rotation_correction::Union{Float64,Nothing}               = nothing
    tilt_correction::Union{Float64,Nothing}                   = nothing
    extra_vars::Dict{String,Any}                              = Dict{String,Any}()
end

# ── Volume (CfRadial §4) ─────────────────────────────────────────────────────

"""
    Volume

Top-level CfRadial 2.1 container. Required global attrs/vars are spec-mandated;
everything else is optional. `sweeps` is a `Vector{SweepGroup}`. The struct is
immutable; field addition mutates `SweepGroup.fields` (its `OrderedDict`),
which is allowed.
"""
Base.@kwdef struct Volume
    conventions::String      = "Cf/Radial"
    version::String          = "2.1"
    title::String            = ""
    institution::String      = ""
    source::String           = ""
    history::String          = ""
    instrument_name::String  = ""
    site_name::String        = ""

    references::String       = ""
    comment::String          = ""
    scan_name::String        = ""
    scan_id::Int             = 0
    platform_is_mobile::Bool = false
    ray_times_increase::Bool = true
    simulated_data::Bool     = false

    volume_number::Int        = 0
    platform_type::String     = "fixed"
    instrument_type::String   = "radar"
    primary_axis::String      = "axis_z"
    time_coverage_start::DateTime
    time_coverage_end::DateTime
    latitude::Float64
    longitude::Float64
    altitude::Float64
    altitude_agl::Union{Float64,Nothing} = nothing

    sweeps::Vector{SweepGroup}

    radar_parameters::Union{RadarParameters,Nothing}                = nothing
    radar_calibration::Union{RadarCalibration,Nothing}              = nothing
    georeference_correction::Union{GeoreferenceCorrection,Nothing}  = nothing

    status_str::Union{String,Nothing} = nothing

    extra_attrs::Dict{String,Any} = Dict{String,Any}()
end

# ── Convenience accessors ────────────────────────────────────────────────────

Base.length(v::Volume) = length(v.sweeps)
Base.iterate(v::Volume, state=1) =
    state > length(v.sweeps) ? nothing : (v.sweeps[state], state + 1)
Base.getindex(v::Volume, i::Int) = v.sweeps[i]
Base.firstindex(v::Volume) = 1
Base.lastindex(v::Volume) = length(v.sweeps)

"""
    n_rays(sweep::SweepGroup) -> Int
    n_rays(volume::Volume) -> Int

Total number of rays. For a `SweepGroup`, `length(sweep.time)`. For a `Volume`,
the sum across sweeps.
"""
n_rays(s::SweepGroup) = length(s.time)
n_rays(v::Volume) = sum(n_rays(s) for s in v.sweeps; init=0)

"""
    field_names(sweep::SweepGroup) -> Vector{String}
    field_names(volume::Volume) -> Vector{String}

Canonical field names. For a sweep, in the order they appear in
`sweep.fields`. For a volume, the union across sweeps in alphabetical order.
"""
field_names(s::SweepGroup) = collect(keys(s.fields))
function field_names(v::Volume)
    s = Set{String}()
    for sw in v.sweeps
        union!(s, keys(sw.fields))
    end
    return sort!(collect(s))
end

"""
    has_field(sweep::SweepGroup, name) -> Bool
    has_field(volume::Volume, name) -> Bool

Check for presence of a field. For a volume, returns true if any sweep has it.
"""
has_field(s::SweepGroup, name::AbstractString) = haskey(s.fields, String(name))
has_field(v::Volume, name::AbstractString) =
    any(has_field(s, name) for s in v.sweeps)

"""
    add_field!(sweep::SweepGroup, name, data, metadata=FieldMetadata())

Add a field to a sweep. `data` must be `(n_rays, n_gates)` matching the sweep's
time × range axes.
"""
function add_field!(sweep::SweepGroup, name::AbstractString,
                    data::AbstractMatrix,
                    metadata::FieldMetadata=FieldMetadata())
    nt, nr = length(sweep.time), length(sweep.range)
    size(data) == (nt, nr) || throw(DimensionMismatch(
        "Field `$(name)` data shape $(size(data)) does not match sweep " *
        "(time, range) = ($(nt), $(nr))"))
    sweep.fields[String(name)] = Field(data, metadata)
    return sweep
end

"""
    remove_field!(sweep::SweepGroup, name) -> Bool

Remove a field by name. Returns true if removed, false if it wasn't present.
"""
function remove_field!(sweep::SweepGroup, name::AbstractString)
    key = String(name)
    haskey(sweep.fields, key) || return false
    delete!(sweep.fields, key)
    return true
end
