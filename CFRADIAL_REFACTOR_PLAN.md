# Daisho.jl CfRadial 2.1 Refactor Plan

**Status:** approved, ready to execute
**Owner:** Michael Bell
**Plan author:** Claude (drafted from a planning conversation; this file is now the source of truth)

---

## 0. How to use this plan

This document is **self-contained**. It does not depend on prior conversation
context. Execute it sequentially from Phase A through Phase D, committing after
each numbered task. The plan was written so that an agent (or human) starting
from a clean context can carry it out.

**Spec references:**
- `docs/CfRadialDoc-v2.1-20190901.pdf` — CfRadial2 v2.1 DRAFT (2019-09-01).
  Section references like "§5.6" point to this document unless prefixed
  with "v1".
- `docs/CfRadialDoc.v1.4.20160801.pdf` — CfRadial v1.4 (2016-08-01).
  Authoritative for the v1 reader. References like "v1 §4.10" point here.

There is no v2.0 spec doc — v2.0 was deprecated in favor of 2.1 in 2019.
Real-world v2 files in the wild may carry `version="2.0"` metadata
(notably anything written by LROSE RadxConvert as of plan write time),
but their structure is 2.1. Treat any v2 file as 2.1 for parsing; do not
key behavior off the `version` attribute value.

**Stop-and-ask discipline.** If you encounter any of the following, stop and
ask the user before guessing:

- A real input file does not match the v1 or v2 layout the readers assume.
  *Do not* invent ad-hoc parsing branches; ask what the file is.
- A required CfRadial 2.1 field is missing from a real file. Spec says required
  fields raise; ask whether to relax in this case.
- A test fixture is needed and no suitable one exists in `test/fixtures/`.
- Any user-facing rename or deletion that this plan does not explicitly call
  out (the plan enumerates everything that goes away in §4.D).
- The `MomentParameters` shape change (§3.3) breaks an existing test in a way
  not covered by §4.D.4 below.
- A package dependency needs adding (e.g. `OrderedCollections`).

**Auto-mode disposition.** Once a phase begins, work through its tasks
sequentially. After each phase, run `julia --project -e 'using Pkg; Pkg.test()'`
and confirm the documented test count target before moving to the next phase.

---

## 1. Goals & Non-goals

### 1.1 Goals

1. Replace Daisho's home-grown `radar` struct with a CfRadial 2.1
   spec-compliant data model (`Volume` containing `SweepGroup`s containing
   `Field`s with rich `FieldMetadata`).
2. Read both CfRadial1 (flat NetCDF3) and CfRadial2 (grouped NetCDF4) files
   into the same internal `Volume` representation.
3. Write CfRadial2.1-compliant output via a single full writer.
4. Provide a format-preserving "modify in place" path that avoids forcing v1
   archives to convert when only field-level changes are needed.
5. Eliminate the four 1300-line copy-pasted `write_qced_cfradial_*` writers in
   `src/radar.jl`.
6. Eliminate the `raw_moment_dict` / `qc_moment_dict` parallel-dict pattern.
   Fields are looked up by canonical name from `sweep.fields[name]`. The
   spec's `is_quality_field` / `qualified_variables` / `ancillary_variables`
   express QC relationships when needed.
7. Adopt CfRadial's `_FillValue` (true missing) vs `_Undetect` (clear air,
   scanned, no echo) distinction at the metadata level. Internal arrays may
   continue to use the numeric sentinels `-32768.0` and `-9999.0` for storage,
   but the meaning is recorded in `field.metadata.fill_value` and
   `field.metadata.undetect`.
8. Validate `Volume` instances against the spec on demand
   (`validate_spec(volume; strict=false)`).

### 1.2 Non-goals (out of scope for this refactor)

- The legacy `write_gridded_radar_*` family in `src/gridding.jl`. Springsteel
  handles new gridded writes; the legacy gridded writers stay as-is and will
  be deprecated in a separate effort. **See `MULTISWEEP_GRIDDING_PLAN.md`**
  for the follow-on plan that puts a `GridAccumulator` in front of these
  writers and routes the Volume-typed gridding drivers through it.
- `src/processing.jl` (orchestration script that lives outside the package
  module — not `include`d in `src/Daisho.jl`).
- LIDAR-specific paths (`lidar_parameters`, `lidar_calibration`,
  `lidar_monitoring`). Define the types but do not implement reader/writer
  support; raise an informative error if encountered.
- Spectrum groups (§6 of the spec). Define the type stub; do not implement
  reader/writer support.
- Performance tuning (lazy/streaming reads). Read full files into memory.

---

## 2. Pre-execution checklist

The following items are settled. They are documented here for awareness; do
not ask the user about them.

1. **Test fixtures.** Two real SEAPOL files are in `test/fixtures/`,
   covering the same volume in both formats:
     - `cfrad.20240903_150007.042_to_20240903_150444.596_SEAPOL_SUR.nc`
       — CfRadial v1.4 (LROSE-written, original from SIGMET).
     - `cfrad2.20240903_150007.042_to_20240903_150444.596_SEAPOL_PICCOLO_CIRC_SUR.nc`
       — CfRadial v2 (LROSE RadxConvert output; tagged `version="2.0"` but
       structurally 2.1; see §3.2.6).

   Both files cover 11 sweeps × 2447 range gates. Field set includes:
   `DBZ`, `DBZ_ATTEN_UNCORRECTED`, `DBZ_L2`, `DBZ_TOT`, `DBZ_TOT_L2`,
   `HID_CSU`, `KDP`, `PHIDP`, `PHIDP_L2`, `PID_FOR_QC`, `RATE_ZH`,
   `RATE_CSU_METHOD`, `RATE_CSU_BLENDED`, `RHOHV`, `RHOHV_L2`, `RHOHV_NNC`,
   `RHOHV_NNC_L2`, `SNR`, `SNR_L2`, `SQI`, `SQI_FOR_MASK`, `TEMP_FOR_PID`,
   `VEL`, `VEL_L2`, `WIDTH`, `WIDTH_L2`, `ZDR`, `ZDR_ATTEN_UNCORRECTED`,
   `ZDR_L2`. Round-trip tests use these files.

2. **CfRadial1 dialect coverage.** Target is v1.4 (LROSE convention) per
   the SEAPOL fixture. Older v1.x dialects are out of scope unless a
   representative file is supplied.

3. **Backward compatibility.** Phase D deletes the legacy `radar` struct,
   `read_cfradial(file, moment_dict)`, `initialize_qc_fields`,
   `initialize_moment_dictionaries`, and the four `write_qced_cfradial_*`
   writers. The user has confirmed this is acceptable.

If any test cannot find the fixture files, **stop and ask** before fabricating
substitutes.

---

## 3. Architecture

### 3.1 Data model

Place all type definitions in a new file `src/cfradial.jl`. Order matters
(forward declarations not allowed). Suggested order:

1. `FieldMetadata`
2. `Field`
3. `Georeference`
4. `RadarMonitoring`
5. `LidarMonitoring` (stub)
6. `SpectrumGroup` (stub — minimal struct, marked unimplemented for I/O)
7. `SweepGroup`
8. `RadarParameters`
9. `LidarParameters` (stub)
10. `RadarCalibration` (with sub-struct `RadarCalibrationEntry` for one
    calibration row)
11. `LidarCalibration` (stub)
12. `GeoreferenceCorrection`
13. `Volume`

Use `OrderedDict` from the existing `DataStructures` dep for `fields`. Use
`Base.@kwdef` everywhere to allow optional fields with defaults.

#### 3.1.1 `FieldMetadata` (CfRadial §5.6)

```julia
Base.@kwdef struct FieldMetadata
    standard_name::Union{String,Nothing}     = nothing
    long_name::Union{String,Nothing}         = nothing
    units::Union{String,Nothing}             = nothing
    fill_value::Union{Real,Nothing}          = nothing  # _FillValue
    undetect::Union{Real,Nothing}            = nothing  # _Undetect (ODIM/CfRadial)
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
```

Notes:
- `extra_attrs` is the catch-all for non-spec attrs encountered in real
  files (e.g. `meta_group`, `coordinates_extra`). Reader populates it; writer
  preserves it through `update_cfradial` but **does not** include it in
  `write_cfradial` (which writes spec-only output).
- `fill_value` and `undetect` are typed `Union{Real,Nothing}` because the
  data type matches the field type (could be Int16, Float32, …).

#### 3.1.2 `Field`

```julia
struct Field
    data::AbstractMatrix    # (time, range), already in physical units
    metadata::FieldMetadata
end
```

`data` is always 2-D (time, range) per spec §5.6. Reader applies
`scale_factor`/`add_offset` if present and stores the result; metadata
records the original packing parameters so the writer can repack on output if
desired (or use Float32 directly — see §3.2.4).

#### 3.1.3 `Georeference` (CfRadial §5.4)

```julia
Base.@kwdef struct Georeference
    latitude::Vector{Float64}                                # (time,)
    longitude::Vector{Float64}                               # (time,)
    altitude::Vector{Float64}                                # (time,)
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
```

Latitude/longitude/altitude are required in the georeference subgroup (per
spec table 5.4). All other fields are optional.

#### 3.1.4 `RadarMonitoring` (CfRadial §5.5)

Mirror spec table for radar monitoring. All fields optional `Vector{Float64}`
indexed by time, plus `extra_vars::Dict`. Include:
`measured_transmit_power_h/v`, `measured_sky_noise/cold_noise/hot_noise`,
`phase_difference_transmit_hv`, `antenna_pointing_accuracy_elev/az`,
`calibration_offset_h/v`, `zdr_offset`. Add `extra_vars` for radar-specific
extras.

#### 3.1.5 `SweepGroup` (CfRadial §5)

```julia
Base.@kwdef struct SweepGroup
    sweep_number::Int
    sweep_mode::String                       # "azimuth_surveillance" | "rhi" | …
    fixed_angle::Float64                     # sweep_fixed_angle in spec
    follow_mode::String              = "none"
    prt_mode::String                 = "fixed"
    polarization_mode::String        = "horizontal"
    frequency::Vector{Float64}       = Float64[]
    polarization_sequence::Union{Vector{String},Nothing} = nothing

    # Coordinates (required)
    time::Vector{DateTime}                   # (time,)
    range::Vector{Float64}                   # (range,) meters
    azimuth::Vector{Float64}                 # (time,)
    elevation::Vector{Float64}               # (time,)

    # Range axis attributes (range coordinate metadata)
    range_meters_to_first_gate::Union{Float64,Nothing} = nothing
    range_meters_between_gates::Union{Float64,Nothing} = nothing
    range_spacing_is_constant::Bool                    = true

    # Optional per-ray scan parameters (spec §5.3)
    pulse_width::Union{Vector{Float64},Nothing}       = nothing
    prt::Union{Vector{Float64},Nothing}               = nothing
    prt_ratio::Union{Vector{Float64},Nothing}         = nothing
    prt_sequence::Union{Matrix{Float64},Nothing}      = nothing  # (time, prt)
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

    # Field data — by canonical name, not by column index
    fields::OrderedDict{String,Field}        = OrderedDict{String,Field}()

    # Optional sub-groups
    georeference::Union{Georeference,Nothing}        = nothing
    radar_monitoring::Union{RadarMonitoring,Nothing} = nothing
    spectra::Union{SpectrumGroup,Nothing}            = nothing  # stub

    # Pass-through for non-spec sweep-level attrs
    extra_attrs::Dict{String,Any}                    = Dict{String,Any}()
end
```

#### 3.1.6 `RadarParameters` (CfRadial §7.1)

```julia
Base.@kwdef struct RadarParameters
    antenna_gain_h::Union{Float64,Nothing}     = nothing
    antenna_gain_v::Union{Float64,Nothing}     = nothing
    beam_width_h::Union{Float64,Nothing}       = nothing
    beam_width_v::Union{Float64,Nothing}       = nothing
    receiver_bandwidth::Union{Float64,Nothing} = nothing
    extra_vars::Dict{String,Any}               = Dict{String,Any}()
end
```

#### 3.1.7 `RadarCalibration` (CfRadial §7.3)

```julia
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

struct RadarCalibration
    entries::Vector{RadarCalibrationEntry}     # one per pulse-width
    calib_index::Union{Vector{Int},Nothing}    # per-ray index across the volume
end
```

`calib_index` lives at the volume level since it indexes per-ray. Per-ray
`calib_index` may also appear in sweeps (see `SweepGroup.calib_index`); both
are allowed.

#### 3.1.8 `GeoreferenceCorrection` (CfRadial §7.5)

Single struct of `Union{Float64,Nothing}` corrections (one value per volume).
Mirror table 7.5: `azimuth_correction`, `elevation_correction`,
`range_correction`, `longitude_correction`, `latitude_correction`,
`pressure_altitude_correction`, `radar_altitude_correction`,
`eastward_ground_speed_correction`, `northward_ground_speed_correction`,
`vertical_velocity_correction`, `heading_correction`, `roll_correction`,
`pitch_correction`, `drift_correction`, `rotation_correction`,
`tilt_correction`. Plus `extra_vars`.

#### 3.1.9 `Volume` (CfRadial §4)

```julia
Base.@kwdef struct Volume
    # Required global attrs (§4.1)
    conventions::String      = "Cf/Radial"
    version::String          = "2.1"
    title::String            = ""
    institution::String      = ""
    source::String           = ""
    history::String          = ""
    instrument_name::String  = ""
    site_name::String        = ""

    # Optional global attrs
    references::String       = ""
    comment::String          = ""
    scan_name::String        = ""
    scan_id::Int             = 0
    platform_is_mobile::Bool = false
    ray_times_increase::Bool = true
    simulated_data::Bool     = false

    # Required global vars (§4.3)
    volume_number::Int        = 0
    platform_type::String     = "fixed"
    instrument_type::String   = "radar"
    primary_axis::String      = "axis_z"
    time_coverage_start::DateTime
    time_coverage_end::DateTime
    latitude::Float64                       # at start of volume
    longitude::Float64
    altitude::Float64
    altitude_agl::Union{Float64,Nothing} = nothing

    # Sweeps
    sweeps::Vector{SweepGroup}

    # Optional root-level metadata sub-groups (§7)
    radar_parameters::Union{RadarParameters,Nothing}             = nothing
    radar_calibration::Union{RadarCalibration,Nothing}           = nothing
    georeference_correction::Union{GeoreferenceCorrection,Nothing} = nothing

    # Free-form spec hook
    status_str::Union{String,Nothing} = nothing

    # Pass-through for non-spec global attrs
    extra_attrs::Dict{String,Any}     = Dict{String,Any}()
end
```

`Volume` is **immutable**. To "modify a sweep" you reconstruct the volume
with the modified sweep slot, or accept that `sweep.fields[name] = newfield`
mutates the OrderedDict (which is allowed because `fields` is a mutable
container). Field addition uses `add_field!(sweep::SweepGroup, name, data,
metadata)` for clarity.

#### 3.1.10 Convenience accessors

Implement these on `Volume`:

- `Base.length(volume::Volume)` → number of sweeps.
- `Base.iterate(volume::Volume, ...)` → iterate sweeps.
- `Base.getindex(volume::Volume, i::Int)` → `volume.sweeps[i]`.
- `n_rays(volume::Volume)` → total rays across all sweeps.
- `n_rays(sweep::SweepGroup)` → `length(sweep.time)`.
- `field_names(sweep::SweepGroup)` → `collect(keys(sweep.fields))`.
- `field_names(volume::Volume)` → union across sweeps.
- `has_field(sweep, name)` / `has_field(volume, name)`.
- `add_field!(sweep::SweepGroup, name::String, data::AbstractMatrix,
  metadata::FieldMetadata=FieldMetadata())`.
- `remove_field!(sweep::SweepGroup, name::String)`.

### 3.2 I/O strategy

#### 3.2.1 Reader: `read_cfradial(file::AbstractString) -> Volume`

Place implementation in `src/cfradial_reader.jl`. Auto-detects format:

```julia
function read_cfradial(file::AbstractString)
    NCDataset(file, "r") do ds
        if _is_cfradial2(ds)
            return _read_cfradial2(ds)
        else
            return _read_cfradial1(ds)
        end
    end
end
```

Detection logic for `_is_cfradial2(ds)`:
1. Check global `Conventions` attr — if it contains "Cf/Radial-2" or version
   ≥ 2 in a `version` attr, it's v2.
2. Otherwise check for the global variable `sweep_group_name` and the
   presence of NetCDF4 groups whose names match.
3. If neither, treat as v1.

The reader is **lenient**: missing optional fields → `nothing`; unknown attrs
→ `extra_attrs`/`extra_vars`; unknown variable types → log a debug message
and skip. Required fields raise `ArgumentError` with a clear message naming
the file and missing field.

Reader applies `scale_factor`/`add_offset` to integer-packed fields and
returns physical-units data. The packing parameters are preserved in
`metadata.scale_factor` / `metadata.add_offset` so the writer can repack on
output.

##### CfRadial1 (v1) reader behavior

- v1 stores all rays flat with `time(time)`, `range(range)`,
  `azimuth(time)`, etc. as top-level vars. Sweep boundaries come from
  `sweep_start_ray_index(sweep)` and `sweep_end_ray_index(sweep)`.
- Field variables are at top level with dims `(range, time)` or `(time, range)`
  (varies by file). Reader handles both, normalizes to `(time, range)`.
- Build one `SweepGroup` per sweep slice `(swpstart[i]:swpend[i])`. Per-sweep
  metadata vars (`sweep_mode`, `fixed_angle`, etc.) are top-level arrays of
  length `n_sweeps` in v1 — index by sweep number.
- Per-ray vars (azimuth, elevation, time, pulse_width, etc.) are top-level
  arrays of length `total_rays` — slice by sweep boundaries.
- Build the `Volume` from root-level globals and all sweeps.

##### CfRadial2 (v2) reader behavior

- Open NetCDF4 root group; read globals.
- Iterate `sweep_group_name[sweep]` array; for each name, open the named
  sub-group and build a `SweepGroup`.
- Optional sub-groups (`georeference`, `radar_monitoring`, etc.) are
  sub-groups within the sweep group.
- Optional root metadata sub-groups (`radar_parameters`,
  `radar_calibration`, etc.) are top-level sub-groups in the root.

#### 3.2.2 Full writer: `write_cfradial(volume::Volume, file::AbstractString)`

Place in `src/cfradial_writer.jl`. Always writes spec-compliant CfRadial 2.1
(NetCDF4 with groups). Process:

1. Open output in mode `"c"` (create, NetCDF4).
2. Write all required global attributes from `volume`. Write optional
   attributes that have a non-default value. Append `extra_attrs` after.
3. Write required global variables (`volume_number`, `platform_type`,
   `instrument_type`, `primary_axis`, `time_coverage_start/end`, lat/lon/alt,
   `sweep_group_name`, `sweep_fixed_angle`).
4. For each sweep, create a sub-group named `sweep_<n>` (zero-padded to 4
   digits). Inside:
     - Define `time` and `range` dimensions and coordinate variables
       with required attributes (§5.2).
     - Write sweep metadata variables (sweep_number, sweep_mode, …).
     - Write per-ray variables (azimuth, elevation, pulse_width, …) with
       `(time)` dim.
     - For each field in `sweep.fields`, define a variable with `(time, range)`
       dim and the metadata attributes from `field.metadata`. Write the data.
       Repack to integer with scale_factor/add_offset only if both are
       present in metadata; otherwise write as Float32.
     - If `sweep.georeference !== nothing`, create `georeference` sub-group.
     - Same for `radar_monitoring`.
5. If `volume.radar_parameters !== nothing`, create root sub-group.
6. If `volume.radar_calibration !== nothing`, create root sub-group with
   `calib` dimension and per-entry variables.
7. If `volume.georeference_correction !== nothing`, create root sub-group.

#### 3.2.3 Update writer: `update_cfradial(input_file, output_file, volume::Volume; fields=nothing)`

Place in `src/cfradial_writer.jl`. Format-preserving in-place modifier.
Process:

1. `cp(input_file, output_file; force=true)`.
2. Open `output_file` in mode `"a"` (append).
3. Detect format (same logic as the reader).
4. Determine which fields to update: if `fields=nothing`, use all keys in
   `volume.sweeps[*].fields`; otherwise use the supplied list.
5. Verify ray-layout compatibility: number of rays per sweep in `volume`
   matches the file's `sweep_start_ray_index`/`sweep_end_ray_index` (v1) or
   per-sweep `time` dimension (v2). If not, raise a clear error stating
   `update_cfradial` requires preserved layout; suggest `write_cfradial` for
   restructuring writes.
6. For each field name:
     - **v1 path:** define or get the top-level field variable (range,
       time)/(time, range) — match existing layout if the var exists. Write
       the concatenated data across sweeps.
     - **v2 path:** for each sweep, navigate to `sweep_<n>`, define or get
       the field variable, write the per-sweep data slice.
7. Update the `history` global attribute (append a line with timestamp +
   "modified by Daisho.jl update_cfradial").

#### 3.2.4 Validator: `validate_spec(volume::Volume; strict::Bool=false)`

Place in `src/cfradial_validate.jl`. Returns a `ValidationReport` struct with
`errors::Vector{String}` and `warnings::Vector{String}`. Errors are raised in
`strict=true` mode; warnings are always informational.

Checks:
- All required global attrs present and non-empty (errors if missing).
- All required global vars present (errors).
- At least one sweep (error).
- For each sweep: required attrs/vars present (errors).
- For each field: required `_FillValue` (warning) and `units` (warning).
- For mobile platforms: `georeference` sub-group present in each sweep
  (warning if missing).
- ODIM-style `_Undetect` is informational only.

#### 3.2.5 Type-pun packing helper

CfRadial allows fields stored as scaled integers (`int16` + `scale_factor` +
`add_offset`). A small helper in `cfradial_reader.jl`:

```julia
function _unpack_field(raw, scale_factor, add_offset, fill_value)
    # Apply scale/offset, preserve fill_value as the spec sentinel,
    # return Float32 matrix.
end
```

And the reverse in `cfradial_writer.jl`:

```julia
function _pack_field(data::AbstractMatrix{Float32}, metadata::FieldMetadata)
    # If metadata.scale_factor/.add_offset are set, return packed integer.
    # Otherwise return Float32 unchanged.
end
```

#### 3.2.6 Real-world divergences from spec (LROSE files)

Audit of the v2 fixture (LROSE RadxConvert output) revealed the following
divergences between what real files contain and what the 2.1 spec says.
The reader **must** canonicalize these into spec-shape when populating the
`Volume`. The writer always writes spec-canonical names. Pass-through of
unknown extras goes to `extra_attrs` / `extra_vars`.

| # | Concern | Spec says (2.1) | Fixture has | Reader behavior |
|---|---------|-----------------|-------------|-----------------|
| 1 | Version tag | `version="2.1"` | `version="2.0"` | Ignore the tag value; trust structure (groups present → v2). |
| 2 | `time_coverage_start` / `time_coverage_end` global attrs | struck out in §4.1 | present alongside the variables | Ignore the attrs; read from the global vars. Stash attrs in `extra_attrs` for round-trip. |
| 3 | `start_time` / `end_time` global attrs (v1 vestige) | not in spec | present | Stash in `extra_attrs`. |
| 4 | `time_coverage_start/end` variable type | spec table 4.3 says `double`, units = ISO8601 | stored as `string` | Accept either; prefer string. |
| 5 | Calibration dim name | `calib` per §7.3.1 | `r_calib` | Accept either. |
| 6 | Per-ray calibration index | `calib_index` per §5.3 | `r_calib_index` | Accept either; store on `SweepGroup.calib_index`. |
| 7 | Range start / spacing | `meters_to_center_of_first_gate` and `meters_between_gates` as **attrs** on the `range` coord var | also (or only) as scalar vars `start_range`, `ray_gate_spacing` in sweep group | Prefer the range-var attrs; fall back to scalar vars; store both in `extra_vars` if both present. |
| 8 | `georefs_applied` location | inside `georeference` subgroup §5.4 | as a sweep-level variable | Accept either; store on `Georeference.georefs_applied` in the `Volume` (creates an empty `Georeference` for stationary sweeps if needed). |
| 9 | Status string root var | `status_str` per §4.3 | `status_xml` | Accept either; store on `Volume.status_str`. |
| 10 | `_Undetect` attribute on fields | recommended for clear-air; ODIM convention | absent on every field; convention encoded in differing `_FillValue`s (-9999 vs -32768) | Read `_FillValue` only; leave `metadata.undetect` as `nothing`. Preserve the differing fill values as-is. |
| 11 | Calibration `base_*` names | `base_1km_hc/vc/hx/vx` per §7.3 | `base_dbz_1km_hc/vc/hx/vx` | Map LROSE names → spec names on read. |
| 12 | Calibration extras | n/a | `dbz_correction`, `calibration_time` | Pass through `RadarCalibrationEntry.extra_vars`. |
| 13 | Georeference extras | n/a | `georef_time`, `georef_unit_num`, `georef_unit_id` | Pass through `Georeference.extra_vars`. |
| 14 | `field_names` global attr (v1) | struck out in 2.1; present in v1.4 | n/a in v2 fixture; present in v1 fixture | v1 reader may use it as a hint; do not require it. |
| 15 | Per-sweep monitoring subgroup name | `radar_monitoring` (or `lidar_monitoring`) per §5.5 | `monitoring` | Accept either; v2 reader normalizes to the spec-typed `RadarMonitoring`. |
| 16 | Field `coordinates` attribute value | `"elevation azimuth range"` (stationary) or `"elevation azimuth range heading roll pitch rotation tilt"` (mobile) per §5.6.2 | `"time range"` | Read whatever is present into `metadata.coordinates`; do not require it to match spec. Writer emits spec-conformant value based on `volume.platform_is_mobile`. |
| 17 | Grid mapping name | `radar_lidar_radial_scan` per §3.5 | `azimuthal_equidistant` | Read into the volume's grid-mapping metadata as-is (string pass-through); writer emits `radar_lidar_radial_scan` for spec-canonical output. |

**v1.4-specific notes** (will be expanded as the v1 reader is implemented):

- v1 stores all sweeps as flat top-level variables. Sweep slicing comes from
  `sweep_start_ray_index(sweep)` and `sweep_end_ray_index(sweep)`.
- v1 uses the `meta_group` attribute on instrument/calibration/platform
  variables to indicate logical grouping (e.g. `meta_group =
  "radar_parameters"`). The v1 reader uses this to populate the corresponding
  `Volume` sub-group structs. See v1 §5 for the canonical list of
  `meta_group` values: `instrument_parameters`, `radar_parameters`,
  `lidar_parameters`, `radar_calibration`, `lidar_calibration`,
  `platform_velocity`, `geometry_correction`.
- v1 `Conventions` is `CF-1.7` (or similar), `Sub_conventions` is
  `CF-Radial instrument_parameters radar_parameters radar_calibration ...`,
  and `version` is `CF-Radial-1.4`. v1 vs v2 detection: v2 has the
  `sweep_group_name` variable at root and NetCDF4 groups; v1 does not.

**Writer policy on round-trip preservation:**

- `update_cfradial`: copies the input file. The original LROSE-isms remain
  in the copy. New/modified field variables are written using the file's
  existing convention (e.g. if the file has `start_range` and `ray_gate_spacing`
  as scalar vars, and the user adds a field, only the field is added — the
  layout stays as-is).
- `write_cfradial`: writes spec-canonical names. LROSE-isms that came in
  via `extra_attrs` / `extra_vars` are **dropped** unless the user opts in
  (kwarg `write_extras::Bool=false`). This keeps round-trip from
  accidentally re-leaking deprecated attrs.

### 3.3 Parameter system update (drop raw/qc distinction)

Phase D modifies the runtime parameter system shipped to `src/parameters.jl`:

#### 3.3.1 New `MomentParameters` shape

Old shape (current `parameters.jl`):

```julia
struct MomentParameters
    raw::Vector{String}
    qc::Vector{String}
    grid_type::Vector{Symbol}
    raw_dict::Dict{String,Int}
    qc_dict::Dict{String,Int}
    grid_type_dict::Dict{Int,Symbol}
end
```

New shape:

```julia
struct MomentParameters
    fields::Vector{String}                 # canonical field names of interest
    grid_type::Dict{String,Symbol}         # name → :linear | :weighted | :nearest
end
```

There is no longer a separate "raw" and "qc" list. The `[fields]` TOML
section is just a default list of fields to grid, plus a per-name grid_type
hint for the gridders.

#### 3.3.2 New TOML section

`config/defaults.toml` `[moments]` section is **renamed** to `[fields]`:

```toml
[fields]
names     = ["DBZ", "ZDR", "KDP", "RHOHV", "VEL", "WIDTH", "PHIDP", "SQI"]
grid_type = { DBZ = "linear", ZDR = "linear", KDP = "weighted",
              RHOHV = "weighted", VEL = "weighted", WIDTH = "weighted",
              PHIDP = "weighted", SQI = "weighted" }
```

Loader updates:
- `_moments_from_dict` becomes `_fields_from_dict`, returns
  `MomentParameters(fields::Vector{String}, grid_type::Dict{String,Symbol})`.
- The bundled `defaults.toml` is updated.
- All call sites that reference `p.moments.raw_dict` etc. migrate as part of
  the consumer migrations in Phase C.

#### 3.3.3 Why this rename now

The user's instruction in this plan: drop the raw/qc distinction now ("get
it right now") rather than carry forward a misnomer. The `[moments]` section
was added very recently and has no external callers yet; breakage is
contained to internal tests and the bundled defaults file.

---

## 4. Phases

Each phase is one or more git commits. After each phase, run the test suite
and confirm the documented test-count target before moving on.

**Branch:** stay on `development`. No PRs (single-contributor workflow).

### Phase A — New data model + readers + writers (additive only)

**Goal:** ship the new `Volume` type system, `read_cfradial` (dual-format),
`write_cfradial`, `update_cfradial`, and `validate_spec` without touching any
existing code path.

Old code (`radar` struct, `read_cfradial(file, moment_dict)`,
`write_qced_cfradial_*`, etc.) is **untouched** in this phase.

#### A.1 Create new types

- File: `src/cfradial.jl`. Implement all type definitions from §3.1 of this
  plan. Include `add_field!`, `remove_field!`, and the convenience accessors
  from §3.1.10.
- Add `include("cfradial.jl")` to `src/Daisho.jl` after `include("radar.jl")`.
- Export all new public types: `Volume`, `SweepGroup`, `Field`,
  `FieldMetadata`, `Georeference`, `RadarMonitoring`, `RadarParameters`,
  `RadarCalibration`, `RadarCalibrationEntry`, `GeoreferenceCorrection`,
  `LidarParameters`, `LidarCalibration`, `LidarMonitoring`, `SpectrumGroup`,
  and the helper functions `add_field!`, `remove_field!`, `n_rays`,
  `field_names`, `has_field`.

**Test:** `test/test_cfradial_types.jl` — basic construction, defaults,
add_field!/remove_field!, accessors.

#### A.2 Reader implementation

- File: `src/cfradial_reader.jl`. Implement `read_cfradial(file)` plus the
  internal v1 and v2 paths per §3.2.1.
- `include` after `cfradial.jl` in `src/Daisho.jl`. Export `read_cfradial`.
  *Note:* the existing `read_cfradial(file, moment_dict)` in `src/radar.jl`
  must keep working; Julia multiple dispatch lets the new method coexist
  because the new method has a different arity.

**Test:** `test/test_cfradial_reader.jl`. Required tests:
- **v1 read** of `cfrad.20240903_*_SEAPOL_SUR.nc`: assert
  `length(volume.sweeps) == 11`, `volume.instrument_name == "SEAPOL"`,
  spot-check that `DBZ` field exists in sweep 1, range gate count is 2447.
- **v2 read** of `cfrad2.20240903_*_SEAPOL_PICCOLO_CIRC_SUR.nc`: same
  structural assertions; additionally assert that LROSE divergences
  (§3.2.6 #5–#11) are normalized to spec names in the resulting `Volume`.
- **v1 vs v2 cross-check**: the same SEAPOL volume read via v1 and v2
  paths produces equivalent data — for each sweep, assert
  `volume_v1.sweeps[i].time ≈ volume_v2.sweeps[i].time`,
  `azimuth ≈ azimuth`, `elevation ≈ elevation`, and that field arrays for
  shared field names are within float tolerance. (Caveat: pure equality
  may fail because LROSE's v1→v2 conversion can re-encode types; use
  `≈` or a small tolerance.)
- **Lenient handling**: build a synthetic in-memory NetCDF lacking
  optional vars; assert the corresponding `Volume` fields are `nothing`.
- **Required-missing**: build a synthetic NetCDF missing `latitude`;
  assert `read_cfradial` raises `ArgumentError` naming the missing field.

The test for v2 uses the real fixture, not a synthetic round-trip — so
this test doesn't depend on the writer landing first.

#### A.3 Writer implementation

- File: `src/cfradial_writer.jl`. Implement `write_cfradial(volume, file)`
  per §3.2.2 and `update_cfradial(input, output, volume; fields=nothing)`
  per §3.2.3.
- `include` and export.

**Test:** `test/test_cfradial_writer.jl`:
- **Synthetic round-trip:** build a small synthetic `Volume`,
  `write_cfradial(vol, tmp)`, `read_cfradial(tmp)`, assert
  structural equality for all fields, sub-groups, and metadata.
- **Real-fixture round-trip:** `read_cfradial(v1_fixture) → write_cfradial(tmp) →
  read_cfradial(tmp)` produces a `Volume` equivalent to the first read
  (per-field within float tolerance; LROSE-isms dropped per §3.2.6 writer
  policy). Same starting from the v2 fixture.
- **`update_cfradial` on v1:** read v1 fixture, add a new field
  `DBZ_DAISHO_QC` to one sweep with synthetic data,
  `update_cfradial(v1_fixture, tmp_v1, modified_volume)`,
  re-read → assert new field visible and original fields unchanged.
- **`update_cfradial` on v2:** same pattern starting from the v2 fixture.
- **Layout-mismatch error:** read v1 fixture, drop one ray from a sweep
  (mutate the Volume's azimuth/elevation/time and field arrays in lockstep),
  assert `update_cfradial` raises an error explicitly mentioning the layout
  mismatch and pointing the user toward `write_cfradial`.

#### A.4 Validator implementation

- File: `src/cfradial_validate.jl`. Implement `validate_spec(volume;
  strict=false)` per §3.2.4 and the `ValidationReport` struct.
- Export `validate_spec`, `ValidationReport`.

**Test:** `test/test_cfradial_validate.jl`:
- Valid volume → empty report.
- Volume missing required global attr → strict raises, non-strict reports
  error.
- Volume with no sweeps → strict raises.

#### A.5 Wire into runtests.jl

Add the four new test files to `test/runtests.jl` after the existing
parameter tests.

**Phase A acceptance:**
- All existing tests still pass (no regression).
- New test files pass.
- `julia --project -e 'using Pkg; Pkg.test()'` reports no errors and a test
  count strictly greater than 1404 (the post-Task-2 baseline).

### Phase B — Bridge layer (legacy ↔ Volume)

**Goal:** allow existing positional consumers (`grid_radar_volume`,
`threshold_qc`, etc. operating on the legacy `radar` struct) to be driven
from a `Volume` via a one-call adapter. This is temporary scaffolding; it
goes away in Phase D.

#### B.1 `as_legacy_radar(volume::Volume) -> radar`

- File: `src/cfradial.jl` (append) or a new `src/cfradial_bridge.jl` —
  whichever fits cleaner.
- Behavior:
  - Concatenate per-sweep `azimuth`, `elevation`, `time` arrays into one
    flat array each.
  - Build `swpstart`/`swpend` from sweep ray counts.
  - Build a `moments::Matrix` by concatenating per-sweep `(time, range)`
    matrices for each field, in the column order of
    `field_names(volume)` (alphabetical for stability).
  - Return a `radar` struct.
- Also returns or writes side-info: a `Vector{String}` of canonical field
  names matching the moment-matrix column order (so callers can build a
  moment_dict on demand). Suggest signature
  `as_legacy_radar(volume) -> (radar, field_names::Vector{String})`.

#### B.2 `as_volume(r::radar; field_names::Vector{String}, ...) -> Volume`

- Reverse direction. Used in tests and possibly by callers wanting to write
  a CfRadial2 from a legacy radar.
- Caller supplies the canonical field names corresponding to the matrix
  columns. Sweep boundaries reconstructed from `swpstart`/`swpend`.

**Test:** `test/test_cfradial_bridge.jl`:
- Round-trip: `read_cfradial(v1_fixture)` → `as_legacy_radar` → `as_volume` →
  assert structural equality with the original `Volume`.
- `as_legacy_radar` from a synthetic Volume → matrix shape and dict mapping
  match expectations.

**Phase B acceptance:**
- Bridge round-trip test passes.
- All Phase A tests still pass.

### Phase C — Migrate consumers

Each sub-phase migrates one consumer. After each, full test suite passes.

#### C.1 `qualitycontrol.jl` migration

- Add new method:
  ```julia
  function threshold_qc!(sweep::SweepGroup, p::DaishoParameters;
                          inputs::Vector{String} = ["DBZ","VEL","ZDR","KDP",
                                                    "RHOHV","WIDTH","PHIDP"],
                          out_suffix::String = "_QC")
  ```
  For each input field present in `sweep.fields`, build a thresholded copy
  using `p.qc.sqi_threshold`, `p.qc.snr_threshold`, etc. (same logic as the
  current positional `threshold_qc`). Output field name is
  `input_name * out_suffix`. Output field metadata: copy input metadata, then
  set `is_quality_field = true`, `qualified_variables = [input_name]`,
  `long_name = "QC " * input_long_name`. Also append the output name to the
  input field's `ancillary_variables`.
- Keep the existing positional `threshold_qc` and the
  `threshold_qc(..., p::DaishoParameters)` overload for now (Phase D removes
  them once the Phase D `MomentParameters` rename is done — note that the
  positional overload reads `p.qc.*` which doesn't change, so it survives
  Phase D unchanged).

**Test:** extend `test/test_qualitycontrol.jl` with a `SweepGroup`-based
test of `threshold_qc!`. Verify input fields untouched, output fields
correctly thresholded, metadata correctly set.

#### C.2 `gridding.jl` migration

Each gridding driver gains a `Volume` overload that internally calls
`as_legacy_radar` plus the existing `(p::DaishoParameters)` driver. Example:

```julia
function grid_radar_volume(volume::Volume, output_file::AbstractString,
                            index_time, p::DaishoParameters;
                            heading::Real=-9999.0)
    legacy, _names = as_legacy_radar(volume)
    grid_radar_volume(legacy, output_file, index_time, p; heading=heading)
end
```

Same shape for `grid_radar_latlon_volume`, `grid_radar_rhi`,
`grid_radar_ppi`, `grid_radar_composite`, `grid_radar_column`.

**Test:** extend `test/test_gridding.jl` with one Volume-based smoke test
per driver (small synthetic Volume, write to tempfile, verify file exists
and has expected variables).

#### C.3 `springsteel_adapter.jl` migration

Add `Volume` overloads that mirror the existing `DaishoParameters`-based
overloads:

```julia
function grid_radar_volume_spectral(volume::Volume, output_file::AbstractString,
        index_time, sgrid::SpringsteelGrid, p::DaishoParameters;
        heading::Real=-9999.0, institution::String="", source::String="",
        include_derivatives::Bool=false)
    legacy, _names = as_legacy_radar(volume)
    grid_radar_volume_spectral(legacy, output_file, index_time, sgrid, p;
        heading=heading, institution=institution, source=source,
        include_derivatives=include_derivatives)
end
```

Same shape for `grid_radar_ppi_spectral` and `grid_radar_column_spectral`.

**Test:** extend `test/test_springsteel.jl`.

**Phase C acceptance after each sub-phase:**
- New tests pass.
- All prior tests still pass.

### Phase D — Cleanup

This phase removes dead code and finalizes the parameter rename.

#### D.1 Update `MomentParameters`

- Edit `src/parameters.jl`:
  - Replace the `MomentParameters` struct with the new shape from §3.3.1.
  - Replace `_moments_from_dict` with `_fields_from_dict`.
  - Replace the section name `"moments"` with `"fields"` everywhere in the
    loader.
  - Update `DaishoParameters` constructor to call `_fields_from_dict` for
    section `"fields"`.

- Edit `config/defaults.toml`:
  - Rename `[moments]` to `[fields]`.
  - Replace the `raw`, `qc`, `grid_type` keys with the new shape from §3.3.2.

- Edit `test/test_runtime_parameters.jl`:
  - Update tests to expect the new shape: `p.moments.fields` and
    `p.moments.grid_type`.
  - Remove tests that reference `p.moments.raw`, `.qc`, `.raw_dict`,
    `.qc_dict`, `.grid_type_dict`.

- Anywhere in `src/` (gridding, springsteel_adapter, qualitycontrol) that
  references `p.moments.qc_dict` etc.: update to use
  `p.moments.fields` / `p.moments.grid_type` (after the dict-style migration
  below). Concretely:
  - `p.moments.qc_dict` (used as the gridder's `moment_dict`) → build on the
    fly: `Dict(name => i for (i,name) in enumerate(p.moments.fields))`.
  - `p.moments.grid_type_dict` (Int → Symbol) → build on the fly:
    `Dict(i => p.moments.grid_type[name] for (i,name) in enumerate(p.moments.fields))`.

  Wrap these conversions in two helpers in `parameters.jl`:
  ```julia
  field_index_dict(p::DaishoParameters) =
      Dict{String,Int}(name => i for (i,name) in enumerate(p.moments.fields))

  grid_type_index_dict(p::DaishoParameters) =
      Dict{Int,Symbol}(i => p.moments.grid_type[p.moments.fields[i]]
                       for i in eachindex(p.moments.fields))
  ```
  Update consumer overloads to call these helpers instead of reading
  `qc_dict`/`grid_type_dict`.

#### D.2 Delete legacy code

In `src/radar.jl`:
- Remove the `radar` struct.
- Remove `initialize_moment_dictionaries` (both overloads — positional and
  `DaishoParameters`).
- Remove `initialize_qc_fields`.
- Remove `read_cfradial(file, moment_dict)` (the old positional reader).
- Remove `write_qced_cfradial_sigmet`, `write_qced_cfradial_singlepol`,
  `write_qced_cfradial_dualpol`, `write_qced_cfradial_P3`.
- Keep helper functions that are still useful (e.g. `split_sweeps`,
  `beam_height`, `dB_to_linear`, `linear_to_dB`) but adapt to operate on
  `SweepGroup`/`Volume` instead of `radar` if any callers remain. If a
  helper has no callers, remove it.

In `src/qualitycontrol.jl`:
- Remove the positional `threshold_qc(..., sqi, snr, rhohv, spec_width)` and
  the `(p::DaishoParameters)` overload wrapper. `threshold_qc!(sweep, p)` is
  the new public API.

In `src/gridding.jl`:
- Remove the long-positional driver methods (each of the six). Keep only
  the `(volume::Volume, output_file, time, p::DaishoParameters; ...)` form
  added in Phase C.
- The internal `grid_volume`, `grid_rhi`, `grid_ppi`, `grid_composite`,
  `grid_column` workers operate on the gridpoints + radar arrays — these
  continue to take their existing arguments since they are called from the
  Volume-aware drivers after extraction. Adapt as needed so the bridge call
  in Phase C migrates inline (the `as_legacy_radar` call moves into the
  driver, then is replaced by direct iteration over `volume.sweeps` if
  cleaner).
- The legacy `write_gridded_radar_*` family is **out of scope** (see §1.2);
  leave it untouched.

In `src/springsteel_adapter.jl`:
- Remove the long-positional `grid_radar_volume_spectral`,
  `grid_radar_ppi_spectral`, `grid_radar_column_spectral` methods if they
  exist. Keep only the `Volume`-based forms added in Phase C.

In `src/cfradial.jl` (or `cfradial_bridge.jl`):
- Remove `as_legacy_radar` and `as_volume`. They are temporary scaffolding.

In tests:
- Remove or rewrite all tests that build a legacy `radar` struct directly.
  Replace with `Volume` construction. The synthetic radar helpers in
  `test/test_helpers.jl` are rewritten to build `Volume`s.

#### D.3 Update docs

- `docs/src/api.md`: regenerate / update to list the new types and their
  exported functions.
- `docs/src/guide/radar_io.md`: rewrite around `Volume`, `read_cfradial`,
  `write_cfradial`, `update_cfradial`.
- `docs/src/guide/quality_control.md`: rewrite around
  `threshold_qc!(sweep, p)`.
- `docs/src/guide/gridding.md`: rewrite around `grid_radar_volume(volume,
  file, time, p)` etc.
- `docs/src/index.md`: top-level landing page references updated.

#### D.4 Final test pass

- Full `Pkg.test()` clean.
- New test count target: > 1404 + the new tests added across A–C minus any
  tests removed in D.2. Document the new total in the commit message.

**Phase D acceptance:**
- All tests pass.
- `git grep` finds no references to the deleted types (`radar` struct,
  `raw_moment_dict`, `qc_moment_dict`, `initialize_qc_fields`,
  `write_qced_cfradial_*`, `as_legacy_radar`, `as_volume`).
- Docs build cleanly (`julia --project=docs docs/make.jl`, if applicable).

---

## 5. Test strategy

### 5.1 Fixtures

`test/fixtures/` already contains:
- `cfrad.20240903_150007.042_to_20240903_150444.596_SEAPOL_SUR.nc`
  — real CfRadial v1.4 (LROSE-written from SIGMET, 11 sweeps, ~50 MB).
- `cfrad2.20240903_150007.042_to_20240903_150444.596_SEAPOL_PICCOLO_CIRC_SUR.nc`
  — real CfRadial v2 (LROSE RadxConvert from the v1 file, ~58 MB).

These are large NetCDF binaries. They are **already** in `test/fixtures/`;
do not delete or regenerate them. Tests should reference them by relative
path. Round-trip writes go to `tempname()` files, not into the fixtures
directory.

If your environment lacks these files (e.g. fresh checkout), **stop and ask**
the user how to obtain them rather than synthesizing substitutes — the LROSE
quirks documented in §3.2.6 are exactly what the reader must handle, and a
synthetic stub will not exercise them.

### 5.2 Synthetic Volume builder for tests

`test/test_helpers.jl` gains `synthetic_volume(; n_sweeps=2, n_rays=20,
n_gates=100, fields=["DBZ","VEL"]) -> Volume` returning a deterministic
volume for round-trip and unit tests.

### 5.3 Round-trip is the principal test

Every reader/writer test follows the pattern:
1. Build or read a `Volume`.
2. Operate on it.
3. Write to a tempfile.
4. Read it back.
5. Assert structural equality (a recursive `==`-like helper that compares
   all spec fields and ignores `extra_attrs`/`extra_vars` by default).

### 5.4 Validation tests

For each phase, run `validate_spec(volume; strict=true)` on round-tripped
volumes. Should always pass for well-formed test inputs.

### 5.5 Regression on the existing suite

The existing test suite (1404 tests as of plan write time) covers
synthetic radar data, parameter loading, gridding overloads, springsteel
integration, etc. After each phase's migrations, the full suite must still
pass — phases that change consumer signatures (Phase D) update the existing
tests rather than break them.

---

## 6. Decision log (settled — do not relitigate)

The following design choices were made and are **not** revisitable inside
the execution of this plan. If circumstances force reconsideration, stop
and ask.

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Top-level type is `Volume` (not `Radar`) | CfRadial spec terminology; avoids collision with current lowercase `radar` struct during transition. |
| 2 | Per-sweep storage matches spec; no flat-volume cache | Aligns with §2.2 of spec. Convenience flatten available in bridge layer (Phase B), removed in Phase D. |
| 3 | Drop `raw`/`qc` distinction in `MomentParameters` immediately | "Get it right now" per user instruction. Section renamed to `[fields]`. |
| 4 | Internal model = strictly v2.1; reader handles v1 + v2 | Single internal representation; minimizes downstream complexity. |
| 5 | Full writer is v2.1 only | One writer to maintain; encourages migration. Layer-A `update_cfradial` preserves v1 archives. |
| 6 | `_FillValue` (true missing) and `_Undetect` (clear air) recorded in metadata | Spec-compliant; replaces Daisho-specific sentinel folklore at the API level. Internal arrays still use the numeric sentinels for storage. |
| 7 | Implement all spec sub-group **types** up-front (incl. lidar/spectra stubs) | Closed schema; future-proof. Implement reader/writer for: georeference, radar_parameters, radar_calibration, radar_monitoring, georeference_correction. Lidar* and spectra get type stubs only. |
| 8 | `Volume` is immutable; `SweepGroup.fields` is mutable (`OrderedDict`) | Predictable top-level identity; field addition via `add_field!`. |
| 9 | Use `Base.@kwdef` and `Union{T,Nothing}` for optional spec fields | Matches the spec's "required vs optional" two-tier model and gives nice error messages on missing required fields. |
| 10 | Sequential phases A→D, direct commits to `development` branch | Single contributor; no PR overhead. |
| 11 | Reader canonicalizes LROSE divergences (§3.2.6) to spec names | Internal model is spec-only; downstream code never sees `r_calib_index`/`base_dbz_1km_*`/etc. |
| 12 | `write_cfradial` writes spec-canonical names; LROSE-isms in `extra_*` are dropped unless `write_extras=true` | Prevents accidental re-leakage of deprecated attrs; opt-in for fidelity round-trip if needed. |
| 13 | Treat any `version="2.x"` file with NetCDF4 groups as 2.1 | Real LROSE output tags `version="2.0"`; structure is 2.1. Tag value is unreliable. |

---

## 7. Stop-and-ask triggers (re-emphasized)

You **must** stop and ask the user, not assume, if:

1. The pre-execution checklist (§2) is not satisfiable — no v1 fixture, etc.
2. A real input file does not match the v1 or v2 reader logic. Show the
   user the file's structure (`ncdump -h` output is fine) and ask how to
   handle it.
3. The user's existing test fixtures or scripts depend on something this
   plan deletes in Phase D and the breakage was not anticipated.
4. The legacy `radar` struct has consumers outside `src/Daisho.jl`'s
   `include` list (e.g. the orphan `src/processing.jl`) and you're unsure
   whether to migrate them.
5. A spec ambiguity surfaces — e.g. a CfRadial1 file uses an attribute name
   the spec does not document, and you're unsure whether to pass it through
   `extra_attrs` or interpret it.
6. You're tempted to add a "convenience" API not described here (e.g. a
   shortcut method, an extra TOML knob). Confirm first.
7. Performance becomes a problem (read times > 30s for typical files).
   Ask before introducing lazy loading or caching.
8. The bundled `Springsteel` dep changes major version mid-refactor.

---

## 8. File-by-file change summary

### New files

```
src/cfradial.jl                  # Type definitions + accessors + add_field!
src/cfradial_reader.jl           # read_cfradial(file)
src/cfradial_writer.jl           # write_cfradial, update_cfradial
src/cfradial_validate.jl         # validate_spec, ValidationReport
test/test_cfradial_types.jl
test/test_cfradial_reader.jl
test/test_cfradial_writer.jl
test/test_cfradial_validate.jl
test/test_cfradial_bridge.jl
```

### Modified files

```
src/Daisho.jl                    # add includes + exports
src/parameters.jl                # rename [moments]→[fields], simplify MomentParameters (Phase D)
config/defaults.toml             # [fields] section (Phase D)
src/radar.jl                     # delete legacy code (Phase D)
src/qualitycontrol.jl            # add threshold_qc! (Phase C), drop legacy (Phase D)
src/gridding.jl                  # add Volume overloads (Phase C), drop legacy (Phase D)
src/springsteel_adapter.jl       # add Volume overloads (Phase C), drop legacy (Phase D)
test/runtests.jl                 # wire in new test files
test/test_helpers.jl             # synthetic_volume helper (Phase A)
test/test_radar.jl               # rewrite around Volume (Phase D)
test/test_qualitycontrol.jl      # add threshold_qc! tests (C.1), prune legacy (D.2)
test/test_gridding.jl            # add Volume tests (C.2), prune legacy (D.2)
test/test_springsteel.jl         # add Volume tests (C.3), prune legacy (D.2)
test/test_runtime_parameters.jl  # update for new MomentParameters shape (D.1)
docs/src/api.md                  # regenerate (D.3)
docs/src/guide/radar_io.md       # rewrite (D.3)
docs/src/guide/quality_control.md  # rewrite (D.3)
docs/src/guide/gridding.md       # update (D.3)
docs/src/index.md                # update (D.3)
```

### Deleted (Phase D)

```
- struct radar (in src/radar.jl)
- function initialize_moment_dictionaries (both overloads, src/radar.jl)
- function initialize_qc_fields (src/radar.jl)
- function read_cfradial(file, moment_dict) (src/radar.jl)
- function write_qced_cfradial_sigmet (src/radar.jl)
- function write_qced_cfradial_singlepol (src/radar.jl)
- function write_qced_cfradial_dualpol (src/radar.jl)
- function write_qced_cfradial_P3 (src/radar.jl)
- function threshold_qc(positional...) (src/qualitycontrol.jl)
- long-positional grid_radar_* drivers (src/gridding.jl)
- long-positional grid_radar_*_spectral drivers (src/springsteel_adapter.jl)
- as_legacy_radar, as_volume (bridge layer)
```

---

## 9. Quick reference: where things live

- v2.1 spec PDF: `docs/CfRadialDoc-v2.1-20190901.pdf`
- v1.4 spec PDF: `docs/CfRadialDoc.v1.4.20160801.pdf`
- v2 test fixture: `test/fixtures/cfrad2.20240903_*_SEAPOL_PICCOLO_CIRC_SUR.nc`
- v1 test fixture: `test/fixtures/cfrad.20240903_*_SEAPOL_SUR.nc`
- Default parameters: `config/defaults.toml`
- This plan: `CFRADIAL_REFACTOR_PLAN.md` (repo root)
