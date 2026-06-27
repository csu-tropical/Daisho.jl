const FIXTURE_V1 = joinpath(@__DIR__, "fixtures",
    "cfrad.20240903_150007.042_to_20240903_150444.596_SEAPOL_SUR.nc")
const FIXTURE_V2 = joinpath(@__DIR__, "fixtures",
    "cfrad2.20240903_150007.042_to_20240903_150444.596_SEAPOL_PICCOLO_CIRC_SUR.nc")

@testset "CfRadial reader" begin
    # The real-fixture tests below need the large gitignored .nc files. They skip
    # VISIBLY when those are absent (e.g. in CI). Crucially, the synthetic
    # in-memory testsets further down ALWAYS run, so the reader's core paths stay
    # covered in CI without shipping 100+ MB of radar data.
    if isfile(FIXTURE_V1) && isfile(FIXTURE_V2)

    @testset "v1 fixture read" begin
        v = read_cfradial(FIXTURE_V1)
        @test v isa Volume
        @test length(v.sweeps) == 11
        @test v.instrument_name == "SEAPOL"
        s1 = v.sweeps[1]
        @test length(s1.range) == 2447
        @test haskey(s1.fields, "DBZ")
        # Time coverage end is after start.
        @test v.time_coverage_end > v.time_coverage_start
        @test v.latitude != 0.0
        @test v.longitude != 0.0
    end

    @testset "v2 fixture read + LROSE canonicalization" begin
        v = read_cfradial(FIXTURE_V2)
        @test v isa Volume
        @test length(v.sweeps) == 11
        @test v.instrument_name == "SEAPOL"
        s1 = v.sweeps[1]
        @test length(s1.range) == 2447
        @test haskey(s1.fields, "DBZ")
        # LROSE quirk #6: r_calib_index → calib_index canonicalization.
        @test s1.calib_index !== nothing
        # LROSE quirk #11: base_dbz_1km_* → base_1km_* canonicalization happens.
        # The values themselves may be the fill-value (and thus `nothing` after the
        # missing→nothing coercion); what we check is that a calibration entry
        # exists and the canonical-named keys exist on it.
        @test v.radar_calibration !== nothing
        @test !isempty(v.radar_calibration.entries)
        e = v.radar_calibration.entries[1]
        @test e.pulse_width !== nothing
        @test :base_1km_hc in fieldnames(typeof(e))
        # LROSE quirk #9: status_xml → status_str pass-through.
        @test v.status_str === nothing || v.status_str isa String
    end

    @testset "v1 vs v2 cross-check" begin
        v1 = read_cfradial(FIXTURE_V1)
        v2 = read_cfradial(FIXTURE_V2)
        @test length(v1.sweeps) == length(v2.sweeps)
        for i in 1:length(v1.sweeps)
            s1 = v1.sweeps[i]
            s2 = v2.sweeps[i]
            @test length(s1.time) == length(s2.time)
            @test length(s1.range) == length(s2.range)
            @test all(isapprox.(s1.azimuth, s2.azimuth; atol=0.01))
            @test all(isapprox.(s1.elevation, s2.elevation; atol=0.01))
        end
        # DBZ data should agree where both files have it.
        @test haskey(v1.sweeps[1].fields, "DBZ")
        @test haskey(v2.sweeps[1].fields, "DBZ")
        d1 = v1.sweeps[1].fields["DBZ"].data
        d2 = v2.sweeps[1].fields["DBZ"].data
        @test size(d1) == size(d2)
    end

    else
        @info "Skipping real-fixture reader tests (large .nc fixtures not present); synthetic tests still run"
    end  # end fixture-gated block

    @testset "lenient handling of missing optional vars" begin
        # Synthetic in-memory NetCDF file lacking optional vars.
        tmp = tempname() * ".nc"
        NCDatasets.NCDataset(tmp, "c") do ds
            ds.attrib["Conventions"] = "CF/Radial-1.4"
            ds.attrib["instrument_name"] = "TINY"
            ds.attrib["site_name"] = "TINY"
            ds.attrib["time_coverage_start"] = "2024-01-01T00:00:00Z"
            ds.attrib["time_coverage_end"] = "2024-01-01T00:01:00Z"
            ds.attrib["title"] = "tiny"
            ds.attrib["institution"] = ""
            ds.attrib["source"] = ""
            ds.attrib["history"] = ""
            ds.attrib["references"] = ""
            ds.attrib["comment"] = ""
            ds.attrib["scan_name"] = ""
            ds.attrib["platform_is_mobile"] = "false"
            ds.attrib["ray_times_increase"] = "true"
            NCDatasets.defDim(ds, "time", 3)
            NCDatasets.defDim(ds, "range", 4)
            NCDatasets.defDim(ds, "sweep", 1)
            NCDatasets.defVar(ds, "time", Float64[0.0, 1.0, 2.0], ("time",);
                attrib = DataStructures.OrderedDict("units" => "seconds since 2024-01-01T00:00:00Z"))
            NCDatasets.defVar(ds, "range", Float32[400.0, 500.0, 600.0, 700.0], ("range",))
            NCDatasets.defVar(ds, "azimuth", Float32[0.0, 120.0, 240.0], ("time",))
            NCDatasets.defVar(ds, "elevation", Float32[1.0, 1.0, 1.0], ("time",))
            NCDatasets.defVar(ds, "latitude", 16.0, ())
            NCDatasets.defVar(ds, "longitude", -24.0, ())
            NCDatasets.defVar(ds, "altitude", 50.0, ())
            NCDatasets.defVar(ds, "sweep_start_ray_index", Int32[0], ("sweep",))
            NCDatasets.defVar(ds, "sweep_end_ray_index", Int32[2], ("sweep",))
            NCDatasets.defVar(ds, "fixed_angle", Float32[1.0], ("sweep",))
            data = randn(Float32, 4, 3)
            NCDatasets.defVar(ds, "DBZ", data, ("range", "time");
                attrib = DataStructures.OrderedDict("_FillValue" => Float32(-32768),
                                                   "units" => "dBZ"))
        end
        v = read_cfradial(tmp)
        @test length(v.sweeps) == 1
        @test v.sweeps[1].pulse_width === nothing
        @test v.sweeps[1].nyquist_velocity === nothing
        @test haskey(v.sweeps[1].fields, "DBZ")
        @test size(v.sweeps[1].fields["DBZ"].data) == (3, 4)
        rm(tmp)
    end

    @testset "range/frequency with _FillValue read without error" begin
        # Regression for GitHub issue #3: some writers attach a `_FillValue`
        # attribute to `range` (and `frequency`), so NCDatasets returns the
        # values as `Vector{Union{Missing,Float32}}`. The reader previously did
        # `collect(Float64, ...)` on these, which throws
        # `Cannot convert Missing to Float64` even when no element is actually
        # filled. Routing through `_to_f64vec` (missing → NaN) fixes it.
        tmp = tempname() * ".nc"
        NCDatasets.NCDataset(tmp, "c") do ds
            ds.attrib["Conventions"] = "CF/Radial-1.4"
            ds.attrib["instrument_name"] = "TINY"
            ds.attrib["site_name"] = "TINY"
            ds.attrib["time_coverage_start"] = "2024-01-01T00:00:00Z"
            ds.attrib["time_coverage_end"] = "2024-01-01T00:01:00Z"
            ds.attrib["title"] = "tiny"
            ds.attrib["institution"] = ""
            ds.attrib["source"] = ""
            ds.attrib["history"] = ""
            ds.attrib["references"] = ""
            ds.attrib["comment"] = ""
            ds.attrib["scan_name"] = ""
            ds.attrib["platform_is_mobile"] = "false"
            ds.attrib["ray_times_increase"] = "true"
            NCDatasets.defDim(ds, "time", 3)
            NCDatasets.defDim(ds, "range", 4)
            NCDatasets.defDim(ds, "sweep", 1)
            NCDatasets.defDim(ds, "frequency", 1)
            NCDatasets.defVar(ds, "time", Float64[0.0, 1.0, 2.0], ("time",);
                attrib = DataStructures.OrderedDict("units" => "seconds since 2024-01-01T00:00:00Z"))
            # `range` carries a _FillValue attribute even though every gate is valid.
            NCDatasets.defVar(ds, "range", Float32[400.0, 500.0, 600.0, 700.0], ("range",);
                attrib = DataStructures.OrderedDict("_FillValue" => Float32(-9999)))
            # `frequency` is genuinely filled (missing) here, exercising NaN mapping.
            fvar = NCDatasets.defVar(ds, "frequency", Float32, ("frequency",);
                attrib = DataStructures.OrderedDict("_FillValue" => Float32(-9999)))
            fvar[:] = [missing]
            NCDatasets.defVar(ds, "azimuth", Float32[0.0, 120.0, 240.0], ("time",))
            NCDatasets.defVar(ds, "elevation", Float32[1.0, 1.0, 1.0], ("time",))
            NCDatasets.defVar(ds, "latitude", 16.0, ())
            NCDatasets.defVar(ds, "longitude", -24.0, ())
            NCDatasets.defVar(ds, "altitude", 50.0, ())
            NCDatasets.defVar(ds, "sweep_start_ray_index", Int32[0], ("sweep",))
            NCDatasets.defVar(ds, "sweep_end_ray_index", Int32[2], ("sweep",))
            NCDatasets.defVar(ds, "fixed_angle", Float32[1.0], ("sweep",))
            data = randn(Float32, 4, 3)
            NCDatasets.defVar(ds, "DBZ", data, ("range", "time");
                attrib = DataStructures.OrderedDict("_FillValue" => Float32(-32768),
                                                   "units" => "dBZ"))
        end
        v = read_cfradial(tmp)                       # previously threw
        @test length(v.sweeps) == 1
        @test v.sweeps[1].range == [400.0, 500.0, 600.0, 700.0]  # clean despite _FillValue attr
        @test isnan(v.sweeps[1].frequency[1])        # genuine fill → NaN
        @test haskey(v.sweeps[1].fields, "DBZ")
        rm(tmp)
    end

    @testset "packed Int16 field is unpacked exactly once" begin
        # Regression: a field stored as scaled Int16 (scale_factor) must be
        # decoded to physical units exactly once. The reader reads the RAW
        # stored values (var.var) and applies the scale itself; reading the
        # CF-decoded values and scaling again would be off by scale_factor.
        tmp = tempname() * ".nc"
        # raw stored Int16 (range × time); physical = raw * 0.01, -32768 = fill.
        raw = Int16[ 5000   2500   1000;
                        0  -1000   3000;
                   -32768   4000   4500;
                      100    200    300 ]
        NCDatasets.NCDataset(tmp, "c") do ds
            ds.attrib["Conventions"] = "CF/Radial-1.4"
            ds.attrib["instrument_name"] = "PACK"
            ds.attrib["site_name"] = "PACK"
            ds.attrib["time_coverage_start"] = "2024-01-01T00:00:00Z"
            ds.attrib["time_coverage_end"] = "2024-01-01T00:01:00Z"
            ds.attrib["title"] = "packed"
            ds.attrib["institution"] = ""; ds.attrib["source"] = ""
            ds.attrib["history"] = ""; ds.attrib["references"] = ""
            ds.attrib["comment"] = ""; ds.attrib["scan_name"] = ""
            ds.attrib["platform_is_mobile"] = "false"
            ds.attrib["ray_times_increase"] = "true"
            NCDatasets.defDim(ds, "time", 3)
            NCDatasets.defDim(ds, "range", 4)
            NCDatasets.defDim(ds, "sweep", 1)
            NCDatasets.defVar(ds, "time", Float64[0.0, 1.0, 2.0], ("time",);
                attrib = DataStructures.OrderedDict("units" => "seconds since 2024-01-01T00:00:00Z"))
            NCDatasets.defVar(ds, "range", Float32[400.0, 500.0, 600.0, 700.0], ("range",))
            NCDatasets.defVar(ds, "azimuth", Float32[0.0, 120.0, 240.0], ("time",))
            NCDatasets.defVar(ds, "elevation", Float32[1.0, 1.0, 1.0], ("time",))
            NCDatasets.defVar(ds, "latitude", 16.0, ())
            NCDatasets.defVar(ds, "longitude", -24.0, ())
            NCDatasets.defVar(ds, "altitude", 50.0, ())
            NCDatasets.defVar(ds, "sweep_start_ray_index", Int32[0], ("sweep",))
            NCDatasets.defVar(ds, "sweep_end_ray_index", Int32[2], ("sweep",))
            NCDatasets.defVar(ds, "fixed_angle", Float32[1.0], ("sweep",))
            dv = NCDatasets.defVar(ds, "DBZ", Int16, ("range", "time");
                attrib = DataStructures.OrderedDict(
                    "scale_factor" => Float32(0.01), "add_offset" => Float32(0.0),
                    "_FillValue" => Int16(-32768), "units" => "dBZ"))
            dv.var[:, :] = raw   # write RAW stored ints (bypass CF packing)
        end
        v = read_cfradial(tmp)
        d = v.sweeps[1].fields["DBZ"].data            # (time, range)
        finite = filter(!isnan, vec(d))
        @test maximum(finite) ≈ 50.0 atol = 1e-4      # 5000 * 0.01, NOT 0.5
        @test minimum(finite) ≈ -10.0 atol = 1e-4     # -1000 * 0.01
        @test 25.0 in round.(finite; digits = 4)      # 2500 * 0.01 present
        @test count(isnan, vec(d)) == 1               # the single _FillValue → NaN
        @test v.sweeps[1].fields["DBZ"].metadata.scale_factor ≈ 0.01 atol = 1e-6
        rm(tmp)
    end

    @testset "required-missing raises ArgumentError" begin
        tmp = tempname() * ".nc"
        NCDatasets.NCDataset(tmp, "c") do ds
            ds.attrib["Conventions"] = "CF/Radial-1.4"
            ds.attrib["time_coverage_start"] = "2024-01-01T00:00:00Z"
            ds.attrib["time_coverage_end"] = "2024-01-01T00:01:00Z"
            NCDatasets.defDim(ds, "time", 3)
            NCDatasets.defDim(ds, "range", 4)
            NCDatasets.defVar(ds, "time", Float64[0.0, 1.0, 2.0], ("time",);
                attrib = DataStructures.OrderedDict("units" => "seconds since 2024-01-01T00:00:00Z"))
            NCDatasets.defVar(ds, "range", Float32[400.0, 500.0, 600.0, 700.0], ("range",))
            NCDatasets.defVar(ds, "azimuth", Float32[0.0, 120.0, 240.0], ("time",))
            NCDatasets.defVar(ds, "elevation", Float32[1.0, 1.0, 1.0], ("time",))
            NCDatasets.defVar(ds, "longitude", -24.0, ())
            NCDatasets.defVar(ds, "altitude", 50.0, ())
        end
        @test_throws ArgumentError read_cfradial(tmp)
        rm(tmp)
    end

    @testset "synthetic CfRadial 1.4 full read" begin
        # Build a fuller in-memory v1 file (no large fixtures) to exercise the
        # v1 code path: _read_cfradial1, _read_radar_calibration_v1, and
        # _read_georeference_correction_v1. 2 sweeps × 4 rays, 5 gates, 2 fields.
        tmp = tempname() * ".nc"
        build_synthetic_cfradial_v1(tmp; n_sweeps = 2, rays_per_sweep = 4,
                                    n_gates = 5, field_names = ["DBZ", "VEL"])
        v = read_cfradial(tmp)

        # Globals / scalars
        @test v.version == "CF-Radial-1.4"
        @test v.instrument_name == "SYNV1"
        @test v.site_name == "SYNSITE"
        @test v.scan_name == "SYNSCAN"
        @test v.scan_id == 7
        @test v.platform_is_mobile == true
        @test v.simulated_data == true
        @test v.volume_number == 42
        @test v.platform_type == "vehicle"
        @test v.instrument_type == "radar"
        @test v.primary_axis == "axis_z"
        @test v.status_str == "<status>ok</status>"
        @test v.altitude_agl == 30.0
        @test v.latitude ≈ 16.886 atol = 1e-4
        @test v.longitude ≈ -24.988 atol = 1e-4
        @test v.altitude ≈ 50.0 atol = 1e-6
        @test v.time_coverage_start == DateTime(2024, 8, 27, 21, 40, 0)
        @test v.time_coverage_end == DateTime(2024, 8, 27, 21, 40, 8)
        @test haskey(v.extra_attrs, "non_spec_attr")   # non-spec attr absorbed

        # Sweeps
        @test length(v.sweeps) == 2
        s1 = v.sweeps[1]
        s2 = v.sweeps[2]
        @test s1.sweep_number == 0
        @test s2.sweep_number == 1
        @test s1.sweep_mode == "azimuth_surveillance"   # 2-D char per-sweep read
        @test s1.follow_mode == "none"
        @test s1.prt_mode == "fixed"
        @test s1.polarization_mode == "horizontal"
        @test s1.rays_are_indexed == true
        @test s1.fixed_angle ≈ 1.5 atol = 1e-4
        @test s2.fixed_angle ≈ 2.5 atol = 1e-4
        @test length(s1.time) == 4 && length(s2.time) == 4
        @test length(s1.range) == 5
        @test s1.range_meters_to_first_gate ≈ 400.0 atol = 1e-4
        @test s1.range_meters_between_gates ≈ 100.0 atol = 1e-4
        @test s1.range_spacing_is_constant == true
        @test length(s1.azimuth) == 4
        @test s1.nyquist_velocity !== nothing && all(≈(25.0), s1.nyquist_velocity)
        @test s1.pulse_width !== nothing
        @test s1.n_samples !== nothing && all(==(64), s1.n_samples)
        @test s1.ray_angle_resolution ≈ 1.0 atol = 1e-6
        @test s1.target_scan_rate ≈ 10.0 atol = 1e-6
        @test length(s1.frequency) == 1
        @test s1.frequency[1] ≈ 5.6e9 rtol = 1e-6

        # Fields present, correct gate/ray counts, fill/sentinel semantics
        @test sort(collect(keys(s1.fields))) == ["DBZ", "VEL"]
        @test size(s1.fields["DBZ"].data) == (4, 5)   # (rays, gates)
        @test size(s1.fields["VEL"].data) == (4, 5)
        dbz1 = s1.fields["DBZ"].data
        @test isnan(dbz1[1, 1])             # -32768 → true-missing (NaN)
        @test dbz1[1, 2] ≈ -9999.0 atol = 1e-3  # -9999 clear-air stays finite
        @test count(isnan, vec(dbz1)) == 1

        # Per-sweep georeference sub-group (mobile layout)
        @test s1.georeference !== nothing
        @test length(s1.georeference.latitude) == 4
        @test all(≈(90.0), s1.georeference.heading)

        # Radar calibration parsed (_read_radar_calibration_v1)
        @test v.radar_calibration !== nothing
        @test length(v.radar_calibration.entries) == 2
        e1 = v.radar_calibration.entries[1]
        @test e1.xmit_power_h ≈ 70.0 atol = 1e-4
        @test e1.noise_hc ≈ -75.0 atol = 1e-4
        @test e1.receiver_gain_hc ≈ 40.0 atol = 1e-4
        @test e1.base_1km_hc ≈ -30.0 atol = 1e-4   # LROSE base_dbz_1km_hc → base_1km_hc
        @test e1.time == DateTime(2024, 8, 27, 21, 40, 0)
        @test v.radar_calibration.entries[2].xmit_power_h ≈ 70.5 atol = 1e-4

        # Radar parameters parsed
        @test v.radar_parameters !== nothing
        @test v.radar_parameters.beam_width_h ≈ 1.0 atol = 1e-6

        # Georeference correction parsed (_read_georeference_correction_v1)
        @test v.georeference_correction !== nothing
        @test v.georeference_correction.azimuth_correction ≈ 0.5 atol = 1e-4
        @test v.georeference_correction.elevation_correction ≈ -0.25 atol = 1e-4
        @test v.georeference_correction.roll_correction ≈ 0.1 atol = 1e-4

        rm(tmp)
    end
end
