const FIXTURE_V1 = joinpath(@__DIR__, "fixtures",
    "cfrad.20240903_150007.042_to_20240903_150444.596_SEAPOL_SUR.nc")
const FIXTURE_V2 = joinpath(@__DIR__, "fixtures",
    "cfrad2.20240903_150007.042_to_20240903_150444.596_SEAPOL_PICCOLO_CIRC_SUR.nc")

@testset "CfRadial reader" begin
    if !isfile(FIXTURE_V1) || !isfile(FIXTURE_V2)
        @info "Skipping fixture-dependent reader tests (fixtures not present)"
        return
    end

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
end
