using NCDatasets

# A DaishoParameters carrying a small PPI grid and a [hybrid_scan] block.
function _hybrid_test_params(; kwargs...)
    fields = [
        Daisho.FieldSpec("DBZ", Set([:linear_interp, :define_detection])),
        Daisho.FieldSpec("SQI", Set([:weighted_interp, :define_scanned])),
    ]
    moments = Daisho.MomentParameters(fields)
    cart = Daisho.CartesianGridParameters(
        xmin = -2000.0, xincr = 1000.0, xdim = 5,
        ymin = -2000.0, yincr = 1000.0, ydim = 5,
        zmin = 500.0, zincr = 500.0, zdim = 3)
    grid = Daisho.GridParameters(cartesian = cart)
    hp = Daisho.HybridScanParameters(; enabled = true, kwargs...)
    return Daisho.DaishoParameters(moments, Daisho.QCParameters(),
        Daisho.GriddingParameters(), grid, Daisho.IOParameters(),
        Daisho.SynthesisParameters(), Daisho.EchoProductsParameters(), hp,
        Set([:grid, :gridding, :hybrid_scan]))
end

# One synthetic tilt: `dbz` is the selection field, `heights` a constant beam height
# (metres) so tests control the height gate directly rather than through geometry.
_tilt(elevation, dbz, height; extra = Dict{String,Array{Float32}}()) =
    (elevation = elevation,
     fields = merge(Dict{String,Array{Float32}}("DBZ" => dbz), extra),
     heights = fill(Float64(height), size(dbz)))

@testset "Hybrid scan" begin

    io = Daisho.IOParameters()
    fv = Float32(io.fill_value)
    ud = Float32(io.undetect)

    @testset "base tilt seeds, higher tilt fills the gaps" begin
        hp = Daisho.HybridScanParameters(enabled = true, base_angle = 0.5,
            beam_height_maximum = 1000.0, height_output = "HYBRID_HEIGHT")

        base = fill(20.0f0, 3, 2)
        base[2, 1] = fv        # true-missing: looks upward
        base[3, 2] = ud        # clear air: stays clear air by default
        high = fill(35.0f0, 3, 2)

        out = apply_hybrid_scan([_tilt(2.0, high, 400.0), _tilt(0.5, base, 50.0)],
                                hp; io = io)

        @test out["DBZ"][1, 1] == 20.0f0          # untouched base value
        @test out["DBZ"][2, 1] == 35.0f0          # filled from the 2.0 tilt
        @test out["DBZ"][3, 2] == ud              # clear air preserved
        @test out["elevation_angle"][1, 1] == 0.5f0
        @test out["elevation_angle"][2, 1] == 2.0f0
        @test out["elevation_angle"][3, 2] == 0.5f0
        @test out["HYBRID_HEIGHT"][1, 1] == 50.0f0
        @test out["HYBRID_HEIGHT"][2, 1] == 400.0f0
    end

    @testset "tilts above beam_height_maximum are rejected" begin
        hp = Daisho.HybridScanParameters(enabled = true, base_angle = 0.5,
            beam_height_maximum = 1000.0, height_output = "HYBRID_HEIGHT")

        base = fill(fv, 2, 2)
        too_high = fill(35.0f0, 2, 2)
        low_enough = fill(30.0f0, 2, 2)

        # Only the aloft tilt is available: nothing fills, provenance is fill_value.
        out = apply_hybrid_scan([_tilt(0.5, base, 50.0), _tilt(9.0, too_high, 5000.0)],
                                hp; io = io)
        @test all(out["DBZ"] .== fv)
        @test all(out["elevation_angle"] .== fv)
        @test all(out["HYBRID_HEIGHT"] .== fv)

        # With an in-range tilt present, that one wins over the aloft one.
        out2 = apply_hybrid_scan([_tilt(0.5, base, 50.0),
                                  _tilt(2.0, low_enough, 900.0),
                                  _tilt(9.0, too_high, 5000.0)], hp; io = io)
        @test all(out2["DBZ"] .== 30.0f0)
        @test all(out2["elevation_angle"] .== 2.0f0)
    end

    # `require_detection` is one switch applied identically at the base tilt and above:
    # it defines what counts as a tilt having answered a cell.
    @testset "require_detection: clear air at the base tilt" begin
        base = fill(ud, 2, 2)
        high = fill(35.0f0, 2, 2)
        tilts = [_tilt(0.5, base, 50.0), _tilt(2.0, high, 400.0)]

        # false: the base observed clear air, which is an answer. Nothing looks upward.
        any_measurement = Daisho.HybridScanParameters(enabled = true,
            beam_height_maximum = 1000.0)
        out = apply_hybrid_scan(tilts, any_measurement; io = io)
        @test all(out["DBZ"] .== ud)
        @test all(out["elevation_angle"] .== 0.5f0)

        # true: only a detection answers, so the search climbs to the echo above.
        detection_only = Daisho.HybridScanParameters(enabled = true,
            require_detection = true, beam_height_maximum = 1000.0)
        out2 = apply_hybrid_scan(tilts, detection_only; io = io)
        @test all(out2["DBZ"] .== 35.0f0)
        @test all(out2["elevation_angle"] .== 2.0f0)
    end

    @testset "require_detection: clear air at a higher tilt" begin
        base = fill(fv, 2, 2)          # never measured: looks upward either way
        clear = fill(ud, 2, 2)         # scanned, no echo
        echo = fill(35.0f0, 2, 2)
        tilts = [_tilt(0.5, base, 50.0), _tilt(2.0, clear, 400.0), _tilt(4.0, echo, 800.0)]

        # false: the 2.0 tilt scanned the cell and saw nothing — that is the answer.
        any_measurement = Daisho.HybridScanParameters(enabled = true,
            beam_height_maximum = 1000.0)
        out = apply_hybrid_scan(tilts, any_measurement; io = io)
        @test all(out["DBZ"] .== ud)
        @test all(out["elevation_angle"] .== 2.0f0)

        # true: climb past the clear air to the echo at 4.0.
        detection_only = Daisho.HybridScanParameters(enabled = true,
            require_detection = true, beam_height_maximum = 1000.0)
        out2 = apply_hybrid_scan(tilts, detection_only; io = io)
        @test all(out2["DBZ"] .== 35.0f0)
        @test all(out2["elevation_angle"] .== 4.0f0)
    end

    @testset "the base tilt is never height-gated" begin
        # At long range even the lowest sweep can exceed beam_height_maximum; it is
        # still the best available look and must not be discarded.
        hp = Daisho.HybridScanParameters(enabled = true, beam_height_maximum = 1000.0,
            height_output = "HYBRID_HEIGHT")
        base = fill(20.0f0, 2, 2)
        high = fill(35.0f0, 2, 2)
        out = apply_hybrid_scan([_tilt(0.5, base, 4000.0), _tilt(2.0, high, 500.0)],
                                hp; io = io)
        @test all(out["DBZ"] .== 20.0f0)
        @test all(out["elevation_angle"] .== 0.5f0)
        @test all(out["HYBRID_HEIGHT"] .== 4000.0f0)
    end

    @testset "unanswered cells keep the base tilt and report its provenance" begin
        hp = Daisho.HybridScanParameters(enabled = true, require_detection = true,
            beam_height_maximum = 1000.0)
        # Base is clear air and nothing above is in range: the value stays, and the
        # provenance names the base tilt that measured it (not fill_value).
        out = apply_hybrid_scan([_tilt(0.5, fill(ud, 2, 2), 50.0),
                                 _tilt(9.0, fill(35.0f0, 2, 2), 5000.0)], hp; io = io)
        @test all(out["DBZ"] .== ud)
        @test all(out["elevation_angle"] .== 0.5f0)

        # Base was never measured and nothing fills in: fill_value provenance.
        out2 = apply_hybrid_scan([_tilt(0.5, fill(fv, 2, 2), 50.0),
                                  _tilt(9.0, fill(35.0f0, 2, 2), 5000.0)], hp; io = io)
        @test all(out2["DBZ"] .== fv)
        @test all(out2["elevation_angle"] .== fv)
    end

    @testset "base tilt selection" begin
        base = fill(20.0f0, 2, 2)
        mid  = fill(30.0f0, 2, 2)
        high = fill(35.0f0, 2, 2)
        tilts = [_tilt(1.0, base, 50.0), _tilt(3.0, mid, 300.0), _tilt(6.0, high, 600.0)]

        # Configured angle present (within tolerance) wins.
        hp = Daisho.HybridScanParameters(enabled = true, base_angle = 3.0,
            beam_height_maximum = 1000.0)
        @test all(apply_hybrid_scan(tilts, hp; io = io)["elevation_angle"] .== 3.0f0)

        # Configured angle absent falls back to the lowest available tilt.
        hp_missing = Daisho.HybridScanParameters(enabled = true, base_angle = 0.5,
            beam_height_maximum = 1000.0)
        @test all(apply_hybrid_scan(tilts, hp_missing; io = io)["elevation_angle"] .== 1.0f0)

        # No configured angle at all: lowest available.
        hp_none = Daisho.HybridScanParameters(enabled = true, beam_height_maximum = 1000.0)
        @test all(apply_hybrid_scan(tilts, hp_none; io = io)["elevation_angle"] .== 1.0f0)
    end

    @testset "carried fields" begin
        base = fill(fv, 2, 2)
        high = fill(35.0f0, 2, 2)
        t_base = _tilt(0.5, base, 50.0;
            extra = Dict{String,Array{Float32}}("RATE_CSU_BLENDED" => fill(fv, 2, 2)))
        t_high = _tilt(2.0, high, 400.0;
            extra = Dict{String,Array{Float32}}("RATE_CSU_BLENDED" => fill(7.5f0, 2, 2)))

        # Empty `fields` carries everything present in the tilts.
        hp_all = Daisho.HybridScanParameters(enabled = true, beam_height_maximum = 1000.0)
        out = apply_hybrid_scan([t_base, t_high], hp_all; io = io)
        @test sort(collect(keys(out))) == ["DBZ", "RATE_CSU_BLENDED", "elevation_angle"]
        @test all(out["RATE_CSU_BLENDED"] .== 7.5f0)

        # An explicit list carries only those fields.
        hp_some = Daisho.HybridScanParameters(enabled = true, fields = ["DBZ"],
            beam_height_maximum = 1000.0, elevation_output = "")
        out2 = apply_hybrid_scan([t_base, t_high], hp_some; io = io)
        @test collect(keys(out2)) == ["DBZ"]
    end

    @testset "gridded height field is preferred over geometry" begin
        hp = Daisho.HybridScanParameters(enabled = true, height_field = "HEIGHT",
            beam_height_maximum = 1000.0)
        X = collect(0.0:1000.0:1000.0)
        Y = collect(0.0:1000.0:1000.0)

        # Geometry at 9° over this tiny grid is well under the limit, but the gridded
        # field says the beam is aloft, so the fill must be rejected.
        geometric = hybrid_beam_heights(X, Y, 9.0, hp)
        @test all(geometric .< hp.beam_height_maximum)

        fields = Dict{String,Array{Float32}}("HEIGHT" => fill(5000.0f0, 2, 2))
        heights = Daisho._hybrid_heights(fields, X, Y, 9.0, hp, io)
        @test all(heights .== 5000.0)

        # Sentinel cells in the height field fall back to geometry.
        fields["HEIGHT"][1, 1] = fv
        heights2 = Daisho._hybrid_heights(fields, X, Y, 9.0, hp, io)
        @test heights2[1, 1] == geometric[1, 1]
        @test heights2[2, 2] == 5000.0
    end

    @testset "hybrid_scan_output_names" begin
        hp = Daisho.HybridScanParameters(fields = ["DBZ", "RATE_CSU_BLENDED"],
            height_output = "HYBRID_HEIGHT")
        @test hybrid_scan_output_names(hp) ==
              ["DBZ", "RATE_CSU_BLENDED", "elevation_angle", "HYBRID_HEIGHT"]
        @test hybrid_scan_output_names(
            Daisho.HybridScanParameters(elevation_output = "")) == String[]
    end

    @testset "errors" begin
        hp = Daisho.HybridScanParameters(enabled = true, select_field = "ZDR")
        @test_throws ArgumentError apply_hybrid_scan(
            [_tilt(0.5, fill(1.0f0, 2, 2), 50.0)], hp; io = io)
        @test_throws ArgumentError apply_hybrid_scan(
            NamedTuple[], Daisho.HybridScanParameters(); io = io)
    end

    @testset "[hybrid_scan] config loading" begin
        base = Dict("fields" => Dict("DBZ" => ["linear_interp", "define_detection"]),
                    "io" => Dict("fill_value" => -32768.0, "undetect" => -9999.0))

        p = Daisho.DaishoParameters(merge(base, Dict("hybrid_scan" => Dict(
            "enabled" => true, "base_angle" => 0.5, "beam_height_maximum" => 750.0,
            "fields" => ["DBZ"], "select_field" => "DBZ"))))
        @test :hybrid_scan in p.provided
        @test p.hybrid_scan.enabled
        @test p.hybrid_scan.base_angle == 0.5
        @test p.hybrid_scan.beam_height_maximum == 750.0

        # Absent block default-constructs and is not marked provided.
        p_none = Daisho.DaishoParameters(base)
        @test !(:hybrid_scan in p_none.provided)
        @test p_none.hybrid_scan.base_angle === nothing

        @test_throws ArgumentError Daisho.DaishoParameters(merge(base,
            Dict("hybrid_scan" => Dict("bogus_key" => 1))))
        @test_throws ArgumentError Daisho.DaishoParameters(merge(base,
            Dict("hybrid_scan" => Dict("beam_height_maximum" => -1.0))))
        # select_field must be among an explicit carried-field list.
        @test_throws ArgumentError Daisho.DaishoParameters(merge(base,
            Dict("hybrid_scan" => Dict("fields" => ["ZDR"], "select_field" => "DBZ"))))
        @test_throws ArgumentError Daisho.DaishoParameters(merge(base,
            Dict("hybrid_scan" => Dict("angle_pattern" => "("))))
    end

    @testset "round trip through gridded PPI files" begin
        p = _hybrid_test_params(base_angle = 0.5, beam_height_maximum = 20000.0,
                                angle_pattern = "_([0-9.]+)\\.nc\$")
        v = synthetic_volume(n_sweeps = 2, n_rays = 24, n_gates = 6,
                             fields = ["DBZ", "SQI"])
        files = String[]
        outfile = tempname() * "_hybrid.nc"
        try
            for i in eachindex(v.sweeps)
                f = tempname() * "_ppi.nc"
                one_sweep = Daisho.Volume((fn === :sweeps ? [v.sweeps[i]] :
                                           getfield(v, fn) for fn in fieldnames(Daisho.Volume))...)
                Daisho.grid_radar_ppi(one_sweep, f, v.time_coverage_start, p)
                push!(files, f)
            end

            # The PPI writer records the sweep's tilt both ways.
            for (i, f) in enumerate(files)
                NCDataset(f) do ds
                    @test haskey(ds, "fixed_angle")
                    @test ds["fixed_angle"].var[] ≈ Float32(v.sweeps[i].fixed_angle)
                    @test ds.attrib["fixed_angle"] ≈ Float32(v.sweeps[i].fixed_angle)
                end
                @test hybrid_scan_tilt_angle(f, p.hybrid_scan) ≈ v.sweeps[i].fixed_angle
            end

            written = build_hybrid_scan(files, outfile, p)
            @test "DBZ" in written && "elevation_angle" in written

            NCDataset(outfile) do ds
                @test size(ds["DBZ"]) == (5, 5, 1)
                @test haskey(ds, "latitude") && haskey(ds, "grid_mapping")
                @test !haskey(ds, "fixed_angle")   # spans tilts; must not claim one
                @test ds["DBZ"].attrib["_FillValue"] == Float32(p.io.fill_value)
            end

            # The product reads back through the ordinary gridded-PPI reader.
            g = Daisho.read_gridded_ppi(outfile, p)
            @test size(g.fields["DBZ"]) == (length(g.X), length(g.Y))

            # Every selected cell reports a tilt that was actually in the input set.
            NCDataset(outfile) do ds
                elev = Float32.(Array(ds["elevation_angle"].var))
                angles = Float32.([s.fixed_angle for s in v.sweeps])
                for e in elev
                    @test e == Float32(p.io.fill_value) || e in angles
                end
            end
        finally
            for f in files
                isfile(f) && rm(f)
            end
            isfile(outfile) && rm(outfile)
        end
    end
end
