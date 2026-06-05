using TOML

@testset "Runtime Parameters" begin

    # ── Test helpers ────────────────────────────────────────────────────────
    # `DaishoParameters(path)` is now a strict load — every documented key
    # must be present in the user file. To keep individual tests focused on
    # what they're actually exercising, _write_full_config starts from the
    # bundled template and deep-merges the test's overrides in memory before
    # writing the result to `path` via TOML.print. The merge logic lives
    # only here; it is NOT part of Daisho's runtime config path.

    function _deep_merge_dict(a::AbstractDict, b::AbstractDict)
        out = Dict{String,Any}(String(k) => v for (k, v) in a)
        for (k, v) in b
            ks = String(k)
            if v isa AbstractDict && haskey(out, ks) && out[ks] isa AbstractDict
                out[ks] = _deep_merge_dict(out[ks], v)
            else
                out[ks] = v
            end
        end
        return out
    end

    function _write_full_config(path::AbstractString, overrides::AbstractDict=Dict{String,Any}())
        base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
        merged = _deep_merge_dict(base, overrides)
        open(path, "w") do io
            TOML.print(io, merged)
        end
        return path
    end

    # ── Bundled-template behavior ───────────────────────────────────────────

    @testset "DaishoParameters() loads bundled template" begin
        p = DaishoParameters()
        @test p isa DaishoParameters
        @test p.qc.sqi_threshold == 0.35
        @test p.qc.snr_threshold == 6.0
        @test p.qc.spectrum_width_max == 8.0
        @test p.qc.rhohv_threshold == 0.7
        @test p.gridding.beam_inflation == 0.01
        @test p.gridding.power_threshold == 0.5
        @test Daisho.field_with_tag(p, :define_scanned) == "SQI"
        @test Daisho.field_with_tag(p, :define_detection) == "DBZ"
        @test p.io.fill_value == -32768.0
        @test p.io.undetect == -9999.0
    end

    @testset "MomentParameters fields + tags" begin
        p = DaishoParameters()
        names = [fs.name for fs in p.moments.fields]
        @test "DBZ" in names
        @test "SQI" in names
        dbz = first(fs for fs in p.moments.fields if fs.name == "DBZ")
        width = first(fs for fs in p.moments.fields if fs.name == "WIDTH")
        @test Daisho.interp_of(dbz) == :linear
        @test Daisho.interp_of(width) == :weighted
        @test Daisho.has_tag(dbz, :define_detection)
        fid = Daisho.field_index_dict(p)
        gtd = Daisho.grid_type_index_dict(p)
        # Sorted-by-name column order: DBZ is alphabetically first.
        @test fid["DBZ"] == 1
        @test gtd[1] == :linear
        @test gtd[fid["WIDTH"]] == :weighted
        @test length(fid) == length(p.moments.fields)
    end

    @testset "Cartesian grid template values" begin
        p = DaishoParameters()
        @test p.grid.cartesian.xdim == 501
        @test p.grid.cartesian.xincr == 500.0
        @test p.grid.cartesian.zdim == 37
    end

    @testset "Spectral grid template uses NaturalBC on every side" begin
        p = DaishoParameters()
        s = p.grid.spectral
        @test s.geometry == "RRR"
        @test s.mubar == 3
        @test s.quadrature == :gauss
        for side in (s.bc.xL, s.bc.xR, s.bc.yL, s.bc.yR, s.bc.zL, s.bc.zR)
            @test side isa Springsteel.BoundaryConditions
            @test side.u === nothing
            @test side.du === nothing
            @test side.d2u === nothing
            @test side.robin === nothing
            @test side.periodic == false
        end
    end

    # ── Strict load: overrides via full-config helper ───────────────────────

    @testset "User TOML overrides land on top of template values" begin
        mktemp() do path, io
            close(io)
            _write_full_config(path, Dict(
                "qc" => Dict("sqi_threshold" => 0.5),
                "grid" => Dict("cartesian" => Dict("xdim" => 64)),
            ))
            p = DaishoParameters(path)
            @test p.qc.sqi_threshold == 0.5
            @test p.grid.cartesian.xdim == 64
            # Untouched template values stay put — because the helper wrote
            # them, not because Daisho merged them.
            @test p.qc.snr_threshold == 6.0
            @test p.grid.cartesian.ydim == 501
            @test p.gridding.power_threshold == 0.5
        end
    end

    @testset "BC sub-tables parse to BoundaryConditions" begin
        mktemp() do path, io
            close(io)
            _write_full_config(path, Dict(
                "grid" => Dict("spectral" => Dict("bc" => Dict(
                    "xL" => Dict("type" => "dirichlet", "value" => 1.5),
                    "xR" => Dict("type" => "neumann"),
                    "yL" => Dict("type" => "robin", "alpha" => 1.0, "beta" => -2.0, "gamma" => 3.0),
                    "yR" => Dict("type" => "periodic"),
                ))),
            ))
            p = DaishoParameters(path)
            @test p.grid.spectral.bc.xL.u == 1.5
            @test p.grid.spectral.bc.xR.du == 0.0
            @test p.grid.spectral.bc.yL.robin == (1.0, -2.0, 3.0)
            @test p.grid.spectral.bc.yR.periodic == true
            # Untouched sides still come through as NaturalBC (from template).
            @test p.grid.spectral.bc.zL.u === nothing
        end
    end

    # ── Strict load: validation errors ──────────────────────────────────────

    @testset "Unknown keys raise ArgumentError" begin
        mktemp() do path, io
            close(io)
            _write_full_config(path, Dict(
                "qc" => Dict("not_a_real_key" => 1.0),
            ))
            @test_throws ArgumentError DaishoParameters(path)
        end
    end

    @testset "Missing user file raises ArgumentError" begin
        @test_throws ArgumentError DaishoParameters("/no/such/path/asdfqwer.toml")
    end

    @testset "Optional [qc] absent → defaults + not in provided" begin
        # [qc] is now optional: a config without it default-constructs
        # QCParameters and records its absence in p.provided.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            delete!(base, "qc")
            open(path, "w") do f
                TOML.print(f, base)
            end
            p = DaishoParameters(path)
            @test p.qc == QCParameters()
            @test !(:qc in p.provided)
        end
    end

    @testset "Optional [gridding]/[grid] absent → defaults + provided" begin
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            delete!(base, "gridding")
            delete!(base, "grid")
            open(path, "w") do f
                TOML.print(f, base)
            end
            p = DaishoParameters(path)
            @test p.gridding == GriddingParameters()
            @test !(:gridding in p.provided)
            @test !(:grid in p.provided)
            # A driver that needs [grid] raises a point-of-use error.
            err = try
                Daisho.require_section(p, :grid, :cartesian; for_op="grid_radar_volume")
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("grid_radar_volume", err.msg)
            @test occursin("[grid]", err.msg)
        end
    end

    @testset "Missing [grid.latlon] while [grid.cartesian] present → defaulted" begin
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            delete!(base["grid"], "latlon")
            open(path, "w") do f
                TOML.print(f, base)
            end
            p = DaishoParameters(path)
            @test p.grid.latlon == LatLonGridParameters()
            @test p.grid.cartesian.xdim == 501
        end
    end

    @testset "Mandatory [fields]/[io] absent → ArgumentError" begin
        for sect in ("fields", "io")
            mktemp() do path, io
                close(io)
                base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
                delete!(base, sect)
                open(path, "w") do f
                    TOML.print(f, base)
                end
                err = try
                    DaishoParameters(path)
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin(sect, err.msg)
            end
        end
    end

    @testset "Missing key within a section raises ArgumentError" begin
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            delete!(base["qc"], "sqi_threshold")
            open(path, "w") do f
                TOML.print(f, base)
            end
            err = try
                DaishoParameters(path)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("sqi_threshold", err.msg)
            @test occursin("[qc]", err.msg)
        end
    end

    @testset "Unknown BC type raises ArgumentError" begin
        mktemp() do path, io
            close(io)
            _write_full_config(path, Dict(
                "grid" => Dict("spectral" => Dict("bc" => Dict(
                    "xL" => Dict("type" => "tornado"),
                ))),
            ))
            @test_throws ArgumentError DaishoParameters(path)
        end
    end

    # Replace the whole [fields] table (deep-merge would otherwise keep the
    # template's default fields). `fields` maps name => Vector of tag strings.
    function _write_config_with_fields(path, fields::AbstractDict,
                                       overrides::AbstractDict=Dict{String,Any}())
        base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
        base["fields"] = Dict{String,Any}(String(k) => v for (k, v) in fields)
        merged = _deep_merge_dict(base, overrides)
        open(path, "w") do io
            TOML.print(io, merged)
        end
        return path
    end

    @testset "New-shape parse → FieldSpec.tags and interp_of" begin
        mktemp() do path, io
            close(io)
            _write_config_with_fields(path, Dict(
                "DBZ" => ["linear_interp", "define_detection"],
                "VEL" => ["nearest_interp", "velocity"],
                "SQI" => ["weighted_interp", "define_scanned"],
            ))
            p = DaishoParameters(path)
            byname = Dict(fs.name => fs for fs in p.moments.fields)
            @test Daisho.interp_of(byname["DBZ"]) == :linear
            @test Daisho.interp_of(byname["VEL"]) == :nearest
            @test Daisho.interp_of(byname["SQI"]) == :weighted
            @test Daisho.has_tag(byname["VEL"], :velocity)
            @test Daisho.field_with_tag(p, :define_detection) == "DBZ"
            @test Daisho.field_with_tag(p, :define_scanned) == "SQI"
            @test Daisho.field_with_tag(p, :velocity) == "VEL"
        end
    end

    @testset "Unknown tag rejected (names vocab)" begin
        mktemp() do path, io
            close(io)
            _write_config_with_fields(path, Dict("DBZ" => ["bogus_tag"]))
            err = try; DaishoParameters(path); nothing; catch e; e; end
            @test err isa ArgumentError
            @test occursin("bogus_tag", err.msg)
            @test occursin("linear_interp", err.msg)
        end
    end

    @testset "Two interpolation tags on one field rejected" begin
        mktemp() do path, io
            close(io)
            _write_config_with_fields(path,
                Dict("DBZ" => ["linear_interp", "nearest_interp"]))
            @test_throws ArgumentError DaishoParameters(path)
        end
    end

    @testset "Duplicate singular tag rejected at load" begin
        mktemp() do path, io
            close(io)
            _write_config_with_fields(path, Dict(
                "DBZ" => ["linear_interp", "define_detection"],
                "ZDR" => ["linear_interp", "define_detection"],
            ))
            @test_throws ArgumentError DaishoParameters(path)
        end
    end

    @testset "Zero define_* loads, field_with_tag raises with op name" begin
        mktemp() do path, io
            close(io)
            _write_config_with_fields(path, Dict(
                "DBZ" => ["linear_interp"],
                "VEL" => ["weighted_interp"],
            ))
            p = DaishoParameters(path)   # loads fine
            err = try
                Daisho.field_with_tag(p, :define_detection; for_op="grid_radar_volume")
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("grid_radar_volume", err.msg)
            @test occursin("define_detection", err.msg)
        end
    end

    @testset "Empty [fields] rejected" begin
        mktemp() do path, io
            close(io)
            _write_config_with_fields(path, Dict{String,Any}())
            @test_throws ArgumentError DaishoParameters(path)
        end
    end

    @testset "Migration diagnostics for legacy shapes" begin
        # Legacy [fields] shape.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["fields"] = Dict("names" => ["DBZ"],
                                  "grid_type" => Dict("DBZ" => "linear"))
            open(path, "w") do f; TOML.print(f, base); end
            err = try; DaishoParameters(path); nothing; catch e; e; end
            @test err isa ArgumentError
            @test occursin("flat array of tags", err.msg) ||
                  occursin("define_detection", err.msg)
        end
        # Legacy [gridding] keys.
        mktemp() do path, io
            close(io)
            _write_full_config(path,
                Dict("gridding" => Dict("missing_key" => "SQI")))
            err = try; DaishoParameters(path); nothing; catch e; e; end
            @test err isa ArgumentError
            @test occursin("define_scanned", err.msg)
        end
        # Legacy [io] keys.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["io"] = Dict("fill_value_missing" => -32768.0,
                              "fill_value_clear" => -9999.0)
            open(path, "w") do f; TOML.print(f, base); end
            err = try; DaishoParameters(path); nothing; catch e; e; end
            @test err isa ArgumentError
            @test occursin("fill_value", err.msg) && occursin("undetect", err.msg)
        end
    end

    @testset "Order-independence of column maps and accumulator fields" begin
        f_a = ["DBZ" => ["linear_interp", "define_detection"],
               "VEL" => ["weighted_interp", "velocity"],
               "SQI" => ["weighted_interp", "define_scanned"]]
        mktemp() do p1, io1
            close(io1)
            mktemp() do p2, io2
                close(io2)
                _write_config_with_fields(p1, Dict(f_a))
                _write_config_with_fields(p2, Dict(reverse(f_a)))
                pa = DaishoParameters(p1)
                pb = DaishoParameters(p2)
                @test Daisho.field_index_dict(pa) == Daisho.field_index_dict(pb)
                @test Daisho.grid_type_index_dict(pa) == Daisho.grid_type_index_dict(pb)
            end
        end
    end

    @testset "field_index_dict / grid_type_index_dict helpers" begin
        p = DaishoParameters()
        fid = Daisho.field_index_dict(p)
        gtd = Daisho.grid_type_index_dict(p)
        @test fid["DBZ"] == 1
        @test gtd[1] == :linear
    end

    @testset "create_radar_grid(p) builds correct geometry" begin
        mktemp() do path, io
            close(io)
            _write_config_with_fields(path, Dict(
                    "DBZ" => ["linear_interp"],
                    "VEL" => ["weighted_interp"],
                ),
                Dict("grid" => Dict("spectral" => Dict(
                    "geometry" => "R",
                    "xmin" => 0.0,
                    "xmax" => 1000.0,
                    "xdim" => 4,
                ))),
            )
            p = DaishoParameters(path)
            sgrid = Daisho.create_radar_grid(p)
            @test sgrid isa R_Grid
            @test sgrid.params.iDim == 4 * 3
            @test length(sgrid.params.vars) == 2
            # Sorted column order: DBZ before VEL.
            @test sgrid.params.vars["DBZ"] == 1
            @test sgrid.params.vars["VEL"] == 2
        end
    end

    @testset "DaishoParameters is immutable" begin
        p = DaishoParameters()
        @test_throws ErrorException p.qc = QCParameters()
    end

    # ── Metadata ────────────────────────────────────────────────────────────

    @testset "Metadata template ships generic placeholders" begin
        p = DaishoParameters()
        m = p.grid.metadata
        @test m isa MetadataParameters
        @test m.Conventions == "CF-1.12"
        @test m.institution == "Your Institution"
        @test m.source == "Your radar"
        @test m.instrument == "Radar"
        @test m.title == "Gridded Radar Data"
        @test m.summary == "Gridded Radar Data"
        @test m.creator_name == "Data Creator"
        @test m.creator_email == ""
        @test m.creator_id == ""
        @test m.project == ""
        @test m.platform == ""
        @test m.keywords == "radar, precipitation"
        @test m.processing_level == "Level 4"
        @test m.license == "CC-BY-4.0"
        @test m.references == ""
    end

    @testset "Metadata overrides via [grid.metadata] TOML" begin
        mktemp() do path, io
            close(io)
            _write_full_config(path, Dict(
                "grid" => Dict("metadata" => Dict(
                    "source"        => "NCAR S-PolKa radar",
                    "instrument"    => "S-PolKa",
                    "title"         => "Gridded S-PolKa Radar Data",
                    "summary"       => "Gridded S-PolKa Radar Data",
                    "project"       => "DYNAMO",
                    "platform"      => "Addu Atoll",
                    "creator_name"  => "Jane Doe",
                    "creator_email" => "jane.doe@example.org",
                    "creator_id"    => "https://orcid.org/0000-0001-0000-0000",
                    "keywords"      => "radar, precipitation, spolka",
                    "references"    => "https://doi.org/10.0000/example",
                )),
            ))
            p = DaishoParameters(path)
            m = p.grid.metadata
            @test m.source == "NCAR S-PolKa radar"
            @test m.instrument == "S-PolKa"
            @test m.title == "Gridded S-PolKa Radar Data"
            @test m.summary == "Gridded S-PolKa Radar Data"
            @test m.project == "DYNAMO"
            @test m.platform == "Addu Atoll"
            @test m.creator_name == "Jane Doe"
            @test m.creator_email == "jane.doe@example.org"
            @test m.creator_id == "https://orcid.org/0000-0001-0000-0000"
            @test m.keywords == "radar, precipitation, spolka"
            @test m.references == "https://doi.org/10.0000/example"
            # Untouched template values still present.
            @test m.Conventions == "CF-1.12"
            @test m.institution == "Your Institution"
            @test m.license == "CC-BY-4.0"
        end
    end

    @testset "Unknown [grid.metadata] keys raise ArgumentError" begin
        mktemp() do path, io
            close(io)
            _write_full_config(path, Dict(
                "grid" => Dict("metadata" => Dict("not_a_global_attr" => "oops")),
            ))
            @test_throws ArgumentError DaishoParameters(path)
        end
    end

    # ── print_config ────────────────────────────────────────────────────────

    @testset "print_config(io) writes the bundled template verbatim" begin
        io = IOBuffer()
        print_config(io)
        @test String(take!(io)) == read(Daisho.DEFAULTS_TOML_PATH, String)
    end

    @testset "print_config(path) writes a loadable template" begin
        mktempdir() do dir
            path = joinpath(dir, "template.toml")
            ret = print_config(path)
            @test ret == path
            @test read(path, String) == read(Daisho.DEFAULTS_TOML_PATH, String)
            # Round-trip: loading the just-written file must work and match
            # DaishoParameters() exactly on a representative scalar.
            p = DaishoParameters(path)
            @test p.qc.sqi_threshold == DaishoParameters().qc.sqi_threshold
            @test p.grid.metadata.institution == "Your Institution"
        end
    end

    @testset "print_config(path) refuses to overwrite without force=true" begin
        mktempdir() do dir
            path = joinpath(dir, "template.toml")
            print_config(path)
            @test_throws ArgumentError print_config(path)
            # force=true succeeds.
            @test print_config(path; force=true) == path
        end
    end

end
