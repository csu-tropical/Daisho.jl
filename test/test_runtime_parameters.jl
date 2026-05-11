@testset "Runtime Parameters" begin

    @testset "DaishoParameters() loads bundled defaults" begin
        p = DaishoParameters()
        @test p isa DaishoParameters
        @test p.qc.sqi_threshold == 0.35
        @test p.qc.snr_threshold == 6.0
        @test p.qc.spectrum_width_max == 8.0
        @test p.qc.rhohv_threshold == 0.7
        @test p.gridding.beam_inflation == 0.01
        @test p.gridding.power_threshold == 0.5
        @test p.gridding.missing_key == "SQI"
        @test p.gridding.valid_key == "DBZ"
        @test p.io.fill_value_missing == -32768.0
        @test p.io.fill_value_clear == -9999.0
    end

    @testset "MomentParameters fields + grid_type" begin
        p = DaishoParameters()
        @test p.moments.fields[1] == "DBZ"
        @test "SQI" in p.moments.fields
        @test p.moments.grid_type["DBZ"] == :linear
        @test p.moments.grid_type["KDP"] == :weighted
        # Helper-built index dicts.
        fid = Daisho.field_index_dict(p)
        gtd = Daisho.grid_type_index_dict(p)
        @test fid["DBZ"] == 1
        @test gtd[1] == :linear
        @test gtd[findfirst(==("KDP"), p.moments.fields)] == :weighted
        @test length(p.moments.fields) == length(p.moments.grid_type)
    end

    @testset "Cartesian grid defaults" begin
        p = DaishoParameters()
        @test p.grid.cartesian.xdim == 501
        @test p.grid.cartesian.xincr == 500.0
        @test p.grid.cartesian.zdim == 37
    end

    @testset "Spectral grid BCs default to NaturalBC" begin
        p = DaishoParameters()
        s = p.grid.spectral
        @test s.geometry == "RRR"
        @test s.mubar == 3
        @test s.quadrature == :gauss
        # Natural BC has all slots `nothing` and periodic=false
        for side in (s.bc.xL, s.bc.xR, s.bc.yL, s.bc.yR, s.bc.zL, s.bc.zR)
            @test side isa Springsteel.BoundaryConditions
            @test side.u === nothing
            @test side.du === nothing
            @test side.d2u === nothing
            @test side.robin === nothing
            @test side.periodic == false
        end
    end

    @testset "User TOML deep-merges over defaults" begin
        mktemp() do path, io
            write(io, """
                [qc]
                sqi_threshold = 0.5
                [grid.cartesian]
                xdim = 64
                """)
            close(io)
            p = DaishoParameters(path)
            # Overridden:
            @test p.qc.sqi_threshold == 0.5
            @test p.grid.cartesian.xdim == 64
            # Untouched defaults still present:
            @test p.qc.snr_threshold == 6.0
            @test p.grid.cartesian.ydim == 501
            @test p.gridding.power_threshold == 0.5
        end
    end

    @testset "BC sub-tables parse to BoundaryConditions" begin
        mktemp() do path, io
            write(io, """
                [grid.spectral.bc.xL]
                type = "dirichlet"
                value = 1.5
                [grid.spectral.bc.xR]
                type = "neumann"
                [grid.spectral.bc.yL]
                type = "robin"
                alpha = 1.0
                beta  = -2.0
                gamma = 3.0
                [grid.spectral.bc.yR]
                type = "periodic"
                """)
            close(io)
            p = DaishoParameters(path)
            @test p.grid.spectral.bc.xL.u == 1.5
            @test p.grid.spectral.bc.xR.du == 0.0
            @test p.grid.spectral.bc.yL.robin == (1.0, -2.0, 3.0)
            @test p.grid.spectral.bc.yR.periodic == true
            # Untouched sides still natural
            @test p.grid.spectral.bc.zL.u === nothing
        end
    end

    @testset "Unknown keys raise ArgumentError" begin
        mktemp() do path, io
            write(io, """
                [qc]
                not_a_real_key = 1.0
                """)
            close(io)
            @test_throws ArgumentError DaishoParameters(path)
        end
    end

    @testset "Missing user file raises ArgumentError" begin
        @test_throws ArgumentError DaishoParameters("/no/such/path/asdfqwer.toml")
    end

    @testset "Unknown BC type raises ArgumentError" begin
        mktemp() do path, io
            write(io, """
                [grid.spectral.bc.xL]
                type = "tornado"
                """)
            close(io)
            @test_throws ArgumentError DaishoParameters(path)
        end
    end

    @testset "Field name without grid_type raises ArgumentError" begin
        mktemp() do path, io
            write(io, """
                [fields]
                names = ["DBZ", "MISSING_FIELD"]
                [fields.grid_type]
                DBZ = "linear"
                """)
            close(io)
            @test_throws ArgumentError DaishoParameters(path)
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
        # Override to a tiny 1D grid for speed
        mktemp() do path, io
            write(io, """
                [fields]
                names = ["DBZ", "VEL"]
                [fields.grid_type]
                DBZ = "linear"
                VEL = "weighted"
                [grid.spectral]
                geometry = "R"
                xmin = 0.0
                xmax = 1000.0
                xdim = 4
                """)
            close(io)
            p = DaishoParameters(path)
            sgrid = Daisho.create_radar_grid(p)
            @test sgrid isa R_Grid
            @test sgrid.params.iDim == 4 * 3   # cells * mubar default
            @test length(sgrid.params.vars) == 2
            @test sgrid.params.vars["DBZ"] == 1
            @test sgrid.params.vars["VEL"] == 2
        end
    end

    @testset "DaishoParameters is immutable" begin
        p = DaishoParameters()
        # struct without `mutable` keyword cannot have its fields reassigned
        @test_throws ErrorException p.qc = QCParameters()
    end

    @testset "Metadata defaults populate every CF global attr slot" begin
        p = DaishoParameters()
        m = p.grid.metadata
        @test m isa MetadataParameters
        # CF conventions header.
        @test m.Conventions == "CF-1.12"
        # Daisho default identity — kept identical to historical hard-coded
        # values so behavior is unchanged when a user supplies no overrides.
        @test m.institution == "Colorado State University"
        @test m.creator_name == "Michael M. Bell"
        @test m.creator_email == "mmbell@colostate.edu"
        @test m.platform == "RV METEOR"
        @test m.processing_level == "Level 4"
        @test m.license == "CC-BY-4.0"
        # Optional slot: empty by default, opted-in via TOML.
        @test m.references == ""
    end

    @testset "Metadata overrides via [grid.metadata] TOML" begin
        mktemp() do path, io
            write(io, """
                [grid.metadata]
                source           = "NCAR S-PolKa radar"
                instrument       = "S-PolKa"
                title            = "Level 4 Gridded S-PolKa Radar Data"
                summary          = "Level 4 Gridded S-PolKa Radar Data"
                project          = "DYNAMO"
                platform         = "Addu Atoll"
                creator_name     = "Jane Doe"
                creator_email    = "jane.doe@example.org"
                creator_id       = "https://orcid.org/0000-0001-0000-0000"
                keywords         = "radar, precipitation, spolka"
                references       = "https://doi.org/10.0000/example"
                """)
            close(io)
            p = DaishoParameters(path)
            m = p.grid.metadata
            # Overridden fields.
            @test m.source == "NCAR S-PolKa radar"
            @test m.instrument == "S-PolKa"
            @test m.title == "Level 4 Gridded S-PolKa Radar Data"
            @test m.summary == "Level 4 Gridded S-PolKa Radar Data"
            @test m.project == "DYNAMO"
            @test m.platform == "Addu Atoll"
            @test m.creator_name == "Jane Doe"
            @test m.creator_email == "jane.doe@example.org"
            @test m.creator_id == "https://orcid.org/0000-0001-0000-0000"
            @test m.keywords == "radar, precipitation, spolka"
            @test m.references == "https://doi.org/10.0000/example"
            # Untouched defaults still present.
            @test m.Conventions == "CF-1.12"
            @test m.institution == "Colorado State University"
            @test m.license == "CC-BY-4.0"
        end
    end

    @testset "Unknown [grid.metadata] keys raise ArgumentError" begin
        mktemp() do path, io
            write(io, """
                [grid.metadata]
                not_a_global_attr = "oops"
                """)
            close(io)
            @test_throws ArgumentError DaishoParameters(path)
        end
    end

end
