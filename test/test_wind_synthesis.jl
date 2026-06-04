using TOML

@testset "Wind Synthesis" begin

    # Reusable mini grid_spec factory (3D shapes only for wind synthesis).
    function _wind_spec(shape::Symbol = :volume_3d; n_x=4, n_y=3, n_z=2)
        return GridSpec(
            shape = shape,
            reference_latitude = 16.0,
            reference_longitude = -24.0,
            x_axis = collect(Float64, 1.0:n_x),
            y_axis = collect(Float64, 1.0:n_y),
            z_axis = collect(Float64, 1.0:n_z),
            lat_axis = shape === :latlon_3d ? collect(Float64, 1.0:n_y) : nothing,
            lon_axis = shape === :latlon_3d ? collect(Float64, 1.0:n_x) : nothing,
        )
    end

    # ── Phase 1: scaffolding ────────────────────────────────────────────────

    @testset "WindGridAccumulator construction and dims" begin
        for (shape, trailing) in ((:volume_3d, (2, 3, 4)), (:latlon_3d, (2, 3, 4)))
            spec = _wind_spec(shape)
            @test Daisho.wind_accumulator_dims(spec) == trailing
            acc = WindGridAccumulator(spec, "VEL")
            @test acc.velocity_field == "VEL"
            @test acc.n_unknowns == 2
            # N=2 ⇒ 3 packed symmetric entries; AtWb has 2 component rows.
            @test size(acc.AtWA) == (3, trailing...)
            @test size(acc.M2)   == (3, trailing...)
            @test size(acc.AtWb) == (2, trailing...)
            @test size(acc.weight_total) == trailing
            @test size(acc.gate_count)   == trailing
            @test all(acc.AtWA .== 0.0)
            @test all(acc.AtWb .== 0.0)
            @test all(acc.M2 .== 0.0)
            @test all(acc.weight_total .== 0.0)
            @test all(acc.gate_count .== Int32(0))
            @test isempty(acc.sweeps)
            @test acc.schema_version == Daisho.WIND_ACCUMULATOR_SCHEMA_VERSION
            @test eltype(acc.gate_count) == Int32
        end

        # n_unknowns = 3 (reserved) allocates the larger packed planes.
        acc3 = WindGridAccumulator(_wind_spec(:volume_3d), "VEL"; n_unknowns = 3)
        @test size(acc3.AtWA) == (6, 2, 3, 4)
        @test size(acc3.AtWb) == (3, 2, 3, 4)

        # Invalid n_unknowns rejected.
        @test_throws ArgumentError WindGridAccumulator(_wind_spec(), "VEL"; n_unknowns = 1)
        @test_throws ArgumentError WindGridAccumulator(_wind_spec(), "VEL"; n_unknowns = 4)

        # Non-3D shapes are out of stage-1 scope.
        bad = GridSpec(shape = :ppi_2d, reference_latitude = 0.0,
                       reference_longitude = 0.0,
                       x_axis = [0.0], y_axis = [0.0], z_axis = [0.0])
        @test_throws ArgumentError Daisho.wind_accumulator_dims(bad)
        @test_throws ArgumentError WindGridAccumulator(bad, "VEL")

        # Custom sentinels carry through.
        acc = WindGridAccumulator(_wind_spec(), "VEL";
                                  fill_value = -777.0, undetect = -8888.0)
        @test acc.fill_value == -777.0
        @test acc.undetect == -8888.0
    end

    @testset "WindGridAccumulator from DaishoParameters" begin
        p = DaishoParameters()
        spec = _wind_spec(:volume_3d)
        acc = WindGridAccumulator(spec, p)
        # :velocity tag is on VEL in the bundled template.
        @test acc.velocity_field == "VEL"
        @test acc.fill_value == p.io.fill_value
        @test acc.undetect == p.io.undetect
    end

    @testset "packed symmetric index map" begin
        # N=2 → [aa, ab, bb]; symmetric in (i, j).
        @test Daisho._packed_index(1, 1, 2) == 1
        @test Daisho._packed_index(1, 2, 2) == 2
        @test Daisho._packed_index(2, 1, 2) == 2
        @test Daisho._packed_index(2, 2, 2) == 3
        @test Daisho._n_packed(2) == 3
        # N=3 → [aa, ab, ac, bb, bc, cc].
        for (i, j, k) in ((1,1,1),(1,2,2),(1,3,3),(2,2,4),(2,3,5),(3,3,6))
            @test Daisho._packed_index(i, j, 3) == k
            @test Daisho._packed_index(j, i, 3) == k
        end
        @test Daisho._n_packed(3) == 6
    end

    @testset "CartesianFrame is the identity output frame" begin
        f = CartesianFrame()
        @test component_names(f) == ("U", "V")
        R = rotation_at(f, 1000.0, -2000.0, 500.0)
        @test R == [1.0 0.0; 0.0 1.0]
        # Orthonormal: R'R == I.
        @test R' * R == [1.0 0.0; 0.0 1.0]
    end

    @testset "[synthesis] config load" begin
        # Present in the bundled template.
        p = DaishoParameters()
        @test :synthesis in p.provided
        @test p.synthesis.velocity_variance == 1.0
        @test p.synthesis.max_sigma == [2.0, 2.0]

        # Omitted → defaults, not in provided.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            delete!(base, "synthesis")
            open(path, "w") do f; TOML.print(f, base); end
            p2 = DaishoParameters(path)
            @test p2.synthesis.velocity_variance == SynthesisParameters().velocity_variance
            @test p2.synthesis.max_sigma == SynthesisParameters().max_sigma
            @test !(:synthesis in p2.provided)
        end

        # Custom values round-trip.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["synthesis"] = Dict("velocity_variance" => 4.0,
                                     "max_sigma_1" => 1.5, "max_sigma_2" => 3.0)
            open(path, "w") do f; TOML.print(f, base); end
            p3 = DaishoParameters(path)
            @test p3.synthesis.velocity_variance == 4.0
            @test p3.synthesis.max_sigma == [1.5, 3.0]
        end

        # Unknown key in [synthesis] is rejected.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["synthesis"]["bogus"] = 1.0
            open(path, "w") do f; TOML.print(f, base); end
            @test_throws ArgumentError DaishoParameters(path)
        end

        # Missing key in a present [synthesis] is rejected (strict).
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            delete!(base["synthesis"], "max_sigma_2")
            open(path, "w") do f; TOML.print(f, base); end
            @test_throws ArgumentError DaishoParameters(path)
        end
    end

    # ── Phase 2: streaming accumulation ─────────────────────────────────────

    # One-ray/one-gate sweep with a chosen beam (az, el, range) and velocity.
    function _one_gate_sweep(; az, el, r, vr)
        sweep = SweepGroup(
            sweep_number = 0, sweep_mode = "azimuth_surveillance",
            fixed_angle = Float64(el),
            time = [DateTime(2024, 1, 1, 0, 0, 0)],
            range = [Float64(r)], azimuth = [Float64(az)], elevation = [Float64(el)],
        )
        add_field!(sweep, "VEL", reshape([Float32(vr)], 1, 1),
                   FieldMetadata(units = "m/s", long_name = "VEL"))
        return sweep
    end

    @testset "single gate → exact rank-1 normal system" begin
        p = DaishoParameters()
        lat0, lon0 = 16.0, -24.0
        vr = 5.0

        # East-pointing beam (az=90°): a = sin(az)cos(el) ≈ 1, b ≈ 0, so u is
        # well-determined and v is not. The single gate lands on the (y=0,
        # x≈20 km) column only.
        spec = GridSpec(shape = :volume_3d,
            reference_latitude = lat0, reference_longitude = lon0,
            x_axis = [18000.0, 20000.0, 22000.0],
            y_axis = [-2000.0, 0.0, 2000.0],
            z_axis = [0.0, 47.2])
        acc = WindGridAccumulator(spec, "VEL")
        sweep = _one_gate_sweep(az = 90.0, el = 0.0, r = 20000.0, vr = vr)
        grid_sweep_wind!(acc, sweep, p;
            ref_latitude = lat0, ref_longitude = lon0, ref_altitude = 0.0)

        # Exactly the (j=2, i=2) column receives the gate (both z levels).
        for ix in CartesianIndices((2, 3, 3))
            k, j, i = ix.I
            expected = (j == 2 && i == 2) ? Int32(1) : Int32(0)
            @test acc.gate_count[k, j, i] == expected
        end
        @test length(acc.sweeps) == 1

        for k in 1:2
            w = acc.weight_total[k, 2, 2]
            @test w > 0.0
            Saa, Sab, Sbb = acc.AtWA[1, k, 2, 2], acc.AtWA[2, k, 2, 2], acc.AtWA[3, k, 2, 2]
            Taa, Tab, Tbb = acc.M2[1, k, 2, 2],   acc.M2[2, k, 2, 2],   acc.M2[3, k, 2, 2]
            Sav, Sbv = acc.AtWb[1, k, 2, 2], acc.AtWb[2, k, 2, 2]
            # Single gate ⇒ rank-1: det == 0 and M2 == w·AtWA exactly.
            @test Saa * Sbb - Sab^2 ≈ 0.0 atol = 1e-9
            @test Taa ≈ w * Saa atol = 1e-12
            @test Tab ≈ w * Sab atol = 1e-12
            @test Tbb ≈ w * Sbb atol = 1e-12
            # a ≈ 1, b ≈ 0 for the east beam: S_aa ≈ w, S_bb ≈ 0, S_ab ≈ 0.
            @test Saa ≈ w atol = 1e-3
            @test Sbb ≈ 0.0 atol = 1e-3
            @test Sab ≈ 0.0 atol = 1e-3
            # AtWb = w·v_r·g ⇒ S_av ≈ w·v_r, S_bv ≈ 0.
            @test Sav ≈ w * vr atol = 1e-3
            @test Sbv ≈ 0.0 atol = 1e-3
        end
    end

    @testset "single gate → coefficient roles (north beam)" begin
        p = DaishoParameters()
        lat0, lon0 = 16.0, -24.0
        vr = -7.0
        # North-pointing beam (az=0°): a ≈ 0, b ≈ 1, so v is well-determined.
        spec = GridSpec(shape = :volume_3d,
            reference_latitude = lat0, reference_longitude = lon0,
            x_axis = [-2000.0, 0.0, 2000.0],
            y_axis = [18000.0, 20000.0, 22000.0],
            z_axis = [0.0, 47.2])
        acc = WindGridAccumulator(spec, "VEL")
        sweep = _one_gate_sweep(az = 0.0, el = 0.0, r = 20000.0, vr = vr)
        grid_sweep_wind!(acc, sweep, p;
            ref_latitude = lat0, ref_longitude = lon0, ref_altitude = 0.0)
        # The gate lands on the (y≈20 km, x=0) column: (j=2, i=2).
        @test acc.gate_count[1, 2, 2] == Int32(1)
        w = acc.weight_total[1, 2, 2]
        @test acc.AtWA[3, 1, 2, 2] ≈ w atol = 1e-3   # S_bb ≈ w (b ≈ 1)
        @test acc.AtWA[1, 1, 2, 2] ≈ 0.0 atol = 1e-3 # S_aa ≈ 0 (a ≈ 0)
        @test acc.AtWb[2, 1, 2, 2] ≈ w * vr atol = 1e-3
        @test acc.AtWb[1, 1, 2, 2] ≈ 0.0 atol = 1e-3
    end

    # Grid sized to actually capture synthetic_volume's low, near-range gates
    # (ranges 400–1500 m, el 1–2° ⇒ beam heights ~7–55 m).
    function _near_radar_spec(v)
        return GridSpec(shape = :volume_3d,
            reference_latitude = v.latitude, reference_longitude = v.longitude,
            x_axis = collect(Float64, -1200.0:300.0:1200.0),
            y_axis = collect(Float64, -1200.0:300.0:1200.0),
            z_axis = [0.0, 60.0, 120.0])
    end

    @testset "two sweeps accumulate additively" begin
        p = DaishoParameters()
        v = synthetic_volume(n_sweeps = 2, n_rays = 72, n_gates = 12)
        spec = _near_radar_spec(v)

        comb = WindGridAccumulator(spec, "VEL")
        grid_sweep_wind!(comb, v, 1, p)
        grid_sweep_wind!(comb, v, 2, p)

        a = WindGridAccumulator(spec, "VEL"); grid_sweep_wind!(a, v, 1, p)
        b = WindGridAccumulator(spec, "VEL"); grid_sweep_wind!(b, v, 2, p)

        @test comb.AtWA == a.AtWA .+ b.AtWA
        @test comb.AtWb == a.AtWb .+ b.AtWb
        @test comb.M2 == a.M2 .+ b.M2
        @test comb.weight_total == a.weight_total .+ b.weight_total
        @test comb.gate_count == a.gate_count .+ b.gate_count
        @test length(comb.sweeps) == 2
        # Something actually landed (guards against an all-zero vacuous pass).
        @test sum(comb.gate_count) > 0
    end

    @testset "missing velocity gates are skipped" begin
        p = DaishoParameters()
        v = synthetic_volume(n_sweeps = 1, n_rays = 72, n_gates = 12)
        spec = _near_radar_spec(v)

        full = WindGridAccumulator(spec, "VEL"); grid_sweep_wind!(full, v, 1, p)
        @test sum(full.gate_count) > 0   # baseline has coverage

        # All-missing velocity ⇒ nothing accumulates.
        vnan = deepcopy(v)
        fill!(vnan.sweeps[1].fields["VEL"].data, NaN32)
        none = WindGridAccumulator(spec, "VEL"); grid_sweep_wind!(none, vnan, 1, p)
        @test all(none.AtWA .== 0.0)
        @test all(none.AtWb .== 0.0)
        @test all(none.M2 .== 0.0)
        @test all(none.weight_total .== 0.0)
        @test all(none.gate_count .== Int32(0))

        # Half-missing velocity ⇒ strictly fewer contributing gates.
        vhalf = deepcopy(v)
        d = vhalf.sweeps[1].fields["VEL"].data
        d[1:2:end, :] .= NaN32
        half = WindGridAccumulator(spec, "VEL"); grid_sweep_wind!(half, vhalf, 1, p)
        @test sum(half.gate_count) < sum(full.gate_count)
        @test sum(half.gate_count) > 0
    end

    @testset "shared gate-geometry kernel parity" begin
        # The kernel must reproduce the exact inline formula the scalar gridding
        # path used before extraction. Hold a reference copy here and compare.
        function _ref_geometry(grid_z, yx_point, radar_zyx, beams, g,
                               h_roi, v_roi, pthr)
            dz = grid_z - radar_zyx[g][1]
            r  = beams[g, 3]
            sine_h = ((dz + Daisho.Reff)^2 - r^2 - Daisho.Reff^2) / (2 * r * Daisho.Reff)
            abs(sine_h) < 1.0 || return (NaN, NaN, 0.0)
            el = asin(sine_h)
            dx = yx_point[2] - radar_zyx[g][3]
            dy = yx_point[1] - radar_zyx[g][2]
            az = (pi / 2.0) - atan(dy, dx)
            az < 0 && (az += 2 * pi)
            ad = Daisho.spherical_angle([beams[g, 1], beams[g, 2]], [az, el])
            aw = exp(-ad * 79.43)
            aw < pthr && (aw = 0.0)
            gr = sin(sqrt(dx^2 + dy^2) / Daisho.Reff) * (Daisho.Reff + dz) / cos(el)
            rw = gr / r
            (abs(gr - r) > h_roi || abs(gr - r) > v_roi) && (rw = 0.0)
            return (az, el, rw * aw)
        end

        radar_zyx = [[0.0, 0.0, 0.0], [0.0, 5000.0, -3000.0]]
        beams = [deg2rad(30.0) deg2rad(0.5) 12000.0 200.0;
                 deg2rad(280.0) deg2rad(1.5) 9000.0 250.0]
        h_roi, v_roi, pthr = 1500.0, 1500.0, 0.5
        for g in 1:2, gz in (100.0, 400.0, 1200.0),
                yx in ([6000.0, 4000.0], [-1000.0, 8000.0], [200.0, 200.0])
            got = Daisho._gate_grid_geometry(gz, yx, radar_zyx, beams, g,
                                             h_roi, v_roi, pthr)
            ref = _ref_geometry(gz, yx, radar_zyx, beams, g, h_roi, v_roi, pthr)
            for k in 1:3
                if isnan(ref[k])
                    @test isnan(got[k])
                else
                    @test got[k] ≈ ref[k] atol=1e-12
                end
            end
        end
    end

end
