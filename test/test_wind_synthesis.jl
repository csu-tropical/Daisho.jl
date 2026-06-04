using TOML

# Stub non-identity output frame for the §2.4 frame-parity test. Defined at top
# level because Julia structs cannot be declared inside a @testset's local scope.
struct _RotatedFrame <: Daisho.SynthesisFrame
    θ::Float64
end
Daisho.component_names(::_RotatedFrame) = ("C1", "C2")
Daisho.rotation_at(f::_RotatedFrame, x::Real, y::Real, z::Real) =
    [cos(f.θ) -sin(f.θ); sin(f.θ) cos(f.θ)]

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

    # ── Phase 3: solve, sandwich error, frame rotation, masking ─────────────

    # A (1,1,1)-grid accumulator with one cell set directly, to test the solve
    # in isolation from gridding.
    function _cell_acc(AtWA3, AtWb2, M23; w = 1.0, n = 2,
                       fv = -32768.0, ud = -9999.0)
        spec = GridSpec(shape = :volume_3d, reference_latitude = 0.0,
                        reference_longitude = 0.0,
                        x_axis = [0.0], y_axis = [0.0], z_axis = [0.0])
        acc = WindGridAccumulator(spec, "VEL"; fill_value = fv, undetect = ud)
        acc.AtWA[:, 1, 1, 1] .= AtWA3
        acc.AtWb[:, 1, 1, 1] .= AtWb2
        acc.M2[:, 1, 1, 1]   .= M23
        acc.weight_total[1, 1, 1] = w
        acc.gate_count[1, 1, 1]   = Int32(n)
        return acc
    end

    function _p_synth(; var = 1.0, ms = [2.0, 2.0])
        p0 = DaishoParameters()
        return Daisho.DaishoParameters(p0.moments, p0.qc, p0.gridding, p0.grid,
            p0.io, SynthesisParameters(velocity_variance = var, max_sigma = ms),
            p0.provided)
    end

    # Independent reference solve for a single cell (§2.3).
    function _ref_solve(Saa, Sab, Sbb, Sav, Sbv, Taa, Tab, Tbb, var)
        D = Saa * Sbb - Sab^2
        c11 = Sbb / D; c12 = -Sab / D; c22 = Saa / D
        u = c11 * Sav + c12 * Sbv
        v = c12 * Sav + c22 * Sbv
        p11 = c11 * Taa + c12 * Tab; p12 = c11 * Tab + c12 * Tbb
        p21 = c12 * Taa + c22 * Tab; p22 = c12 * Tab + c22 * Tbb
        cov11 = var * (p11 * c11 + p12 * c12)
        cov22 = var * (p21 * c12 + p22 * c22)
        return (u, v, sqrt(cov11), sqrt(cov22), D)
    end

    @testset "solve + general sandwich covariance" begin
        Saa, Sab, Sbb = 1.36, 0.48, 0.64
        Sav, Sbv = 2.5, -1.1
        Taa, Tab, Tbb = 0.9, 0.3, 0.5   # M2 ≠ AtWA (genuinely weighted)
        var = 1.7
        acc = _cell_acc([Saa, Sab, Sbb], [Sav, Sbv], [Taa, Tab, Tbb]; w = 3.0, n = 4)
        out = finalize_wind(acc, _p_synth(var = var, ms = [1e9, 1e9]))
        u, v, s1, s2, D = _ref_solve(Saa, Sab, Sbb, Sav, Sbv, Taa, Tab, Tbb, var)
        @test out.comp1[1, 1, 1] ≈ u   atol = 1e-10
        @test out.comp2[1, 1, 1] ≈ v   atol = 1e-10
        @test out.sigma1[1, 1, 1] ≈ s1 atol = 1e-10
        @test out.sigma2[1, 1, 1] ≈ s2 atol = 1e-10
        @test out.det[1, 1, 1] ≈ D     atol = 1e-10
        @test out.quality_flag[1, 1, 1] == Int8(0)   # thresholds wide open
        @test out.n_gates[1, 1, 1] == Int32(4)
        @test out.component_names == ("U", "V")
    end

    @testset "equal-weight reduction: cov_cart == σ²·Ci" begin
        # Unit-weight gates ⇒ M2 == AtWA, and the sandwich collapses to σ²·Ci.
        Saa, Sab, Sbb = 1.36, 0.48, 0.64
        var = 2.0
        acc = _cell_acc([Saa, Sab, Sbb], [3.0, 4.0], [Saa, Sab, Sbb]; w = 2.0, n = 2)
        out = finalize_wind(acc, _p_synth(var = var, ms = [1e9, 1e9]))
        D = Saa * Sbb - Sab^2
        @test out.sigma1[1, 1, 1]^2 ≈ var * (Sbb / D) atol = 1e-10
        @test out.sigma2[1, 1, 1]^2 ≈ var * (Saa / D) atol = 1e-10
    end

    @testset "baseline singularity vs near-baseline (non-destructive)" begin
        p = _p_synth(var = 1.0, ms = [2.0, 2.0])
        # Single look direction ⇒ rank-1 AtWA with D == 0 exactly ⇒ singular,
        # components blanked. (AtWA=[1,1,1] is the gg of a 45° unit look; the
        # determinant 1·1 − 1·1 = 0 is exact in Float64.)
        AtWA = [1.0, 1.0, 1.0]
        acc_sing = _cell_acc(AtWA, [5.0, 5.0], AtWA; w = 1.0, n = 1)
        out_s = finalize_wind(acc_sing, p)
        @test out_s.quality_flag[1, 1, 1] == Daisho.QFLAG_SINGULAR
        @test out_s.comp1[1, 1, 1] == acc_sing.fill_value
        @test out_s.comp2[1, 1, 1] == acc_sing.fill_value
        @test out_s.sigma1[1, 1, 1] == acc_sing.fill_value
        @test out_s.det[1, 1, 1] ≈ 0.0 atol = 1e-9

        # Near-baseline: two almost-parallel looks ⇒ small but positive D ⇒
        # large σ. Components and σ are still WRITTEN (non-destructive); only the
        # σ-threshold bits are set.
        g1 = (sin(deg2rad(10.0)), cos(deg2rad(10.0)))
        g2 = (sin(deg2rad(11.0)), cos(deg2rad(11.0)))
        Saa = g1[1]^2 + g2[1]^2; Sab = g1[1]*g1[2] + g2[1]*g2[2]
        Sbb = g1[2]^2 + g2[2]^2
        acc_near = _cell_acc([Saa, Sab, Sbb], [0.5, 0.5], [Saa, Sab, Sbb]; w = 2.0, n = 2)
        out_n = finalize_wind(acc_near, p)
        @test out_n.det[1, 1, 1] > 0.0
        @test out_n.comp1[1, 1, 1] != acc_near.fill_value   # preserved
        @test isfinite(out_n.sigma1[1, 1, 1])
        @test isfinite(out_n.sigma2[1, 1, 1])
        @test out_n.quality_flag[1, 1, 1] != Int8(0)         # at least one σ bit
        @test (out_n.quality_flag[1, 1, 1] & Daisho.QFLAG_SINGULAR) == Int8(0)
    end

    @testset "independent per-component σ gating" begin
        # East-heavy geometry: u well-determined (σ1 small), v poorly (σ2 large).
        Saa = 3.09; Sab = 0.285; Sbb = 0.9025
        D = Saa * Sbb - Sab^2
        s1 = sqrt(Sbb / D); s2 = sqrt(Saa / D)         # var = 1, M2 = AtWA
        @test s1 < s2
        acc = _cell_acc([Saa, Sab, Sbb], [1.0, 1.0], [Saa, Sab, Sbb]; w = 4.0, n = 4)
        # Threshold between σ1 and σ2 ⇒ only the σ2 bit trips.
        mid = 0.5 * (s1 + s2)
        out = finalize_wind(acc, _p_synth(var = 1.0, ms = [mid, mid]))
        @test (out.quality_flag[1, 1, 1] & Daisho.QFLAG_SIGMA1) == Int8(0)
        @test (out.quality_flag[1, 1, 1] & Daisho.QFLAG_SIGMA2) != Int8(0)
        # Tighten only component 1 ⇒ only the σ1 bit trips (independent).
        out1 = finalize_wind(acc, _p_synth(var = 1.0, ms = [s1 * 0.5, 1e9]))
        @test (out1.quality_flag[1, 1, 1] & Daisho.QFLAG_SIGMA1) != Int8(0)
        @test (out1.quality_flag[1, 1, 1] & Daisho.QFLAG_SIGMA2) == Int8(0)
    end

    @testset "no-data flag where nothing accumulated" begin
        acc = _cell_acc([0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0, 0.0]; w = 0.0, n = 0)
        out = finalize_wind(acc, _p_synth())
        @test out.quality_flag[1, 1, 1] == Daisho.QFLAG_NODATA
        @test out.comp1[1, 1, 1] == acc.fill_value
        @test out.n_gates[1, 1, 1] == Int32(0)
    end

    @testset "frame rotation == rotating accumulated matrices (§2.4)" begin
        # finalize(acc, RotatedFrame θ) must equal finalize(acc', Cartesian)
        # where acc' carries the rotated coefficients (AtWA'=R·AtWA·Rᵀ,
        # AtWb'=R·AtWb, M2'=R·M2·Rᵀ).
        θ = deg2rad(37.0)
        cθ, sθ = cos(θ), sin(θ)
        Saa, Sab, Sbb = 1.36, 0.48, 0.64
        Sav, Sbv = 2.5, -1.1
        Taa, Tab, Tbb = 0.9, 0.3, 0.5
        var = 1.3
        p = _p_synth(var = var, ms = [1e9, 1e9])

        acc = _cell_acc([Saa, Sab, Sbb], [Sav, Sbv], [Taa, Tab, Tbb]; w = 3.0, n = 4)
        out_frame = finalize_wind(acc, p; frame = _RotatedFrame(θ))

        # Rotate the symmetric packed matrices: M' = R·M·Rᵀ, with R=[cθ -sθ; sθ cθ].
        function _rot_packed(Maa, Mab, Mbb)
            # full M
            m11, m12, m22 = Maa, Mab, Mbb
            # RM
            a11 = cθ * m11 - sθ * m12; a12 = cθ * m12 - sθ * m22
            a21 = sθ * m11 + cθ * m12; a22 = sθ * m12 + cθ * m22
            # (RM)Rᵀ ; Rᵀ = [cθ sθ; -sθ cθ]
            n11 = a11 * cθ - a12 * sθ
            n12 = a11 * sθ + a12 * cθ
            n22 = a21 * sθ + a22 * cθ
            return (n11, n12, n22)
        end
        rAtWA = collect(_rot_packed(Saa, Sab, Sbb))
        rM2   = collect(_rot_packed(Taa, Tab, Tbb))
        rAtWb = [cθ * Sav - sθ * Sbv, sθ * Sav + cθ * Sbv]
        acc2 = _cell_acc(rAtWA, rAtWb, rM2; w = 3.0, n = 4)
        out_cart = finalize_wind(acc2, p)

        @test out_frame.component_names == ("C1", "C2")
        @test out_frame.comp1[1, 1, 1]  ≈ out_cart.comp1[1, 1, 1]  atol = 1e-9
        @test out_frame.comp2[1, 1, 1]  ≈ out_cart.comp2[1, 1, 1]  atol = 1e-9
        @test out_frame.sigma1[1, 1, 1] ≈ out_cart.sigma1[1, 1, 1] atol = 1e-9
        @test out_frame.sigma2[1, 1, 1] ≈ out_cart.sigma2[1, 1, 1] atol = 1e-9
    end

    @testset "analytic uniform-wind recovery (two crossing radars)" begin
        p = _p_synth(var = 1.0, ms = [5.0, 5.0])
        u_true, v_true = 8.0, -3.0
        # Multi-elevation PPIs from a southern and a western radar (~90° beam
        # crossing near the origin) give well-conditioned points along the
        # central column.
        volA, volB = make_synthetic_dual_doppler(u = u_true, v = v_true,
            ref_lat = 16.0, ref_lon = -24.0)
        spec = GridSpec(shape = :volume_3d,
            reference_latitude = 16.0, reference_longitude = -24.0,
            x_axis = [-2000.0, 0.0, 2000.0],
            y_axis = [-2000.0, 0.0, 2000.0],
            z_axis = collect(Float64, 200.0:200.0:1600.0))
        acc = WindGridAccumulator(spec, "VEL")
        for s in eachindex(volA.sweeps); grid_sweep_wind!(acc, volA, s, p); end
        for s in eachindex(volB.sweeps); grid_sweep_wind!(acc, volB, s, p); end
        out = finalize_wind(acc, p)

        nz, ny, nx = size(out.comp1)
        nsolved = 0
        for ix in CartesianIndices((nz, ny, nx))
            if out.quality_flag[ix] == Int8(0)
                nsolved += 1
                @test out.comp1[ix] ≈ u_true atol = 0.1
                @test out.comp2[ix] ≈ v_true atol = 0.1
            end
        end
        @test nsolved > 0          # genuinely recovered some well-conditioned points
        @test length(acc.sweeps) == length(volA.sweeps) + length(volB.sweeps)
    end

    # ── Phase 4: persistence + multi-file merge ─────────────────────────────

    @testset "JLD2 round-trip" begin
        spec = _wind_spec(:volume_3d)
        acc = WindGridAccumulator(spec, "VEL"; fill_value = -777.0, undetect = -8888.0)
        acc.AtWA[:, 1, 1, 1] .= [1.5, 0.3, 2.0]
        acc.AtWb[:, 1, 1, 1] .= [4.0, -2.0]
        acc.M2[:, 2, 3, 4]   .= [0.5, 0.1, 0.9]
        acc.weight_total[1, 1, 1] = 3.0
        acc.gate_count[1, 1, 1]   = Int32(5)
        push!(acc.sweeps, SweepProvenance(instrument_name = "RADARA",
            sweep_number = 0, fixed_angle = 0.5,
            time_start = DateTime(2024, 1, 1, 0, 0, 0)))

        path = tempname() * ".jld2"
        try
            save_wind_accumulator(path, acc)
            rt = load_wind_accumulator(path)
            @test rt.velocity_field == acc.velocity_field
            @test rt.n_unknowns == acc.n_unknowns
            @test rt.AtWA == acc.AtWA
            @test rt.AtWb == acc.AtWb
            @test rt.M2 == acc.M2
            @test rt.weight_total == acc.weight_total
            @test rt.gate_count == acc.gate_count
            @test rt.fill_value == -777.0
            @test rt.undetect == -8888.0
            @test rt.schema_version == Daisho.WIND_ACCUMULATOR_SCHEMA_VERSION
            @test rt.grid_spec.x_axis == acc.grid_spec.x_axis
            @test length(rt.sweeps) == 1
            @test rt.sweeps[1].instrument_name == "RADARA"
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "merge equivalence: grid-together == grid-separately-then-merge" begin
        p = DaishoParameters()
        v = synthetic_volume(n_sweeps = 2, n_rays = 72, n_gates = 12)
        spec = _near_radar_spec(v)

        together = WindGridAccumulator(spec, "VEL")
        grid_sweep_wind!(together, v, 1, p)
        grid_sweep_wind!(together, v, 2, p)

        a = WindGridAccumulator(spec, "VEL"); grid_sweep_wind!(a, v, 1, p)
        b = WindGridAccumulator(spec, "VEL"); grid_sweep_wind!(b, v, 2, p)
        merge_wind_accumulators!(a, b)

        @test a.AtWA == together.AtWA
        @test a.AtWb == together.AtWb
        @test a.M2 == together.M2
        @test a.weight_total == together.weight_total
        @test a.gate_count == together.gate_count
        @test length(a.sweeps) == 2
        @test sum(together.gate_count) > 0   # non-vacuous

        # The merged single-stage products agree too.
        oa = finalize_wind(a, p); ot = finalize_wind(together, p)
        @test oa.comp1 == ot.comp1
        @test oa.sigma2 == ot.sigma2
        @test oa.quality_flag == ot.quality_flag
    end

    @testset "merge strict compatibility" begin
        spec = _wind_spec(:volume_3d)
        dst = WindGridAccumulator(spec, "VEL")
        # velocity_field mismatch.
        @test_throws ArgumentError merge_wind_accumulators!(
            dst, WindGridAccumulator(spec, "VR"))
        # n_unknowns mismatch.
        @test_throws ArgumentError merge_wind_accumulators!(
            dst, WindGridAccumulator(spec, "VEL"; n_unknowns = 3))
        # grid_spec mismatch.
        @test_throws ArgumentError merge_wind_accumulators!(
            dst, WindGridAccumulator(_wind_spec(:volume_3d; n_x = 5), "VEL"))
        # [io] sentinel mismatch.
        @test_throws ArgumentError merge_wind_accumulators!(
            dst, WindGridAccumulator(spec, "VEL"; fill_value = -1.0))
    end

    # ── Phase 5: NetCDF output ──────────────────────────────────────────────

    @testset "write_wind_synthesis NetCDF output" begin
        p = _p_synth(var = 1.0, ms = [5.0, 5.0])
        volA, volB = make_synthetic_dual_doppler(u = 8.0, v = -3.0)
        spec = GridSpec(shape = :volume_3d,
            reference_latitude = 16.0, reference_longitude = -24.0,
            x_axis = [-2000.0, 0.0, 2000.0],
            y_axis = [-2000.0, 0.0, 2000.0],
            z_axis = collect(Float64, 200.0:200.0:1600.0))
        acc = WindGridAccumulator(spec, "VEL")
        for s in eachindex(volA.sweeps); grid_sweep_wind!(acc, volA, s, p); end
        for s in eachindex(volB.sweeps); grid_sweep_wind!(acc, volB, s, p); end
        out = finalize_wind(acc, p)

        nx, ny, nz = length(spec.x_axis), length(spec.y_axis), length(spec.z_axis)
        file = tempname() * ".nc"
        try
            write_wind_synthesis(file, out, p; index_time = DateTime(2024, 1, 1, 0, 0, 0))
            ds = NCDataset(file, "r")
            try
                @test ds.dim["X"] == nx
                @test ds.dim["Y"] == ny
                @test ds.dim["Z"] == nz
                @test ds.dim["time"] == 1
                for vn in ("U", "V", "USTD", "VSTD", "DET", "NGATES", "QFLAG",
                           "X", "Y", "Z", "time", "latitude", "longitude", "grid_mapping")
                    @test haskey(ds, vn)
                end
                @test size(ds["U"]) == (nx, ny, nz, 1)
                @test ds["U"].attrib["standard_name"] == "eastward_wind"
                @test ds["V"].attrib["standard_name"] == "northward_wind"
                @test haskey(ds["U"].attrib, "_FillValue")
                @test ds["QFLAG"].attrib["flag_masks"] == Int8[1, 2, 4, 8]
                @test eltype(ds["NGATES"].var[:, :, :, 1]) <: Integer
                @test ds["grid_mapping"].attrib["grid_mapping_name"] == "transverse_mercator"
                # Round-trip a recovered value at a solved cell.
                solved = nothing
                for ci in CartesianIndices(out.quality_flag)
                    if out.quality_flag[ci] == Int8(0); solved = ci; break; end
                end
                @test solved !== nothing
                if solved !== nothing
                    k, j, i = solved.I
                    @test Float32(out.comp1[k, j, i]) ≈ ds["U"][i, j, k, 1] atol = 1e-4
                    @test Float32(out.comp2[k, j, i]) ≈ ds["V"][i, j, k, 1] atol = 1e-4
                end
            finally
                close(ds)
            end
        finally
            isfile(file) && rm(file)
        end
    end

end
