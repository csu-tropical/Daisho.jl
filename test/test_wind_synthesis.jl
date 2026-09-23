using TOML

# Stub non-identity output frame for the §2.4 frame-parity test. Defined at top
# level because Julia structs cannot be declared inside a @testset's local scope.
struct _RotatedFrame <: Daisho.SynthesisFrame
    θ::Float64
end
Daisho.component_names(::_RotatedFrame) = ("C1", "C2")
Daisho.rotation_at(f::_RotatedFrame, x::Real, y::Real, z::Real) =
    [cos(f.θ) -sin(f.θ); sin(f.θ) cos(f.θ)]

# Reference copy of the pre-unification `_grid_sweep_wind_3d!` worker, writing
# into caller-supplied planes. Held here so the unified traversal's wind output
# can be asserted byte-identical to the standalone wind worker it replaced.
function _ref_grid_sweep_wind_3d!(AtWA, AtWb, M2, weight_total, gate_count,
        grid_spec, velkey, N, sweep, p, ref_latitude, ref_longitude, ref_altitude)
    g = grid_spec; gd = p.gridding
    TM = Daisho.CoordRefSystems.shift(
        Daisho.TransverseMercator{1.0, g.reference_latitude, Daisho.WGS84Latest},
        lonₒ = g.reference_longitude)
    grid_origin, radar_zyx, beams, n_gates_s, _ =
        Daisho._sweep_zyx_and_beams(sweep, ref_latitude, ref_longitude, ref_altitude, TM)
    gate_yx  = Daisho._sweep_gate_yx(radar_zyx, beams)
    balltree = Daisho.BallTree(gate_yx)
    gridpoints = Daisho._materialize_gridpoints_3d(g, TM, grid_origin)
    nx = length(g.x_axis); ny = length(g.y_axis); nz = length(g.z_axis)
    beam_coef = Daisho.BEAM_COEF_1DEG
    horizontal_roi = if g.shape === :latlon_3d
        latrad = g.reference_latitude * pi / 180.0
        fac_lat = 111.13209 - 0.56605 * cos(2.0 * latrad)
        fac_lon = 111.41513 * cos(latrad)
        deg_km = sqrt(fac_lat^2 + fac_lon^2)
        degincr = ny >= 2 ? (g.lat_axis[2]-g.lat_axis[1]) : (nx>=2 ? (g.lon_axis[2]-g.lon_axis[1]) : 0.01)
        deg_km * 1000.0 * degincr * gd.horizontal_roi_factor
    else
        xincr = nx >= 2 ? (g.x_axis[2]-g.x_axis[1]) : 0.0; xincr * gd.horizontal_roi_factor
    end
    zincr = nz >= 2 ? (g.z_axis[2]-g.z_axis[1]) : 0.0
    vertical_roi = zincr * gd.vertical_roi_factor
    power_threshold = gd.power_threshold
    beam_cutoff = Daisho._beam_cutoff(power_threshold, beam_coef)
    s = sin(beam_cutoff)
    range_guard_min = gd.range_guard_min; range_weight_max = gd.range_weight_max
    for ii in CartesianIndices((ny, nx))
        j_y, i_x = ii.I
        yx_point = [gridpoints[1, j_y, i_x, 2], gridpoints[1, j_y, i_x, 3]]
        origin_dist = Daisho.euclidean(yx_point, [0.0, 0.0])
        R_q = (horizontal_roi + origin_dist * s) / (1.0 - s)
        gates = Daisho.inrange(balltree, yx_point, R_q)
        isempty(gates) && continue
        gcoef = Vector{Float64}(undef, N)
        for k_z in 1:nz
            grid_z = gridpoints[k_z, j_y, i_x, 1]
            for g_flat in gates
                ray = Daisho._ray_of(g_flat,n_gates_s); gate = Daisho._gate_in_ray(g_flat,n_gates_s)
                vr = Daisho._gate_value(sweep, velkey, ray, gate)
                ismissing(vr) && continue
                az, el, w = Daisho._gate_grid_geometry(grid_z, yx_point, radar_zyx, beams,
                    gate_yx, g_flat, horizontal_roi, vertical_roi, beam_cutoff, beam_coef,
                    range_guard_min, range_weight_max)
                w > 0.0 || continue
                abs(el) > deg2rad(p.synthesis.max_elevation) && continue
                Daisho._wind_coeffs!(gcoef, az, el, N)
                for a in 1:N, b in a:N
                    pk = Daisho._packed_index(a,b,N); gg = gcoef[a]*gcoef[b]
                    AtWA[pk,k_z,j_y,i_x] += w*gg; M2[pk,k_z,j_y,i_x] += w*w*gg
                end
                for a in 1:N; AtWb[a,k_z,j_y,i_x] += w*vr*gcoef[a]; end
                weight_total[k_z,j_y,i_x] += w; gate_count[k_z,j_y,i_x] += Int32(1)
            end
        end
    end
    return nothing
end

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
            @test isempty(acc.scalar.sweeps)
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
        @test p.synthesis.max_elevation == 45.0

        # Omitted → defaults, not in provided.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            delete!(base, "synthesis")
            open(path, "w") do f; TOML.print(f, base); end
            p2 = DaishoParameters(path)
            @test p2.synthesis.velocity_variance == SynthesisParameters().velocity_variance
            @test p2.synthesis.max_sigma == SynthesisParameters().max_sigma
            @test p2.synthesis.max_elevation == SynthesisParameters().max_elevation
            @test !(:synthesis in p2.provided)
        end

        # Custom values round-trip; max_elevation is optional (defaults when absent).
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["synthesis"] = Dict("velocity_variance" => 4.0,
                                     "max_sigma_1" => 1.5, "max_sigma_2" => 3.0)
            open(path, "w") do f; TOML.print(f, base); end
            p3 = DaishoParameters(path)
            @test p3.synthesis.velocity_variance == 4.0
            @test p3.synthesis.max_sigma == [1.5, 3.0]
            @test p3.synthesis.max_elevation == 45.0   # absent ⇒ default
        end

        # Explicit max_elevation round-trips.
        mktemp() do path, io
            close(io)
            base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
            base["synthesis"]["max_elevation"] = 30.0
            open(path, "w") do f; TOML.print(f, base); end
            @test DaishoParameters(path).synthesis.max_elevation == 30.0
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
        grid_sweep!(acc, sweep, p;
            ref_latitude = lat0, ref_longitude = lon0, ref_altitude = 0.0)

        # Exactly the (j=2, i=2) column receives the gate (both z levels).
        for ix in CartesianIndices((2, 3, 3))
            k, j, i = ix.I
            expected = (j == 2 && i == 2) ? Int32(1) : Int32(0)
            @test acc.gate_count[k, j, i] == expected
        end
        @test length(acc.scalar.sweeps) == 1

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
        grid_sweep!(acc, sweep, p;
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
        grid_sweep!(comb, v, 1, p)
        grid_sweep!(comb, v, 2, p)

        a = WindGridAccumulator(spec, "VEL"); grid_sweep!(a, v, 1, p)
        b = WindGridAccumulator(spec, "VEL"); grid_sweep!(b, v, 2, p)

        # Edge-referenced inclusion lets a cell receive gates from BOTH sweeps,
        # so combine-then-sum and sum-then-combine differ only by floating-point
        # association (~1e-16); the additive property holds to float precision.
        # The integer gate_count is exactly additive.
        @test comb.AtWA ≈ a.AtWA .+ b.AtWA
        @test comb.AtWb ≈ a.AtWb .+ b.AtWb
        @test comb.M2 ≈ a.M2 .+ b.M2
        @test comb.weight_total ≈ a.weight_total .+ b.weight_total
        @test comb.gate_count == a.gate_count .+ b.gate_count
        @test length(comb.scalar.sweeps) == 2
        # Something actually landed (guards against an all-zero vacuous pass).
        @test sum(comb.gate_count) > 0
    end

    @testset "missing velocity gates are skipped" begin
        p = DaishoParameters()
        v = synthetic_volume(n_sweeps = 1, n_rays = 72, n_gates = 12)
        spec = _near_radar_spec(v)

        full = WindGridAccumulator(spec, "VEL"); grid_sweep!(full, v, 1, p)
        @test sum(full.gate_count) > 0   # baseline has coverage

        # All-missing velocity ⇒ nothing accumulates.
        vnan = deepcopy(v)
        fill!(vnan.sweeps[1].fields["VEL"].data, NaN32)
        none = WindGridAccumulator(spec, "VEL"); grid_sweep!(none, vnan, 1, p)
        @test all(none.AtWA .== 0.0)
        @test all(none.AtWb .== 0.0)
        @test all(none.M2 .== 0.0)
        @test all(none.weight_total .== 0.0)
        @test all(none.gate_count .== Int32(0))

        # Half-missing velocity ⇒ strictly fewer contributing gates.
        vhalf = deepcopy(v)
        d = vhalf.sweeps[1].fields["VEL"].data
        d[1:2:end, :] .= NaN32
        half = WindGridAccumulator(spec, "VEL"); grid_sweep!(half, vhalf, 1, p)
        @test sum(half.gate_count) < sum(full.gate_count)
        @test sum(half.gate_count) > 0
    end

    @testset "shared gate-geometry kernel parity" begin
        # The kernel must reproduce the edge-referenced inclusion + weight math
        # (§3 of the Edge-Referenced ROI plan). Hold an independent reference copy
        # here and compare: footprint-based d_h/d_v inclusion, no per-gate power
        # cut, range_weight with the radial tolerance.
        function _ref_geometry(grid_z, yx_point, radar_zyx, beams, gate_yx, g,
                               h_roi, v_roi, beam_cutoff, beam_coef)
            dz = grid_z - radar_zyx[g][1]
            r  = beams[g, 3]
            sine_h = ((dz + Daisho.Reff)^2 - r^2 - Daisho.Reff^2) / (2 * r * Daisho.Reff)
            abs(sine_h) < 1.0 || return (NaN, NaN, 0.0)
            el = asin(sine_h)
            dx = yx_point[2] - radar_zyx[g][3]
            dy = yx_point[1] - radar_zyx[g][2]
            az = (pi / 2.0) - atan(dy, dx)
            az < 0 && (az += 2 * pi)
            footprint = r * sin(beam_cutoff)
            d_h = sqrt((gate_yx[1, g] - yx_point[1])^2 + (gate_yx[2, g] - yx_point[2])^2)
            d_v = abs(beams[g, 4] - grid_z)
            (d_h <= h_roi + footprint && d_v <= v_roi + footprint) ||
                return (az, el, 0.0)
            ad = Daisho.spherical_angle([beams[g, 1], beams[g, 2]], [az, el])
            aw = exp(-ad * beam_coef)
            gr = sin(sqrt(dx^2 + dy^2) / Daisho.Reff) * (Daisho.Reff + dz) / cos(el)
            rw = gr / r
            (abs(gr - r) > h_roi || abs(gr - r) > v_roi) && (rw = 0.0)
            return (az, el, rw * aw)
        end

        radar_zyx = [[0.0, 0.0, 0.0], [0.0, 5000.0, -3000.0]]
        beams = [deg2rad(30.0) deg2rad(0.5) 12000.0 200.0;
                 deg2rad(280.0) deg2rad(1.5) 9000.0 250.0]
        gate_yx = Daisho._sweep_gate_yx(radar_zyx, beams)
        beam_coef = Daisho.BEAM_COEF_1DEG
        # A wide ROI so some (but not all) of the synthetic gates pass inclusion,
        # exercising both branches of the edge-referenced cut.
        h_roi, v_roi = 6000.0, 6000.0
        beam_cutoff = Daisho._beam_cutoff(0.5, beam_coef)
        for g in 1:2, gz in (100.0, 400.0, 1200.0),
                yx in ([6000.0, 4000.0], [-1000.0, 8000.0], [200.0, 200.0])
            got = Daisho._gate_grid_geometry(gz, yx, radar_zyx, beams, gate_yx, g,
                                             h_roi, v_roi, beam_cutoff, beam_coef)
            ref = _ref_geometry(gz, yx, radar_zyx, beams, gate_yx, g,
                                h_roi, v_roi, beam_cutoff, beam_coef)
            for k in 1:3
                if isnan(ref[k])
                    @test isnan(got[k])
                else
                    @test got[k] ≈ ref[k] atol=1e-12
                end
            end
        end
    end

    @testset "wind folds into the shared traversal (scalar + wind from one pass)" begin
        # A WindGridAccumulator run must (a) produce scalar grids byte-identical
        # to a standalone ScalarGridAccumulator run, and (b) produce wind planes
        # byte-identical to the pre-unification standalone wind worker. Together
        # these prove the single geometry pass correctly drives both products.
        p = DaishoParameters()
        v = synthetic_volume(n_sweeps = 2, n_rays = 72, n_gates = 12)
        spec = _near_radar_spec(v)

        wind_acc = WindGridAccumulator(spec, p)
        for s in eachindex(v.sweeps); grid_sweep!(wind_acc, v, s, p); end

        # (a) embedded scalar grids == standalone scalar grids.
        scalar_acc = ScalarGridAccumulator(spec, p)
        for s in eachindex(v.sweeps); grid_sweep!(scalar_acc, v, s, p); end
        @test wind_acc.scalar.fields == scalar_acc.fields
        @test wind_acc.scalar.weighted_sum == scalar_acc.weighted_sum
        @test wind_acc.scalar.weight_total == scalar_acc.weight_total
        @test wind_acc.scalar.coverage     == scalar_acc.coverage
        @test any(scalar_acc.coverage .== Int8(2))   # non-vacuous

        # (b) wind planes == reference standalone wind worker.
        nz, ny, nx = Daisho.wind_accumulator_dims(spec)
        rAtWA = zeros(Float64, 3, nz, ny, nx); rAtWb = zeros(Float64, 2, nz, ny, nx)
        rM2 = zeros(Float64, 3, nz, ny, nx); rwt = zeros(Float64, nz, ny, nx)
        rgc = zeros(Int32, nz, ny, nx)
        for s in eachindex(v.sweeps)
            _ref_grid_sweep_wind_3d!(rAtWA, rAtWb, rM2, rwt, rgc, spec, "VEL", 2,
                v.sweeps[s], p, Float64(v.latitude), Float64(v.longitude), Float64(v.altitude))
        end
        @test wind_acc.AtWA == rAtWA
        @test wind_acc.AtWb == rAtWb
        @test wind_acc.M2 == rM2
        @test wind_acc.weight_total == rwt
        @test wind_acc.gate_count == rgc
        @test sum(wind_acc.gate_count) > 0           # non-vacuous
    end

    @testset "build_accumulator selects mode from config" begin
        p = DaishoParameters()   # bundled template has a :velocity-tagged field
        spec = _wind_spec(:volume_3d)
        acc = build_accumulator(spec, p)
        @test acc isa WindGridAccumulator
        @test acc.velocity_field == "VEL"

        # Drop the velocity tag ⇒ scalar accumulator.
        novel = filter(fs -> !(:velocity in fs.tags), p.moments.fields)
        p2 = Daisho.DaishoParameters(Daisho.MomentParameters(novel),
            p.qc, p.gridding, p.grid, p.io, p.synthesis, p.provided)
        acc2 = build_accumulator(spec, p2)
        @test acc2 isa ScalarGridAccumulator
        @test !(acc2 isa WindGridAccumulator)
    end

    @testset "max_elevation gates steep beams out of the wind solve only" begin
        p0 = DaishoParameters()
        mkp(maxel) = Daisho.DaishoParameters(p0.moments, p0.qc, p0.gridding, p0.grid,
            p0.io, SynthesisParameters(velocity_variance = 1.0, max_sigma = [1e9, 1e9],
                                       max_elevation = maxel), p0.provided)

        # Wind-gate count: the multi-elevation crossing radars give a spread of
        # line-of-sight elevations, so a lower cap admits strictly fewer gates.
        volA, volB = make_synthetic_dual_doppler(u = 8.0, v = -3.0)
        wspec = GridSpec(shape = :volume_3d,
            reference_latitude = 16.0, reference_longitude = -24.0,
            x_axis = [-2000.0, 0.0, 2000.0], y_axis = [-2000.0, 0.0, 2000.0],
            z_axis = collect(Float64, 200.0:200.0:1600.0))
        wgates(maxel) = begin
            acc = WindGridAccumulator(wspec, mkp(maxel))
            for s in eachindex(volA.sweeps); grid_sweep!(acc, volA, s, mkp(maxel)); end
            for s in eachindex(volB.sweeps); grid_sweep!(acc, volB, s, mkp(maxel)); end
            sum(acc.gate_count)
        end
        n90, n2, n0 = wgates(90.0), wgates(2.0), wgates(0.0)
        @test n90 > 0          # baseline: gates contribute to the wind solve
        @test n0 == 0          # cap 0 excludes every gate from the wind
        @test n2 < n90         # an intermediate cap removes the steeper gates
        @test n0 <= n2 <= n90

        # Scalar grids (populated DBZ/SQI volume) are untouched by the wind cap.
        v = synthetic_volume(n_sweeps = 2, n_rays = 72, n_gates = 12)
        sspec = _near_radar_spec(v)
        sc(maxel) = begin
            acc = WindGridAccumulator(sspec, mkp(maxel))
            for s in eachindex(v.sweeps); grid_sweep!(acc, v, s, mkp(maxel)); end
            acc
        end
        a90, a0 = sc(90.0), sc(0.0)
        @test a90.scalar.coverage     == a0.scalar.coverage
        @test a90.scalar.weighted_sum == a0.scalar.weighted_sum
        @test any(a90.scalar.coverage .== Int8(2))            # non-vacuous scalar grid
        @test sum(a0.gate_count) == 0 && sum(a90.gate_count) > 0   # wind still gated
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

    function _p_synth(; var = 1.0, ms = [2.0, 2.0], maxel = 45.0)
        p0 = DaishoParameters()
        return Daisho.DaishoParameters(p0.moments, p0.qc, p0.gridding, p0.grid,
            p0.io, SynthesisParameters(velocity_variance = var, max_sigma = ms,
                                       max_elevation = maxel),
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
        for s in eachindex(volA.sweeps); grid_sweep!(acc, volA, s, p); end
        for s in eachindex(volB.sweeps); grid_sweep!(acc, volB, s, p); end
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
        @test length(acc.scalar.sweeps) == length(volA.sweeps) + length(volB.sweeps)
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
        push!(acc.scalar.sweeps, SweepProvenance(instrument_name = "RADARA",
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
            @test rt.scalar.grid_spec.x_axis == acc.scalar.grid_spec.x_axis
            @test length(rt.scalar.sweeps) == 1
            @test rt.scalar.sweeps[1].instrument_name == "RADARA"
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "merge equivalence: grid-together == grid-separately-then-merge" begin
        p = DaishoParameters()
        v = synthetic_volume(n_sweeps = 2, n_rays = 72, n_gates = 12)
        spec = _near_radar_spec(v)

        together = WindGridAccumulator(spec, "VEL")
        grid_sweep!(together, v, 1, p)
        grid_sweep!(together, v, 2, p)

        a = WindGridAccumulator(spec, "VEL"); grid_sweep!(a, v, 1, p)
        b = WindGridAccumulator(spec, "VEL"); grid_sweep!(b, v, 2, p)
        merge_wind_accumulators!(a, b)

        # Float arrays agree to floating-point association (cells now overlap
        # both sweeps under edge-referenced inclusion); gate_count is exact.
        @test a.AtWA ≈ together.AtWA
        @test a.AtWb ≈ together.AtWb
        @test a.M2 ≈ together.M2
        @test a.weight_total ≈ together.weight_total
        @test a.gate_count == together.gate_count
        @test length(a.scalar.sweeps) == 2
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
        for s in eachindex(volA.sweeps); grid_sweep!(acc, volA, s, p); end
        for s in eachindex(volB.sweeps); grid_sweep!(acc, volB, s, p); end
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
                @test "time" in NCDatasets.unlimited(ds.dim)   # record dim for concatenation
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

    @testset "write_grid_products: wind accumulator → scalar + wind in one file" begin
        p = _p_synth(var = 1.0, ms = [5.0, 5.0])
        volA, volB = make_synthetic_dual_doppler(u = 8.0, v = -3.0)
        spec = GridSpec(shape = :volume_3d,
            reference_latitude = 16.0, reference_longitude = -24.0,
            x_axis = [-2000.0, 0.0, 2000.0], y_axis = [-2000.0, 0.0, 2000.0],
            z_axis = collect(Float64, 200.0:200.0:1600.0))
        acc = build_accumulator(spec, p)          # velocity-tagged ⇒ WindGridAccumulator
        @test acc isa WindGridAccumulator
        for s in eachindex(volA.sweeps); grid_sweep!(acc, volA, s, p); end
        for s in eachindex(volB.sweeps); grid_sweep!(acc, volB, s, p); end
        out = finalize_wind(acc, p)

        nx, ny, nz = length(spec.x_axis), length(spec.y_axis), length(spec.z_axis)
        file = tempname() * ".nc"
        try
            write_grid_products(file, acc, p; index_time = DateTime(2024, 1, 1, 0, 0, 0))
            ds = NCDataset(file, "r")
            try
                @test ds.dim["X"] == nx && ds.dim["Y"] == ny && ds.dim["Z"] == nz
                @test "time" in NCDatasets.unlimited(ds.dim)   # record dim for concatenation
                # Shared coords + wind vars + every configured scalar field.
                for vn in ("U", "V", "USTD", "VSTD", "DET", "NGATES", "QFLAG",
                           "X", "Y", "Z", "time", "latitude", "longitude", "grid_mapping")
                    @test haskey(ds, vn)
                end
                for f in acc.scalar.fields
                    @test haskey(ds, f)
                    @test size(ds[f]) == (nx, ny, nz, 1)
                end
                @test size(ds["U"]) == (nx, ny, nz, 1)
                @test ds["U"].attrib["standard_name"] == "eastward_wind"
                @test ds["V"].attrib["standard_name"] == "northward_wind"
                @test ds["QFLAG"].attrib["flag_masks"] == Int8[1, 2, 4, 8]
                @test ds["grid_mapping"].attrib["grid_mapping_name"] == "transverse_mercator"
                # Round-trip a recovered wind value at a solved cell.
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

    @testset "write_grid_products masks σ-flagged winds (mask_quality)" begin
        # Tight σ thresholds so some solvable cells are flagged (QFLAG != 0).
        p = _p_synth(var = 1.0, ms = [0.5, 0.5])
        volA, volB = make_synthetic_dual_doppler(u = 8.0, v = -3.0)
        spec = GridSpec(shape = :volume_3d,
            reference_latitude = 16.0, reference_longitude = -24.0,
            x_axis = [-2000.0, 0.0, 2000.0], y_axis = [-2000.0, 0.0, 2000.0],
            z_axis = collect(Float64, 200.0:200.0:1600.0))
        acc = WindGridAccumulator(spec, p)
        for s in eachindex(volA.sweeps); grid_sweep!(acc, volA, s, p); end
        for s in eachindex(volB.sweeps); grid_sweep!(acc, volB, s, p); end
        out = finalize_wind(acc, p)
        nflagged = count(!=(Int8(0)), out.quality_flag)
        nsolvable = count(ci -> out.comp1[ci] != out.fill_value, CartesianIndices(out.comp1))
        @test nflagged > 0 && nsolvable > 0   # there is something to mask

        fmask = tempname() * ".nc"; ffull = tempname() * ".nc"
        try
            write_grid_products(fmask, acc, p; index_time = DateTime(2024,1,1))           # mask_quality=true (default)
            write_grid_products(ffull, acc, p; index_time = DateTime(2024,1,1),
                                mask_quality = false)
            dm = NCDataset(fmask, "r"); dfll = NCDataset(ffull, "r")
            try
                # NCDatasets maps _FillValue → missing on read.
                Um = dm["U"][:,:,:,1]; Uf = dfll["U"][:,:,:,1]
                qf = dm["QFLAG"][:,:,:,1]
                # Masked file: U is fill (missing) exactly where QFLAG != 0.
                for ci in CartesianIndices(qf)
                    qf[ci] != Int8(0) && @test ismissing(Um[ci])
                end
                # The full (unmasked) file keeps finite winds at some flagged cells.
                flagged_finite_full = count(ci -> qf[ci] != Int8(0) &&
                    !ismissing(Uf[ci]), CartesianIndices(qf))
                @test flagged_finite_full > 0
                # Masked strictly removes those.
                @test count(!ismissing, Um) < count(!ismissing, Uf)
                # QFLAG itself is identical (diagnostics unmasked).
                @test dm["QFLAG"][:,:,:,1] == dfll["QFLAG"][:,:,:,1]
            finally
                close(dm); close(dfll)
            end
        finally
            isfile(fmask) && rm(fmask); isfile(ffull) && rm(ffull)
        end
        # The accumulator / SynthesisOutput are untouched by writing.
        @test count(ci -> out.comp1[ci] != out.fill_value, CartesianIndices(out.comp1)) == nsolvable
    end

    @testset "write_grid_products: scalar accumulator has no wind variables" begin
        p = DaishoParameters()
        v = synthetic_volume(n_sweeps = 2, n_rays = 72, n_gates = 12)
        spec = GridSpec(shape = :volume_3d,
            reference_latitude = v.latitude, reference_longitude = v.longitude,
            x_axis = collect(Float64, -1200.0:300.0:1200.0),
            y_axis = collect(Float64, -1200.0:300.0:1200.0),
            z_axis = [0.0, 60.0, 120.0])
        acc = ScalarGridAccumulator(spec, p)
        for s in eachindex(v.sweeps); grid_sweep!(acc, v, s, p); end

        nx, ny, nz = length(spec.x_axis), length(spec.y_axis), length(spec.z_axis)
        file = tempname() * ".nc"
        try
            write_grid_products(file, acc, p; index_time = v.time_coverage_start)
            ds = NCDataset(file, "r")
            try
                for f in acc.fields
                    @test haskey(ds, f)
                end
                # No wind product written for a scalar accumulator.
                for vn in ("U", "V", "USTD", "VSTD", "DET", "NGATES", "QFLAG")
                    @test !haskey(ds, vn)
                end
                @test haskey(ds, "grid_mapping") && haskey(ds, "latitude")
            finally
                close(ds)
            end
        finally
            isfile(file) && rm(file)
        end
    end


    # ── Variational Phase 5: stage-1 extensions (N=3, fall speed, gate error) ──

    # Full-positional DaishoParameters with a custom [synthesis] / [reference_state].
    function _p_gate(; fs = nothing, ge = nothing, n3 = false, maxel = 45.0,
                     var = 1.0, ref = ReferenceStateParameters(), moments = nothing)
        p0 = DaishoParameters()
        syn = SynthesisParameters(velocity_variance = var, max_sigma = [2.0, 2.0],
            max_elevation = maxel, include_vertical_velocity = n3,
            fall_speed = fs, gate_error = ge)
        return Daisho.DaishoParameters(moments === nothing ? p0.moments : moments,
            p0.qc, p0.gridding, p0.grid, p0.io, syn, p0.echo, p0.hybrid_scan,
            ref, p0.provided)
    end
    # Exponential reference with a temperature lapse ⇒ a freezing level exists.
    _ref_warm() = ReferenceStateParameters(temperature_surface_k = 288.15,
                                           lapse_rate_k_m = 0.0065)
    function _mutated_config(f!)
        path = tempname() * ".toml"
        base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
        f!(base)
        open(path, "w") do io; TOML.print(io, base); end
        return path
    end
    _load_err(path) = try; DaishoParameters(path); nothing; catch e; e; end

    @testset "[synthesis] fall_speed / gate_error / include_vertical_velocity config" begin
        # Template (new keys commented out) still loads; defaults off.
        p = DaishoParameters()
        @test p.synthesis.include_vertical_velocity == false
        @test p.synthesis.fall_speed === nothing
        @test p.synthesis.gate_error === nothing
        # New tags on the template fields.
        @test Daisho.field_with_tag_or(p, :reflectivity) == "DBZ"
        @test Daisho.field_with_tag_or(p, :spectrum_width) == "WIDTH"

        # Uncommenting the documented template blocks loads them. A block runs
        # from its commented header to the first non-comment line; only
        # "# key = value" lines are uncommented (prose lines use "#   ").
        txt = read(Daisho.DEFAULTS_TOML_PATH, String)
        @test occursin("# [synthesis.fall_speed]", txt)
        @test occursin("# [synthesis.gate_error]", txt)
        out = String[]
        inblock = false
        for ln in split(txt, '\n')
            if startswith(ln, "# [synthesis.fall_speed]") || startswith(ln, "# [synthesis.gate_error]")
                inblock = true
            elseif inblock && !startswith(ln, "#")
                inblock = false
            end
            if inblock && startswith(ln, "# ") && !startswith(ln, "#  ") &&
               !occursin("melting_layer_bottom_m", ln)
                push!(out, ln[3:end])
            else
                push!(out, ln)
            end
        end
        path = tempname() * ".toml"
        write(path, join(out, '\n'))
        pu = DaishoParameters(path)
        @test pu.synthesis.fall_speed == Daisho.FallSpeedParameters()
        @test pu.synthesis.gate_error == Daisho.GateErrorParameters()
        rm(path)

        # Custom values round-trip; include_vertical_velocity parses.
        path = _mutated_config() do b
            b["synthesis"]["include_vertical_velocity"] = true
            b["synthesis"]["fall_speed"] = Dict("rain_coefficient" => 3.0,
                "mixed_phase_dbz" => 15.0, "rain_dbz" => 35.0,
                "melting_layer_bottom_m" => 4000.0, "melting_layer_depth_m" => 500.0)
            b["synthesis"]["gate_error"] = Dict("fall_speed_error_m_s" => 1.5,
                "spectrum_width_error_factor" => 0.5, "error_guard_min_m_s" => 0.8)
        end
        pc = DaishoParameters(path)
        @test pc.synthesis.include_vertical_velocity
        fsp = pc.synthesis.fall_speed
        @test fsp.rain_coefficient == 3.0
        @test fsp.rain_exponent == 0.107          # absent ⇒ default
        @test fsp.mixed_phase_dbz == 15.0 && fsp.rain_dbz == 35.0
        @test fsp.melting_layer_bottom_m == 4000.0
        @test fsp.melting_layer_depth_m == 500.0
        @test pc.synthesis.gate_error ==
              Daisho.GateErrorParameters(1.5, 0.5, 0.8)
        rm(path)

        # Empty sub-table ⇒ enabled with all defaults.
        path = _mutated_config() do b
            b["synthesis"]["fall_speed"] = Dict{String,Any}()
        end
        @test isnan(DaishoParameters(path).synthesis.fall_speed.melting_layer_bottom_m)
        rm(path)

        # Unknown key in each sub-table names the section.
        for (tbl, sec) in (("fall_speed", "[synthesis.fall_speed]"),
                           ("gate_error", "[synthesis.gate_error]"))
            path = _mutated_config() do b
                b["synthesis"][tbl] = Dict("bogus" => 1.0)
            end
            e = _load_err(path)
            @test e isa ArgumentError && occursin(sec, e.msg)
            rm(path)
        end

        # Validation errors.
        for bad in (Dict("mixed_phase_dbz" => 30.0, "rain_dbz" => 30.0),
                    Dict("mixed_phase_dbz" => 40.0, "rain_dbz" => 30.0),
                    Dict("rain_coefficient" => 0.0),
                    Dict("snow_exponent" => -0.1),
                    Dict("melting_layer_depth_m" => 0.0))
            path = _mutated_config() do b
                b["synthesis"]["fall_speed"] = bad
            end
            @test _load_err(path) isa ArgumentError
            rm(path)
        end
        for bad in (Dict("error_guard_min_m_s" => 0.0),
                    Dict("fall_speed_error_m_s" => -1.0))
            path = _mutated_config() do b
                b["synthesis"]["gate_error"] = bad
            end
            @test _load_err(path) isa ArgumentError
            rm(path)
        end
        path = _mutated_config() do b
            b["synthesis"]["include_vertical_velocity"] = "yes"
        end
        @test _load_err(path) isa ArgumentError
        rm(path)

        # Samurai-key migration diagnostics name the Daisho key.
        for (k, daisho_key, where) in (
                ("radar_fallspeed_error", "fall_speed_error_m_s", :top),
                ("radar_sw_error", "spectrum_width_error_factor", :top),
                ("radar_min_error", "error_guard_min_m_s", :sub),
                ("horizontal_radar_appx", "include_vertical_velocity", :top),
                ("max_radar_elevation", "max_elevation", :top))
            path = _mutated_config() do b
                if where === :top
                    b["synthesis"][k] = 1.0
                else
                    b["synthesis"]["gate_error"] = Dict(k => 1.0)
                end
            end
            e = _load_err(path)
            @test e isa ArgumentError
            @test occursin(daisho_key, e.msg) && occursin(k, e.msg)
            rm(path)
        end

        # New tags: a duplicated singular tag is rejected.
        for tag in ("reflectivity", "spectrum_width")
            path = _mutated_config() do b
                push!(b["fields"]["SQI"], tag)
            end
            e = _load_err(path)
            @test e isa ArgumentError && occursin(tag, e.msg)
            rm(path)
        end
    end

    @testset "GateModel construction" begin
        # Neither block ⇒ no model.
        @test Daisho.GateModel(_p_gate()) === nothing
        acc0 = WindGridAccumulator(_wind_spec(), _p_gate())
        @test acc0.gate_model === nothing
        @test acc0.n_unknowns == 2

        # Gate error only: no reference needed.
        ge = Daisho.GateErrorParameters()
        gm = Daisho.GateModel(_p_gate(ge = ge, var = 2.0))
        @test gm.gate_error == ge && gm.fall_speed === nothing
        @test gm.reference === nothing
        @test gm.velocity_variance == 2.0
        @test gm.spectrum_width_field == "WIDTH"

        # Fall speed: exponential reference built internally; freezing level
        # derived from the lapse rate when melting_layer_bottom_m is NaN.
        fs = Daisho.FallSpeedParameters()
        gm = Daisho.GateModel(_p_gate(fs = fs, ref = _ref_warm()))
        @test gm.reflectivity_field == "DBZ"
        @test gm.reference isa ExponentialReferenceProfile
        @test gm.melting_layer_bottom_m ≈ 15.0 / 0.0065
        # Explicit value wins.
        gm2 = Daisho.GateModel(_p_gate(
            fs = Daisho.FallSpeedParameters(melting_layer_bottom_m = 4200.0)))
        @test gm2.melting_layer_bottom_m == 4200.0
        # No temperature and no explicit bottom ⇒ error explaining the fix.
        e = try; Daisho.GateModel(_p_gate(fs = fs)); nothing; catch x; x; end
        @test e isa ArgumentError && occursin("melting_layer_bottom_m", e.msg)
        # Explicit reference is used as given.
        r = ExponentialReferenceProfile(1.0, 9000.0, 280.0, 0.005)
        @test Daisho.GateModel(_p_gate(fs = fs); reference = r).reference == r
        # Sounding source needs an explicit reference (needs the grid).
        snd = ReferenceStateParameters(source = "sounding", sounding_file = "x.txt")
        e = try; Daisho.GateModel(_p_gate(fs = fs, ref = snd)); nothing; catch x; x; end
        @test e isa ArgumentError && occursin("reference", e.msg)
        # Fall speed without a reflectivity-tagged field ⇒ error.
        p0 = DaishoParameters()
        norefl = Daisho.MomentParameters([Daisho.FieldSpec(f.name,
            setdiff(f.tags, Set([:reflectivity]))) for f in p0.moments.fields])
        e = try
            Daisho.GateModel(_p_gate(fs = fs, ref = _ref_warm(), moments = norefl)); nothing
        catch x; x; end
        @test e isa ArgumentError && occursin("reflectivity", e.msg)

        # Equality is field-wise.
        a = Daisho.GateModel(_p_gate(fs = fs, ge = ge, ref = _ref_warm()))
        b = Daisho.GateModel(_p_gate(fs = fs, ge = ge, ref = _ref_warm()))
        @test a == b
        @test a != Daisho.GateModel(_p_gate(fs = fs, ref = _ref_warm()))
        @test a != Daisho.GateModel(_p_gate(fs = Daisho.FallSpeedParameters(rain_dbz = 31.0),
                                            ge = ge, ref = _ref_warm()))
        # NaN temperature fields compare equal on the exponential profile.
        @test ExponentialReferenceProfile(1.2, 8500.0, NaN, 0.0065) ==
              ExponentialReferenceProfile(1.2, 8500.0, NaN, 0.0065)
        @test ExponentialReferenceProfile(1.2, 8500.0, NaN, 0.0065) !=
              ExponentialReferenceProfile(1.2, 8000.0, NaN, 0.0065)
    end

    @testset "terminal_fall_speed (samurai relations)" begin
        fs = Daisho.FallSpeedParameters()
        ref = ExponentialReferenceProfile(1.2, 8500.0, NaN, 0.0065)
        mb = 4000.0; top = mb + fs.melting_layer_depth_m
        tfs(dbz, z) = Daisho.terminal_fall_speed(fs, dbz, z, ref, mb)
        dcor(z) = (1.2 / reference_density(ref, z))^0.45
        vtr(dbz, z) = -dcor(z) * 2.6 * (10.0^(dbz / 10))^0.107
        vts(dbz, z) = -dcor(z) * 0.817 * (10.0^(dbz / 10))^0.063

        # Rain at z=0 (DCOR = 1).
        @test tfs(40.0, 0.0) ≈ -2.6 * 10^(4 * 0.107)
        @test tfs(30.0, 0.0) ≈ -2.6 * 1000^0.107
        # Below the melting layer samurai uses rain for every dBZ.
        @test tfs(5.0, 1000.0) ≈ vtr(5.0, 1000.0)
        # Above the melting layer: snow below mixed_phase_dbz, rain above
        # rain_dbz, linear in dBZ between.
        z = 7000.0
        @test tfs(10.0, z) ≈ vts(10.0, z)
        @test tfs(45.0, z) ≈ vtr(45.0, z)
        @test tfs(25.0, z) ≈ 0.5 * (vts(25.0, z) + vtr(25.0, z))
        # Continuity at the dBZ blend edges.
        @test tfs(20.0 + 1e-9, z) ≈ tfs(20.0, z) atol = 1e-8
        @test tfs(30.0 + 1e-9, z) ≈ tfs(30.0, z) atol = 1e-8
        # Height blend across the melting layer: rain at the bottom, the
        # above-layer (dBZ-blended) value at the top, linear between.
        @test tfs(10.0, mb) ≈ vtr(10.0, mb)
        @test tfs(10.0, top) ≈ vts(10.0, top)
        zm = mb + 250.0
        @test tfs(10.0, zm) ≈ 0.75 * vtr(10.0, zm) + 0.25 * vts(10.0, zm)
        @test tfs(10.0, mb - 1e-7) ≈ tfs(10.0, mb) atol = 1e-8
        @test tfs(10.0, top + 1e-7) ≈ tfs(10.0, top) atol = 1e-8
        # Density correction: falls faster aloft (DCOR > 1).
        @test dcor(9000.0) > 1.0
        @test tfs(10.0, 9000.0) ≈ dcor(9000.0) * vts(10.0, 0.0)
        # Always downward.
        @test all(tfs(d, zz) < 0 for d in -10.0:5.0:60.0, zz in 0.0:500.0:10000.0)
    end

    @testset "gate_sigma" begin
        ge = Daisho.GateErrorParameters()   # σ_fs 2.0, SW factor 1.0, guard 1.0
        # Horizontal beam, no SW ⇒ guard.
        @test Daisho.gate_sigma(ge, 0.0, NaN) == 1.0
        # Steep beam: |sin el|·σ_fs.
        @test Daisho.gate_sigma(ge, deg2rad(60.0), NaN) ≈ sin(deg2rad(60.0)) * 2.0
        @test Daisho.gate_sigma(ge, -deg2rad(60.0), NaN) ≈ sin(deg2rad(60.0)) * 2.0
        # SW term adds.
        @test Daisho.gate_sigma(ge, deg2rad(60.0), 1.5) ≈ sin(deg2rad(60.0)) * 2.0 + 1.5
        ge2 = Daisho.GateErrorParameters(spectrum_width_error_factor = 0.5)
        @test Daisho.gate_sigma(ge2, 0.0, 4.0) ≈ 2.0
        # Missing SW omitted (not propagated as NaN).
        @test Daisho.gate_sigma(ge2, deg2rad(30.0), NaN) == 1.0
    end

    @testset "include_vertical_velocity ⇒ N=3 planes; finalize == N=2" begin
        p3 = _p_gate(n3 = true)
        acc3 = WindGridAccumulator(_wind_spec(), p3)
        @test acc3.n_unknowns == 3
        @test size(acc3.AtWA, 1) == 6 && size(acc3.M2, 1) == 6
        @test size(acc3.AtWb, 1) == 3
        @test build_accumulator(_wind_spec(), p3).n_unknowns == 3

        # Same gates in both runs (max_elevation = 90 on the N=2 run; N=3
        # ignores the cap because steep beams observe w).
        p2 = _p_gate(maxel = 90.0)
        p3 = _p_gate(n3 = true, maxel = 5.0)     # cap must not apply at N=3
        volA, volB = make_synthetic_dual_doppler(u = 8.0, v = -3.0)
        spec = GridSpec(shape = :volume_3d,
            reference_latitude = 16.0, reference_longitude = -24.0,
            x_axis = [-2000.0, 0.0, 2000.0], y_axis = [-2000.0, 0.0, 2000.0],
            z_axis = collect(Float64, 200.0:200.0:1600.0))
        a2 = WindGridAccumulator(spec, p2); a3 = WindGridAccumulator(spec, p3)
        for vol in (volA, volB), s in eachindex(vol.sweeps)
            grid_sweep!(a2, vol, s, p2); grid_sweep!(a3, vol, s, p3)
        end
        @test a3.gate_count == a2.gate_count
        @test sum(a3.gate_count) > 0
        # (u,v) block of the packed 3×3 equals the 2×2 planes exactly.
        @test a3.AtWA[[1, 2, 4], :, :, :] == a2.AtWA
        @test a3.M2[[1, 2, 4], :, :, :] == a2.M2
        @test a3.AtWb[1:2, :, :, :] == a2.AtWb
        @test any(a3.AtWA[6, :, :, :] .> 0)       # w row populated
        o2 = finalize_wind(a2, p2); o3 = finalize_wind(a3, p3)
        @test o3.comp1 == o2.comp1
        @test o3.comp2 == o2.comp2
        @test o3.sigma1 == o2.sigma1
        @test o3.sigma2 == o2.sigma2
        @test o3.det == o2.det
        @test o3.quality_flag == o2.quality_flag
        @test count(==(Int8(0)), o3.quality_flag) > 0
    end

    @testset "no gate model ⇒ planes bit-identical to the pre-change loop" begin
        v = synthetic_volume(n_sweeps = 2, n_rays = 72, n_gates = 12)
        spec = _near_radar_spec(v)
        for (N, p) in ((2, _p_gate()), (3, _p_gate(n3 = true, maxel = 90.0)))
            acc = WindGridAccumulator(spec, p)
            @test acc.gate_model === nothing && acc.n_unknowns == N
            for s in eachindex(v.sweeps); grid_sweep!(acc, v, s, p); end
            np = Daisho._n_packed(N)
            nz, ny, nx = Daisho.wind_accumulator_dims(spec)
            rAtWA = zeros(np, nz, ny, nx); rAtWb = zeros(N, nz, ny, nx)
            rM2 = zeros(np, nz, ny, nx); rwt = zeros(nz, ny, nx)
            rgc = zeros(Int32, nz, ny, nx)
            for s in eachindex(v.sweeps)
                _ref_grid_sweep_wind_3d!(rAtWA, rAtWb, rM2, rwt, rgc, spec, "VEL", N,
                    v.sweeps[s], p, Float64(v.latitude), Float64(v.longitude),
                    Float64(v.altitude))
            end
            @test acc.AtWA == rAtWA
            @test acc.AtWb == rAtWb
            @test acc.M2 == rM2
            @test acc.weight_total == rwt
            @test acc.gate_count == rgc
            @test sum(acc.gate_count) > 0
        end
    end

    # One-cell accumulator + a hand-built contribution: exercises the per-gate
    # model math in isolation from the geometry.
    function _one_cell_gate(p; az = 30.0, el = 20.0, w = 0.7, beam_z = 1500.0,
                            vr = 6.0, dbz = 35.0, sw = 1.2)
        spec = GridSpec(shape = :volume_3d, reference_latitude = 0.0,
            reference_longitude = 0.0, x_axis = [0.0], y_axis = [0.0], z_axis = [beam_z])
        acc = WindGridAccumulator(spec, p)
        sweep = SweepGroup(sweep_number = 0, sweep_mode = "azimuth_surveillance",
            fixed_angle = el, time = [DateTime(2024, 1, 1)], range = [10000.0],
            azimuth = [az], elevation = [el])
        for (name, val) in (("VEL", vr), ("DBZ", dbz), ("WIDTH", sw))
            add_field!(sweep, name, reshape([Float32(val)], 1, 1),
                       FieldMetadata(units = "x", long_name = name))
        end
        c = Daisho.GateContribution(1, 1, 1, deg2rad(az), deg2rad(el), w, beam_z)
        Daisho.accumulate_cell!(acc, Daisho.CartesianCell(1, 1, 1, beam_z, 1), sweep,
            Int[], [c], Daisho.SweepKeys("SQI", "DBZ"), p)
        g = zeros(acc.n_unknowns)
        Daisho._wind_coeffs!(g, deg2rad(az), deg2rad(el), acc.n_unknowns)
        return acc, g, Float64(Float32(vr)), Float64(Float32(dbz)), Float64(Float32(sw))
    end
    function _packed_outer(g, s)
        N = length(g)
        out = zeros(Daisho._n_packed(N))
        for a in 1:N, b in a:N
            out[Daisho._packed_index(a, b, N)] = s * (g[a] * g[b])   # same association as the accumulator
        end
        return out
    end

    @testset "gate model: weight scaling and fall-speed correction (one gate)" begin
        w, el, var = 0.7, 20.0, 2.0
        ge = Daisho.GateErrorParameters()
        fs = Daisho.FallSpeedParameters()
        for N3 in (false, true)
            # Gate error only: w′ = w·σ²_vr/σ_m², v_r unchanged.
            p = _p_gate(ge = ge, var = var, n3 = N3)
            acc, g, vr, _, sw = _one_cell_gate(p; w = w, el = el)
            σm = Daisho.gate_sigma(ge, deg2rad(el), sw)
            @test σm ≈ sin(deg2rad(el)) * 2.0 + sw
            wp = w * var / σm^2
            @test acc.AtWA[:, 1, 1, 1] ≈ _packed_outer(g, wp) rtol = 1e-14
            @test acc.M2[:, 1, 1, 1] ≈ _packed_outer(g, wp^2) rtol = 1e-14
            @test acc.AtWb[:, 1, 1, 1] ≈ wp * vr .* g rtol = 1e-14
            @test acc.weight_total[1, 1, 1] ≈ wp rtol = 1e-14
            @test acc.gate_count[1, 1, 1] == 1

            # Fall speed only: weight unchanged, ṽ = v_r − w_t·sin(el).
            p = _p_gate(fs = fs, var = var, n3 = N3, ref = _ref_warm())
            acc, g, vr, dbz, _ = _one_cell_gate(p; w = w, el = el)
            gm = acc.gate_model
            wt = Daisho.terminal_fall_speed(fs, dbz, 1500.0, gm.reference,
                                            gm.melting_layer_bottom_m)
            @test wt < 0
            @test acc.AtWA[:, 1, 1, 1] == _packed_outer(g, w)
            @test acc.AtWb[:, 1, 1, 1] ≈ w * (vr - wt * sin(deg2rad(el))) .* g rtol = 1e-14
            @test acc.weight_total[1, 1, 1] == w

            # Both.
            p = _p_gate(fs = fs, ge = ge, var = var, n3 = N3, ref = _ref_warm())
            acc, g, vr, dbz, sw = _one_cell_gate(p; w = w, el = el)
            wp = w * var / Daisho.gate_sigma(ge, deg2rad(el), sw)^2
            @test acc.AtWA[:, 1, 1, 1] ≈ _packed_outer(g, wp) rtol = 1e-14
            @test acc.AtWb[:, 1, 1, 1] ≈ wp * (vr - wt * sin(deg2rad(el))) .* g rtol = 1e-14

            # Missing reflectivity ⇒ no fall-speed correction; missing SW ⇒
            # SW term omitted.
            acc, g, vr, _, _ = _one_cell_gate(p; w = w, el = el, dbz = NaN, sw = NaN)
            wp = w * var / Daisho.gate_sigma(ge, deg2rad(el), NaN)^2
            @test acc.AtWb[:, 1, 1, 1] ≈ wp * vr .* g rtol = 1e-14
            @test acc.AtWA[:, 1, 1, 1] ≈ _packed_outer(g, wp) rtol = 1e-14
        end

        # max_elevation applies at N=2 only.
        acc, _, _, _, _ = _one_cell_gate(_p_gate(maxel = 10.0); el = 20.0)
        @test acc.gate_count[1, 1, 1] == 0
        acc, _, _, _, _ = _one_cell_gate(_p_gate(maxel = 10.0, n3 = true); el = 20.0)
        @test acc.gate_count[1, 1, 1] == 1
    end

    @testset "JLD2 round trip at schema v2 with a GateModel" begin
        @test Daisho.WIND_ACCUMULATOR_SCHEMA_VERSION == 2
        p = _p_gate(fs = Daisho.FallSpeedParameters(), ge = Daisho.GateErrorParameters(),
                    ref = _ref_warm(), n3 = true)
        acc, _, _, _, _ = _one_cell_gate(p)
        @test acc.gate_model isa Daisho.GateModel
        path = tempname() * ".jld2"
        try
            save_wind_accumulator(path, acc)
            rt = load_wind_accumulator(path)
            @test rt.gate_model == acc.gate_model
            @test rt.gate_model.reference == acc.gate_model.reference
            @test rt.n_unknowns == 3
            @test rt.AtWA == acc.AtWA && rt.AtWb == acc.AtWb && rt.M2 == acc.M2
            @test rt.weight_total == acc.weight_total
            @test rt.schema_version == 2
        finally
            isfile(path) && rm(path)
        end

        # A schema-1 file is refused.
        old = WindGridAccumulator(; (f => getfield(acc, f)
            for f in fieldnames(WindGridAccumulator) if f !== :schema_version)...,
            schema_version = 1)
        path = tempname() * ".jld2"
        try
            Daisho.JLD2.jldopen(path, "w") do f
                f["wind_accumulator"] = old
            end
            e = try; load_wind_accumulator(path); nothing; catch x; x; end
            @test e isa ArgumentError && occursin("schema", e.msg)
        finally
            isfile(path) && rm(path)
        end
    end

    @testset "merge refuses differing gate models / N" begin
        spec = _wind_spec()
        fs = Daisho.FallSpeedParameters()
        pa = _p_gate(fs = fs, ref = _ref_warm())
        pb = _p_gate(fs = Daisho.FallSpeedParameters(rain_dbz = 32.0), ref = _ref_warm())
        a = WindGridAccumulator(spec, pa)
        @test_throws ArgumentError merge_wind_accumulators!(a, WindGridAccumulator(spec, pb))
        @test_throws ArgumentError merge_wind_accumulators!(a, WindGridAccumulator(spec, _p_gate()))
        @test_throws ArgumentError merge_wind_accumulators!(
            WindGridAccumulator(spec, _p_gate()), a)
        @test_throws ArgumentError merge_wind_accumulators!(
            WindGridAccumulator(spec, _p_gate()), WindGridAccumulator(spec, _p_gate(n3 = true)))
        # Equal models merge.
        b = WindGridAccumulator(spec, pa)
        b.AtWA[1, 1, 1, 1] = 2.0
        merge_wind_accumulators!(a, b)
        @test a.AtWA[1, 1, 1, 1] == 2.0
    end

end
