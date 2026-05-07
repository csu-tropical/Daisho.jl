@testset "Quality Control" begin

    @testset "fix_SEAPOL_RHOHV!" begin
        # Create a volume with RHOHV and UNKNOWN_ID_82
        raw_names = ["DBZ", "RHOHV", "UNKNOWN_ID_82"]
        moment_dict = Dict("DBZ" => 1, "RHOHV" => 2, "UNKNOWN_ID_82" => 3)
        vol = make_synthetic_radar(n_rays=5, n_gates=10, n_moments=3)

        # Set UNKNOWN_ID_82 to a known value
        # The transformation is: (UNKNOWN_ID_82 + 32640 - 1) / 65533.0
        unknown_val = 32000.0
        vol.moments[:, 3] .= unknown_val
        vol.moments[:, 2] .= 0.0  # Initial RHOHV

        Daisho.fix_SEAPOL_RHOHV!(vol, moment_dict)

        # RHOHV should now be (32000 + 32640 - 1) / 65533 ≈ 0.9863
        expected = (unknown_val + 32640 - 1) / 65533.0
        @test all(vol.moments[:, 2] .≈ expected)
    end

    @testset "threshold_qc - below threshold" begin
        n = 50
        raw_moments = Array{Union{Missing, Float32}}(undef, n, 3)
        raw_moments[:, 1] .= 20.0   # DBZ
        raw_moments[:, 2] .= 5.0    # VEL
        raw_moments[:, 3] .= 0.0    # SQI (below threshold)
        raw_moments[1:25, 3] .= 0.5  # first half above threshold

        raw_dict = Dict("DBZ" => 1, "VEL" => 2, "SQI" => 3)
        qc_dict = Dict("DBZ" => 1, "VEL" => 2)

        qc_moments = Array{Union{Missing, Float32}}(undef, n, 2)
        qc_moments[:, 1] .= raw_moments[:, 1]
        qc_moments[:, 2] .= raw_moments[:, 2]

        result = Daisho.threshold_qc(raw_moments, raw_dict, qc_moments, qc_dict,
            "SQI", 0.3, "SQI", true)

        # First 25 gates have SQI=0.5 > 0.3, should survive
        @test !ismissing(result[1, 1])
        @test !ismissing(result[25, 1])

        # Last 25 gates have SQI=0.0 < 0.3, should be set to missing
        @test ismissing(result[26, 1])
        @test ismissing(result[50, 1])
    end

    @testset "threshold_qc - above threshold (spectrum width)" begin
        n = 50
        raw_moments = Array{Union{Missing, Float32}}(undef, n, 3)
        raw_moments[:, 1] .= 20.0   # DBZ
        raw_moments[:, 2] .= 5.0    # VEL
        raw_moments[:, 3] .= 10.0   # WIDTH (above threshold)
        raw_moments[1:25, 3] .= 2.0  # first half below threshold

        raw_dict = Dict("DBZ" => 1, "VEL" => 2, "WIDTH" => 3)
        qc_dict = Dict("DBZ" => 1, "VEL" => 2)

        qc_moments = Array{Union{Missing, Float32}}(undef, n, 2)
        qc_moments[:, 1] .= raw_moments[:, 1]
        qc_moments[:, 2] .= raw_moments[:, 2]

        result = Daisho.threshold_qc(raw_moments, raw_dict, qc_moments, qc_dict,
            "WIDTH", 8.0, "WIDTH", false)

        # First 25 gates have WIDTH=2.0 < 8.0, should survive (below=false means remove if >= threshold)
        @test !ismissing(result[1, 1])

        # Last 25 gates have WIDTH=10.0 > 8.0, should be missing
        @test ismissing(result[26, 1])
    end

    @testset "threshold_qc - missing_key exclusion" begin
        n = 20
        raw_moments = Array{Union{Missing, Float32}}(undef, n, 3)
        raw_moments[:, 1] .= 20.0   # DBZ
        raw_moments[:, 2] .= 5.0    # VEL
        raw_moments[:, 3] .= 0.0    # SQI below threshold

        raw_dict = Dict("DBZ" => 1, "VEL" => 2, "SQI" => 3)
        qc_dict = Dict("DBZ" => 1, "SQI" => 2)

        qc_moments = Array{Union{Missing, Float32}}(undef, n, 2)
        qc_moments[:, 1] .= 20.0
        qc_moments[:, 2] .= 0.0

        result = Daisho.threshold_qc(raw_moments, raw_dict, qc_moments, qc_dict,
            "SQI", 0.3, "SQI", true)

        # DBZ should be set to missing (SQI below threshold)
        @test ismissing(result[1, qc_dict["DBZ"]])

        # SQI itself should NOT be modified (it's the missing_key)
        @test !ismissing(result[1, qc_dict["SQI"]])
    end

    @testset "smooth_sqi is not implemented" begin
        @test_throws ErrorException Daisho.smooth_sqi([0.5, 0.6, 0.7])
    end

    @testset "despeckle" begin
        n_gates = 20
        n_rays = 5
        n_moments = 2
        moment_dict = Dict("DBZ" => 1, "VEL" => 2)
        moments = Array{Union{Missing, Float32}}(missing, n_gates * n_rays, n_moments)

        # Create a small speckle (size 2) surrounded by missing data
        # In ray 1, gates 5-6 have data
        for g in 5:6
            moments[g, :] .= 10.0
        end

        # Create a larger feature (size 5) that should survive
        for g in 10:14
            moments[g, :] .= 20.0
        end

        result = Daisho.despeckle(3, copy(moments), moment_dict, n_gates, n_rays)

        # The speckle (size 2, <= 3) should be removed
        @test ismissing(result[5, 1])
        @test ismissing(result[6, 1])

        # The larger feature (size 5, > 3) should survive
        @test !ismissing(result[10, 1])
        @test !ismissing(result[14, 1])
    end

    @testset "despeckle - all missing" begin
        n_gates = 10
        n_rays = 3
        moment_dict = Dict("DBZ" => 1)
        moments = Array{Union{Missing, Float32}}(missing, n_gates * n_rays, 1)

        result = Daisho.despeckle(3, copy(moments), moment_dict, n_gates, n_rays)
        @test all(ismissing.(result))
    end

    @testset "despeckle - error on invalid speckle" begin
        moment_dict = Dict("DBZ" => 1)
        moments = Array{Union{Missing, Float32}}(missing, 10, 1)
        @test_throws ErrorException Daisho.despeckle(0, moments, moment_dict, 10, 1)
        @test_throws ErrorException Daisho.despeckle(-1, moments, moment_dict, 10, 1)
    end

    @testset "despeckle_azimuthal" begin
        n_gates = 10
        n_rays = 20
        n_moments = 1
        moment_dict = Dict("DBZ" => 1)
        moments = Array{Union{Missing, Float32}}(missing, n_gates * n_rays, n_moments)

        # Create a small azimuthal speckle in gate 1, rays 3-4
        for r in 3:4
            moments[(r-1)*n_gates + 1, 1] = 10.0  # gate 1 of rays 3 and 4
        end

        # Create a larger azimuthal feature in gate 1, rays 10-16
        for r in 10:16
            moments[(r-1)*n_gates + 1, 1] = 20.0
        end

        result = Daisho.despeckle_azimuthal(3, copy(moments), moment_dict, n_gates, n_rays)

        # Small speckle (size 2) should be removed
        @test ismissing(result[2*n_gates + 1, 1])  # ray 3, gate 1

        # Larger feature (size 7) should survive
        @test !ismissing(result[9*n_gates + 1, 1])  # ray 10, gate 1
    end

    @testset "stddev_phidp_threshold" begin
        n_gates = 20
        n_rays = 3
        moment_dict = Dict("DBZ" => 1, "PHIDP" => 2)
        moments = Array{Union{Missing, Float64}}(undef, n_gates * n_rays, 2)
        moments[:, 1] .= 20.0  # DBZ

        # Create smooth phidp for rays 1-2 (low std dev)
        for r in 1:2
            for g in 1:n_gates
                idx = (r-1)*n_gates + g
                moments[idx, 2] = 50.0 + 0.1 * g  # Smooth linear increase
            end
        end

        # Create noisy phidp for ray 3 (high std dev)
        for g in 1:n_gates
            idx = 2*n_gates + g
            moments[idx, 2] = 50.0 + 50.0 * randn()
        end

        result = Daisho.stddev_phidp_threshold(copy(moments), moment_dict, n_gates, n_rays,
            5, 5.0)

        # Smooth rays should mostly survive
        @test !ismissing(result[10, 1])  # Middle of smooth ray 1

        # Note: The noisy ray may have some gates removed, but we can't guarantee all
        # since it depends on the random seed
    end

    @testset "stddev_phidp_threshold - window validation" begin
        moment_dict = Dict("PHIDP" => 1)
        moments = Array{Union{Missing, Float64}}(undef, 10, 1)
        @test_throws ErrorException Daisho.stddev_phidp_threshold(moments, moment_dict, 10, 1, 0)
        @test_throws ErrorException Daisho.stddev_phidp_threshold(moments, moment_dict, 10, 1, 4)
    end

    @testset "remove_platform_motion!" begin
        vol = make_synthetic_radar(n_rays=4, n_gates=5, n_moments=1)
        moment_dict = Dict("VEL" => 1)

        # Set known velocities
        vel_original = 5.0
        vol.moments[:, 1] .= vel_original

        # Set platform motion: moving east at 10 m/s
        vol.ew_platform[:] .= 10.0
        vol.ns_platform[:] .= 0.0
        vol.w_platform[:] .= 0.0
        vol.azimuth[:] .= [0.0, 90.0, 180.0, 270.0]  # N, E, S, W

        result = Daisho.remove_platform_motion!(vol, copy(vol.moments), moment_dict)

        # For azimuth 90 (east), platform_vr = 10*cos(el)*sin(90°) ≈ 10*cos(1°)
        el_rad = deg2rad(1.0)
        platform_vr_east = 10.0 * cos(el_rad) * sin(deg2rad(90.0))
        n_gates = 5

        # Check that velocity was corrected for eastward ray
        corrected_vel = vel_original + platform_vr_east
        @test result[(1*n_gates + 1), 1] ≈ corrected_vel atol=0.1

        # For azimuth 0 (north), platform_vr should be ~0 (only NS component matters)
        platform_vr_north = 10.0 * cos(el_rad) * sin(deg2rad(0.0))
        corrected_vel_n = vel_original + platform_vr_north
        @test result[1, 1] ≈ corrected_vel_n atol=0.1
    end

    @testset "threshold_height" begin
        vol = make_synthetic_radar(n_rays=5, n_gates=10, n_moments=2)
        raw_dict = Dict("DBZ" => 1, "VEL" => 2)
        qc_dict = Dict("DBZ" => 1, "VEL" => 2)

        vol.moments[:, 1] .= 20.0  # DBZ
        vol.moments[:, 2] .= 5.0   # VEL
        qc_moments = Array{Union{Missing, Float32}}(undef, 50, 2)
        qc_moments[:, 1] .= 20.0
        qc_moments[:, 2] .= 5.0

        # Use a very high threshold that should remove all data at low elevations
        result = Daisho.threshold_height(vol, raw_dict, qc_moments, qc_dict, 100000.0)

        # All gates should be removed since beam height < 100km for short ranges
        # (depends on specific geometry, but at 1° el and short range, heights are small)
        removed_count = count(ismissing, result[:, 1])
        @test removed_count > 0
    end

    @testset "add_azimuthal_offset" begin
        vol = make_synthetic_radar(n_rays=4, n_gates=5)
        vol.azimuth[:] .= [0.0, 90.0, 180.0, 350.0]

        result = Daisho.add_azimuthal_offset(vol, 20.0)

        @test result[1] ≈ 20.0
        @test result[2] ≈ 110.0
        @test result[3] ≈ 200.0
        @test result[4] ≈ 10.0  # 350 + 20 = 370 -> mod 360 = 10
    end

    @testset "add_azimuthal_offset - wrapping" begin
        vol = make_synthetic_radar(n_rays=3, n_gates=5)
        vol.azimuth[:] .= [355.0, 0.0, 5.0]

        result = Daisho.add_azimuthal_offset(vol, -10.0)

        @test result[1] ≈ 345.0
        @test result[2] ≈ 350.0
        @test result[3] ≈ 355.0
    end

    @testset "mask_sector" begin
        n_rays = 4
        n_gates = 5
        vol = make_synthetic_radar(n_rays=n_rays, n_gates=n_gates, n_moments=2)
        vol.azimuth[:] .= [0.0, 90.0, 180.0, 270.0]

        raw_dict = Dict("DBZ" => 1, "VEL" => 2)
        qc_dict = Dict("DBZ" => 1, "VEL" => 2)

        qc_moments = Array{Union{Missing, Float32}}(undef, n_gates * n_rays, 2)
        qc_moments .= 10.0

        # Heading is 0 for all rays, mask sector between 80-100 degrees
        heading = fill(Float32(0.0), n_rays)

        result = Daisho.mask_sector(vol, raw_dict, qc_moments, qc_dict,
            heading, 80.0, 100.0, "DBZ")

        # Ray at 90° should be masked (heading-relative az = 90 - 0 = 90, in [80,100])
        for g in 1:n_gates
            idx = 1 * n_gates + g  # ray 2 (0-indexed: ray index 1 = azimuth 90°)
            @test ismissing(result[idx, 1])
        end

        # Ray at 0° should NOT be masked (heading-relative az = 0, not in [80,100])
        @test !ismissing(result[1, 1])
    end

    @testset "mask_sector - azimuth wrapping with mod" begin
        n_rays = 4
        n_gates = 5
        vol = make_synthetic_radar(n_rays=n_rays, n_gates=n_gates, n_moments=1)
        vol.azimuth[:] .= [10.0, 90.0, 180.0, 350.0]

        raw_dict = Dict("DBZ" => 1)
        qc_dict = Dict("DBZ" => 1)

        qc_moments = Array{Union{Missing, Float32}}(undef, n_gates * n_rays, 1)
        qc_moments .= 10.0

        # Heading is 20 for all rays
        # Heading-relative azimuths: 10-20=-10 -> mod(360)=350, 90-20=70, 180-20=160, 350-20=330
        heading = fill(Float32(20.0), n_rays)

        result = Daisho.mask_sector(vol, raw_dict, qc_moments, qc_dict,
            heading, 340.0, 360.0, "DBZ")

        # Ray at 10° should be masked (heading-relative = 350, in [340,360])
        for g in 1:n_gates
            @test ismissing(result[g, 1])
        end

        # Ray at 90° should not be masked (heading-relative = 70, not in [340,360])
        @test !ismissing(result[n_gates + 1, 1])
    end


    @testset "threshold_qc!(sweep, p) sweep-aware overload" begin
        t0 = DateTime(2024, 9, 3, 15, 0, 0)
        n_rays = 4
        n_gates = 5
        sweep = SweepGroup(
            sweep_number = 0,
            sweep_mode = "azimuth_surveillance",
            fixed_angle = 1.0,
            time = [t0 + Second(r) for r in 0:(n_rays-1)],
            range = collect(Float64, 400.0:100.0:(400.0 + 100.0 * (n_gates-1))),
            azimuth = collect(Float64, 0.0:90.0:(90.0*(n_rays-1))),
            elevation = fill(1.0, n_rays),
        )
        # DBZ field (the input to QC)
        dbz = fill(Float32(20.0), n_rays, n_gates)
        add_field!(sweep, "DBZ", dbz, FieldMetadata(units="dBZ", long_name="raw_dbz"))
        # SQI: half the gates fail the SQI threshold
        sqi = fill(Float32(0.5), n_rays, n_gates)
        sqi[:, 1:2] .= 0.1f0  # below sqi_threshold=0.35
        add_field!(sweep, "SQI", sqi, FieldMetadata(units=""))
        # SNR: above threshold
        snr = fill(Float32(20.0), n_rays, n_gates)
        add_field!(sweep, "SNR", snr, FieldMetadata(units="dB"))
        # RHOHV: above threshold
        rhohv = fill(Float32(0.99), n_rays, n_gates)
        add_field!(sweep, "RHOHV", rhohv, FieldMetadata(units=""))
        # WIDTH: below max
        width = fill(Float32(2.0), n_rays, n_gates)
        add_field!(sweep, "WIDTH", width, FieldMetadata(units="m/s"))

        p = DaishoParameters()
        threshold_qc!(sweep, p; inputs = ["DBZ"])
        @test haskey(sweep.fields, "DBZ_QC")
        out = sweep.fields["DBZ_QC"].data
        @test all(isnan, out[:, 1:2])
        @test !any(isnan, out[:, 3:end])
        @test sweep.fields["DBZ_QC"].metadata.is_quality_field
        @test sweep.fields["DBZ_QC"].metadata.qualified_variables == ["DBZ"]
        # Input DBZ should now have ancillary_variables containing DBZ_QC.
        @test "DBZ_QC" in sweep.fields["DBZ"].metadata.ancillary_variables
    end

end
