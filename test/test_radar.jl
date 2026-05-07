# After the CfRadial 2.1 refactor, the public radar API is `Volume`. This file
# tests only the small set of utility helpers that survive the cleanup
# (beam_height, dB conversions, get_radar_orientation, plus the internal
# scaffolding `radar` struct used by the gridding workers via the bridge).

@testset "Radar" begin

    @testset "internal radar scaffolding constructs" begin
        # Build a small synthetic Volume and verify the bridge produces a `radar`
        # with the expected shape. (`radar` is internal, not exported.)
        v = synthetic_volume(n_sweeps=1, n_rays=4, n_gates=5, fields=["DBZ"])
        legacy, names = as_legacy_radar(v)
        @test length(legacy.azimuth) == 4
        @test length(legacy.range) == 5
        @test size(legacy.moments) == (4 * 5, 1)
        @test names == ["DBZ"]
    end

    @testset "split_sweeps internal helper" begin
        # synthetic_volume: 2 sweeps × 10 rays each = 20 total rays
        v = synthetic_volume(n_sweeps=2, n_rays=10, n_gates=5, fields=["DBZ"])
        legacy, _ = as_legacy_radar(v)
        sweeps = Daisho.split_sweeps(legacy)
        @test length(sweeps) == 2
        @test length(sweeps[1].azimuth) == 10
        @test length(sweeps[2].azimuth) == 10
    end

    @testset "beam_height" begin
        # At 0 elevation, close range, height should be approximately radar_height
        h = Daisho.beam_height(0.0, 0.0, 50.0)
        @test h ≈ 50.0

        # At 0 elevation, 100 km range, beam height should be above ground due to Earth curvature
        h = Daisho.beam_height(100000.0, 0.0, 0.0)
        @test h > 0.0

        # At 90 degrees elevation, beam height ≈ slant_range + radar_height
        h = Daisho.beam_height(10000.0, 90.0, 50.0)
        @test h ≈ 10050.0 atol=10.0

        # Typical case: 1 degree elevation, 50 km range
        h = Daisho.beam_height(50000.0, 1.0, 0.0)
        @test h > 800.0
        @test h < 1200.0

        # Negative elevation (possible for airborne radar)
        h = Daisho.beam_height(10000.0, -5.0, 5000.0)
        @test h < 5000.0
    end

    @testset "dB_to_linear" begin
        @test Daisho.dB_to_linear([0.0])[1] ≈ 1.0
        @test Daisho.dB_to_linear([10.0])[1] ≈ 10.0
        @test Daisho.dB_to_linear([-10.0])[1] ≈ 0.1
        @test Daisho.dB_to_linear([20.0])[1] ≈ 100.0
        @test Daisho.dB_to_linear([0.0, 10.0, 20.0]) ≈ [1.0, 10.0, 100.0]
    end

    @testset "linear_to_dB" begin
        @test Daisho.linear_to_dB([1.0])[1] ≈ 0.0
        @test Daisho.linear_to_dB([10.0])[1] ≈ 10.0
        @test Daisho.linear_to_dB([100.0])[1] ≈ 20.0
    end

    @testset "dB_to_linear / linear_to_dB round-trip" begin
        original = [0.0, 5.0, 10.0, 20.0, 30.0, -5.0]
        result = Daisho.linear_to_dB(Daisho.dB_to_linear(original))
        @test result ≈ original atol=1e-10
    end

    @testset "dB_to_linear!" begin
        moment = [0.0, 10.0, 20.0]
        Daisho.dB_to_linear!(moment)
        @test moment ≈ [1.0, 10.0, 100.0]
    end

    @testset "linear_to_dB!" begin
        moment = [1.0, 10.0, 100.0]
        Daisho.linear_to_dB!(moment)
        @test moment ≈ [0.0, 10.0, 20.0]
    end

    @testset "get_radar_orientation - no heading/pitch/roll" begin
        # Use the synthetic CfRadial helper which intentionally omits orientation vars.
        filepath = joinpath(@__DIR__, "fixtures", "test_orientation.nc")
        create_synthetic_cfradial(filepath, n_sweeps=1, n_rays=5, n_gates=10)
        result = Daisho.get_radar_orientation(filepath)
        @test size(result, 1) == 5
        @test size(result, 2) == 3
        @test all(isnan.(result[:, 1]))
        @test all(isnan.(result[:, 2]))
        @test all(isnan.(result[:, 3]))
        rm(filepath, force=true)
    end

end
