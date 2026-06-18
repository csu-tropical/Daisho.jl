using DelimitedFiles

# Reference fixtures generated from CSU_RadarTools (test/fixtures/generate_fixtures.py).
const FHC_FIXDIR = joinpath(@__DIR__, "fixtures")

_read_fix(name) = (d = readdlm(joinpath(FHC_FIXDIR, name), ','; header = true); d[1])

@testset "FHC (csu_fhc_summer)" begin

    @testset "beta_mbf kernel" begin
        # At the center the membership is exactly 1.
        @test beta_mbf(5.0, 5.0, 2.0, 3.0) == 1.0
        # One width from center: 1/(1+1^b) = 0.5 for any b.
        @test beta_mbf(7.0, 5.0, 2.0, 3.0) ≈ 0.5
        @test beta_mbf(3.0, 5.0, 2.0, 1.0) ≈ 0.5
        # Monotone falloff away from center.
        @test beta_mbf(20.0, 5.0, 2.0, 3.0) < beta_mbf(10.0, 5.0, 2.0, 3.0)
    end

    @testset "coefficient tables load for all bands" begin
        for band in ("S", "C", "X")
            sets = get_mbf_sets_summer(band; use_temp = true)
            for k in (:DZ, :DR, :KD, :LD, :RH, :T)
                @test length(sets[k].m) == FHC_N_TYPES
                @test length(sets[k].a) == FHC_N_TYPES
                @test length(sets[k].b) == FHC_N_TYPES
            end
        end
        # temp_factor divides the temperature MBF coefficients.
        s1 = get_mbf_sets_summer("S"; use_temp = true, temp_factor = 1.0)
        s2 = get_mbf_sets_summer("S"; use_temp = true, temp_factor = 2.0)
        @test s2[:T].a ≈ s1[:T].a ./ 2.0
        @test s2[:DZ].a == s1[:DZ].a   # only T is scaled
    end

    @testset "scalar input returns a scalar class" begin
        c = csu_fhc_summer(; dz = 45.0, zdr = 0.2, kdp = 1.0, rho = 0.99,
                           T = 5.0, band = "S")
        @test c isa Integer
        @test 1 <= c <= FHC_N_TYPES
    end

    @testset "matches Python reference fixtures exactly" begin
        for band in ("S", "C", "X"), tag in ("withT", "noT")
            data = _read_fix("fhc_summer_$(band)_$(tag).csv")
            dz = Float64.(data[:, 1]); zdr = Float64.(data[:, 2])
            kdp = Float64.(data[:, 3]); rho = Float64.(data[:, 4])
            T = Float64.(data[:, 5]); hid_py = Int.(data[:, 6])
            use_temp = tag == "withT"
            hid = csu_fhc_summer(; dz = dz, zdr = zdr, kdp = kdp, rho = rho,
                T = (use_temp ? T : nothing), use_temp = use_temp,
                method = :hybrid, band = band)
            @test hid == hid_py
        end
    end

    @testset "masked / sentinel reflectivity is unclassified" begin
        io = IOParameters()
        dz = [45.0, io.fill_value, io.undetect, NaN]
        hid = csu_fhc_summer(; dz = dz, zdr = fill(0.5, 4), kdp = fill(1.0, 4),
                             rho = fill(0.99, 4), T = fill(5.0, 4), band = "S",
                             fill_value = io.fill_value, undetect = io.undetect)
        @test 1 <= hid[1] <= FHC_N_TYPES
        @test hid[2] == 0 && hid[3] == 0 && hid[4] == 0
    end
end
