using DelimitedFiles

const RAIN_FIXDIR = joinpath(@__DIR__, "fixtures")
_read_rainfix(name) = (d = readdlm(joinpath(RAIN_FIXDIR, name), ','; header = true); d[1])

# NaN-aware approximate equality (numpy yields NaN for negative-base fractional
# powers, which the Julia port reproduces via _safe_pow).
_approx_or_nan(a, b; rtol = 1e-6, atol = 1e-8) =
    (isnan(a) && isnan(b)) || (!isnan(a) && !isnan(b) && isapprox(a, b; rtol = rtol, atol = atol))

@testset "Rain rate (CSU blended tropical)" begin

    @testset "component relations vs Python fixtures" begin
        for band in ("S", "C", "X")
            data = _read_rainfix("rain_components_$(band).csv")
            dz = Float64.(data[:, 1]); zdr = Float64.(data[:, 2]); kdp = Float64.(data[:, 3])
            cols = (
                ("RATE_Z",        calc_rain_zr(dz; a = Daisho.RAIN_RZ_ALL.a,  b = Daisho.RAIN_RZ_ALL.b),  Float64.(data[:, 4])),
                ("RATE_Z_CONV",   calc_rain_zr(dz; a = Daisho.RAIN_RZ_CONV.a, b = Daisho.RAIN_RZ_CONV.b), Float64.(data[:, 5])),
                ("RATE_Z_STRAT",  calc_rain_zr(dz; a = Daisho.RAIN_RZ_STRAT.a,b = Daisho.RAIN_RZ_STRAT.b),Float64.(data[:, 6])),
                ("RATE_KDP",      calc_rain_kdp(kdp, band),         Float64.(data[:, 7])),
                ("RATE_KDP_ZDR",  calc_rain_kdp_zdr(kdp, zdr, band),Float64.(data[:, 8])),
                ("RATE_Z_ZDR",    calc_rain_z_zdr(dz, zdr, band),   Float64.(data[:, 9])),
            )
            for (nm, jl, py) in cols
                @test all(_approx_or_nan(jl[i], py[i]) for i in eachindex(py))
            end
        end
    end

    @testset "blended: plain and convective/stratiform map" begin
        for band in ("S", "C", "X")
            d = _read_rainfix("rain_blended_$(band)_plain.csv")
            dz = Float64.(d[:, 1]); zdr = Float64.(d[:, 2]); kdp = Float64.(d[:, 3])
            rain_py = Float64.(d[:, 4]); meth_py = Int.(d[:, 5])
            rain, meth = calc_blended_rain_tropical(; dz = dz, zdr = zdr, kdp = kdp, band = band)
            @test all(_approx_or_nan(rain[i], rain_py[i]) for i in eachindex(rain_py))
            @test meth == meth_py

            d = _read_rainfix("rain_blended_$(band)_cs.csv")
            dz = Float64.(d[:, 1]); zdr = Float64.(d[:, 2]); kdp = Float64.(d[:, 3])
            cs = Float64.(d[:, 4]); rain_py = Float64.(d[:, 5]); meth_py = Int.(d[:, 6])
            rain, meth = calc_blended_rain_tropical(; dz = dz, zdr = zdr, kdp = kdp, cs = cs, band = band)
            @test all(_approx_or_nan(rain[i], rain_py[i]) for i in eachindex(rain_py))
            @test meth == meth_py
        end
    end

    @testset "blended with FHC ice masking (corrected hail-only method)" begin
        for band in ("S", "C", "X")
            d = _read_rainfix("rain_blended_$(band)_fhc.csv")
            dz = Float64.(d[:, 1]); zdr = Float64.(d[:, 2]); kdp = Float64.(d[:, 3])
            fhc = Float64.(d[:, 4]); rain_py = Float64.(d[:, 5]); meth_py = Int.(d[:, 6])

            # Default: corrected ice method. Rain values are identical to Python.
            rain, meth = calc_blended_rain_tropical(; dz = dz, zdr = zdr, kdp = kdp,
                                                    fhc = fhc, band = band)
            @test all(_approx_or_nan(rain[i], rain_py[i]) for i in eachindex(rain_py))
            # Method matches Python everywhere except ice cells that are not
            # hail-with-Kdp/Z, which the corrected port labels invalid (-1) instead
            # of 2 (the Python bug).
            for i in eachindex(meth_py)
                cond_kdpz = kdp[i] >= 0.3 && dz[i] >= 38.0
                is_ice = fhc[i] > 2 && fhc[i] < 10
                is_hail_kdpz = fhc[i] == 9 && cond_kdpz
                if is_ice && !is_hail_kdpz
                    @test meth[i] == -1
                else
                    @test meth[i] == meth_py[i]
                end
            end

            # Opting into the Python bug reproduces its method labels exactly.
            _, meth_bug = calc_blended_rain_tropical(; dz = dz, zdr = zdr, kdp = kdp,
                fhc = fhc, band = band, correct_ice_method = false)
            @test meth_bug == meth_py
        end
    end

    @testset "scalar input and sentinel handling" begin
        r, m = calc_blended_rain_tropical(; dz = 50.0, zdr = 0.1, kdp = 2.0, band = "S")
        @test r isa Real && m isa Integer
        # Out-of-range reflectivity → no rain, invalid method.
        r, m = calc_blended_rain_tropical(; dz = -20.0, zdr = 0.1, kdp = 2.0, band = "S")
        @test r == 0.0 && m == -1
    end
end
