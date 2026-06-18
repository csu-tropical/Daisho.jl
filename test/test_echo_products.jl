using NCDatasets

# Build a DaishoParameters carrying polarimetric fields and a [echo] block.
function _echo_test_params(; enabled::Bool, rain_components = ["RATE_Z", "RATE_KDP"])
    fields = [
        Daisho.FieldSpec("DBZ",   Set([:linear_interp, :define_detection])),
        Daisho.FieldSpec("ZDR",   Set([:linear_interp])),
        Daisho.FieldSpec("KDP",   Set([:linear_interp])),
        Daisho.FieldSpec("RHOHV", Set([:weighted_interp])),
        Daisho.FieldSpec("SQI",   Set([:weighted_interp, :define_scanned])),
    ]
    moments = Daisho.MomentParameters(fields)
    small_cart = Daisho.CartesianGridParameters(
        xmin = -2000.0, xincr = 1000.0, xdim = 5,
        ymin = -2000.0, yincr = 1000.0, ydim = 5,
        zmin = 500.0, zincr = 500.0, zdim = 3)
    grid = Daisho.GridParameters(cartesian = small_cart)
    dp = Daisho.EchoProductsParameters(
        enabled = enabled, band = "S",
        compute_fhc = true, compute_blended_rain = true,
        rain_components = rain_components,
        use_temp = true, temp_factor = 1.0,
        temperature = TemperatureProfile([0.0, 2000.0, 6000.0], [25.0, 10.0, -15.0]))
    return Daisho.DaishoParameters(moments, Daisho.QCParameters(),
        Daisho.GriddingParameters(), grid, Daisho.IOParameters(),
        Daisho.SynthesisParameters(), dp, Set([:grid, :gridding, :echo]))
end

@testset "Echo products (HID + rain) on gridded data" begin

    @testset "TemperatureProfile interpolation" begin
        prof = TemperatureProfile([0.0, 1000.0, 5000.0], [20.0, 10.0, -20.0])
        @test temperature_celsius(prof, 0.0) == 20.0
        @test temperature_celsius(prof, 1000.0) == 10.0
        @test temperature_celsius(prof, 500.0) ≈ 15.0          # midpoint
        @test temperature_celsius(prof, 3000.0) ≈ 10.0 + (3000-1000)/(5000-1000)*(-30.0)
        @test temperature_celsius(prof, -500.0) == 20.0        # clamp low
        @test temperature_celsius(prof, 9000.0) == -20.0       # clamp high
        # Unsorted input is sorted on construction.
        prof2 = TemperatureProfile([5000.0, 0.0, 1000.0], [-20.0, 20.0, 10.0])
        @test temperature_celsius(prof2, 500.0) ≈ 15.0
        # Vector sampling.
        @test temperature_celsius(prof, [0.0, 1000.0]) == [20.0, 10.0]
    end

    @testset "apply_echo_products in memory" begin
        io = IOParameters()
        nx, ny, nz = 4, 3, 2
        dbz = fill(40.0f0, nx, ny, nz)
        dbz[1, 1, 1] = Float32(io.fill_value)   # true missing
        dbz[2, 1, 1] = Float32(io.undetect)     # clear air
        fields = Dict{String,Any}(
            "DBZ" => dbz,
            "ZDR" => fill(1.0f0, nx, ny, nz),
            "KDP" => fill(1.5f0, nx, ny, nz),
            "RHOHV" => fill(0.98f0, nx, ny, nz))
        z_axis = [500.0, 1500.0]
        heights = Daisho._heights_array((nx, ny, nz), z_axis, 3)
        dp = Daisho.EchoProductsParameters(enabled = true, band = "S",
            rain_components = ["RATE_Z", "RATE_KDP", "RATE_Z_ZDR", "RATE_KDP_ZDR"],
            use_temp = true,
            temperature = TemperatureProfile([0.0, 5000.0], [20.0, -20.0]))
        out = apply_echo_products(fields, dp; io = io, heights = heights)

        @test haskey(out, "HID_CSU") && haskey(out, "RATE_CSU_BLENDED")
        @test all(haskey(out, k) for k in ("RATE_Z", "RATE_KDP", "RATE_Z_ZDR", "RATE_KDP_ZDR"))
        for v in values(out)
            @test size(v) == (nx, ny, nz)
            @test eltype(v) == Float32
        end
        # Sentinel policy: missing → fill_value, clear-air → undetect (HID) / 0 (rain).
        @test out["HID_CSU"][1, 1, 1] == Float32(io.fill_value)
        @test out["HID_CSU"][2, 1, 1] == Float32(io.undetect)
        @test out["RATE_CSU_BLENDED"][1, 1, 1] == Float32(io.fill_value)
        @test out["RATE_CSU_BLENDED"][2, 1, 1] == 0.0f0
        # Valid cells classify within range and produce a finite rain rate.
        @test 1 <= out["HID_CSU"][3, 3, 1] <= FHC_N_TYPES
        @test isfinite(out["RATE_CSU_BLENDED"][3, 3, 1]) && out["RATE_CSU_BLENDED"][3, 3, 1] >= 0
    end

    @testset "inline gridding writes echo products" begin
        v = synthetic_volume(n_sweeps = 2, n_rays = 24, n_gates = 6,
                             fields = ["DBZ", "ZDR", "KDP", "RHOHV", "SQI"])
        p = _echo_test_params(enabled = true)
        outfile = tempname() * "_vol.nc"
        try
            Daisho.grid_radar_volume(v, outfile, v.time_coverage_start, p)
            NCDataset(outfile) do ds
                for name in ("HID_CSU", "RATE_CSU_BLENDED", "RATE_Z", "RATE_KDP")
                    @test haskey(ds, name)
                    @test size(ds[name]) == (5, 5, 3, 1)
                end
                @test ds["RATE_CSU_BLENDED"].attrib["units"] == "mm/hr"
            end
        finally
            isfile(outfile) && rm(outfile)
        end
    end

    @testset "standalone add_echo_products! matches inline result" begin
        v = synthetic_volume(n_sweeps = 2, n_rays = 24, n_gates = 6,
                             fields = ["DBZ", "ZDR", "KDP", "RHOHV", "SQI"])
        p_on = _echo_test_params(enabled = true)
        p_off = _echo_test_params(enabled = false)
        f_inline = tempname() * "_inline.nc"
        f_standalone = tempname() * "_standalone.nc"
        try
            Daisho.grid_radar_volume(v, f_inline, v.time_coverage_start, p_on)        # inline
            Daisho.grid_radar_volume(v, f_standalone, v.time_coverage_start, p_off)   # no echo products
            written = add_echo_products!(f_standalone, p_on)                # then append
            @test "HID_CSU" in written && "RATE_CSU_BLENDED" in written

            NCDataset(f_inline) do a
                NCDataset(f_standalone) do b
                    for name in ("HID_CSU", "RATE_CSU_BLENDED", "RATE_Z", "RATE_KDP")
                        @test isequal(Array(a[name].var), Array(b[name].var))
                    end
                end
            end
        finally
            isfile(f_inline) && rm(f_inline)
            isfile(f_standalone) && rm(f_standalone)
        end
    end

    @testset "multi-time concatenated file (loops over time)" begin
        io = IOParameters()
        nx, ny, nz, nt = 4, 3, 2, 2
        # Two time slices with different reflectivity so HID/rain differ per time.
        dbz = Array{Float32}(undef, nx, ny, nz, nt)
        dbz[:, :, :, 1] .= 30.0f0
        dbz[:, :, :, 2] .= Float32(io.fill_value)   # second time entirely missing
        zdr = fill(1.0f0, nx, ny, nz, nt)
        kdp = fill(1.2f0, nx, ny, nz, nt)
        rho = fill(0.98f0, nx, ny, nz, nt)

        f = tempname() * "_multitime.nc"
        try
            NCDataset(f, "c") do ds
                ds.dim["X"] = nx; ds.dim["Y"] = ny; ds.dim["Z"] = nz; ds.dim["time"] = nt
                defVar(ds, "X", Float32, ("X",))[:] = collect(0:nx-1) .* 1000.0
                defVar(ds, "Y", Float32, ("Y",))[:] = collect(0:ny-1) .* 1000.0
                defVar(ds, "Z", Float32, ("Z",))[:] = [500.0, 1500.0]
                defVar(ds, "DBZ", Float32, ("X", "Y", "Z", "time"))[:] = dbz
                defVar(ds, "ZDR", Float32, ("X", "Y", "Z", "time"))[:] = zdr
                defVar(ds, "KDP", Float32, ("X", "Y", "Z", "time"))[:] = kdp
                defVar(ds, "RHOHV", Float32, ("X", "Y", "Z", "time"))[:] = rho
            end
            p = _echo_test_params(enabled = true)
            written = add_echo_products!(f, p)
            @test "HID_CSU" in written

            NCDataset(f) do ds
                @test size(ds["HID_CSU"]) == (nx, ny, nz, nt)
                hid = Array(ds["HID_CSU"].var)
                # Per-slice result must equal applying the algorithm to that slice.
                heights = Daisho._heights_array((nx, ny, nz), [500.0, 1500.0], 3)
                for t in 1:nt
                    sl = Dict{String,Any}("DBZ" => dbz[:, :, :, t],
                        "ZDR" => zdr[:, :, :, t], "KDP" => kdp[:, :, :, t],
                        "RHOHV" => rho[:, :, :, t])
                    ref = apply_echo_products(sl, p.echo; io = io, heights = heights)
                    @test isequal(hid[:, :, :, t], ref["HID_CSU"])
                end
                # Different reflectivity per time should change the classification.
                @test hid[:, :, :, 1] != hid[:, :, :, 2]
            end
        finally
            isfile(f) && rm(f)
        end
    end
end
