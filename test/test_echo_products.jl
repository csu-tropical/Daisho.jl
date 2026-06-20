using NCDatasets

# Build a DaishoParameters carrying polarimetric fields and a [echo] block.
function _echo_test_params(; enabled::Bool, rain_components = ["RATE_Z", "RATE_KDP"],
        temp_source::Symbol = :profile, temp_field_units::String = "C")
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
        temp_source = temp_source, temp_field_units = temp_field_units,
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
        # Sentinel policy: missing → fill_value, clear-air → undetect for BOTH HID
        # and rain (the undetect flag is preserved, never collapsed to 0).
        @test out["HID_CSU"][1, 1, 1] == Float32(io.fill_value)
        @test out["HID_CSU"][2, 1, 1] == Float32(io.undetect)
        @test out["RATE_CSU_BLENDED"][1, 1, 1] == Float32(io.fill_value)
        @test out["RATE_CSU_BLENDED"][2, 1, 1] == Float32(io.undetect)
        @test out["RATE_Z"][1, 1, 1] == Float32(io.fill_value)
        @test out["RATE_Z"][2, 1, 1] == Float32(io.undetect)
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

            # The Fields-API reader surfaces echo outputs even though they are NOT
            # listed in [fields] (echo products are appended post-grid).
            @test !("HID_CSU" in [fs.name for fs in p.moments.fields])
            g = Daisho.read_gridded_radar(outfile, p)
            for name in ("HID_CSU", "RATE_CSU_BLENDED", "RATE_Z", "RATE_KDP")
                @test haskey(g.fields, name)
                @test size(g.fields[name]) == (length(g.X), length(g.Y), length(g.Z))
            end
            @test echo_output_names(p.echo) ==
                  ["HID_CSU", "RATE_CSU_BLENDED", "RATE_Z", "RATE_KDP"]
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

    @testset "temperature source: gridded field" begin
        io = IOParameters()
        nx, ny, nz = 4, 3, 2
        dbz = fill(40.0f0, nx, ny, nz)
        zdr = fill(1.0f0, nx, ny, nz)
        kdp = fill(1.5f0, nx, ny, nz)
        rho = fill(0.98f0, nx, ny, nz)
        prof = TemperatureProfile([0.0, 5000.0], [20.0, -20.0])
        z_axis = [1000.0, 4000.0]
        heights = Daisho._heights_array((nx, ny, nz), z_axis, 3)
        # A 3-D temperature field equal to the profile sampled at each cell.
        tfield = Float32.(map(h -> temperature_celsius(prof, h), heights))

        dp_prof = Daisho.EchoProductsParameters(enabled = true, band = "S",
            use_temp = true, temp_source = :profile, temperature = prof)
        dp_field = Daisho.EchoProductsParameters(enabled = true, band = "S",
            use_temp = true, temp_source = :field, temp_field = "TEMP_FOR_PID")

        base = Dict{String,Any}("DBZ" => dbz, "ZDR" => zdr, "KDP" => kdp, "RHOHV" => rho)
        fields_c = merge(base, Dict("TEMP_FOR_PID" => tfield))

        out_prof = apply_echo_products(base, dp_prof; io = io, heights = heights)
        out_field = apply_echo_products(fields_c, dp_field; io = io, heights = heights)

        @testset "field (°C) == profile sampled at the same heights" begin
            @test out_field["HID_CSU"] == out_prof["HID_CSU"]
        end

        @testset "Kelvin units convert to identical result" begin
            dp_k = Daisho.EchoProductsParameters(enabled = true, band = "S",
                use_temp = true, temp_source = :field, temp_field = "TEMP_FOR_PID",
                temp_field_units = "K")
            fields_k = merge(base, Dict("TEMP_FOR_PID" => tfield .+ 273.15f0))
            out_k = apply_echo_products(fields_k, dp_k; io = io, heights = heights)
            @test out_k["HID_CSU"] == out_field["HID_CSU"]
        end

        @testset "sentinel/NaN temperature cells skip the T term" begin
            t_sent = copy(tfield)
            t_sent[1, 1, 1] = Float32(io.fill_value)
            t_sent[2, 1, 1] = Float32(io.undetect)
            out_s = apply_echo_products(merge(base, Dict("TEMP_FOR_PID" => t_sent)),
                                        dp_field; io = io, heights = heights)
            # Those cells classify as if no temperature was supplied.
            c_noT = csu_fhc_summer(; dz = 40.0, zdr = 1.0, kdp = 1.5, rho = 0.98,
                                   band = "S", use_temp = false)
            @test out_s["HID_CSU"][1, 1, 1] == Float32(c_noT)
            @test out_s["HID_CSU"][2, 1, 1] == Float32(c_noT)
        end

        @testset "missing field errors clearly" begin
            @test_throws ArgumentError apply_echo_products(base, dp_field;
                                                           io = io, heights = heights)
        end

        @testset "reference_state is not yet implemented" begin
            dp_ref = Daisho.EchoProductsParameters(enabled = true, band = "S",
                use_temp = true, temp_source = :reference_state)
            @test_throws ArgumentError apply_echo_products(base, dp_ref;
                                                           io = io, heights = heights)
        end

        @testset "standalone add_echo_products! reads gridded temperature" begin
            nt = 2
            dbz4 = Array{Float32}(undef, nx, ny, nz, nt)
            dbz4[:, :, :, 1] .= 45.0f0
            dbz4[:, :, :, 2] .= 18.0f0
            zdr4 = fill(0.5f0, nx, ny, nz, nt)
            kdp4 = fill(1.0f0, nx, ny, nz, nt)
            rho4 = fill(0.97f0, nx, ny, nz, nt)
            temp4 = Array{Float32}(undef, nx, ny, nz, nt)
            temp4[:, :, :, 1] .= 15.0f0
            temp4[:, :, :, 2] .= -10.0f0

            f = tempname() * "_tempfield.nc"
            try
                NCDataset(f, "c") do ds
                    ds.dim["X"] = nx; ds.dim["Y"] = ny; ds.dim["Z"] = nz; ds.dim["time"] = nt
                    defVar(ds, "X", Float32, ("X",))[:] = collect(0:nx-1) .* 1000.0
                    defVar(ds, "Y", Float32, ("Y",))[:] = collect(0:ny-1) .* 1000.0
                    defVar(ds, "Z", Float32, ("Z",))[:] = [500.0, 1500.0]
                    defVar(ds, "DBZ", Float32, ("X", "Y", "Z", "time"))[:] = dbz4
                    defVar(ds, "ZDR", Float32, ("X", "Y", "Z", "time"))[:] = zdr4
                    defVar(ds, "KDP", Float32, ("X", "Y", "Z", "time"))[:] = kdp4
                    defVar(ds, "RHOHV", Float32, ("X", "Y", "Z", "time"))[:] = rho4
                    defVar(ds, "TEMP_FOR_PID", Float32, ("X", "Y", "Z", "time"))[:] = temp4
                end
                p = _echo_test_params(enabled = true, temp_source = :field)
                written = add_echo_products!(f, p)
                @test "HID_CSU" in written
                NCDataset(f) do ds
                    hid = Array(ds["HID_CSU"].var)
                    for t in 1:nt
                        sl = Dict{String,Any}("DBZ" => dbz4[:, :, :, t],
                            "ZDR" => zdr4[:, :, :, t], "KDP" => kdp4[:, :, :, t],
                            "RHOHV" => rho4[:, :, :, t], "TEMP_FOR_PID" => temp4[:, :, :, t])
                        ref = apply_echo_products(sl, p.echo; io = io)
                        @test isequal(hid[:, :, :, t], ref["HID_CSU"])
                    end
                end

                # Missing temperature variable with temp_source=:field errors.
                f2 = tempname() * "_notemp.nc"
                try
                    NCDataset(f2, "c") do ds
                        ds.dim["X"] = nx; ds.dim["Y"] = ny; ds.dim["Z"] = nz; ds.dim["time"] = nt
                        defVar(ds, "X", Float32, ("X",))[:] = collect(0:nx-1) .* 1000.0
                        defVar(ds, "Y", Float32, ("Y",))[:] = collect(0:ny-1) .* 1000.0
                        defVar(ds, "Z", Float32, ("Z",))[:] = [500.0, 1500.0]
                        defVar(ds, "DBZ", Float32, ("X", "Y", "Z", "time"))[:] = dbz4
                        defVar(ds, "ZDR", Float32, ("X", "Y", "Z", "time"))[:] = zdr4
                        defVar(ds, "KDP", Float32, ("X", "Y", "Z", "time"))[:] = kdp4
                        defVar(ds, "RHOHV", Float32, ("X", "Y", "Z", "time"))[:] = rho4
                    end
                    @test_throws ArgumentError add_echo_products!(f2, p)
                finally
                    isfile(f2) && rm(f2)
                end
            finally
                isfile(f) && rm(f)
            end
        end
    end

    @testset "beam-height field enables profile temperature on PPI" begin
        io = IOParameters()
        # Config with a beam_height HEIGHT field; small grid within gate range.
        fields = [
            Daisho.FieldSpec("DBZ",   Set([:linear_interp, :define_detection])),
            Daisho.FieldSpec("ZDR",   Set([:linear_interp])),
            Daisho.FieldSpec("KDP",   Set([:weighted_interp])),
            Daisho.FieldSpec("RHOHV", Set([:weighted_interp])),
            Daisho.FieldSpec("SQI",   Set([:weighted_interp, :define_scanned])),
            Daisho.FieldSpec("HEIGHT", Set([:beam_height])),
        ]
        moments = Daisho.MomentParameters(fields)
        sc = Daisho.CartesianGridParameters(
            xmin = -2000.0, xincr = 250.0, xdim = 17,
            ymin = -2000.0, yincr = 250.0, ydim = 17,
            zmin = 500.0, zincr = 500.0, zdim = 1)
        prof = TemperatureProfile([0.0, 3000.0, 8000.0], [20.0, 0.0, -30.0])
        dp = Daisho.EchoProductsParameters(enabled = true, band = "S",
            use_temp = true, temp_source = :profile, height_field = "HEIGHT",
            temperature = prof)
        p = Daisho.DaishoParameters(moments, Daisho.QCParameters(),
            Daisho.GriddingParameters(), Daisho.GridParameters(cartesian = sc),
            io, Daisho.SynthesisParameters(), dp, Set([:grid, :gridding, :echo]))

        v = synthetic_volume(n_sweeps = 1, n_rays = 180, n_gates = 24,
                             fields = ["DBZ", "ZDR", "KDP", "RHOHV", "SQI"])
        v.sweeps[1].elevation .= 10.0   # visible height gradient with range

        f = tempname() * "_height_ppi.nc"
        try
            Daisho.grid_radar_ppi(v, f, v.time_coverage_start, p)
            g = Daisho.read_gridded_ppi(f, p)

            @testset "HEIGHT field is gridded and physical" begin
                @test haskey(g.fields, "HEIGHT")
                H = Daisho.mask_sentinels(g.fields["HEIGHT"], io)
                dbz = g.fields["DBZ"]
                # Valid exactly where DBZ is a real measurement.
                @test all(isfinite(H[i]) for i in eachindex(dbz)
                          if !Daisho._dp_invalid(dbz[i], io))
                # Increases with horizontal range and tracks beam_height.
                X = g.X; Y = g.Y
                pts = sort([(sqrt(X[i]^2 + Y[j]^2), H[i, j])
                            for i in eachindex(X), j in eachindex(Y) if isfinite(H[i, j])])
                @test pts[1][2] < pts[end][2]
                rmax, hmax = pts[end]
                @test isapprox(hmax, Daisho.beam_height(rmax, 10.0, 50.0); rtol = 0.1)
            end

            @testset "profile-via-height equals field source with same T" begin
                # T field = profile sampled at the gridded HEIGHT (sentinels kept).
                H = g.fields["HEIGHT"]
                tfield = similar(H, Float32)
                for i in eachindex(H)
                    tfield[i] = Daisho._dp_invalid(H[i], io) ? Float32(io.fill_value) :
                                Float32(temperature_celsius(prof, H[i]))
                end
                base = Dict{String,Any}(k => g.fields[k]
                    for k in ("DBZ", "ZDR", "KDP", "RHOHV", "HEIGHT"))
                dp_field = Daisho.EchoProductsParameters(enabled = true, band = "S",
                    use_temp = true, temp_source = :field, temp_field = "T_FROM_H")
                out_prof = apply_echo_products(base, dp; io = io)
                out_field = apply_echo_products(merge(base, Dict("T_FROM_H" => tfield)),
                                                dp_field; io = io)
                @test out_prof["HID_CSU"] == out_field["HID_CSU"]
            end

            @testset "temperature actually changes the PPI classification" begin
                # Controlled, realistic inputs spanning warm-to-cold heights so the
                # temperature term clearly matters (synthetic_volume DBZ is not
                # physical). HEIGHT 200 m → 8000 m maps to ~+19 °C → −30 °C.
                n = 6
                base = Dict{String,Any}(
                    "DBZ"    => fill(30.0f0, n),
                    "ZDR"    => fill(0.3f0, n),
                    "KDP"    => fill(0.2f0, n),
                    "RHOHV"  => fill(0.98f0, n),
                    "HEIGHT" => Float32.(collect(range(200.0, 8000.0; length = n))))
                dp_noT = Daisho.EchoProductsParameters(enabled = true, band = "S",
                    use_temp = true, temp_source = :profile, height_field = "",
                    temperature = prof)  # no height field, 2-D ⇒ temperature inactive
                out_T = apply_echo_products(base, dp; io = io)
                out_noT = apply_echo_products(base, dp_noT; io = io, heights = nothing)
                @test out_T["HID_CSU"] != out_noT["HID_CSU"]
            end

            @testset "standalone add_echo_products! reproduces inline (height field)" begin
                # Grid again without echo, then append via the standalone path.
                p_off = Daisho.DaishoParameters(moments, Daisho.QCParameters(),
                    Daisho.GriddingParameters(), Daisho.GridParameters(cartesian = sc),
                    io, Daisho.SynthesisParameters(),
                    Daisho.EchoProductsParameters(enabled = false), Set([:grid, :gridding, :echo]))
                f2 = tempname() * "_height_standalone.nc"
                try
                    Daisho.grid_radar_ppi(v, f2, v.time_coverage_start, p_off)
                    add_echo_products!(f2, p)
                    NCDataset(f) do a
                        NCDataset(f2) do b
                            @test isequal(Array(a["HID_CSU"].var), Array(b["HID_CSU"].var))
                        end
                    end
                finally
                    isfile(f2) && rm(f2)
                end
            end
        finally
            isfile(f) && rm(f)
        end
    end
end
