# End-to-end tests of the variational pipeline (Phase 10):
# `analyze_radar_volumes_variational` grids a synthetic dual-Doppler pair
# through the real `WindGridAccumulator` on the Springsteel quadrature mish,
# solves the 3DVAR problem, and writes one NetCDF file with the scalar
# spectral fields and the analysis (`write_variational_netcdf`).

using Dates
using NCDatasets
using TOML
using Springsteel

# Box offset: the pipeline centres the grid on the CENTROID of the radar
# reference positions; with radars at (15.8, -24.0) and (16.0, -24.2) that is
# (15.9, -24.1), where the point (16.0, -24.0) sits ≈ (10.7, 11.1) km. The
# box below is the Phase-9 box (±12 km around (16, -24), off the radar
# baseline) expressed in the centroid frame.
const _PIPE_X0 = 10_700.0
const _PIPE_Y0 = 11_000.0

# Same template as `_solve_load` in test_variational_solve.jl (6×6×3 cells,
# mubar 3 → 18×18×9 mish; W Dirichlet at the surface; 3-unknown
# accumulation; zero background), with the box shifted to the centroid frame.
function _pipe_load(mutate! = identity)
    path = tempname() * ".toml"
    base = TOML.parsefile(Daisho.DEFAULTS_TOML_PATH)
    for (ax, lo, hi, c) in (("i", _PIPE_X0 - 12000.0, _PIPE_X0 + 12000.0, 6),
                            ("j", _PIPE_Y0 - 12000.0, _PIPE_Y0 + 12000.0, 6),
                            ("k", 0.0, 3000.0, 3))
        base["grid"]["springsteel"][ax]["min"] = lo
        base["grid"]["springsteel"][ax]["max"] = hi
        base["grid"]["springsteel"][ax]["cells"] = c
    end
    base["grid"]["springsteel"]["k"]["bc"]["W"] = Dict("min" => "dirichlet")
    base["synthesis"]["include_vertical_velocity"] = true
    base["synthesis"]["max_elevation"] = 90.0
    base["variational"] = Dict{String,Any}(
        "enabled" => true,
        "cg_progress_interval" => 0,
        "state" => ["U", "V", "W"],
        "background" => "zero",
        "mass_continuity_sigma_per_s" => 1.0e-4,
        "max_inner_iterations" => 10000,
        "inner_tolerance" => 1.0e-6,
        "background_error" => Dict(
            "default" => Dict("sigma" => 10.0, "filter_length_m" => [4000.0, 4000.0, 1000.0]),
            "W"       => Dict("sigma" => 2.0,  "filter_length_m" => [4000.0, 4000.0, 1000.0])),
    )
    mutate!(base)
    open(path, "w") do io; TOML.print(io, base); end
    return DaishoParameters(path)
end

# (n_x·n_y·n_z, …) regular-grid slab (z fastest) → (n_x, n_y, n_z).
_pipe_xyz(col, nx, ny, nz) = permutedims(reshape(col, nz, ny, nx), (3, 2, 1))

# A file variable as an (x, y, z) array, dropping the size-1 time dimension.
function _pipe_read(ds, name)
    v = ds[name]
    d = dimnames(v)
    return length(d) == 4 && first(d) == "time" ? Array(v[1, :, :, :]) : Array(v[:, :, :])
end

@testset "Variational pipeline (Phase 10)" begin
    ref_lat, ref_lon = 15.9, -24.1
    a_uy, a_vx = 2.0e-4, -1.0e-4
    wind = (x, y, z) -> (8.0 + a_uy * (y - _PIPE_Y0), -3.0 + a_vx * (x - _PIPE_X0), 0.0)
    vort_true = a_vx - a_uy
    volA, volB = make_synthetic_dual_doppler_field(wind; ref_lat = ref_lat, ref_lon = ref_lon,
        radarA_lat = 15.8, radarA_lon = -24.0, radarB_lat = 16.0, radarB_lon = -24.2,
        dbz = (x, y, z) -> 30.0)
    vols = [volA, volB]
    index_time = DateTime(2026, 9, 23, 12, 0, 0)

    p = _pipe_load()
    sgrid = Daisho.create_radar_grid(p)
    file = tempname() * ".nc"
    local result, prob
    try
        t_run = @elapsed (result, prob) = analyze_radar_volumes_variational(
            vols, file, index_time, sgrid, p; institution = "CSU", source = "synthetic",
            return_problem = true)
        @info "Phase 10 pipeline run" seconds = t_run cg = repr(result.stats.inner_iterations)

        @testset "return values" begin
            @test result isa VariationalResult
            @test prob isa VariationalProblem
            @test result.converged
            @test prob.stage1 isa Daisho.SynthesisOutput
            # The state grid holds the analysis (evaluate_analysis! ran).
            sg = prob.state_grid
            colU = sg.params.vars["U"]
            @test maximum(abs.(sg.physical[:, colU, 1] .- vec(analysis_field(result, "U")))) < 1e-12
        end

        @testset "regular-grid file contents" begin
            @test isfile(file)
            NCDataset(file) do ds
                for name in ("DBZ", "U", "V", "W", "VORT", "DIV", "MCRESIDUAL",
                             "grid_mapping", "time", "x", "y", "z")
                    @test haskey(ds, name)
                end
                for name in ("U_BACKGROUND", "U_INCREMENT", "U_x", "U_MISH", "USTD",
                             "QFLAG_MISH", "NOBS_VR", "RHOBAR", "x_mish")
                    @test !haskey(ds, name)
                end
                @test ds["time"][1] == index_time
                # Same dims, element type and fill as the scalar fields.
                for name in ("U", "V", "W", "VORT", "DIV", "MCRESIDUAL")
                    @test dimnames(ds[name]) == dimnames(ds["DBZ"])
                    @test eltype(ds[name].var) == eltype(ds["DBZ"].var)
                    @test ds[name].attrib["grid_mapping"] == "grid_mapping"
                end
                @test dimnames(ds["DBZ"]) == ("time", "x", "y", "z")
                @test ds["U"].attrib["standard_name"] == "eastward_wind"
                @test ds["W"].attrib["standard_name"] == "upward_air_velocity"
                @test ds["VORT"].attrib["standard_name"] == "atmosphere_relative_vorticity"
                @test ds["DIV"].attrib["standard_name"] == "divergence_of_wind"
                @test ds["MCRESIDUAL"].attrib["units"] == "s-1"
                # Global attributes: the write_radar_netcdf set plus the solve summary.
                @test ds.attrib["institution"] == "CSU"
                @test ds.attrib["reference_latitude"] ≈ ref_lat
                @test ds.attrib["variational_background"] == "zero"
                @test ds.attrib["variational_outer_iterations"] == result.stats.outer_iterations
                @test ds.attrib["variational_cg_iterations"] == sum(result.stats.inner_iterations)
                @test ds.attrib["variational_converged"] == 1
                @test ds.attrib["variational_quality_flag_map"] == "nearest"
                @test ds.attrib["variational_mask_quality_flags"] == ""

                # Stage-1 quality flag resampled to the regular grid.
                @test haskey(ds, "QFLAG")
                @test dimnames(ds["QFLAG"]) == dimnames(ds["DBZ"])
                @test eltype(ds["QFLAG"].var) == Int8
                @test !haskey(ds["QFLAG"].attrib, "_FillValue")
                @test ds["QFLAG"].attrib["flag_masks"] == Int8[1, 2, 4, 8]
                @test ds["QFLAG"].attrib["flag_meanings"] ==
                      "sigma1_above_threshold sigma2_above_threshold singular_geometry no_data"
                @test ds["QFLAG"].attrib["grid_mapping"] == "grid_mapping"
                @test occursin("nearest", ds["QFLAG"].attrib["comment"])

                x = Float64.(ds["x"][:]); y = Float64.(ds["y"][:]); z = Float64.(ds["z"][:])
                gp = sgrid.params
                @test x == collect(LinRange(gp.iMin, gp.iMax, gp.i_regular_out))
                @test y == collect(LinRange(gp.jMin, gp.jMax, gp.j_regular_out))
                @test z == collect(LinRange(gp.kMin, gp.kMax, gp.k_regular_out))
                nx, ny, nz = length(x), length(y), length(z)

                U = _pipe_read(ds, "U"); V = _pipe_read(ds, "V"); W = _pipe_read(ds, "W")
                @test size(U) == (nx, ny, nz)
                # QFLAG = the nearest-mish resampling of stage 1's flag; with
                # no mask the winds stay finite at flagged points too.
                gzq = Daisho.get_springsteel_gridpoints_zyx(sgrid)
                halfq = (sgrid.ibasis.data[1, 1, 1].params.DX, sgrid.jbasis.data[1, 1, 1].params.DX,
                         sgrid.kbasis.data[1, 1, 1].params.DX) ./ 2
                Q = _pipe_read(ds, "QFLAG")
                @test Q == Daisho.regular_quality_flag(prob.stage1.quality_flag,
                    (gzq[1, 1, :, 3], gzq[1, :, 1, 2], gzq[:, 1, 1, 1]), (x, y, z), halfq, :nearest)
                @test all(isfinite, U) && all(isfinite, V) && all(isfinite, W)

                # Parity with Springsteel's regular-grid transform of the
                # analysis coefficients.
                sg = prob.state_grid
                names = Daisho.variable_names(prob.state)
                Daisho.coefficients_to_springsteel_spectral!(sg, result.coefficients, names)
                reg = Springsteel.regularGridTransform(sg, x, y, z)
                cU, cV, cW = sg.params.vars["U"], sg.params.vars["V"], sg.params.vars["W"]
                @test maximum(abs.(U .- _pipe_xyz(reg[:, cU, 1], nx, ny, nz))) < 1e-10
                @test maximum(abs.(V .- _pipe_xyz(reg[:, cV, 1], nx, ny, nz))) < 1e-10
                @test maximum(abs.(W .- _pipe_xyz(reg[:, cW, 1], nx, ny, nz))) < 1e-10
                vort = _pipe_xyz(reg[:, cV, 2] .- reg[:, cU, 4], nx, ny, nz)
                div = _pipe_xyz(reg[:, cU, 2] .+ reg[:, cV, 4], nx, ny, nz)
                dl = [Daisho.dlnrho_dz(prob.reference, zz) for zz in z]
                mc = div .+ _pipe_xyz(reg[:, cW, 6], nx, ny, nz) .+ W .* reshape(dl, 1, 1, :)
                @test maximum(abs.(_pipe_read(ds, "VORT") .- vort)) < 1e-12
                @test maximum(abs.(_pipe_read(ds, "DIV") .- div)) < 1e-12
                @test maximum(abs.(_pipe_read(ds, "MCRESIDUAL") .- mc)) < 1e-12

                # Truth in the interior (one cell in from the lateral
                # boundaries), all regular levels.
                interior = [abs(x[i] - _PIPE_X0) < 8000.0 && abs(y[j] - _PIPE_Y0) < 8000.0
                            for i in 1:nx, j in 1:ny, k in 1:nz]
                @test count(interior) > 0
                Ut = [wind(x[i], y[j], z[k])[1] for i in 1:nx, j in 1:ny, k in 1:nz]
                Vt = [wind(x[i], y[j], z[k])[2] for i in 1:nx, j in 1:ny, k in 1:nz]
                eU = maximum(abs.(U[interior] .- Ut[interior]))
                eV = maximum(abs.(V[interior] .- Vt[interior]))
                eW = maximum(abs.(W[interior]))
                @test eU < 0.3
                @test eV < 0.3
                @test eW < 0.5
                mcf = _pipe_read(ds, "MCRESIDUAL")
                mc_rms = sqrt(sum(abs2, mcf) / length(mcf))
                @test mc_rms < 1e-5
                vortf = _pipe_read(ds, "VORT")
                vort_err = sqrt(sum(abs2, vortf[interior] .- vort_true) / count(interior))
                @test vort_err < 0.2 * abs(vort_true)
                @info "Phase 10 regular-grid analysis" max_err_U = eU max_err_V = eV max_abs_W = eW mc_rms = mc_rms vort_rms_err = vort_err
            end
        end

        @testset "scalar fields match grid_radar_volume_spectral" begin
            sgrid2 = Daisho.create_radar_grid(p)
            file2 = tempname() * ".nc"
            try
                Daisho.grid_radar_volume_spectral(vols, file2, nothing, sgrid2, p)
                @test isequal(sgrid2.physical, sgrid.physical)
                @test isequal(sgrid2.spectral, sgrid.spectral)
                NCDataset(file) do ds1
                    NCDataset(file2) do ds2
                        @test !haskey(ds2, "time")          # historical layout unchanged
                        for name in keys(Daisho.field_index_dict(p))
                            @test isequal(_pipe_read(ds1, name), _pipe_read(ds2, name))
                        end
                        @test isequal(ds1["x"][:], ds2["x"][:])
                    end
                end
            finally
                rm(file2; force = true)
            end
        end

        @testset "background, increment, derivatives, mish (writer on the same solve)" begin
            p_all = _pipe_load() do b
                b["variational"]["output_background"] = true
                b["variational"]["output_increment"] = true
                b["variational"]["output_mish"] = true
            end
            file3 = tempname() * ".nc"
            try
                ref = Daisho._volumes_reference_position(vols)
                write_variational_netcdf(file3, sgrid, prob, result, volA, p_all;
                    index_time = index_time, ref_lat = ref[1], ref_lon = ref[2],
                    include_derivatives = true)
                NCDataset(file3) do ds
                    # Background + increment = analysis on the regular grid.
                    for name in ("U", "V", "W")
                        @test haskey(ds, name * "_BACKGROUND")
                        @test haskey(ds, name * "_INCREMENT")
                        A = _pipe_read(ds, name)
                        B = _pipe_read(ds, name * "_BACKGROUND")
                        I = _pipe_read(ds, name * "_INCREMENT")
                        @test maximum(abs.(B .+ I .- A)) < 1e-6
                        @test dimnames(ds[name * "_BACKGROUND"]) == dimnames(ds["DBZ"])
                    end
                    # Zero background.
                    @test all(iszero, _pipe_read(ds, "U_BACKGROUND"))

                    # Derivatives: the same suffixes Springsteel writes for scalars.
                    for sfx in ("_x", "_xx", "_y", "_yy", "_z", "_zz")
                        @test haskey(ds, "DBZ" * sfx)
                        for name in ("U", "V", "W")
                            @test haskey(ds, name * sfx)
                            @test dimnames(ds[name * sfx]) == dimnames(ds["DBZ" * sfx])
                        end
                    end
                    x = Float64.(ds["x"][:]); y = Float64.(ds["y"][:]); z = Float64.(ds["z"][:])
                    nx, ny, nz = length(x), length(y), length(z)
                    sg = prob.state_grid
                    Daisho.coefficients_to_springsteel_spectral!(sg, result.coefficients,
                        Daisho.variable_names(prob.state))
                    reg = Springsteel.regularGridTransform(sg, x, y, z)
                    cU, cV = sg.params.vars["U"], sg.params.vars["V"]
                    @test maximum(abs.(_pipe_read(ds, "U_x") .- _pipe_xyz(reg[:, cU, 2], nx, ny, nz))) < 1e-12
                    @test maximum(abs.(_pipe_read(ds, "V_y") .- _pipe_xyz(reg[:, cV, 4], nx, ny, nz))) < 1e-12
                    @test maximum(abs.(_pipe_read(ds, "DIV") .- (_pipe_read(ds, "U_x") .+ _pipe_read(ds, "V_y")))) < 1e-12

                    # Mish lattice.
                    nzm, nym, nxm = prob.state.dims
                    @test ds.dim["x_mish"] == nxm
                    @test ds.dim["y_mish"] == nym
                    @test ds.dim["z_mish"] == nzm
                    gz = Daisho.get_springsteel_gridpoints_zyx(sgrid)
                    @test ds["x_mish"][:] == gz[1, 1, :, 3]
                    @test ds["y_mish"][:] == gz[1, :, 1, 2]
                    @test ds["z_mish"][:] == gz[:, 1, 1, 1]
                    for name in ("U_MISH", "V_MISH", "W_MISH", "U_STAGE1", "V_STAGE1",
                                 "USTD", "VSTD", "DET", "NGATES", "QFLAG_MISH", "NOBS_VR")
                        @test haskey(ds, name)
                        @test dimnames(ds[name]) == ("time", "x_mish", "y_mish", "z_mish")
                    end
                    # U_MISH is written as Float64: exact round trip.
                    @test eltype(ds["U_MISH"].var) == Float64
                    @test _pipe_read(ds, "U_MISH") == permutedims(analysis_field(result, "U"), (3, 2, 1))
                    @test _pipe_read(ds, "W_MISH") == permutedims(analysis_field(result, "W"), (3, 2, 1))
                    @test eltype(ds["NOBS_VR"].var) == Int8
                    @test _pipe_read(ds, "NOBS_VR") == permutedims(result.nobs_vr, (3, 2, 1))
                    @test eltype(ds["QFLAG_MISH"].var) == Int8
                    @test _pipe_read(ds, "QFLAG_MISH") == permutedims(prob.stage1.quality_flag, (3, 2, 1))
                    # The regular-grid QFLAG coexists with the mish one.
                    @test dimnames(ds["QFLAG"]) == dimnames(ds["DBZ"])
                    # Stage-1 diagnostics: QC'd (blanked where QFLAG != 0) with the [io] fill.
                    @test ds["USTD"].attrib["_FillValue"] == Float32(p.io.fill_value)
                    ok = permutedims(prob.stage1.quality_flag, (3, 2, 1)) .== 0
                    us1 = ds["U_STAGE1"].var[1, :, :, :]
                    @test us1[ok] == Float32.(permutedims(prob.stage1.comp1, (3, 2, 1))[ok])
                    @test all(==(Float32(p.io.fill_value)), us1[.!ok])
                    @test dimnames(ds["RHOBAR"]) == ("z_mish",)
                    @test ds["RHOBAR"][:] ≈ [Daisho.reference_density(prob.reference, zz)
                                             for zz in prob.state.z_mish]
                    @test ds["RHOBAR"].attrib["standard_name"] == "air_density"
                end
            finally
                rm(file3; force = true)
            end
        end

        @testset "quality-flag masking (writer on the same solve)" begin
            ref = Daisho._volumes_reference_position(vols)
            q1 = prob.stage1.quality_flag
            @info "Stage-1 QFLAG bit counts" sigma1 = count(!iszero, q1 .& Int8(1)) sigma2 = count(!iszero, q1 .& Int8(2)) singular = count(!iszero, q1 .& Int8(4)) no_data = count(!iszero, q1 .& Int8(8)) n = length(q1)
            function _write_masked(mapname, flags; kw...)
                pm = _pipe_load() do b
                    b["variational"]["quality_flag_map"] = mapname
                    b["variational"]["mask_quality_flags"] = flags
                    for (k, v) in kw; b["variational"][String(k)] = v; end
                end
                f = tempname() * ".nc"
                write_variational_netcdf(f, sgrid, prob, result, volA, pm;
                    index_time = index_time, ref_lat = ref[1], ref_lon = ref[2],
                    include_derivatives = true)
                return f
            end
            # NCDatasets reads the NaN fill back as `missing`.
            blank(A) = map(a -> ismissing(a) || isnan(a), A)
            winds = ("U", "V", "W", "VORT", "DIV", "MCRESIDUAL", "U_x", "W_zz",
                     "U_BACKGROUND", "V_INCREMENT")
            files = String[]
            try
                # Explicit empty list: bit-identical to the default writer.
                f0 = _write_masked("nearest", String[]; output_background = true,
                                   output_increment = true)
                push!(files, f0)
                fm = _write_masked("nearest", ["no_data"]; output_background = true,
                                   output_increment = true)
                push!(files, fm)
                NCDataset(file) do ds_ref
                    NCDataset(f0) do ds0
                        for name in ("U", "V", "W", "VORT", "DIV", "MCRESIDUAL", "QFLAG", "DBZ")
                            @test isequal(_pipe_read(ds0, name), _pipe_read(ds_ref, name))
                        end
                    end
                end
                NCDataset(fm) do ds
                    @test ds.attrib["variational_mask_quality_flags"] == "no_data"
                    Q = _pipe_read(ds, "QFLAG")
                    masked = (Q .& Int8(8)) .!= 0
                    @test any(masked)
                    @test !all(masked)
                    for name in winds
                        A = _pipe_read(ds, name)
                        @test blank(A) == masked
                    end
                    NCDataset(f0) do ds0
                        @test isequal(_pipe_read(ds, "DBZ"), _pipe_read(ds0, "DBZ"))
                        @test isequal(_pipe_read(ds, "DBZ_x"), _pipe_read(ds0, "DBZ_x"))
                        @test isequal(_pipe_read(ds, "QFLAG"), _pipe_read(ds0, "QFLAG"))
                        U0 = _pipe_read(ds0, "U")
                        @test _pipe_read(ds, "U")[.!masked] == U0[.!masked]
                    end
                end

                # worst ⊇ nearest ⊇ best, in masked points.
                all_flags = [m for (m, _) in Daisho.QFLAG_MEANINGS]
                nmask = Dict{String,Int}()
                Q0 = NCDataset(ds -> _pipe_read(ds, "QFLAG"), f0)
                dbz0 = NCDataset(ds -> _pipe_read(ds, "DBZ"), f0)
                for mapname in ("worst", "nearest", "best")
                    f = _write_masked(mapname, all_flags)
                    push!(files, f)
                    NCDataset(f) do ds
                        @test ds.attrib["variational_quality_flag_map"] == mapname
                        @test occursin(mapname, ds["QFLAG"].attrib["comment"])
                        nmask[mapname] = count(blank(_pipe_read(ds, "U")))
                        @test blank(_pipe_read(ds, "U")) == (_pipe_read(ds, "QFLAG") .!= 0)
                        @test isequal(_pipe_read(ds, "DBZ"), dbz0)
                    end
                end
                @info "Masked regular points per quality_flag_map" worst = nmask["worst"] nearest = nmask["nearest"] best = nmask["best"] n = length(Q0)
                @test nmask["worst"] >= nmask["nearest"] >= nmask["best"]
                @test nmask["worst"] > nmask["best"]
            finally
                foreach(f -> rm(f; force = true), files)
            end
        end
    finally
        rm(file; force = true)
    end

    @testset "pipeline flags and errors" begin
        # write_output = false skips the file.
        file4 = tempname() * ".nc"
        r4 = analyze_radar_volumes_variational(vols, file4, index_time,
            Daisho.create_radar_grid(p), p; write_output = false)
        @test r4 isa VariationalResult
        @test !isfile(file4)

        # Single-volume convenience form reaches the same code path.
        @test hasmethod(analyze_radar_volumes_variational,
            Tuple{Volume,AbstractString,Any,typeof(sgrid),DaishoParameters})

        p_off = _pipe_load(b -> (b["variational"]["enabled"] = false))
        err = try
            analyze_radar_volumes_variational(vols, tempname() * ".nc", index_time,
                Daisho.create_radar_grid(p_off), p_off)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("enabled", err.msg)

        p_novel = _pipe_load(b -> (b["fields"]["VEL"] = ["weighted_interp"]))
        err = try
            analyze_radar_volumes_variational(vols, tempname() * ".nc", index_time,
                Daisho.create_radar_grid(p_novel), p_novel)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("velocity", err.msg)

        # Volumes without any gate of the velocity field (here: the field is
        # renamed so the volumes' VEL never matches) → a clear error instead
        # of a silent zero analysis.
        p_nogate = _pipe_load(b -> begin
            b["fields"]["VU"] = b["fields"]["VEL"]
            delete!(b["fields"], "VEL")
        end)
        err = try
            analyze_radar_volumes_variational(vols, tempname() * ".nc", index_time,
                Daisho.create_radar_grid(p_nogate), p_nogate)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("no gate of the velocity field `VU`", err.msg)

        # A hand-built grid whose vars do not match [fields] is refused.
        bad = Daisho.create_radar_grid(p.grid.springsteel, Dict("DBZ" => 1);
                                       allowed = Set(["DBZ", "W"]))
        @test_throws ArgumentError analyze_radar_volumes_variational(
            vols, tempname() * ".nc", index_time, bad, p)
        @test_throws ArgumentError analyze_radar_volumes_variational(
            Volume[], tempname() * ".nc", index_time, sgrid, p)
    end

    @testset "regular_quality_flag" begin
        # 3 × 1 × 1 cells, mubar 3 Gauss points → 9 × 3 × 3 mish; the regular
        # output points are the 4 × 2 × 2 cell nodes.
        pq = _pipe_load() do b
            for (ax, c) in (("i", 3), ("j", 1), ("k", 1))
                b["grid"]["springsteel"][ax]["min"] = 0.0
                b["grid"]["springsteel"][ax]["max"] = 3000.0 * c
                b["grid"]["springsteel"][ax]["cells"] = c
            end
        end
        g = Daisho.create_radar_grid(pq)
        gz = Daisho.get_springsteel_gridpoints_zyx(g)
        xm, ym, zm = gz[1, 1, :, 3], gz[1, :, 1, 2], gz[:, 1, 1, 1]
        @test (length(xm), length(ym), length(zm)) == (9, 3, 3)
        gp = g.params
        x = collect(LinRange(gp.iMin, gp.iMax, gp.i_regular_out))
        y = collect(LinRange(gp.jMin, gp.jMax, gp.j_regular_out))
        z = collect(LinRange(gp.kMin, gp.kMax, gp.k_regular_out))
        @test x ≈ [0.0, 3000.0, 6000.0, 9000.0]
        half = (1500.0, 1500.0, 1500.0)
        rq(q, m) = Daisho.regular_quality_flag(q, (xm, ym, zm), (x, y, z), half, m)

        # Nearest: the flag encodes the mish x index. Boundary nodes take the
        # first/last mish point; an interior node sits midway between two
        # Gauss points, and the tie goes to the lower index.
        qi = Int8[i for k in 1:3, j in 1:3, i in 1:9]
        out = rq(qi, :nearest)
        @test size(out) == (4, 2, 2)
        @test out[:, 1, 1] == Int8[1, 3, 6, 9]
        @test all(out[:, j, k] == out[:, 1, 1] for j in 1:2, k in 1:2)
        @test rq(qi, "nearest") == out                   # string method accepted

        # A flag at one mish x column (all y, z). Index 3 (0.887·DX) is the
        # nearest of node 2 and within half a cell of no other node.
        q3 = zeros(Int8, 3, 3, 9); q3[:, :, 3] .= Int8(8)
        @test rq(q3, :nearest)[:, 1, 1] == Int8[0, 8, 0, 0]
        @test rq(q3, :worst)[:, 1, 1] == Int8[0, 8, 0, 0]
        @test all(iszero, rq(q3, :best))                 # node 2 also sees indices 2, 4, 5
        # Index 2 (the cell-1 midpoint, exactly half a cell from nodes 1 and
        # 2) is nobody's nearest, but within half a cell of both.
        q2 = zeros(Int8, 3, 3, 9); q2[:, :, 2] .= Int8(4)
        @test all(iszero, rq(q2, :nearest))
        @test rq(q2, :worst)[:, 1, 1] == Int8[4, 4, 0, 0]
        @test all(iszero, rq(q2, :best))
        # Best keeps a bit only when every neighbour carries it: node 1's
        # neighbourhood is mish x indices {1, 2}.
        q12 = zeros(Int8, 3, 3, 9); q12[:, :, 1:2] .= Int8(1)
        @test rq(q12, :best)[:, 1, 1] == Int8[1, 0, 0, 0]
        @test rq(q12, :worst)[:, 1, 1] == Int8[1, 1, 0, 0]
        # A single mish point flagged (not a whole column): worst spreads it
        # to the y/z nodes within half a cell too.
        qp = zeros(Int8, 3, 3, 9); qp[1, 1, 1] = Int8(2)
        @test rq(qp, :nearest)[1, 1, 1] == Int8(2)
        @test count(!iszero, rq(qp, :nearest)) == 1
        @test count(!iszero, rq(qp, :worst)) == 1
        @test all(iszero, rq(qp, :best))

        # best ⊆ nearest ⊆ worst, bitwise, on a random field.
        qr = rand(Int8(0):Int8(15), 3, 3, 9)
        b, n, w = rq(qr, :best), rq(qr, :nearest), rq(qr, :worst)
        @test all(iszero, b .& .~n)
        @test all(iszero, n .& .~w)

        @test_throws ArgumentError rq(qi, :median)
        @test_throws ArgumentError rq(qi, "any")
        @test_throws DimensionMismatch Daisho.regular_quality_flag(qi, (ym, ym, zm),
            (x, y, z), half, :nearest)
    end
end
