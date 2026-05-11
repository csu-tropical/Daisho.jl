@testset "GridAccumulator multi-file workflow" begin
    function _small_p()
        p = DaishoParameters()
        small_cart = Daisho.CartesianGridParameters(
            xmin = -1500.0, xincr = 500.0, xdim = 7,
            ymin = -1500.0, yincr = 500.0, ydim = 7,
            zmin = 500.0,   zincr = 500.0, zdim = 3)
        return Daisho.DaishoParameters(p.moments, p.qc, p.gridding,
            Daisho.GridParameters(cartesian = small_cart,
                                  latlon = p.grid.latlon,
                                  rhi = p.grid.rhi,
                                  spectral = p.grid.spectral),
            p.io)
    end

    @testset "round-trip per-sweep persistence == direct gridding" begin
        p = _small_p()
        v = synthetic_volume(n_sweeps = 2, n_rays = 36, n_gates = 6)
        gs = build_grid_spec(:volume_3d, v, p)

        # Direct path.
        direct_out = tempname() * "_direct.nc"
        per_sweep_jld = tempname() * "_per_sweep.jld2"
        leg_jld = tempname() * "_leg.jld2"
        leg_out = tempname() * "_leg.nc"
        per_sweep_files = String[]
        try
            Daisho.grid_radar_volume(v, direct_out, v.time_coverage_start, p)

            # Per-sweep persistence: one file per sweep, combine, finalize.
            for i in eachindex(v.sweeps)
                fpath = tempname() * "_s$(i).jld2"
                push!(per_sweep_files, fpath)
                grid_sweep_to_file(v, i, fpath, p; grid_spec = gs,
                                   merge_into_existing = false)
            end
            combine_accumulator_files(per_sweep_files, leg_jld)
            finalize_accumulator_file(leg_jld, leg_out, p;
                                      index_time = v.time_coverage_start)

            ds_direct = NCDataset(direct_out, "r")
            ds_leg    = NCDataset(leg_out, "r")
            try
                for fname in p.moments.fields
                    haskey(ds_direct, fname) || continue
                    haskey(ds_leg, fname)    || continue
                    d = Float64.(coalesce.(ds_direct[fname][:, :, :, :], -32768.0))
                    l = Float64.(coalesce.(ds_leg[fname][:, :, :, :],    -32768.0))
                    @test size(d) == size(l)
                    for ix in CartesianIndices(size(d))
                        a, b = d[ix], l[ix]
                        if a > -9000 && b > -9000
                            @test isapprox(a, b; atol = 1e-3)
                        else
                            @test a == b
                        end
                    end
                end
            finally
                close(ds_direct); close(ds_leg)
            end
        finally
            for f in (direct_out, per_sweep_jld, leg_jld, leg_out)
                isfile(f) && rm(f)
            end
            for f in per_sweep_files
                isfile(f) && rm(f)
            end
        end
    end

    @testset "merge_into_existing rolling JLD2" begin
        p = _small_p()
        v = synthetic_volume(n_sweeps = 2, n_rays = 36, n_gates = 6)
        gs = build_grid_spec(:volume_3d, v, p)

        rolling = tempname() * "_rolling.jld2"
        rolling_out = tempname() * "_rolling.nc"
        per_sweep_files = String[]
        combined_jld = tempname() * "_combined.jld2"
        combined_out = tempname() * "_combined.nc"
        try
            grid_sweep_to_file(v, 1, rolling, p; grid_spec = gs,
                               merge_into_existing = false)
            for i in 2:length(v.sweeps)
                grid_sweep_to_file(v, i, rolling, p; grid_spec = gs,
                                   merge_into_existing = true)
            end
            finalize_accumulator_file(rolling, rolling_out, p;
                                      index_time = v.time_coverage_start)

            # Multi-file path.
            for i in eachindex(v.sweeps)
                fpath = tempname() * "_s$(i).jld2"
                push!(per_sweep_files, fpath)
                grid_sweep_to_file(v, i, fpath, p; grid_spec = gs,
                                   merge_into_existing = false)
            end
            combine_accumulator_files(per_sweep_files, combined_jld)
            finalize_accumulator_file(combined_jld, combined_out, p;
                                      index_time = v.time_coverage_start)

            ds_a = NCDataset(rolling_out, "r")
            ds_b = NCDataset(combined_out, "r")
            try
                for fname in p.moments.fields
                    haskey(ds_a, fname) || continue
                    haskey(ds_b, fname) || continue
                    da = Float64.(coalesce.(ds_a[fname][:, :, :, :], -32768.0))
                    db = Float64.(coalesce.(ds_b[fname][:, :, :, :], -32768.0))
                    @test size(da) == size(db)
                    for ix in CartesianIndices(size(da))
                        a, b = da[ix], db[ix]
                        if a > -9000 && b > -9000
                            @test isapprox(a, b; atol = 1e-3)
                        else
                            @test a == b
                        end
                    end
                end
            finally
                close(ds_a); close(ds_b)
            end
        finally
            for f in (rolling, rolling_out, combined_jld, combined_out)
                isfile(f) && rm(f)
            end
            for f in per_sweep_files
                isfile(f) && rm(f)
            end
        end
    end

    @testset "merge refuses field_folds=true across files" begin
        # Build two single-sweep accumulators, both flagging VEL as
        # field_folds. combine_accumulator_files must raise.
        p = _small_p()
        v = synthetic_volume(n_sweeps = 1, n_rays = 12, n_gates = 4)
        gs = build_grid_spec(:volume_3d, v, p)

        # Build an accumulator with field_folds=true for VEL by hand.
        ff = Bool[fname == "VEL" for fname in p.moments.fields]
        acc1 = GridAccumulator(gs, p.moments.fields, p.moments.grid_type;
                                field_folds = ff)
        acc2 = GridAccumulator(gs, p.moments.fields, p.moments.grid_type;
                                field_folds = ff)
        # Populate something in VEL so the merge attempt has content.
        vel_idx = findfirst(==("VEL"), p.moments.fields)
        if vel_idx !== nothing
            acc1.weighted_sum[vel_idx, 1, 1, 1] = 1.0
            acc1.weight_total[vel_idx, 1, 1, 1] = 1.0
            acc2.weighted_sum[vel_idx, 1, 1, 1] = 2.0
            acc2.weight_total[vel_idx, 1, 1, 1] = 1.0
        end

        f1 = tempname() * "_a.jld2"
        f2 = tempname() * "_b.jld2"
        fout = tempname() * "_out.jld2"
        try
            save_accumulator(f1, acc1)
            save_accumulator(f2, acc2)
            err = nothing
            try
                combine_accumulator_files([f1, f2], fout)
            catch e
                err = e
            end
            @test err isa ArgumentError
            @test occursin("wind retrieval", err.msg)
        finally
            for f in (f1, f2, fout)
                isfile(f) && rm(f)
            end
        end
    end

    @testset "grid_sweep_to_file refuses overwrite without merge" begin
        p = _small_p()
        v = synthetic_volume(n_sweeps = 1, n_rays = 12, n_gates = 4)
        gs = build_grid_spec(:volume_3d, v, p)
        path = tempname() * "_x.jld2"
        try
            grid_sweep_to_file(v, 1, path, p; grid_spec = gs,
                               merge_into_existing = false)
            @test isfile(path)
            err = nothing
            try
                grid_sweep_to_file(v, 1, path, p; grid_spec = gs,
                                   merge_into_existing = false)
            catch e
                err = e
            end
            @test err isa ArgumentError
        finally
            isfile(path) && rm(path)
        end
    end
end
