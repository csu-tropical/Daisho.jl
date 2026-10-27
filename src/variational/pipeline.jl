# The file-writing variational pipeline (plan §5): grid radar volumes through
# the stage-1 `WindGridAccumulator` on the Springsteel quadrature mish, run the
# scalar spectral products exactly as `grid_radar_volume_spectral` does, build
# and solve the `VariationalProblem`, and write one NetCDF file holding the
# scalar fields and the analysis on the regular output grid (plus, optionally,
# mish-lattice diagnostics).

"""
    analyze_radar_volumes_variational(volumes::AbstractVector{<:Volume}, output_file, index_time,
        sgrid::RRR_Grid, p::DaishoParameters; heading = -9999.0, institution = "", source = "",
        include_derivatives = false, frame = CartesianFrame(), reference = nothing,
        nchunks = Threads.nthreads(), write_output = true, return_problem = false)
        -> VariationalResult  (or (VariationalResult, VariationalProblem) with return_problem)
    analyze_radar_volumes_variational(volume::Volume, output_file, index_time, sgrid, p; kwargs...)

The variational (3DVAR) wind analysis from radar volumes to a NetCDF file.

1. Requires `[variational] enabled = true` (`ArgumentError` otherwise) and a
   grid whose variable map matches `[fields]` (as for
   [`grid_radar_volume_spectral`](@ref)).
2. Centres the grid on the **centroid** of the volumes' reference positions
   ([`volume_reference_position`](@ref)), builds the accumulator on `sgrid`'s
   quadrature lattice ([`build_springsteel_grid_spec`](@ref),
   [`build_accumulator`](@ref) with the reference-density profile
   `reference`, default [`build_reference_profile`](@ref)`(p, sgrid)`) — a
   [`WindGridAccumulator`](@ref), so one `[fields]` entry must carry the
   `velocity` tag (`ArgumentError` otherwise) — and grids every sweep of every
   volume.
3. Runs the scalar spectral products on `sgrid` exactly as
   [`grid_radar_volume_spectral`](@ref) does (the scalar fields written are
   identical).
4. Builds the [`VariationalProblem`](@ref) from the accumulator
   (stage-1 [`finalize_wind`](@ref) in `frame`, state vector, background,
   radial-velocity super-obs, mass continuity, scalar value obs), solves it
   ([`solve_variational`](@ref)) and loads the analysis into the state grid
   ([`evaluate_analysis!`](@ref)).
5. With `write_output = true`, writes `output_file` with
   [`write_variational_netcdf`](@ref); `index_time` (a `DateTime`, or
   `nothing` for no time dimension) becomes the file's `time` coordinate.

Returns the [`VariationalResult`](@ref); with `return_problem = true`, the
tuple `(result, problem)` so the stage-1 product (`problem.stage1`), the
observation set and the state grid (holding the analysis) are reachable.
"""
function analyze_radar_volumes_variational(volumes::AbstractVector{<:Volume},
        output_file::AbstractString, index_time, sgrid::RRR_Grid, p::DaishoParameters;
        heading::Real = -9999.0, institution::String = "", source::String = "",
        include_derivatives::Bool = false, frame::SynthesisFrame = CartesianFrame(),
        reference::Union{Nothing,AbstractReferenceProfile} = nothing,
        nchunks::Integer = Threads.nthreads(), write_output::Bool = true,
        return_problem::Bool = false)
    for_op = "analyze_radar_volumes_variational"
    p.variational.enabled || throw(ArgumentError(
        "$(for_op): [variational] enabled = false; set `enabled = true` in the " *
        "[variational] block to run the variational analysis"))
    isempty(volumes) && throw(ArgumentError("$(for_op): no volumes given"))
    _require_radar_vars(sgrid, p; for_op = for_op)
    ref_lat, ref_lon = _volumes_reference_position(volumes)
    spec = build_springsteel_grid_spec(sgrid;
        reference_latitude = ref_lat, reference_longitude = ref_lon)
    ref = reference === nothing ? build_reference_profile(p, sgrid) : reference
    acc = build_accumulator(spec, p; reference = ref)
    acc isa WindGridAccumulator || throw(ArgumentError(
        "$(for_op): the variational analysis needs the stage-1 wind accumulator, " *
        "which requires a [fields] entry tagged `velocity` (the radial velocity); " *
        "none of $(join(sort([fs.name for fs in p.moments.fields]), ", ")) carries it"))
    _grid_volumes!(acc, sgrid, volumes, p; heading = heading)
    if sum(acc.gate_count) == 0
        velname = field_with_tag(p, :velocity; for_op = for_op)
        throw(ArgumentError(
            "$(for_op): no gate of the velocity field `$(velname)` was gridded, so " *
            "there are no radial-velocity observations. Check that the volumes " *
            "carry a field named `$(velname)` (read_cfradial warns when a file has " *
            "no fields), that its values are not all missing, and that the " *
            "[grid.springsteel] domain overlaps the radar coverage."))
    end

    radar_grid = finalize_grid(acc.scalar)
    _scalar_spectral_products!(sgrid, radar_grid, p)

    prob = VariationalProblem(acc, radar_grid, sgrid, p;
                              frame = frame, reference = ref, nchunks = nchunks)
    result = solve_variational(prob)
    evaluate_analysis!(prob.state_grid, result)

    if write_output
        write_variational_netcdf(output_file, sgrid, prob, result, volumes[1], p;
            index_time = index_time, ref_lat = ref_lat, ref_lon = ref_lon,
            institution = institution, source = source, heading = heading,
            include_derivatives = include_derivatives)
    end
    return return_problem ? (result, prob) : result
end

analyze_radar_volumes_variational(volume::Volume, output_file::AbstractString, index_time,
                                  sgrid::RRR_Grid, p::DaishoParameters; kwargs...) =
    analyze_radar_volumes_variational([volume], output_file, index_time, sgrid, p; kwargs...)

# ── NetCDF writer ─────────────────────────────────────────────────────────────

# Regular-grid slots Springsteel writes per variable (value + first/second
# derivatives), with its Cartesian suffixes.
const _VAR_REGULAR_SUFFIXES = ("", "_x", "_xx", "_y", "_yy", "_z", "_zz")

# (n_x·n_y·n_z) regular slab, z fastest → (n_x, n_y, n_z).
_regular_xyz(col::AbstractVector, nx::Int, ny::Int, nz::Int) =
    permutedims(reshape(col, nz, ny, nx), (3, 2, 1))

# The regular-grid evaluation (n_x·n_y·n_z, nvars, 7) of the Daisho-layout
# coefficient vector `a` on the state grid. Leaves `grid.spectral` holding `a`.
function _state_regular(grid, a::AbstractVector{Float64}, names, x, y, z)
    coefficients_to_springsteel_spectral!(grid, a, names)
    return Springsteel.regularGridTransform(grid, x, y, z)
end

"""
    regular_quality_flag(qflag, mish, regular, half_cell, method) -> Array{Int8,3}

Resample the stage-1 bitwise quality flag `qflag` (`(nz, ny, nx)`, on the
quadrature lattice) onto the regular output points.

`mish = (x_mish, y_mish, z_mish)` and `regular = (x, y, z)` are the
ascending coordinates per axis; `half_cell = (DX, DY, DZ) ./ 2`. `method`
(`Symbol` or string) selects the rule, per regular point:

- `:nearest` — the flag of the nearest mish point (per axis; an exact tie
  goes to the lower index);
- `:worst` — the bitwise OR over every mish point within half a cell
  (`|x_mish − x| ≤ DX/2` on each axis, inclusive), so any flagged neighbour
  flags the point;
- `:best` — the bitwise AND over the same neighbourhood, so a bit survives
  only if every neighbour carries it.

An axis whose neighbourhood is empty falls back to the nearest index, so
`best ⊆ nearest ⊆ worst` bitwise. Returns `(nx, ny, nz)` (the writer's
layout). Any other `method` throws `ArgumentError`.
"""
function regular_quality_flag(qflag::AbstractArray{Int8,3},
                              mish::NTuple{3,AbstractVector{<:Real}},
                              regular::NTuple{3,AbstractVector{<:Real}},
                              half_cell::NTuple{3,Real}, method::Union{Symbol,AbstractString})
    m = Symbol(method)
    m in (:nearest, :worst, :best) || throw(ArgumentError(
        "regular_quality_flag: method must be :nearest, :worst or :best, got $(repr(method))"))
    nzm, nym, nxm = size(qflag)
    map(length, mish) == (nxm, nym, nzm) || throw(DimensionMismatch(
        "regular_quality_flag: mish coordinate lengths $(map(length, mish)) do not match " *
        "the flag's (nx, ny, nz) = $((nxm, nym, nzm))"))
    sx, sy, sz = (_qflag_index_sets(mish[a], regular[a], half_cell[a], m) for a in 1:3)
    out = Array{Int8,3}(undef, length(sx), length(sy), length(sz))
    for k in eachindex(sz), j in eachindex(sy), i in eachindex(sx)
        if m === :nearest
            out[i, j, k] = qflag[sz[k][1], sy[j][1], sx[i][1]]
        else
            acc = m === :worst ? Int8(0) : Int8(-1)       # OR identity / AND identity (all bits)
            for kk in sz[k], jj in sy[j], ii in sx[i]
                f = qflag[kk, jj, ii]
                acc = m === :worst ? (acc | f) : (acc & f)
            end
            out[i, j, k] = acc
        end
    end
    return out
end

# Per regular coordinate on one axis, the mish indices it draws its flag
# from: the nearest (ties → lower index) for :nearest; all within half a cell
# (inclusive, with a small relative tolerance) for :worst/:best, falling back
# to the nearest when none is.
function _qflag_index_sets(xm::AbstractVector{<:Real}, xr::AbstractVector{<:Real},
                           half::Real, method::Symbol)
    isempty(xm) && throw(ArgumentError("regular_quality_flag: empty mish axis"))
    tol = 1e-9 * abs(half)
    sets = Vector{Vector{Int}}(undef, length(xr))
    for (r, x) in enumerate(xr)
        dmin, inear = Inf, 1
        for (i, xi) in enumerate(xm)
            d = abs(xi - x)
            if d < dmin - tol
                dmin, inear = d, i
            end
        end
        if method === :nearest
            sets[r] = [inear]
        else
            s = [i for (i, xi) in enumerate(xm) if abs(xi - x) <= half + tol]
            sets[r] = isempty(s) ? [inear] : s
        end
    end
    return sets
end

"""
    write_variational_netcdf(output_file, sgrid, prob::VariationalProblem,
        result::VariationalResult, volume::Volume, p::DaishoParameters;
        index_time = nothing, ref_lat, ref_lon, institution = "", source = "",
        heading = -9999.0, include_derivatives = false) -> output_file

Write the variational analysis and the gridded scalar fields to one CF-1.12
NetCDF file.

`sgrid` is the obs/scalar grid after its spectral transform (as left by
[`analyze_radar_volumes_variational`](@ref) or
[`grid_radar_volume_spectral`](@ref)); `volume` supplies the time-coverage
metadata. `p` selects the optional outputs (`[variational] output_background`,
`output_increment`, `output_mish`), so the same solve can be written with
different settings.

# Layout
The file is created by [`write_radar_netcdf`](@ref) (Springsteel's regular
output grid: dims `x, y, z`, the scalar fields, `grid_mapping`, the radar
global attributes), with `time` = `index_time` (a `DateTime`, or `nothing`
for no time dimension). Every regular-grid field is `Float64` with a `NaN`
fill on `("time", "x", "y", "z")` (Julia order; `("x", "y", "z")` without
`time`), exactly like the scalars. Appended:

- **Analysis** (regular grid, from Springsteel's `regularGridTransform` of
  the analysis coefficients): `U`, `V`, `W` (the state's wind-role
  variables, by name; CF `eastward_wind`/`northward_wind`/`upward_air_velocity`).
  A `[fields]`-derived state variable (e.g. `DBZ`) is written as
  `<name>_ANALYSIS`, since `<name>` is the gridded scalar field.
- `VORT = ∂v/∂x − ∂u/∂y`, `DIV = ∂u/∂x + ∂v/∂y` (s⁻¹), and — when the state
  has `W` — `MCRESIDUAL = DIV + ∂w/∂z + w·d(ln ρ̄)/dz` (s⁻¹, ρ̄ from
  `prob.reference` at the regular `z`).
- `include_derivatives = true`: the six derivative fields of every analysis
  variable with Springsteel's scalar suffixes (`_x, _xx, _y, _yy, _z, _zz`).
- `output_background`: `<name>_BACKGROUND` (the background coefficients
  `a_g`); `output_increment`: `<name>_INCREMENT` (analysis − background
  coefficients), so `BACKGROUND + INCREMENT = analysis`.
- `QFLAG` (when the problem carries a stage-1 product, `prob.stage1`): the
  stage-1 bitwise quality flag (CF `flag_masks`/`flag_meanings`, see
  `Daisho.QFLAG_MEANINGS`) resampled from the quadrature lattice to the
  regular points by `Daisho.regular_quality_flag` with the rule
  `[variational] quality_flag_map` (`"nearest"`, `"worst"`, `"best"`);
  `Int8`, no fill, on the regular dims. It is stage 1's diagnosis of the
  per-point dual-Doppler geometry, not of the variational analysis: the
  solve can recover winds at points stage 1 flagged (baseline, a single σ
  above threshold) but has nothing to fit where there was no data.
- Masking: when `[variational] mask_quality_flags` is non-empty, every
  regular-grid wind-derived field — the wind-role analysis variables (all
  derivative suffixes), their `_BACKGROUND`/`_INCREMENT`, `VORT`, `DIV`,
  `MCRESIDUAL` — is set to `NaN` where `QFLAG` carries any listed bit.
  Gridded scalars and `<name>_ANALYSIS` of scalar state variables are never
  masked. With the default empty list nothing is masked.
- `output_mish`: dims `x_mish, y_mish, z_mish` (the quadrature coordinates)
  with `<name>_MISH` (`Float64`, the mish analysis `result.analysis`, exact),
  the stage-1 product (`U_STAGE1`, `V_STAGE1`, `USTD`, `VSTD`, `DET`,
  `NGATES`, and the flag itself as `QFLAG_MISH` — the QC'd product blanked
  where the flag `!= 0`, `Float32` with the `[io]` fill, as in
  [`write_wind_synthesis`](@ref)), `NOBS_VR`
  (`Int8`), all on `("time", "x_mish", "y_mish", "z_mish")`, and `RHOBAR` on
  `("z_mish",)` (the ρ̄ the continuity constraint used).

Global attributes: those of [`write_radar_netcdf`](@ref) plus
`variational_background`, `variational_outer_iterations`,
`variational_cg_iterations` (total CG iterations), `variational_converged`
(`1`/`0`), `variational_quality_flag_map` (the resampling rule) and
`variational_mask_quality_flags` (the masked flag meanings, space-separated;
empty when none).

Mutates `prob.state_grid.spectral` (it holds the analysis coefficients on
return); call from a single thread.
"""
function write_variational_netcdf(output_file::AbstractString, sgrid::RRR_Grid,
        prob::VariationalProblem, result::VariationalResult, volume::Volume,
        p::DaishoParameters; index_time = nothing, ref_lat::Real, ref_lon::Real,
        institution::String = "", source::String = "", heading::Real = -9999.0,
        include_derivatives::Bool = false)
    state = prob.state
    names = variable_names(state)
    iu, iv, iw = wind_indices(state)
    wind_vars = iw === nothing ? (iu, iv) : (iu, iv, iw)
    tsec = _nc_time_seconds(index_time)

    # 1. Scalars on the regular grid (creates the file).
    write_radar_netcdf(String(output_file), sgrid, volume, field_index_dict(p);
        ref_lat = ref_lat, ref_lon = ref_lon, institution = institution,
        source = source, heading = heading, include_derivatives = include_derivatives,
        time = tsec)

    # 2. The analysis on the same regular points.
    gp = sgrid.params
    x = collect(LinRange(gp.iMin, gp.iMax, gp.i_regular_out))
    y = collect(LinRange(gp.jMin, gp.jMax, gp.j_regular_out))
    z = collect(LinRange(gp.kMin, gp.kMax, gp.k_regular_out))
    nx, ny, nz = length(x), length(y), length(z)
    sg = prob.state_grid
    col(name) = sg.params.vars[name]

    a_bg = prob.cf.a_g
    reg_bg = p.variational.output_background ?
        _state_regular(sg, a_bg, names, x, y, z) : nothing
    reg_inc = p.variational.output_increment ?
        _state_regular(sg, result.coefficients .- a_bg, names, x, y, z) : nothing
    reg = _state_regular(sg, result.coefficients, names, x, y, z)   # leaves the analysis loaded

    fidx = field_index_dict(p)
    outname(v) = haskey(fidx, names[v]) ? names[v] * "_ANALYSIS" : names[v]

    # Stage-1 quality flag resampled to the regular points, and the optional
    # wind mask it drives (computed before any wind field is written).
    qmap = p.variational.quality_flag_map
    qreg = nothing
    if prob.stage1 !== nothing
        gz = get_springsteel_gridpoints_zyx(sgrid)      # (kDim, jDim, iDim, 3): [z, y, x]
        half = (sgrid.ibasis.data[1, 1, 1].params.DX, sgrid.jbasis.data[1, 1, 1].params.DX,
                sgrid.kbasis.data[1, 1, 1].params.DX) ./ 2
        qreg = regular_quality_flag(prob.stage1.quality_flag,
            (gz[1, 1, :, 3], gz[1, :, 1, 2], gz[:, 1, 1, 1]), (x, y, z), half, Symbol(qmap))
    end
    mbits = qflag_bits(p.variational.mask_quality_flags)
    mask = (qreg === nothing || mbits == 0) ? nothing : (qreg .& mbits) .!= 0
    _masked(data) = mask === nothing ? data : (d = copy(data); d[mask] .= NaN; d)
    _maybe_masked(v, data) = v in wind_vars ? _masked(data) : data

    NCDataset(output_file, "a") do ds
        has_time = haskey(ds.dim, "time")
        dims = has_time ? ("time", "x", "y", "z") : ("x", "y", "z")

        function _attrs(name, base)
            a = OrderedDict{String,Any}()
            for (k, v) in base; a[k] = v; end
            a["grid_mapping"] = "grid_mapping"
            return a
        end
        function _var_attrs(v, suffix)
            if v in wind_vars
                sn, ln = _cf_component_attrs(names[v])
                base = OrderedDict{String,Any}("long_name" => ln, "units" => "m s-1")
                sn === nothing || (base["standard_name"] = sn)
            else
                base = OrderedDict{String,Any}("long_name" => names[v] * " analysis",
                                               "units" => state.vars[v].units)
            end
            suffix == "" || (base["long_name"] *= " ($(suffix))")
            return _attrs(names[v], base)
        end
        function _defreg!(name, data, attrs)
            vv = defVar(ds, name, Float64, dims; deflatelevel = 4, fillvalue = NaN,
                        attrib = attrs)
            _put_xyz!(vv, dims, data)
        end

        nslots = include_derivatives ? 7 : 1
        for v in 1:nvars(state), slot in 1:nslots
            sfx = _VAR_REGULAR_SUFFIXES[slot]
            _defreg!(outname(v) * sfx,
                     _maybe_masked(v, _regular_xyz(reg[:, col(names[v]), slot], nx, ny, nz)),
                     _var_attrs(v, sfx))
        end
        for (tag, R) in (("_BACKGROUND", reg_bg), ("_INCREMENT", reg_inc))
            R === nothing && continue
            for v in 1:nvars(state)
                a = _var_attrs(v, "")
                a["long_name"] *= tag == "_BACKGROUND" ? " background" : " analysis increment"
                _defreg!(names[v] * tag, _maybe_masked(v, _regular_xyz(R[:, col(names[v]), 1], nx, ny, nz)), a)
            end
        end

        cu, cv = col(names[iu]), col(names[iv])
        div = reg[:, cu, 2] .+ reg[:, cv, 4]
        vort = reg[:, cv, 2] .- reg[:, cu, 4]
        _defreg!("VORT", _masked(_regular_xyz(vort, nx, ny, nz)),
                 _attrs("VORT", variable_attrib_dict["VORT"]))
        _defreg!("DIV", _masked(_regular_xyz(div, nx, ny, nz)),
                 _attrs("DIV", variable_attrib_dict["DIV"]))
        if iw !== nothing
            cw = col(names[iw])
            dl = [Float64(dlnrho_dz(prob.reference, zz)) for zz in z]
            mc = _regular_xyz(div .+ reg[:, cw, 6], nx, ny, nz) .+
                 _regular_xyz(reg[:, cw, 1], nx, ny, nz) .* reshape(dl, 1, 1, :)
            _defreg!("MCRESIDUAL", _masked(mc),
                     _attrs("MCRESIDUAL", variable_attrib_dict["MCRESIDUAL"]))
        end
        if qreg !== nothing
            qv = defVar(ds, "QFLAG", Int8, dims; attrib = _qflag_attrs(
                "grid_mapping" => "grid_mapping",
                "comment" => "stage-1 quality flag resampled from the quadrature " *
                             "lattice to the output grid: $(qmap)"))
            _put_xyz!(qv, dims, qreg)
        end

        # 3. Mish-lattice diagnostics.
        if p.variational.output_mish
            _write_variational_mish!(ds, sgrid, prob, result, has_time)
        end

        # 4. Solve summary.
        ds.attrib["variational_background"] = p.variational.background
        ds.attrib["variational_outer_iterations"] = Int32(result.stats.outer_iterations)
        ds.attrib["variational_cg_iterations"] = Int32(sum(result.stats.inner_iterations; init = 0))
        ds.attrib["variational_converged"] = Int32(result.converged)
        ds.attrib["variational_quality_flag_map"] = qmap
        ds.attrib["variational_mask_quality_flags"] = join(p.variational.mask_quality_flags, " ")
    end
    return output_file
end

function _write_variational_mish!(ds, sgrid, prob::VariationalProblem,
                                  result::VariationalResult, has_time::Bool)
    state = prob.state
    nzm, nym, nxm = state.dims
    gz = get_springsteel_gridpoints_zyx(sgrid)          # (kDim, jDim, iDim, 3): [z, y, x]
    defDim(ds, "x_mish", nxm)
    defDim(ds, "y_mish", nym)
    defDim(ds, "z_mish", nzm)
    for (name, vals, long) in (("x_mish", gz[1, 1, :, 3], "easting of the quadrature (mish) points"),
                               ("y_mish", gz[1, :, 1, 2], "northing of the quadrature (mish) points"),
                               ("z_mish", gz[:, 1, 1, 1], "altitude of the quadrature (mish) points"))
        defVar(ds, name, Float64, (name,); attrib = OrderedDict{String,Any}(
            "units" => "m", "long_name" => long))[:] = collect(Float64, vals)
    end
    dims = has_time ? ("time", "x_mish", "y_mish", "z_mish") : ("x_mish", "y_mish", "z_mish")
    _xyz(A) = permutedims(A, (3, 2, 1))

    iu, iv, iw = wind_indices(state)
    for (v, name) in enumerate(variable_names(state))
        attrs = OrderedDict{String,Any}()
        if v in (iu, iv, iw)
            sn, ln = _cf_component_attrs(name)
            attrs["long_name"] = ln * " (analysis at the quadrature points)"
            sn === nothing || (attrs["standard_name"] = sn)
            attrs["units"] = "m s-1"
        else
            attrs["long_name"] = "$(name) analysis at the quadrature points"
            attrs["units"] = state.vars[v].units
        end
        attrs["grid_mapping"] = "grid_mapping"
        vv = defVar(ds, name * "_MISH", Float64, dims; deflatelevel = 4, fillvalue = NaN,
                    attrib = attrs)
        _put_xyz!(vv, dims, _xyz(state_view(result.analysis, state, v)))
    end

    if prob.stage1 !== nothing
        _write_wind_data_vars!(ds, _quality_masked(prob.stage1); dims = dims,
                               component_suffix = "_STAGE1", coordinates = nothing,
                               qflag_name = "QFLAG_MISH")
    end
    if result.nobs_vr !== nothing
        attrs = OrderedDict{String,Any}()
        for (k, v) in variable_attrib_dict["NOBS_VR"]; attrs[k] = v; end
        attrs["grid_mapping"] = "grid_mapping"
        _put_xyz!(defVar(ds, "NOBS_VR", Int8, dims; attrib = attrs), dims, _xyz(result.nobs_vr))
    end
    rho = OrderedDict{String,Any}()
    for (k, v) in variable_attrib_dict["RHOBAR"]; rho[k] = v; end
    defVar(ds, "RHOBAR", Float64, ("z_mish",); attrib = rho)[:] =
        [Float64(reference_density(prob.reference, zz)) for zz in state.z_mish]
    return ds
end
