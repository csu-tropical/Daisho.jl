# Observation builders (plan §2, §3.6): radial-velocity super-obs from the
# stage-1 per-point normal systems, anelastic mass-continuity pseudo-obs, and
# value obs of `[fields]`-derived state variables.

# Unpack the packed symmetric N×N normal matrix at (k, j, i).
function _unpack_normal(acc::WindGridAccumulator, k::Int, j::Int, i::Int)
    N = acc.n_unknowns
    S = zeros(N, N)
    for a in 1:N, b in a:N
        v = acc.AtWA[_packed_index(a, b, N), k, j, i]
        S[a, b] = v
        S[b, a] = v
    end
    return S, acc.AtWb[:, k, j, i]
end

"""
    radial_velocity_superobs(acc::WindGridAccumulator, state::StateVector, p::DaishoParameters)
        -> (group::ObsGroup{RadialVelocitySuperObs}, nobs_vr::Array{Int8,3})

Turn the stage-1 per-point normal systems into super-observations (plan §2).
The accumulator planes must lie on the state mish (`size(acc.AtWA)[2:4] ==
state.dims`). At each point with gates and positive total weight, with
`S = Σ w g gᵀ` and `b = Σ w ṽ g`:

- the system is the full `(u, v, w)` 3×3 (requires a `:wind_w` state
  variable and `acc.n_unknowns == 3`, i.e. `[synthesis]
  include_vertical_velocity = true`), or its `(u, v)` 2×2 block when
  `[variational.radial_velocity] horizontal_approximation = true` (N = 2 or
  3). A 3×3 whose `S[3,3] == 0` exactly (every gate at zero elevation, so w
  is structurally unobserved) is reduced to the 2×2 block;
- `S = Σ_r λ_r e_r e_rᵀ`; each `r` with `λ_r > 0` and
  `λ_r > eigenvalue_guard_relative · λ_max` emits one
  [`RadialVelocitySuperObs`](@ref) with `e = e_r` (zero-padded to 3),
  value `y_r = e_rᵀb / λ_r` and `σ_r = √(σ²_vr · error_inflation / λ_r)`
  (inverse variance `λ_r / (σ²_vr · error_inflation)`, `σ²_vr =
  [synthesis] velocity_variance`).

Then `Σ_r r_r (e_rᵀx − y_r)² = Σ_m w_m (ṽ_m − g_mᵀx)² / σ²_vr + const` (up to
the guard-dropped directions): the rows reproduce the gate cost and its
gradient `2(Sx − b)/σ²_vr` exactly. Eigenvector signs are arbitrary; `y_r`
flips with `e_r`, so each row (and its cost) is sign-invariant.

`nobs_vr[k, j, i] ∈ 0:3` counts the rows per point (a diagnostic).
"""
function radial_velocity_superobs(acc::WindGridAccumulator, state::StateVector, p::DaishoParameters)
    size(acc.AtWA)[2:4] == state.dims || throw(DimensionMismatch(
        "radial_velocity_superobs: accumulator planes $(size(acc.AtWA)[2:4]) != state mish dims " *
        "$(state.dims); the stage-1 accumulator must be built on the state grid's mish"))
    rv = p.variational.radial_velocity
    N = acc.n_unknowns
    horizontal = rv.horizontal_approximation
    if !horizontal
        has_role(state, :wind_w) || throw(ArgumentError(
            "radial_velocity_superobs: the state has no `wind_w` variable, so the (u, v, w) " *
            "super-obs rows cannot be represented; add \"W\" to `[variational] state` or set " *
            "`[variational.radial_velocity] horizontal_approximation = true`"))
        N == 3 || throw(ArgumentError(
            "radial_velocity_superobs: the accumulator holds the 2-unknown (u, v) system but the " *
            "state has W; set `[synthesis] include_vertical_velocity = true` (3-unknown " *
            "accumulation) or `[variational.radial_velocity] horizontal_approximation = true`"))
    end
    σ2 = p.synthesis.velocity_variance * rv.error_inflation
    guard = rv.eigenvalue_guard_relative
    nz, ny, nx = state.dims
    counts = zeros(Int8, nz, ny, nx)
    obs = RadialVelocitySuperObs[]
    for i in 1:nx, j in 1:ny, k in 1:nz
        (acc.gate_count[k, j, i] == 0 || acc.weight_total[k, j, i] <= 0.0) && continue
        S, b = _unpack_normal(acc, k, j, i)
        n = (horizontal || N == 2 || S[3, 3] == 0.0) ? 2 : 3
        if n < N
            S = S[1:n, 1:n]; b = b[1:n]
        end
        F = eigen(Symmetric(S))
        λmax = maximum(F.values)
        λmax > 0.0 || continue
        loc = MishIndex(k, j, i)
        for r in 1:n
            λ = F.values[r]
            (λ > 0.0 && λ > guard * λmax) || continue
            e = view(F.vectors, :, r)
            β = dot(e, b)
            push!(obs, RadialVelocitySuperObs(loc, β / λ, sqrt(σ2 / λ),
                                              (e[1], e[2], n == 3 ? e[3] : 0.0)))
            counts[k, j, i] += Int8(1)
        end
    end
    return ObsGroup(:radial_velocity, obs), counts
end

"""
    build_radial_velocity_superobs(acc, state, p) -> ObsGroup{RadialVelocitySuperObs}

The group of [`radial_velocity_superobs`](@ref).
"""
build_radial_velocity_superobs(acc::WindGridAccumulator, state::StateVector, p::DaishoParameters) =
    radial_velocity_superobs(acc, state, p)[1]

"""
    radial_velocity_superobs_count(acc, state, p) -> Array{Int8,3}

Per-point super-obs row count (`NOBS_VR` diagnostic) of
[`radial_velocity_superobs`](@ref).
"""
radial_velocity_superobs_count(acc::WindGridAccumulator, state::StateVector, p::DaishoParameters) =
    radial_velocity_superobs(acc, state, p)[2]

"""
    build_mass_continuity_obs(state::StateVector, reference::AbstractReferenceProfile,
                              p::DaishoParameters) -> ObsGroup{MassContinuityObs}

One [`MassContinuityObs`](@ref) per mish point with σ =
`[variational] mass_continuity_sigma_per_s` and
`dlnrho_dz(reference, state.z_mish[k])`; an empty group when
`[variational] mass_continuity = false`.
"""
function build_mass_continuity_obs(state::StateVector, reference::AbstractReferenceProfile,
                                   p::DaishoParameters)
    obs = MassContinuityObs[]
    p.variational.mass_continuity || return ObsGroup(:mass_continuity, obs)
    σ = p.variational.mass_continuity_sigma_per_s
    nz, ny, nx = state.dims
    dl = [Float64(dlnrho_dz(reference, z)) for z in state.z_mish]
    sizehint!(obs, state.n_mish)
    for i in 1:nx, j in 1:ny, k in 1:nz
        push!(obs, MassContinuityObs(MishIndex(k, j, i), σ, dl[k]))
    end
    return ObsGroup(:mass_continuity, obs)
end

"""
    build_scalar_value_obs(radar_grid::AbstractArray{<:Real,4}, state::StateVector,
                           p::DaishoParameters) -> ObsGroup{ScalarValueObs}

Value observations of every `:gridded_field` state variable from the gridded
scalar mish array `radar_grid` (`(n_fields, nz, ny, nx)`, columns by
[`field_index_dict`](@ref), as from [`finalize_grid`](@ref)). Per mish
point: `[io] fill_value` or `NaN` (true missing) → no obs; `[io] undetect`
(scanned, no echo) → an obs of the variable's `undetect_value`; otherwise an
obs of the gridded value. σ is the variable's `[variational.scalar_obs]`
sigma. Empty when the state has no `:gridded_field` variables.
"""
function build_scalar_value_obs(radar_grid::AbstractArray{<:Real,4}, state::StateVector,
                                p::DaishoParameters)
    obs = ScalarValueObs[]
    vars = [v for (v, var) in enumerate(state.vars) if var.source === :gridded_field]
    isempty(vars) && return ObsGroup(:scalar_value, obs)
    size(radar_grid)[2:4] == state.dims || throw(DimensionMismatch(
        "build_scalar_value_obs: radar_grid mish dims $(size(radar_grid)[2:4]) != state dims $(state.dims)"))
    fidx = field_index_dict(p)
    fill_value = p.io.fill_value
    undetect = p.io.undetect
    nz, ny, nx = state.dims
    for v in vars
        var = state.vars[v]
        c = fidx[var.name]
        1 <= c <= size(radar_grid, 1) || throw(DimensionMismatch(
            "build_scalar_value_obs: field `$(var.name)` column $(c) outside radar_grid's " *
            "$(size(radar_grid, 1)) fields"))
        for i in 1:nx, j in 1:ny, k in 1:nz
            x = Float64(radar_grid[c, k, j, i])
            (isnan(x) || x == fill_value) && continue
            val = x == undetect ? var.undetect_value : x
            push!(obs, ScalarValueObs(MishIndex(k, j, i), v, val, var.obs_sigma))
        end
    end
    return ObsGroup(:scalar_value, obs)
end

"""
    build_observation_set(acc, radar_grid, state, reference, p;
                          extra::AbstractVector{<:ObsGroup} = ObsGroup[]) -> ObservationSet

The standard observation set, non-empty groups only, in the order:
radial-velocity super-obs from `acc` (skipped when `acc === nothing`),
mass continuity ([`build_mass_continuity_obs`](@ref)), scalar value obs from
`radar_grid` ([`build_scalar_value_obs`](@ref); `radar_grid` may be
`nothing` only when the state has no `:gridded_field` variables), then
`extra`.
"""
function build_observation_set(acc::Union{Nothing,WindGridAccumulator},
                               radar_grid::Union{Nothing,AbstractArray{<:Real,4}},
                               state::StateVector, reference::AbstractReferenceProfile,
                               p::DaishoParameters;
                               extra::AbstractVector{<:ObsGroup} = ObsGroup[])
    set = ObservationSet()
    add!(g) = nobs(g) > 0 && push_group!(set, g)
    acc === nothing || add!(build_radial_velocity_superobs(acc, state, p))
    add!(build_mass_continuity_obs(state, reference, p))
    if radar_grid === nothing
        any(v -> v.source === :gridded_field, state.vars) && throw(ArgumentError(
            "build_observation_set: the state has `[fields]`-derived variables " *
            "($(join([v.name for v in state.vars if v.source === :gridded_field], ", "))) " *
            "but no radar_grid was given"))
    else
        add!(build_scalar_value_obs(radar_grid, state, p))
    end
    for g in extra
        push_group!(set, g)
    end
    return set
end
