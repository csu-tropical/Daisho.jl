# The variational state vector: configurable control-variable composition
# (winds plus optional `[fields]`-derived scalars), role resolution, per-axis
# boundary conditions, and the state-variable Springsteel grid derived from
# the obs/scalar grid's geometry.
#
# Layout (plan §3.3): mish flat index per variable is column-major
# `(kDim, jDim, iDim)` (k fastest), matching the accumulator plane order and
# `daisho_to_springsteel_index`. The state vector itself is variable-major:
# variable `v` occupies `offsets[v]+1 : offsets[v]+n_mish`.

"""
    STATE_ROLE_VOCAB

`Symbol` form of [`VARIATIONAL_ROLE_VOCAB`](@ref) (`:wind_u`, `:wind_v`,
`:wind_w`, `:reflectivity`, `:scalar`), derived from it so the two vocabularies
cannot drift apart.
"""
const STATE_ROLE_VOCAB = Symbol.(VARIATIONAL_ROLE_VOCAB)

"""
    STATE_SINGULAR_ROLES

Roles that at most one state variable may carry: `:wind_u`, `:wind_v`,
`:wind_w`, `:reflectivity`. `:scalar` is not singular — any number of
`[fields]`-derived state variables may carry it.
"""
const STATE_SINGULAR_ROLES = (:wind_u, :wind_v, :wind_w, :reflectivity)

# ── State transforms ─────────────────────────────────────────────────────────

"""
    AbstractStateTransform

Hook between a state variable's control-space value `c` and its physical
value, evaluated pointwise at mish coordinate `z` (metres). Observation
operators always see the physical value ([`to_physical`](@ref)); the analytic
Jacobian [`d_physical_d_control`](@ref) lets a nonlinear transform's
observation operator linearize without finite differences.

Only [`IdentityTransform`](@ref) ships in this phase. Later additions
(density-weighted momentum, log-transformed scalars) plug in here without
touching `StateVector` or the operator layer.
"""
abstract type AbstractStateTransform end

"""
    IdentityTransform()

`to_physical(c, z) = c`. The only transform in this phase; state winds are
physical `u, v, w` (plan §1: "State winds are physical `u, v, w` (not
ρ-weighted)").
"""
struct IdentityTransform <: AbstractStateTransform end

"""
    to_physical(t::AbstractStateTransform, c, z) -> Real
    to_control(t::AbstractStateTransform, u, z) -> Real

Map a control-space value `c` to its physical value (and back) at mish
height `z` (metres). [`IdentityTransform`](@ref): both are the identity.
"""
to_physical(::IdentityTransform, c::Real, z::Real) = c
to_control(::IdentityTransform, u::Real, z::Real) = u

"""
    d_physical_d_control(t::AbstractStateTransform, c, z) -> Real

The pointwise Jacobian `∂(to_physical)/∂c` at control value `c` and height
`z`; `1` for [`IdentityTransform`](@ref).
"""
d_physical_d_control(::IdentityTransform, c::Real, z::Real) = 1.0

# ── Background-error specifications ──────────────────────────────────────────

"""
    BackgroundErrorSpec

The diagonal scaling `D` of the control-variable chain `C = S∘D∘F` for one
state variable: either a single scalar σ_b ([`UniformSigma`](@ref)) or a
per-mish-point map ([`FieldSigma`](@ref)). See [`sigma_field`](@ref),
[`sigma_vector`](@ref).
"""
abstract type BackgroundErrorSpec end

"""
    UniformSigma(sigma)

A single background-error standard deviation applied at every mish point.
`sigma` must be positive.
"""
struct UniformSigma <: BackgroundErrorSpec
    sigma::Float64
    function UniformSigma(sigma::Real)
        isfinite(sigma) && sigma > 0.0 || throw(ArgumentError(
            "UniformSigma: sigma must be finite and positive (got $(sigma))."))
        return new(Float64(sigma))
    end
end

"""
    FieldSigma(sigma::Array{Float64,3})

A per-mish-point background-error standard deviation map, shape
`(nz, ny, nx)` (matching [`StateVector`](@ref)'s `dims`). Every entry must be
finite and positive. Populated from a future Background product (Phase 9);
not produced by [`StateVector`](@ref) itself in this phase.
"""
struct FieldSigma <: BackgroundErrorSpec
    sigma::Array{Float64,3}
    function FieldSigma(sigma::Array{Float64,3})
        all(x -> isfinite(x) && x > 0.0, sigma) || throw(ArgumentError(
            "FieldSigma: every entry must be finite and positive."))
        return new(sigma)
    end
end

"""
    sigma_field(s::BackgroundErrorSpec, dims::NTuple{3,Int}) -> Array{Float64,3}

The background-error standard deviation at every mish point, shaped `dims`
(`(nz, ny, nx)`): [`UniformSigma`](@ref) broadcasts its scalar,
[`FieldSigma`](@ref) is returned as-is after a size check.
"""
sigma_field(s::UniformSigma, dims::NTuple{3,Int}) = fill(s.sigma, dims)
function sigma_field(s::FieldSigma, dims::NTuple{3,Int})
    size(s.sigma) == dims || throw(DimensionMismatch(
        "sigma_field: FieldSigma size $(size(s.sigma)) != state dims $(dims)"))
    return s.sigma
end

"""
    sigma_vector(s::BackgroundErrorSpec, dims::NTuple{3,Int}) -> Vector{Float64}

[`sigma_field`](@ref) flattened in the k-fastest mish order (`vec` of a
column-major `(nz, ny, nx)` array), matching [`state_view`](@ref)'s layout.
"""
sigma_vector(s::BackgroundErrorSpec, dims::NTuple{3,Int}) = vec(sigma_field(s, dims))

# ── StateVariable ─────────────────────────────────────────────────────────────

"""
    StateVariable

One control-variable entry: its role, units, provenance, background-error
specification, mish-space filter length, resolved per-axis boundary
conditions, and (for a `[fields]`-derived scalar) its value-observation
parameters. Built by [`StateVector`](@ref); not constructed directly except
in tests.

# Fields
- `name::String`: state-vector name (`"U"`/`"V"`/`"W"` or a `[fields]` name).
- `role::Symbol`: one of [`STATE_ROLE_VOCAB`](@ref).
- `units::String`: `"m s-1"` for wind roles, `"dBZ"` for `:reflectivity`,
  `"1"` for `:scalar`.
- `source::Symbol`: `:retrieved` (U/V/W, not observed directly) or
  `:gridded_field` (a `[fields]` name, whose gridded mish value is a direct
  value observation).
- `background_error::BackgroundErrorSpec`: the diagonal scaling `D`.
- `filter_length_m::NTuple{3,Float64}`: mish-space filter length (metres)
  along `(i, j, k)`, from [`BackgroundErrorEntry`](@ref).
- `bc_min`/`bc_max::NTuple{3,Springsteel.BoundaryConditions}`: resolved
  boundary conditions per axis `(i, j, k)`, from
  `[grid.springsteel.<axis>.bc]`.
- `transform::AbstractStateTransform`: control ↔ physical hook (default
  [`IdentityTransform`](@ref)).
- `obs_sigma::Float64`: `:gridded_field` only; `NaN` otherwise.
- `undetect_value::Float64`: `:gridded_field` only; `NaN` otherwise.
"""
Base.@kwdef struct StateVariable
    name::String
    role::Symbol
    units::String
    source::Symbol
    background_error::BackgroundErrorSpec
    filter_length_m::NTuple{3,Float64}
    bc_min::NTuple{3,BoundaryConditions}
    bc_max::NTuple{3,BoundaryConditions}
    transform::AbstractStateTransform = IdentityTransform()
    obs_sigma::Float64 = NaN
    undetect_value::Float64 = NaN
end

# Refuse a boundary condition whose data live in Springsteel's `ahat`
# background coefficient vector (rank-3 inhomogeneous, R3X — the only ahat
# form reachable from a `BoundaryConditions` object; R1T1X is set only at
# runtime via `set_ahat_neumann!`, never by `_bc_to_spline_dict`). The
# variational state's increment is homogeneous by construction (plan §3.5),
# so this data would be silently unusable. `AxisSpline` re-checks this on the
# actual Springsteel spline (`_has_ahat`) when the state grid is built; this
# is the earlier, name-bearing check.
function _reject_ahat_bc(bc::BoundaryConditions, ctx::String)
    if bc_rank(bc) == 3 && is_inhomogeneous(bc)
        throw(ArgumentError(
            "$(ctx): inhomogeneous rank-3 boundary condition carries data in " *
            "Springsteel's `ahat` background vector (R3X); unsupported for a " *
            "variational state variable, whose increment is homogeneous."))
    end
    return bc
end

# Resolve state variable `name`'s role (plan §3.5 priority order): explicit
# `[variational.roles]` override, then the output frame's component names,
# then "W", then the `reflectivity`-tagged `[fields]` entry, then any other
# `[fields]` name, else `ArgumentError`.
function _resolve_role(p::DaishoParameters, name::String, frame::SynthesisFrame,
                       fidx::Dict{String,Int})
    haskey(p.variational.roles, name) && return p.variational.roles[name]
    cn = component_names(frame)
    name == cn[1] && return :wind_u
    name == cn[2] && return :wind_v
    name == "W" && return :wind_w
    refl = field_with_tag_or(p, :reflectivity, "")
    !isempty(refl) && name == refl && return :reflectivity
    haskey(fidx, name) && return :scalar
    throw(ArgumentError(
        "[variational] state entry `$(name)` does not resolve to a role: it " *
        "is not a `[variational.roles]` override, not the output frame's " *
        "component names ($(cn[1]), $(cn[2])), not \"W\", not the " *
        "`reflectivity`-tagged `[fields]` entry ($(isempty(refl) ? "none" : refl)), " *
        "and not a configured `[fields]` name ($(join(sort(collect(keys(fidx))), ", ")))."))
end

const _AXIS_NAMES = ("i", "j", "k")

# `(bc_min, bc_max)` for `name` on one axis: its own per-variable override in
# `axis.bc_min`/`axis.bc_max`, else the axis' "default" (always present — see
# `_springsteel_bc_from_dict`).
function _resolve_axis_bc(axis::SpringsteelAxisConfig, name::String, axname::String)
    bc_min = _reject_ahat_bc(get(axis.bc_min, name, axis.bc_min["default"]),
        "[grid.springsteel.$(axname).bc]: state variable `$(name)` min side")
    bc_max = _reject_ahat_bc(get(axis.bc_max, name, axis.bc_max["default"]),
        "[grid.springsteel.$(axname).bc]: state variable `$(name)` max side")
    return bc_min, bc_max
end

# ── StateVector ────────────────────────────────────────────────────────────────

"""
    StateVector

The configured control-variable composition: an ordered list of
[`StateVariable`](@ref)s and the mish geometry they share.

# Fields
- `vars::Vector{StateVariable}`: state variables in `[variational] state`
  order.
- `dims::NTuple{3,Int}`: mish dimensions `(nz, ny, nx)` = `(kDim, jDim,
  iDim)`.
- `n_mish::Int`: `prod(dims)`.
- `offsets::Vector{Int}`: variable `v` occupies
  `offsets[v]+1 : offsets[v]+n_mish` of the flat state vector.
- `index::Dict{String,Int}`: state-variable name → `v`.
- `roles::Dict{Symbol,Int}`: singular role ([`STATE_SINGULAR_ROLES`](@ref)) →
  `v`.
- `z_mish::Vector{Float64}`: vertical mish coordinates (length `dims[1]`),
  ascending, shared by every `(i, j)` column of the grid.

See also: [`state_view`](@ref), [`wind_indices`](@ref), [`has_role`](@ref),
[`role_index`](@ref), [`filter_lengths_m`](@ref),
[`background_sigma_vector`](@ref), [`create_state_grid`](@ref).
"""
struct StateVector
    vars::Vector{StateVariable}
    dims::NTuple{3,Int}
    n_mish::Int
    offsets::Vector{Int}
    index::Dict{String,Int}
    roles::Dict{Symbol,Int}
    z_mish::Vector{Float64}
end

"""
    StateVector(p::DaishoParameters, sgrid::RRR_Grid; frame::SynthesisFrame = CartesianFrame())

Build the [`StateVector`](@ref) named by `p.variational.state`, on the mish
geometry of `sgrid` (the obs/scalar grid built by [`create_radar_grid`](@ref)
— `sgrid` need only share `p`'s `[grid.springsteel]` geometry; it need not
carry the state variables). See the module docstring for role resolution,
boundary-condition resolution, and layout.
"""
function StateVector(p::DaishoParameters, sgrid::RRR_Grid;
                     frame::SynthesisFrame = CartesianFrame())
    names = p.variational.state
    cfg = p.grid.springsteel
    fidx = field_index_dict(p)
    gp = sgrid.params
    dims = (gp.kDim, gp.jDim, gp.iDim)
    n_mish = dims[1] * dims[2] * dims[3]

    z_mish = Vector{Float64}(sgrid.kbasis.data[1, 1, 1].mishPoints)
    length(z_mish) == dims[1] || throw(DimensionMismatch(
        "StateVector: grid k mish length $(length(z_mish)) != kDim $(dims[1])"))
    issorted(z_mish) || throw(ArgumentError(
        "StateVector: grid k mish points are not ascending"))

    vars = Vector{StateVariable}(undef, length(names))
    index = Dict{String,Int}()
    roles = Dict{Symbol,Int}()
    for (v, name) in enumerate(names)
        haskey(index, name) && throw(ArgumentError(
            "[variational] state: duplicate entry `$(name)`."))
        index[name] = v

        role = _resolve_role(p, name, frame, fidx)
        if role in STATE_SINGULAR_ROLES
            haskey(roles, role) && throw(ArgumentError(
                "[variational] state: role `$(role)` is assigned to both " *
                "`$(names[roles[role]])` and `$(name)`; each of " *
                "$(STATE_SINGULAR_ROLES) may be assigned to at most one " *
                "state variable."))
            roles[role] = v
        end

        is_field = haskey(fidx, name)
        source = is_field ? :gridded_field : :retrieved
        units = role === :reflectivity ? "dBZ" : (role === :scalar ? "1" : "m s-1")

        obs_sigma = NaN
        undetect_value = NaN
        if is_field
            entry = p.variational.scalar_obs[name]   # Phase 6 validated presence
            obs_sigma = entry.sigma
            undetect_value = entry.undetect_value
        end

        be = background_error_for(p.variational, name)

        bc_min_i, bc_max_i = _resolve_axis_bc(cfg.i, name, "i")
        bc_min_j, bc_max_j = _resolve_axis_bc(cfg.j, name, "j")
        bc_min_k, bc_max_k = _resolve_axis_bc(cfg.k, name, "k")

        vars[v] = StateVariable(; name = name, role = role, units = units,
            source = source, background_error = UniformSigma(be.sigma),
            filter_length_m = be.filter_length_m,
            bc_min = (bc_min_i, bc_min_j, bc_min_k),
            bc_max = (bc_max_i, bc_max_j, bc_max_k),
            obs_sigma = obs_sigma, undetect_value = undetect_value)
    end

    haskey(roles, :wind_u) || throw(ArgumentError(
        "[variational] state: no state variable resolves to role `wind_u`."))
    haskey(roles, :wind_v) || throw(ArgumentError(
        "[variational] state: no state variable resolves to role `wind_v`."))
    p.variational.mass_continuity && !haskey(roles, :wind_w) && throw(ArgumentError(
        "[variational] mass_continuity = true requires a state variable with " *
        "role `wind_w`."))

    offsets = [(v - 1) * n_mish for v in 1:length(vars)]
    return StateVector(vars, dims, n_mish, offsets, index, roles, z_mish)
end

_state_range(s::StateVector, v::Int) = (s.offsets[v] + 1):(s.offsets[v] + s.n_mish)

"""
    Base.length(s::StateVector) -> Int

Length of the flat state vector: `nvars(s) * s.n_mish`.
"""
Base.length(s::StateVector) = s.n_mish * length(s.vars)

"""
    nvars(s::StateVector) -> Int

Number of state variables.
"""
nvars(s::StateVector) = length(s.vars)

"""
    variable_names(s::StateVector) -> Vector{String}

State-variable names, in state order.
"""
variable_names(s::StateVector) = [v.name for v in s.vars]

"""
    state_index_dict(s::StateVector) -> Dict{String,Int}

Name → 1-based index map, in the exact form Springsteel's `vars` grid
parameter expects (see [`create_state_grid`](@ref)).
"""
state_index_dict(s::StateVector) = Dict{String,Int}(s.index)

"""
    has_role(s::StateVector, role::Symbol) -> Bool

Whether some state variable carries singular role `role`.
"""
has_role(s::StateVector, role::Symbol) = haskey(s.roles, role)

"""
    role_index(s::StateVector, role::Symbol) -> Int

The 1-based state-variable index carrying singular role `role`, or
`ArgumentError` if none does.
"""
function role_index(s::StateVector, role::Symbol)
    haskey(s.roles, role) || throw(ArgumentError(
        "StateVector: no state variable has role `$(role)`."))
    return s.roles[role]
end

"""
    wind_indices(s::StateVector) -> (iu::Int, iv::Int, iw::Union{Int,Nothing})

The state-variable indices of the `:wind_u`, `:wind_v`, and (if present)
`:wind_w` roles; `iw` is `nothing` when no state variable carries `:wind_w`.
"""
wind_indices(s::StateVector) = (role_index(s, :wind_u), role_index(s, :wind_v),
                                has_role(s, :wind_w) ? role_index(s, :wind_w) : nothing)

"""
    state_view(x::AbstractVector, s::StateVector, v::Int)
    state_view(x::AbstractVector, s::StateVector, name::String)
    state_view(x::AbstractVector, s::StateVector, role::Symbol)

A `(nz, ny, nx)` (`s.dims`) reshaped view of state variable `v` (by index,
name, or singular role) within flat state vector `x` (`length(x) ==
length(s)`). `x[state's flat range][k + (j-1)*nz + (i-1)*nz*ny]` ==
`state_view(x, s, v)[k, j, i]` — the same k-fastest order as
`daisho_to_springsteel_index`.
"""
function state_view(x::AbstractVector, s::StateVector, v::Int)
    return reshape(view(x, _state_range(s, v)), s.dims)
end
state_view(x::AbstractVector, s::StateVector, name::String) =
    state_view(x, s, s.index[name])
state_view(x::AbstractVector, s::StateVector, role::Symbol) =
    state_view(x, s, role_index(s, role))

"""
    filter_lengths_m(s::StateVector) -> Matrix{Float64}

Mish-space filter lengths (metres), `3 × nvars(s)`, in the **`(k, j, i)`**
axis order used by [`MishFilter`](@ref)/`_SplineGridGeometry` (axis 1 = k,
axis 2 = j, axis 3 = i) — the transpose of `StateVariable.filter_length_m`'s
`(i, j, k)` config order (row 1 of the result is variable `v`'s k-axis
length, i.e. `StateVariable.filter_length_m[3]`, and so on).
"""
function filter_lengths_m(s::StateVector)
    out = Matrix{Float64}(undef, 3, nvars(s))
    for (v, var) in enumerate(s.vars)
        fi, fj, fk = var.filter_length_m
        out[1, v] = fk
        out[2, v] = fj
        out[3, v] = fi
    end
    return out
end

"""
    background_sigma_vector(s::StateVector) -> Vector{Float64}

The diagonal `D` of the control-variable chain as a flat, variable-major
vector of length `length(s)`: each state variable's
`sigma_vector(var.background_error, s.dims)` in its own block.
"""
function background_sigma_vector(s::StateVector)
    out = Vector{Float64}(undef, length(s))
    for (v, var) in enumerate(s.vars)
        out[_state_range(s, v)] = sigma_vector(var.background_error, s.dims)
    end
    return out
end

# ── State grid ───────────────────────────────────────────────────────────────

"""
    create_state_grid(p::DaishoParameters, sgrid::RRR_Grid, state::StateVector) -> RRR_Grid

Build the **state grid**: a Springsteel grid with the same geometry as
`sgrid` (bounds, cell counts, mubar, quadrature, `regular_out`) but variables
`state_index_dict(state)` and each state variable's own resolved boundary
conditions (plan §3.3). Implemented as
`create_radar_grid(p.grid.springsteel, state_index_dict(state))`, which
independently reproduces `sgrid`'s geometry from the same
`[grid.springsteel]` config and fills `BCL/BCR/BCD/BCU/BCB/BCT` from the same
per-variable `bc_min`/`bc_max` dicts `state`'s boundary conditions were
resolved from — so the two are consistent by construction; `allowed` widens
to every configured `[fields]` name (not just `state`'s), so a BC override
meant for the obs/scalar grid's own fields does not spuriously fail state-grid
construction. `spline_filter` is left empty (exports are not re-filtered).

Throws if the resulting grid's geometry disagrees with `sgrid` (a
same-`[grid.springsteel]` precondition violation) or carries a
`spline_filter`.
"""
function create_state_grid(p::DaishoParameters, sgrid::RRR_Grid, state::StateVector)
    cfg = p.grid.springsteel
    vars = state_index_dict(state)
    allowed = union(Set{String}(keys(field_index_dict(p))),
                    Set{String}(variable_names(state)))
    state_grid = create_radar_grid(cfg, vars; allowed = allowed)

    gp, sp = state_grid.params, sgrid.params
    (gp.iDim, gp.jDim, gp.kDim) == (sp.iDim, sp.jDim, sp.kDim) || throw(DimensionMismatch(
        "create_state_grid: state grid physical dims $((gp.iDim, gp.jDim, gp.kDim)) " *
        "!= obs grid dims $((sp.iDim, sp.jDim, sp.kDim)); `sgrid` and `p.grid.springsteel` " *
        "must share the same geometry."))
    (gp.b_iDim, gp.b_jDim, gp.b_kDim) == (sp.b_iDim, sp.b_jDim, sp.b_kDim) || throw(DimensionMismatch(
        "create_state_grid: state grid coefficient dims $((gp.b_iDim, gp.b_jDim, gp.b_kDim)) " *
        "!= obs grid dims $((sp.b_iDim, sp.b_jDim, sp.b_kDim))"))
    isempty(gp.spline_filter) || throw(ArgumentError(
        "create_state_grid: state grid must carry no spline_filter (got keys " *
        "$(collect(keys(gp.spline_filter))); exports must not be re-filtered)."))
    z_state = Vector{Float64}(state_grid.kbasis.data[1, 1, 1].mishPoints)
    z_state == state.z_mish || throw(ArgumentError(
        "create_state_grid: state grid k mish points disagree with " *
        "StateVector.z_mish."))

    return state_grid
end
