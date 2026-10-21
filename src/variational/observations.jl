# Observation types of the variational solver (plan §3.6).
#
# An observation is a value `y`, a standard deviation σ (inverse variance
# `r = 1/σ²`), a location, and an operator written as a list of terms
# `(var, deriv, weight)` on the state variables (`deriv`: 0 = value,
# 1 = ∂x (i), 2 = ∂y (j), 3 = ∂z (k)). Concrete types are dispatched on,
# replacing samurai's switch statements; homogeneous groups of them are
# converted to the Phase-4 backends:
#
#   MishIndex locations → MishObsBackend  (linear, or nonlinear via its model hook)
#   NodeIndex locations → MishObsBackend on the node lattice (linear only)
#   XYZ locations       → ScatteredObsBackend (linear only)
#
# The backend slot of a term is `deriv + 1` (backend slots: 1 = value,
# 2 = ∂x, 3 = ∂y, 4 = ∂z).

# ── Locations ────────────────────────────────────────────────────────────────

"""
    ObsLocation

Supertype of observation locations: [`MishIndex`](@ref) (a mish/quadrature
point of the state grid), [`NodeIndex`](@ref) (a B-spline node of the state
grid) or [`XYZ`](@ref) (an arbitrary point in metres).
"""
abstract type ObsLocation end

"""
    MishIndex(k, j, i)

A mish point of the state's `(nz, ny, nx)` lattice (1-based; `k` vertical,
`j` y, `i` x), i.e. the same index as the stage-1 accumulator planes
`[:, k, j, i]`. See [`mish_flat`](@ref).
"""
struct MishIndex <: ObsLocation
    k::Int
    j::Int
    i::Int
end

"""
    NodeIndex(k, j, i)

A B-spline node of the state grid's `(cells_k+1, cells_j+1, cells_i+1)` node
lattice (1-based; `k` vertical, `j` y, `i` x; node `n` of an axis sits at
`min + (n−1)·DX`, so the lattice is the default regular output grid). See
[`node_flat`](@ref) and `StateVector.node_dims`. Supported for linear
observations only.
"""
struct NodeIndex <: ObsLocation
    k::Int
    j::Int
    i::Int
end

"""
    XYZ(x, y, z)

An arbitrary location (metres, the grid's i, j, k coordinates) inside the
state grid's domain. Supported for linear observations only.
"""
struct XYZ <: ObsLocation
    x::Float64
    y::Float64
    z::Float64
end

"""
    mish_flat(loc::MishIndex, dims::NTuple{3,Int}) -> Int

The k-fastest flat mish index `k + (j−1)·nz + (i−1)·nz·ny` of `loc` on a
`dims = (nz, ny, nx)` lattice — the point index of [`MishObsBackend`](@ref)
and of [`state_view`](@ref). Out-of-range indices throw `ArgumentError`.
"""
function mish_flat(loc::MishIndex, dims::NTuple{3,Int})
    nz, ny, nx = dims
    (1 <= loc.k <= nz && 1 <= loc.j <= ny && 1 <= loc.i <= nx) || throw(ArgumentError(
        "mish_flat: MishIndex($(loc.k), $(loc.j), $(loc.i)) outside the $(dims) (nz, ny, nx) lattice"))
    return loc.k + (loc.j - 1) * nz + (loc.i - 1) * nz * ny
end

"""
    node_flat(loc::NodeIndex, node_dims::NTuple{3,Int}) -> Int

The k-fastest flat index of `loc` on the `node_dims = (nz, ny, nx)` node
lattice (`StateVector.node_dims`) — the point index of a node-lattice
[`MishObsBackend`](@ref). Out-of-range indices throw `ArgumentError`.
"""
function node_flat(loc::NodeIndex, dims::NTuple{3,Int})
    nz, ny, nx = dims
    (1 <= loc.k <= nz && 1 <= loc.j <= ny && 1 <= loc.i <= nx) || throw(ArgumentError(
        "node_flat: NodeIndex($(loc.k), $(loc.j), $(loc.i)) outside the $(dims) (nz, ny, nx) node lattice"))
    return loc.k + (loc.j - 1) * nz + (loc.i - 1) * nz * ny
end

# ── Terms ─────────────────────────────────────────────────────────────────────

"""
    ObsTerm(var, deriv, weight)

One term of an observation operator: `weight × [deriv of state variable
var]` at the observation location. `deriv`: `0` = value, `1` = ∂x (i),
`2` = ∂y (j), `3` = ∂z (k); the backend slot is `deriv + 1`.
"""
struct ObsTerm
    var::Int
    deriv::Int
    weight::Float64
    function ObsTerm(var::Integer, deriv::Integer, weight::Real)
        0 <= deriv <= 3 || throw(ArgumentError(
            "ObsTerm: deriv must be 0 (value), 1 (∂x), 2 (∂y) or 3 (∂z); got $(deriv)"))
        return new(Int(var), Int(deriv), Float64(weight))
    end
end

_backend_term(t::ObsTerm) = (t.var, t.deriv + 1, t.weight)

# ── Interface ────────────────────────────────────────────────────────────────

"""
    AbstractObservation

Supertype of variational observation types. Interface (the first four are
required; the rest have defaults):

- `obs_value(o) -> Float64`: the observed value.
- `obs_sigma(o) -> Float64`: its standard deviation (> 0);
  `obs_inverse_variance(o) = 1/σ²` by default.
- `obs_location(o) -> ObsLocation`: a [`MishIndex`](@ref), [`NodeIndex`](@ref) or [`XYZ`](@ref).
- `obs_terms(o, state::StateVector) -> Vector{ObsTerm}`: the linear operator
  row (for nonlinear types, the initial linearisation).
- `obs_kind(o) -> Symbol`: the group label (default: a per-type constant).
- `is_linear(o)`: `true` by default.
- `obs_slots(o, state)`: the distinct `(var, deriv+1)` of `obs_terms`, in
  term order — the `sample` layout of nonlinear types.
- `model_value(o, sample, state)`: nonlinear model `h(x)` from the `sample`
  (current values of `obs_slots` at the location, in that order); default
  is the linear sum of the terms.
- `linearize(o, sample, state)`: tangent-linear weights, one per
  `obs_terms` entry in that order; default is the term weights.

Observations are grouped into homogeneous [`ObsGroup`](@ref)s, collected in
an [`ObservationSet`](@ref) and converted to backends by
[`build_backends`](@ref).
"""
abstract type AbstractObservation end

"""
    obs_value(o::AbstractObservation) -> Float64

The observed value.
"""
function obs_value end

"""
    obs_sigma(o::AbstractObservation) -> Float64

The observation standard deviation (> 0).
"""
function obs_sigma end

"""
    obs_location(o::AbstractObservation) -> ObsLocation

Where the observation is located.
"""
function obs_location end

"""
    obs_terms(o::AbstractObservation, state::StateVector) -> Vector{ObsTerm}

The linear operator row of `o` on `state` (nonlinear types: the initial
linearisation, used until the first outer-loop linearisation).
"""
function obs_terms end

"""
    obs_inverse_variance(o::AbstractObservation) -> Float64

`1 / obs_sigma(o)^2`.
"""
obs_inverse_variance(o::AbstractObservation) = 1.0 / obs_sigma(o)^2

"""
    obs_kind(o::AbstractObservation) -> Symbol

Diagnostic label of `o` (e.g. `:radial_velocity`); by default a constant of
the type, `obs_kind(typeof(o))`.
"""
obs_kind(o::AbstractObservation) = obs_kind(typeof(o))

is_linear(::AbstractObservation) = true

"""
    obs_slots(o::AbstractObservation, state::StateVector) -> Vector{Tuple{Int,Int}}

Distinct `(var, deriv + 1)` backend slots of `obs_terms(o, state)`, in term
order: the layout of the `sample` passed to [`model_value`](@ref) and
[`linearize`](@ref).
"""
function obs_slots(o::AbstractObservation, state::StateVector)
    out = Tuple{Int,Int}[]
    for t in obs_terms(o, state)
        s = (t.var, t.deriv + 1)
        s in out || push!(out, s)
    end
    return out
end

"""
    model_value(o::AbstractObservation, sample, state::StateVector) -> Float64

The (possibly nonlinear) model equivalent of `o`; `sample[q]` is the current
value of `obs_slots(o, state)[q]` at the observation location. Default: the
linear sum `Σ_t weight_t · sample[slot(t)]`.
"""
function model_value(o::AbstractObservation, sample, state::StateVector)
    slots = obs_slots(o, state)
    s = 0.0
    for t in obs_terms(o, state)
        s += t.weight * sample[findfirst(==((t.var, t.deriv + 1)), slots)]
    end
    return s
end

"""
    linearize(o::AbstractObservation, sample, state::StateVector) -> weights

Tangent-linear weights of `o` at `sample` (see [`model_value`](@ref)), one
per `obs_terms(o, state)` entry, in that order. Default: the term weights.
"""
linearize(o::AbstractObservation, sample, state::StateVector) =
    [t.weight for t in obs_terms(o, state)]

function _check_value_sigma(what, value, sigma)
    isfinite(value) || throw(ArgumentError("$(what): value must be finite (got $(value))"))
    (isfinite(sigma) && sigma > 0) || throw(ArgumentError(
        "$(what): sigma must be finite and > 0 (got $(sigma))"))
    return nothing
end

# ── Concrete types ───────────────────────────────────────────────────────────

"""
    RadialVelocitySuperObs(loc::MishIndex, value, sigma, e::NTuple{3,Float64})

One eigen-row of a stage-1 per-point radial-velocity normal system (plan §2,
built by [`build_radial_velocity_superobs`](@ref)): the obs `eᵀ(u, v, w) =
value` with inverse variance `λ/(σ²_vr·inflation)`. Terms `(u, val, e[1])`,
`(v, val, e[2])`, plus `(w, val, e[3])` only when the state has a `:wind_w`
variable and `e[3] ≠ 0` (a horizontal-approximation or structurally
w-blind row carries `e[3] = 0` exactly and no w term). Kind
`:radial_velocity`.
"""
struct RadialVelocitySuperObs <: AbstractObservation
    loc::MishIndex
    value::Float64
    sigma::Float64
    e::NTuple{3,Float64}
    function RadialVelocitySuperObs(loc::MishIndex, value::Real, sigma::Real, e::NTuple{3,<:Real})
        _check_value_sigma("RadialVelocitySuperObs", value, sigma)
        all(isfinite, e) || throw(ArgumentError("RadialVelocitySuperObs: e must be finite"))
        return new(loc, Float64(value), Float64(sigma), Float64.(e))
    end
end
obs_value(o::RadialVelocitySuperObs) = o.value
obs_sigma(o::RadialVelocitySuperObs) = o.sigma
obs_location(o::RadialVelocitySuperObs) = o.loc
obs_kind(::Type{RadialVelocitySuperObs}) = :radial_velocity
function obs_terms(o::RadialVelocitySuperObs, state::StateVector)
    iu, iv, iw = wind_indices(state)
    t = [ObsTerm(iu, 0, o.e[1]), ObsTerm(iv, 0, o.e[2])]
    if o.e[3] != 0.0
        iw === nothing && throw(ArgumentError(
            "RadialVelocitySuperObs: the row has a w component (e[3] = $(o.e[3])) but the " *
            "state has no `wind_w` variable"))
        push!(t, ObsTerm(iw, 0, o.e[3]))
    end
    return t
end

"""
    MassContinuityObs(loc::Union{MishIndex,NodeIndex}, sigma, dlnrho_dz)

Anelastic mass-continuity pseudo-observation at a mish point or a B-spline node (`loc`): value 0,
operator `∂u/∂x + ∂v/∂y + ∂w/∂z + w·d(ln ρ̄)/dz` (terms `(u,∂x,1)`,
`(v,∂y,1)`, `(w,∂z,1)`, and `(w,val,dlnrho_dz)` unless `dlnrho_dz == 0`).
`sigma` is in s⁻¹ (`[variational] mass_continuity_sigma_per_s`). Requires a
`:wind_w` state variable. Kind `:mass_continuity`.
"""
struct MassContinuityObs <: AbstractObservation
    loc::Union{MishIndex,NodeIndex}
    sigma::Float64
    dlnrho_dz::Float64
    function MassContinuityObs(loc::Union{MishIndex,NodeIndex}, sigma::Real, dlnrho_dz::Real)
        _check_value_sigma("MassContinuityObs", 0.0, sigma)
        isfinite(dlnrho_dz) || throw(ArgumentError("MassContinuityObs: dlnrho_dz must be finite"))
        return new(loc, Float64(sigma), Float64(dlnrho_dz))
    end
end
obs_value(::MassContinuityObs) = 0.0
obs_sigma(o::MassContinuityObs) = o.sigma
obs_location(o::MassContinuityObs) = o.loc
obs_kind(::Type{MassContinuityObs}) = :mass_continuity
function obs_terms(o::MassContinuityObs, state::StateVector)
    iu, iv, iw = wind_indices(state)
    iw === nothing && throw(ArgumentError(
        "MassContinuityObs requires a state variable with role `wind_w`"))
    t = [ObsTerm(iu, 1, 1.0), ObsTerm(iv, 2, 1.0), ObsTerm(iw, 3, 1.0)]
    o.dlnrho_dz == 0.0 || push!(t, ObsTerm(iw, 0, o.dlnrho_dz))
    return t
end

"""
    ScalarValueObs(loc::MishIndex, var::Int, value, sigma)

Direct observation of state variable `var`'s value at a mish point (the
gridded `[fields]` value of a `:gridded_field` state variable; see
[`build_scalar_value_obs`](@ref)). Kind `:scalar_value`.
"""
struct ScalarValueObs <: AbstractObservation
    loc::MishIndex
    var::Int
    value::Float64
    sigma::Float64
    function ScalarValueObs(loc::MishIndex, var::Integer, value::Real, sigma::Real)
        _check_value_sigma("ScalarValueObs", value, sigma)
        var >= 1 || throw(ArgumentError("ScalarValueObs: var must be ≥ 1 (got $(var))"))
        return new(loc, Int(var), Float64(value), Float64(sigma))
    end
end
obs_value(o::ScalarValueObs) = o.value
obs_sigma(o::ScalarValueObs) = o.sigma
obs_location(o::ScalarValueObs) = o.loc
obs_kind(::Type{ScalarValueObs}) = :scalar_value
obs_terms(o::ScalarValueObs, ::StateVector) = [ObsTerm(o.var, 0, 1.0)]

"""
    PointObs(loc::XYZ, value, sigma, terms::Vector{ObsTerm}; kind = :point)

A generic linear observation at an arbitrary point: `Σ_t weight_t ·
[deriv_t of var_t](x, y, z) = value`. Converted to a
[`ScatteredObsBackend`](@ref).
"""
struct PointObs <: AbstractObservation
    loc::XYZ
    value::Float64
    sigma::Float64
    terms::Vector{ObsTerm}
    kind::Symbol
    function PointObs(loc::XYZ, value::Real, sigma::Real, terms::AbstractVector{ObsTerm};
                      kind::Symbol = :point)
        _check_value_sigma("PointObs", value, sigma)
        isempty(terms) && throw(ArgumentError("PointObs: at least one term required"))
        return new(loc, Float64(value), Float64(sigma), collect(terms), kind)
    end
end
obs_value(o::PointObs) = o.value
obs_sigma(o::PointObs) = o.sigma
obs_location(o::PointObs) = o.loc
obs_kind(o::PointObs) = o.kind
obs_kind(::Type{PointObs}) = :point
obs_terms(o::PointObs, ::StateVector) = o.terms

"""
    WindSpeedObs(loc, value, sigma)

Nonlinear horizontal wind-speed observation `√(u² + v²) = value` (exists to
exercise the Gauss–Newton outer loop). `obs_slots = [(u, 1), (v, 1)]`;
`linearize` gives `(u/s, v/s)` with `s = max(√(u²+v²), 1e-6)` (zero weights
at calm points). The initial `obs_terms` weights, used only before the first
linearisation, are `1/√2` each (a nonzero placeholder so the operator's
sparsity pattern is populated). Only [`MishIndex`](@ref) locations are
supported by [`build_backends`](@ref) (nonlinear obs need the mish backend's
model hook). Kind `:wind_speed`.
"""
struct WindSpeedObs <: AbstractObservation
    loc::ObsLocation
    value::Float64
    sigma::Float64
    function WindSpeedObs(loc::ObsLocation, value::Real, sigma::Real)
        _check_value_sigma("WindSpeedObs", value, sigma)
        return new(loc, Float64(value), Float64(sigma))
    end
end
obs_value(o::WindSpeedObs) = o.value
obs_sigma(o::WindSpeedObs) = o.sigma
obs_location(o::WindSpeedObs) = o.loc
obs_kind(::Type{WindSpeedObs}) = :wind_speed
is_linear(::WindSpeedObs) = false
function obs_terms(::WindSpeedObs, state::StateVector)
    iu, iv, _ = wind_indices(state)
    return [ObsTerm(iu, 0, sqrt(0.5)), ObsTerm(iv, 0, sqrt(0.5))]
end
function obs_slots(::WindSpeedObs, state::StateVector)
    iu, iv, _ = wind_indices(state)
    return [(iu, 1), (iv, 1)]
end
model_value(::WindSpeedObs, sample, ::StateVector) = sqrt(sample[1]^2 + sample[2]^2)
function linearize(::WindSpeedObs, sample, ::StateVector)
    u, v = sample[1], sample[2]
    s = max(sqrt(u^2 + v^2), 1e-6)
    return (u / s, v / s)
end

# ── Groups and sets ──────────────────────────────────────────────────────────

"""
    ObsGroup{T<:AbstractObservation}(kind::Symbol, obs::Vector{T})
    ObsGroup(kind, obs::AbstractVector{T})
    ObsGroup(obs::AbstractVector{T})          # kind = obs_kind(T)

A homogeneous group of observations (one concrete type, hence homogeneous
linearity and sample layout), converted to one backend by
[`build_backends`](@ref). `kind` labels the backend and the group's
[`cost_by_group`](@ref) entry.
"""
struct ObsGroup{T<:AbstractObservation}
    kind::Symbol
    obs::Vector{T}
end
ObsGroup(kind::Symbol, obs::AbstractVector{T}) where {T<:AbstractObservation} =
    ObsGroup{T}(kind, collect(obs))
ObsGroup(obs::AbstractVector{T}) where {T<:AbstractObservation} = ObsGroup(obs_kind(T), obs)

nobs(g::ObsGroup) = length(g.obs)

"""
    ObservationSet(groups::Vector{ObsGroup} = ObsGroup[])

An ordered collection of [`ObsGroup`](@ref)s. See [`push_group!`](@ref),
[`build_backends`](@ref), [`build_obsset`](@ref), [`cost_by_group`](@ref).
"""
struct ObservationSet
    groups::Vector{ObsGroup}
end
ObservationSet() = ObservationSet(ObsGroup[])

nobs(set::ObservationSet) = sum((nobs(g) for g in set.groups); init = 0)

"""
    kinds(set::ObservationSet) -> Vector{Symbol}

The group kinds, in group order (kinds may repeat).
"""
kinds(set::ObservationSet) = [g.kind for g in set.groups]

"""
    push_group!(set::ObservationSet, group::ObsGroup) -> set

Append a group.
"""
push_group!(set::ObservationSet, group::ObsGroup) = (push!(set.groups, group); set)

# ── Conversion to backends ───────────────────────────────────────────────────

function _group_location_type(g::ObsGroup)
    L = typeof(obs_location(g.obs[1]))
    for o in g.obs
        obs_location(o) isa L || throw(ArgumentError(
            "ObsGroup :$(g.kind): mixed location types ($(L) and $(typeof(obs_location(o)))); " *
            "a group must be all MishIndex, all NodeIndex or all XYZ"))
    end
    L <: MishIndex || L <: NodeIndex || L <: XYZ || throw(ArgumentError(
        "ObsGroup :$(g.kind): unsupported location type $(L)"))
    return L
end

# Nonlinear groups: the MishObsBackend's `slots` are the distinct (var, slot)
# in first-appearance order over the point-sorted observations, and its model
# receives the sample in that order. Requiring every observation to share
# `obs_slots` and to list its terms' distinct slots in exactly that order
# makes the derived order equal `obs_slots` whatever the sort permutation.
function _check_nonlinear_group(g::ObsGroup, state::StateVector)
    sl = obs_slots(g.obs[1], state)
    for (m, o) in enumerate(g.obs)
        obs_slots(o, state) == sl || throw(ArgumentError(
            "ObsGroup :$(g.kind): nonlinear observations of one group must share obs_slots " *
            "(observation $(m) differs)"))
        seq = Tuple{Int,Int}[]
        for t in obs_terms(o, state)
            s = (t.var, t.deriv + 1)
            s in seq || push!(seq, s)
        end
        seq == sl || throw(ArgumentError(
            "ObsGroup :$(g.kind): observation $(m)'s terms cover slots $(seq) but obs_slots " *
            "is $(sl); a nonlinear observation's terms must list exactly obs_slots, in order"))
    end
    return sl
end

function _mish_backend(g::ObsGroup, state::StateVector; lattice::Symbol = :mish)
    n = nobs(g)
    dims = (nvars(state), lattice === :nodes ? state.n_nodes : state.n_mish)
    points = lattice === :nodes ? [node_flat(obs_location(o), state.node_dims) for o in g.obs] :
                                  [mish_flat(obs_location(o), state.dims) for o in g.obs]
    y = [obs_value(o) for o in g.obs]
    rinv = [obs_inverse_variance(o) for o in g.obs]
    terms = [[_backend_term(t) for t in obs_terms(o, state)] for o in g.obs]
    linear = is_linear(g.obs[1])
    all(o -> is_linear(o) == linear, g.obs) || throw(ArgumentError(
        "ObsGroup :$(g.kind): mixes linear and nonlinear observations"))
    if linear
        return MishObsBackend(g.kind, dims, points, y, rinv, terms; lattice = lattice)
    end
    lattice === :mish || throw(ArgumentError(
        "ObsGroup :$(g.kind): nonlinear observations at NodeIndex locations are not supported " *
        "(nonlinear observations need MishIndex locations)"))
    sl = _check_nonlinear_group(g, state)
    obs = g.obs
    model = (sample, m) -> (model_value(obs[m], sample, state), linearize(obs[m], sample, state))
    b = MishObsBackend(g.kind, dims, points, y, rinv, terms; model = model)
    b.slots == sl || error("ObsGroup :$(g.kind): backend slot order $(b.slots) != obs_slots $(sl) " *
                           "(internal invariant violated)")
    n == nobs(b) || error("internal: backend size mismatch")
    return b
end

function _scattered_backend(g::ObsGroup, state::StateVector, state_grid)
    all(is_linear, g.obs) || throw(ArgumentError(
        "ObsGroup :$(g.kind): nonlinear observations at XYZ locations are not supported " *
        "(nonlinear observations need MishIndex locations)"))
    xyz = zeros(nobs(g), 3)
    for (m, o) in enumerate(g.obs)
        l = obs_location(o)
        xyz[m, 1] = l.x; xyz[m, 2] = l.y; xyz[m, 3] = l.z
    end
    y = [obs_value(o) for o in g.obs]
    rinv = [obs_inverse_variance(o) for o in g.obs]
    terms = [[_backend_term(t) for t in obs_terms(o, state)] for o in g.obs]
    return ScatteredObsBackend(g.kind, state_grid, variable_names(state), xyz, y, rinv, terms)
end

"""
    build_backends(set::ObservationSet, state::StateVector, state_grid) -> Vector{AbstractObsBackend}

One backend per non-empty group, in group order: [`MishIndex`](@ref)
groups → [`MishObsBackend`](@ref) (nonlinear groups get the model hook
`(sample, m) -> (model_value(obs[m], …), linearize(obs[m], …))`, with the
backend's slot order checked equal to `obs_slots`); [`NodeIndex`](@ref)
groups → [`MishObsBackend`](@ref) with `lattice = :nodes` (linear only);
[`XYZ`](@ref) groups → [`ScatteredObsBackend`](@ref) on `state_grid` (linear
only; nonlinear → `ArgumentError`).
"""
function build_backends(set::ObservationSet, state::StateVector, state_grid)
    out = AbstractObsBackend[]
    for g in set.groups
        nobs(g) == 0 && continue
        L = _group_location_type(g)
        push!(out, L <: MishIndex ? _mish_backend(g, state) :
                   L <: NodeIndex ? _mish_backend(g, state; lattice = :nodes) :
                                    _scattered_backend(g, state, state_grid))
    end
    return out
end

"""
    build_obsset(set::ObservationSet, state::StateVector, state_grid; nchunks = Threads.nthreads()) -> ObsSet

`ObsSet(state_grid, variable_names(state), build_backends(set, state, state_grid); nchunks)`.
"""
build_obsset(set::ObservationSet, state::StateVector, state_grid;
             nchunks::Integer = Threads.nthreads()) =
    ObsSet(state_grid, variable_names(state), build_backends(set, state, state_grid); nchunks = nchunks)

"""
    cost_by_group(set::ObservationSet, obsset::ObsSet, residual) -> Dict{Symbol,Float64}

[`cost_by_backend`](@ref) labelled by kind: `½ Σ rinv·residual²` per group
kind (summed over groups sharing a kind; every kind of `set` has an entry,
`0.0` for empty groups). `obsset` must come from [`build_obsset`](@ref) on
`set`.
"""
function cost_by_group(set::ObservationSet, obsset::ObsSet, residual::AbstractVector{<:Real})
    out = Dict{Symbol,Float64}(k => 0.0 for k in kinds(set))
    nonempty = [g.kind for g in set.groups if nobs(g) > 0]
    length(nonempty) == length(obsset.backends) || throw(DimensionMismatch(
        "cost_by_group: $(length(nonempty)) non-empty groups but $(length(obsset.backends)) backends"))
    for (k, c) in zip(nonempty, cost_by_backend(obsset, residual))
        out[k] += c
    end
    return out
end
