# The variational problem (plan §5): background, background-error chain
# C = S∘D∘F, observation set, incremental cost function, solve, and the
# analysis result with its diagnostics (analysis/background/increment fields,
# derivative slabs, divergence, vorticity, anelastic mass-continuity residual).
#
# This file holds the in-memory API only; the file-writing pipeline
# (`analyze_radar_volumes_variational`, `write_variational_netcdf`) is a later
# phase.

# ── Background ────────────────────────────────────────────────────────────────

const _BACKGROUND_MODES = (:zero, :synthesis, :custom)
const _BACKGROUND_CONFIG_MODES = ("zero", "synthesis")

"""
    Background(x::Vector{Float64}, sigma::Vector{Float64}, mode::Symbol)
    Background(state::StateVector, p::DaishoParameters)                        # :zero
    Background(state::StateVector, stage1::SynthesisOutput, p::DaishoParameters)  # :synthesis
    Background(state::StateVector, nothing, p::DaishoParameters)               # from `[variational] background`

The background (first guess) of the variational analysis and the diagonal
`D` of its background-error chain `C = S∘D∘F`, both on the state mish.

# Fields
- `x::Vector{Float64}`: the mish-space background `x_g`, flat and
  variable-major like the state vector (`length(state)`; see
  [`state_view`](@ref)). Its spline projection `a_g = S x_g` is the outer
  loop's starting point.
- `sigma::Vector{Float64}`: the per-point background-error standard deviation
  (same layout), every entry finite and `> 0` so `D` is positive definite.
- `mode::Symbol`: `:zero`, `:synthesis`, or `:custom` (the raw constructor's
  label for a caller-supplied background).

# Modes
- `:zero` (`[variational] background = "zero"`, samurai-like): `x = 0`,
  `sigma = background_sigma_vector(state)` (each variable's
  `[variational.background_error]` σ).
- `:synthesis` (`background = "synthesis"`): the stage-1 multi-Doppler
  product seeds the horizontal winds. For the `:wind_u`/`:wind_v` state
  variables, `x = stage1.comp1`/`comp2` where `stage1.quality_flag == 0`
  (else `0`), and `sigma = background_error_inflation · stage1.sigma1`/`sigma2`
  there (else the variable's config σ). The stage-1 σ is additionally
  guarded from below by `1e-3 ·` the config σ: at heavily overdetermined
  points the sandwich σ can be tiny, which would make `D` nearly singular and
  pin the analysis to the background. Every other variable (W, scalars) keeps
  its config σ and a zero background. Stage-1 and the radial-velocity
  super-obs come from the same gates, so this mode counts the radar data
  twice; `background_error_inflation ≫ 1` weakens the background to
  compensate (documented double-counting guard).

`stage1` must be the product of [`finalize_wind`](@ref) on the accumulator
built on this state's mish (`size(stage1.comp1) == state.dims`, else
`DimensionMismatch`), with the same output frame as the
[`StateVector`](@ref): its `component_names` must equal the names of the
`:wind_u`/`:wind_v` state variables (`ArgumentError` otherwise). Note the
radial-velocity super-obs always constrain Cartesian `(u, v)`, so the
`:synthesis` background is only meaningful with the default
`CartesianFrame`.

The three-argument form with `nothing` dispatches on
`p.variational.background`: `"zero"` → the `:zero` background; `"synthesis"`
→ `ArgumentError` (a stage-1 product is required).
"""
struct Background
    x::Vector{Float64}
    sigma::Vector{Float64}
    mode::Symbol
    function Background(x::AbstractVector{<:Real}, sigma::AbstractVector{<:Real}, mode::Symbol)
        length(x) == length(sigma) || throw(DimensionMismatch(
            "Background: x has length $(length(x)) but sigma has $(length(sigma))"))
        mode in _BACKGROUND_MODES || throw(ArgumentError(
            "Background: mode must be one of $(_BACKGROUND_MODES), got :$(mode)"))
        all(isfinite, x) || throw(ArgumentError("Background: x must be finite everywhere"))
        all(s -> isfinite(s) && s > 0.0, sigma) || throw(ArgumentError(
            "Background: sigma must be finite and > 0 everywhere (D must be positive definite)"))
        return new(Vector{Float64}(x), Vector{Float64}(sigma), mode)
    end
end

function _check_background_config(p::DaishoParameters)
    p.variational.background in _BACKGROUND_CONFIG_MODES || throw(ArgumentError(
        "[variational] background = \"$(p.variational.background)\" is not valid; " *
        "expected one of $(join(_BACKGROUND_CONFIG_MODES, ", "))"))
    return p.variational.background
end

function Background(state::StateVector, p::DaishoParameters)
    _check_background_config(p)
    return Background(zeros(length(state)), background_sigma_vector(state), :zero)
end

function Background(state::StateVector, ::Nothing, p::DaishoParameters)
    mode = _check_background_config(p)
    mode == "synthesis" && throw(ArgumentError(
        "Background: [variational] background = \"synthesis\" needs the stage-1 " *
        "SynthesisOutput (from finalize_wind), got `nothing`"))
    return Background(state, p)
end

function Background(state::StateVector, stage1::SynthesisOutput, p::DaishoParameters)
    _check_background_config(p)
    for a in (stage1.comp1, stage1.comp2, stage1.sigma1, stage1.sigma2, stage1.quality_flag)
        size(a) == state.dims || throw(DimensionMismatch(
            "Background: stage-1 arrays have size $(size(a)) but the state mish is " *
            "$(state.dims); finalize_wind must run on the accumulator built on the state mish"))
    end
    iu, iv, _ = wind_indices(state)
    names = (state.vars[iu].name, state.vars[iv].name)
    Tuple(stage1.component_names) == names || throw(ArgumentError(
        "Background: stage-1 component names $(stage1.component_names) differ from the " *
        "state's wind_u/wind_v variables $(names); finalize_wind and StateVector must use " *
        "the same output frame"))
    infl = p.variational.background_error_inflation
    x = zeros(length(state))
    sigma = background_sigma_vector(state)
    q = stage1.quality_flag
    for (v, comp, sig) in ((iu, stage1.comp1, stage1.sigma1), (iv, stage1.comp2, stage1.sigma2))
        xv = state_view(x, state, v)
        sv = state_view(sigma, state, v)
        @inbounds for idx in eachindex(q)
            q[idx] == 0 || continue
            xv[idx] = comp[idx]
            sv[idx] = max(infl * sig[idx], 1e-3 * sv[idx])   # guard vs the config σ
        end
    end
    return Background(x, sigma, :synthesis)
end

# ── VariationalProblem ───────────────────────────────────────────────────────

"""
    VariationalProblem

Everything the incremental 3DVAR solve needs, assembled on the state mish.

# Fields
- `state::StateVector`, `state_grid` (the RRR state grid of
  [`create_state_grid`](@ref)).
- `background::Background`.
- `observations::ObservationSet` and its backend form `obsset::ObsSet`.
- `S::SplineProjection`, `D::DiagonalScaling` (per-point map of
  `background.sigma`), `F::MishFilter`, `C = ChainOp(S, D, F)` — the
  control-to-coefficient operator `B^{1/2}`.
- `cf::IncrementalCostFunction` (background coefficients `a_g = S x_g`).
- `nobs_vr`: per-point radial-velocity super-obs row counts (`NOBS_VR`
  diagnostic) when the super-obs were built from an accumulator, else
  `nothing`.
- `stage1`: the stage-1 [`SynthesisOutput`](@ref) when built from an
  accumulator, else `nothing`.
- `reference::AbstractReferenceProfile` (ρ̄(z) for continuity diagnostics).
- `params::DaishoParameters`.

See [`solve_variational`](@ref).
"""
struct VariationalProblem{G,TC,CF}
    state::StateVector
    state_grid::G
    background::Background
    observations::ObservationSet
    S::SplineProjection
    D::DiagonalScaling
    F::MishFilter
    C::TC
    obsset::ObsSet
    cf::CF
    nobs_vr::Union{Nothing,Array{Int8,3}}
    stage1::Union{Nothing,SynthesisOutput}
    reference::AbstractReferenceProfile
    params::DaishoParameters
end

# Filter lengths (metres → cells) for `MishFilter`'s convenience form. The
# Gaussian's length is its σ; the Lanczos kernel's is its lobe half-width `a`,
# which must be an integer number of cells: round to the nearest integer ≥ 1
# (0 stays 0 = unfiltered).
function _state_filter_cells(state_grid, state::StateVector, kind::Symbol)
    cells = filter_lengths_cells(state_grid, variable_names(state), filter_lengths_m(state))
    if kind === :lanczos
        cells = map(L -> (isnan(L) || iszero(L)) ? L : max(1.0, round(L)), cells)
    end
    return cells
end

"""
    VariationalProblem(state, state_grid, observations::ObservationSet, background::Background,
                       reference::AbstractReferenceProfile, p::DaishoParameters;
                       nobs_vr = nothing, stage1 = nothing, nchunks = Threads.nthreads())

In-memory constructor from parts. Builds

- `S = SplineProjection(state_grid, names)`;
- `D = DiagonalScaling` with the per-point map `background.sigma` of every
  variable;
- `F = MishFilter(state_grid, names, cells; kind = Symbol(p.variational.filter_type))`
  where `cells = filter_lengths_cells(state_grid, names, filter_lengths_m(state))`:
  each variable's `[variational.background_error] filter_length_m` is the
  Gaussian σ (or the Lanczos half-width `a`, rounded to a whole number ≥ 1 of
  cells) in metres, converted to cells per axis by that axis' cell width;
- `C = ChainOp(S, D, F)`;
- `obsset = build_obsset(observations, state, state_grid; nchunks)`;
- `a_g = S · background.x` and `cf = IncrementalCostFunction(C, obsset, a_g)`.
"""
function VariationalProblem(state::StateVector, state_grid,
                            observations::ObservationSet, background::Background,
                            reference::AbstractReferenceProfile, p::DaishoParameters;
                            nobs_vr::Union{Nothing,Array{Int8,3}} = nothing,
                            stage1::Union{Nothing,SynthesisOutput} = nothing,
                            nchunks::Integer = Threads.nthreads())
    n = length(state)
    length(background.x) == n || throw(DimensionMismatch(
        "VariationalProblem: background has length $(length(background.x)), state has $(n)"))
    nobs_vr === nothing || size(nobs_vr) == state.dims || throw(DimensionMismatch(
        "VariationalProblem: nobs_vr has size $(size(nobs_vr)), state mish is $(state.dims)"))
    names = variable_names(state)
    nm = state.n_mish
    S = SplineProjection(state_grid, names; nchunks = nchunks)
    sig = Union{Float64,Vector{Float64}}[background.sigma[_state_range(state, v)] for v in 1:nvars(state)]
    D = DiagonalScaling(sig, nm)
    kind = Symbol(p.variational.filter_type)
    F = MishFilter(state_grid, names, _state_filter_cells(state_grid, state, kind);
                   kind = kind, nchunks = nchunks)
    C = ChainOp(S, D, F)
    obsset = build_obsset(observations, state, state_grid; nchunks = nchunks)
    a_g = zeros(size(S, 1))
    apply!(a_g, S, background.x)
    cf = IncrementalCostFunction(C, obsset, a_g)
    return VariationalProblem(state, state_grid, background, observations, S, D, F, C,
                              obsset, cf, nobs_vr, stage1, reference, p)
end

"""
    VariationalProblem(acc::WindGridAccumulator, radar_grid, sgrid, p::DaishoParameters;
                       frame = CartesianFrame(), reference = nothing,
                       nchunks = Threads.nthreads())

High-level constructor from a filled stage-1 accumulator built on `sgrid`'s
quadrature mish (`build_accumulator(build_springsteel_grid_spec(sgrid; …), p)`
then `grid_sweep!` over every sweep):

1. `stage1 = finalize_wind(acc, p; frame)` (kept as `prob.stage1`);
2. `reference` defaults to [`build_reference_profile`](@ref)`(p, sgrid)`;
3. `state = StateVector(p, sgrid; frame)`,
   `state_grid = create_state_grid(p, sgrid, state)`;
4. `background = Background(state, stage1 or nothing, p)` per
   `[variational] background`;
5. observations: the radial-velocity super-obs of `acc`
   ([`radial_velocity_superobs`](@ref), which also yields `nobs_vr`), mass
   continuity, and value obs of `[fields]`-derived state variables from
   `radar_grid` (the gridded scalar mish, `finalize_grid(acc.scalar)`; may be
   `nothing` when the state has no such variables) — the standard
   [`build_observation_set`](@ref) order, with the super-obs built once.
"""
function VariationalProblem(acc::WindGridAccumulator,
                            radar_grid::Union{Nothing,AbstractArray{<:Real,4}},
                            sgrid, p::DaishoParameters;
                            frame::SynthesisFrame = CartesianFrame(),
                            reference::Union{Nothing,AbstractReferenceProfile} = nothing,
                            nchunks::Integer = Threads.nthreads())
    mode = _check_background_config(p)
    stage1 = finalize_wind(acc, p; frame = frame)
    ref = reference === nothing ? build_reference_profile(p, sgrid) : reference
    state = StateVector(p, sgrid; frame = frame)
    state_grid = create_state_grid(p, sgrid, state)
    background = Background(state, mode == "synthesis" ? stage1 : nothing, p)
    group, nobs_vr = radial_velocity_superobs(acc, state, p)
    obs = build_observation_set(nothing, radar_grid, state, ref, p)
    nobs(group) > 0 && pushfirst!(obs.groups, group)
    return VariationalProblem(state, state_grid, obs, background, ref, p;
                              nobs_vr = nobs_vr, stage1 = stage1, nchunks = nchunks)
end

"""
    solver_options(p::DaishoParameters) -> VarSolverOptions

[`VarSolverOptions`](@ref) from `[variational]`: `max_outer_iterations`,
`max_inner_iterations`, `inner_tolerance`, `outer_tolerance` (other options
at their defaults).
"""
solver_options(p::DaishoParameters) = VarSolverOptions(
    max_outer = p.variational.max_outer_iterations,
    max_inner = p.variational.max_inner_iterations,
    inner_tolerance = p.variational.inner_tolerance,
    outer_tolerance = p.variational.outer_tolerance)

# ── Result ────────────────────────────────────────────────────────────────────

"""
    VariationalResult

Output of [`solve_variational`](@ref). All mish-space vectors are flat and
variable-major (see [`state_view`](@ref)).

# Fields
- `state::StateVector`.
- `coefficients`: analysis spline coefficients `a_g + C ξ_total` (Daisho
  layout; export with [`evaluate_analysis!`](@ref)).
- `analysis`: the spline analysis evaluated on the mish (value slot of every
  variable) — the smooth, boundary-condition-exact analysis.
- `background`: the mish background `x_g`.
- `increment`: the mish-space increment `D F ξ_total` (before the spline
  projection, so NOT spline-smoothed; `analysis ≈ S-smoothed(background +
  increment)`).
- `control`: `ξ_total`.
- `stats::VarSolverStats`.
- `cost_by_group`: `½ Σ r (y − h(a))²` per observation kind at the analysis.
- `nobs_vr`: the problem's super-obs row counts (or `nothing`).
- `converged::Bool`.
"""
struct VariationalResult
    state::StateVector
    coefficients::Vector{Float64}
    analysis::Vector{Float64}
    background::Vector{Float64}
    increment::Vector{Float64}
    control::Vector{Float64}
    stats::VarSolverStats
    cost_by_group::Dict{Symbol,Float64}
    nobs_vr::Union{Nothing,Array{Int8,3}}
    converged::Bool
end

"""
    solve_variational(prob::VariationalProblem; opts = solver_options(prob.params)) -> VariationalResult

Run [`solve_incremental!`](@ref) on `prob.cf` (which resets and then
accumulates `ξ_total`, so repeated calls are independent), then form the
analysis coefficients, the mish analysis (one [`SplineEvaluation`](@ref)
value pass), the mish increment `D F ξ_total`, and the per-kind observation
cost at the analysis.
"""
function solve_variational(prob::VariationalProblem;
                           opts::VarSolverOptions = solver_options(prob.params))
    cf = prob.cf
    state = prob.state
    stats = solve_incremental!(cf, opts)
    ξ = copy(control_increment(cf))
    a = zeros(size(prob.C, 1))
    analysis_coefficients!(a, cf)

    names = variable_names(state)
    E = SplineEvaluation(prob.state_grid, names, [(v, 1) for v in 1:nvars(state)])
    analysis = zeros(size(E, 1))
    apply!(analysis, E, a)

    tmp = zeros(length(ξ))
    increment = zeros(length(ξ))
    apply!(tmp, prob.F, ξ)
    apply!(increment, prob.D, tmp)

    yh = zeros(nobs(prob.obsset))
    evaluate!(yh, prob.obsset, a)
    cbg = cost_by_group(prob.observations, prob.obsset, obs_values(prob.obsset) .- yh)

    return VariationalResult(state, a, analysis, copy(prob.background.x), increment, ξ,
                             stats, cbg, prob.nobs_vr, stats.converged)
end

"""
    analysis_field(r::VariationalResult, key) -> Array{Float64,3}
    background_field(r::VariationalResult, key) -> Array{Float64,3}
    increment_field(r::VariationalResult, key) -> Array{Float64,3}

A `(nz, ny, nx)` copy of one state variable's analysis, background, or
mish-space increment; `key` is a state-variable name (`"U"`), singular role
(`:wind_u`), or index.
"""
analysis_field(r::VariationalResult, key::Union{Int,String,Symbol}) =
    copy(state_view(r.analysis, r.state, key))
background_field(r::VariationalResult, key::Union{Int,String,Symbol}) =
    copy(state_view(r.background, r.state, key))
increment_field(r::VariationalResult, key::Union{Int,String,Symbol}) =
    copy(state_view(r.increment, r.state, key))

"""
    evaluate_analysis!(state_grid, r::VariationalResult) -> state_grid

Write the analysis coefficients into `state_grid.spectral`
([`coefficients_to_springsteel_spectral!`](@ref)) and run
`Springsteel.gridTransform!` so `state_grid.physical` holds the analysis and
its derivatives on the mish (and Springsteel's `regularGridTransform` can
resample it). Call from a single thread (`gridTransform!` mutates shared
spline buffers).
"""
function evaluate_analysis!(state_grid, r::VariationalResult)
    coefficients_to_springsteel_spectral!(state_grid, r.coefficients, variable_names(r.state))
    Springsteel.gridTransform!(state_grid)
    return state_grid
end

"""
    analysis_derivatives(prob::VariationalProblem, r::VariationalResult)
        -> Dict{String,NamedTuple{(:value, :dx, :dy, :dz)}}

The analysis value and first derivatives `∂/∂x, ∂/∂y, ∂/∂z` (per metre) of
every state variable on the mish, each `(nz, ny, nx)`, from one
[`SplineEvaluation`](@ref) pass over all four slots of every variable.
"""
function analysis_derivatives(prob::VariationalProblem, r::VariationalResult)
    state = prob.state
    names = variable_names(state)
    nv = nvars(state)
    E = SplineEvaluation(prob.state_grid, names, [(v, s) for v in 1:nv for s in 1:4])
    y = zeros(size(E, 1))
    apply!(y, E, r.coefficients)
    out = Dict{String,NamedTuple{(:value, :dx, :dy, :dz),NTuple{4,Array{Float64,3}}}}()
    for v in 1:nv
        b = 4 * (v - 1)
        out[names[v]] = (value = copy(slab_view(y, E, b + 1)), dx = copy(slab_view(y, E, b + 2)),
                         dy = copy(slab_view(y, E, b + 3)), dz = copy(slab_view(y, E, b + 4)))
    end
    return out
end

function _wind_derivatives(prob::VariationalProblem, der)
    iu, iv, iw = wind_indices(prob.state)
    names = variable_names(prob.state)
    return der[names[iu]], der[names[iv]], iw === nothing ? nothing : der[names[iw]]
end

"""
    horizontal_divergence(prob, r; derivatives = analysis_derivatives(prob, r)) -> Array{Float64,3}

`∂u/∂x + ∂v/∂y` of the analysis on the mish (`:wind_u`/`:wind_v` roles).
"""
function horizontal_divergence(prob::VariationalProblem, r::VariationalResult;
                               derivatives = analysis_derivatives(prob, r))
    du, dv, _ = _wind_derivatives(prob, derivatives)
    return du.dx .+ dv.dy
end

"""
    vertical_vorticity(prob, r; derivatives = analysis_derivatives(prob, r)) -> Array{Float64,3}

`∂v/∂x − ∂u/∂y` of the analysis on the mish.
"""
function vertical_vorticity(prob::VariationalProblem, r::VariationalResult;
                            derivatives = analysis_derivatives(prob, r))
    du, dv, _ = _wind_derivatives(prob, derivatives)
    return dv.dx .- du.dy
end

"""
    mass_continuity_residual(prob, r; derivatives = analysis_derivatives(prob, r)) -> Array{Float64,3}

The anelastic mass-continuity residual of the analysis on the mish,
`∂u/∂x + ∂v/∂y + ∂w/∂z + w·d(ln ρ̄)/dz` (s⁻¹), with ρ̄ from
`prob.reference`. Requires a `:wind_w` state variable (`ArgumentError`
otherwise).
"""
function mass_continuity_residual(prob::VariationalProblem, r::VariationalResult;
                                  derivatives = analysis_derivatives(prob, r))
    du, dv, dw = _wind_derivatives(prob, derivatives)
    dw === nothing && throw(ArgumentError(
        "mass_continuity_residual: the state has no `wind_w` variable"))
    dl = [Float64(dlnrho_dz(prob.reference, z)) for z in prob.state.z_mish]
    return du.dx .+ dv.dy .+ dw.dz .+ dw.value .* reshape(dl, :, 1, 1)
end
