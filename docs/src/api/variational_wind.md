# Variational Wind Analysis (Stage 2)

The incremental 3DVAR `(u, v, w)` analysis built on the stage-1
[`WindGridAccumulator`](@ref). See the
[Variational Wind guide](../guide/variational_wind.md) for configuration and
workflow, and the [theory page](../theory/variational_wind.md) for the
mathematics. The `[variational]` configuration structs
([`VariationalParameters`](@ref) and its sub-tables) are listed on the
[Parameters](parameters.md) page.

## Pipeline

```@docs
Daisho.analyze_radar_volumes_variational
Daisho.write_variational_netcdf
```

## Problem and result

```@docs
Daisho.VariationalProblem
Daisho.Background
Daisho.synthesis_smooth_fit
Daisho.synthesis_sigma_map
Daisho.synthesis_level_profile
Daisho.solve_variational
Daisho.VariationalResult
Daisho.analysis_field
Daisho.background_field
Daisho.increment_field
Daisho.evaluate_analysis!
Daisho.analysis_derivatives
Daisho.horizontal_divergence
Daisho.vertical_vorticity
Daisho.mass_continuity_residual
```

## State vector

```@docs
Daisho.StateVector
Daisho.StateVariable
Daisho.create_state_grid
Daisho.state_view
Daisho.wind_indices
Daisho.has_role
Daisho.role_index
Daisho.STATE_ROLE_VOCAB
Daisho.STATE_SINGULAR_ROLES
Daisho.filter_lengths_m
Daisho.background_sigma_vector
Daisho.AbstractStateTransform
Daisho.IdentityTransform
Daisho.to_physical
Daisho.d_physical_d_control
Daisho.BackgroundErrorSpec
Daisho.UniformSigma
Daisho.FieldSigma
Daisho.sigma_field
Daisho.sigma_vector
```

## Observations

`XYZ` is not exported (the name would clash with `ColorTypes.XYZ` for
plotting users); write `Daisho.XYZ`.

```@docs
Daisho.AbstractObservation
Daisho.RadialVelocitySuperObs
Daisho.MassContinuityObs
Daisho.ScalarValueObs
Daisho.PointObs
Daisho.WindSpeedObs
Daisho.MishIndex
Daisho.XYZ
Daisho.ObsTerm
Daisho.ObsGroup
Daisho.ObservationSet
Daisho.build_observation_set
Daisho.build_radial_velocity_superobs
Daisho.radial_velocity_superobs
Daisho.build_mass_continuity_obs
Daisho.build_scalar_value_obs
```

## Solver

```@docs
Daisho.VarSolverOptions
Daisho.VarSolverStats
Daisho.solver_options
```

## Stage-1 extensions

Gate-level physics added to the stage-1 accumulator for the variational
stage (`[synthesis.fall_speed]`, `[synthesis.gate_error]`).

```@docs
Daisho.FallSpeedParameters
Daisho.GateErrorParameters
Daisho.GateModel
Daisho.terminal_fall_speed
Daisho.gate_sigma
```

## Operator layer (internal)

The unexported linear operators, observation backends, cost function and
Gauss–Newton/CG driver the problem is assembled from are documented on the
[Variational Operator Layer](variational_operators.md) page.
