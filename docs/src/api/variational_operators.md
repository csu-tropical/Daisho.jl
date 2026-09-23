# Variational Operator Layer (Internal)

Not exported: the linear operators, observation backends, cost function and
Gauss–Newton/CG driver the problem is assembled from. Stable enough to build
custom problems on, but qualified (`Daisho.`) and subject to change. The public
API built on them is on the [Variational Wind Analysis](variational_wind.md)
page.

```@docs
Daisho.AbstractLinearOp
Daisho.apply!
Daisho.ChainOp
Daisho.DiagonalScaling
Daisho.PencilScratch
Daisho.SplineProjection
Daisho.SplineEvaluation
Daisho.slab_view
Daisho.coefficients_to_springsteel_spectral!
Daisho.sa!
Daisho.sb!
Daisho.sbt!
Daisho.eval!
Daisho.scatter!
Daisho.AxisFilter
Daisho.axis_filter
Daisho.MishFilter
Daisho.filter_length_cells
Daisho.filter_lengths_cells
Daisho.filter_normalisation
Daisho.has_filter
Daisho.ObsSet
Daisho.MishObsBackend
Daisho.ScatteredObsBackend
Daisho.build_backends
Daisho.evaluate!
Daisho.apply_H!
Daisho.apply_Ht!
Daisho.linearize!
Daisho.obs_values
Daisho.obs_rinv
Daisho.slots_needed
Daisho.mish_flat
Daisho.IncrementalCostFunction
Daisho.HessianOperator
Daisho.hessian_mul!
Daisho.cost
Daisho.cost_nonlinear
Daisho.cost_by_backend
Daisho.gradient!
Daisho.analysis_coefficients!
Daisho.solve_incremental!
```

## Observation internals

```@docs
Daisho.nobs
Daisho.is_linear
Daisho.linearize
Daisho.model_value
Daisho.inverse_variance
Daisho.push_group!
Daisho.build_obsset
Daisho.relinearize!
Daisho.cost_by_group
```
