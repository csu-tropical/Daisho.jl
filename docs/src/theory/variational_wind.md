# Variational Wind Solver (Stage 2)

Stage 2 is an incremental, samurai-style 3DVAR analysis that produces a
`(u, v, w)` wind field satisfying anelastic mass continuity, built on top of
stage 1 (the [dual-Doppler retrieval](wind_synthesis.md)) and on Springsteel's
cubic B-spline quadrature grid. Where stage 1 solves each grid point
independently from an unconstrained least-squares system, stage 2 couples
every point through a global cost function whose control variable lives on
the spline quadrature lattice (the *mish*) rather than on the B-spline
coefficients themselves — the key departure from samurai that makes boundary
conditions exact (§2).

Daisho's implementation is a from-scratch Julia port of the samurai
formulation (Dennis et al. 2022), reusing Springsteel's B-spline machinery
(Ooyama 2002) but replacing samurai's coefficient-space recursive filter,
hand-coded truncated Newton solver, and switch-statement observation types
with a mish-space filter, an analytic Gauss–Newton/CG solve (Krylov.jl), and
multiple dispatch over observation types. §7 below itemizes the differences.

## 1. The incremental cost function

The analysis increment relative to a background `a_g` (spline coefficients)
is written through a **control variable** `ξ`:

```math
J(\xi) = \tfrac{1}{2}\|\xi\|^2 + \tfrac{1}{2}\sum_m r_m\left([H_k C \xi]_m - d_m\right)^2,
\qquad r_m = 1/\sigma_m^2
```

```math
d_m = y_m - h_m(a_k), \qquad a_k = a_g + C\,\xi_{\text{total}}
```

```math
\nabla J(\xi) = \xi + C^\top H_k^\top R\,(H_k C \xi - d), \qquad
\mathrm{Hess} = I + C^\top H_k^\top R H_k C
```

`d` is the innovation at the current outer-loop linearisation `a_k`; `H_k` is
the observation operator's tangent linear at `a_k` (identical to `H` for
linear observations); `R = \mathrm{diag}(r_m)` is the observation inverse
variance. The first term `½‖ξ‖²` is the implicit background penalty: because
`ξ` already carries the σ_b scaling (through `D`, below), a unit-variance
prior on `ξ` *is* the Gaussian background term of the traditional
`J = (x-x_b)^\top B^{-1}(x-x_b)/2 + \dots` formulation, with `B = C C^\top`.

**Why the mish, not the coefficients?** The control variable is defined on the
quadrature mish because that is where the background-error covariance model
(a spatial filter times a diagonal σ) is easiest to specify, and — critically
— because applying the boundary-condition projector `S` (§2) *after*
filtering, rather than before, makes the increment's boundary conditions
exact by construction (§2.3). samurai's control variable lived on the
B-spline coefficients and filtered them directly with a recursive filter,
which does not respect a coefficient-space boundary condition and so
corrupted it (samurai problem 2 of §1).

Implemented in `src/variational/cost_function.jl`
(`IncrementalCostFunction`, `hessian_mul!`, `cost`, `gradient!`) and
`solver.jl` (`solve_incremental!`).

## 2. The control-variable chain `C = S∘D∘F`

`C` maps a mish-space control vector `ξ` to spline coefficients `a`,
factored as three operators applied right to left —
`ChainOp(S, D, F)` applies `F`, then `D`, then `S` —
with the exact adjoint `Cᵀ = Fᵀ Dᵀ Sᵀ` applied left to right. Every factor is
implemented with its **exact** transpose (verified by dot-product/adjoint
tests, not approximated), so the whole chain, the Hessian, and CG all use
consistent, allocation-free operators after warm-up.

### 2.1 `F` — mish-space spatial filter

`F` is a separable, per-axis, per-variable convolution on the mish points,
built from Springsteel's spline-direction kernel (`GaussianFilter` or
`LanczosFilter`) and stored as a sparse row-normalised matrix
(`AxisFilter`, `MishFilter`):

```math
F_{ij} = \frac{K(x_j - x_i)}{W_i}\ \text{for } |x_j - x_i| \le \text{support},
\qquad W_i = \sum_j K_{ij}
```

`F` is **not symmetric**: `(F - F^\top)_{ij} = K_{ij}(1/W_i - 1/W_j)`, which
is nonzero within one kernel support of a domain boundary (the truncated
window there gives adjacent points different row sums `W`) and, on a
non-uniform (Gauss) mish, even in the interior at the size of the truncated
kernel tail. `Fᵀ` is applied as the exact sparse transpose — never
approximated as `F` — so the adjoint and the Hessian stay exact. (An
optional `symmetric = true` variant, `W_q^{1/2}(K/\kappa)W_q^{1/2}` with the
SB quadrature weights, is available on `AxisFilter`/`MishFilter` for future
preconditioning work but is not reachable from `[variational]` config today.)

Filter lengths are configured in **metres** per state variable and axis
(`[variational.background_error.<name>] filter_length_m = [i, j, k]`) and
converted to cells internally (`filter_lengths_cells`). Because `F` is not
symmetric, the *effective* correlation length of the background covariance `B = C C^\top` seen by an
observation is that of `F F^\top`, which for a Gaussian kernel of standard
deviation σ (in cells) is **√2·σ**, not σ — a common point of confusion
migrating filter lengths from samurai (§7).

### 2.2 `D` — diagonal background-error scaling

`D` is a diagonal operator (`DiagonalScaling`)
holding each state variable's background-error standard deviation σ_b,
either a single scalar (`UniformSigma`) or a
per-mish-point map (`FieldSigma`). `D` is applied
*after* `F` (order `D·F`, i.e. filter white noise, then scale by σ_b) so a
per-point σ map is respected and matches samurai's `SC` factor. `D` is
symmetric, so its own adjoint equals itself.

### 2.3 `S` — spline projection

`S` projects a filtered, scaled mish field onto B-spline coefficients that
satisfy the state variable's boundary conditions **exactly**. Per axis,
`S = S_A \circ S_B` (tensor product across the three axes, in `k, j, i`
order):

```math
S_B = E_0^\top \cdot \mathrm{diag}(w), \qquad w_i = \Delta x \cdot q_i
\quad\text{(Springsteel's \texttt{\_sb\_matrix}, quadrature weights $q$)}
```

```math
S_A = \Gamma^\top\left(\Gamma (P+Q) \Gamma^\top\right)^{-1}\Gamma
\qquad\text{(symmetric: } S_A^\top = S_A\text{)}
```

`Γ` is the boundary-condition operator that maps the full coefficient space
onto its boundary-consistent subspace (Springsteel's `GammaBC`); `(P+Q)` is
the spline mass-plus-penalty matrix; the middle factor's Cholesky is
pre-factored per axis (`pqFactor`, applied with `ldiv!`). Because `S_A`
projects through `Γ` and back, **every coefficient vector produced by `S`
already lies in the boundary-consistent subspace** — the increment satisfies
homogeneous boundary conditions exactly, with no further correction step.

Implemented in `src/variational/spline_operator.jl`
(`AxisSpline`, `sb!`,
`sa!`, `SplineProjection`).
`Sᵀ` reverses the sweep order (`i, j, k`) and reverses the per-axis order to
`sa!` then `sbt!` (the transpose of `sb!`, `u = w \odot (E_0 b)`) — again the
exact transpose, not an approximation.

**Filter-then-project is why boundary conditions come out exact.** samurai
applied its recursive filter directly to the B-spline coefficients, *after*
the `S_A` projection (or interleaved with it in ways that did not commute
with `Γ`), so the filtered coefficients no longer satisfied the boundary
condition the projection had just enforced — the corruption is samurai
problem 2 of §1. Daisho filters strictly in physical (mish) space and
projects last, so the projection is the very last operation applied to the
control path and its boundary-condition guarantee is never subsequently
disturbed.

`coefficients_to_springsteel_spectral!` (`spline_operator.jl`) converts
Daisho's `(b_k, b_j, b_i)` k-fastest coefficient layout to Springsteel's
i-fastest spectral storage (`B = (P+Q)\cdot a` per axis, since `a = \Gamma^\top x`
and `S_A(B) = a`), so `Springsteel.gridTransform!` reproduces the analysis
field and its derivatives for output.

## 3. The observation operator `H`

Observations act on the **spline coefficient vector** `a` (the output of
`C`), not directly on `ξ`, so `H_k C` composes with the control chain. An
observation is a weighted sum of *terms* `(variable, derivative slot,
weight)`, slots `1 = value, 2 = ∂i (x), 3 = ∂j (y), 4 = ∂k (z)`:

```math
h_m(a) = \sum_t w_t \cdot \left[\text{slot}_t \text{ of variable}_t\right](\text{location}_m)
```

Two backends implement `H`/`Hᵀ` (`MishObsBackend`, `ScatteredObsBackend`, unified by `ObsSet`):

- **Mish-located** observations (radial-velocity super-obs, mass continuity,
  gridded scalar fields — anything indexed by a `MishIndex`) share one `SplineEvaluation` tensor-product pass to the union of `(variable,
  slot)` slabs the whole backend needs, followed by per-observation weighted
  sums; the adjoint scatters into the slabs and applies one evaluation
  adjoint pass. This is the backend every observation type in this phase
  uses.
- **Scattered** observations at arbitrary `(x, y, z)` (`XYZ` locations, `PointObs`) build a sparse
  `nobs × ncoef` matrix from the 4×4×4 tensor-product B-spline stencil at
  each point (Springsteel's single-point `SItransform`, non-mish-aligned);
  linear only.

Both are exact adjoints of their forward operator (dot-product tested), and
`ObsSet` itself is an `AbstractLinearOp`, so
the whole chain `H_k C` (and its transpose) composes through the same
generic interface as `S`, `D`, `F`.

## 4. Super-observations from the stage-1 normal system

Rather than re-visiting every gate, the radar data term reuses stage 1's
per-point accumulated normal system directly. At each mish point, stage 1
(`WindGridAccumulator`) has already formed

```math
S_p = \sum_m w_m\, g_m g_m^\top, \qquad b_p = \sum_m w_m\, \tilde v_m\, g_m,
\qquad g = [\sin\mathrm{az}\cos\mathrm{el},\ \cos\mathrm{az}\cos\mathrm{el},\ \sin\mathrm{el}]
```

(the fall-speed-corrected value `ṽ` and error-scaled weight `w`, §5). Since

```math
\sum_m w_m(\tilde v_m - g_m^\top x)^2 = x^\top S_p x - 2 x^\top b_p + \text{const},
```

eigendecomposing `S_p = \sum_r \lambda_r e_r e_r^\top` and setting
`\beta_r = e_r^\top b_p`, each direction with `λ_r > 0` and `λ_r` above a
relative guard (`eigenvalue_guard_relative · λ_max`) becomes one
observation (`RadialVelocitySuperObs`):

```math
y_r = \beta_r / \lambda_r, \qquad r_r = \lambda_r / (\sigma_{vr}^2 \cdot \text{inflation})
```

with terms `(u, val, e_r[1])`, `(v, val, e_r[2])`, and `(w, val, e_r[3])`
only when the state has a `wind_w` variable and `e_r[3] \neq 0`. Summed over
the retained rows, `\sum_r r_r(e_r^\top x - y_r)^2` reproduces the original
weighted gate cost (and its gradient `2(S_p x - b_p)/\sigma_{vr}^2`) exactly,
up to the constant and the rows dropped by the guard. Degenerate geometry —
a single look, or a point near the baseline — has a rank-deficient `S_p`, so
fewer rows are emitted; nothing is inverted, and no gate row is discarded
silently (the guard only drops directions the data genuinely do not
constrain). The per-point row count `NOBS_VR ∈ 0:3` is a diagnostic
(`radial_velocity_superobs_count`).

A `(u, v, w)` full 3×3 system requires `[synthesis] include_vertical_velocity
= true` (stage 1 accumulating the 3-unknown system, `g` including `sin el`);
`[variational.radial_velocity] horizontal_approximation = true` drops the
`w` row unconditionally (samurai `horizontal_radar_appx`), reproducing the
dual-Doppler `(u, v)` approximation and permitting the 2-unknown
accumulation. A structurally w-blind 3×3 point (every contributing gate at
zero elevation, `S_p[3,3] = 0` exactly) is reduced to its 2×2 block
automatically. Implemented in `src/variational/superobs.jl`
(`radial_velocity_superobs`).

## 5. Per-gate error model and the fall-speed correction

Both corrections are applied **upstream**, at stage-1 accumulation, so they
propagate into the super-obs eigen-split without any special-casing in the
variational layer itself.

**Fall speed** (`terminal_fall_speed`, a
port of samurai's `VarDriver3D.cpp` `MetObs::radar` branch): from the gate's
reflectivity (dBZ) and the configured reference-density profile ρ̄(z),

```math
Z = 10^{\mathrm{dBZ}/10}, \qquad
\mathrm{DCOR} = \left(\bar\rho(0)/\bar\rho(z)\right)^{p}
```

```math
V_R = -c_{\text{rain}} Z^{e_{\text{rain}}}\,\mathrm{DCOR}
\quad\text{(Joss \& Waldvogel 1971)}, \qquad
V_S = -c_{\text{snow}} Z^{e_{\text{snow}}}\,\mathrm{DCOR}
\quad\text{(Atlas et al. 1973)}
```

with the density exponent `p` from Beard (1985). Below `mixed_phase_dbz` the
above-melting-layer speed is pure snow, above `rain_dbz` pure rain, linear
in dBZ between the two; across the melting layer (bottom = the configured
value or the reference profile's freezing level, over
`melting_layer_depth_m`) the fall speed blends linearly in height from the
rain relation to the above-layer value, continuous in both dBZ and height.
The corrected gate value entering the normal system is `\tilde v = v_r -
w_t\sin(\mathrm{el})`.

**Per-gate error** (`gate_sigma`, samurai
`DopplerError`):

```math
\sigma_m = \max\left(\sigma_{\min},\ |\sin\mathrm{el}|\cdot\sigma_{fs} + \mathrm{SW}\cdot c_{sw}\right)
```

folded into the gridding weight as `w' = w \cdot \sigma_{vr}^2/\sigma_m^2` —
the identity that keeps the stage-1 normal system, its eigen-split, and the
sandwich covariance formula valid: setting `σ_m² = σ_vr²` everywhere
(disabled) recovers `w' = w` exactly, bit-identical to the uncorrected path.

## 6. Anelastic mass continuity

Mass continuity is enforced as a **pseudo-observation** of value 0 at every
mish point (`MassContinuityObs`):

```math
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z}
+ w\,\frac{d\ln\bar\rho}{dz} = 0, \qquad \sigma = \texttt{mass\_continuity\_sigma\_per\_s}
```

terms `(u, ∂x, 1)`, `(v, ∂y, 1)`, `(w, ∂z, 1)`, and `(w, val, dlnρ̄/dz)`
(omitted where `dlnρ̄/dz = 0`, e.g. a neutrally-stratified test profile).
Because it is a pseudo-observation of a spline *derivative* rather than a
finite-difference or vertically-integrated constraint, mass continuity
couples every mish point through the same `H_k C` machinery as any other
observation — this is the advantage samurai gets from formulating continuity
this way rather than integrating `w` from a divergence profile. `dlnρ̄/dz`
comes from the configured `[reference_state]` profile
([`dlnrho_dz`](@ref Daisho.dlnrho_dz)), evaluated once per k-level.

## 7. The Gauss–Newton outer loop

```math
\xi_{\text{total}} = 0
```
```math
\text{for } k = 1 \dots \texttt{max\_outer\_iterations}:
```
```math
a_k = a_g + C\,\xi_{\text{total}}, \qquad d_k = y - h(a_k)
\quad\text{(relinearise nonlinear obs at } a_k\text{)}
```
```math
\text{rhs} = C^\top H_k^\top R\, d_k - \xi_{\text{total}} = -\nabla J(\xi_{\text{total}})
```
```math
\text{CG:}\quad (I + C^\top H_k^\top R H_k C)\,\delta\xi = \text{rhs}, \qquad x_0 = 0
```
```math
\xi_{\text{total}} \mathrel{+}= \delta\xi
```

(`relinearize!`, `solve_incremental!`). `rhs` is exactly the negative gradient of the
**true, possibly nonlinear** cost `J` at `ξ_total`, evaluated with the true
residual `d_k = y - h(a_k)` (not the quadratic model's residual), while the
CG solve uses the Gauss–Newton Hessian at the current linearisation. Each
outer iteration is therefore a full Gauss–Newton step on the *exact*
nonlinear problem.

**For a fully linear observation set, one outer iteration is exact.** `H_k`
does not depend on `a_k` when every observation is linear, so the Hessian is
the exact (not approximate) Hessian of the quadratic `J`, and solving
`\mathrm{Hess}\cdot\delta\xi = \text{rhs}` finds its exact minimiser in one
step; a second outer iteration relinearises at that minimiser and finds
`\delta\xi \approx 0` (the gradient there is already zero, up to the CG
tolerance) — a property exercised directly by the test suite
(`max_outer=2` giving `‖δξ₂‖ ≪ ‖δξ₁‖`).

**`ξ_total` bookkeeping and the samurai inconsistency.** samurai's outer
loop added each increment to the *background state itself* and reset its
control variable to zero for the next iteration. That conflates two
different reference points across iterations: the `½‖ξ‖²` control-space
penalty of iteration `k+1` is then computed relative to the *already
adjusted* background, silently discarding the control cost already paid for
the part of the increment accepted in iteration `k`. Daisho never resets:
`a_k` is always computed from the original background `a_g` plus the *full*
accumulated `ξ_total`, and the control penalty `½‖ξ_total‖²` inside `J` is
therefore always evaluated consistently against the same reference. Only
`H_k` (the *linearisation point* of the observation operator) changes across
outer iterations, not the meaning of `ξ`.

## 8. The Hessian and conjugate gradient

```math
\mathrm{Hess} = I + C^\top H_k^\top R H_k C
```

is symmetric positive definite with eigenvalues `≥ 1` (the identity
dominates), so `HessianOperator` (a
matrix-free `Base.size`/`Base.eltype`/`LinearAlgebra.mul!` wrapper around
`hessian_mul!`) is solved with `Krylov.cg!`
(Krylov.jl), reusing one `Krylov.CgWorkspace` across outer iterations, `x0 =
0`, relative tolerance `inner_tolerance`, iteration cap
`max_inner_iterations`. A single Hessian product is one forward-and-adjoint
pass through `C` and one through the observation set — no matrix is ever
formed. **Conditioning** is driven by `(\sigma_b/\sigma_o)^2 \sum w` (large
background σ or dense, low-error observation coverage steepens the
quadratic term relative to the identity); a preconditioner slot (`M =` in
`VarSolverOptions`, Krylov's convention —
`mul!` for an inverse-Hessian approximation, `ldiv!` for a
`Factorization`) is exposed but untested and unset by default (`nothing`).

## 9. Background modes

`[variational] background`:

- `"zero"` (default, samurai-like): the background is 0 everywhere for every
  state variable, with σ_b from `[variational.background_error]`.
- `"synthesis"`: `U`, `V` take the stage-1 `(u, v)` analysis
  (`stage1.comp1`/`comp2`) wherever `quality_flag == 0`, else 0; the
  background σ is `FieldSigma(background_error_inflation ·
  stage1.sigma1/sigma2)`, floored at the configured σ where the stage-1
  point was flagged (so a masked point falls back to the config prior
  rather than an unreliable stage-1 σ). `W` always comes from
  `background_error`, since stage 1 has no `w`.

**Double-counting caveat.** The `"synthesis"` background and the
radial-velocity super-obs term (§4) are built from the *same* underlying
gates: the super-obs are the eigen-split of the very normal system stage 1
solved to produce the background. Using both without correction lets that
data count twice — once as the prior mean, once as the likelihood.
`background_error_inflation` (multiplying σ_b, hence *weakening* the
background term relative to the always-present radar likelihood) is the
guard against this; it does not eliminate the double count, it trades it off
explicitly. The `"zero"` mode has no such coupling — its background carries
no information from the current volume at all.

## 10. Differences from samurai

- **Observation weights are `1/σ²`, not `1/σ`.** samurai's cost function
  used inverse-σ weights (a legacy of its normalization); Daisho uses the
  standard `1/σ²` inverse variance throughout. Every ported σ value must be
  re-derived, not copied.
- **Super-observations, not individual gates.** samurai's cost function
  visits every gate; Daisho visits the stage-1 accumulator's per-point
  eigen-split (§4), which reproduces the identical weighted least-squares
  cost and gradient at a fraction of the observation count, and for free
  from an accumulation Daisho already performs.
- **Physical `(u, v, w)`, not ρ-weighted momentum.** State winds are
  physical velocities; continuity carries the `w\,d\ln\bar\rho/dz` term
  explicitly rather than folding it into a ρ-weighted state variable. A
  transform hook (`AbstractStateTransform`) is reserved for a future density-weighted
  or log-transformed state without touching the operator layer.
- **Continuity on the mish, not at spline nodes.** The pseudo-observation is
  evaluated at every quadrature point (via `H`, the same tensor-product
  spline evaluation as every other observation), not at the coefficient
  nodes samurai used.
- **Filter lengths in metres, and `√2` effective correlation length.** §2.1;
  samurai configured filter lengths in grid units and its recursive filter's
  effective correlation length equals its configured σ (a symmetric filter);
  Daisho's non-symmetric, exact-adjoint `F` has an effective `B = FF^\top`
  correlation length of `√2` times the configured (cell) length.
- **No storm-motion advection term.** Out of scope for this phase (plan
  §1); a single synthesis time is assumed, as in stage 1.
- **Flexible state vector**, not samurai's fixed seven variables:
  `StateVector` is built from `[variational]
  state`, any `[fields]` name may be appended as a directly-observed scalar
  state variable (§ guide, "Adding a scalar to the state").
- **Multiple dispatch over observation types**, not a hard-coded switch:
  `AbstractObservation` subtypes
  (`RadialVelocitySuperObs`,
  `MassContinuityObs`,
  `ScalarValueObs`, `PointObs`, `WindSpeedObs`) each supply
  `obs_terms`/`model_value`/`linearize`; adding an observation type is a new
  subtype, not a new branch in existing code.

## 11. Future extensions

- **Thermodynamic/microphysical state.** The role vocabulary
  ([`VARIATIONAL_ROLE_VOCAB`](@ref Daisho.VARIATIONAL_ROLE_VOCAB)) already
  reserves roles beyond wind and reflectivity, and the transform hook
  (§10) is the intended seam for a density-weighted or log-transformed
  thermodynamic variable — no redesign of `StateVector` or the operator
  layer is anticipated.
- **Terrain / immersed-boundary pseudo-observations.** `PointObs` at arbitrary `XYZ` locations already exists; a terrain
  mask would add a group of zero-value pseudo-observations of the
  wind-normal component at terrain-following points, out of scope for this
  phase.
- **Gate-level observations.** `ScatteredObsBackend` supports arbitrary off-mish locations today
  (used for `PointObs`); a per-gate radial-velocity
  observation type would reuse it directly instead of the super-obs
  reduction, at the cost of one basis stencil per gate.
- **Cross-variable / flow-dependent B.** `D` and `F` are currently diagonal
  and per-variable; a flow-dependent or cross-variable background-error
  covariance would replace `D` (or add a cross term to `C`) without
  changing the cost function, Hessian, or solver.
- **Preconditioning.** The `M =` hook on `VarSolverOptions` is wired to `Krylov.cg!` but unset by default;
  a physically-motivated preconditioner (e.g. a coarse-grid or diagonal
  approximation of `\mathrm{Hess}`) is the natural way to reduce the 100–300
  CG iterations per outer loop the reference grid size implies (plan §7).

## References

- Dennis, J. M., Baker, A. H., Dobbins, B., Bell, M. M., Sun, J., Kim, Y., &
  Cha, T.-Y. (2022). Enabling efficient execution of a variational data
  assimilation application. *The International Journal of High Performance
  Computing Applications*. DOI: 10.1177/10943420221119801
- Ooyama, K. V. (2002). The cubic-spline transform method: Basic definitions
  and tests in a 1D single domain. *Monthly Weather Review*, 130(10),
  2392–2415.
- Miller, L. J., & Anderson, S. M. (1991). *CEDRIC — Cartesian Space Data
  Processor*, Appendix F (as cited by the [stage-1 theory page](wind_synthesis.md)).
- Joss, J., & Waldvogel, A. (1971). Raindrop size distribution and sampling
  size errors. *Journal of the Atmospheric Sciences*, 27, 566–569 (rain
  fall-speed relation).
- Atlas, D., Srivastava, R. C., & Sekhon, R. S. (1973). Doppler radar
  characteristics of precipitation at vertical incidence. *Reviews of
  Geophysics and Space Physics*, 11, 1–35 (snow fall-speed relation).
- Beard, K. V. (1985). Simple altitude adjustments to raindrop velocities
  for Doppler radar analysis. *Journal of Atmospheric and Oceanic
  Technology*, 2, 468–471 (density correction exponent).
