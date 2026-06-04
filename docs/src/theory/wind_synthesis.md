# Multi-Doppler Wind Synthesis (Stage 1)

Stage 1 is a gridpoint-independent, overdetermined, weighted least-squares
**dual-Doppler horizontal wind retrieval** `(u, v)`, streamed through an
accumulator that mirrors the [`GridAccumulator`](@ref) model and carries a
preserved per-component uncertainty. It follows CEDRIC appendix F (Miller &
Anderson, 1991), §1, "Cartesian components of motion from radial velocities".

It is **not** vertical velocity, mass continuity, the variational solution,
fallspeed/advection correction, dealiasing, or spectral-grid integration — those
are stage 2 / upstream QC / follow-ons.

## Observation equation

For each contributing gate `m` (from **any** radar or sweep) reaching a grid
point, with the vertical term dropped (`W` neglected — the two-equation
dual-Doppler approximation):

```math
v_{r,m} \approx u\, a_m + v\, b_m, \qquad
a_m = \sin(\mathrm{az}_m)\cos(\mathrm{el}_m), \qquad
b_m = \cos(\mathrm{az}_m)\cos(\mathrm{el}_m)
```

where ``\mathrm{az}_m`` is the azimuth clockwise from +y (true north) and
``\mathrm{el}_m`` the refraction-corrected elevation. Both are the **effective**
line-of-sight angles from the gate's radar origin to the grid point — exactly the
`gridpt_az`/`gridpt_el` the shared gate-geometry kernel already computes for the
scalar gridding path. (The deferred three-unknown extension restores the third
column ``c_m = \sin(\mathrm{el}_m)``, the coefficient of `W`.)

## Weighted least squares → streamed normal system

Minimizing ``\sum_m w_m (u\,a_m + v\,b_m - v_{r,m})^2`` with the existing
gridding weight ``w_m = \text{range\_weight}\cdot\text{angle\_weight}``, each gate
contributes a **rank-1 update** to a per-gridpoint normal system. Writing
``g_m = [a_m, b_m]``:

```math
A^\top W A \mathrel{+}= w_m\, g_m g_m^\top, \qquad
A^\top W b \mathrel{+}= w_m\, v_{r,m}\, g_m, \qquad
M_2 \mathrel{+}= w_m^2\, g_m g_m^\top
```

No per-radar pre-averaging and no retention of individual gate rows: multi-radar
synthesis is simply continuing to grid more sweeps into the same accumulator. All
updates are in the **Cartesian** basis. The symmetric ``A^\top W A`` and ``M_2``
are stored packed (`[S_aa, S_ab, S_bb]`).

## Per-gridpoint solve and covariance

At finalize, each grid point is solved independently:

```math
D = S_{aa}S_{bb} - S_{ab}^2, \qquad
C_i = (A^\top W A)^{-1} = \frac{1}{D}\begin{bmatrix} S_{bb} & -S_{ab} \\ -S_{ab} & S_{aa}\end{bmatrix},
\qquad \begin{bmatrix}u\\v\end{bmatrix} = C_i \begin{bmatrix}S_{av}\\S_{bv}\end{bmatrix}
```

The covariance is the equal-variance **"sandwich"**, with ``M_2`` the second
accumulated matrix:

```math
\mathrm{cov} = \sigma_{vr}^2\, C_i\, M_2\, C_i
```

Each gate's radial velocity is modeled with the same variance ``\sigma_{vr}^2``
(diagonal, geometry-independent — the standard "1 m/s, diagonal" assumption); the
beam-power×range weights are interpolation/representativeness weights, not a noise
model. ``\sigma_{vr}^2`` (`velocity_variance`) is a decoupled scalar that only
rescales to physical units; setting it to 1 gives normalized (CEDRIC USTD/VSTD)
σ. At **equal gate weights** ``M_2 = A^\top W A`` and ``\mathrm{cov} =
\sigma_{vr}^2 C_i``, recovering the unweighted CEDRIC Eq-13 ``\sum g^2`` form. As
the grid point approaches the baseline ``D \to 0`` and the covariance blows up —
the direct baseline-instability detector.

## Output frame

The Cartesian solution is rotated into an active orthonormal **output frame**
``R = `` [`rotation_at`](@ref)`(frame, x, y, z)`:

```math
\begin{bmatrix}c_1\\c_2\end{bmatrix} = R\begin{bmatrix}u\\v\end{bmatrix},
\qquad \mathrm{cov}_{\text{frame}} = R\,\mathrm{cov}\,R^\top,
\qquad \sigma_i = \sqrt{(\mathrm{cov}_{\text{frame}})_{ii}}
```

Because `R` depends only on the grid point (not the gate), it factors out of every
accumulated sum, so **accumulation stays Cartesian** and the frame is a pure
finalize-time view — one accumulator can be finalized in several frames without
re-gridding. Nothing hardcodes U=zonal/V=meridional. Stage 1 ships only the
identity [`CartesianFrame`](@ref) (`(c_1,c_2)=(u,v)`); plane-aligned (RHI),
polar/cylindrical (hurricane), and 3D coplane frames are pure additions later.

## Quality control (non-destructive)

Quality is judged **solely** by the per-component normalized uncertainty
(CEDRIC DTEST style), evaluated **in the output frame**, against the independent
thresholds `max_sigma[1]`/`max_sigma[2]` — one component can be well-determined
while the other is not. There is no radar-count and no gate-count threshold: the
adequacy of the geometry (how many radars, crossing angle, baseline proximity) is
already expressed in σ.

Masking is **non-destructive**: where the system is solvable, `comp1, comp2,
sigma1, sigma2` are always written, and a separate `quality_flag` records which
σ threshold(s) failed — so stage 2 can recover those points using global
information. Two intrinsic (non-tunable) states blank the components to
`fill_value`:

| Flag bit | Meaning |
|---------|---------|
| `0x01` | σ₁ above `max_sigma[1]` (in-frame) |
| `0x02` | σ₂ above `max_sigma[2]` (in-frame) |
| `0x04` | **singular**: ≥1 gate but degenerate geometry (single look / baseline) |
| `0x08` | **no data**: no weighted gate reached the point |

The singular guard is scale-relative (``D \le \varepsilon\, S_{aa}S_{bb}``): a
numerically rank-1 single-look system is caught (rather than producing a
falsely-confident ``(0,0)`` with ``\sigma = 0``), while a *physically*
near-baseline pair keeps a positive ``D`` and survives, its instability
surfacing as a large σ caught by the thresholds.

## Stage-2 handoff

The [`WindGridAccumulator`](@ref) is the stage-2 handoff artifact: its
`AtWA`/`AtWb`/`M2`/`weight_total`/counts are exactly what a variational stage
needs to recover masked baseline points. [`finalize_wind`](@ref) produces the
single-stage *product*; the accumulator is the richer *input* for stage 2.
Because accumulation is linear, [`merge_wind_accumulators!`](@ref) across files
and radars is an exact elementwise sum.
