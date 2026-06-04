# Multi-Doppler Wind Synthesis

Daisho's stage-1 wind synthesis retrieves the horizontal wind `(u, v)` from the
radial velocities of two or more radars, by streaming each sweep into a
per-gridpoint weighted least-squares normal system and solving each grid point
independently. See [the theory page](../theory/wind_synthesis.md) for the math.

## Workflow

The accumulator mirrors the [multi-sweep gridding accumulator](gridding.md):
allocate once, grid sweeps from any radars into it, then finalize.

```julia
using Daisho

p = DaishoParameters("mygrid.toml")          # needs a :velocity-tagged field
                                             # and a [synthesis] block

# A Cartesian grid the wind is solved on.
spec = build_grid_spec(:volume_3d, volume_radarA, p)

# One accumulator collects contributions from every radar/sweep.
acc = WindGridAccumulator(spec, p)

for s in eachindex(volume_radarA.sweeps)
    grid_sweep_wind!(acc, volume_radarA, s, p)
end
for s in eachindex(volume_radarB.sweeps)
    grid_sweep_wind!(acc, volume_radarB, s, p)
end

# Solve every grid point, with the default Cartesian (U, V) output frame.
out = finalize_wind(acc, p)

# Write a CF-1.12 gridded NetCDF: U, V, USTD, VSTD, DET, NGATES, QFLAG.
write_wind_synthesis("wind.nc", out, p; index_time = volume_radarA.time_coverage_start)
```

There is **no radar-index argument** — nothing about the synthesis depends on
which radar a gate came from. Sweeps from different radars are gridded into the
same accumulator exactly as sweeps from one radar are.

## Configuration

The `[synthesis]` block is optional (it default-constructs) and controls only the
QC, which is the per-component uncertainty:

```toml
[synthesis]
velocity_variance = 1.0   # assumed σ_vr² (m/s)²; 1.0 → normalized σ units
max_sigma_1       = 2.0   # max normalized σ on output component 1 (U)
max_sigma_2       = 2.0   # max normalized σ on output component 2 (V)
```

The two thresholds are **independent** — a geometry that determines one component
well but not the other flags only the poor one. There is no radar-count or
gate-count threshold; geometry adequacy already shows up in σ.

The velocity field is the one tagged `velocity` in `[fields]`:

```toml
[fields]
VEL = ["weighted_interp", "velocity"]
```

## Output and quality flags

[`finalize_wind`](@ref) returns a [`SynthesisOutput`](@ref) with the two wind
components, their per-component σ, the normal-equation determinant `det`, the
contributing-gate count `n_gates`, and a `quality_flag`. Masking is
**non-destructive**: wherever the system is solvable, the components and σ are
always written; the flag only records which σ threshold failed, so a later
stage can still use those points.

| `quality_flag` bit | Meaning |
|--------------------|---------|
| `0` | solvable, both σ within thresholds |
| `0x01` | σ₁ above `max_sigma_1` |
| `0x02` | σ₂ above `max_sigma_2` |
| `0x04` | singular geometry (single look / baseline) — components blanked |
| `0x08` | no data reached this point — components blanked |

## Persistence and multi-file merge

Because the accumulation is linear, accumulators can be persisted per file and
combined later — gridding sweeps A and B into one accumulator equals gridding
them separately and merging:

```julia
save_wind_accumulator("radarA.jld2", accA)
save_wind_accumulator("radarB.jld2", accB)

dst = load_wind_accumulator("radarA.jld2")
merge_wind_accumulators!(dst, load_wind_accumulator("radarB.jld2"))
out = finalize_wind(dst, p)
```

## Output frames

The wind is expressed in a pluggable orthonormal output frame. Stage 1 ships the
identity [`CartesianFrame`](@ref) (components `U`, `V`); because accumulation is
Cartesian and frame-agnostic, the frame is applied only at finalize, so one
accumulator can be finalized in several frames:

```julia
out_uv = finalize_wind(acc, p)                          # CartesianFrame (U, V)
# Future: finalize_wind(acc, p; frame = PolarFrame(center))  # tangential/radial
```

## Assumptions

- Input velocity is **dealiased** and uses the positive-away-from-radar
  convention (dealiasing is upstream QC, not stage 1).
- A single synthesis time (no advection); inputs should be reasonably
  contemporaneous.
- Reflectivity/fallspeed is not used (`W` neglected).
- A gate's own velocity validity (non-missing value) gates its participation,
  independent of the `define_detection`/`define_scanned` roles.
