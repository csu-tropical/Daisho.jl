# Reference State

A hydrostatic reference-density profile ρ̄(z), configured through the optional
`[reference_state]` block and surfaced as
[`ReferenceStateParameters`](@ref Daisho.ReferenceStateParameters) on the
loaded [`DaishoParameters`](@ref). Shared by consumers that need a base-state
density or temperature profile: variational-stage anelastic mass continuity,
the radar fall-speed correction, and (a future phase) FHC temperature. See the
[Reference State guide](../guide/reference_state.md) for the sounding format
and worked configuration.

```@docs
Daisho.ReferenceStateParameters
Daisho.AbstractReferenceProfile
Daisho.ExponentialReferenceProfile
Daisho.SoundingReferenceProfile
Daisho.reference_density
Daisho.dlnrho_dz
Daisho.reference_temperature_kelvin
Daisho.freezing_level_m
Daisho.build_reference_profile
```
