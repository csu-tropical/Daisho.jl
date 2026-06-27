# Echo Products

Post-gridding products derived from gridded radar fields: fuzzy-logic
hydrometeor identification (FHC), rain-rate retrievals, and the temperature
profile they draw on. These live under the `[echo]` configuration namespace and
are driven through [`EchoProductsParameters`](@ref Daisho.EchoProductsParameters).

## Echo Products Driver

```@docs
Daisho.EchoProductsParameters
Daisho.apply_echo_products
Daisho.add_echo_products!
Daisho.echo_output_names
```

## Fuzzy Hydrometeor Classification

```@docs
Daisho.csu_fhc_summer
Daisho.get_mbf_sets_summer
Daisho.beta_mbf
Daisho.FHC_SUMMER_CLASSES
Daisho.FHC_N_TYPES
Daisho.DEFAULT_FHC_WEIGHTS
```

## Rain Rate

```@docs
Daisho.calc_blended_rain_tropical
Daisho.calc_rain_zr
Daisho.calc_rain_kdp
Daisho.calc_rain_kdp_zdr
Daisho.calc_rain_z_zdr
```

### Rain-Rate Coefficients

```@docs
Daisho.RAIN_RZ_ALL
Daisho.RAIN_RZ_CONV
Daisho.RAIN_RZ_STRAT
Daisho.RAIN_BAND_COEFFS
```

## Temperature Profile

```@docs
Daisho.TemperatureProfile
Daisho.read_temperature_profile
Daisho.temperature_celsius
```
