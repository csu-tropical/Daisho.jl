# Test fixtures

This directory holds NetCDF radar files used by the integration tests in
`test/test_cfradial_*.jl`. The files are large (~50–60 MB each) and are not
committed to the repository.

Tests that require these fixtures check for them at runtime and **skip
gracefully** if they are absent — the rest of the suite continues to run.

## Required fixtures

For full reader/writer/bridge coverage, place the following SEAPOL CfRadial
files in this directory:

- `cfrad.20240903_150007.042_to_20240903_150444.596_SEAPOL_SUR.nc`
  — CfRadial v1.4 (LROSE-converted from a SIGMET RAW file).
- `cfrad2.20240903_150007.042_to_20240903_150444.596_SEAPOL_PICCOLO_CIRC_SUR.nc`
  — CfRadial v2 (LROSE `RadxConvert` output from the v1 file above).

Both cover the same volume: 11 sweeps × 2447 range gates, with the field set
documented in §2.1 of the CfRadial 2.1 refactor plan
(`CFRADIAL_REFACTOR_PLAN.md`).

## Where to obtain them

These are private SEA-POL volumes from PICCOLO. Contact the maintainer
(`mmbell@colostate.edu`) for access. Long term, we expect to either
host them on Zenodo with a download script, or move them into Git LFS.

## Test artifacts

Tests may create transient `.nc` files in this directory (e.g.
`test_orientation.nc`, `test_cfradial.nc`) and remove them on teardown.
The `*.nc` glob in `.gitignore` keeps those out of git as well.
