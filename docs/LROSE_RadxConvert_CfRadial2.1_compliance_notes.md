# CfRadial 2.1 Compliance Notes — LROSE RadxConvert Output

**Reporter:** Michael M. Bell &lt;mmbell@colostate.edu&gt; (Daisho.jl maintainer,
Colorado State University)
**Date prepared:** 2026-05-06
**Spec assessed against:**
[CfRadial Data File Format v2.1 DRAFT, 2019-09-01](https://github.com/NCAR/CfRadial/blob/master/docs/CfRadialDoc-v2.1-20190901.pdf)
**LROSE component:** RadxConvert (verified against the 2026-04-25 release;
exact build string from sample file: `driver = "RadxConvert(NCAR)"`,
`history` line indicates `Cf2RadxFile` writer class). Findings below also
reproduced on a slightly earlier build — the only diff between the two
builds is a CF-conformant casing fix on calibration `units` attrs from
`"db"` to `"dB"` (a useful improvement, not on the list below).

---

## 1. Context

We are implementing a CfRadial reader/writer in
[Daisho.jl](https://github.com/csu-tropical/Daisho.jl) (a Julia radar processing
package). To validate the implementation against real-world v2 files, we
audited a CfRadial2 file produced by the latest LROSE RadxConvert. We chose
the spec as our internal model of truth and decided to canonicalize any
on-disk divergences in our reader. This doc is to help the broader LROSE team would find
address RadxConvert's spec conformance in a future
release. We have nothing but appreciation for the spec and the LROSE toolset; please
treat this as constructive feedback rather than a defect list.

## 2. Sample file

The findings below are from a SEA-POL surveillance volume (CSU shipboard
C-band dual-pol radar) processed through SIGMET → RadxConvert (CfRadial 1.4
intermediate) → RadxConvert (CfRadial 2.x output).

| | |
|---|---|
| Filename | `cfrad2.20240903_150007.042_to_20240903_150444.596_SEAPOL_PICCOLO_CIRC_SUR.nc` |
| Time coverage | 2024-09-03 15:00:07Z – 15:04:44Z |
| Instrument | SEA-POL (C-band, ship-mounted, dual-pol) |
| Scan type | Surveillance, 11 elevation sweeps |
| Sweep size | 248 rays × 2447 range gates each |
| Field count | 29 (DBZ, ZDR, KDP, RHOHV, VEL, WIDTH, PHIDP, SQI, SNR, plus L2 / NNC / ATTEN_UNCORRECTED variants and rate / hydrometeor products) |
| File header version tag | `:version = "2.0"` (see §3 item 1) |

A corresponding CfRadial 1.4 file from the same volume is also available; we
cross-checked structural equivalence between the two during reader
development.

## 3. Divergences from CfRadial 2.1 spec

Items are grouped by area. Each row lists the spec reference, what we
observed in the file, and a suggested resolution. None of the items
invalidate the file's usefulness — our reader handles all of them — but
each is a place where the file as written would not pass a strict spec
check.

### 3.1 Root group — global attributes

| # | Concern | Spec (§) | File has | Suggested action |
|---|---------|----------|----------|------------------|
| 1 | `version` global attribute value | §1.4 history table — v2.0 was deprecated 2019-09-01 in favour of v2.1 | `:version = "2.0"` | Bump to `"2.1"` once the implementation matches v2.1. Even if the structure is intentionally still 2.0-flavoured, the deprecation warrants documentation. Note: downstream readers should not rely on this string for behaviour; we ignore it and dispatch on structure. |
| 2 | `time_coverage_start` global attribute | §4.1 table — struck through; spec moved this to a global *variable* in §4.3 | present (and the variable is also present) | Drop the global attribute; keep the variable. |
| 3 | `time_coverage_end` global attribute | §4.1 — struck through | present (and the variable is also present) | Drop the attribute; keep the variable. |
| 4 | `start_time` / `end_time` global attributes | not in v2 spec (vestigial v1) | present | Drop. |
| 5 | `original_format`, `driver`, `created` global attributes | not in §4.1 | present | These are useful provenance metadata. Either propose them as ODIM-style optional extensions to §4.1 (with explicit type/description), or document that they are LROSE conventions that downstream tools may safely ignore. |
| 6 | `Conventions` value | spec §4.1 says `"Cf/Radial"` | matches | OK |

### 3.2 Root group — global variables

| # | Concern | Spec (§) | File has | Suggested action |
|---|---------|----------|----------|------------------|
| 7 | `time_coverage_start` / `time_coverage_end` variable type | §4.3 table says `double` with units "seconds since YYYY-MM-DD HH:MM:SS" or "seconds since YYYY-MM-DDThh:mm:ssZ" | stored as `string` (NC_STRING) | The spec is internally inconsistent here: the `units` field of §4.3 is itself an ISO 8601 string (literally `"seconds since YYYY-MM-DD..."`) which suggests numeric encoding, but the *value* commonly stored — and what the file holds — is the ISO 8601 timestamp itself. We suggest the spec be clarified: either (a) require `double` with a numeric reference epoch and units = "seconds since &lt;epoch&gt;", or (b) require `string` and drop the `units` confusion. The LROSE choice (string) is the more practically useful one. |
| 8 | `status_str` root variable | §4.3 table — name is `status_str` | named `status_xml` | Rename to `status_str` for spec compliance. (The XML payload itself is fine; only the variable name differs.) |

### 3.3 Sweep group

| # | Concern | Spec (§) | File has | Suggested action |
|---|---------|----------|----------|------------------|
| 9 | Per-ray calibration index variable name | §5.3 table — `calib_index(time)` | `r_calib_index(time)` | Rename to `calib_index`. |
| 10 | Range start / gate spacing as scalar variables | §5.2.2 lists these as **attributes** of the `range` coordinate variable: `meters_to_center_of_first_gate`, `meters_between_gates` | both attributes are present (good) AND the file additionally defines scalar variables `start_range` and `ray_gate_spacing` | The attributes are sufficient and are what the spec requires. The scalar variables are redundant and not in the spec. Suggest dropping the scalars. |
| 11 | `georefs_applied` location | §5.4 table places `georefs_applied(time)` inside the `georeference` sub-group | placed at sweep-group level (alongside `azimuth`, `elevation`, etc.) | Move into the `georeference` sub-group. |

### 3.4 Calibration sub-group

| # | Concern | Spec (§) | File has | Suggested action |
|---|---------|----------|----------|------------------|
| 12 | Calibration dimension name | §7.3.1 — `calib` | `r_calib` | Rename to `calib`. |
| 13 | Reflectivity-at-1km variable names | §7.3.2 — `base_1km_hc`, `base_1km_vc`, `base_1km_hx`, `base_1km_vx` | `base_dbz_1km_hc/vc/hx/vx` | Rename to spec form. |
| 14 | `dbz_correction` calibration variable | not listed in §7.3.2 (spec lists `zdr_correction`, `ldr_correction_h`, `ldr_correction_v`, `system_phidp`, but not a generic dBZ correction) | present | Either add `dbz_correction` to the §7.3.2 table (with units "dB") or drop from the writer. We suspect this is a useful variable that simply was never added to the spec table. |
| 15 | `calibration_time(r_calib)` variable | §7.3.2 lists `time(calib)` (note: no prefix) | name is `calibration_time`; type is `string` | Spec name is `time`; we suggest the writer use that for symmetry with the sweep-level `time` variable, even though it sits in a different group. |

### 3.5 Monitoring sub-group

| # | Concern | Spec (§) | File has | Suggested action |
|---|---------|----------|----------|------------------|
| 16 | Sub-group name | §3 fig 3.2 and §5.5 — `radar_monitoring` (or `lidar_monitoring` for lidars) | `monitoring` (no prefix) | Rename to `radar_monitoring`. The spec's prefixed name disambiguates sensor types in mixed deployments. |

### 3.6 Georeference sub-group

| # | Concern | Spec (§) | File has | Suggested action |
|---|---------|----------|----------|------------------|
| 17 | Extra `georef_time(time)` variable | not in §5.4 table | present | Either add to §5.4 (rationale: useful when georef arrives on a different cadence than radar rays) or drop from writer. |
| 18 | `georef_unit_num`, `georef_unit_id` variables | not in §5.4 table | present | Same as #17 — propose for inclusion or drop. |

### 3.7 Field-level metadata

| # | Concern | Spec (§) | File has | Suggested action |
|---|---------|----------|----------|------------------|
| 19 | Use of `_Undetect` attribute | §5.6 lists `_Undetect` (ODIM convention) as a way to encode *gates that were scanned but produced no valid echo* — distinct from `_FillValue` (true missing) | absent on every field; the SEA-POL convention encodes this distinction by using **different `_FillValue` values** on different fields (-9999 for clear-air, -32768 for true missing) | We strongly support adopting `_Undetect` end-to-end. The current per-field-fill-value convention is fragile and easy to misread: a downstream reader has to know that "if the fill value is -9999 it means clear air and not missing" — which is impossible to infer from the file alone. Adopting `_Undetect` makes the distinction self-describing per spec. |
| 20 | `coordinates` attribute value | §5.6.2 — for stationary platforms `"elevation azimuth range"`; for mobile `"elevation azimuth range heading roll pitch rotation tilt"` | `"time range"` | The spec mandates the spatial coordinates here. The file uses temporal axis names instead. Recommend correcting to spec form. |
| 21 | `grid_mapping` attribute | spec uses `radar_lidar_radial_scan` per §3.5 | references `grid_mapping` (the variable name); `grid_mapping` itself has `grid_mapping_name = "azimuthal_equidistant"` | The spec defines a CfRadial-specific `radar_lidar_radial_scan` grid mapping (§3.5) for radial-coordinate data. The file uses the more generic `azimuthal_equidistant`. This is debatable — `azimuthal_equidistant` is a real CF projection and may be more interoperable with non-radar tooling — but a spec-compliant writer should output `radar_lidar_radial_scan`. |

### 3.8 Spec ambiguities encountered

These are not LROSE issues; they are points where the v2.1 spec itself is
unclear. Flagging in case the team finds them useful.

- **§4.3 `time_coverage_start` type vs units** — see item #7 above.
- **§3.5 grid mapping** — the spec introduces `radar_lidar_radial_scan` but
  notes "there is no corresponding projection in PROJ.4". A writer guidance
  note ("when in doubt, use the radial scan mapping; tools that need a
  PROJ-compatible projection should add an additional `grid_mapping` var")
  would help.
- **§5.6 `_FillValue` vs `_Undetect`** — the spec mentions `_Undetect`
  briefly but does not state whether a writer SHOULD emit it for every
  field with a "no echo" possibility, or whether `_FillValue` alone is
  acceptable. A SHOULD/MUST recommendation here would make conformance
  testable.

## 4. Cross-reference: corresponding v1.4 file

The same volume is also available as a CfRadial 1.4 file (output of the
preceding RadxConvert step from SIGMET). For interest, that file has:

- `Conventions = "CF-1.7"`,
  `Sub_conventions = "CF-Radial instrument_parameters radar_parameters radar_calibration platform_velocity"`,
  `version = "CF-Radial-1.4"`. ✓ as expected.
- `field_names` global attribute present (struck-through in v2.1 §4.1, but
  a v1.4 fixture, so this is correct).
- All instrument / calibration metadata is at root level with the v1.4
  `meta_group` attribute pointing to logical groupings — i.e. the v1.4
  pattern that v2 lifts into actual NetCDF4 groups.

The v1.4 file appears spec-conformant. No comparable issue list to report
for that side.

## 5. What we did about it in Daisho.jl

For reference and in case it's useful to other downstream tooling:

- We adopted the v2.1 spec as our **internal data model**, irrespective of
  on-disk format.
- Our reader auto-detects v1.4 vs v2 (groups present → v2; otherwise v1).
- The reader **canonicalises** every divergence in §3 above to the spec
  form when populating the in-memory volume. Truly non-spec extras are
  preserved in `extra_attrs` / `extra_vars` dictionaries for round-trip.
- Our full writer emits spec-canonical CfRadial 2.1 only. LROSE-isms in
  `extra_*` are dropped unless the caller opts in.
- Our format-preserving "modify in place" path (`update_cfradial`) keeps
  the input file's layout — so a v1.4 archive stays v1.4, and an LROSE-v2
  archive keeps its LROSE-isms.

This means a Daisho-written file should pass the strict spec check; a file
read in via Daisho and re-written through `update_cfradial` will preserve
the source's quirks.

## 6. Contact

Happy to discuss any of the above, share the test fixtures, or contribute
fixes to LROSE if the team would find that helpful.

— Michael Bell, mmbell@colostate.edu
