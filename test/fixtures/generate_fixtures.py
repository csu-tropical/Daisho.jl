#!/usr/bin/env python3
"""Generate reference fixtures from CSU_RadarTools for verifying the Daisho.jl port.

Runs the original Python `csu_fhc_summer`, the four component rain-rate relations,
and `calc_blended_rain_tropical` over seeded, representative input arrays and writes
the inputs + outputs as CSV files that the Julia test suite reads back and asserts
against (HID exactly; rain rates within a tight tolerance).

This script is committed for reproducibility. It is version-independent: the
compiled Cython `hid_beta_f` is replaced by a pure-Python equivalent (identical
formula) so no specific CPython build is required.

Usage:
    <python> generate_fixtures.py /path/to/CSU_RadarTools /path/to/output_dir
"""
import os
import sys
import types

import numpy as np


def install_hid_beta_stub(csu_pkg_path):
    """Inject a pure-Python csu_radartools.calc_kdp_ray_fir.hid_beta_f.

    The Cython kernel computes exactly 1/(1+(((x-m)/a)**2)**b); the pure-Python
    form is numerically identical, so generated fixtures are unaffected.
    """
    sys.path.insert(0, os.path.dirname(csu_pkg_path.rstrip("/")))
    stub = types.ModuleType("csu_radartools.calc_kdp_ray_fir")

    def hid_beta_f(ngates, x_arr, a, b, m):
        x = np.asarray(x_arr, dtype="float64")
        return 1.0 / (1.0 + (((x - m) / a) ** 2) ** b)

    stub.hid_beta_f = hid_beta_f
    sys.modules["csu_radartools.calc_kdp_ray_fir"] = stub


def main():
    csu_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser("~/Development/CSU_RadarTools/csu_radartools")
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))

    install_hid_beta_stub(csu_path)
    from csu_radartools import csu_fhc, common
    from csu_radartools import csu_blended_rain_tropical as cbt

    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(20260616)

    n = 600
    dz = rng.uniform(-10.0, 65.0, n)
    zdr = rng.uniform(-1.0, 5.0, n)
    kdp = rng.uniform(-0.5, 5.0, n)
    rho = rng.uniform(0.80, 1.00, n)
    T = rng.uniform(-40.0, 30.0, n)

    bands = ["S", "C", "X"]

    # ---- FHC summer: with and without temperature, all bands ----
    for band in bands:
        for use_temp in (True, False):
            hid = csu_fhc.csu_fhc_summer(
                dz=dz, zdr=zdr, kdp=kdp, rho=rho,
                T=(T if use_temp else None), use_temp=use_temp,
                method="hybrid", band=band)
            tag = "withT" if use_temp else "noT"
            header = "dz,zdr,kdp,rho,T,hid"
            data = np.column_stack([dz, zdr, kdp, rho, T, hid.astype(float)])
            np.savetxt(os.path.join(outdir, f"fhc_summer_{band}_{tag}.csv"),
                       data, delimiter=",", header=header, comments="",
                       fmt="%.10g")

    # ---- Component rain rates (band-specific coefficients from the predef block) ----
    # Coefficients transcribed from calc_blended_rain_tropical's predef branch.
    band_coeffs = {
        "S": dict(kdp_zdr=(96.5726, 0.9315, -2.1140 / 10.),
                  z_zdr=(0.0085, 0.9237, -5.2389 / 10.),
                  kdp=(59.5202, 0.7451)),
        "C": dict(kdp_zdr=(45.6976, 0.8763, -1.6718 / 10.),
                  z_zdr=(0.0086, 0.9088, -4.2059 / 10.),
                  kdp=(34.5703, 0.7331)),
        "X": dict(kdp_zdr=(28.1289, 0.9194, -1.6876 / 10.),
                  z_zdr=(0.0085, 0.9294, -4.4580 / 10.),
                  kdp=(21.9729, 0.7221)),
    }
    # Band-independent R-Z coefficients.
    rz_all = (216.0, 1.39)
    rz_conv = (126.0, 1.39)
    rz_strat = (291.0, 1.55)

    for band in bands:
        c = band_coeffs[band]
        r_z_all = common.calc_rain_zr(dz, a=rz_all[0], b=rz_all[1])
        r_z_conv = common.calc_rain_zr(dz, a=rz_conv[0], b=rz_conv[1])
        r_z_strat = common.calc_rain_zr(dz, a=rz_strat[0], b=rz_strat[1])
        r_kdp = common.calc_rain_kdp(kdp, a=c["kdp"][0], b=c["kdp"][1])
        r_kdp_zdr = common.calc_rain_kdp_zdr(kdp, zdr, a=c["kdp_zdr"][0],
                                             b=c["kdp_zdr"][1], c=c["kdp_zdr"][2])
        r_z_zdr = common.calc_rain_z_zdr(dz, zdr, a=c["z_zdr"][0],
                                         b=c["z_zdr"][1], c=c["z_zdr"][2])
        header = "dz,zdr,kdp,r_z_all,r_z_conv,r_z_strat,r_kdp,r_kdp_zdr,r_z_zdr"
        data = np.column_stack([dz, zdr, kdp, r_z_all, r_z_conv, r_z_strat,
                                r_kdp, r_kdp_zdr, r_z_zdr])
        np.savetxt(os.path.join(outdir, f"rain_components_{band}.csv"),
                   data, delimiter=",", header=header, comments="", fmt="%.10g")

    # ---- Blended tropical rain: plain, with cs map, with fhc ----
    cs = rng.integers(0, 4, n).astype(float)         # 0,1,2,3
    fhc = rng.integers(1, 11, n).astype(float)       # 1..10
    for band in bands:
        # (a) no cs, no fhc
        rain, meth = cbt.calc_blended_rain_tropical(dz=dz, zdr=zdr, kdp=kdp, band=band)
        data = np.column_stack([dz, zdr, kdp, rain, meth.astype(float)])
        np.savetxt(os.path.join(outdir, f"rain_blended_{band}_plain.csv"),
                   data, delimiter=",", header="dz,zdr,kdp,rain,method",
                   comments="", fmt="%.10g")

        # (b) with cs map (exercises methods 5/6/4)
        rain, meth = cbt.calc_blended_rain_tropical(dz=dz, zdr=zdr, kdp=kdp, cs=cs, band=band)
        data = np.column_stack([dz, zdr, kdp, cs, rain, meth.astype(float)])
        np.savetxt(os.path.join(outdir, f"rain_blended_{band}_cs.csv"),
                   data, delimiter=",", header="dz,zdr,kdp,cs,rain,method",
                   comments="", fmt="%.10g")

        # (c) with fhc (exercises ice masking / hail). Records Python's (buggy)
        # method; the Julia test accounts for the intentional hail-only correction.
        rain, meth = cbt.calc_blended_rain_tropical(dz=dz, zdr=zdr, kdp=kdp, fhc=fhc, band=band)
        data = np.column_stack([dz, zdr, kdp, fhc, rain, meth.astype(float)])
        np.savetxt(os.path.join(outdir, f"rain_blended_{band}_fhc.csv"),
                   data, delimiter=",", header="dz,zdr,kdp,fhc,rain,method",
                   comments="", fmt="%.10g")

    print(f"Fixtures written to {outdir}")


if __name__ == "__main__":
    main()
