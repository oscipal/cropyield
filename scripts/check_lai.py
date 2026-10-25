"""Checks of src/lai.py (SNAP LAI network).

    python scripts/check_lai.py              # downloads the Sentinel Hub reference script (or --js <path>)

1. Weights: re-parses every constant from the reference JavaScript and evaluates it as a plain scalar
   transcription; must equal src.lai.snap_lai on random spectra (|diff| < 1e-4).
2. Plausibility on typical spectra (bare soil ~0, dense green canopy several m2 m-2).
3. Sensitivity to the assumed angles (view zenith 0-11°, relative azimuth cos -1..1, overpass 10:00-11:00) on the
   loader's real reflectances of a few non-Swiss fields: median |delta LAI|.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import lai as L  # noqa: E402

URL = "https://raw.githubusercontent.com/sentinel-hub/custom-scripts/main/sentinel-2/lai/script.js"


def reference(js: str):
    """Scalar LAI function built only from the constants in the JS text."""
    num = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    norm = {m[0]: (float(m[1]), float(m[2])) for m in
            re.findall(r"var (\w+)_norm = normalize\(sample\.\w+, (" + num + r"), (" + num + r")\)", js)}
    any_num = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    vz = re.search(r"viewZen_norm = normalize\(Math\.cos\(.*?\), (" + any_num + r"), (" + any_num + r")\)", js).groups()
    sz = re.search(r"sunZen_norm\s*= normalize\(Math\.cos\(.*?\), (" + any_num + r"), (" + any_num + r")\)", js).groups()
    den = re.search(r"denormalize\(l2, (" + num + r"), (" + num + r")\)", js).groups()
    neurons = []
    for i in range(1, 6):
        body = re.search(rf"function neuron{i}\(.*?\{{(.*?)return", js, re.S).group(1)
        neurons.append([float(c.replace(" ", "")) for c in re.findall(r"([-+]\s*\d+\.\d+)", body)])
    l2 = [float(c.replace(" ", "")) for c in
          re.findall(r"([-+]\s*\d+\.\d+)", re.search(r"function layer2\(.*?\{(.*?)return", js, re.S).group(1))]
    order = ["b03", "b04", "b05", "b06", "b07", "b8a", "b11", "b12"]
    assert all(len(n) == 12 for n in neurons) and len(l2) == 6 and set(order) <= set(norm)
    nz = lambda x, lo, hi: 2 * (x - lo) / (hi - lo) - 1  # noqa: E731

    def f(bands, cos_vz, cos_sz, cos_raz):
        v = [nz(b, *norm[k]) for b, k in zip(bands, order)]
        v += [nz(cos_vz, float(vz[0]), float(vz[1])), nz(cos_sz, float(sz[0]), float(sz[1])), cos_raz]
        n = [2 / (1 + math.exp(-2 * (w[0] + sum(a * b for a, b in zip(w[1:], v))))) - 1 for w in neurons]
        out = l2[0] + sum(a * b for a, b in zip(l2[1:], n))
        return 0.5 * (out + 1) * (float(den[1]) - float(den[0])) + float(den[0])
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--js", type=Path, default=None)
    a = ap.parse_args()
    js = a.js.read_text() if a.js else urllib.request.urlopen(URL, timeout=30).read().decode()
    ref = reference(js)
    ok = True

    rng = np.random.default_rng(0)
    refl = rng.uniform(0.0, 0.5, (500, 8)).astype(np.float32)
    csz = rng.uniform(0.4, 0.93, 500)
    mine = L.snap_lai(refl, list(range(8)), csz)
    cvz = math.cos(math.radians(L.VIEW_ZENITH_DEG))
    theirs = np.clip([ref(r, cvz, c, L.COS_REL_AZIMUTH) for r, c in zip(refl.tolist(), csz)], 0, L.LAI_MAX)
    d = float(np.abs(mine - theirs).max())
    ok &= d < 1e-4
    print(f"1. weights vs reference script: max |diff| {d:.2e} on 500 random spectra -> {'OK' if d < 1e-4 else 'FAIL'}")

    #            B03   B04   B05   B06   B07   B8A   B11   B12
    spectra = {"bare soil": [0.10, 0.13, 0.15, 0.17, 0.18, 0.20, 0.28, 0.24],
               "sparse crop": [0.07, 0.07, 0.11, 0.20, 0.24, 0.27, 0.22, 0.14],
               "dense green crop": [0.05, 0.03, 0.08, 0.30, 0.42, 0.46, 0.20, 0.09]}
    vals = {k: float(L.snap_lai(np.array([v], np.float32), list(range(8)), np.array([0.8]))[0])
            for k, v in spectra.items()}
    plaus = vals["bare soil"] < 0.5 and 0.5 < vals["sparse crop"] < vals["dense green crop"] and vals["dense green crop"] > 2
    ok &= plaus
    print("2. typical spectra:", {k: round(v, 2) for k, v in vals.items()}, "->", "OK" if plaus else "FAIL")

    # 3. sensitivity on real reflectances (non-Swiss pixels only; values are not printed)
    from src.loader import load
    d = load("south_america/loyo/all", role="train", features=["s2"], grid="10d", pixels_per_field=20,
             countries=["Argentina", "Uruguay"], verbose=False)
    X = d.dyn.astype(np.float32).reshape(-1, 12)
    X = X[np.isfinite(X).all(1)][:200_000]
    bi = [["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"].index(b) for b in L.BANDS]
    sz = L.cos_sun_zenith(np.full(len(X), -33.0), np.full(len(X), 20.0))
    base = L.snap_lai(X, bi, sz)
    print(f"3. sensitivity on {len(X):,} clear Argentine / Uruguayan pixel-steps (median base LAI {np.median(base):.2f}):")
    for name, kw in [("view zenith 0°", dict(cos_vz=1.0)), ("view zenith 11°", dict(cos_vz=math.cos(math.radians(11)))),
                     ("rel. azimuth cos -1", dict(cos_raz=-1.0)), ("rel. azimuth cos +1", dict(cos_raz=1.0))]:
        print(f"   {name:22s} median |dLAI| {np.median(np.abs(L.snap_lai(X, bi, sz, **kw) - base)):.3f}")
    for hour in (10.0, 11.0):
        old, L.OVERPASS_SOLAR_HOUR = L.OVERPASS_SOLAR_HOUR, hour
        s2 = L.cos_sun_zenith(np.full(len(X), -33.0), np.full(len(X), 20.0))
        L.OVERPASS_SOLAR_HOUR = old
        print(f"   overpass {hour:.0f}:00            median |dLAI| {np.median(np.abs(L.snap_lai(X, bi, s2) - base)):.3f}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
