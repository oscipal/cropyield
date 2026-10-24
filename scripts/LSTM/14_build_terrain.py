"""Terrain features per pixel, recomputed from the raw DEM of every field-year with one method for all countries.

    python scripts/LSTM/14_build_terrain.py             # -> <work_dir>/lstm/terrain.npy + terrain.json
    python scripts/LSTM/14_build_terrain.py --check 20  # 20 field-years per country, summary only, no output

Why: the delivered terrain bands (static.npy / NetCDF, raw dem/*.tif) are inconsistent between countries. slope is
in an unknown unit (values 0-22,000), curvature is clipped to [-1, 1] (Brazil: integers), twi is missing for all
German and most South American pixels and on different scales elsewhere, and the Brazilian DEM is integer metres
in 30 m blocks while the others are smooth floats. Only the elevation itself is usable as delivered.

Method, on the field's full raster (10 m grid of the S2 images and the yield mask, rows/cols of meta.parquet):
    elevation     the raw DEM (m), nodata -> NaN
    rel_elevation elevation minus the median elevation of the field's labelled pixels (m): low / high spots
    slope         degrees, from a local quadratic surface fitted per pixel by Gaussian-weighted least squares
                  (sigma SIGMA_PX pixels, ~30 m SRTM scale) over the valid pixels inside the raster, so pixels at
                  the raster edge and next to holes are not biased (plain smoothing + finite differences
                  underestimated the slope at the edge by a third)
    aspect_sin    sin of the aspect = east component of the downslope direction (aspect clockwise from north);
    aspect_cos    cos of the aspect = north component; both 0 where slope < FLAT_DEG (aspect undefined)
    curvature     minus the Laplacian of the fitted surface, 1/km; > 0 convex (crest), < 0 concave (hollow)
TWI is not recomputed: it needs flow accumulation over the upstream catchment, which a DEM clipped to the field
cannot give.

Output (row order of lstm/data/meta.parquet):
    <work_dir>/lstm/terrain.npy   float32 (pixels, 6)
    <work_dir>/lstm/terrain.json  names, units, method parameters, counts
Only counts are printed for Switzerland (CLAUDE.md); quantiles are printed for the YieldSAT countries.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_paths  # noqa: E402

_spec = importlib.util.spec_from_file_location("s2grid", Path(__file__).with_name("11_build_s2_grid.py"))
s2grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2grid)

NAMES = ["elevation", "rel_elevation", "slope", "aspect_sin", "aspect_cos", "curvature"]
UNITS = {"elevation": "m", "rel_elevation": "m", "slope": "deg", "aspect_sin": "-1..1 (east +)",
         "aspect_cos": "-1..1 (north +)", "curvature": "1/km (convex +)"}
PIXEL_M = 10.0
SIGMA_PX = 1.5   # Gaussian smoothing before derivatives (~30 m SRTM scale)
FLAT_DEG = 0.1   # below this slope the aspect is set to 0 / 0
PRIVATE = ("Switzerland",)  # counts only


def read_dem(field_dir: Path) -> np.ndarray:
    path = next((field_dir / "dem").glob("dem-*.tif"))
    with rasterio.open(path) as r:
        z = r.read(1).astype(np.float64)
        nodata = r.nodata
    if nodata is not None:
        z[z == nodata] = np.nan
    z[~np.isfinite(z)] = np.nan
    return z


def quadratic_fit(z: np.ndarray, sigma: float) -> np.ndarray:
    """Per pixel, a Gaussian-weighted least-squares fit z ~ c0 + c1 e + c2 n + c3 e^2 + c4 e n + c5 n^2 over the
    valid pixels of a (2 r + 1)^2 window (e east, n north, in metres). Only pixels inside the raster and not NaN
    enter, so the fit is unbiased at the raster edge and next to holes. Returns (rows, cols, 6), NaN where z is
    NaN or too few neighbours."""
    r = int(np.ceil(3 * sigma))
    dr, dc = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    e, n = dc * PIXEL_M, -dr * PIXEL_M
    w = np.exp(-(dr ** 2 + dc ** 2) / (2 * sigma ** 2))
    basis = [np.ones_like(e), e, n, e * e, e * n, n * n]
    valid = np.isfinite(z)
    zv, wv = np.where(valid, z, 0.0), valid.astype(np.float64)
    k = len(basis)
    A = np.empty(z.shape + (k, k))
    b = np.empty(z.shape + (k,))
    for i in range(k):
        for j in range(i, k):
            A[..., i, j] = A[..., j, i] = correlate(wv, w * basis[i] * basis[j], mode="constant", cval=0.0)
        b[..., i] = correlate(zv, w * basis[i], mode="constant", cval=0.0)
    ok = valid & (correlate(wv, np.ones_like(w), mode="constant") >= 3 * k)
    out = np.full(z.shape + (k,), np.nan)
    out[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    return out


def terrain(z: np.ndarray) -> dict[str, np.ndarray]:
    """Rasters of slope (deg), aspect sin/cos and curvature (1/km) from a DEM in metres on the 10 m grid."""
    c = quadratic_fit(z, SIGMA_PX)
    dz_east, dz_north = c[..., 1], c[..., 2]
    grad = np.hypot(dz_east, dz_north)
    slope = np.degrees(np.arctan(grad))
    # downslope direction (-dz_east, -dz_north); aspect clockwise from north -> sin = east, cos = north
    flat = slope < FLAT_DEG
    with np.errstate(invalid="ignore", divide="ignore"):
        a_sin = np.where(flat, 0.0, -dz_east / grad)
        a_cos = np.where(flat, 0.0, -dz_north / grad)
    curv = -2.0 * (c[..., 3] + c[..., 5]) * 1000.0   # minus Laplacian, 1/m -> 1/km, convex positive
    nan = ~np.isfinite(grad)
    for a in (a_sin, a_cos):
        a[nan] = np.nan
    return {"slope": slope, "aspect_sin": a_sin, "aspect_cos": a_cos, "curvature": curv}


def build_field(field_dir: Path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    z = read_dem(field_dir)
    t = terrain(z)
    elev = z[rows, cols]
    rel = elev - np.nanmedian(elev) if np.isfinite(elev).any() else elev
    cols_out = {"elevation": elev, "rel_elevation": rel, **{k: v[rows, cols] for k, v in t.items()}}
    return np.stack([cols_out[n] for n in NAMES], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- workers

_G: dict = {}


def _init(meta_path, out_path):
    m = pd.read_parquet(meta_path, columns=["row", "col"])
    _G.update(rows=m["row"].to_numpy(), cols=m["col"].to_numpy(),
              out=None if out_path is None else np.load(out_path, mmap_mode="r+"))


def _work(task):
    i, d, a, b = task
    x = build_field(d, _G["rows"][a:b], _G["cols"][a:b])
    if _G["out"] is not None:
        _G["out"][a:b] = x
        return i, None
    return i, x


# --------------------------------------------------------------------------- reporting

def summary(x: np.ndarray, country: np.ndarray, delivered_dem: np.ndarray | None) -> None:
    """NaN counts per country and band; quantiles for the YieldSAT countries; elevation vs the delivered band."""
    rows = []
    for c in pd.unique(country):
        m = country == c
        row = {"country": c, "pixels": int(m.sum())}
        for j, n in enumerate(NAMES):
            row[f"nan_{n}"] = int(np.isnan(x[m, j]).sum())
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    q = []
    for c in pd.unique(country):
        if c in PRIVATE:
            continue
        m = country == c
        for j, n in enumerate(NAMES):
            v = x[m, j][np.isfinite(x[m, j])]
            if v.size:
                p = np.percentile(v, [1, 50, 99])
                q.append({"country": c, "band": n, "p1": p[0], "p50": p[1], "p99": p[2]})
    print(pd.DataFrame(q).round(3).to_string(index=False))
    if delivered_dem is not None:
        keep = ~np.isin(country, PRIVATE)
        a, b = x[keep, 0], delivered_dem[keep]
        both = np.isfinite(a) & np.isfinite(b)
        print(f"elevation vs delivered dem band (YieldSAT countries): {both.sum():,} pixels compared, "
              f"{int((np.abs(a[both] - b[both]) > 0.01).sum()):,} differ by > 0.01 m, "
              f"{int((np.isfinite(a) != np.isfinite(b)).sum()):,} differ in being missing")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", type=int, default=0, help="N field-years per country, print summary, no output")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    paths = load_paths()
    data_dir = paths["work_dir"] / "lstm" / "data"
    meta_path = data_dir / "meta.parquet"
    meta = pd.read_parquet(meta_path, columns=["country", "field"])
    fields = s2grid.field_dirs(paths, meta)
    static_names = json.loads((data_dir / "info.json").read_text())["static_bands"]
    static = np.load(data_dir / "static.npy", mmap_mode="r")
    dem_col = static_names.index("dem")
    print(f"{len(fields):,} field-years, {len(meta):,} pixels", flush=True)

    if args.check:
        pick = fields.groupby("country").sample(n=args.check, random_state=0, replace=False) \
            if (fields.groupby("country").size() >= args.check).all() \
            else fields.groupby("country").head(args.check)
        tasks = [(i, r.dir, r.start, r.stop) for i, r in pick.iterrows()]
        with Pool(args.workers, _init, (meta_path, None)) as pool:
            res = dict(pool.imap_unordered(_work, tasks))
        idx = np.concatenate([np.arange(fields.at[i, "start"], fields.at[i, "stop"]) for i in pick.index])
        x = np.concatenate([res[i] for i in pick.index])
        summary(x, meta["country"].to_numpy()[idx], np.asarray(static[idx, dem_col]))
        return

    out_path = paths["work_dir"] / "lstm" / "terrain.npy"
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(len(meta), len(NAMES)))
    del out
    t0 = time.time()
    tasks = [(i, r.dir, r.start, r.stop) for i, r in fields.iterrows()]
    with Pool(args.workers, _init, (meta_path, out_path)) as pool:
        for j, _ in enumerate(pool.imap_unordered(_work, tasks, chunksize=4), 1):
            if j % 200 == 0 or j == len(tasks):
                print(f"  {j:,}/{len(tasks):,} field-years ({time.time() - t0:.0f} s)", flush=True)
    x = np.load(out_path, mmap_mode="r")
    country = meta["country"].to_numpy()
    summary(np.asarray(x), country, np.asarray(static[:, dem_col]))
    info = {"names": NAMES, "units": UNITS, "pixel_m": PIXEL_M, "sigma_px": SIGMA_PX, "flat_deg": FLAT_DEG,
            "n_pixels": len(meta), "nan_per_band": {n: int(np.isnan(x[:, j]).sum()) for j, n in enumerate(NAMES)},
            "source": "scripts/LSTM/14_build_terrain.py"}
    (out_path.with_suffix(".json")).write_text(json.dumps(info, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
