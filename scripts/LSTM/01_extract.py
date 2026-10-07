"""Extract all countries once into memmaps for the pixel LSTM.

Outputs in <work_dir>/lstm/data:
    dyn.npy       float32 (pixels, 24, 16)  bands that change over time (S2 + weather), NaN outside the season
    static.npy    float32 (pixels, 104)     bands repeated at every time step in the NetCDF (soil, DEM, coords)
    meta.parquet  one row per pixel: country, field, farm, crop, year, row, col, target
    info.json     band names of both arrays and row ranges per country

Storing the static bands once instead of 24 times keeps the full dataset at ~24 GB instead of ~140 GB.
The model repeats them over time again (input fusion), so its input equals the NetCDF ``sample``.
Only pixels with a non-NaN target are kept.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_paths  # noqa: E402
from src.data import iter_blocks, load_meta, open_dataset  # noqa: E402

COUNTRIES = ["Germany", "Argentina", "Brazil", "Uruguay"]
NC_NAME = "merge_s2-soil-dem-weather-coords.nc"
DYNAMIC = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12",
           "temp_mean", "temp_max", "temp_min", "total_prec"]
META_COLS = ["field", "farm", "crop", "year", "row", "col", "target"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=100_000)
    args = ap.parse_args()

    paths = load_paths()
    out_dir = paths["work_dir"] / "lstm" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets, metas, masks = {}, {}, {}
    for c in COUNTRIES:
        ds = open_dataset(paths["preprocessed"] / c / NC_NAME)
        meta = load_meta(ds)
        masks[c] = meta["target"].notna().to_numpy()
        datasets[c], metas[c] = ds, meta
        print(f"{c}: {len(meta):,} pixels, {masks[c].sum():,} with target", flush=True)

    # band order differs between the country files: always index bands by name
    bands = {c: [str(b) for b in datasets[c]["band"].values] for c in COUNTRIES}
    for c in COUNTRIES[1:]:
        assert set(bands[c]) == set(bands[COUNTRIES[0]]), f"band set differs in {c}"
    static = [b for b in bands[COUNTRIES[0]] if b not in DYNAMIC]

    n_total = sum(int(m.sum()) for m in masks.values())
    n_times = datasets[COUNTRIES[0]].sizes["time_step"]
    dyn = np.lib.format.open_memmap(out_dir / "dyn.npy", mode="w+", dtype=np.float32,
                                    shape=(n_total, n_times, len(DYNAMIC)))
    stat = np.lib.format.open_memmap(out_dir / "static.npy", mode="w+", dtype=np.float32,
                                     shape=(n_total, len(static)))

    pos, ranges, meta_parts = 0, {}, []
    for c in COUNTRIES:
        ds, mask, start_pos, t0 = datasets[c], masks[c], pos, time.time()
        dyn_idx = [bands[c].index(b) for b in DYNAMIC]
        static_idx = [bands[c].index(b) for b in static]
        n = ds.sizes["index"]
        for start, stop in iter_blocks(n, args.block):
            m = mask[start:stop]
            if not m.any():
                continue
            arr = ds["sample"].isel(index=slice(start, stop)).transpose("index", "time_step", "band").values[m]
            s = arr[:, :, static_idx]
            # static bands must be identical at every time step, otherwise they belong in DYNAMIC
            if not np.allclose(s, s[:, :1], equal_nan=True):
                bad = [static[j] for j in np.flatnonzero(~np.isclose(s, s[:, :1], equal_nan=True).all((0, 1)))]
                raise ValueError(f"{c}: bands vary over time but are treated as static: {bad}")
            dyn[pos:pos + len(arr)] = arr[:, :, dyn_idx]
            stat[pos:pos + len(arr)] = s[:, 0]
            pos += len(arr)
            print(f"  {c}: pixels {stop:,}/{n:,} ({time.time() - t0:.0f} s)", flush=True)
        ranges[c] = [start_pos, pos]
        part = metas[c].loc[mask, [col for col in META_COLS if col in metas[c]]].reset_index()
        part.insert(1, "country", c)
        meta_parts.append(part)
    assert pos == n_total
    dyn.flush()
    stat.flush()

    meta = pd.concat(meta_parts, ignore_index=True)
    for col in ("country", "field", "farm", "crop", "year"):
        meta[col] = meta[col].astype(str)
    meta.to_parquet(out_dir / "meta.parquet", index=False)

    fields = pd.read_parquet(paths["work_dir"] / "fields.parquet")
    missing = set(fields["field"]) ^ set(meta["field"].unique())
    if missing:
        print(f"WARNING: {len(missing)} fields differ between meta and fields.parquet, e.g. {sorted(missing)[:5]}")

    n_nan_static = int(np.isnan(stat[:]).any(axis=1).sum())
    info = {"dynamic_bands": DYNAMIC, "static_bands": static, "n_pixels": n_total, "n_times": n_times,
            "country_rows": ranges, "pixels_with_nan_static": n_nan_static}
    (out_dir / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "static_bands"}, indent=1))


if __name__ == "__main__":
    main()
