"""ERA5-Land soil water and weather per field-year on the fixed time grids of the S2 cubes, with anomalies against a
1995-2015 climatology of the field's cell.

    python scripts/LSTM/16_build_era5land_grid.py                 # every grid folder: data_w10, data_w15, data_w30
    python scripts/LSTM/16_build_era5land_grid.py --grids data_w10

Input: <work_dir>/era5land/daily/<cell>.parquet and cells.parquet (scripts/LSTM/15_download_era5land.py). Windows
as in 13_build_weather_grid.py: window k = days [start + k d, start + (k + 1) d) on the cube's day axis
(src/season.py: day_of, day 0 = 2 January of the year before harvest). Features per window:

    swvl1, swvl2, swvl3     volumetric soil water, layers 0-7 / 7-28 / 28-100 cm (m3 m-3), window mean
    swvl_rz                 0-100 cm, thickness-weighted (7, 21, 72 cm)
    swvl1_anom, swvl_rz_anom   minus the cell's climatology of the same calendar days (m3 m-3)
    t2m_c, tmax_c           mean daily mean / maximum 2 m temperature (degC)
    tp_mm                   precipitation sum (mm)
    ssrd_mj                 mean daily surface solar radiation (MJ m-2 d-1)
    vpd_kpa                 mean daily vapour pressure deficit from daily mean t2m and d2m (Tetens; kPa)
    t2m_anom_c, tp_anom_mm, vpd_anom_kpa   anomalies against the climatology (mean / sum of daily anomalies)

Climatology: per cell and day of year, mean over 1995-2015 (before the dataset's first harvest year, so no held-out
year enters it), smoothed with a 31-day circular moving average. One value per field-year, the same for all its
pixels. Outputs <work_dir>/lstm/<grid>/era5land.npy float32 (n_fields, n_times, n_features) and era5land.json, rows in
the field order of meta.parquet (as weather.npy). Only counts are printed, never values (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import country_dirs, load_paths  # noqa: E402
from src.season import read_season  # noqa: E402

FEATURES = {"swvl1": "m3 m-3", "swvl2": "m3 m-3", "swvl3": "m3 m-3", "swvl_rz": "m3 m-3", "swvl1_anom": "m3 m-3",
            "swvl_rz_anom": "m3 m-3", "t2m_c": "degC", "tmax_c": "degC", "tp_mm": "mm", "ssrd_mj": "MJ m-2 d-1",
            "vpd_kpa": "kPa", "t2m_anom_c": "degC", "tp_anom_mm": "mm", "vpd_anom_kpa": "kPa"}
SUM = ("tp_mm", "tp_anom_mm")                      # summed over the window; all others are means
CLIM_YEARS = (1995, 2015)
GRIDS = ["data_w10", "data_w15", "data_w30"]


def _svp(t_c):
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def cell_daily(path: Path) -> pd.DataFrame:
    """Daily table with the derived columns and their anomalies, indexed by date."""
    d = pd.read_parquet(path).set_index("date")
    d["swvl_rz"] = (7 * d["swvl1"] + 21 * d["swvl2"] + 72 * d["swvl3"]) / 100.0
    d["vpd_kpa"] = np.maximum(_svp(d["t2m_c"]) - _svp(d["d2m_c"]), 0.0)
    doy = np.minimum(d.index.dayofyear, 365) - 1
    clim_rows = (d.index.year >= CLIM_YEARS[0]) & (d.index.year <= CLIM_YEARS[1])
    for src, dst in (("swvl1", "swvl1_anom"), ("swvl_rz", "swvl_rz_anom"), ("t2m_c", "t2m_anom_c"),
                     ("tp_mm", "tp_anom_mm"), ("vpd_kpa", "vpd_anom_kpa")):
        c = pd.Series(d[src].to_numpy()[clim_rows]).groupby(doy[clim_rows]).mean().reindex(range(365)).to_numpy()
        c = np.convolve(np.r_[c[-15:], c, c[:15]], np.ones(31) / 31, mode="valid")    # circular 31-day mean
        d[dst] = d[src].to_numpy() - c[doy]
    return d


def windows(d: pd.DataFrame, jan1, start: int, days: int, n_times: int) -> np.ndarray:
    out = np.full((n_times, len(FEATURES)), np.nan, dtype=np.float32)
    cols = list(FEATURES)
    vals = d[cols]
    for k in range(n_times):
        a = pd.Timestamp(jan1 + timedelta(days=start + k * days + 1))     # day_of(date) = (date - jan1).days - 1
        g = vals.loc[a:a + pd.Timedelta(days=days - 1)]
        if len(g) < days:
            continue
        out[k] = np.where(np.isin(cols, SUM), g.sum().to_numpy(), g.mean().to_numpy())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", default=GRIDS)
    args = ap.parse_args()
    paths = load_paths()
    work, lstm = paths["work_dir"], paths["work_dir"] / "lstm"
    order = pd.read_parquet(work / "viz" / "field_index.parquet", columns=["field"])["field"].tolist()
    table = pd.read_parquet(work / "fields.parquet", columns=["field", "country"]).set_index("field")
    cells = pd.read_parquet(work / "era5land" / "cells.parquet").set_index("field")["cell"]
    grids = []
    for g in args.grids:
        info = json.loads((lstm / g / "info.json").read_text())
        grids.append((info.get("start_day", 0), info.get("window_days", 30), info["n_times"]))
    arrs = [np.full((len(order), n, len(FEATURES)), np.nan, np.float32) for _, _, n in grids]
    by_cell = pd.Series(order).groupby(cells.reindex(order).to_numpy()).groups
    no_cell = 0
    for i, (cell, idx) in enumerate(by_cell.items()):
        path = work / "era5land" / "daily" / f"{cell}.parquet"
        if not path.exists():
            no_cell += len(idx)
            continue
        d = cell_daily(path)
        for r in idx:
            f = order[r]
            jan1 = read_season(country_dirs(paths, table.at[f, "country"])[1] / f)["jan1"]
            for arr, g in zip(arrs, grids):
                arr[r] = windows(d, jan1, *g)
        if (i + 1) % 50 == 0:
            print(f"{i + 1} / {len(by_cell)} cells", flush=True)
    for g, (start, days, n), arr in zip(args.grids, grids, arrs):
        np.save(lstm / g / "era5land.npy", arr)
        (lstm / g / "era5land.json").write_text(json.dumps(
            {"fields": order, "features": list(FEATURES), "units": FEATURES, "start_day": start, "window_days": days,
             "n_times": n, "climatology_years": CLIM_YEARS, "source": "scripts/LSTM/16_build_era5land_grid.py"},
            indent=1))
        print(f"{g}: {arr.shape}, windows with data {np.isfinite(arr[..., 0]).mean():.1%}", flush=True)
    print(f"{len(order) - no_cell} field-years with ERA5-Land, {no_cell} without a downloaded cell")


if __name__ == "__main__":
    main()
