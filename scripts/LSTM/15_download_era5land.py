"""ERA5-Land point time series (soil water + weather) for every 0.1 deg cell that holds a field, from the Copernicus CDS.

    python scripts/LSTM/15_download_era5land.py              # all cells, 2 requests in parallel, skips finished cells
    python scripts/LSTM/15_download_era5land.py --limit 2    # first 2 cells only (test)

Needs ~/.cdsapirc (url + personal access token) and the ERA5-Land licence accepted on the CDS website. Dataset
`reanalysis-era5-land-timeseries` (hourly, ~9 km, 1950-present), 1995-2024 so that a 30-year climatology per cell is
available for anomalies. A field's cell = the ERA5-Land grid point (multiples of 0.1 deg) nearest to the centroid of
its outline in `<work_dir>/field_shapes.gpkg`.

Outputs in <work_dir>/era5land/:
    cells.parquet            field (field-year id as in fields.parquet) -> cell, lat, lon
    daily/<cell>.parquet     daily values per cell: swvl1..4 (m3 m-3, mean; layers 0-7, 7-28, 28-100, 100-289 cm),
                             t2m_c / d2m_c (degC, mean), tmax_c / tmin_c (degC), tp_mm (mm d-1, sum), ssrd_mj (MJ m-2 d-1)
Only counts are printed, never values or coordinates (CLAUDE.md: the Swiss outlines are among the fields).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_paths  # noqa: E402

DATASET = "reanalysis-era5-land-timeseries"
VARIABLES = [f"volumetric_soil_water_level_{i}" for i in (1, 2, 3, 4)] + [
    "2m_temperature", "2m_dewpoint_temperature", "total_precipitation", "surface_solar_radiation_downwards"]
DATES = "1995-01-01/2024-12-31"


def cell_table(work: Path) -> pd.DataFrame:
    import geopandas as gpd
    g = gpd.read_file(work / "field_shapes.gpkg", layer="all")
    cen = g.geometry.to_crs(3857).centroid.to_crs(4326)
    lat, lon = np.round(cen.y.to_numpy() * 10) / 10, np.round(cen.x.to_numpy() * 10) / 10
    cell = [f"{'n' if a >= 0 else 's'}{abs(a):05.1f}_{'e' if o >= 0 else 'w'}{abs(o):05.1f}" for a, o in zip(lat, lon)]
    return pd.DataFrame({"field": g["field"].to_numpy(), "country": g["country"].to_numpy(), "cell": cell,
                         "lat": lat, "lon": lon})


def to_daily(csv: Path) -> pd.DataFrame:
    """Hourly CSV -> daily table (UTC days). Precipitation and radiation: the time series dataset gives hourly
    amounts, summed per day; checked in --limit runs that the daily totals are plausible (not accumulations)."""
    h = pd.read_csv(csv, parse_dates=["valid_time"]).set_index("valid_time")
    d = pd.DataFrame(index=h.resample("D").mean().index)
    r = h.resample("D")
    for k in ("swvl1", "swvl2", "swvl3", "swvl4"):
        d[k] = r[k].mean().astype("float32")
    d["t2m_c"] = (r["t2m"].mean() - 273.15).astype("float32")
    d["tmax_c"] = (r["t2m"].max() - 273.15).astype("float32")
    d["tmin_c"] = (r["t2m"].min() - 273.15).astype("float32")
    d["d2m_c"] = (r["d2m"].mean() - 273.15).astype("float32")
    d["tp_mm"] = (r["tp"].sum() * 1000.0).astype("float32")
    d["ssrd_mj"] = (r["ssrd"].sum() / 1e6).astype("float32")
    d["n_hours"] = r["t2m"].count().astype("int16")
    return d.rename_axis("date").reset_index()


def fetch(cell: str, lat: float, lon: float, out: Path) -> str:
    import cdsapi
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        z = Path(tmp) / "x.zip"
        for attempt in range(3):
            try:
                cdsapi.Client(quiet=True, progress=False).retrieve(DATASET, {
                    "variable": VARIABLES, "location": {"longitude": float(lon), "latitude": float(lat)},
                    "date": [DATES], "data_format": "csv"}, str(z))
                break
            except Exception as e:  # noqa: BLE001  (CDS queue / network hiccups)
                if attempt == 2:
                    return f"{cell} FAILED {type(e).__name__}: {str(e)[:200]}"
                time.sleep(60)
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp)
        csvs = sorted(Path(tmp).glob("*.csv"))
        parts = [pd.read_csv(c, parse_dates=["valid_time"]).drop(columns=["latitude", "longitude"]) for c in csvs]
        merged = parts[0]
        for p in parts[1:]:
            merged = merged.merge(p, on="valid_time", how="outer")
        merged.to_csv(Path(tmp) / "all.csv", index=False)
        d = to_daily(Path(tmp) / "all.csv")
        d.to_parquet(out.with_suffix(".tmp"))
        shutil.move(out.with_suffix(".tmp"), out)
    miss = int(d[["swvl1", "t2m_c"]].isna().all(axis=1).sum())
    return f"{cell} ok, {len(d)} days, {miss} days without data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)   # CDS rejects requests beyond its per-dataset queue limit
    a = ap.parse_args()
    work = Path(load_paths()["work_dir"])
    root = work / "era5land"
    (root / "daily").mkdir(parents=True, exist_ok=True)
    t = cell_table(work)
    t.to_parquet(root / "cells.parquet")
    cells = t.drop_duplicates("cell")[["cell", "lat", "lon"]]
    print(f"{len(t)} field-years, {len(cells)} cells ({t.drop_duplicates('cell').groupby('country').size().to_dict()})",
          flush=True)
    todo = [r for r in cells.itertuples() if not (root / "daily" / f"{r.cell}.parquet").exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(cells) - len(todo) if not a.limit else 'n/a'} done before, {len(todo)} to fetch", flush=True)
    t0, n_ok, n_fail = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        jobs = [ex.submit(fetch, r.cell, r.lat, r.lon, root / "daily" / f"{r.cell}.parquet") for r in todo]
        for i, j in enumerate(as_completed(jobs), 1):
            msg = j.result()
            n_fail += "FAILED" in msg
            n_ok += "FAILED" not in msg
            # the cell id holds the rounded coordinates: print only a running count, and failures without the id
            print(f"[{i}/{len(todo)}] {'FAILED ' + msg.split('FAILED', 1)[1] if 'FAILED' in msg else 'ok'} "
                  f"({msg.split(', ', 1)[1] if 'ok' in msg else ''}) {time.time() - t0:.0f} s", flush=True)
    print(f"##### era5land done: {n_ok} ok, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
