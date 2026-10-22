"""Weather features per field-year on the fixed time grids of the S2 cubes, from the raw daily ERA5 CSVs.

    python scripts/LSTM/13_build_weather_grid.py                 # every grid folder: data, data_w10, data_w15, data_w30
    python scripts/LSTM/13_build_weather_grid.py --grids data_w10

The YieldSAT weather bands in lstm/data/dyn.npy are sums between consecutive *selected scenes* (Kelvin and metres,
both scene days counted), so a value mostly encodes the gap length. Here every window of the grid gets the same
calendar days (day convention of src/season.py, windows [start + k d, start + (k + 1) d)) and readable units:

    tmean_c, tmax_c, tmin_c   mean of the daily values (degC)
    prec_mm                   precipitation summed over the window (mm; windows have equal length)
    gdd                       growing degree days in the window, sum of max(0, Tmean - Tbase), Tbase per crop (TBASE)
    gdd_cum, prec_cum_mm      cumulative since sowing, at the window end (NaN before sowing, frozen after harvest)
    frost_days, heat_days     days with Tmin < 0 degC / Tmax > HEAT_C degC
    rh_mean                   mean relative humidity (%)            } NaN where the CSV lacks RH_mean / S_rad /
                                                                    } Temp_dew (many Argentina and Brazil files)
    srad                      mean daily S_rad of the CSV (unit as delivered, likely Wh m-2 d-1)
    vpd_kpa                   mean vapour pressure deficit from Tmean and Temp_dew (Tetens)
    season_frac               share of the window's days between sowing and harvest (models may mask with it)

A window not fully covered by the CSV (sowing - 60 d .. harvest + 60 d) is NaN. The values are the same for all
pixels of a field, so the output is per field-year, in the row order of lstm/data/meta.parquet:

    <work_dir>/lstm/<grid>/weather.npy    float32 (n_fields, n_times, n_features)
    <work_dir>/lstm/<grid>/weather.json   fields, features, units, grid (start_day, window_days, n_times), TBASE

Only counts are printed, never values (CLAUDE.md: the Swiss CSVs are read by this script only).
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import country_dirs, load_paths  # noqa: E402
from src.season import day_of, read_season  # noqa: E402

TBASE = {"wheat": 0.0, "winter_cereal": 0.0, "rapeseed": 0.0, "corn": 10.0, "soybean": 10.0}
HEAT_C = 30.0
FEATURES = {"tmean_c": "degC", "tmax_c": "degC", "tmin_c": "degC", "prec_mm": "mm", "gdd": "degC d",
            "gdd_cum": "degC d", "prec_cum_mm": "mm", "frost_days": "d", "heat_days": "d", "rh_mean": "%",
            "srad": "S_rad as delivered (likely Wh m-2 d-1)", "vpd_kpa": "kPa", "season_frac": "0-1"}
GRIDS = ["data", "data_w10", "data_w15", "data_w30"]


def daily(field_dir: Path, crop: str) -> tuple[pd.DataFrame, dict]:
    """Daily table on the cube's day axis with the derived daily columns, and the season of the field."""
    s = read_season(field_dir)
    w = pd.read_csv(next((field_dir / "weather").glob("*.csv")), parse_dates=["Date"])
    d = pd.DataFrame({"day": [day_of(t.date(), s["jan1"]) for t in w["Date"]]})
    d["tmean_c"], d["tmax_c"], d["tmin_c"] = w["Temp_mean"] - 273.15, w["Temp_max"] - 273.15, w["Temp_min"] - 273.15
    d["prec_mm"] = w["Total_prec"] * 1000.0
    d["gdd"] = np.maximum(d["tmean_c"] - TBASE.get(crop, 0.0), 0.0)
    d["frost_days"] = (d["tmin_c"] < 0).astype(float)
    d["heat_days"] = (d["tmax_c"] > HEAT_C).astype(float)
    # RH, S_rad and Temp_dew are missing in many South American CSVs (only Temp_* and Total_prec): NaN there
    nan = pd.Series(np.nan, index=w.index)
    d["rh_mean"], d["srad"] = w.get("RH_mean", nan), w.get("S_rad", nan)
    es = lambda t: 0.6108 * np.exp(17.27 * t / (t + 237.3))
    d["vpd_kpa"] = es(d["tmean_c"]) - es(w.get("Temp_dew", nan) - 273.15)
    in_season = (d["day"] >= s["sow_day"]) & (d["day"] <= s["harvest_day"])
    d["in_season"] = in_season.astype(float)
    d["gdd_cum"] = d["gdd"].where(in_season, 0.0).cumsum().where(d["day"] >= s["sow_day"])
    d["prec_cum_mm"] = d["prec_mm"].where(in_season, 0.0).cumsum().where(d["day"] >= s["sow_day"])
    return d.set_index("day").sort_index(), s


MEAN = ["tmean_c", "tmax_c", "tmin_c", "rh_mean", "srad", "vpd_kpa"]
SUM = ["prec_mm", "gdd", "frost_days", "heat_days"]
LAST = ["gdd_cum", "prec_cum_mm"]


def windows(d: pd.DataFrame, start: int, days: int, n_times: int) -> np.ndarray:
    out = np.full((n_times, len(FEATURES)), np.nan, dtype=np.float32)
    col = {f: i for i, f in enumerate(FEATURES)}
    for k in range(n_times):
        a = start + k * days
        g = d.loc[a:a + days - 1]
        if len(g) < days:          # not fully covered by the CSV
            continue
        for f in MEAN:
            out[k, col[f]] = g[f].mean()
        for f in SUM:
            out[k, col[f]] = g[f].sum()
        for f in LAST:
            out[k, col[f]] = g[f].iloc[-1]
        out[k, col["season_frac"]] = g["in_season"].mean()
    return out


def _work(task):
    field_dir, crop, grids = task
    try:
        d, _ = daily(field_dir, crop)
    except (FileNotFoundError, StopIteration):
        return None
    return [windows(d, *g) for g in grids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", default=GRIDS, help="folders under <work_dir>/lstm with an info.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    paths = load_paths()
    lstm = paths["work_dir"] / "lstm"
    meta = pd.read_parquet(paths["work_dir"] / "viz" / "field_index.parquet", columns=["field"]) \
        if (paths["work_dir"] / "viz" / "field_index.parquet").exists() else None
    if meta is None:
        raise SystemExit("run viz/build_index.py first (field order of meta.parquet)")
    table = pd.read_parquet(paths["work_dir"] / "fields.parquet", columns=["field", "country", "crop"]).set_index("field")
    order = meta["field"].tolist()
    grids = []
    for g in args.grids:
        info = json.loads((lstm / g / "info.json").read_text())
        grids.append((info.get("start_day", 0), info.get("window_days", 30), info["n_times"]))
    tasks = [(country_dirs(paths, table.at[f, "country"])[1] / f, table.at[f, "crop"], grids) for f in order]
    with Pool(args.workers) as pool:
        res = pool.map(_work, tasks, chunksize=16)
    missing = [f for f, r in zip(order, res) if r is None]
    for gi, (g, (start, days, n_times)) in enumerate(zip(args.grids, grids)):
        arr = np.stack([r[gi] if r is not None else np.full((n_times, len(FEATURES)), np.nan, np.float32)
                        for r in res])
        np.save(lstm / g / "weather.npy", arr)
        (lstm / g / "weather.json").write_text(json.dumps(
            {"fields": order, "features": list(FEATURES), "units": FEATURES, "start_day": start, "window_days": days,
             "n_times": n_times, "tbase": TBASE, "heat_c": HEAT_C, "source": "scripts/LSTM/13_build_weather_grid.py"},
            indent=1))
        covered = np.isfinite(arr[..., 0])
        full_season = np.nan_to_num(arr[..., -1]) == 1
        print(f"{g}: {arr.shape}, windows covered by the CSV {covered.mean():.1%}, fully in season "
              f"{full_season.mean():.1%}, with RH {np.isfinite(arr[..., list(FEATURES).index('rh_mean')]).mean():.1%}",
              flush=True)
    print(f"{len(order) - len(missing)} field-years with weather, {len(missing)} without a CSV")


if __name__ == "__main__":
    main()
