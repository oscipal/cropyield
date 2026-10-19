"""Complete description of the three Swiss NetCDF files: schema, variables, band list and groups, categorical
decodings, time axis, per-band normalisation stats, label sanity, field-year table, comparison with YieldSAT.

Writes report/01_netcdf.md and tables/swiss_field_years.csv, tables/swiss_bands.csv.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from swiss_lib import (DYNAMIC, FOLDERS, TABLES, Report, swiss_bands, swiss_ds, swiss_meta, swiss_static_names,
                       yieldsat_static_names)
from src.data import CATEGORICAL_VARS, get_mapping  # noqa: E402


def band_group(b: str) -> str:
    if b in DYNAMIC[:12]:
        return "S2 reflectance (dynamic)"
    if b in DYNAMIC[12:]:
        return "ERA5 weather (dynamic, summed/meaned between selected scenes)"
    if re.search(r"_\d+-\d+_uncertainty$", b):
        return "SoilGrids uncertainty (static, 6 depths)"
    if re.search(r"_\d+-\d+$", b):
        return "SoilGrids value (static, 6 depths)"
    if b in ("dem", "aspect", "slope", "curvature", "twi"):
        return "SRTM terrain (static)"
    return "other static (coordinates etc.)"


def main() -> None:
    r = Report("01_netcdf", "SwissYield preprocessed NetCDF (YieldSAT format)")
    ds = swiss_ds("all")

    r.h("Files")
    rows = []
    for part in FOLDERS:
        d = swiss_ds(part)
        rows.append({"file": FOLDERS[part], "pixels(index)": d.sizes["index"], "time_steps": d.sizes["time_step"],
                     "bands": d.sizes["band"], "field_years(attrs)": len(d.attrs)})
    r.table(pd.DataFrame(rows))
    r.p("DUP1 = Perich farm (farm1), DUP2 = farms 2+3, 'Switzerland' = union. One index entry = one 10 m pixel of one "
        "field-year; `sample[index, time_step, band]` float32 is the model input.")

    r.h("Variables")
    r.table(pd.DataFrame([{"variable": v, "dims": str(ds[v].dims), "dtype": str(ds[v].dtype),
                           "attrs": ", ".join(ds[v].attrs)[:80]} for v in ds.data_vars]))
    r.p("Coordinates: " + ", ".join(f"{c}{ds[c].dims}" for c in ds.coords))

    r.h("Categorical variables (code -> label)")
    for v in CATEGORICAL_VARS:
        try:
            m = get_mapping(ds, v)
        except KeyError:
            continue
        labels = list(m.values())
        r.p(f"- `{v}`: {len(m)} labels" + (f": {labels}" if len(m) <= 30 else f", e.g. {labels[:3]} ..."))
    r.p("`row`/`col` are the pixel position in the field raster (uint8 codes decoded by src.data); `times[index, time_step]` is the date of the selected scene per step.")

    meta = swiss_meta()
    r.h("Time axis (date of the selected scene per time step, pooled over pixels)")
    times = pd.DataFrame(ds["times"].values[::37])  # strided subsample of pixels is enough for the date range
    rows = []
    for k in range(times.shape[1]):
        t = pd.to_datetime(times[k], errors="coerce").dropna()
        doy = t.dt.dayofyear
        rows.append({"step": k, "window_start_doy": 30 * k + 1, "scene_doy_min": doy.min(), "scene_doy_median": doy.median(),
                     "scene_doy_max": doy.max(), "scene_year_range": f"{t.dt.year.min()}-{t.dt.year.max()}", "n": len(t)})
    r.table(pd.DataFrame(rows))

    r.h("Bands")
    bands = swiss_bands(ds)
    stats = pd.DataFrame({"band": bands, **{s: ds[f"stats-{s}"].values for s in ("mean", "std", "min", "max")}})
    stats["group"] = [band_group(b) for b in bands]
    stats.to_csv(TABLES / "swiss_bands.csv", index=False)
    r.table(stats.groupby("group").agg(n=("band", "size"), first=("band", "first"), last=("band", "last")))
    ys = yieldsat_static_names()
    st = swiss_static_names(ds)
    r.p(f"Static bands: Swiss {len(st)} / YieldSAT {len(ys)}; identical set: {set(st) == set(ys)}. "
        f"Band order identical to YieldSAT countries: not guaranteed - always index by name.")
    r.p("`stats-*` hold the dataset-level mean/std/min/max per band (normalisation statistics stored by YieldSAT; "
        "check whether they equal YieldSAT's before reusing them):")
    r.table(stats[~stats.band.str.contains(r"_\d+-\d+")].drop(columns="group"))
    r.p("SoilGrids bands follow `<property>_<depth>[_uncertainty]` with depths 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm "
        "and properties: " + ", ".join(sorted({re.sub(r"_\d+-\d+.*", "", b) for b in bands if re.search(r"_\d+-\d+$", b)})))

    r.h("Pixels, labels")
    meta["labelled"] = meta["target"].notna()
    r.table(meta.groupby("source").agg(pixels=("target", "size"), labelled=("labelled", "sum"), field_years=("field_year", "nunique")))
    bad = meta.groupby("source")["target"].agg(nan=lambda s: s.isna().sum(), le0=lambda s: (s <= 0).sum(), gt20=lambda s: (s > 20).sum())
    r.p("Target sanity (NaN / <=0 / >20 t/ha):")
    r.table(bad)
    r.p(f"Duplicate (field_year,row,col): {int(meta.duplicated(['field_year', 'row', 'col']).sum())}. "
        "Target unit t/ha (wet mass corrected to 15 % moisture); no -1 sentinel in the NetCDF.")

    g = meta.groupby("field_year")
    t = g["target"]
    table = pd.DataFrame({
        "source": g["source"].first(), "farm": g["farm"].first(), "crop": g["crop"].first(), "year": g["year"].first(),
        "n_pixels": g.size(), "n_labelled": g["labelled"].sum(), "row_min": g["row"].min(), "row_max": g["row"].max(),
        "col_min": g["col"].min(), "col_max": g["col"].max(), "yield_mean": t.mean(), "yield_std": t.std(),
        "yield_min": t.min(), "yield_p5": t.quantile(0.05), "yield_median": t.median(), "yield_p95": t.quantile(0.95),
        "yield_max": t.max(), "seeding_date": g["seeding_date"].first(), "harvesting_date": g["harvesting_date"].first(),
        "seeding_date_type": g["seeding_date_type"].first()})
    season = pd.to_datetime(table["harvesting_date"], errors="coerce") - pd.to_datetime(table["seeding_date"], errors="coerce")
    table["season_days"] = season.dt.days
    table.to_csv(TABLES / "swiss_field_years.csv")

    r.h("Field-years")
    r.table(table.groupby(["source", "farm", "crop"]).size().unstack(fill_value=0))
    r.table(table.groupby(["source", "year"]).size().unstack(fill_value=0))
    r.table(table.groupby("source")[["n_pixels", "yield_mean", "yield_std", "season_days"]].describe().T)
    r.p("Seeding date types: " + str(table.groupby(["source", "seeding_date_type"]).size().to_dict()))
    r.p("Full table: `tables/swiss_field_years.csv`.")
    r.save()


if __name__ == "__main__":
    main()
