"""Profile of the Raw/ folder: metadata JSON schema, raster grids/CRS/extent, field areas, terrain and soil value
ranges, S2 scene dates and gaps, SCL class shares, weather CSV columns, yield-mask coverage vs the NetCDF.

Exact coordinates are not written to the report (only a rounded overall bounding box and distances).
Writes report/04_raw.md and tables/swiss_raw_fields.csv.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds

from swiss_lib import RAW, TABLES, Report, swiss_meta

SCL = {0: "no_data", 1: "saturated", 2: "dark_area", 3: "cloud_shadow", 4: "vegetation", 5: "bare_soil", 6: "water",
       7: "unclassified", 8: "cloud_medium", 9: "cloud_high", 10: "thin_cirrus", 11: "snow"}
COORD_KEY = ("lat", "lon", "coord", "geometry", "bbox", "location", "centroid", "polygon")


def describe_values(vals: list) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "all null"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return f"numeric, min {min(vals):.4g}, median {np.median(vals):.4g}, max {max(vals):.4g}"
    if all(isinstance(v, (list, dict)) for v in vals):
        return f"{type(vals[0]).__name__}, length {min(map(len, vals))}-{max(map(len, vals))}"
    d = sorted({str(v) for v in vals})
    return f"{len(d)} distinct: {d[:12]}" + (" ..." if len(d) > 12 else "")


def metadata_schema(r: Report, fields) -> None:
    keys: dict[str, list] = defaultdict(list)
    n = 0
    for f in fields:
        js = json.loads(next(f.glob("metadata-*.json")).read_text())
        n += 1
        for k, v in js.items():
            keys[k].append(v)
    rows = []
    for k, vals in keys.items():
        hide = any(c in k.lower() for c in COORD_KEY)
        rows.append({"key": k, "present_in": f"{len(vals)}/{n}", "type": Counter(type(v).__name__ for v in vals).most_common(2).__str__(),
                     "values": "(hidden: location)" if hide else describe_values(vals)})
    r.table(pd.DataFrame(rows))


def grid_info(fields, r: Report) -> pd.DataFrame:
    rows = []
    for f in fields:
        with rasterio.open(next((f / "yield_masks").glob("*.tif"))) as ym:
            y = ym.read(1)
            valid = y >= 0
            l, b, rr, t = transform_bounds(ym.crs, "EPSG:4326", *ym.bounds)
            d = {"field": f.name, "crs": str(ym.crs), "res_m": ym.res[0], "height": ym.height, "width": ym.width,
                 "dtype": ym.dtypes[0], "nodata": ym.nodata, "lon": (l + rr) / 2, "lat": (b + t) / 2,
                 "mask_labelled_px": int(valid.sum()), "mask_total_px": y.size,
                 "area_ha": float(valid.sum() * ym.res[0] * ym.res[1] / 1e4),
                 "mask_yield_min": float(y[valid].min()) if valid.any() else np.nan,
                 "mask_yield_mean": float(y[valid].mean()) if valid.any() else np.nan,
                 "mask_yield_max": float(y[valid].max()) if valid.any() else np.nan,
                 "n_nonneg1_negative": int(((y < 0) & (y != -1)).sum())}
        scenes = sorted((f / "s2_images").glob("*.tif"))
        with rasterio.open(scenes[0]) as s2:
            d.update(s2_bands=s2.count, s2_dtype=s2.dtypes[0], s2_same_grid=(s2.shape == ym.shape and s2.transform == ym.transform),
                     s2_desc=";".join(str(x) for x in s2.descriptions), s2_nodata=s2.nodata)
        for name in ("dem", "aspect", "slope", "curvature", "twi"):
            with rasterio.open(next((f / "dem").glob(f"{name}-*.tif"))) as src:
                a = src.read(1).astype(float)
                d[f"{name}_mean"] = np.nanmean(np.where(a == src.nodata, np.nan, a)) if src.nodata is not None else a.mean()
                d[f"{name}_res"] = src.res[0]
                d[f"{name}_same_grid"] = src.shape == ym.shape
        with rasterio.open(next((f / "soil").glob("clay_*.tif"))) as src:
            d.update(soil_bands=src.count, soil_res=src.res[0], soil_shape=str(src.shape), soil_desc=";".join(map(str, src.descriptions))[:120])
        rows.append(d)
    t = pd.DataFrame(rows)
    t["source"] = t["field"].str.extract(r"(DUP\d)")
    t["farm"] = t["field"].str.extract(r"(farm\d)")
    return t


def scene_table(fields) -> tuple[pd.DataFrame, Counter]:
    rows, scl_counts = [], Counter()
    for f in fields:
        dates = sorted(pd.to_datetime([p.stem[-8:] for p in (f / "s2_images").glob("*.tif")], format="%Y%m%d"))
        gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
        cloud = []
        for p in sorted((f / "scl_masks").glob("*.tif")):
            with rasterio.open(p) as src:
                a = src.read(1)
            u, c = np.unique(a, return_counts=True)
            scl_counts.update(dict(zip(u.tolist(), c.tolist())))
            cloud.append(np.isin(a, (8, 9)).mean())
        rows.append({"field": f.name, "n_scenes": len(dates), "first": dates[0].date(), "last": dates[-1].date(),
                     "max_gap_d": gaps.max(), "median_gap_d": float(np.median(gaps)), "scl_cloud_8_9_frac_mean": float(np.mean(cloud)),
                     "scenes_with_cloud_gt_50pct": int((np.array(cloud) > 0.5).sum())})
    return pd.DataFrame(rows), scl_counts


def weather_table(fields, r: Report) -> None:
    frames = []
    for f in fields:
        df = pd.read_csv(next((f / "weather").glob("*.csv")))
        df["field"] = f.name
        frames.append(df)
    cols = Counter(tuple(df.columns) for df in frames)
    r.p(f"Columns (identical in all files: {len(cols) == 1}): `{list(frames[0].columns)}`")
    allw = pd.concat(frames, ignore_index=True)
    num = allw.select_dtypes("number")
    r.table(num.describe().T)
    date_col = next((c for c in allw.columns if "date" in c.lower() or "time" in c.lower()), None)
    if date_col:
        d = pd.to_datetime(allw[date_col], errors="coerce")
        per = allw.assign(_d=d).groupby("field")["_d"].agg(["min", "max", "count"])
        per["days"] = (per["max"] - per["min"]).dt.days + 1
        r.p(f"Daily rows per file: min {per['count'].min()}, median {per['count'].median()}, max {per['count'].max()}; "
            f"date column `{date_col}`; rows == span in days for all: {bool((per['count'] == per['days']).all())}")
    r.p(f"NaN share per column: {allw.isna().mean().round(4).to_dict()}")


def main() -> None:
    fields = sorted(p for p in RAW.iterdir() if p.is_dir())
    r = Report("04_raw", "SwissYield raw files (rasters, metadata, weather)")

    r.h("Metadata JSON schema (union over the 73 files)")
    metadata_schema(r, fields)

    g = grid_info(fields, r)
    r.h("Grids and CRS")
    r.p(f"CRS values: {dict(Counter(g.crs))}; resolution (m): {dict(Counter(g.res_m))}; yield mask dtype {set(g.dtype)}, nodata {set(g.nodata)}; "
        f"all S2 on the yield-mask grid: {bool(g.s2_same_grid.all())}; all DEM layers on that grid: "
        f"{bool(g[[c for c in g if c.endswith('_same_grid') and c != 's2_same_grid']].all().all())}")
    r.p(f"S2 tif: {set(g.s2_bands)} bands, dtype {set(g.s2_dtype)}, nodata {set(g.s2_nodata)}; band descriptions: {Counter(g.s2_desc).most_common(1)[0][0]}")
    r.p(f"Soil tif: bands {set(g.soil_bands)}, resolution {set(g.soil_res)} m, shapes {Counter(g.soil_shape).most_common(5)}, descriptions e.g. {g.soil_desc.iloc[0]}")
    r.p(f"Yield mask pixels < 0 other than -1: {int(g.n_nonneg1_negative.sum())}")

    r.h("Extent (WGS84, rounded)")
    r.p(f"Overall: lon {g.lon.min():.2f}..{g.lon.max():.2f}, lat {g.lat.min():.2f}..{g.lat.max():.2f}")
    r.table(g.groupby(["source", "farm"]).agg(fields=("field", "size"), lon_min=("lon", "min"), lon_max=("lon", "max"),
                                              lat_min=("lat", "min"), lat_max=("lat", "max")).round(2))
    km = lambda a, b: np.hypot((a.lon.mean() - b.lon.mean()) * 111.32 * np.cos(np.radians(a.lat.mean())), (a.lat.mean() - b.lat.mean()) * 111.2)  # noqa: E731
    farms = {k: v for k, v in g.groupby("farm")}
    r.p("Distance between farm centres (km): " + ", ".join(f"{a}-{b}: {km(farms[a], farms[b]):.0f}" for a in farms for b in farms if a < b))

    r.h("Field geometry (10 m pixels)")
    r.table(g.groupby("source")[["height", "width", "mask_total_px", "mask_labelled_px", "area_ha"]].describe().T)
    r.h("Terrain per field (raster means over the field)")
    r.table(g.groupby("source")[["dem_mean", "slope_mean", "aspect_mean", "curvature_mean", "twi_mean"]].describe().T)
    r.h("Raw yield-mask values (t/ha, labelled pixels), per source")
    r.table(g.groupby("source")[["mask_yield_min", "mask_yield_mean", "mask_yield_max"]].describe().T)

    sc, scl = scene_table(fields)
    r.h("S2 scenes")
    r.table(sc.describe().T)
    tot = sum(scl.values())
    r.p("SCL class share over all scene pixels: " + ", ".join(f"{SCL.get(k, k)} {v / tot:.1%}" for k, v in sorted(scl.items())))
    r.p("Preprocessing keeps only SCL 4 (vegetation) and 5 (bare soil) pixels; per time step it picks the scene with the fewest SCL 8+9 pixels.")

    r.h("Weather CSVs (daily ERA5, seeding-60 d .. harvest+60 d)")
    weather_table(fields, r)

    # raw labelled pixels vs NetCDF labelled pixels per field-year
    meta = swiss_meta()
    nc = meta[meta.target.notna()].groupby("field").size().rename("nc_labelled_px")
    g = g.merge(nc, left_on="field", right_index=True, how="left").merge(sc, on="field")
    g["nc_over_mask"] = g["nc_labelled_px"] / g["mask_labelled_px"]
    r.h("NetCDF labelled pixels vs raw yield-mask pixels")
    r.table(g.groupby("source")[["nc_over_mask"]].describe().T)
    g.drop(columns=["s2_desc", "soil_desc", "lon", "lat"]).to_csv(TABLES / "swiss_raw_fields.csv", index=False)
    r.p("Per-field table (no coordinates): `tables/swiss_raw_fields.csv`.")
    r.save()


if __name__ == "__main__":
    main()
