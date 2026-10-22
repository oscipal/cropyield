"""One-off files for the visualizer: pixel row range and bounding box of every field-year, and light field
outlines for the web map.

    python viz/build_index.py      # -> <work_dir>/viz/{field_index,field_shapes_display,field_dates}.parquet

Pixels of a field are contiguous rows in lstm/data (as in FieldSet, scripts/LSTM/field_image.py), so one
field-year is the slice [start, end) of dyn.npy / static.npy / meta.parquet. The polygons of field_shapes.gpkg
trace every pixel (3.3M vertices, 0.3M holes); for display they are closed by 15 m, holes and parts < 400 m2 are
dropped and the outline simplified by 8 m (about 50x fewer vertices). Sowing and harvest dates come from the raw
metadata JSON of every field (src/season.py). Only counts are printed (CLAUDE.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import country_dirs, load_paths  # noqa: E402
from src.season import read_season  # noqa: E402


def main() -> None:
    paths = load_paths()
    work = paths["work_dir"]
    meta = pq.read_table(work / "lstm" / "data" / "meta.parquet", columns=["field", "row", "col"]).to_pandas()
    codes, names = pd.factorize(meta["field"])
    start = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    end = np.r_[start[1:], len(codes)]
    if len(start) != len(names):
        raise ValueError("pixels of a field are not contiguous")
    rows, cols = meta["row"].to_numpy(np.int32), meta["col"].to_numpy(np.int32)
    idx = pd.DataFrame({
        "field": names[codes[start]], "start": start, "end": end,
        "row_min": np.minimum.reduceat(rows, start), "col_min": np.minimum.reduceat(cols, start),
        "row_max": np.maximum.reduceat(rows, start), "col_max": np.maximum.reduceat(cols, start)})
    out = work / "viz"
    out.mkdir(exist_ok=True)
    idx.to_parquet(out / "field_index.parquet", index=False)
    n_table = len(pd.read_parquet(work / "fields.parquet", columns=["field"]))
    print(f"{len(idx)} field-years indexed ({len(meta):,} pixels); fields.parquet has {n_table}")
    dates = field_dates(paths)
    dates.to_parquet(out / "field_dates.parquet", index=False)
    print(f"{len(dates)} field-years with dates; missing sowing {int(dates['sow_day'].isna().sum())}, missing harvest "
          f"{int(dates['harvest_day'].isna().sum())}; seeding_date_type: "
          f"{dates['seeding_date_type'].value_counts(dropna=False).to_dict()}")
    shapes = display_shapes(work / "field_shapes.gpkg")
    shapes.to_parquet(out / "field_shapes_display.parquet")
    print(f"{len(shapes)} display outlines, {int(shapes.count_coordinates().sum()):,} vertices")


def field_dates(paths: dict) -> pd.DataFrame:
    fields = pd.read_parquet(paths["work_dir"] / "fields.parquet", columns=["field", "country"])
    rows = [{"field": f, **read_season(country_dirs(paths, c)[1] / f)} for f, c in zip(fields["field"], fields["country"])]
    t = pd.DataFrame(rows)
    for c in ("seeding_date", "harvesting_date", "jan1"):
        t[c] = pd.to_datetime(t[c])
    return t.astype({"sow_day": "Int32", "harvest_day": "Int32"})


def display_shapes(gpkg: Path):
    import geopandas as gpd
    g = gpd.read_file(gpkg, layer="all")[["field", "geometry"]].to_crs(3857)

    def outer(geom):
        parts = [shapely.Polygon(p.exterior) for p in getattr(geom, "geoms", [geom]) if p.area > 400]
        return shapely.MultiPolygon(parts) if parts else geom

    geom = g.geometry.buffer(15, resolution=2).buffer(-15, resolution=2).apply(outer).simplify(8)
    return gpd.GeoDataFrame({"field": g["field"]}, geometry=geom, crs=3857).to_crs(4326)


if __name__ == "__main__":
    main()
