"""Export field footprints as a GeoPackage for QGIS: work_dir/field_shapes.gpkg.

One polygon (multipolygon) per field-year, outlining its labelled yield pixels (yield mask > -1 in the raw
GeoTIFF), in EPSG:4326. Attributes come from fields.parquet (country, farm, crop, year, n_pixels, yield_mean,
physical_field, region). One layer per country plus the combined layer "all".
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import country_dirs, load_paths  # noqa: E402

MASK = "mean_scaled_yield_masked_regional_statistical_outlier.tif"


def footprint(tif: Path):
    with rasterio.open(tif) as r:
        a = r.read(1)
        ok = (a > -1).astype(np.uint8)
        geoms = [shape(g) for g, v in shapes(ok, mask=ok.astype(bool), transform=r.transform)]
        return gpd.GeoSeries([unary_union(geoms)], crs=r.crs).to_crs(4326).iloc[0]


def main() -> None:
    paths = load_paths()
    work = paths["work_dir"]
    fields = pd.read_parquet(work / "fields.parquet")
    geoms = []
    for name, country in zip(fields["field"], fields["country"]):
        tif = country_dirs(paths, country)[1] / name / "yield_masks" / MASK   # Switzerland: own data root
        try:
            geoms.append(footprint(tif))
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"{name}: {e}")
            geoms.append(None)
    gdf = gpd.GeoDataFrame(fields, geometry=geoms, crs=4326)
    print("missing geometry:", gdf.geometry.isna().sum())
    gdf = gdf[gdf.geometry.notna()]
    out = work / "field_shapes.gpkg"
    out.unlink(missing_ok=True)
    gdf.to_file(out, layer="all", driver="GPKG")
    for c, g in gdf.groupby("country"):
        g.to_file(out, layer=c, driver="GPKG")
    print(f"wrote {out} ({len(gdf)} field-years)")


if __name__ == "__main__":
    main()
