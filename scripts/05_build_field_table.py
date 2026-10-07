"""One row per field-year for all countries: fields.parquet in work_dir.

Columns: field, country, farm, crop, year, n_pixels, yield_mean, physical_field, region.

field_shared_name is per field-year, so the same physical field recurs across years under different
names. physical_field groups field-years whose pixel footprints overlap (IoU > 0.5 on a 20 m grid; pixel
positions from row/col and the raw GeoTIFF transform). Double cropping (two field-years of the same year
on the same land) is therefore one physical field.
region is the farm, except that farms are merged (connected components) when they share a physical field
or when any of their fields lie within MERGE_KM of each other (edge to edge), so that held-out regions in
leave-regions-out splits are never next to training regions.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_paths  # noqa: E402
from src.data import load_meta, open_dataset  # noqa: E402

COUNTRIES = ["Germany", "Argentina", "Brazil", "Uruguay"]
NC_NAME = "merge_s2-soil-dem-weather-coords.nc"
# common metric CRS per country for comparing footprints (fields span two UTM zones in DE and AR)
COUNTRY_CRS = {"Germany": "EPSG:32632", "Argentina": "EPSG:32720", "Brazil": "EPSG:32722",
               "Uruguay": "EPSG:32721"}
IOU_MIN = 0.5
CELL_M = 20.0
MERGE_KM = 10.0  # farms with fields closer than this form one region (fields of different countries are >140 km apart)


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> dict:
        return {x: self.find(x) for x in self.parent}


def field_table(meta: pd.DataFrame, country: str) -> pd.DataFrame:
    fields = meta.groupby("field", observed=True).agg(
        farm=("farm", "first"), crop=("crop", "first"), year=("year", "first"),
        n_pixels=("target", "size"), yield_mean=("target", "mean"),
        n_farm=("farm", "nunique"), n_crop=("crop", "nunique"), n_year=("year", "nunique"))
    bad = fields[(fields[["n_farm", "n_crop", "n_year"]] > 1).any(axis=1)]
    assert bad.empty, f"{country}: fields with several farms/crops/years: {bad.index[:5].tolist()}"
    fields = fields.drop(columns=["n_farm", "n_crop", "n_year"]).reset_index()
    fields.insert(1, "country", country)
    for c in ("field", "farm", "crop", "year"):
        fields[c] = fields[c].astype(str)
    fields["farm"] = country + "_" + fields["farm"]
    return fields


def footprints(raw_dir: Path, meta: pd.DataFrame, names: np.ndarray, crs: str) -> list[set | None]:
    """Set of occupied CELL_M grid cells (in ``crs``) per field-year; None where no raw DEM exists."""
    idx = meta.groupby("field", observed=True).indices
    rows, cols = meta["row"].to_numpy(), meta["col"].to_numpy()
    out: list[set | None] = []
    for name in names:
        tif = raw_dir / name / "dem" / f"dem-{name}.tif"
        if not tif.exists():
            out.append(None)
            continue
        with rasterio.open(tif) as r:
            src_crs, tr = r.crs, r.transform
        i = idx[name]
        x, y = tr * (cols[i] + 0.5, rows[i] + 0.5)
        if src_crs != crs:
            x, y = transform(src_crs, crs, x, y)
        cells = np.floor(np.column_stack([x, y]) / CELL_M).astype(np.int64)
        out.append(set(map(tuple, np.unique(cells, axis=0))))
    return out


def physical_ids(names: np.ndarray, cells: list[set | None]) -> np.ndarray:
    """Group field-years whose footprints overlap with IoU > IOU_MIN."""
    uf = UnionFind(range(len(names)))
    ok = [i for i, c in enumerate(cells) if c]
    # candidate pairs via bounding boxes of the cell sets, exact IoU only for those
    bb = np.array([[min(x for x, _ in cells[i]), min(y for _, y in cells[i]),
                    max(x for x, _ in cells[i]), max(y for _, y in cells[i])] for i in ok])
    for j, i in enumerate(ok):
        cand = np.flatnonzero((bb[:, 0] <= bb[j, 2]) & (bb[:, 2] >= bb[j, 0]) &
                              (bb[:, 1] <= bb[j, 3]) & (bb[:, 3] >= bb[j, 1]))
        for c in cand[cand > j]:
            k = ok[c]
            inter = len(cells[i] & cells[k])
            if inter and inter / (len(cells[i]) + len(cells[k]) - inter) > IOU_MIN:
                uf.union(i, k)
    roots = uf.groups()
    # name each physical field after its first field-year (sorted), stable across runs
    first = {}
    for i in sorted(range(len(names)), key=lambda i: names[i]):
        first.setdefault(roots[i], names[i])
    return np.array([first[roots[i]] for i in range(len(names))])


def outline(cells: set) -> np.ndarray:
    """Centres (m) of the cells on the border of a footprint: enough for edge-to-edge distances."""
    c = np.array(sorted(cells))
    inner = [(x, y) for x, y in c if {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)} <= cells]
    border = np.array(sorted(set(map(tuple, c)) - set(inner)))
    return (border + 0.5) * CELL_M


def regions(fields: pd.DataFrame, cells: list[set | None]) -> np.ndarray:
    """Farms linked by a shared physical field or by fields within MERGE_KM form one region."""
    farm = fields["farm"].to_numpy()
    uf = UnionFind(fields["farm"].unique())
    for farms in fields.groupby("physical_field")["farm"].unique():
        for f in farms[1:]:
            uf.union(farms[0], f)
    # proximity: candidate pairs from bounding boxes, exact edge distance on the footprint outlines
    gap = MERGE_KM * 1000
    ok = [i for i, c in enumerate(cells) if c]
    pts = {i: outline(cells[i]) for i in ok}
    bb = np.array([[*pts[i].min(0), *pts[i].max(0)] for i in ok])
    for j, i in enumerate(ok):
        dx = np.maximum(0, np.maximum(bb[:, 0] - bb[j, 2], bb[j, 0] - bb[:, 2]))
        dy = np.maximum(0, np.maximum(bb[:, 1] - bb[j, 3], bb[j, 1] - bb[:, 3]))
        for c in np.flatnonzero(np.hypot(dx, dy) <= gap):
            k = ok[c]
            if k <= i or uf.find(farm[i]) == uf.find(farm[k]):
                continue
            d = np.sqrt(((pts[i][:, None, :] - pts[k][None, :, :]) ** 2).sum(-1)).min() - CELL_M
            if d <= gap:
                uf.union(farm[i], farm[k])
    roots = uf.groups()
    members: dict[str, list[str]] = {}
    for farm, root in roots.items():
        members.setdefault(root, []).append(farm)
    label = {root: "+".join(sorted(fs)) for root, fs in members.items()}
    return fields["farm"].map(lambda f: label[roots[f]]).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="*", default=COUNTRIES)
    args = ap.parse_args()
    paths = load_paths()

    tables = []
    for country in args.countries:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta = load_meta(open_dataset(paths["preprocessed"] / country / NC_NAME))
        meta = meta[meta["target"].notna()]
        meta["field"] = meta["field"].astype(str)
        fields = field_table(meta, country)
        cells = footprints(paths["raw"] / country, meta, fields["field"].to_numpy(), COUNTRY_CRS[country])
        del meta
        missing = sum(c is None for c in cells)
        if missing:
            print(f"WARNING {country}: {missing} field-years without raw DEM; each is its own physical field")
        fields["physical_field"] = physical_ids(fields["field"].to_numpy(), cells)
        fields["region"] = regions(fields, cells)
        n_merged = (fields.groupby("region")["farm"].nunique() > 1).sum()
        print(f"{country}: {len(fields)} field-years, {fields['physical_field'].nunique()} physical fields, "
              f"{fields['farm'].nunique()} farms -> {fields['region'].nunique()} regions "
              f"({n_merged} merged); crops {fields['crop'].value_counts().to_dict()}", flush=True)
        tables.append(fields)

    out = pd.concat(tables, ignore_index=True)
    assert out["field"].is_unique
    path = paths["work_dir"] / "fields.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out)} field-years)")


if __name__ == "__main__":
    main()
