"""Helpers for Part A (data inspection). Used by notebooks/00_inspect.ipynb."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .data import FeatureReader, iter_blocks, pixel_dim

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".gz", ".bz2", ".xz", ".rar")
SOIL_KEYWORDS = ("bdod", "cec", "cfvo", "clay", "nitrogen", "ocd", "ocs", "phh2o", "sand", "silt", "soc", "soil")
RAW_KEYWORDS = ("quality", "qual", "flag", "support", "std", "count", "n_points", "npoints", "points", "n_obs")


# --------------------------------------------------------------------------- 1. files

def list_tree(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rows.append({"path": str(p.relative_to(root)), "size_mb": p.stat().st_size / 1e6})
    return pd.DataFrame(rows, columns=["path", "size_mb"])


def summarize_tree(tree: pd.DataFrame, depth: int = 2) -> pd.DataFrame:
    """Aggregate file counts/sizes per directory prefix of the given depth."""
    if tree.empty:
        return tree
    prefix = tree["path"].map(lambda s: "/".join(Path(s).parts[:depth]))
    return (tree.assign(prefix=prefix).groupby("prefix")
            .agg(n_files=("path", "size"), size_mb=("size_mb", "sum")).reset_index())


def check_archives(root: str | Path, tree: pd.DataFrame) -> pd.DataFrame:
    """For each archive, check whether an extracted sibling (same stem) exists."""
    root = Path(root)
    rows = []
    for rel in tree["path"]:
        if not rel.lower().endswith(ARCHIVE_SUFFIXES):
            continue
        p = root / rel
        stem = p.name
        for suf in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
            if stem.lower().endswith(suf):
                stem = stem[: -len(suf)]
                break
        target = p.parent / stem
        rows.append({
            "archive": rel,
            "size_mb": p.stat().st_size / 1e6,
            "extracted_path_exists": target.exists(),
            "n_extracted_files": sum(1 for f in target.rglob("*") if f.is_file()) if target.is_dir() else 0,
        })
    return pd.DataFrame(rows, columns=["archive", "size_mb", "extracted_path_exists", "n_extracted_files"])


# --------------------------------------------------------------------------- 2. structure

def dataset_overview(ds: xr.Dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    dims = pd.DataFrame({"dim": list(ds.sizes), "size": list(ds.sizes.values())})
    rows = []
    for name, var in ds.variables.items():
        rows.append({
            "name": name,
            "kind": "coord" if name in ds.coords else "data",
            "dims": ",".join(var.dims),
            "shape": "x".join(map(str, var.shape)),
            "dtype": str(var.dtype),
            "size_mb": var.size * var.dtype.itemsize / 1e6,
            "attrs": ", ".join(list(var.attrs)[:8]) + (" ..." if len(var.attrs) > 8 else ""),
        })
    return dims, pd.DataFrame(rows)


# --------------------------------------------------------------------------- 4./5. counts and target

def counts_table(meta: pd.DataFrame) -> pd.DataFrame:
    return (meta.groupby(["crop", "year", "farm"], observed=True)
            .agg(n_fields=("field", "nunique"), n_pixels=("target", "size")).reset_index())


def field_means(meta: pd.DataFrame) -> pd.DataFrame:
    return (meta.groupby("field_year", observed=True)
            .agg(field=("field", "first"), crop=("crop", "first"), year=("year", "first"),
                 farm=("farm", "first"), target_mean=("target", "mean"), n_pixels=("target", "size"),
                 n_target_nan=("target", lambda s: int(s.isna().sum())))
            .reset_index())


def field_target_stats(fields: pd.DataFrame) -> pd.DataFrame:
    return (fields.groupby(["crop", "year"], observed=True)["target_mean"]
            .describe(percentiles=[0.25, 0.5, 0.75]).reset_index())


def ground_truth_check(ds: xr.Dataset, fields: pd.DataFrame) -> pd.DataFrame:
    """Compare pixel-mean target with attributes named ``<field>_<...>_yield_ground_truth``."""
    attrs = {**ds.attrs, **ds["target"].attrs}
    gt = {k: v for k, v in attrs.items() if k.endswith("yield_ground_truth")}
    names = sorted(fields["field"].astype(str).unique(), key=len, reverse=True)
    rows = []
    for key, value in gt.items():
        field = next((f for f in names if key.startswith(f + "_")), None)
        middle = key[len(field) + 1: -len("yield_ground_truth")].strip("_") if field else None
        rows.append({"attr": key, "field": field, "middle": middle, "gt": float(np.asarray(value).ravel()[0])})
    gt_df = pd.DataFrame(rows, columns=["attr", "field", "middle", "gt"])
    if gt_df.empty:
        return gt_df
    f = fields.assign(field=fields["field"].astype(str), year=fields["year"].astype(str))
    merged = gt_df.merge(f[["field", "year", "crop", "target_mean"]], on="field", how="left")
    # if the middle part contains a year, keep only the matching field-year
    has_year = merged["middle"].fillna("").str.contains(r"(?:19|20)\d\d")
    year_ok = merged.apply(lambda r: str(r["year"]) in str(r["middle"]), axis=1)
    merged = merged[~has_year | year_ok].copy()
    merged["diff"] = merged["target_mean"] - merged["gt"]
    merged["identical_1e-4"] = merged["diff"].abs() < 1e-4
    return merged


# --------------------------------------------------------------------------- 6. inputs

def scan_s2(reader: FeatureReader, meta: pd.DataFrame, bands: list[str],
            block: int = 100_000, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One pass over all pixels: NaN fraction per (time, band) and mean NDVI per (crop, year, time)."""
    b04, b08 = bands.index("B04"), bands.index("B08")
    T = reader.n_times
    nan_count = np.zeros((T, len(bands)), dtype=np.int64)
    groups = meta[["crop", "year"]].astype(str).agg("|".join, axis=1)
    codes, uniques = pd.factorize(groups)
    ndvi_sum = np.zeros((len(uniques), T))
    ndvi_n = np.zeros((len(uniques), T))
    for start, stop in iter_blocks(reader.n_pixels, block):
        x = reader.read(bands, start, stop)
        nan_count += np.isnan(x).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = (x[..., b08] - x[..., b04]) / (x[..., b08] + x[..., b04])
        ndvi[~np.isfinite(ndvi)] = np.nan
        c = codes[start:stop]
        valid = ~np.isnan(ndvi)
        np.add.at(ndvi_sum, c, np.where(valid, ndvi, 0.0))
        np.add.at(ndvi_n, c, valid.astype(np.float64))
        if verbose:
            print(f"  scanned {stop:,}/{reader.n_pixels:,}", flush=True)
    nan_frac = pd.DataFrame(nan_count / reader.n_pixels, columns=bands)
    nan_frac.index.name = "t"
    with np.errstate(invalid="ignore"):
        mean = ndvi_sum / ndvi_n
    ndvi_df = (pd.DataFrame(mean, index=pd.Index(uniques, name="crop|year")).reset_index()
               .melt(id_vars="crop|year", var_name="t", value_name="ndvi"))
    ndvi_df[["crop", "year"]] = ndvi_df["crop|year"].str.split("|", expand=True)
    return nan_frac, ndvi_df.drop(columns="crop|year")


def _first_pixel_per_field(meta: pd.DataFrame) -> pd.Series:
    return meta.reset_index().groupby("field_year", observed=True)["pixel"].first().sort_values()


def times_per_field(ds: xr.Dataset, meta: pd.DataFrame) -> pd.DataFrame:
    """Calendar dates of each timestep for one representative pixel per field-year."""
    times = ds["times"]
    pdim = pixel_dim(ds)
    first = _first_pixel_per_field(meta)
    if times.dims and times.dims[0] == pdim:
        vals = times.isel({pdim: first.to_numpy()}).values
    else:
        vals = np.broadcast_to(times.values, (len(first),) + times.shape)
    vals = np.asarray(vals)
    if vals.dtype.kind != "M":
        units = times.attrs.get("units", "")
        if "since" in units:
            vals = xr.coding.times.decode_cf_datetime(vals, units, times.attrs.get("calendar"))
    df = pd.DataFrame(vals.reshape(len(first), -1), index=first.index)
    df.columns.name = "t"
    return df


def times_summary(times_df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(times_df.dtypes.iloc[0], np.datetime64):
        return times_df.describe().T
    rows = []
    for t in times_df.columns:
        s = pd.to_datetime(times_df[t])
        doy = s.dt.dayofyear
        rows.append({"t": t, "median": s.median().date(), "min": s.min().date(), "max": s.max().date(),
                     "doy_median": doy.median(), "doy_iqr_days": doy.quantile(0.75) - doy.quantile(0.25),
                     "doy_range_days": doy.max() - doy.min()})
    return pd.DataFrame(rows)


def step_lengths_days(times_df: pd.DataFrame) -> np.ndarray | None:
    """Median length (days) of each timestep, from consecutive dates; last step = previous one."""
    if not np.issubdtype(times_df.dtypes.iloc[0], np.datetime64):
        return None
    d = times_df.apply(pd.to_datetime).diff(axis=1).iloc[:, 1:]
    lengths = d.apply(lambda c: c.dt.days).median().to_numpy()
    return np.append(lengths, lengths[-1])


def seeding_type_share(meta: pd.DataFrame) -> pd.DataFrame:
    f = meta.groupby("field_year", observed=True).first()
    tab = pd.crosstab([f["crop"], f["year"]], f["seeding_date_type"], normalize="index")
    tab["n_fields"] = f.groupby(["crop", "year"], observed=True).size()
    return tab.reset_index()


def feature_stats(reader: FeatureReader, names: list[str], n_blocks: int = 20,
                  block: int = 50_000, step_days: np.ndarray | None = None) -> pd.DataFrame:
    """Statistics of selected features over evenly spaced pixel blocks.

    For temp_* features the value divided by the timestep length is reported (K/day and degC),
    to test whether the stored value is a sum of daily Kelvin temperatures.
    """
    starts = np.linspace(0, max(reader.n_pixels - block, 0), n_blocks).astype(int)
    data = np.concatenate([reader.read(names, s, min(s + block, reader.n_pixels)) for s in np.unique(starts)])
    rows = []
    for i, name in enumerate(names):
        x = data[..., i]
        v = x[np.isfinite(x)]
        if v.size == 0:
            rows.append({"feature": name, "n_finite": 0})
            continue
        vmax = v.max()
        row = {"feature": name, "n_finite": v.size, "nan_frac": 1 - v.size / x.size,
               "min": v.min(), "p01": np.percentile(v, 1), "mean": v.mean(), "p99": np.percentile(v, 99),
               "max": vmax, "frac_eq_max": np.mean(v == vmax), "frac_ge_30000": np.mean(v >= 30000),
               "n_unique": len(np.unique(v))}
        if name.startswith("temp") and step_days is not None and x.ndim == 2 and x.shape[1] == len(step_days):
            per_day = x / step_days[None, :]
            row["mean_per_day_K"] = np.nanmean(per_day)
            row["mean_per_day_degC"] = np.nanmean(per_day) - 273.15
        rows.append(row)
    return pd.DataFrame(rows)


def soil_features(names: list[str]) -> list[str]:
    return [n for n in names if any(k in n.lower() for k in SOIL_KEYWORDS)]


def temp_features(names: list[str]) -> list[str]:
    return [n for n in names if n.lower().startswith("temp")]


# --------------------------------------------------------------------------- 7. figures

def plot_yield_maps(meta: pd.DataFrame, field_years: list[str], axes=None):
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, len(field_years), figsize=(5 * len(field_years), 4))
    for ax, fy in zip(np.atleast_1d(axes), field_years):
        f = meta[meta["field_year"] == fy]
        grid = f.pivot_table(index="row", columns="col", values="target", aggfunc="mean")
        im = ax.imshow(grid.to_numpy(), cmap="viridis", origin="upper")
        ax.set_title(f"{fy}\n{f['crop'].iloc[0]}, n={len(f)} px")
        ax.set_xticks([]), ax.set_yticks([])
        plt.colorbar(im, ax=ax, label="target (t/ha)")
    return axes


def plot_ndvi(ndvi_df: pd.DataFrame):
    import matplotlib.pyplot as plt
    crops = sorted(ndvi_df["crop"].unique())
    fig, axes = plt.subplots(1, len(crops), figsize=(5 * len(crops), 4), sharey=True, squeeze=False)
    for ax, crop in zip(axes[0], crops):
        for year, g in ndvi_df[ndvi_df["crop"] == crop].groupby("year"):
            ax.plot(g["t"], g["ndvi"], marker="o", ms=3, label=year)
        ax.set_title(crop), ax.set_xlabel("timestep"), ax.legend(title="year")
    axes[0, 0].set_ylabel("mean NDVI (B08, B04)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 8. raw data

def match_raw_to_fields(raw_tree: pd.DataFrame, field_names) -> pd.DataFrame:
    """Match raw files to field_shared_name via path components (exact), then substring."""
    names = set(map(str, field_names))
    by_len = sorted(names, key=len, reverse=True)
    rows = []
    for rel in raw_tree["path"]:
        parts = [Path(rel).stem, *Path(rel).parts[:-1]]
        hit = next((p for p in parts if p in names), None)
        how = "exact" if hit else None
        if hit is None:
            hit = next((n for n in by_len if n in rel), None)
            how = "substring" if hit else None
        rows.append({"path": rel, "field": hit, "match": how})
    return pd.DataFrame(rows)


def describe_raw_file(path: str | Path) -> dict:
    """Structure of one raw file plus columns/attrs that look like quality or support-size info."""
    path = Path(path)
    suf = "".join(path.suffixes).lower()
    info: dict = {"path": str(path), "suffix": suf}
    fields: list[str] = []
    if suf.endswith((".nc", ".nc4", ".h5", ".hdf5")):
        with xr.open_dataset(path) as ds:
            info["repr"] = repr(ds)
            fields = list(ds.variables) + list(ds.attrs)
    elif suf.endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(path) as src:
            info["profile"] = dict(src.profile)
            info["descriptions"] = src.descriptions
            info["tags"] = src.tags()
            fields = [d or "" for d in src.descriptions] + list(src.tags())
    elif suf.endswith(".csv"):
        df = pd.read_csv(path, nrows=1000)
        info["head"], info["dtypes"] = df.head(), df.dtypes
        fields = list(df.columns)
    elif suf.endswith(".parquet"):
        df = pd.read_parquet(path)
        info["shape"], info["head"], info["dtypes"] = df.shape, df.head(), df.dtypes
        fields = list(df.columns)
    elif suf.endswith((".gpkg", ".shp", ".geojson")):
        import geopandas as gpd
        gdf = gpd.read_file(path, rows=1000)
        info["crs"], info["head"], info["dtypes"] = gdf.crs, gdf.head(), gdf.dtypes
        fields = list(gdf.columns)
    elif suf.endswith(".json"):
        import json
        obj = json.loads(path.read_text(encoding="utf-8"))
        info["keys"] = list(obj)[:50] if isinstance(obj, dict) else f"list[{len(obj)}]"
        fields = list(obj) if isinstance(obj, dict) else []
    elif suf.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
        info["shape"], info["dtype"] = arr.shape, arr.dtype
    else:
        info["note"] = "unknown format"
    info["quality_support_candidates"] = [
        f for f in map(str, fields) if any(re.search(k, f, re.I) for k in RAW_KEYWORDS)
    ]
    return info


# --------------------------------------------------------------------------- overview markdown

def write_overview(sections: dict[str, pd.DataFrame | str], path: str | Path, max_rows: int = 60) -> None:
    lines = ["# Data overview (YieldSAT Germany)", "", "Generated by notebooks/00_inspect.ipynb.", ""]
    for title, content in sections.items():
        lines += [f"## {title}", ""]
        if isinstance(content, pd.DataFrame):
            df = content if len(content) <= max_rows else content.head(max_rows)
            lines.append(df.to_markdown(index=False, floatfmt=".4g"))
            if len(content) > max_rows:
                lines.append(f"\n_{len(content) - max_rows} more rows omitted._")
        else:
            lines.append(str(content))
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
