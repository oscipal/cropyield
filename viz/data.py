"""Data access for the visualizer. Everything is read from the derived files in <work_dir> (never the raw data
folders); large arrays are memory-mapped and sliced one field-year at a time."""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_paths  # noqa: E402
from src.splits import load_split  # noqa: E402

PATHS = load_paths()
WORK = PATHS["work_dir"]
RUNS = WORK / "lstm" / "runs"
GRIDS = {"30-day (lstm/data)": "data", "10-day": "data_w10", "15-day": "data_w15", "30-day S2 only": "data_w30"}
WEATHER = ["temp_mean", "temp_max", "temp_min", "total_prec"]
warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN steps outside the season
SETS = {0: "train", 1: "val", 2: "test"}
# default configs on the Runs page: the models of docs/results_summary.md (scripts/LSTM/10_summary.py)
SUMMARY_MODELS = ["mean", "mean_crop_country", "lstm_s2", "lstm_s2_adm", "lstm_v2_s2", "lstm_patch_s2",
                  "lstm_patch_v3_s2", "convlstm_s2"]


# --------------------------------------------------------------------------- fields

@st.cache_data
def fields_table() -> pd.DataFrame:
    return pd.read_parquet(WORK / "fields.parquet")


@st.cache_data
def field_index() -> pd.DataFrame:
    path = WORK / "viz" / "field_index.parquet"
    if not path.exists():
        st.error(f"{path} is missing: run `python viz/build_index.py` once.")
        st.stop()
    return pd.read_parquet(path).set_index("field")


@st.cache_data
def field_shapes():
    """Display outline per field-year (EPSG:4326, viz/build_index.py) with the field table columns; fields without
    a polygon in field_shapes.gpkg are absent."""
    import geopandas as gpd
    path = WORK / "viz" / "field_shapes_display.parquet"
    if not path.exists():
        st.error(f"{path} is missing: run `python viz/build_index.py` once.")
        st.stop()
    return gpd.read_parquet(path).merge(fields_table(), on="field", how="inner")


# --------------------------------------------------------------------------- pixel inputs

@st.cache_data
def grid_info(grid: str) -> dict:
    return json.loads((WORK / "lstm" / grid / "info.json").read_text())


@st.cache_resource
def _arrays(grid: str) -> tuple[np.ndarray, np.ndarray]:
    d = WORK / "lstm" / grid
    return np.load(d / "dyn.npy", mmap_mode="r"), np.load(d / "static.npy", mmap_mode="r")


def time_labels(grid: str) -> list[str]:
    """Window start of every time step; day 0 = 1 January of the year before harvest (Y-1)."""
    info = grid_info(grid)
    start, days = info.get("start_day", 0), info.get("window_days", 30)
    out = []
    for k in range(info["n_times"]):
        d = start + k * days
        yr = "Y-1" if d < 365 else "Y"
        out.append(f"{(date(2001, 1, 1) + timedelta(days=d % 365)).strftime('%d %b')} {yr}")
    return out


@st.cache_data(max_entries=8)
def field_inputs(field: str, grid: str) -> dict:
    """Pixels of one field-year: grid position inside its bounding box, target, dynamic (n, T, B) float32 with
    NaN = missing, static (n, 104), band names."""
    ix = field_index().loc[field]
    a, b = int(ix["start"]), int(ix["end"])
    dyn_mm, static_mm = _arrays(grid)
    dyn = np.asarray(dyn_mm[a:b])
    if dyn.dtype == np.uint16:
        miss = dyn == grid_info(grid).get("missing_value", 65535)
        dyn = dyn.astype(np.float32)
        dyn[miss] = np.nan
    meta = pq.ParquetFile(WORK / "lstm" / "data" / "meta.parquet")
    rows = _read_rows(meta, a, b, ["row", "col", "target"])
    info = grid_info(grid)
    return {"lr": rows["row"].to_numpy(np.int32) - int(ix["row_min"]),
            "lc": rows["col"].to_numpy(np.int32) - int(ix["col_min"]),
            "h": int(ix["row_max"] - ix["row_min"] + 1), "w": int(ix["col_max"] - ix["col_min"] + 1),
            "target": rows["target"].to_numpy(np.float32), "dyn": dyn, "static": np.asarray(static_mm[a:b]),
            "dyn_bands": info["dynamic_bands"], "static_bands": grid_info("data")["static_bands"]}


def _read_rows(pf: pq.ParquetFile, a: int, b: int, columns: list[str]) -> pd.DataFrame:
    """Rows [a, b) of a parquet file, reading only the row groups that hold them."""
    bounds = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
    groups = [i for i in range(pf.num_row_groups) if bounds[i] < b and bounds[i + 1] > a]
    t = pf.read_row_groups(groups, columns=columns).to_pandas()
    return t.iloc[a - bounds[groups[0]]: b - bounds[groups[0]]].reset_index(drop=True)


def to_image(values: np.ndarray, lr: np.ndarray, lc: np.ndarray, h: int, w: int) -> np.ndarray:
    """Scatter per-pixel values (n,) or (n, C) into an (h, w[, C]) image, NaN where there is no pixel."""
    img = np.full((h, w) + values.shape[1:], np.nan, dtype=np.float32)
    img[lr, lc] = values
    return img


def band(inp: dict, name: str) -> np.ndarray:
    """(n, T) values of one dynamic band."""
    return inp["dyn"][:, :, inp["dyn_bands"].index(name)]


def ndvi(inp: dict) -> np.ndarray:
    nir, red = band(inp, "B08"), band(inp, "B04")
    with np.errstate(divide="ignore", invalid="ignore"):
        return (nir - red) / (nir + red)


# --------------------------------------------------------------------------- runs

def _parse_split(name: str) -> tuple[str, str, str]:
    scope, method, crop = name.split("__")
    return scope, method, crop


@st.cache_data(ttl=120)
def list_runs() -> pd.DataFrame:
    """Every <config>/<scope>__<method>__<crop> folder, finished (metrics.json) or not."""
    rows = []
    for cfg in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for d in sorted(p for p in cfg.iterdir() if p.is_dir() and p.name.count("__") == 2):
            scope, method, crop = _parse_split(d.name)
            rows.append({"config": cfg.name, "split": f"{scope}/{method}/{crop}", "scope": scope, "method": method,
                         "crop": crop, "path": str(d), "done": (d / "metrics.json").exists(),
                         "has_val": (d / "preds_val.parquet").exists(),
                         "has_history": (d / "history.csv").exists()})
    return pd.DataFrame(rows)


@st.cache_data(ttl=120)
def metrics_long() -> pd.DataFrame:
    """One row per run x set x group ("all" or "Country|crop") with the ensemble metrics."""
    rows = []
    for r in list_runs().query("done").itertuples():
        m = json.loads((Path(r.path) / "metrics.json").read_text())
        for role, groups in m["metrics"].items():
            members = m.get("metrics_members", {}).get(role) or []
            for g, vals in groups.items():
                spread = [mm[g]["field_r2"] for mm in members if g in mm]
                rows.append({"config": r.config, "split": r.split, "scope": r.scope, "method": r.method,
                             "crop": r.crop, "set": role, "group": g, **vals,
                             "member_std_field_r2": float(np.std(spread)) if len(spread) > 1 else np.nan,
                             "best_epoch": m.get("best_epoch"), "path": r.path})
    return pd.DataFrame(rows)


@st.cache_data(ttl=120)
def run_summary(path: str) -> dict:
    m = json.loads((Path(path) / "metrics.json").read_text())
    return {k: m.get(k) for k in ("split", "config", "best_epoch", "epochs_run", "n_pixels", "n_fields")}


@st.cache_data(ttl=120)
def history(path: str) -> pd.DataFrame:
    return pd.read_csv(Path(path) / "history.csv")


@st.cache_data(max_entries=3)
def preds(path: str, role: str) -> pd.DataFrame:
    pf = pq.ParquetFile(Path(path) / f"preds_{role}.parquet")
    cols = [c for c in ("field", "country", "crop", "year", "row", "col", "target", "pred", "pred_std")
            if c in pf.schema_arrow.names]
    return pf.read(columns=cols).to_pandas()


@st.cache_data(max_entries=3)
def field_level(path: str, role: str) -> pd.DataFrame:
    """Field-year means of target and prediction, sorted by absolute error."""
    p = preds(path, role)
    f = p.groupby("field", sort=False).agg(country=("country", "first"), crop=("crop", "first"),
                                           year=("year", "first"), target=("target", "mean"),
                                           pred=("pred", "mean"), n_pixels=("target", "size")).reset_index()
    f["error"] = f["pred"] - f["target"]
    return f.sort_values("error", key=np.abs, ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- splits

@st.cache_data
def split_groups() -> pd.DataFrame:
    import zarr
    g = zarr.open_group(str(PATHS["split_suite"]), mode="r")
    rows = [{"scope": s, "method": m, "crop": c}
            for s in sorted(g.group_keys()) for m in sorted(g[s].group_keys()) for c in sorted(g[s][m].group_keys())]
    return pd.DataFrame(rows)


@st.cache_data
def split_table(scope: str, method: str, crop: str) -> tuple[pd.DataFrame, dict]:
    """Field table of one split with a ``set`` column (train/val/test) and the split's attributes."""
    s = load_split(PATHS["split_suite"], scope, method, crop)
    sets = {f: name for name in ("train", "val", "test") for f in s[f"{name}_fields"]}
    t = fields_table()
    t = t[t["field"].isin(sets)].copy()
    t["set"] = t["field"].map(sets)
    return t.reset_index(drop=True), s["attrs"]


# --------------------------------------------------------------------------- season, weather, scenes

@st.cache_data
def field_dates() -> pd.DataFrame:
    """Sowing / harvest dates and their day on the cube axis (viz/build_index.py, src/season.py)."""
    return pd.read_parquet(WORK / "viz" / "field_dates.parquet").set_index("field")


def step_dates(field: str, grid: str) -> pd.DataFrame:
    """Calendar start / end date of every window of a grid for one field-year, and the share of its days in season."""
    info, s = grid_info(grid), field_dates().loc[field]
    start, days = info.get("start_day", 0), info.get("window_days", 30)
    k = np.arange(info["n_times"])
    a, b = start + k * days, start + (k + 1) * days - 1
    jan1 = pd.Timestamp(s["jan1"])
    lo, hi = int(s["sow_day"]), int(s["harvest_day"])
    in_season = (np.minimum(b, hi) - np.maximum(a, lo) + 1).clip(0) / days
    # day d is the date jan1 + d + 1 (src/season.py: day = (date - jan1).days - 1)
    return pd.DataFrame({"step": k, "start": jan1 + pd.to_timedelta(a + 1, "D"),
                         "end": jan1 + pd.to_timedelta(b + 1, "D"), "in_season": in_season})


@st.cache_data
def weather_grid(grid: str) -> tuple[np.ndarray, dict]:
    """Weather features per field-year on a grid (scripts/LSTM/13_build_weather_grid.py)."""
    d = WORK / "lstm" / grid
    if not (d / "weather.npy").exists():
        return None, None
    return np.load(d / "weather.npy"), json.loads((d / "weather.json").read_text())


def field_weather(field: str, grid: str) -> pd.DataFrame | None:
    arr, info = weather_grid(grid)
    if arr is None:
        return None
    w = pd.DataFrame(arr[info["fields"].index(field)], columns=info["features"])
    return pd.concat([step_dates(field, grid), w], axis=1)


@st.cache_data(max_entries=16)
def daily_weather(field: str) -> pd.DataFrame:
    """The field's raw daily ERA5 CSV in readable units (°C, mm); read by the app, not printed anywhere."""
    from src.config import country_dirs
    country = fields_table().set_index("field").at[field, "country"]
    w = pd.read_csv(next((country_dirs(PATHS, country)[1] / field / "weather").glob("*.csv")), parse_dates=["Date"])
    out = pd.DataFrame({"date": w["Date"], "tmean_c": w["Temp_mean"] - 273.15, "tmax_c": w["Temp_max"] - 273.15,
                        "tmin_c": w["Temp_min"] - 273.15, "prec_mm": w["Total_prec"] * 1000.0})
    for src, dst in (("RH_mean", "rh_mean"), ("S_rad", "srad")):
        if src in w:
            out[dst] = w[src]
    s = field_dates().loc[field]
    sown = out["date"] >= pd.Timestamp(s["seeding_date"])
    in_season = sown & (out["date"] <= pd.Timestamp(s["harvesting_date"]))
    out["prec_cum_mm"] = out["prec_mm"].where(in_season, 0.0).cumsum().where(sown)
    return out


@st.cache_data
def scene_windows() -> pd.DataFrame | None:
    """Per field x 30-day window: chosen scene, valid shares and class (scripts/LSTM/12_scene_availability.py)."""
    path = WORK / "scene_availability" / "windows_w30.parquet"
    return pd.read_parquet(path).set_index(["field", "step"]) if path.exists() else None


# --------------------------------------------------------------------------- soil and terrain

DEPTHS = ["0-5", "5-15", "15-30", "30-60", "60-100", "100-200"]
# SoilGrids 2.0 mapped units as stored in the cube (sand + silt + clay = 1000 g/kg): name, display unit, factor
SOIL = {"clay": ("Clay", "%", 0.1), "silt": ("Silt", "%", 0.1), "sand": ("Sand", "%", 0.1),
        "phh2o": ("pH (H₂O)", "", 0.1), "soc": ("Organic carbon", "g/kg", 0.1),
        "cec": ("Cation exchange capacity", "cmol(c)/kg", 0.1), "nitrogen": ("Nitrogen", "g/kg", 0.01),
        "cfvo": ("Coarse fragments", "vol %", 0.1)}
TERRAIN = {"dem": ("Elevation", "m"), "slope": ("Slope", "RichDEM, as delivered"), "aspect": ("Aspect", "°"),
           "twi": ("Topographic wetness index", ""), "curvature": ("Curvature", "")}
TEXTURE = ["sand", "loamy sand", "sandy loam", "loam", "silt loam", "silt", "sandy clay loam", "clay loam",
           "silty clay loam", "sandy clay", "silty clay", "clay"]


def soil_values(inp: dict, prop: str, depth: str, uncertainty: bool = False) -> np.ndarray:
    name = f"{prop}_{depth}" + ("_uncertainty" if uncertainty else "")
    return inp["static"][:, inp["static_bands"].index(name)] * SOIL[prop][2]


def texture_class(sand, silt, clay) -> np.ndarray:
    """USDA soil texture class index into TEXTURE (-1 = no data) from sand / silt / clay in %."""
    sand, silt, clay = (np.asarray(x, dtype=float) for x in (sand, silt, clay))
    tot = sand + silt + clay
    with np.errstate(invalid="ignore", divide="ignore"):
        sand, silt, clay = sand / tot * 100, silt / tot * 100, clay / tot * 100
    conds = [
        silt + 1.5 * clay < 15,
        (silt + 1.5 * clay >= 15) & (silt + 2 * clay < 30),
        ((clay >= 7) & (clay < 20) & (sand > 52) & (silt + 2 * clay >= 30))
        | ((clay < 7) & (silt < 50) & (silt + 2 * clay >= 30)),
        (clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand <= 52),
        ((silt >= 50) & (clay >= 12) & (clay < 27)) | ((silt >= 50) & (silt < 80) & (clay < 12)),
        (silt >= 80) & (clay < 12),
        (clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45),
        (clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45),
        (clay >= 27) & (clay < 40) & (sand <= 20),
        (clay >= 35) & (sand > 45),
        (clay >= 40) & (silt >= 40),
        (clay >= 40) & (sand <= 45) & (silt < 40),
    ]
    out = np.select(conds, np.arange(len(TEXTURE)), default=-1)
    return np.where(np.isfinite(tot) & (tot > 0), out, -1)


# --------------------------------------------------------------------------- models

# menu entry -> configs (first = main config); configs on disk but not listed go to "Other"
MODEL_FAMILIES = {
    "Baselines": ["mean", "mean_crop_country"],
    "LSTM v1 (paper)": ["lstm_s2", "lstm_s2_adm"],
    "LSTM v2": ["lstm_v2_s2"],
    "Patch LSTM": ["lstm_patch_v3_s2", "lstm_patch_s2", "lstm_patch_v3_w10_s2", "patch_aug_spec", "patch_aug_time",
                   "patch_soil_dem", "patch_weather"],
    "ConvLSTM": ["convlstm_s2", "convlstm_base", "convlstm_h64", "convlstm_k9", "convlstm_l2", "convlstm_w10",
                 "convlstm_w10_s1", "convlstm_w15", "convlstm_w30"],
    "Field image": ["field_image_s2_screen", "field_image_s2_crop256", "field_image_s2_fmean",
                    "field_image_s2_small", "field_image_s2_tf"],
}


@st.cache_data(ttl=120)
def model_families() -> dict[str, list[str]]:
    """MODEL_FAMILIES restricted to configs that have a run folder, plus "Other" for unlisted ones."""
    on_disk = list(dict.fromkeys(list_runs()["config"]))
    fam = {k: [c for c in v if c in on_disk] for k, v in MODEL_FAMILIES.items()}
    fam = {k: v for k, v in fam.items() if v}
    listed = {c for v in fam.values() for c in v}
    other = [c for c in on_disk if c not in listed]
    return fam | ({"Other": other} if other else {})


def family_of(config: str) -> str:
    return next((k for k, v in model_families().items() if config in v), "Other")


@st.cache_data
def config_info(config: str) -> dict:
    """Config file, its header comment (the screens describe their change there) and the summary description."""
    from src.config import REPO_ROOT
    path = next(iter(sorted((REPO_ROOT / "configs").rglob(f"{config}.yaml"))), None)
    text = path.read_text() if path else ""
    header = " ".join(l.lstrip("# ").strip() for l in text.splitlines()[:6] if l.startswith("#"))
    return {"path": str(path.relative_to(REPO_ROOT)) if path else None, "yaml": text, "header": header,
            "desc": _summary_descriptions().get(config, ""),
            "screen": bool(path) and path.parent.name.endswith("_screen")}


@st.cache_data
def _summary_descriptions() -> dict[str, str]:
    """Model descriptions of scripts/LSTM/10_summary.py (config -> text)."""
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("summary10", REPO / "scripts" / "LSTM" / "10_summary.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {cfg.strip("`"): text for _, cfg, text in mod.DESCRIPTIONS}
    except Exception:   # the report module is optional for the viewer
        return {}


REPO = Path(__file__).resolve().parents[1]


@st.cache_data
def split_membership() -> pd.DataFrame:
    """Set (train / val / test) of every field-year in every split of the suite."""
    rows = []
    for g in split_groups().itertuples():
        t, _ = split_table(g.scope, g.method, g.crop)
        rows.append(pd.DataFrame({"field": t["field"], "split": f"{g.scope}/{g.method}/{g.crop}", "set": t["set"]}))
    return pd.concat(rows, ignore_index=True)
