"""Shared helpers for the SwissYield analysis scripts.

SwissYield is in the YieldSAT preprocessed format (same variables, same 24 time steps), so it is read with
src.data and compared against YieldSAT through the memmaps that scripts/LSTM/01_extract.py wrote
(<work_dir>/lstm/data). Only labelled pixels (non-NaN target) are used, as in training.

Outputs go to <work_dir>/swissyield/{tables,figures,cache}, never into the data folders or the repository.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_paths  # noqa: E402
from src.data import iter_blocks, load_meta, open_dataset  # noqa: E402

NC_NAME = "merge_s2-soil-dem-weather-coords.nc"
FOLDERS = {"all": "Switzerland", "DUP1": "Switzerland_DUP1", "DUP2": "Switzerland_DUP2"}
DYNAMIC = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12",
           "temp_mean", "temp_max", "temp_min", "total_prec"]
S2 = DYNAMIC[:12]
WEATHER = DYNAMIC[12:]

PATHS = load_paths()
OUT = PATHS["work_dir"] / "swissyield"
TABLES, FIGURES, CACHE, REPORT = OUT / "tables", OUT / "figures", OUT / "cache", OUT / "report"
RAW = PATHS["swissyield"] / "Raw" / "Switzerland"
for _d in (TABLES, FIGURES, CACHE, REPORT):
    _d.mkdir(parents=True, exist_ok=True)
YS_DIR = PATHS["work_dir"] / "lstm" / "data"

# Okabe-Ito, colour-blind safe
COLORS = {
    "Swiss DUP1": "#0072B2",
    "Swiss DUP2": "#D55E00",
    "Swiss all": "#000000",
    "YieldSAT Germany wheat": "#009E73",
    "YieldSAT Germany": "#CC79A7",
    "YieldSAT wheat": "#999999",
}


# --------------------------------------------------------------------------- Swiss
def swiss_ds(part: str = "all"):
    return open_dataset(PATHS["swissyield"] / "Preprocessed" / FOLDERS[part] / NC_NAME)


def swiss_meta() -> pd.DataFrame:
    """Pixel metadata of the full Swiss file plus `source` (DUP1 = farm1/Perich, DUP2 = farm2+farm3)."""
    cache = CACHE / "swiss_meta.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    meta = load_meta(swiss_ds("all"))
    dup1_fields = set(load_meta(swiss_ds("DUP1"))["field"].astype(str))
    meta["source"] = np.where(meta["field"].astype(str).isin(dup1_fields), "DUP1", "DUP2")
    for col in meta.columns:
        if meta[col].dtype == object:
            meta[col] = meta[col].astype(str)
    meta.to_parquet(cache)
    return meta


def swiss_bands(ds) -> list[str]:
    return [str(b) for b in ds["band"].values]


def swiss_static_names(ds) -> list[str]:
    return [b for b in swiss_bands(ds) if b not in DYNAMIC]


def swiss_blocks(block: int = 50_000):
    """Yield (meta rows, dyn (n,24,16), static (n,S)) for the labelled pixels of each pixel block."""
    ds, meta = swiss_ds("all"), swiss_meta()
    bands = swiss_bands(ds)
    dyn_idx = [bands.index(b) for b in DYNAMIC]
    st_idx = [bands.index(b) for b in swiss_static_names(ds)]
    labelled = meta["target"].notna().to_numpy()
    for start, stop in iter_blocks(len(meta), block):
        m = labelled[start:stop]
        if not m.any():
            continue
        arr = ds["sample"].isel(index=slice(start, stop)).transpose("index", "time_step", "band").values[m]
        yield meta.iloc[start:stop][m], arr[:, :, dyn_idx], arr[:, 0][:, st_idx]


# --------------------------------------------------------------------------- YieldSAT
def yieldsat_meta() -> pd.DataFrame:
    meta = pd.read_parquet(YS_DIR / "meta.parquet")
    meta["field_year"] = meta["field"] + "_" + meta["year"]
    return meta


def yieldsat_static_names() -> list[str]:
    return json.loads((YS_DIR / "info.json").read_text())["static_bands"]


YS_SELECT = {
    "YieldSAT Germany wheat": lambda m: (m["country"] == "Germany") & (m["crop"] == "wheat"),
    "YieldSAT Germany": lambda m: m["country"] == "Germany",
    "YieldSAT wheat": lambda m: m["crop"] == "wheat",
}


def yieldsat_blocks(select, block: int = 200_000):
    meta = yieldsat_meta()
    sel = np.flatnonzero(select(meta).to_numpy())
    dyn = np.load(YS_DIR / "dyn.npy", mmap_mode="r")
    stat = np.load(YS_DIR / "static.npy", mmap_mode="r")
    for a, b in iter_blocks(len(sel), block):
        idx = sel[a:b]
        yield meta.iloc[idx], np.asarray(dyn[idx]), np.asarray(stat[idx])


# --------------------------------------------------------------------------- derived
def ndvi(dyn: np.ndarray) -> np.ndarray:
    b8, b4 = dyn[..., DYNAMIC.index("B08")], dyn[..., DYNAMIC.index("B04")]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (b8 - b4) / (b8 + b4)


def sample_group(name: str, n: int, seed: int = 0) -> dict:
    """Random labelled-pixel sample of one comparison group: meta, dyn, static, static_names.

    Groups: 'Swiss all|DUP1|DUP2' and the YS_SELECT keys. Cached as npz + parquet.
    """
    stem = CACHE / f"sample_{name.replace(' ', '_')}_{n}_{seed}"
    if stem.with_suffix(".npz").exists():
        z = np.load(stem.with_suffix(".npz"), allow_pickle=False)
        return {"meta": pd.read_parquet(stem.with_suffix(".parquet")), "dyn": z["dyn"], "static": z["static"],
                "static_names": list(z["static_names"])}
    rng = np.random.default_rng(seed)
    if name.startswith("Swiss"):
        meta = swiss_meta()
        sel = meta["target"].notna() if name == "Swiss all" else meta["target"].notna() & (meta["source"] == name.split()[1])
        total = int(sel.sum())
        frac = min(1.0, n / total)
        parts = []
        for m, d, s in swiss_blocks():
            keep = (m["source"] == name.split()[1]).to_numpy().copy() if name != "Swiss all" else np.ones(len(m), bool)
            keep &= rng.random(len(m)) < frac
            parts.append((m[keep], d[keep], s[keep]))
        static_names = swiss_static_names(swiss_ds())
    else:
        meta = yieldsat_meta()
        sel = np.flatnonzero(YS_SELECT[name](meta).to_numpy())
        idx = np.sort(rng.choice(sel, size=min(n, len(sel)), replace=False))
        dyn = np.load(YS_DIR / "dyn.npy", mmap_mode="r")
        stat = np.load(YS_DIR / "static.npy", mmap_mode="r")
        parts = [(meta.iloc[idx], np.asarray(dyn[idx]), np.asarray(stat[idx]))]
        static_names = yieldsat_static_names()
    meta_s = pd.concat([p[0] for p in parts])
    dyn_s, st_s = np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
    meta_s.to_parquet(stem.with_suffix(".parquet"))
    np.savez(stem.with_suffix(".npz"), dyn=dyn_s, static=st_s, static_names=np.array(static_names))
    return {"meta": meta_s, "dyn": dyn_s, "static": st_s, "static_names": static_names}


def field_aggregates(name: str) -> dict:
    """Per field-year: pixel count, target mean/std, mean of every dynamic band per time step (NaN-aware),
    count of valid values per step and mean of every static band. Streams all labelled pixels once; cached."""
    stem = CACHE / f"fieldagg_{name.replace(' ', '_')}"
    if stem.with_suffix(".npz").exists():
        z = np.load(stem.with_suffix(".npz"), allow_pickle=False)
        return {"table": pd.read_parquet(stem.with_suffix(".parquet")), "dyn_mean": z["dyn_mean"],
                "dyn_n": z["dyn_n"], "static_mean": z["static_mean"], "static_names": list(z["static_names"])}
    blocks = swiss_blocks() if name.startswith("Swiss") else yieldsat_blocks(YS_SELECT[name])
    acc: dict[str, dict] = {}
    for m, dyn, stat in blocks:
        fy = m["field_year"].to_numpy()
        for key in np.unique(fy):
            k = fy == key
            a = acc.setdefault(key, {"n": 0, "t": 0.0, "t2": 0.0, "ds": 0.0, "dn": 0, "ss": 0.0,
                                     "row": m[k].iloc[0][["field", "farm", "crop", "year"]
                                                         + (["source"] if "source" in m else [])].to_dict()})
            t = m["target"].to_numpy(np.float64)[k]
            a["n"] += k.sum()
            a["t"] += t.sum()
            a["t2"] += (t ** 2).sum()
            a["ds"] = a["ds"] + np.nansum(dyn[k], axis=0)
            a["dn"] = a["dn"] + (~np.isnan(dyn[k])).sum(axis=0)
            a["ss"] = a["ss"] + np.nansum(stat[k], axis=0)
    keys = sorted(acc)
    table = pd.DataFrame([{"field_year": k, **acc[k]["row"], "n_pixels": acc[k]["n"],
                           "yield_mean": acc[k]["t"] / acc[k]["n"],
                           "yield_std": np.sqrt(max(acc[k]["t2"] / acc[k]["n"] - (acc[k]["t"] / acc[k]["n"]) ** 2, 0))}
                          for k in keys])
    dyn_n = np.stack([acc[k]["dn"] for k in keys])
    dyn_mean = np.stack([acc[k]["ds"] for k in keys]) / np.where(dyn_n == 0, np.nan, dyn_n)
    static_mean = np.stack([acc[k]["ss"] / acc[k]["n"] for k in keys])
    static_names = swiss_static_names(swiss_ds()) if name.startswith("Swiss") else yieldsat_static_names()
    table.to_parquet(stem.with_suffix(".parquet"))
    np.savez(stem.with_suffix(".npz"), dyn_mean=dyn_mean, dyn_n=dyn_n, static_mean=static_mean,
             static_names=np.array(static_names))
    return {"table": table, "dyn_mean": dyn_mean, "dyn_n": dyn_n, "static_mean": static_mean,
            "static_names": static_names}


# --------------------------------------------------------------------------- plotting
def clip_range(arrays: list[np.ndarray], lo: float = 0.5, hi: float = 99.5) -> tuple[float, float]:
    """Common plotting range: widest [lo, hi] percentile range over the groups."""
    pairs = [np.nanpercentile(a, [lo, hi]) for a in arrays if np.isfinite(a).any()]
    return min(p[0] for p in pairs), max(p[1] for p in pairs)


def draw_hists(ax, series: dict[str, np.ndarray], bins, density: bool = True, **kw) -> None:
    """Overlaid step histograms (density by default so differently sized groups compare)."""
    for name, a in series.items():
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            continue
        ax.hist(a, bins=bins, density=density, histtype="step", lw=1.5, color=COLORS.get(name), label=name, **kw)
    ax.tick_params(labelsize=8)


def save(fig, name: str) -> Path:
    path = FIGURES / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)
    return path


# --------------------------------------------------------------------------- report
def md_table(df: pd.DataFrame, floatfmt: str = "{:.4g}") -> str:
    df = df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)) + "\n"


class Report:
    """Collects a markdown section (also printed) and writes it to <work_dir>/swissyield/report/<name>.md;
    99_assemble_report.py joins the sections into profile.md."""

    def __init__(self, name: str, title: str):
        self.name, self.parts = name, [f"# {title}\n"]
        print(f"\n##### {title}")

    def h(self, text: str) -> None:
        self.parts.append(f"\n## {text}\n")
        print(f"\n== {text}")

    def p(self, text: str) -> None:
        self.parts.append(text + "\n")
        print(text)

    def table(self, df: pd.DataFrame, **kw) -> None:
        self.parts.append(md_table(df, **kw))
        print(df.to_string(max_rows=60))

    def save(self) -> None:
        path = REPORT / f"{self.name}.md"
        path.write_text("\n".join(self.parts))
        print("wrote", path)
