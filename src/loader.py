"""One loader for all model inputs: pick the features (src/features.py), the time grid and axis, and the pixels.

    from src.loader import load, fit_scaler, YieldDataset

    d = load("world/loyo/all", role="train", grid="10d",
             features=["s2", "indices", "observed", "weather:tmean_c,prec_mm", "terrain", "soil:0-30cm", "crop"])
    d.dyn       (pixels, T, F_dyn)  physical units, NaN = missing (no S2 observation / outside the weather record)
    d.static    (pixels, F_static)
    d.y         yield t/ha;  d.meta: field, farm, crop, country, year, row, col
    d.dyn_names, d.static_names, d.units;  d.days: start day of every step relative to the time origin (`align`)
    sc = fit_scaler(d)                              # on the training set
    ds = YieldDataset(d, sc)                        # torch Dataset: normalised, NaN -> 0; batched __getitems__
    # or normalised at load time (one float16 rounding, as the training script does):
    sc = fit_scaler(split="world/loyo/all", features=[...], grid="10d")      # streamed, nothing kept
    d = load("world/loyo/all", "val", features=[...], grid="10d", scaler=sc)  # z-scored, NaN -> 0

Time axis (dynamic features):
    grid     "30d", "15d", "10d": windows of the S2 cubes rebuilt from the raw scenes (scripts/LSTM/11_build_s2_grid.py,
             start day 210, best scene per window, clear pixels only) with the weather on the same windows
             (scripts/LSTM/13_build_weather_grid.py).
    align    "calendar": day 0 = 1 January of the year before the harvest year (src/season.py), the grid as built.
             "sowing" / "harvest": every field-year shifted so that step 0 is the window that contains its sowing /
             harvest date. Shifts are whole windows, so the alignment is exact to one window (10 / 15 / 30 days).
             Steps that fall outside the built grid are missing (NaN, observed 0).
    season   (first, last) day relative to the origin; keeps the steps whose start day d.days is in [first, last).
             Defaults: calendar (210, 720) = the whole grid, sowing (0, 360), harvest (-360, 0) (everything up to
             the harvest window). E.g. align="harvest", season=(-360, -60): inputs end two months before harvest.
             Sowing dates: Uruguay all "Estimated" (day 314-315); 16 German and 19 Swiss field-years use a fixed
             290 / 333 days before harvest.

Pixels (applied in this order, the same for every feature):
    split + role, or fields       which field-years
    countries, crops, years       keep only these
    edge_buffer_m                 drop pixels closer than this to the field edge (distance from the pixel centre
                                  to the nearest pixel centre without yield, so 10 = outermost ring; holes in the
                                  yield map count as edge). The Swiss Perich farm is already buffered by 20 m.
    min_observed                  drop pixels with fewer clear S2 observations on the selected steps
    pixels_per_field, seed        random subsample per field-year
    patch=k                       any odd k: neighbour table (pixels, k*k) over the loaded pixels of the same field-year
                                  (-1 = no pixel there, or not loaded because of a filter / subsample)

Terrain: scripts/LSTM/14_build_terrain.py. Soil: src/soil.py. Memory: d.dyn is held in RAM (world 10-day training
set, S2 + indices + observed: 8.3 M pixels x 51 x 16 x 2 bytes = 13.5 GB as float16, the default); the size is
printed before loading.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src import features as F
from src import soil
from src.config import country_dirs, load_paths
from src.season import read_season
from src.splits import load_split

GRIDS = {"30d": "data_w30", "15d": "data_w15", "10d": "data_w10"}
ROLES = ("train", "val", "test")
ALIGN = ("calendar", "sowing", "harvest")
DEFAULT_SEASON = {"sowing": (0, 360), "harvest": (-360, 0)}   # calendar: the whole grid
S2_MISSING = 65535
PIXEL_M = 10.0
BLOCK = 200_000


@dataclass
class Data:
    dyn: np.ndarray
    static: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    dyn_names: list[str]
    static_names: list[str]
    units: dict[str, str]
    days: np.ndarray                 # start day of every step relative to the origin of `align`
    grid: str
    rows: np.ndarray                 # pixel rows in lstm/data/meta.parquet
    nbr: np.ndarray | None = None    # (pixels, k*k) neighbour positions in this Data, -1 = absent / not loaded
    normalized: bool = False         # loaded with a scaler: z-scored, NaN -> 0
    align: str = "calendar"

    def __len__(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:
        return (f"Data({len(self):,} pixels, {self.meta['field'].nunique():,} fields, grid {self.grid}, "
                f"align {self.align} days {self.days[0]}..{self.days[-1] if len(self.days) else '-'}, "
                f"dyn {self.dyn.shape} {self.dyn.dtype}, static {self.static.shape})")


# --------------------------------------------------------------------------- shared tables

@lru_cache(maxsize=1)
def _base() -> dict:
    work = load_paths()["work_dir"] / "lstm"
    info = json.loads((work / "data" / "info.json").read_text())
    meta = pd.read_parquet(work / "data" / "meta.parquet")
    meta["country"] = meta["country"].astype(str)
    f = meta["field"].to_numpy()
    start = np.r_[0, np.flatnonzero(f[1:] != f[:-1]) + 1]
    fields = pd.DataFrame({"field": f[start], "country": meta["country"].to_numpy()[start], "start": start,
                           "stop": np.r_[start[1:], len(f)]}).set_index("field")
    assert fields.index.is_unique, "rows of a field-year are not contiguous in meta.parquet"
    return {"work": work, "meta": meta, "fields": fields, "static_names": info["static_bands"]}


@lru_cache(maxsize=1)
def seasons() -> pd.DataFrame:
    """sow_day, harvest_day (on the cube's day axis) and seeding_date_type per field-year, from the raw metadata
    JSONs (src/season.py); cached in <work_dir>/lstm/seasons.parquet."""
    base = _base()
    path = base["work"] / "seasons.parquet"
    if path.exists():
        return pd.read_parquet(path)
    paths = load_paths()
    rows = []
    for field, r in base["fields"].iterrows():
        s = read_season(country_dirs(paths, r["country"])[1] / field)
        rows.append({"field": field, "sow_day": s["sow_day"], "harvest_day": s["harvest_day"],
                     "seeding_date_type": s["seeding_date_type"]})
    t = pd.DataFrame(rows).set_index("field")
    t.to_parquet(path)
    return t


# --------------------------------------------------------------------------- pixel selection

def select_rows(split: str | None, role: str | list[str], fields: list[str] | None = None) -> np.ndarray:
    """Pixel rows (sorted) of a split role, or of explicit field names, or of everything."""
    meta = _base()["meta"]
    if fields is None and split is not None:
        scope, method, crop = split.split("/")
        s = load_split(load_paths()["split_suite"], scope, method, crop)
        roles = [role] if isinstance(role, str) else list(role)
        bad = [r for r in roles if r not in ROLES]
        if bad:
            raise ValueError(f"role must be in {ROLES}, got {bad}")
        fields = [f for r in roles for f in s[f"{r}_fields"]]
    if fields is None:
        return np.arange(len(meta))
    rows = np.flatnonzero(meta["field"].isin(fields).to_numpy())
    missing = set(fields) - set(meta["field"].iloc[rows].unique())
    if missing:
        raise KeyError(f"{len(missing)} fields have no pixels, e.g. {sorted(missing)[:3]}")
    return rows


def subsample(rows: np.ndarray, field: np.ndarray, per_field: int, seed: int) -> np.ndarray:
    """At most per_field random rows of every field (fixed by seed), sorted."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    keep = pd.Series(order).groupby(field[order]).head(per_field).to_numpy()
    return np.sort(rows[keep])


def edge_distance(rows: np.ndarray) -> np.ndarray:
    """Distance (m) from each pixel centre to the nearest pixel centre of its field-year raster without yield
    (outside the field, or a hole in the yield map); computed over all labelled pixels of the field-year."""
    from scipy.ndimage import distance_transform_edt
    base = _base()
    meta, fields = base["meta"], base["fields"]
    r_all, c_all = meta["row"].to_numpy().astype(np.int64), meta["col"].to_numpy().astype(np.int64)
    out = np.empty(len(rows), dtype=np.float32)
    field = meta["field"].to_numpy()[rows]
    bounds = np.r_[0, np.flatnonzero(field[1:] != field[:-1]) + 1, len(rows)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        f = fields.loc[field[a]]
        rr, cc = r_all[f["start"]:f["stop"]], c_all[f["start"]:f["stop"]]
        r0, c0 = rr.min() - 1, cc.min() - 1
        mask = np.zeros((rr.max() - r0 + 2, cc.max() - c0 + 2), dtype=bool)   # 1-pixel empty border
        mask[rr - r0, cc - c0] = True
        dist = distance_transform_edt(mask) * PIXEL_M
        out[a:b] = dist[r_all[rows[a:b]] - r0, c_all[rows[a:b]] - c0]
    return out


def neighbours(rows: np.ndarray, k: int) -> np.ndarray:
    """(len(rows), k*k) positions in `rows` of the k x k neighbourhood (row-major, centre at k*k // 2) within the
    same field-year; -1 where there is no labelled pixel or it is not in `rows`."""
    if k < 1 or k % 2 == 0:
        raise ValueError(f"patch size must be odd, got {k}")
    base = _base()
    meta = base["meta"]
    r_all, c_all = meta["row"].to_numpy().astype(np.int64), meta["col"].to_numpy().astype(np.int64)
    field = meta["field"].to_numpy()[rows]
    h = k // 2
    out = np.full((len(rows), k * k), -1, dtype=np.int64)
    bounds = np.r_[0, np.flatnonzero(field[1:] != field[:-1]) + 1, len(rows)]
    dr, dc = (x.ravel() for x in np.mgrid[-h:h + 1, -h:h + 1])
    for a, b in zip(bounds[:-1], bounds[1:]):
        rr, cc = r_all[rows[a:b]], c_all[rows[a:b]]
        r0, c0 = rr.min() - h, cc.min() - h
        grid = np.full((rr.max() - r0 + h + 1, cc.max() - c0 + h + 1), -1, dtype=np.int64)
        grid[rr - r0, cc - c0] = np.arange(a, b)
        out[a:b] = grid[(rr - r0)[:, None] + dr[None, :], (cc - c0)[:, None] + dc[None, :]]
    return out


# --------------------------------------------------------------------------- plan

def _plan(split=None, role="train", features=("s2", "indices", "observed"), grid="10d", fields=None,
          countries=None, crops=None, years=None, edge_buffer_m=0.0, min_observed=0, pixels_per_field=None,
          seed=0, align="calendar", season=None) -> dict:
    """Resolved features, time axis and pixel rows of a load() / fit_scaler() call."""
    if grid not in GRIDS:
        raise ValueError(f"grid must be one of {list(GRIDS)}, got {grid!r}")
    if align not in ALIGN:
        raise ValueError(f"align must be one of {ALIGN}, got {align!r}")
    base = _base()
    meta_all, work = base["meta"], base["work"]
    gdir = work / GRIDS[grid]
    info = json.loads((gdir / "info.json").read_text())
    T, start, wd = info["n_times"], info["start_day"], info["window_days"]
    cats = {c: sorted(meta_all[c].astype(str).unique()) for c in F.CATEGORICAL}
    groups = F.resolve(list(features), base["static_names"], cats)
    dyn_g = [g for g in groups if g.dynamic]

    # time axis: step m of the output = grid window k_f + m of field f, d.days = m * wd
    lo, hi = season if season is not None else DEFAULT_SEASON.get(align, (start, start + T * wd))
    if not lo < hi:
        raise ValueError(f"season must be (first, last) with first < last, got {season}")
    m = np.arange(-(-lo // wd), -(-hi // wd))           # steps whose start day m * wd lies in [lo, hi)
    plan = {"meta_all": meta_all, "work": work, "gdir": gdir, "groups": groups, "dyn_g": dyn_g,
            "sta_g": [g for g in groups if not g.dynamic], "T_grid": T, "start": start, "wd": wd, "steps": m,
            "align": align}

    rows = select_rows(split, role, fields)
    for col, keep in (("country", countries), ("crop", crops), ("year", years)):
        if keep is not None:
            v = meta_all[col].to_numpy()[rows]
            rows = rows[np.isin(v.astype(str), [str(x) for x in keep])]
    if edge_buffer_m:
        rows = rows[edge_distance(rows) > edge_buffer_m]
    if min_observed:
        obs = [g for g in F.resolve(["observed"], base["static_names"], cats)]
        cnt = np.concatenate([x[..., 0].sum(1) for _, x in _dyn_blocks(plan, obs, rows)]) if len(rows) else []
        rows = rows[np.asarray(cnt) >= min_observed]
    if pixels_per_field:
        rows = subsample(rows, meta_all["field"].to_numpy()[rows], pixels_per_field, seed)
    plan["rows"] = rows
    plan["dyn_names"] = [n for g in dyn_g for n in g.names]
    plan["sta_names"] = [n for g in plan["sta_g"] for n in g.names]
    return plan


def _origin_window(plan: dict, field: np.ndarray) -> np.ndarray:
    """Grid window index of the time origin of every row's field-year (may lie outside the grid)."""
    start, wd = plan["start"], plan["wd"]
    if plan["align"] == "calendar":
        return np.full(len(field), -(start // wd), dtype=np.int64)   # day 0
    s = seasons()
    day = s[{"sowing": "sow_day", "harvest": "harvest_day"}[plan["align"]]].reindex(field).to_numpy()
    if pd.isna(day).any():
        raise ValueError(f"{int(pd.isna(day).sum())} pixels have no {plan['align']} date")
    return np.floor_divide(day.astype(np.int64) - start, wd)


# --------------------------------------------------------------------------- loading

def load(split: str | None = None, role: str | list[str] = "train", features: list[str] = ("s2", "indices",
         "observed"), grid: str = "10d", fields: list[str] | None = None, *, align: str = "calendar",
         season: tuple[int, int] | None = None, countries: list[str] | None = None, crops: list[str] | None = None,
         years: list[int] | None = None, edge_buffer_m: float = 0.0, min_observed: int = 0,
         pixels_per_field: int | None = None, seed: int = 0, patch: int | None = None, dtype=np.float16,
         scaler: Scaler | None = None, verbose: bool = True) -> Data:
    """Inputs and target of one split role (e.g. "world/loyo/all", "train"), or of explicit `fields`, or of all
    pixels (split=None). See the module docstring. With `scaler` (from fit_scaler) the features are z-scored in
    float32 before they are stored and NaN -> 0 (d.normalized = True): one rounding to `dtype` only, the form
    the training script uses."""
    p = _plan(split, role, features, grid, fields, countries, crops, years, edge_buffer_m, min_observed,
              pixels_per_field, seed, align, season)
    meta_all, rows, steps = p["meta_all"], p["rows"], p["steps"]
    dyn_names, sta_names = p["dyn_names"], p["sta_names"]
    n, T = len(rows), len(steps)
    if verbose:
        gb = n * T * len(dyn_names) * np.dtype(dtype).itemsize / 1e9
        print(f"load {split or 'fields'} {role if split else ''}: {n:,} pixels, grid {grid}, align {align} "
              f"({T} steps, days {steps[0] * p['wd'] if T else '-'}..{steps[-1] * p['wd'] if T else '-'}), "
              f"{len(dyn_names)} dynamic + {len(sta_names)} static features, dyn {gb:.1f} GB {np.dtype(dtype)}",
              flush=True)

    meta = meta_all.iloc[rows].reset_index(drop=True)
    dyn = np.empty((n, T, len(dyn_names)), dtype=dtype)
    for a, x in _dyn_blocks(p, p["dyn_g"], rows):
        dyn[a:a + len(x)] = x if scaler is None else _apply(x, scaler.dyn_mean, scaler.dyn_std)

    static = _static(p, rows)

    units = {nm: u for g in p["groups"] for nm, u in zip(g.names, g.units)}
    if scaler is not None:
        if (scaler.dyn_names, scaler.static_names) != (dyn_names, sta_names):
            raise ValueError("scaler was fitted on other features")
        static = _apply(static, scaler.static_mean, scaler.static_std)
    return Data(dyn, static, meta["target"].to_numpy(np.float32), meta.drop(columns=["target"]), dyn_names,
                sta_names, units, steps * p["wd"], grid, rows, neighbours(rows, patch) if patch else None,
                scaler is not None, align)


def _static(plan: dict, rows: np.ndarray) -> np.ndarray:
    """(len(rows), F_static) float32 static features in physical units."""
    work, meta_all = plan["work"], plan["meta_all"]
    n = len(rows)
    static = np.empty((n, len(plan["sta_names"])), dtype=np.float32)
    st_mm = np.load(work / "data" / "static.npy", mmap_mode="r")
    j = 0
    for g in plan["sta_g"]:
        k = len(g.names)
        if g.group == "terrain":
            tj = json.loads((work / "terrain.json").read_text())
            tm = np.load(work / "terrain.npy", mmap_mode="r")
            static[:, j:j + k] = np.asarray(tm[rows])[:, [tj["names"].index(c) for c in g.arg]]
        elif g.group == "soil":
            for a in range(0, n, BLOCK):
                static[a:a + BLOCK, j:j + k] = soil.compute(st_mm, rows[a:a + BLOCK], g.arg)
        elif g.group == "coords":
            static[:, j:j + k] = np.asarray(st_mm[rows])[:, g.arg]
        else:  # one-hot crop / country
            v = meta_all[g.group].astype(str).to_numpy()[rows]
            static[:, j:j + k] = (v[:, None] == np.array(g.arg)[None, :]).astype(np.float32)
        j += k
    return static


def _dyn_blocks(plan: dict, dyn_g: list, rows: np.ndarray):
    """Yield (offset, float32 block (b, steps, F)) of the dynamic features of `rows` on the plan's time axis,
    in physical units; steps outside the built grid are NaN (observed 0)."""
    if not dyn_g or not len(rows):
        return
    gdir, T, wd, steps = plan["gdir"], plan["T_grid"], plan["wd"], plan["steps"]
    field = plan["meta_all"]["field"].to_numpy()[rows]
    k0 = _origin_window(plan, field)
    s2_mm = np.load(gdir / "dyn.npy", mmap_mode="r")
    if any(g.group == "weather" for g in dyn_g):
        wj = json.loads((gdir / "weather.json").read_text())
        assert (wj["start_day"], wj["window_days"], wj["n_times"]) == (plan["start"], wd, T), "weather grid != S2"
        wx = np.load(gdir / "weather.npy")
        pos = {f: i for i, f in enumerate(wj["fields"])}
        wfield = pd.Series(field).map(pos).to_numpy()
        wcol = {f: i for i, f in enumerate(wj["features"])}
    obs_cols = []
    for a in range(0, len(rows), BLOCK):
        raw = np.asarray(s2_mm[rows[a:a + BLOCK]])
        refl = np.where(raw == S2_MISSING, np.float32(np.nan), raw.astype(np.float32) / np.float32(10000.0))
        parts, c = [], 0
        for g in dyn_g:
            if g.group == "s2":
                parts.append(refl[..., [F.S2_BANDS.index(b) for b in g.arg]])
            elif g.group == "indices":
                with np.errstate(divide="ignore", invalid="ignore"):
                    for i in g.arg:
                        x, y = (refl[..., F.S2_BANDS.index(b)] for b in F.INDICES[i])
                        v = (x - y) / (x + y)
                        v[~np.isfinite(v)] = np.nan
                        parts.append(np.clip(v, -1.0, 1.0)[..., None])
            elif g.group == "observed":
                obs_cols = [c]
                parts.append(np.isfinite(refl[..., 3:4]).astype(np.float32))   # B04
            elif g.group == "weather":
                parts.append(wx[wfield[a:a + BLOCK]][..., [wcol[n] for n in g.arg]].astype(np.float32))
            c += len(g.names)
        x = np.concatenate(parts, axis=-1)                                   # (b, T_grid, F)
        src = k0[a:a + BLOCK, None] + steps[None, :]                         # (b, steps) grid window per step
        inside = (src >= 0) & (src < T)
        out = np.take_along_axis(x, np.clip(src, 0, T - 1)[..., None], axis=1)
        out[~inside] = np.nan
        for oc in obs_cols:
            out[..., oc][~inside] = 0.0
        yield a, out


def _apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """z-score in float32, NaN / inf -> 0."""
    z = (x.astype(np.float32) - mean.astype(np.float32)) / std.astype(np.float32)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- normalisation and torch

NOT_SCALED = ("observed",)   # 0/1 flags and one-hots keep their values


@dataclass
class Scaler:
    dyn_mean: np.ndarray
    dyn_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    dyn_names: list[str]
    static_names: list[str]


def _stats(x: np.ndarray, block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean / std over all axes but the last."""
    if x.shape[-1] == 0:
        return np.zeros(0), np.ones(0)
    return _stats_blocks(x[a:a + block] for a in range(0, len(x), block))


def _stats_blocks(blocks) -> tuple[np.ndarray, np.ndarray]:
    s = ss = cnt = 0.0
    for b in blocks:
        v = np.asarray(b, dtype=np.float64).reshape(-1, b.shape[-1])
        ok = np.isfinite(v)
        v = np.where(ok, v, 0.0)
        s, ss, cnt = s + v.sum(0), ss + (v * v).sum(0), cnt + ok.sum(0)
    if np.isscalar(cnt):  # no blocks: no dynamic features
        return np.zeros(0), np.ones(0)
    mean = s / np.maximum(cnt, 1)
    std = np.sqrt(np.maximum(ss / np.maximum(cnt, 1) - mean ** 2, 0.0))
    std[~(std > 1e-12)] = 1.0
    return mean, std


def fit_scaler(d: Data | None = None, **selection) -> Scaler:
    """z-score statistics per feature (NaN ignored); flags and one-hots are left as they are.
    fit_scaler(d): from loaded (not normalised) Data. fit_scaler(split=..., features=..., grid=..., <any pixel /
    time options of load>): streamed from the files in float32 without keeping the data, so that
    load(..., scaler=sc) rounds only once."""
    if d is not None:
        if d.normalized:
            raise ValueError("Data is already normalised")
        dm, ds = _stats(d.dyn)
        sm, ss = _stats(d.static)
        dyn_names, sta_names, units = d.dyn_names, d.static_names, d.units
    else:
        p = _plan(**selection)
        dm, ds = _stats_blocks(x for _, x in _dyn_blocks(p, p["dyn_g"], p["rows"]))
        sm, ss = _stats(_static(p, p["rows"]))
        dyn_names, sta_names = p["dyn_names"], p["sta_names"]
        units = {nm: u for g in p["groups"] for nm, u in zip(g.names, g.units)}
    for names, m, s in ((dyn_names, dm, ds), (sta_names, sm, ss)):
        for i, nm in enumerate(names):
            if nm in NOT_SCALED or units.get(nm) == "one-hot":
                m[i], s[i] = 0.0, 1.0
    return Scaler(dm, ds, sm, ss, list(dyn_names), list(sta_names))


class YieldDataset:
    """torch Dataset over a Data: item i -> (dyn (T, F) or with patch (K, T, F + 1), static (S,), y), float32,
    z-scored with the scaler and NaN -> 0 (the `observed` feature says which steps are real). With a patch the
    last channel is 1 where the neighbour exists. Supports batched fetching (__getitems__)."""

    def __init__(self, d: Data, scaler: Scaler | None = None):
        import torch
        if scaler is None and not d.normalized:
            raise ValueError("pass a scaler, or load the Data with scaler=...")
        if scaler is not None and (scaler.dyn_names, scaler.static_names) != (d.dyn_names, d.static_names):
            raise ValueError("scaler was fitted on other features")
        self.torch, self.d, self.sc = torch, d, scaler

    def __len__(self) -> int:
        return len(self.d)

    def _dyn(self, idx: np.ndarray) -> np.ndarray:
        if self.d.normalized:
            return self.d.dyn[idx].astype(np.float32)
        return _apply(self.d.dyn[idx], self.sc.dyn_mean, self.sc.dyn_std)

    def __getitems__(self, idx):
        idx = np.asarray(idx)
        st = self.d.static[idx] if self.d.normalized else _apply(self.d.static[idx], self.sc.static_mean,
                                                                   self.sc.static_std)
        if self.d.nbr is None:
            xd = self._dyn(idx)
        else:
            nb = self.d.nbr[idx]                                     # (B, K)
            present = nb >= 0
            xd = self._dyn(np.maximum(nb, 0).ravel()).reshape(nb.shape + self.d.dyn.shape[1:])
            xd *= present[..., None, None]
            flag = np.broadcast_to(present[..., None, None], xd.shape[:-1] + (1,)).astype(np.float32)
            xd = np.concatenate([xd, flag], axis=-1)
        t = self.torch
        return [(t.from_numpy(xd[i]), t.from_numpy(st[i]), t.tensor(self.d.y[j])) for i, j in enumerate(idx)]

    def __getitem__(self, i: int):
        return self.__getitems__([i])[0]
