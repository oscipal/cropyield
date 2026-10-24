"""One loader for all model inputs: pick the features (src/features.py) and the time grid, get arrays.

    from src.loader import load, fit_scaler, YieldDataset

    d = load("world/loyo/all", role="train", grid="10d",
             features=["s2", "indices", "observed", "weather:tmean_c,prec_mm", "terrain", "soil:0-30cm", "crop"])
    d.dyn       (pixels, T, F_dyn)  physical units, NaN = missing (no S2 observation / outside the weather record)
    d.static    (pixels, F_static)
    d.y         yield t/ha;  d.meta: field, farm, crop, country, year, row, col
    d.dyn_names, d.static_names, d.units;  d.days: first day of every window (day 0 = 1 Jan of the year before
                harvest, src/season.py)
    sc = fit_scaler(d)                              # on the training set
    ds = YieldDataset(d, sc)                        # torch Dataset: normalised, NaN -> 0; batched __getitems__
    # or normalised at load time (one float16 rounding, as the training script does):
    sc = fit_scaler(split="world/loyo/all", features=[...], grid="10d")      # streamed, nothing kept
    d = load("world/loyo/all", "val", features=[...], grid="10d", scaler=sc)  # z-scored, NaN -> 0

Grids: "30d", "15d", "10d" = the S2 cubes rebuilt from the raw scenes (scripts/LSTM/11_build_s2_grid.py, start
day 210, best scene per window, clear pixels only) and the weather on the same windows
(scripts/LSTM/13_build_weather_grid.py). Terrain: scripts/LSTM/14_build_terrain.py. Soil: src/soil.py.

Memory: d.dyn is held in RAM. World 10-day training set, S2 + indices + observed: 8.3 M pixels x 51 x 16 x 2 bytes
= 13.5 GB as float16 (the default); use pixels_per_field to subsample. The size is printed before loading.
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
from src.config import load_paths
from src.splits import load_split

GRIDS = {"30d": "data_w30", "15d": "data_w15", "10d": "data_w10"}
ROLES = ("train", "val", "test")
S2_MISSING = 65535
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
    days: np.ndarray
    grid: str
    rows: np.ndarray                 # pixel rows in lstm/data/meta.parquet
    nbr: np.ndarray | None = None    # (pixels, k*k) neighbour positions in this Data, -1 = absent / not loaded
    normalized: bool = False         # loaded with a scaler: z-scored, NaN -> 0

    def __len__(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:
        return (f"Data({len(self):,} pixels, {self.meta['field'].nunique():,} fields, grid {self.grid}, "
                f"dyn {self.dyn.shape} {self.dyn.dtype}, static {self.static.shape})")


@lru_cache(maxsize=1)
def _base() -> dict:
    work = load_paths()["work_dir"] / "lstm"
    info = json.loads((work / "data" / "info.json").read_text())
    meta = pd.read_parquet(work / "data" / "meta.parquet")
    meta["country"] = meta["country"].astype(str)
    return {"work": work, "meta": meta, "static_names": info["static_bands"]}


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


def _plan(split, role, features, grid, fields, pixels_per_field, seed) -> dict:
    """Resolved features, grid and pixel rows of a load() call."""
    if grid not in GRIDS:
        raise ValueError(f"grid must be one of {list(GRIDS)}, got {grid!r}")
    base = _base()
    meta_all, work = base["meta"], base["work"]
    gdir = work / GRIDS[grid]
    info = json.loads((gdir / "info.json").read_text())
    cats = {c: sorted(meta_all[c].astype(str).unique()) for c in F.CATEGORICAL}
    groups = F.resolve(list(features), base["static_names"], cats)
    dyn_g = [g for g in groups if g.dynamic]
    sta_g = [g for g in groups if not g.dynamic]
    rows = select_rows(split, role, fields)
    if pixels_per_field:
        rows = subsample(rows, meta_all["field"].to_numpy()[rows], pixels_per_field, seed)
    return {"meta_all": meta_all, "work": work, "gdir": gdir, "rows": rows, "groups": groups,
            "grid": (info["n_times"], info["start_day"], info["window_days"]), "dyn_g": dyn_g, "sta_g": sta_g,
            "dyn_names": [n for g in dyn_g for n in g.names], "sta_names": [n for g in sta_g for n in g.names]}


def load(split: str | None = None, role: str | list[str] = "train", features: list[str] = ("s2", "indices",
         "observed"), grid: str = "10d", fields: list[str] | None = None, pixels_per_field: int | None = None,
         seed: int = 0, patch: int | None = None, dtype=np.float16, scaler: Scaler | None = None,
         verbose: bool = True) -> Data:
    """Inputs and target of one split role (e.g. "world/loyo/all", "train"), or of explicit `fields`, or of all
    pixels (split=None). See the module docstring. With `scaler` (from fit_scaler) the features are z-scored in
    float32 before they are stored and NaN -> 0 (d.normalized = True): one rounding to `dtype` only, the form
    the training script uses."""
    p = _plan(split, role, features, grid, fields, pixels_per_field, seed)
    meta_all, work, gdir, rows = p["meta_all"], p["work"], p["gdir"], p["rows"]
    T, start, wd = p["grid"]
    groups, dyn_g, sta_g, dyn_names, sta_names = p["groups"], p["dyn_g"], p["sta_g"], p["dyn_names"], p["sta_names"]
    n = len(rows)
    if verbose:
        gb = n * T * len(dyn_names) * np.dtype(dtype).itemsize / 1e9
        print(f"load {split or 'fields'} {role if split else ''}: {n:,} pixels, grid {grid} ({T} steps), "
              f"{len(dyn_names)} dynamic + {len(sta_names)} static features, dyn {gb:.1f} GB {np.dtype(dtype)}",
              flush=True)

    meta = meta_all.iloc[rows].reset_index(drop=True)
    dyn = np.empty((n, T, len(dyn_names)), dtype=dtype)
    for a, x in _dyn_blocks(gdir, dyn_g, rows, meta["field"].to_numpy(), (start, wd, T)):
        dyn[a:a + len(x)] = x if scaler is None else _apply(x, scaler.dyn_mean, scaler.dyn_std)

    static = np.empty((n, len(sta_names)), dtype=np.float32)
    st_mm = np.load(work / "data" / "static.npy", mmap_mode="r")
    j = 0
    for g in sta_g:
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
            v = meta[g.group].astype(str).to_numpy()
            static[:, j:j + k] = (v[:, None] == np.array(g.arg)[None, :]).astype(np.float32)
        j += k

    nbr = None
    if patch:
        nb = np.load(work / "data" / f"nbr_k{patch}.npy", mmap_mode="r")
        loc = np.full(len(meta_all), -1, dtype=np.int64)
        loc[rows] = np.arange(n)
        g = np.asarray(nb[rows])
        nbr = np.where(g >= 0, loc[np.maximum(g, 0)], -1)

    units = {nm: u for g in groups for nm, u in zip(g.names, g.units)}
    if scaler is not None:
        if (scaler.dyn_names, scaler.static_names) != (dyn_names, sta_names):
            raise ValueError("scaler was fitted on other features")
        static = _apply(static, scaler.static_mean, scaler.static_std)
    return Data(dyn, static, meta["target"].to_numpy(np.float32), meta.drop(columns=["target"]), dyn_names,
                sta_names, units, start + wd * np.arange(T), grid, rows, nbr, scaler is not None)


def _dyn_blocks(gdir: Path, dyn_g: list, rows: np.ndarray, field: np.ndarray, grid: tuple[int, int, int]):
    """Yield (offset, float32 block (b, T, F_dyn)) of the dynamic features of `rows`, in physical units."""
    if not dyn_g:
        return
    s2_mm = np.load(gdir / "dyn.npy", mmap_mode="r")
    if any(g.group == "weather" for g in dyn_g):
        wj = json.loads((gdir / "weather.json").read_text())
        assert (wj["start_day"], wj["window_days"], wj["n_times"]) == grid, "weather grid != S2 grid"
        wx = np.load(gdir / "weather.npy")
        pos = {f: i for i, f in enumerate(wj["fields"])}
        wfield = pd.Series(field).map(pos).to_numpy()
        wcol = {f: i for i, f in enumerate(wj["features"])}
    for a in range(0, len(rows), BLOCK):
        raw = np.asarray(s2_mm[rows[a:a + BLOCK]])
        refl = np.where(raw == S2_MISSING, np.float32(np.nan), raw.astype(np.float32) / np.float32(10000.0))
        parts = []
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
                parts.append(np.isfinite(refl[..., 3:4]).astype(np.float32))   # B04
            elif g.group == "weather":
                parts.append(wx[wfield[a:a + BLOCK]][..., [wcol[c] for c in g.arg]].astype(np.float32))
        yield a, np.concatenate(parts, axis=-1)


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


def fit_scaler(d: Data | None = None, *, split: str | None = None, role: str = "train",
               features: list[str] = ("s2", "indices", "observed"), grid: str = "10d",
               fields: list[str] | None = None, pixels_per_field: int | None = None, seed: int = 0) -> Scaler:
    """z-score statistics per feature (NaN ignored); flags and one-hots are left as they are.
    fit_scaler(d): from loaded (not normalised) Data. fit_scaler(split=..., features=..., grid=...): streamed from
    the files in float32 without keeping the data, so that load(..., scaler=sc) rounds only once."""
    if d is not None:
        if d.normalized:
            raise ValueError("Data is already normalised")
        dm, ds = _stats(d.dyn)
        sm, ss = _stats(d.static)
        dyn_names, sta_names, units = d.dyn_names, d.static_names, d.units
    else:
        p = _plan(split, role, features, grid, fields, pixels_per_field, seed)
        T, start, wd = p["grid"]
        field = p["meta_all"]["field"].to_numpy()[p["rows"]]
        dm, ds = _stats_blocks(x for _, x in _dyn_blocks(p["gdir"], p["dyn_g"], p["rows"], field, (start, wd, T)))
        static_specs = [s for s in features if s.partition(":")[0] in F.STATIC_GROUPS]
        st = load(split, role, static_specs, grid, fields, pixels_per_field, seed, verbose=False)
        sm, ss = _stats(st.static)
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
