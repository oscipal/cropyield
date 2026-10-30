"""Scene-sequence inputs: every usable Sentinel-2 scene of a field-year as one time step, no windows
(world variant wv_scenes, `inputs.scenes: true` in scripts/LSTM/04_train_v2.py).

    S, names = load_scene_set(rows, stats)        # rows of lstm/data/meta.parquet (one split role)
    S.gather(b, nbr_b)                            # (B, K, T_batch, F + 1) patches, or (B, T_batch, F) without nbr

Store: <work_dir>/lstm/scenes/ (scripts/LSTM/19_build_s2_scenes.py): per field-year its scenes with day in
[210, 720) and >= 5 % clear pixels on the field, clear pixels only. All pixels of a field-year share its scene list,
so the sequences of a pixel and its neighbours have the same length and dates.

Features per step (F = 19): the 12 bands (reflectance), NDVI, NDRE, NDMI, observed (1 = pixel clear in this
scene), scene (1 = a real scene, 0 = padding), day (scene day on the cube axis, day 0 = 1 January of the year before
the harvest year) and gap (days since the previous kept scene, the first one counted from day 210). Bands, indices,
day and gap are z-scored with training statistics (fit_scene_stats), missing values -> 0.

Batches are left-padded to the longest sequence in the batch, so the last step is always the last scene. The
neighbour-present channel of a patch does not depend on the step (ConvLSTMModel reads it at step 0).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from src.config import load_paths
from src.features import INDICES, S2_BANDS

FEATURES = S2_BANDS + list(INDICES) + ["observed", "scene", "day", "gap"]
SCALED = S2_BANDS + list(INDICES) + ["day", "gap"]
MISSING = 65535
FIRST_DAY = 210


@lru_cache(maxsize=1)
def _store():
    d = load_paths()["work_dir"] / "lstm" / "scenes"
    info = json.loads((d / "info.json").read_text())
    values = np.memmap(d / "values.u16", dtype=np.uint16, mode="r", shape=tuple(info["shape"]))
    fy = pd.read_parquet(d / "fields.parquet")
    sc = pd.read_parquet(d / "scenes.parquet").sort_values(["field", "k"])
    days = {f: g["day"].to_numpy() for f, g in sc.groupby("field")}
    if info["bands"] != list(S2_BANDS):
        raise ValueError(f"scene store bands {info['bands']} != {S2_BANDS}")
    return values, fy, days


def _features(raw: np.ndarray, days: np.ndarray) -> np.ndarray:
    """raw (P, n, 12) uint16 -> (P, n, 19) float32 in physical units, NaN = missing."""
    refl = np.where(raw == MISSING, np.float32(np.nan), raw.astype(np.float32) / np.float32(10000.0))
    parts = [refl]
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in INDICES:
            x, y = (refl[..., S2_BANDS.index(b)] for b in INDICES[i])
            v = (x - y) / (x + y)
            v[~np.isfinite(v)] = np.nan
            parts.append(np.clip(v, -1.0, 1.0)[..., None])
    P, n = raw.shape[:2]
    gap = np.diff(np.r_[FIRST_DAY, days]).astype(np.float32)
    parts += [np.isfinite(refl[..., 3:4]).astype(np.float32), np.ones((P, n, 1), np.float32),
              np.broadcast_to(days.astype(np.float32)[None, :, None], (P, n, 1)),
              np.broadcast_to(gap[None, :, None], (P, n, 1))]
    return np.concatenate(parts, axis=-1)


def _blocks(rows: np.ndarray):
    """Yield (positions in rows, field-year n_scenes, float32 features (len, n, 19)) per field-year of `rows`."""
    values, fy, days = _store()
    starts = fy["start"].to_numpy()
    f_of = np.searchsorted(starts, rows, side="right") - 1
    bounds = np.r_[0, np.flatnonzero(np.diff(f_of)) + 1, len(rows)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        r = fy.iloc[f_of[a]]
        n = int(r["n_scenes"])
        p = rows[a:b] - int(r["start"])
        if n == 0:
            yield a, b, 0, np.zeros((b - a, 0, len(FEATURES)), np.float32)
            continue
        raw = np.asarray(values[int(r["offset"]):int(r["offset"]) + (int(r["stop"]) - int(r["start"])) * n])
        yield a, b, n, _features(raw.reshape(-1, n, 12)[p], days[r["field"]])


def fit_scene_stats(rows: np.ndarray, max_per_field: int = 200, seed: int = 0) -> dict:
    """Mean / std of the scaled features over the training pixel-scenes (up to max_per_field random pixels per
    field-year), NaN ignored."""
    rng = np.random.default_rng(seed)
    s = ss = cnt = 0.0
    for _, _, n, x in _blocks(np.sort(rows)):
        if n == 0:
            continue
        if len(x) > max_per_field:
            x = x[rng.choice(len(x), max_per_field, replace=False)]
        v = x.reshape(-1, x.shape[-1]).astype(np.float64)
        ok = np.isfinite(v)
        v = np.where(ok, v, 0.0)
        s, ss, cnt = s + v.sum(0), ss + (v * v).sum(0), cnt + ok.sum(0)
    mean = s / np.maximum(cnt, 1)
    std = np.sqrt(np.maximum(ss / np.maximum(cnt, 1) - mean ** 2, 0.0))
    std[~(std > 1e-12)] = 1.0
    keep = np.array([f in SCALED for f in FEATURES])
    mean, std = np.where(keep, mean, 0.0), np.where(keep, std, 1.0)
    return {"names": FEATURES, "mean": mean.tolist(), "std": std.tolist()}


@dataclass
class SceneSet:
    """Pixel-scene features of one split role: flat (N, F) float16 z-scored, row i's scenes at
    flat[off[i]: off[i] + n[i]]."""
    flat: torch.Tensor
    off: torch.Tensor
    n: torch.Tensor

    def __len__(self) -> int:
        return len(self.off)

    @property
    def shape(self):
        return (len(self.off), int(self.n.max()) if len(self.n) else 0, self.flat.shape[1])

    @property
    def device(self):
        return self.flat.device

    def element_size(self) -> int:
        return self.flat.element_size()

    def nelement(self) -> int:
        return self.flat.nelement()

    def to(self, device) -> "SceneSet":
        return SceneSet(self.flat.to(device), self.off.to(device), self.n.to(device))

    def gather(self, b: torch.Tensor, nbr: torch.Tensor | None = None) -> torch.Tensor:
        """Rows b (B,) -> (B, T, F); with nbr (B, K) local neighbour rows (-1 = missing) -> (B, K, T, F + 1), the
        last channel 1 where the neighbour exists. Left-padded with zeros to T = the longest sequence in b."""
        dev = self.flat.device
        b = b.to(dev)
        n = self.n[b].long()                                              # (B,)
        T = max(int(n.max()), 1)
        t = torch.arange(T, device=dev)
        pad = (T - n)[:, None]                                            # (B, 1)
        real = t[None, :] >= pad                                          # (B, T)
        if nbr is None:
            idx = self.off[b].long()[:, None] + (t[None, :] - pad)
            x = self.flat[idx.clamp(min=0, max=len(self.flat) - 1)]
            return x * real[..., None].to(x.dtype)
        nbr = nbr.to(dev).long()
        present = nbr >= 0                                                # (B, K)
        off = self.off[nbr.clamp(min=0)].long()                           # (B, K)
        idx = off[:, :, None] + (t[None, None, :] - pad[:, :, None])     # (B, K, T)
        x = self.flat[idx.clamp(min=0, max=len(self.flat) - 1)]          # (B, K, T, F)
        x = x * (real[:, None, :] & present[:, :, None])[..., None].to(x.dtype)
        v = present[:, :, None, None].to(x.dtype).expand(-1, -1, T, 1)
        return torch.cat([x, v], dim=-1)


def load_scene_set(rows: np.ndarray, stats: dict, dtype=torch.float16) -> SceneSet:
    """SceneSet of `rows` (sorted rows of meta.parquet, e.g. Data.rows of the loader), z-scored with `stats`."""
    mean = np.asarray(stats["mean"], np.float32)
    std = np.asarray(stats["std"], np.float32)
    sizes = np.zeros(len(rows), np.int64)
    for a, b, n, _ in _blocks_sizes(rows):
        sizes[a:b] = n
    off = np.r_[0, np.cumsum(sizes)[:-1]]
    flat = np.empty((int(sizes.sum()), len(FEATURES)), dtype=np.float16)
    for a, b, n, x in _blocks(rows):
        if n == 0:
            continue
        z = np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        flat[off[a]:off[a] + (b - a) * n] = z.reshape(-1, len(FEATURES))
    print(f"  scene set: {len(rows):,} pixels, {len(flat):,} pixel-scenes (median {int(np.median(sizes)) if len(sizes) else 0}"
          f", max {int(sizes.max()) if len(sizes) else 0} per pixel), {flat.nbytes / 1e9:.1f} GB", flush=True)
    return SceneSet(torch.from_numpy(flat).to(dtype), torch.from_numpy(off), torch.from_numpy(sizes.astype(np.int32)))


def _blocks_sizes(rows: np.ndarray):
    _, fy, _ = _store()
    starts = fy["start"].to_numpy()
    f_of = np.searchsorted(starts, rows, side="right") - 1
    bounds = np.r_[0, np.flatnonzero(np.diff(f_of)) + 1, len(rows)]
    n_sc = fy["n_scenes"].to_numpy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield a, b, int(n_sc[f_of[a]]), None


def augment_scenes(xd: torch.Tensor, aug: dict, gen: torch.Generator, stats: dict) -> torch.Tensor:
    """Training augmentation on gathered scene patches (B, K, T, F + 1) or (B, T, F):
      day_shift: s     add k ~ U{-s..s} days to the day channel of every real step of a sample (calendar jitter)
      step_dropout: p  each real scene of a sample becomes cloudy with probability p (bands, indices, observed = 0)"""
    if not aug:
        return xd
    x = xd if xd.dim() == 4 else xd[:, None]
    B, K, T, _ = x.shape
    dev = x.device
    i_scene, i_day = FEATURES.index("scene"), FEATURES.index("day")
    real = x[:, K // 2, :, i_scene] > 0.5                                     # (B, T), same for every neighbour
    x = x.clone()
    s = int(aug.get("day_shift", 0))
    if s > 0:
        k = torch.randint(-s, s + 1, (B, 1, 1), generator=gen, device=dev).to(x.dtype)
        x[..., i_day] += (k / float(stats["std"][i_day])) * real[:, None, :].to(x.dtype)
    p = float(aug.get("step_dropout", 0.0))
    if p > 0:
        drop = (torch.rand(B, 1, T, generator=gen, device=dev) < p) & real[:, None, :]
        cols = len(S2_BANDS) + len(INDICES) + 1                                # bands, indices, observed
        x[..., :cols] = torch.where(drop[..., None], torch.zeros_like(x[..., :cols]), x[..., :cols])
    return x if xd.dim() == 4 else x[:, 0]
