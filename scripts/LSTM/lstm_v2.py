"""LSTM v2: additions on top of the paper LSTM (lstm_lib.py), each switchable in the config.

* inputs.indices        spectral indices computed from the raw bands (ndvi, ndre, ndmi)
* inputs.mask_channel   per time step 1 = observed, 0 = missing (outside the season); otherwise missing
                        steps are indistinguishable from average observations after z-scoring
* inputs.categorical    one-hot of meta columns (crop, country) appended to the static inputs
* target.standardize_by z-score the yield per group (e.g. crop x country) with training statistics;
                        predictions are transformed back to t/ha
* train.pixels_per_field  each epoch samples at most this many pixels per field, so every field counts
                        about equally (field-level metrics weight fields equally) and epochs are short
* model.dropout, train.weight_decay
* ensemble.n_members    Deep Ensemble: independent seeds, prediction = mean over members

Kept separate from lstm_lib.py so the baseline (02_train.py) stays unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from lstm_lib import expand_bands

INDICES = {"ndvi": ("B08", "B04"), "ndre": ("B08", "B05"), "ndmi": ("B08", "B11")}


class FeatureSpec:
    """Builds model inputs from the raw memmaps (dynamic: pixels x 24 x 16, static: pixels x 104)."""

    def __init__(self, inputs: dict, info: dict):
        dyn_avail, stat_avail = info["dynamic_bands"], info["static_bands"]
        self.dyn_bands = expand_bands(inputs["dynamic"], dyn_avail)
        self.dyn_cols = [dyn_avail.index(b) for b in self.dyn_bands]
        self.indices = list(inputs.get("indices") or [])
        unknown = [i for i in self.indices if i not in INDICES]
        if unknown:
            raise KeyError(f"unknown indices {unknown}, available {list(INDICES)}")
        self.index_cols = [(dyn_avail.index(INDICES[i][0]), dyn_avail.index(INDICES[i][1])) for i in self.indices]
        self.mask_channel = bool(inputs.get("mask_channel", False))
        self.obs_col = dyn_avail.index("B02")
        self.stat_bands = expand_bands(inputs.get("static") or [], stat_avail)
        self.stat_cols = [stat_avail.index(b) for b in self.stat_bands]
        self.categorical = list(inputs.get("categorical") or [])
        self.fill = float(inputs.get("fill_value", 0.0))

    @property
    def dyn_names(self) -> list[str]:
        return self.dyn_bands + self.indices + (["observed"] if self.mask_channel else [])

    def build_dyn(self, raw: np.ndarray) -> np.ndarray:
        """raw (n, T, 16) -> (n, T, F) float32; NaN where missing, except the mask channel."""
        raw = np.asarray(raw, dtype=np.float32)
        parts = [raw[..., self.dyn_cols]]
        with np.errstate(divide="ignore", invalid="ignore"):
            for a, b in self.index_cols:
                v = (raw[..., a] - raw[..., b]) / (raw[..., a] + raw[..., b])
                v[~np.isfinite(v)] = np.nan
                parts.append(np.clip(v, -1.0, 1.0)[..., None])
        if self.mask_channel:
            parts.append(np.isfinite(raw[..., self.obs_col])[..., None].astype(np.float32))
        return np.concatenate(parts, axis=-1)

    # ---------------------------------------------------------------- normalisation
    def fit(self, dyn: np.ndarray, stat: np.ndarray, idx: np.ndarray, meta: pd.DataFrame, block: int = 200_000):
        """Mean/std per feature over the training pixels idx; categories seen in training."""
        nd = len(self.dyn_names)
        s, ss, n = np.zeros(nd), np.zeros(nd), np.zeros(nd)
        ks = len(self.stat_cols)
        s2, ss2, n2 = np.zeros(ks), np.zeros(ks), np.zeros(ks)
        for i in range(0, len(idx), block):
            b = idx[i:i + block]
            x = self.build_dyn(dyn[b]).reshape(-1, nd).astype(np.float64)
            v = np.isfinite(x)
            x = np.where(v, x, 0.0)
            s, ss, n = s + x.sum(0), ss + (x ** 2).sum(0), n + v.sum(0)
            if ks:
                x = np.asarray(stat[b], dtype=np.float64)[:, self.stat_cols]
                v = np.isfinite(x)
                x = np.where(v, x, 0.0)
                s2, ss2, n2 = s2 + x.sum(0), ss2 + (x ** 2).sum(0), n2 + v.sum(0)
        self.dyn_mean, self.dyn_std = _mean_std(s, ss, n)
        self.stat_mean, self.stat_std = _mean_std(s2, ss2, n2)
        if self.mask_channel:  # keep the 0/1 indicator as is
            self.dyn_mean[-1], self.dyn_std[-1] = 0.0, 1.0
        self.categories = {c: sorted(meta[c].iloc[idx].astype(str).unique()) for c in self.categorical}
        return self

    def transform(self, dyn, stat, idx: np.ndarray, meta: pd.DataFrame, block: int = 200_000):
        """Normalised float16 tensors: dynamic (n, T, F) and static (n, S + one-hot)."""
        T = dyn.shape[1]
        xd = torch.empty((len(idx), T, len(self.dyn_names)), dtype=torch.float16)
        n_cat = sum(len(v) for v in self.categories.values())
        xs = torch.zeros((len(idx), len(self.stat_cols) + n_cat), dtype=torch.float16)
        for i in range(0, len(idx), block):
            b = idx[i:i + block]
            x = (self.build_dyn(dyn[b]) - self.dyn_mean) / self.dyn_std
            xd[i:i + len(b)] = torch.from_numpy(np.nan_to_num(x, nan=self.fill, posinf=self.fill, neginf=self.fill))
            if self.stat_cols:
                x = (np.asarray(stat[b], dtype=np.float32)[:, self.stat_cols] - self.stat_mean) / self.stat_std
                xs[i:i + len(b), :len(self.stat_cols)] = torch.from_numpy(
                    np.nan_to_num(x, nan=self.fill, posinf=self.fill, neginf=self.fill))
        col = len(self.stat_cols)
        for c, cats in self.categories.items():  # unseen category -> all zeros
            codes = pd.Categorical(meta[c].iloc[idx].astype(str), categories=cats).codes
            rows = np.flatnonzero(codes >= 0)
            xs[torch.from_numpy(rows), torch.from_numpy(col + codes[rows].astype(np.int64))] = 1.0
            col += len(cats)
        return xd, xs

    def state(self) -> dict:
        return {"dyn_names": self.dyn_names, "stat_bands": self.stat_bands, "categories": self.categories,
                "dyn_mean": self.dyn_mean.tolist(), "dyn_std": self.dyn_std.tolist(),
                "stat_mean": self.stat_mean.tolist(), "stat_std": self.stat_std.tolist()}


def _mean_std(s, ss, n):
    mean = s / np.maximum(n, 1)
    std = np.sqrt(np.maximum(ss / np.maximum(n, 1) - mean ** 2, 0.0))
    std[~(std > 1e-12)] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class TargetScaler:
    """z-score the yield per group with training statistics. Falls back to coarser groups (dropping
    the last column at a time, finally the global statistics) for groups absent from training."""

    def __init__(self, by: list[str] | None):
        self.by = list(by or [])

    def fit(self, meta: pd.DataFrame, idx: np.ndarray):
        y = meta["target"].iloc[idx]
        self.global_ = (float(y.mean()), float(y.std(ddof=0)) or 1.0)
        self.levels = []  # finest grouping first
        for k in range(len(self.by), 0, -1):
            cols = self.by[:k]
            g = pd.DataFrame({"key": _keys(meta.iloc[idx], cols), "y": y.to_numpy()}).groupby("key")["y"]
            stats = g.agg(["mean", "std"])
            stats["std"] = stats["std"].fillna(self.global_[1]).replace(0, self.global_[1])
            self.levels.append((cols, stats))
        return self

    def params(self, meta: pd.DataFrame, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sub = meta.iloc[idx]
        mean = np.full(len(idx), np.nan)
        std = np.full(len(idx), np.nan)
        for cols, stats in self.levels:
            todo = np.isnan(mean)
            if not todo.any():
                break
            keys = _keys(sub[todo], cols)
            mean[todo] = stats["mean"].reindex(keys).to_numpy()
            std[todo] = stats["std"].reindex(keys).to_numpy()
        todo = np.isnan(mean)
        mean[todo], std[todo] = self.global_
        return mean.astype(np.float32), std.astype(np.float32)

    def state(self) -> dict:
        return {"by": self.by, "global": self.global_,
                "groups": {"|".join(cols): stats.to_dict("index") for cols, stats in self.levels}}


def _keys(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    key = df[cols[0]].astype(str)
    for c in cols[1:]:
        key = key + "|" + df[c].astype(str)
    return key.to_numpy()


class LSTMv2(nn.Module):
    """Paper LSTM (2 x LSTM 128, FC 128 -> BN -> ReLU -> FC 1) with optional dropout between the LSTM
    layers and before the output layer."""

    def __init__(self, n_in: int, hidden_size: int = 128, num_layers: int = 2, head_size: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, head_size), nn.BatchNorm1d(head_size), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(head_size, 1))

    def forward(self, xd: torch.Tensor, xs: torch.Tensor | None = None) -> torch.Tensor:
        if xs is not None and xs.shape[-1] > 0:
            xd = torch.cat([xd, xs[:, None, :].expand(-1, xd.shape[1], -1)], dim=-1)
        _, (h, _) = self.lstm(xd)
        return self.head(h[-1]).squeeze(-1)


def sample_per_field(field_codes: np.ndarray, k: int | None, rng: np.random.Generator) -> np.ndarray:
    """Positions of at most k random pixels per field (all pixels if k is None)."""
    if not k:
        return rng.permutation(len(field_codes))
    order = np.lexsort((rng.random(len(field_codes)), field_codes))
    f = field_codes[order]
    start = np.r_[0, np.flatnonzero(f[1:] != f[:-1]) + 1]
    rank = np.arange(len(f)) - np.repeat(start, np.diff(np.r_[start, len(f)]))
    return rng.permutation(order[rank < k])


def field_balanced_mse(err2: np.ndarray, field_codes: np.ndarray) -> float:
    """Mean over fields of the per-field mean squared error."""
    s = np.bincount(field_codes, weights=err2)
    n = np.bincount(field_codes)
    return float(np.mean(s[n > 0] / n[n > 0]))


# --------------------------------------------------------------------------- spatial patches

def gather_patches(Xd: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
    """Xd (n, T, F) of one set, nbr (B, K) local row indices (-1 = missing) -> (B, K, T, F + 1).
    Missing neighbours are zero; the extra last channel is 1 where the neighbour exists."""
    valid = nbr >= 0
    x = Xd[nbr.clamp(min=0)]
    x = x * valid[:, :, None, None].to(x.dtype)
    v = valid[:, :, None, None].to(x.dtype).expand(-1, -1, x.shape[2], 1)
    return torch.cat([x, v], dim=-1)


class PatchLSTM(nn.Module):
    """Pixel with its k x k neighbourhood: a small CNN encodes the patch at every time step (weights shared
    over time), its output plus the centre pixel's own features feed the LSTM of LSTMv2.

    Input xd (B, K=k*k, T, C) from gather_patches, xs (B, S) static features of the centre pixel."""

    def __init__(self, n_dyn: int, n_static: int, k: int = 5, conv_channels: int = 32, hidden_size: int = 128,
                 num_layers: int = 2, head_size: int = 128, dropout: float = 0.0):
        super().__init__()
        self.k = k
        c = n_dyn + 1  # + neighbour-present channel
        self.cnn = nn.Sequential(
            nn.Conv2d(c, conv_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, 3, padding=1), nn.ReLU())
        # patch summary: CNN features at the centre and averaged over the neighbours present
        self.lstm = LSTMv2(2 * conv_channels + c + n_static, hidden_size, num_layers, head_size, dropout)

    def forward(self, xd: torch.Tensor, xs: torch.Tensor | None = None) -> torch.Tensor:
        B, K, T, C = xd.shape
        img = xd.permute(0, 2, 3, 1).reshape(B * T, C, self.k, self.k)
        f = self.cnn(img)                                             # (B*T, ch, k, k)
        present = img[:, -1:]                                          # neighbour-present mask
        centre = f[:, :, self.k // 2, self.k // 2]
        mean = (f * present).sum((2, 3)) / present.sum((2, 3)).clamp(min=1)
        feats = torch.cat([centre, mean, xd[:, K // 2].reshape(B * T, C)], dim=-1).reshape(B, T, -1)
        return self.lstm(feats, xs)
