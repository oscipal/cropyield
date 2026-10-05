"""Per-fold band normalisation (train-pixel statistics), then NaN -> 0."""
from __future__ import annotations

import numpy as np


def band_stats(X: np.ndarray, idx: np.ndarray, block: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware mean and std per band over the given pixels and all timesteps."""
    idx = np.sort(idx)
    B = X.shape[-1]
    s, ss, n = np.zeros(B), np.zeros(B), np.zeros(B)
    for i in range(0, len(idx), block):
        x = np.asarray(X[idx[i:i + block]], dtype=np.float64).reshape(-1, B)
        valid = np.isfinite(x)
        x = np.where(valid, x, 0.0)
        s += x.sum(0)
        ss += (x ** 2).sum(0)
        n += valid.sum(0)
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X: np.ndarray, idx: np.ndarray, mean: np.ndarray, std: np.ndarray,
              block: int = 50_000) -> np.ndarray:
    """Return normalised copy of X[idx] (order of idx preserved) with NaN replaced by 0."""
    out = np.empty((len(idx),) + X.shape[1:], dtype=np.float32)
    for i in range(0, len(idx), block):
        x = (np.asarray(X[idx[i:i + block]], dtype=np.float32) - mean) / std
        out[i:i + block] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return out
