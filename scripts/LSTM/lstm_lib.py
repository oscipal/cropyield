"""Shared code for the pixel LSTM: band selection, normalisation, model.

Model and training follow the LSTM of the YieldSAT benchmark (Miranda et al., CVPR 2026, ref. [36] =
Pathak et al., IGARSS 2023): 2 stacked LSTM layers with 128 hidden units, then FC 128 -> BatchNorm -> ReLU
-> FC 1; Adam, lr 1e-3, batch 1024, 50 epochs, early stopping after 8 epochs without improvement on
validation data.
"""
from __future__ import annotations

import re

import numpy as np
import torch
from torch import nn

S2 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
WEATHER = ["temp_mean", "temp_max", "temp_min", "total_prec"]
DEM = ["aspect", "curvature", "dem", "slope", "twi"]
COORDS = ["coord_x", "coord_y", "coord_z"]
_SOIL = re.compile(r"^(cec|cfvo|clay|nitrogen|phh2o|sand|silt|soc)_\d+-\d+$")
_SOIL_UNC = re.compile(r"^(cec|cfvo|clay|nitrogen|phh2o|sand|silt|soc)_\d+-\d+_uncertainty$")


def expand_bands(spec: list[str], available: list[str]) -> list[str]:
    """Expand group names (s2, weather, soil, soil_uncertainty, dem, coords) and keep explicit band names."""
    groups = {"s2": S2, "weather": WEATHER, "dem": DEM, "coords": COORDS,
              "soil": [b for b in available if _SOIL.match(b)],
              "soil_uncertainty": [b for b in available if _SOIL_UNC.match(b)]}
    out: list[str] = []
    for item in spec:
        out += groups.get(item, [item])
    missing = [b for b in out if b not in available]
    if missing:
        raise KeyError(f"bands not in data: {missing}")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate bands in {spec}")
    return out


# --------------------------------------------------------------------------- normalisation

def _blocks(idx: np.ndarray, block: int):
    for i in range(0, len(idx), block):
        yield i, idx[i:i + block]


def band_stats(arr: np.ndarray, cols: list[int], idx: np.ndarray, block: int = 200_000):
    """NaN-aware mean/std per selected band over the pixels ``idx`` (and all time steps, if 3-D)."""
    k = len(cols)
    s, ss, n = np.zeros(k), np.zeros(k), np.zeros(k)
    for _, b in _blocks(idx, block):
        x = np.asarray(arr[b], dtype=np.float64)[..., cols].reshape(-1, k)
        valid = np.isfinite(x)
        x = np.where(valid, x, 0.0)
        s += x.sum(0)
        ss += (x ** 2).sum(0)
        n += valid.sum(0)
    mean = s / np.maximum(n, 1)
    std = np.sqrt(np.maximum(ss / np.maximum(n, 1) - mean ** 2, 0.0))
    std[~(std > 1e-12)] = 1.0  # constant or empty bands: centre only
    return mean.astype(np.float32), std.astype(np.float32)


def normalized(arr: np.ndarray, cols: list[int], idx: np.ndarray, mean: np.ndarray, std: np.ndarray,
               fill: float = 0.0, block: int = 200_000) -> torch.Tensor:
    """(x - mean) / std for arr[idx][..., cols] as a float16 tensor; NaN (outside the season) -> fill."""
    out = torch.empty((len(idx),) + arr.shape[1:-1] + (len(cols),), dtype=torch.float16)
    for i, b in _blocks(idx, block):
        x = (np.asarray(arr[b], dtype=np.float32)[..., cols] - mean) / std
        out[i:i + len(b)] = torch.from_numpy(np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill))
    return out


# --------------------------------------------------------------------------- model

class PaperLSTM(nn.Module):
    """(batch, time, dyn) [+ (batch, static) repeated over time] -> 2x LSTM(128) -> last hidden state
    -> FC 128 -> BatchNorm -> ReLU -> FC 1."""

    def __init__(self, n_in: int, hidden_size: int = 128, num_layers: int = 2, head_size: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, head_size), nn.BatchNorm1d(head_size), nn.ReLU(),
                                  nn.Linear(head_size, 1))

    def forward(self, xd: torch.Tensor, xs: torch.Tensor | None = None) -> torch.Tensor:
        if xs is not None and xs.shape[-1] > 0:
            xd = torch.cat([xd, xs[:, None, :].expand(-1, xd.shape[1], -1)], dim=-1)
        _, (h, _) = self.lstm(xd)
        return self.head(h[-1]).squeeze(-1)
