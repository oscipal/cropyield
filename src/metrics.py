"""Field- and pixel-level metrics as in the paper (R2 and RMSE)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def r2(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def field_level(preds: pd.DataFrame, key: str = "field_year") -> pd.DataFrame:
    """Mean of pixel predictions per field vs. mean of pixel targets per field."""
    agg = {"target": ("target", "mean"), "pred": ("pred", "mean"), "n_pixels": ("target", "size")}
    for col in ("fold", "year", "farm", "field"):
        if col in preds and col != key:
            agg[col] = (col, "first")
    return preds.groupby(key, observed=True).agg(**agg).reset_index()


def compute_metrics(preds: pd.DataFrame) -> dict:
    fields = field_level(preds)
    return {
        "field_r2": r2(fields["target"], fields["pred"]),
        "field_rmse": rmse(fields["target"], fields["pred"]),
        "pixel_r2": r2(preds["target"], preds["pred"]),
        "pixel_rmse": rmse(preds["target"], preds["pred"]),
        "n_fields": len(fields),
        "n_pixels": len(preds),
    }
