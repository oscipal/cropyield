"""Cross-validation over years (src.splitting.CV_YEARS): pool the ensemble predictions of a config's folds and score them.

Each fold <scope>/cv<year>/all is trained once and predicts its test year and its val year. Pooled over the folds,
every field-year of a CV year has one test and one val prediction from a model that never saw that year. Fold cv2021
equals world/loyo/all (test 2021, val 2022), so a run of world/loyo/all counts as fold 2021 when the config has no
cv2021 run.

Scores on field means (mean prediction and mean target over the pixels of a field-year), ensemble prediction only,
and only for country x crop x year cells with at least CV_MIN_FIELDS field-years:
  group (country x crop, all CV years pooled): R2 and rRMSE (RMSE in % of the mean observed field yield). Includes the
        variation between years, so getting each year's level right counts.
  cell  (country x crop x year): R2, rRMSE and bias (mean prediction / mean observed - 1, in %).
Used by scripts/LSTM/22_cv_report.py and the visualizer's Results page.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.splitting import CV_MIN_FIELDS, CV_YEARS

COLS = ["country", "crop", "year", "field_year", "target", "pred"]


def fold_dirs(config_dir: Path, scope: str = "world") -> dict[str, Path]:
    """CV year -> finished run folder of that fold (world__loyo__all stands in for the fold with the same years)."""
    out = {}
    for y in CV_YEARS[scope]:
        d = config_dir / f"{scope}__cv{y}__all"
        if (d / "metrics.json").exists():
            out[y] = d
    loyo = config_dir / f"{scope}__loyo__all"
    if (loyo / "metrics.json").exists():
        a = json.loads((loyo / "metrics.json").read_text()).get("split_attrs", {})
        y = (a.get("test_years") or [None])[0]
        if y in CV_YEARS[scope] and y not in out:
            i = CV_YEARS[scope].index(y)
            if a.get("val_years") == [CV_YEARS[scope][(i + 1) % len(CV_YEARS[scope])]]:
                out[y] = loyo
    return dict(sorted(out.items()))


def field_preds(dirs: dict[str, Path], role: str, drop_countries: tuple[str, ...] = ()) -> pd.DataFrame:
    """Field-mean target and ensemble prediction of every fold's val or test set, with the fold year."""
    parts = []
    for y, d in dirs.items():
        p = pd.read_parquet(d / f"preds_{role}.parquet", columns=COLS)
        p = p[~p["country"].astype(str).isin(drop_countries)]
        f = p.groupby(["country", "crop", "year", "field_year"], observed=True)[["target", "pred"]].mean()
        parts.append(f.reset_index().assign(fold=y))
    if not parts:
        return pd.DataFrame(columns=COLS + ["fold"])
    f = pd.concat(parts, ignore_index=True)
    for c in ("country", "crop", "year"):
        f[c] = f[c].astype(str)
    return f


def _scores(g: pd.DataFrame) -> pd.Series:
    y, p = g["target"].to_numpy(), g["pred"].to_numpy()
    sst = ((y - y.mean()) ** 2).sum()
    return pd.Series({"n_fields": len(g), "r2": 1 - ((y - p) ** 2).sum() / sst if sst > 0 else np.nan,
                      "rrmse": 100 * np.sqrt(((y - p) ** 2).mean()) / y.mean(),
                      "bias": 100 * (p.mean() / y.mean() - 1)})


def scores(f: pd.DataFrame, min_fields: int = CV_MIN_FIELDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(groups, cells): scores per country x crop over the pooled CV years, and per country x crop x year; only cells
    with >= min_fields field-years count (also in the pooled groups)."""
    n = f.groupby(["country", "crop", "year"])["field_year"].transform("size")
    f = f[n >= min_fields]
    if f.empty:
        return (pd.DataFrame(columns=["country", "crop", "n_years", "n_fields", "r2", "rrmse", "bias"]),
                pd.DataFrame(columns=["country", "crop", "year", "n_fields", "r2", "rrmse", "bias"]))
    cells = f.groupby(["country", "crop", "year"]).apply(_scores, include_groups=False).reset_index()
    groups = f.groupby(["country", "crop"]).apply(_scores, include_groups=False).reset_index()
    groups.insert(2, "n_years", f.groupby(["country", "crop"])["year"].nunique().to_numpy())
    for t in (cells, groups):
        t["n_fields"] = t["n_fields"].astype(int)
    return groups, cells


def summary(f: pd.DataFrame, min_fields: int = CV_MIN_FIELDS) -> dict:
    """Headline numbers of one config and set: medians over groups and over cells, and one pooled rRMSE."""
    groups, cells = scores(f, min_fields)
    n = f.groupby(["country", "crop", "year"])["field_year"].transform("size") if len(f) else pd.Series(dtype=int)
    k = f[n >= min_fields] if len(f) else f
    return {"folds": f["fold"].nunique() if len(f) else 0, "n_groups": len(groups), "n_cells": len(cells),
            "group_r2": groups["r2"].median(), "cell_r2": cells["r2"].median(),
            "group_rrmse": groups["rrmse"].median(), "cell_rrmse": cells["rrmse"].median(),
            "pooled_rrmse": (100 * np.sqrt(((k["target"] - k["pred"]) ** 2).mean()) / k["target"].mean()
                             if len(k) else np.nan)}


def paired(a: pd.DataFrame, b: pd.DataFrame, metric: str, keys: list[str]) -> tuple[float, int, int]:
    """Median over the common rows (groups or cells) of a - b, the number of rows where a is better (higher R2,
    lower rRMSE) and the number of common rows."""
    m = a.merge(b, on=keys, suffixes=("_a", "_b"))
    if m.empty:
        return np.nan, 0, 0
    d = m[f"{metric}_a"] - m[f"{metric}_b"]
    better = (d > 0) if metric == "r2" else (d < 0)
    return float(d.median()), int(better.sum()), len(m)
