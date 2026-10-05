"""CV splits stored as a zarr store with one entry per field.

Layout:
    field      (field,)        field_shared_name
    farm       (field,)        farm of the field (stratum)
    test_fold  (field,)  int8  fold in which the field is in the test set
    inner_val  (fold, field) bool  field is in the inner validation split of that fold
Training fields of fold k are all fields with test_fold != k and not inner_val[k].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def save_splits(path: str | Path, fields: np.ndarray, farms: np.ndarray, test_fold: np.ndarray,
                inner_val: np.ndarray, attrs: dict) -> None:
    ds = xr.Dataset(
        {
            "farm": ("field", np.asarray(farms, dtype=object)),
            "test_fold": ("field", np.asarray(test_fold, dtype=np.int8)),
            "inner_val": (("fold", "field"), np.asarray(inner_val, dtype=bool)),
        },
        coords={"field": np.asarray(fields, dtype=object), "fold": np.arange(inner_val.shape[0])},
        attrs=attrs,
    )
    ds.to_zarr(path, mode="w")


def load_splits(path: str | Path) -> list[dict]:
    """Return one dict per fold with train_fields, val_fields and test_fields."""
    ds = xr.open_zarr(path).load()
    fields = ds["field"].values.astype(str)
    test_fold = ds["test_fold"].values
    folds = []
    for k in ds["fold"].values:
        test = test_fold == k
        val = ds["inner_val"].sel(fold=k).values
        train = ~test & ~val
        folds.append({"fold": int(k), "train_fields": list(fields[train]),
                      "val_fields": list(fields[val]), "test_fields": list(fields[test])})
    return folds
