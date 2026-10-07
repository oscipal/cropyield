"""Load one split of the split suite (scripts/06_make_split_suite.py).

The suite is one zarr store with a group per scope/method/crop; each group holds one row per field-year
with ``split`` 0 = train, 1 = val, 2 = test (see docs/splits.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def load_split(path: str | Path, scope: str, method: str, crop: str) -> dict:
    """Field names of train/val/test and the split's attributes."""
    ds = xr.open_zarr(path, group=f"{scope}/{method}/{crop}", consolidated=False).load()
    fields = np.array([str(f) for f in ds["field"].values])  # zarr v3 may return StringDType
    split = ds["split"].values
    return {"train_fields": list(fields[split == 0]), "val_fields": list(fields[split == 1]),
            "test_fields": list(fields[split == 2]), "attrs": dict(ds.attrs)}
