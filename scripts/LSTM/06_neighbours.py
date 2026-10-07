"""Neighbour table for patch models: for every pixel the row indices of its k x k neighbourhood.

Output <work_dir>/lstm/data/nbr_k{k}.npy, int32 (pixels, k*k), row-major over the window (centre at k*k // 2),
-1 where the neighbour is not a labelled pixel of the same field-year (outside the field or no yield).
Neighbours never cross field-years, so a split assigns a pixel and all its neighbours to the same set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_paths  # noqa: E402


def neighbour_table(rows: np.ndarray, cols: np.ndarray, field_start: np.ndarray, k: int) -> np.ndarray:
    """rows/cols of all pixels; field_start: first row index of every field (fields are contiguous)."""
    r = k // 2
    out = np.full((len(rows), k * k), -1, dtype=np.int32)
    bounds = np.r_[field_start, len(rows)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        rr, cc = rows[a:b].astype(np.int64), cols[a:b].astype(np.int64)
        r0, c0 = rr.min(), cc.min()
        grid = np.full((rr.max() - r0 + 1 + 2 * r, cc.max() - c0 + 1 + 2 * r), -1, dtype=np.int32)
        gr, gc = rr - r0 + r, cc - c0 + r
        grid[gr, gc] = np.arange(a, b, dtype=np.int32)
        j = 0
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                out[a:b, j] = grid[gr + dr, gc + dc]
                j += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="window size (odd)")
    args = ap.parse_args()
    assert args.k % 2 == 1, "window size must be odd"

    data_dir = load_paths()["work_dir"] / "lstm" / "data"
    meta = pd.read_parquet(data_dir / "meta.parquet", columns=["field", "row", "col"])
    f = meta["field"].to_numpy()
    start = np.flatnonzero(np.r_[True, f[1:] != f[:-1]])
    assert len(start) == meta["field"].nunique(), "pixels of a field must be contiguous in meta.parquet"

    nbr = neighbour_table(meta["row"].to_numpy(), meta["col"].to_numpy(), start, args.k)
    centre = args.k * args.k // 2
    assert (nbr[:, centre] == np.arange(len(nbr))).all(), "centre of every window must be the pixel itself"
    np.save(data_dir / f"nbr_k{args.k}.npy", nbr)
    print(f"wrote nbr_k{args.k}.npy {nbr.shape}; share of neighbours present: {(nbr >= 0).mean():.3f}")


if __name__ == "__main__":
    main()
