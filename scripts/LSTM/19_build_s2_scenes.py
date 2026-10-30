"""All Sentinel-2 scenes per field-year, without time windows: the store behind the scene-sequence input
(`inputs.scenes`, world variant wv_scenes) and the GDD grid (scripts/LSTM/20_build_gdd_grid.py).

    python scripts/LSTM/19_build_s2_scenes.py                 # -> <work_dir>/lstm/scenes/
    python scripts/LSTM/19_build_s2_scenes.py --limit 20      # quick test on the first 20 field-years (separate folder)

Same rules as the windowed cubes (scripts/LSTM/11_build_s2_grid.py): scene day = date minus 1 January of the year
before the harvest year, minus 1; pixel values only where the scene's SCL class is 4 or 5 (vegetation, bare soil),
raw DN, saturated 65535 stored as 65534, 65535 = missing. Kept: every scene with day in [210, 720) (the day range of
the S2 grids) in which at least MIN_CLEAR of the field's labelled pixels are clear; the others carry no usable
information and would only lengthen the sequences.

Outputs in <work_dir>/lstm/scenes/:
    values.u16      raw uint16, shape (n_pixel_scenes, 12) row-major: per field-year (meta.parquet order) its pixels
                    one after the other, each with its kept scenes in day order (pixel p of field-year f starts at
                    offset_f + (p - start_f) * n_scenes_f)
    fields.parquet  field, start, stop (rows in meta.parquet), n_scenes, offset (row in values.u16)
    scenes.parquet  field, k (position in the field-year's sequence), day, clear_share (share of the field's pixels
                    with SCL 4 / 5)
    info.json       shape, bands, day range, MIN_CLEAR
Only counts are printed, never values (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from src.config import load_paths  # noqa: E402

grid = importlib.import_module("11_build_s2_grid")
BANDS, MISSING, CLEAR = grid.BANDS, grid.MISSING, grid.CLEAR
START_DAY, END_DAY = 210, 720
MIN_CLEAR = 0.05

_G: dict = {}


def _init(meta_path):
    m = pd.read_parquet(meta_path, columns=["row", "col"])
    _G.update(rows=m["row"].to_numpy(), cols=m["col"].to_numpy())


def _work(task):
    """(index, uint16 (pixels, n_kept, 12), [(day, clear_share)], n_scenes_in_range)."""
    i, d, a, b = task
    rows, cols = _G["rows"][a:b], _G["cols"][a:b]
    kept, info, n_in = [], [], 0
    for day, img, scl_path in grid.scenes(d)[1]:
        if not START_DAY <= day < END_DAY:
            continue
        n_in += 1
        with rasterio.open(scl_path) as r:
            clear = np.isin(r.read(1)[rows, cols], CLEAR)
        share = float(clear.mean())
        if share < MIN_CLEAR:
            continue
        with rasterio.open(img) as r:
            v = r.read()[:, rows, cols].T                                 # (pixels, 12)
        v = np.where(v == MISSING, MISSING - 1, v)
        kept.append(np.where(clear[:, None], v, MISSING).astype(np.uint16))
        info.append((day, share))
    cube = np.stack(kept, axis=1) if kept else np.zeros((b - a, 0, len(BANDS)), np.uint16)
    return i, cube, info, n_in


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N field-years only (test build in scenes_test/)")
    args = ap.parse_args()
    paths = load_paths()
    lstm = paths["work_dir"] / "lstm"
    meta_path = lstm / "data" / "meta.parquet"
    fields = grid.field_dirs(paths, pd.read_parquet(meta_path, columns=["field", "country"]))
    if args.limit:
        fields = fields.iloc[:args.limit]
    out = lstm / ("scenes_test" if args.limit else "scenes")
    out.mkdir(exist_ok=True)
    tasks = [(i, r.dir, r.start, r.stop) for i, r in fields.iterrows()]
    n_sc, offset, rows_sc, n_in_all = np.zeros(len(fields), np.int64), np.zeros(len(fields), np.int64), [], 0
    pos, t0 = 0, time.time()
    with open(out / "values.u16", "wb") as fh, Pool(args.workers, _init, (meta_path,)) as pool:
        for j, (i, cube, info, n_in) in enumerate(pool.imap(_work, tasks, chunksize=2)):
            fh.write(np.ascontiguousarray(cube).tobytes())
            n_sc[j], offset[j] = cube.shape[1], pos
            pos += cube.shape[0] * cube.shape[1]
            n_in_all += n_in
            rows_sc += [(fields.at[i, "field"], k, day, share) for k, (day, share) in enumerate(info)]
            if (j + 1) % 100 == 0:
                print(f"  {j + 1} / {len(tasks)} field-years, {pos:,} pixel-scenes, {time.time() - t0:.0f} s",
                      flush=True)
    t = fields[["field", "start", "stop"]].copy()
    t["n_scenes"], t["offset"] = n_sc, offset
    t.to_parquet(out / "fields.parquet", index=False)
    pd.DataFrame(rows_sc, columns=["field", "k", "day", "clear_share"]).to_parquet(out / "scenes.parquet", index=False)
    (out / "info.json").write_text(json.dumps({
        "shape": [int(pos), len(BANDS)], "dtype": "uint16", "bands": BANDS, "missing_value": int(MISSING),
        "day_range": [START_DAY, END_DAY], "min_clear": MIN_CLEAR, "source": "scripts/LSTM/19_build_s2_scenes.py"},
        indent=1))
    print(f"done: {len(t)} field-years, {n_in_all:,} scenes in the day range, {int(n_sc.sum()):,} kept "
          f"(median {int(np.median(n_sc))}, max {int(n_sc.max())} per field-year), {pos:,} pixel-scenes, "
          f"{pos * len(BANDS) * 2 / 1e9:.1f} GB, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
