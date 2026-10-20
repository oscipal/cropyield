"""Sentinel-2 cubes on a finer time grid, built from the raw scenes (same rules as the YieldSAT cube).

    python scripts/LSTM/11_build_s2_grid.py --window-days 10            # -> <work_dir>/lstm/data_w10/
    python scripts/LSTM/11_build_s2_grid.py --check 5                   # verify against lstm/data, no output
    python scripts/LSTM/11_build_s2_grid.py --verify                    # whole data_w30 against lstm/data

Rules (checked with --check to reproduce lstm/data/dyn.npy with 30-day windows from day 0): day of a scene =
its date minus 1 January of the year before the harvest year (metadata-<field>.json: harvesting_date), minus 1,
so the 30-day windows are the intervals (30 k, 30 (k + 1)] of days since 1 January; window = (day - start_day)
// days;
per window the scene with the fewest SCL 8 + 9 (cloud) pixels on the raster, ties to the latest; pixel values
only where SCL is 4 or 5 (vegetation, bare soil), else missing; values are the raw DN.
Known difference (--verify): 410 of 12.4M pixels in 9 Brazilian field-years have SCL 0 in the chosen scene and a
value in the YieldSAT cube; here they are missing. Keeping SCL 0 everywhere is wrong (311k pixels then differ).

Outputs in <work_dir>/lstm/data_w<days>/ (pixel rows identical to lstm/data/meta.parquet):
    dyn.npy       uint16 (pixels, T, 12)  the 12 S2 bands, 65535 = missing (saturated DN 65535 stored as 65534)
    info.json     dynamic_bands, n_times, window_days, start_day, country_rows
    static.npy, meta.parquet, nbr_k*.npy  symlinks to lstm/data
The grid covers [start_day, 720) like the 24 x 30-day cube; start_day is the largest multiple of 30 before the
first scene with a clear pixel on its field (earlier scenes would only add empty steps; printed). Only counts and match statistics are printed, never pixel values (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import country_dirs, load_paths  # noqa: E402

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
END_DAY = 720          # 24 x 30 days, end of the original cube
MISSING = np.uint16(65535)
CLOUD, CLEAR = (8, 9), (4, 5)


def scenes(field_dir: Path) -> tuple[date, list[tuple[int, Path, Path]]]:
    """(1 Jan of the year before the harvest year, [(day, image, scl)]) of one field-year, sorted by day.
    day = days since that 1 January minus 1 (day -1 = 1 January itself, outside the grid)."""
    meta = json.loads((field_dir / f"metadata-{field_dir.name}.json").read_text())
    # harvest year from harvesting_date: the label `year` is off by one in a few field-years
    harvest = meta.get("harvesting_date")
    year = datetime.strptime(harvest, "%d.%m.%Y").year if harvest else int(meta["year"])
    jan1 = date(year - 1, 1, 1)
    out = []
    for img in sorted((field_dir / "s2_images").glob("S2_L2A_*.tif")):
        d = img.stem.rsplit("_", 1)[1]
        scl = field_dir / "scl_masks" / f"S2_L2A_SCL_{d}.tif"
        out.append(((datetime.strptime(d, "%Y%m%d").date() - jan1).days - 1, img, scl))
    return jan1, sorted(out)


def build_field(field_dir: Path, rows: np.ndarray, cols: np.ndarray, days: int, start: int, n_times: int
                ) -> tuple[np.ndarray, int, int, int]:
    """(len(rows), n_times, 12) uint16 cube of one field-year, number of scenes in the grid, windows filled,
    clear values clipped from 65535 to 65534."""
    _, sc = scenes(field_dir)
    best: dict[int, tuple[int, int, Path, np.ndarray]] = {}
    n_in = 0
    for day, img, scl_path in sc:
        if not start <= day < start + n_times * days:
            continue
        n_in += 1
        k = (day - start) // days
        with rasterio.open(scl_path) as r:
            scl = r.read(1)
        clouds = int(np.isin(scl, CLOUD).sum())
        # fewest cloud pixels; ties: the later scene (sorted by day, so >= keeps the latest)
        if k not in best or (-clouds, day) >= (-best[k][0], best[k][1]):
            best[k] = (clouds, day, img, scl)
    out = np.full((len(rows), n_times, len(BANDS)), MISSING, dtype=np.uint16)
    n_clip = 0
    for k, (_, _, img, scl) in best.items():
        with rasterio.open(img) as r:
            v = r.read()[:, rows, cols].T                             # (pixels, 12)
        clear = np.isin(scl[rows, cols], CLEAR)
        sat = clear[:, None] & (v == MISSING)
        n_clip += int(sat.sum())
        out[:, k] = np.where(clear[:, None], np.where(sat, MISSING - 1, v), MISSING)
    return out, n_in, len(best), n_clip


def field_dirs(paths: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per field-year: country, field, first and last row in meta, raw folder."""
    f = meta["field"].to_numpy()
    start = np.r_[0, np.flatnonzero(f[1:] != f[:-1]) + 1]
    stop = np.r_[start[1:], len(f)]
    t = pd.DataFrame({"country": meta["country"].to_numpy()[start], "field": f[start], "start": start, "stop": stop})
    assert t["field"].is_unique, "rows of a field are not contiguous in meta.parquet"
    t["dir"] = [country_dirs(paths, c)[1] / name for c, name in zip(t["country"], t["field"])]
    return t


# --------------------------------------------------------------------------- workers

_G: dict = {}


def _init(meta_path, out_path, days, start, n_times):
    m = pd.read_parquet(meta_path, columns=["row", "col"])
    _G.update(rows=m["row"].to_numpy(), cols=m["col"].to_numpy(), days=days, start=start, n_times=n_times,
              out=None if out_path is None else np.load(out_path, mmap_mode="r+"))


def _work(task):
    i, d, a, b = task
    cube, n_in, n_win, n_clip = build_field(d, _G["rows"][a:b], _G["cols"][a:b], _G["days"], _G["start"],
                                            _G["n_times"])
    if _G["out"] is not None:
        _G["out"][a:b] = cube
        return i, n_in, n_win, n_clip, None
    return i, n_in, n_win, n_clip, cube


def _earliest(task):
    """Earliest scene day in [0, END_DAY) with a clear pixel on the field, number of scenes outside the grid."""
    d, a, b = task
    rows, cols = _G["rows"][a:b], _G["cols"][a:b]
    sc = scenes(d)[1]
    outside = sum(not 0 <= day < END_DAY for day, _, _ in sc)
    for day, _, scl_path in sc:
        if 0 <= day < END_DAY:
            with rasterio.open(scl_path) as r:
                if np.isin(r.read(1)[rows, cols], CLEAR).any():
                    return day, outside
    return np.nan, outside


def earliest_days(fields: pd.DataFrame, meta_path: Path, workers: int) -> pd.DataFrame:
    tasks = [(r.dir, r.start, r.stop) for r in fields.itertuples()]
    with Pool(workers, _init, (meta_path, None, 30, 0, 24)) as pool:
        res = pool.map(_earliest, tasks, chunksize=8)
    return pd.DataFrame(res, columns=["first", "outside"], index=fields.index)


# --------------------------------------------------------------------------- main

def meta_rows(meta_path, a, b):
    m = pd.read_parquet(meta_path, columns=["row", "col"]).iloc[a:b]
    return m["row"].to_numpy(), m["col"].to_numpy()


def diagnose(r, want, meta_path):
    """Which raw scene each window of the reference was taken from: the scene whose clear mask and values equal
    the reference on the field. Prints day numbers and the window choice of the current rule only."""
    rc = meta_rows(meta_path, r.start, r.stop)
    jan1, sc = scenes(r.dir)
    meta = json.loads((r.dir / f"metadata-{r.dir.name}.json").read_text())
    rel = {k: (datetime.strptime(meta[k], "%d.%m.%Y").date() - jan1).days
           for k in ("seeding_date", "harvesting_date") if meta.get(k)}
    src = {}
    for day, img, scl_path in sc:
        with rasterio.open(scl_path) as s:
            clear = np.isin(s.read(1)[rc], CLEAR)
        with rasterio.open(img) as s:
            v = s.read()[:, rc[0], rc[1]].T.astype(np.float32)
        for k in range(want.shape[1]):
            w = want[:, k]
            ok = ~np.isnan(w[:, 0])
            if ok.any() and np.array_equal(ok, clear) and np.array_equal(w[ok], v[ok]):
                src.setdefault(k, []).append(day)
    days = [s[0] for s in sc]
    print(f"      seeding/harvest day {rel}; scene days {days[0]}..{days[-1]} ({len(days)})")
    print("      reference window <- scene day: " + ", ".join(f"{k}<-{d}" for k, d in sorted(src.items())))
    empty = [k for k in range(want.shape[1]) if np.isnan(want[:, k, 0]).all()
             and any(d // 30 == k for d in days)]
    print(f"      reference windows empty although scenes exist: {empty}")


def check(fields, meta_path, data_dir, n, workers, seed=0):
    """Rebuild n random field-years per country with 30-day windows from day 0 and compare with lstm/data."""
    ref = np.load(data_dir / "dyn.npy", mmap_mode="r")
    info = json.loads((data_dir / "info.json").read_text())
    bidx = [info["dynamic_bands"].index(b) for b in BANDS]
    pick = fields.groupby("country").sample(min(n, fields["country"].value_counts().min()), random_state=seed)
    tasks = [(i, r.dir, r.start, r.stop) for i, r in pick.iterrows()]
    bad = 0
    with Pool(workers, _init, (meta_path, None, 30, 0, 24)) as pool:
        for i, _, n_win, n_clip, cube in pool.imap_unordered(_work, tasks):
            r = fields.loc[i]
            want = np.asarray(ref[r.start:r.stop][:, :, bidx])
            want[want == float(MISSING)] = MISSING - 1    # the only intended difference: saturated DN clipped
            got = np.where(cube == MISSING, np.nan, cube.astype(np.float32))
            same_nan = np.array_equal(np.isnan(got), np.isnan(want))
            same = same_nan and np.array_equal(got[~np.isnan(got)], want[~np.isnan(want)])
            n_valid = int((~np.isnan(want[:, :, 0])).sum())
            print(f"  {r.country:12s} {r.field:55s} windows {n_win:2d}  valid pixel-steps {n_valid:7,}  "
                  f"clipped {n_clip:5,}  "
                  f"{'OK' if same else 'MISMATCH' + ('' if same_nan else ' (missing positions differ)')}",
                  flush=True)
            bad += not same
            if not same:
                diagnose(r, want, meta_path)
    print(f"check: {len(tasks) - bad} / {len(tasks)} field-years identical")
    if bad:
        raise SystemExit(1)


def verify(data_dir: Path, built_dir: Path, allow: int = 0, block: int = 500_000) -> None:
    """Compare a whole 30-day build with lstm/data: same values and missing positions after start_day, and the
    reference windows before start_day empty."""
    info = json.loads((built_dir / "info.json").read_text())
    assert info["window_days"] == 30, "verify needs the 30-day build"
    ref = np.load(data_dir / "dyn.npy", mmap_mode="r")
    got = np.load(built_dir / "dyn.npy", mmap_mode="r")
    bidx = [json.loads((data_dir / "info.json").read_text())["dynamic_bands"].index(b) for b in BANDS]
    s = info["start_day"] // 30
    assert got.shape[1] == ref.shape[1] - s
    n_bad = n_early = 0
    bad_rows, bad_win = [], []
    for i in range(0, len(ref), block):
        r = np.asarray(ref[i:i + block][:, :, bidx])
        n_early += int((~np.isnan(r[:, :s])).any((1, 2)).sum())
        r = r[:, s:]
        r[r == float(MISSING)] = MISSING - 1
        g = np.asarray(got[i:i + block])
        g = np.where(g == MISSING, np.nan, g.astype(np.float32))
        diff = (~((r == g) | (np.isnan(r) & np.isnan(g)))).any(2)            # (pixels, T)
        rows = np.flatnonzero(diff.any(1))
        n_bad += len(rows)
        bad_rows.append(rows + i)
        bad_win.append([np.flatnonzero(diff[j]) + s for j in rows])
        print(f"  {min(i + block, len(ref)):,}/{len(ref):,} pixels, differing {n_bad:,}", flush=True)
    print(f"verify: {n_bad:,} pixels differ, {n_early:,} pixels have reference values before start day")
    if n_bad:
        meta = pd.read_parquet(data_dir / "meta.parquet", columns=["field"])
        rows = np.concatenate(bad_rows)
        wins = [w for part in bad_win for w in part]
        d = pd.DataFrame({"field": meta["field"].to_numpy()[rows], "windows": [tuple(w) for w in wins]})
        print(d.groupby("field").agg(pixels=("windows", "size"),
                                     windows=("windows", lambda x: sorted(set(w for t in x for w in t))))
              .to_string())
    if n_bad > allow or n_early:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=10, choices=[30, 15, 10])
    ap.add_argument("--start-day", type=int, default=None, help="default: from the earliest scene")
    ap.add_argument("--check", type=int, default=0, help="verify N field-years per country against lstm/data")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true", help="compare lstm/data_w30 with lstm/data, no build")
    ap.add_argument("--allow-diff", type=int, default=0, help="--verify passes with at most this many pixels differing")
    args = ap.parse_args()

    paths = load_paths()
    data_dir = paths["work_dir"] / "lstm" / "data"
    meta_path = data_dir / "meta.parquet"
    meta = pd.read_parquet(meta_path, columns=["country", "field"])
    fields = field_dirs(paths, meta)
    print(f"{len(fields):,} field-years, {len(meta):,} pixels", flush=True)

    if args.verify:
        verify(data_dir, paths["work_dir"] / "lstm" / "data_w30", args.allow_diff)
        return
    if args.check:
        check(fields, meta_path, data_dir, args.check, args.workers)
        return

    first = earliest_days(fields, meta_path, args.workers)
    print("earliest scene day with a clear pixel on the field, per country (days after 1 Jan of the year "
          f"before harvest; scenes outside [0, {END_DAY}) are not in the original cube and stay out):")
    print(first.groupby(fields["country"]).agg(first_min=("first", "min"), first_median=("first", "median"),
                                               scenes_outside=("outside", "sum")).to_string(), flush=True)
    start = args.start_day if args.start_day is not None else int(first["first"].min() // 30 * 30)
    n_times = (END_DAY - start) // args.window_days
    print(f"grid: start day {start}, {args.window_days}-day windows, {n_times} steps", flush=True)

    out_dir = paths["work_dir"] / "lstm" / f"data_w{args.window_days}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dyn.npy"
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(len(meta), n_times, len(BANDS)))
    del out

    t0, n_in, n_win, n_clip = time.time(), 0, 0, 0
    tasks = [(i, r.dir, r.start, r.stop) for i, r in fields.iterrows()]
    with Pool(args.workers, _init, (meta_path, out_path, args.window_days, start, n_times)) as pool:
        for j, (_, a, b, c, _) in enumerate(pool.imap_unordered(_work, tasks, chunksize=4), 1):
            n_in += a
            n_win += b
            n_clip += c
            if j % 100 == 0 or j == len(tasks):
                print(f"  {j:,}/{len(tasks):,} field-years ({time.time() - t0:.0f} s)", flush=True)
    print(f"scenes in the grid {n_in:,}, filled windows {n_win:,} "
          f"(mean per field-year {n_win / len(tasks):.1f} of {n_times}), saturated values clipped {n_clip:,}")

    src_info = json.loads((data_dir / "info.json").read_text())
    info = {"dynamic_bands": BANDS, "n_pixels": len(meta), "n_times": n_times, "window_days": args.window_days,
            "start_day": start, "missing_value": int(MISSING), "country_rows": src_info["country_rows"],
            "static_bands": src_info["static_bands"], "source": "scripts/LSTM/11_build_s2_grid.py"}
    (out_dir / "info.json").write_text(json.dumps(info, indent=1))
    for p in [data_dir / "static.npy", meta_path, *sorted(data_dir.glob("nbr_k*.npy"))]:
        link = out_dir / p.name
        if not link.exists():
            link.symlink_to(p)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
