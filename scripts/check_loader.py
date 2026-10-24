"""Checks of src/loader.py against the existing pipeline (step 5 of .claude/plans/data_loader.md).

    python scripts/check_loader.py

1. S2 + indices + observed on every grid equal scripts/LSTM/lstm_v2.py FeatureSpec.build_dyn on the same cube
   (reflectance = DN / 10000, so the bands are compared after scaling, the indices directly).
2. Weather equals <grid>/weather.npy of the pixel's field; terrain equals terrain.npy; soil single layers equal
   static.npy x unit factor.
3. Split membership equals src/splits.py; shapes, names and days follow info.json; YieldDataset batches are finite.
4. Options: patch k=5 equals the prebuilt nbr_k5.npy, k=3 is the centre of k=7; align="sowing" step m equals grid
   window (sow window + m); season slices the calendar grid; edge distances equal a brute-force search; the scaler
   streamed with filters equals the scaler of the loaded data.
Only counts and pass/fail are printed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "LSTM"))
from lstm_v2 import FeatureSpec  # noqa: E402

from src import soil  # noqa: E402
from src.config import load_paths  # noqa: E402
from src.loader import GRIDS, YieldDataset, _base, edge_distance, fit_scaler, load, neighbours, seasons  # noqa: E402
from src.splits import load_split  # noqa: E402

SPLIT = "europe/loyo/all"
OK = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global OK
    OK &= bool(cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{' (' + detail + ')' if detail else ''}", flush=True)


def main() -> None:
    work = load_paths()["work_dir"] / "lstm"
    for grid, folder in GRIDS.items():
        print(f"grid {grid}")
        info = json.loads((work / folder / "info.json").read_text())
        d = load(SPLIT, "val", ["s2", "indices", "observed", "weather:all", "terrain", "soil:clay,soc@5-15cm"],
                 grid=grid, dtype=np.float32, verbose=False)
        check("shape / days", d.dyn.shape[1] == info["n_times"] and d.days[0] == info["start_day"]
              and d.days[1] - d.days[0] == info["window_days"], f"{d.dyn.shape}")
        spec = FeatureSpec({"dynamic": ["s2"], "indices": ["ndvi", "ndre", "ndmi"], "mask_channel": True}, info)
        ref = spec.build_dyn(np.load(work / folder / "dyn.npy", mmap_mode="r")[d.rows])
        ref[..., :12] /= 10000.0
        a = d.dyn[..., :16]
        same_nan = np.array_equal(np.isnan(a), np.isnan(ref))
        close = np.allclose(np.nan_to_num(a), np.nan_to_num(ref), atol=1e-6)
        check("s2 + indices + observed = FeatureSpec", same_nan and close)

        wj = json.loads((work / folder / "weather.json").read_text())
        w = np.load(work / folder / "weather.npy")
        pos = {f: i for i, f in enumerate(wj["fields"])}
        wref = w[d.meta["field"].map(pos).to_numpy()]
        check("weather = weather.npy", np.array_equal(d.dyn[..., 16:], wref, equal_nan=True))

        tref = np.load(work / "terrain.npy", mmap_mode="r")[d.rows]
        check("terrain = terrain.npy", np.array_equal(d.static[:, :6], tref, equal_nan=True))
        names = json.loads((work / "data" / "info.json").read_text())["static_bands"]
        st = np.load(work / "data" / "static.npy", mmap_mode="r")[d.rows]
        sref = np.stack([st[:, names.index("clay_5-15")] * 0.1, st[:, names.index("soc_5-15")] * 0.1], axis=1)
        check("soil single layer = static x factor", np.allclose(d.static[:, 6:8], sref, atol=1e-4))

    print("splits and torch")
    s = load_split(load_paths()["split_suite"], *SPLIT.split("/"))
    for role in ("train", "val", "test"):
        d = load(SPLIT, role, ["observed"], grid="30d", verbose=False)
        check(f"{role} fields = split store", set(d.meta["field"]) == set(s[f"{role}_fields"]),
              f"{d.meta['field'].nunique()} fields")
    d = load(SPLIT, "train", ["s2", "indices", "observed", "terrain", "crop"], grid="30d", patch=5, verbose=False)
    ds = YieldDataset(d, fit_scaler(d))
    xd, xs, y = (np.stack([b[i].numpy() for b in ds.__getitems__(np.arange(0, len(d), 997))]) for i in range(3))
    check("YieldDataset patch batch finite", np.isfinite(xd).all() and np.isfinite(xs).all(), f"{xd.shape}")
    check("one-hot left unscaled", set(np.unique(xs[:, -1])) <= {0.0, 1.0})
    print("options")
    work_meta = _base()["meta"]
    base = load(SPLIT, "val", ["s2:B04", "observed", "weather:tmean_c"], grid="10d", dtype=np.float32, verbose=False)
    nb5 = np.load(work / "data" / "nbr_k5.npy", mmap_mode="r")[base.rows]
    loc = np.full(len(work_meta), -1)
    loc[base.rows] = np.arange(len(base.rows))
    check("patch 5 = nbr_k5.npy", np.array_equal(neighbours(base.rows, 5), np.where(nb5 >= 0, loc[np.maximum(nb5, 0)], -1)))
    n7 = neighbours(base.rows, 7)
    check("patch 3 = centre of patch 7", np.array_equal(neighbours(base.rows, 3),
                                                       n7[:, [r * 7 + c for r in range(2, 5) for c in range(2, 5)]]))
    sw = load(SPLIT, "val", ["s2:B04", "observed", "weather:tmean_c"], grid="10d", align="sowing", dtype=np.float32,
              verbose=False)
    k0 = (seasons()["sow_day"].reindex(sw.meta["field"]).to_numpy() - 210) // 10
    ok = True
    for i in range(0, len(sw), 101):
        for m in range(len(sw.days)):
            k = k0[i] + m
            ref = base.dyn[i, k] if 0 <= k < base.dyn.shape[1] else np.array([np.nan, 0, np.nan], np.float32)
            ok &= np.array_equal(sw.dyn[i, m], ref, equal_nan=True)
    check("align sowing = shifted calendar grid", ok, f"days {sw.days[0]}..{sw.days[-1]}")
    cal = load(SPLIT, "val", ["observed"], grid="10d", season=(300, 400), dtype=np.float32, verbose=False)
    check("season (300, 400) = calendar slice", np.array_equal(cal.dyn[..., 0], base.dyn[:, 9:19, 1]))
    d = edge_distance(base.rows)
    fld = base.meta["field"].iloc[0]
    allr = np.flatnonzero(work_meta["field"].to_numpy() == fld)
    R, C = (work_meta[c].to_numpy().astype(int)[allr] for c in ("row", "col"))
    occupied = set(zip(R, C))
    empty = np.array([(i, j) for i in range(R.min() - 1, R.max() + 2) for j in range(C.min() - 1, C.max() + 2)
                      if (i, j) not in occupied])
    sel = np.flatnonzero(base.meta["field"].to_numpy() == fld)
    pts = np.stack([base.meta["row"].to_numpy().astype(int)[sel], base.meta["col"].to_numpy().astype(int)[sel]], 1)
    brute = np.sqrt(((empty[None] - pts[:, None]) ** 2).sum(-1)).min(1) * 10
    check("edge distance = brute force (one field)", np.allclose(brute, d[sel]))
    kw = dict(split=SPLIT, role="train", features=["s2", "observed", "terrain", "crop"], grid="10d", align="sowing",
              edge_buffer_m=20, min_observed=5, pixels_per_field=300)
    a, b = fit_scaler(**kw), fit_scaler(load(**kw, verbose=False))
    check("streamed scaler = scaler of loaded data", np.allclose(a.dyn_mean, b.dyn_mean, rtol=1e-3)
          and np.allclose(a.static_mean, b.static_mean, rtol=1e-5))
    print("ALL OK" if OK else "SOME CHECKS FAILED")
    sys.exit(0 if OK else 1)


if __name__ == "__main__":
    main()
