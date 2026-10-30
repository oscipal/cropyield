"""Peak-NDVI baseline per split, saved as a regular run: per country x crop OLS of field-mean yield on field peak NDVI,
fitted on the split's train field-years (the `peak_ndvi` baseline of scripts/loyo_eval.py, whose functions are reused
here; Meroni et al. 2021 benchmark). Every val / test pixel gets its field-year's prediction.

    python scripts/LSTM/17_baseline_peak_ndvi.py                          # the six pooled all-crop splits
    python scripts/LSTM/17_baseline_peak_ndvi.py --splits europe/loyo/all

Peak NDVI = max over the 10-day steps that start between sowing and harvest (src.loader.seasons()) of the field-mean
NDVI of 100 random pixels per field-year; a step counts only if >= 25 % of those pixels are clear. Fit per country x
crop on the train field-years (pooled over the country's crops if < 5 train fields with a peak); val / test
field-years without a valid peak get the training mean of their country x crop. The peaks of train + val field-years
are computed in one call as in loyo_eval.py (the 100-pixel sample depends on the set loaded), so val reproduces
docs/loyo_eval.md; the test field-years are loaded in a second call.

Writes the layout of 04_train_v2.py, so the viewer and the reports treat the baseline like a one-member model:
<work_dir>/lstm/runs/peak_ndvi/<scope>__<method>__<crop>/metrics.json + preds_val.parquet + preds_test.parquet
(pred = pred_m0, pred_std = 0; metrics_member_mean = metrics).

CLAUDE.md: no Swiss data values; the console shows R2 and counts only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
import loyo_eval as E  # noqa: E402
import src.loader as L  # noqa: E402
from src.config import load_paths  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.splits import load_split  # noqa: E402

NAME = "peak_ndvi"
SPLITS = [f"{s}/{m}/all" for s in ("world", "europe", "south_america") for m in ("loyo", "loro")]
ROLES = {0: "train", 1: "val", 2: "test"}
COLS = ["pixel", "country", "field", "farm", "crop", "year", "row", "col", "target"]
CHECK = ("europe/loyo/all", "south_america/loyo/all", -0.418)   # docs/loyo_eval.md: median val subset R2, all fields


def group_metrics(preds: pd.DataFrame, col: str = "pred") -> dict:
    p = preds.assign(pred=preds[col])
    out = {"all": compute_metrics(p)}
    for (country, crop), g in p.groupby(["country", "crop"], observed=True):
        out[f"{country}|{crop}"] = compute_metrics(g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    args = ap.parse_args()

    L._base = E._lean_base                                   # this process only: low-memory pixel table
    paths = load_paths()
    meta = E._lean_base()["meta"]
    ft = pd.read_parquet(paths["work_dir"] / "fields.parquet").set_index("field")
    run_root = paths["work_dir"] / "lstm" / "runs" / NAME
    val_metrics = {}

    for name in args.splits:
        scope, method, crop = name.split("/")
        s = load_split(paths["split_suite"], scope, method, crop)
        tr, va, te = (ft.loc[s[k]] for k in ("train_fields", "val_fields", "test_fields"))
        print(f"{name}: peak NDVI of {len(tr) + len(va)} train + val and {len(te)} test field-years", flush=True)
        peak = pd.concat([E.peak_ndvi(list(tr.index) + list(va.index)), E.peak_ndvi(list(te.index))])

        role_of = {f: r for r, d in enumerate((tr, va, te)) for f in d.index}
        cat_role = pd.Series(meta["field"].cat.categories).map(role_of).fillna(-1).to_numpy(int)
        role = cat_role[meta["field"].cat.codes.to_numpy()]
        idx = {r: np.flatnonzero(role == r) for r in ROLES}
        metrics, info = {}, {}
        out_dir = run_root / name.replace("/", "__")
        out_dir.mkdir(parents=True, exist_ok=True)
        for r, ev in ((1, va), (2, te)):
            b, info[ROLES[r]] = E.baselines(tr, ev, peak)
            field_pred = b["peak_ndvi"]["m0"]
            p = meta.iloc[idx[r]][COLS].reset_index(drop=True)
            p["field"] = p["field"].astype(str)
            for c in ("country", "farm", "crop", "year"):
                p[c] = p[c].cat.remove_unused_categories()
            p["pred_m0"] = p["field"].map(field_pred).astype(np.float32)
            p["pred"] = p["pred_m0"]
            p["pred_std"] = np.float32(0)
            p["field_year"] = p["field"]
            if p["pred"].isna().any():
                raise ValueError(f"{name} {ROLES[r]}: field-years without a prediction")
            fm = p.groupby("field", observed=True)["target"].mean()
            info[ROLES[r]]["target_is_yield_mean"] = bool(np.allclose(fm, ev["yield_mean"].reindex(fm.index),
                                                                      rtol=1e-4, atol=1e-4))
            p.to_parquet(out_dir / f"preds_{ROLES[r]}.parquet", index=False)
            metrics[ROLES[r]] = group_metrics(p)

        summary = {"split": name, "config": NAME, "split_attrs": s["attrs"], "best_epoch": None, "epochs_run": 0,
                   "n_pixels": {ROLES[r]: int(len(idx[r])) for r in idx},
                   "n_fields": {ROLES[r]: int(meta["field"].iloc[idx[r]].nunique()) for r in idx},
                   "metrics": metrics, "metrics_member_mean": metrics,
                   "metrics_members": {role: [m] for role, m in metrics.items()},
                   "baseline": {"min_fields": E.MIN_FIELDS, "min_clear": E.MIN_CLEAR, "pixels": E.PIXELS, **info}}
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1, default=str))
        val_metrics[name] = metrics["val"]
        mv, mt = metrics["val"]["all"], metrics["test"]["all"]
        print(f"{NAME} {name:24s} val field R2 {mv['field_r2']:6.2f} ({mv['n_fields']} fields) | test field R2 "
              f"{mt['field_r2']:6.2f} ({mt['n_fields']} fields) | {summary['n_pixels']} pixels | "
              f"pooled fits val/test {info['val']['pooled_fits']}/{info['test']['pooled_fits']}, no peak "
              f"train/val/test {info['val']['no_peak_train']}/{info['val']['no_peak_val']}/"
              f"{info['test']['no_peak_val']} | target = yield_mean: "
              f"{info['val']['target_is_yield_mean'] and info['test']['target_is_yield_mean']}", flush=True)

    *check, want = CHECK
    if all(c in val_metrics for c in check):   # median over the val subsets with >= 5 fields
        r2s = [m["field_r2"] for c in check for g, m in val_metrics[c].items()
               if g != "all" and m["n_fields"] >= E.MIN_FIELDS]
        got = float(np.median(r2s))
        print(f"check vs docs/loyo_eval.md: median val field R2 over {len(r2s)} subsets {got:.3f} (doc {want:.3f}) "
              f"{'ok' if abs(got - want) < 5e-4 else 'MISMATCH'}", flush=True)


if __name__ == "__main__":
    main()
