"""Mean baselines per split: every test pixel gets the mean yield of the training pixels (train + val; no
model selection is needed), either overall or per group (e.g. crop x country, falling back to coarser groups
when a group is missing from training).

Writes metrics.json in the same layout as 02_train.py, so 03_report.py treats a baseline like a model:
<work_dir>/lstm/runs/<config name>/<scope>__<method>__<crop>/metrics.json (+ preds_test.parquet).
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
sys.path.insert(0, str(HERE))
from lstm_v2 import TargetScaler  # noqa: E402
from src.config import REPO_ROOT, load_paths, load_yaml  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.splits import load_split  # noqa: E402


def all_split_names(store: Path) -> list[str]:
    import zarr
    root = zarr.open_group(store, mode="r")
    return [f"{s}/{m}/{c}" for s in sorted(root.group_keys()) for m in sorted(root[s].group_keys())
            for c in sorted(root[s][m].group_keys())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "baseline_mean.yaml"))
    ap.add_argument("--splits", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = load_paths()
    meta = pd.read_parquet(paths["work_dir"] / "lstm" / "data" / "meta.parquet")
    run_root = paths["work_dir"] / "lstm" / "runs" / cfg["name"]
    by = cfg.get("group_by") or []

    for name in args.splits or all_split_names(paths["split_suite"]):
        scope, method, crop = name.split("/")
        s = load_split(paths["split_suite"], scope, method, crop)
        role_of = {f: r for r, key in ((0, "train_fields"), (1, "val_fields"), (2, "test_fields")) for f in s[key]}
        role = meta["field"].map(role_of).to_numpy()
        fit = np.flatnonzero((role == 0) | (role == 1))
        test = np.flatnonzero(role == 2)

        # TargetScaler: per-group training means with fallback to coarser groups, then the global mean
        mean, _ = TargetScaler(by).fit(meta, fit).params(meta, test)
        p = meta.iloc[test][["pixel", "country", "field", "farm", "crop", "year", "target"]].reset_index(drop=True)
        p["pred"] = mean
        p["field_year"] = p["field"]
        metrics = {"all": compute_metrics(p)}
        for (country, c), g in p.groupby(["country", "crop"], observed=True):
            metrics[f"{country}|{c}"] = compute_metrics(g)

        out_dir = run_root / name.replace("/", "__")
        out_dir.mkdir(parents=True, exist_ok=True)
        p.to_parquet(out_dir / "preds_test.parquet", index=False)
        summary = {"split": name, "config": cfg["name"], "split_attrs": s["attrs"], "best_epoch": None,
                   "epochs_run": 0, "n_pixels": {"fit": int(len(fit)), "test": int(len(test))},
                   "n_fields": {"train": int(meta["field"].iloc[fit].nunique()),
                                "test": int(meta["field"].iloc[test].nunique())},
                   "metrics": {"test": metrics}, "group_by": by}
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1, default=str))
        m = metrics["all"]
        print(f"{cfg['name']:18s} {name:28s} field R2 {m['field_r2']:6.2f} RMSE {m['field_rmse']:5.2f} | "
              f"pixel R2 {m['pixel_r2']:6.2f}", flush=True)


if __name__ == "__main__":
    main()
