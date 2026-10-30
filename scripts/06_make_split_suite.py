"""Build all single train/val/test splits (scope x method x crop) into one zarr store.

Store layout (paths.yaml: split_suite), one group per ``{scope}/{method}/{crop}``; cross-validation folds over years
(src.splitting.cv_folds, --methods cv) are groups ``{scope}/cv{year}/all``:
    split           (field,)  int8   0 train, 1 val, 2 test
    country, farm, region, physical_field, crop, year   (field,)  str
    attrs: chosen test/val years or regions, score components, achieved shares
Statistics: scripts/07_split_report.py -> docs/splits.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_paths  # noqa: E402
from src.splits import load_split  # noqa: E402
from src.splitting import CV_YEARS, ROLES, SCOPES, cv_folds, loro_split, loyo_split, subset  # noqa: E402

METHODS = {"loyo": loyo_split, "loro": loro_split}
ALL_METHODS = [*METHODS, "cv"]   # cv: the folds of CV_YEARS[scope], crop "all" only
META_COLS = ["country", "farm", "region", "physical_field", "crop", "year"]


def check(sub: pd.DataFrame, role: np.ndarray, method: str) -> None:
    assert len(role) == len(sub) and set(np.unique(role)) == {0, 1, 2}
    crops = set(sub["crop"])
    assert set(sub.loc[role == 0, "crop"]) == crops, "train misses a crop"
    if method in ("loyo", "cv"):
        ty, vy = set(sub.loc[role == 2, "year"]), set(sub.loc[role == 1, "year"])
        assert len(ty) == len(vy) == 1 and not ty & vy
        assert not set(sub.loc[role == 0, "year"]) & (ty | vy)
    else:
        for col in ("region", "physical_field"):
            assert (sub.groupby(col)["role"].nunique() == 1).all(), f"{col} spans several sets"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20_000, help="random assignments tried per LORO split")
    ap.add_argument("--scopes", nargs="*", default=list(SCOPES), choices=list(SCOPES))
    ap.add_argument("--methods", nargs="*", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--crops", nargs="*", default=None, help="e.g. all; default: every crop plus all")
    args = ap.parse_args()
    paths = load_paths()
    fields = pd.read_parquet(paths["work_dir"] / "fields.parquet")
    store = paths["split_suite"]
    # a full run rebuilds the store; with any filter only the selected groups are (re)written
    partial = (set(args.scopes), set(args.methods), args.crops) != (set(SCOPES), set(ALL_METHODS), None)
    if store.exists() and not partial:
        shutil.rmtree(store)

    summary = []
    for scope in args.scopes:
        for method in args.methods:
            if method == "cv":
                if scope not in CV_YEARS or (args.crops is not None and "all" not in args.crops):
                    continue
                sub = subset(fields, scope, "all")
                jobs = [(name, "all", sub, role, info) for name, role, info in cv_folds(sub, CV_YEARS[scope])]
            else:
                crops = sorted(subset(fields, scope, "all", method)["crop"].unique())
                jobs = []
                for crop in (crops + ["all"] if args.crops is None else args.crops):
                    sub = subset(fields, scope, crop, method)
                    role, info = (METHODS[method](sub) if method == "loyo"
                                  else METHODS[method](sub, n_trials=args.trials))
                    jobs.append((method, crop, sub, role, info))
            for name, crop, sub, role, info in jobs:
                check(sub.assign(role=role), role, method)
                group = f"{scope}/{name}/{crop}"
                shares = {f"share_{k}": float((role == v).mean()) for k, v in ROLES.items()}
                attrs = {"scope": scope, "method": method, "crop": crop, "countries": SCOPES[scope],
                         **info, **shares}
                ds = xr.Dataset({"split": ("field", role.astype(np.int8)),
                                 **{c: ("field", sub[c].astype(str).to_numpy(dtype=object)) for c in META_COLS}},
                                coords={"field": sub["field"].to_numpy(dtype=object)},
                                attrs=json.loads(json.dumps(attrs, default=lambda o: o.item())))
                ds.to_zarr(store, group=group, mode="w", consolidated=False)

                loaded = load_split(store, scope, name, crop)
                for rname, code in ROLES.items():
                    assert sorted(loaded[f"{rname}_fields"]) == sorted(sub.loc[role == code, "field"])

                held = (f"test {info['test_years']} / val {info['val_years']}" if method in ("loyo", "cv") else
                        f"test {len(info['test_regions'])} / val {len(info['val_regions'])} regions")
                summary.append({"split": group, "n": len(sub), "held out": held,
                                "train/val/test": " / ".join(f"{shares[f'share_{k}']:.0%}" for k in ROLES),
                                "score": round(info["score"], 3)})
                print(f"{group:32s} n={len(sub):5d}  {held:40s} "
                      f"{summary[-1]['train/val/test']}  score {info['score']:.3f}", flush=True)

    (paths["work_dir"] / "logs").mkdir(exist_ok=True)
    log = paths["work_dir"] / "logs" / "06_split_suite_summary.csv"
    if partial and log.exists():
        old = pd.read_csv(log)
        summary = old[~old["split"].isin([r["split"] for r in summary])].to_dict("records") + summary
    pd.DataFrame(summary).to_csv(log, index=False)
    print(f"wrote {store}; run scripts/07_split_report.py for docs/splits.md")


if __name__ == "__main__":
    main()
