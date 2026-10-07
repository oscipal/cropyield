"""Build all single train/val/test splits (scope x method x crop) into one zarr store.

Store layout (paths.yaml: split_suite), one group per ``{scope}/{method}/{crop}``:
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
from src.splitting import ROLES, SCOPES, loro_split, loyo_split, subset  # noqa: E402

METHODS = {"loyo": loyo_split, "loro": loro_split}
META_COLS = ["country", "farm", "region", "physical_field", "crop", "year"]


def check(sub: pd.DataFrame, role: np.ndarray, method: str) -> None:
    assert len(role) == len(sub) and set(np.unique(role)) == {0, 1, 2}
    crops = set(sub["crop"])
    assert set(sub.loc[role == 0, "crop"]) == crops, "train misses a crop"
    if method == "loyo":
        ty, vy = set(sub.loc[role == 2, "year"]), set(sub.loc[role == 1, "year"])
        assert len(ty) == len(vy) == 1 and not ty & vy
        assert not set(sub.loc[role == 0, "year"]) & (ty | vy)
    else:
        for col in ("region", "physical_field"):
            assert (sub.groupby(col)["role"].nunique() == 1).all(), f"{col} spans several sets"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20_000, help="random assignments tried per LORO split")
    args = ap.parse_args()
    paths = load_paths()
    fields = pd.read_parquet(paths["work_dir"] / "fields.parquet")
    store = paths["split_suite"]
    if store.exists():
        shutil.rmtree(store)

    summary = []
    for scope in SCOPES:
        crops = sorted(subset(fields, scope, "all")["crop"].unique())
        for method, fn in METHODS.items():
            for crop in crops + ["all"]:
                sub = subset(fields, scope, crop)
                role, info = fn(sub) if method == "loyo" else fn(sub, n_trials=args.trials)
                check(sub.assign(role=role), role, method)
                group = f"{scope}/{method}/{crop}"
                shares = {f"share_{k}": float((role == v).mean()) for k, v in ROLES.items()}
                attrs = {"scope": scope, "method": method, "crop": crop, "countries": SCOPES[scope],
                         **info, **shares}
                ds = xr.Dataset({"split": ("field", role.astype(np.int8)),
                                 **{c: ("field", sub[c].astype(str).to_numpy(dtype=object)) for c in META_COLS}},
                                coords={"field": sub["field"].to_numpy(dtype=object)},
                                attrs=json.loads(json.dumps(attrs, default=lambda o: o.item())))
                ds.to_zarr(store, group=group, mode="w", consolidated=False)

                loaded = load_split(store, scope, method, crop)
                for name, code in ROLES.items():
                    assert sorted(loaded[f"{name}_fields"]) == sorted(sub.loc[role == code, "field"])

                held = (f"test {info['test_years']} / val {info['val_years']}" if method == "loyo" else
                        f"test {len(info['test_regions'])} / val {len(info['val_regions'])} regions")
                summary.append({"split": group, "n": len(sub), "held out": held,
                                "train/val/test": " / ".join(f"{shares[f'share_{k}']:.0%}" for k in ROLES),
                                "score": round(info["score"], 3)})
                print(f"{group:32s} n={len(sub):5d}  {held:40s} "
                      f"{summary[-1]['train/val/test']}  score {info['score']:.3f}", flush=True)

    (paths["work_dir"] / "logs").mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(paths["work_dir"] / "logs" / "06_split_suite_summary.csv", index=False)
    print(f"wrote {store}; run scripts/07_split_report.py for docs/splits.md")


if __name__ == "__main__":
    main()
