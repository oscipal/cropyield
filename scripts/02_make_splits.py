"""CV10 splits: StratifiedGroupKFold (groups = field, strata = farm) + inner 10 % val split per fold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT, load_paths, load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "ger_r_lstm.yaml"))
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = load_paths()
    seed = cfg["seed"]
    n_splits = cfg["splits"]["n_splits"]
    meta = pd.read_parquet(paths["work_dir"] / cfg["data"]["subdir"] / "meta.parquet",
                           columns=["field", "farm", "target"])

    groups = meta["field"].to_numpy()
    strata = pd.factorize(meta["farm"])[0]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for k, (train_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(meta)), strata, groups)):
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg["splits"]["inner_val_fraction"], random_state=seed + k)
        fit_rel, val_rel = next(gss.split(train_idx, groups=groups[train_idx]))
        fit_fields = sorted(set(groups[train_idx[fit_rel]]))
        val_fields = sorted(set(groups[train_idx[val_rel]]))
        test_fields = sorted(set(groups[test_idx]))
        assert not set(fit_fields) & set(test_fields) and not set(val_fields) & set(test_fields)
        assert not set(fit_fields) & set(val_fields)
        folds.append({"fold": k, "train_fields": fit_fields, "val_fields": val_fields, "test_fields": test_fields})
        test_farms = meta.loc[test_idx, "farm"].value_counts().to_dict()
        print(f"fold {k}: train {len(fit_fields)} / val {len(val_fields)} / test {len(test_fields)} fields; "
              f"test pixels {len(test_idx):,}; test farms {test_farms}")

    all_test = [f for fold in folds for f in fold["test_fields"]]
    assert len(all_test) == len(set(all_test)) == meta["field"].nunique()

    out = {"seed": seed, "n_splits": n_splits, "method": "StratifiedGroupKFold(shuffle=True)",
           "group": "field", "stratify": "farm",
           "inner_val": f"GroupShuffleSplit(test_size={cfg['splits']['inner_val_fraction']}, random_state=seed+fold)",
           "folds": folds}
    out_path = paths["splits_dir"] / cfg["splits"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
