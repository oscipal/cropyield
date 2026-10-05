"""Extract rapeseed (GER-R) pixels once: X.npy (pixels x 24 x 12) + meta.parquet."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT, load_paths, load_yaml  # noqa: E402
from src.data import FeatureReader, load_meta, open_dataset  # noqa: E402

CROP_ALIASES = {"rapeseed": ("rapeseed", "rape", "raps", "canola", "oilseed")}


def select_crop(labels, crop: str) -> str:
    uniques = sorted(set(map(str, labels)))
    aliases = CROP_ALIASES.get(crop.lower(), (crop.lower(),))
    hits = [u for u in uniques if u.lower() == crop.lower()] or \
           [u for u in uniques if any(a in u.lower() for a in aliases)]
    if len(hits) != 1:
        raise ValueError(f"Crop '{crop}' matches {hits}; available labels: {uniques}. Use --crop-label.")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "ger_r_lstm.yaml"))
    ap.add_argument("--crop-label", default=None, help="exact decoded crop label (overrides config)")
    ap.add_argument("--block", type=int, default=100_000)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = load_paths()
    out_dir = paths["work_dir"] / cfg["data"]["subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    bands = cfg["data"]["bands"]

    ds = open_dataset(paths["germany_nc"])
    meta = load_meta(ds)
    crop_label = args.crop_label or select_crop(meta["crop"], cfg["data"]["crop"])
    mask = (meta["crop"].astype(str) == crop_label).to_numpy() & meta["target"].notna().to_numpy()
    n_crop = int((meta["crop"].astype(str) == crop_label).sum())
    print(f"crop '{crop_label}': {n_crop:,} pixels, {mask.sum():,} with valid target")

    reader = FeatureReader(ds)
    print(reader)
    X = np.lib.format.open_memmap(out_dir / "X.npy", mode="w+", dtype=np.float32,
                                  shape=(int(mask.sum()), reader.n_times, len(bands)))
    reader.extract(bands, mask, out=X, block=args.block)
    X.flush()

    sel = meta[mask].reset_index()  # keeps original pixel index as column "pixel"
    cols = [c for c in ("pixel", "field", "field_year", "farm", "year", "crop", "row", "col", "target",
                        "seeding_date_type") if c in sel]
    sel = sel[cols]
    for c in ("field", "field_year", "farm", "year", "crop", "seeding_date_type"):
        if c in sel:
            sel[c] = sel[c].astype(str)
    sel.to_parquet(out_dir / "meta.parquet", index=False)

    info = {"source": str(paths["germany_nc"]), "crop_label": crop_label, "bands": bands,
            "shape": list(X.shape), "n_fields": int(sel["field"].nunique()),
            "n_field_years": int(sel["field_year"].nunique()), "n_farms": int(sel["farm"].nunique())}
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
