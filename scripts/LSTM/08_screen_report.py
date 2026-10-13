"""Compare screening runs on val (or test) per split x country x crop subset, member by member.

    python scripts/LSTM/08_screen_report.py --configs lstm_patch_s2 patch_weather ... --members 2

Reads preds_<role>.parquet of <work_dir>/lstm/runs/<config>/<split>/ and uses the first N members' predictions
(pred_m0 ... pred_m<N-1>) so a 5-member reference run is comparable to 2-member screening runs. Per subset
(>= 5 fields) and member: field R2, correlation of field means, offset share = mean(err)^2 / mean(err^2) over
fields (how much of the field error is a constant level shift), within-field correlation of pixel deviations
from the field mean. Writes <work_dir>/lstm/screen/<name>.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_paths  # noqa: E402
from src.metrics import field_level, r2  # noqa: E402

DEFAULT_SPLITS = ["germany/loyo/all", "germany/loro/all", "south_america/loyo/all", "south_america/loro/all"]
MIN_FIELDS = 5


def subset_stats(g: pd.DataFrame, col: str) -> dict:
    fl = field_level(g.assign(pred=g[col]))
    err = fl["pred"].to_numpy() - fl["target"].to_numpy()
    dev_t = g["target"] - g.groupby("field_year")["target"].transform("mean")
    dev_p = g[col] - g.groupby("field_year")[col].transform("mean")
    return {"field_r2": r2(fl["target"], fl["pred"]),
            "corr": float(np.corrcoef(fl["target"], fl["pred"])[0, 1]),
            "offset_share": float(err.mean() ** 2 / np.mean(err ** 2)),
            "within_corr": float(np.corrcoef(dev_t, dev_p)[0, 1]) if dev_t.std() > 0 and dev_p.std() > 0
            else float("nan")}


def collect(run_root: Path, splits: list[str], role: str, n_members: int) -> pd.DataFrame:
    rows = []
    for split in splits:
        f = run_root / split.replace("/", "__") / f"preds_{role}.parquet"
        if not f.exists():
            print(f"missing {f}", file=sys.stderr)
            continue
        p = pd.read_parquet(f)
        cols = [c for c in (f"pred_m{m}" for m in range(n_members)) if c in p]
        for (country, crop), g in p.groupby(["country", "crop"], observed=True):
            if g["field_year"].nunique() < MIN_FIELDS:
                continue
            for m, col in enumerate(cols):
                rows.append({"split": split, "subset": f"{country}|{crop}", "member": m, **subset_stats(g, col),
                             "n_fields": g["field_year"].nunique()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="run names")
    ap.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    ap.add_argument("--role", choices=["val", "test"], default="val")
    ap.add_argument("--members", type=int, default=2, help="use the first N members")
    ap.add_argument("--reference", default="lstm_patch_s2")
    ap.add_argument("--name", default=None, help="output file name (default: first config)")
    args = ap.parse_args()

    work = load_paths()["work_dir"] / "lstm"
    res = {c: collect(work / "runs" / c, args.splits, args.role, args.members) for c in args.configs}
    if args.reference not in res:
        res[args.reference] = collect(work / "runs" / args.reference, args.splits, args.role, args.members)

    def per_subset(df):  # mean and spread over members
        g = df.groupby(["split", "subset"])
        return g["field_r2"].mean(), g["field_r2"].std(ddof=0).fillna(0.0), g[["corr", "offset_share", "within_corr"]].mean()

    ref_r2 = per_subset(res[args.reference])[0]
    lines = [f"# Screening on {args.role}, first {args.members} members, reference `{args.reference}`", ""]
    lines += ["## Summary", "", "| config | subsets | median field R2 | median member spread | above reference "
              "| median corr | median offset share | median within-field corr |", "|---|---|---|---|---|---|---|---|"]
    for c, df in res.items():
        if df.empty:
            continue
        r, sp, other = per_subset(df)
        common = r.index.intersection(ref_r2.index)
        above = f"{int((r[common] > ref_r2[common]).sum())}/{len(common)}"
        lines.append(f"| {c} | {len(r)} | {r.median():.3f} | {sp.median():.3f} | {above} | "
                     f"{other['corr'].median():.2f} | {other['offset_share'].median():.2f} | "
                     f"{other['within_corr'].median():.2f} |")
    lines += ["", "## Field R2 per subset (mean ± spread over members)", ""]
    names = [c for c in res if not res[c].empty]
    lines += ["| split | subset | fields | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    stats = {c: per_subset(res[c]) for c in names}
    nf = res[names[0]].groupby(["split", "subset"])["n_fields"].first()
    for key in nf.index:
        cells = [f"{stats[c][0][key]:.2f} ± {stats[c][1][key]:.2f}" if key in stats[c][0].index else "-"
                 for c in names]
        lines.append(f"| {key[0]} | {key[1].replace('|', ' / ')} | {nf[key]} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    out = work / "screen"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name or args.configs[0]}_{args.role}.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
