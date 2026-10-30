"""Cross-validation over years: pooled test (and val) scores of configs trained on the folds <scope>/cv<year>/all.

    python scripts/LSTM/22_cv_report.py --configs cv_1d_aug cv_base_aug_fz --ref cv_1d_aug --out docs/results_cv_<topic>.md

Folds, scores and the 10-field minimum: src/cv.py, src/splitting.py (CV_YEARS, CV_MIN_FIELDS). Ensemble predictions
only. Switzerland is left out of the doc and the console (CLAUDE.md); the visualizer shows it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_paths  # noqa: E402
from src.cv import field_preds, fold_dirs, paired, scores, summary  # noqa: E402
from src.splitting import CV_MIN_FIELDS, CV_YEARS  # noqa: E402

DROP = ("Switzerland",)


def fmt(x, spec=".2f") -> str:
    return "–" if pd.isna(x) else format(x, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--ref", default=None, help="config the others are compared with (default: the first)")
    ap.add_argument("--scope", default="world")
    ap.add_argument("--title", default="Cross-validation over years")
    ap.add_argument("--note", default="", help="extra header text: what is new")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ref = args.ref or args.configs[0]
    runs = load_paths()["work_dir"] / "lstm" / "runs"
    years = CV_YEARS[args.scope]

    F, G, C, S = {}, {}, {}, {}
    for c in dict.fromkeys(args.configs + [ref]):
        dirs = fold_dirs(runs / c, args.scope)
        for role in ("test", "val"):
            F[c, role] = field_preds(dirs, role, DROP)
            G[c, role], C[c, role] = scores(F[c, role])
            S[c, role] = summary(F[c, role])
        print(f"{c}: folds {', '.join(dirs) or 'none'}", flush=True)

    L = [f"# {args.title}", "",
         f"Folds: `{args.scope}/cv<year>/all` for the test years {', '.join(years)} (each also val once: the next "
         f"year, cyclic); 2016, 2017, 2024 and later years are always train. Fold 2021 = `{args.scope}/loyo/all`.",
         f"Ensemble field scores; only country × crop × year cells with ≥ {CV_MIN_FIELDS} fields count. "
         "Group = country × crop with all its CV years pooled (includes the between-year variation); cell = one "
         "group in one year. rRMSE = RMSE in % of the mean observed field yield. Switzerland is left out of this "
         "doc (shown in the visualizer).",
         "Caveats: 6 test years; cells have 10–236 fields; the val scores come from the year used for early "
         "stopping of the same fold, so they are slightly optimistic."]
    if args.note:
        L += ["", args.note]
    for role in ("test", "val"):
        L += ["", f"## {role}: overview", "",
              f"Paired columns: median over the common groups (cells) of the difference to `{ref}`, with the number "
              "better.", "",
              "| config | folds | median group R² | median cell R² | median group rRMSE % | pooled rRMSE % "
              "| group R² vs ref | cell R² vs ref | cell rRMSE vs ref |", "|---|---|---|---|---|---|---|---|---|"]
        for c in args.configs:
            s = S[c, role]
            cmp = ["–"] * 3
            if c != ref:
                cmp = []
                for tab, keys, met, sp in ((G, ["country", "crop"], "r2", "+.2f"),
                                           (C, ["country", "crop", "year"], "r2", "+.2f"),
                                           (C, ["country", "crop", "year"], "rrmse", "+.1f")):
                    d, nb, n = paired(tab[c, role], tab[ref, role], met, keys)
                    cmp.append(f"{fmt(d, sp)} ({nb} / {n})")
            L.append(f"| `{c}` | {s['folds']} / {len(years)} | {fmt(s['group_r2'], '+.2f')} | "
                     f"{fmt(s['cell_r2'], '+.2f')} | {fmt(s['group_rrmse'], '.1f')} | {fmt(s['pooled_rrmse'], '.1f')} "
                     f"| " + " | ".join(cmp) + " |")
        L += ["", f"### {role}: per group (all CV years pooled), R² / rRMSE %", "",
              "| group | years | fields | " + " | ".join(f"`{c}`" for c in args.configs) + " |",
              "|---|---|---|" + "---|" * len(args.configs)]
        keys = G[ref, role][["country", "crop", "n_years", "n_fields"]]
        for k in keys.itertuples(index=False):
            cells = []
            for c in args.configs:
                g = G[c, role]
                r = g[(g["country"] == k.country) & (g["crop"] == k.crop)]
                cells.append("–" if r.empty else f"{fmt(r['r2'].iloc[0], '+.2f')} / {fmt(r['rrmse'].iloc[0], '.1f')}")
            L.append(f"| {k.country} {k.crop} | {k.n_years} | {k.n_fields} | " + " | ".join(cells) + " |")
        for met, sp, label in (("r2", "+.2f", "R²"), ("bias", "+.1f", "bias % (mean prediction / mean observed − 1)")):
            L += ["", f"### {role}: per cell, {label}", ""]
            for c in args.configs:
                t = C[c, role].pivot_table(index=["country", "crop"], columns="year", values=met)
                if t.empty:
                    continue
                L += [f"`{c}`:", "", "| group | " + " | ".join(t.columns) + " |", "|---|" + "---|" * len(t.columns)]
                L += [f"| {i[0]} {i[1]} | " + " | ".join(fmt(v, sp) for v in row) + " |" for i, row in t.iterrows()]
                L.append("")
    out = Path(args.out)
    out.write_text("\n".join(L).rstrip() + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
