"""Write docs/results_step1.md: own LSTM and B0 results next to the paper values, plus scatterplot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT, load_paths, load_yaml  # noqa: E402
from src.metrics import field_level, summarize  # noqa: E402

# Paper: CV10, S2 only, GER-R, mean +- std over folds (Tab. 13 and 14)
PAPER = {
    "field": {"r2": (0.62, 0.25), "rmse": (0.83, 0.26)},
    "pixel": {"r2": (0.36, 0.14), "rmse": (1.33, 0.23)},
}

DEVIATIONS = [
    "Normalisation per band with mean/std of the training pixels of each fold (inner-train part), "
    "then NaN -> 0. The tutorial fills raw values with -1 and does not normalise.",
    "Hyperparameters not fully given in the paper; chosen: LSTM 1 layer, hidden 64, last hidden state, "
    "linear head, MSE, Adam lr 1e-3, batch 1024, max 50 epochs, early stopping patience 5.",
    "Early stopping on an inner validation split (10 % of training fields, grouped by field); "
    "the test fold is never used for model selection.",
    "Splits: StratifiedGroupKFold(10, shuffle, fixed seed), groups = field_shared_name, strata = farm. "
    "Field-level metrics aggregate per field-year.",
    "Std over folds computed with ddof=0.",
    "Only pixels with non-NaN target are used.",
]


def fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "ger_r_lstm.yaml"))
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = load_paths()
    pred_root = paths["work_dir"] / cfg["data"]["subdir"] / "preds"
    docs = paths["docs_dir"]
    fig_dir = docs / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    models = {"LSTM": f"lstm_{cfg['name']}", "B0": f"b0_{cfg['name']}"}
    summaries, preds_all = {}, {}
    for label, exp in models.items():
        files = sorted((pred_root / exp).glob("fold*.parquet"))
        if not files:
            raise FileNotFoundError(f"No predictions for {exp} in {pred_root / exp}")
        preds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if preds["fold"].nunique() != cfg["splits"]["n_splits"]:
            print(f"WARNING: {exp} has {preds['fold'].nunique()} folds, expected {cfg['splits']['n_splits']}")
        preds_all[label] = preds
        summaries[label] = summarize(preds)

    # main table
    rows = []
    for level in ("field", "pixel"):
        p = PAPER[level]
        rows.append({"Ebene": level.capitalize(), "Modell": "Paper LSTM",
                     "R² (Folds)": fmt(*p["r2"]), "RMSE t/ha (Folds)": fmt(*p["rmse"]),
                     "R² gepoolt": "–", "RMSE gepoolt": "–"})
        for label, s in summaries.items():
            rows.append({"Ebene": level.capitalize(), "Modell": label,
                         "R² (Folds)": fmt(s[f"{level}_r2_mean"], s[f"{level}_r2_std"]),
                         "RMSE t/ha (Folds)": fmt(s[f"{level}_rmse_mean"], s[f"{level}_rmse_std"]),
                         "R² gepoolt": f"{s['pooled'][f'{level}_r2']:.2f}",
                         "RMSE gepoolt": f"{s['pooled'][f'{level}_rmse']:.2f}"})
    table = pd.DataFrame(rows)

    # within one paper fold-std?
    checks = []
    s = summaries["LSTM"]
    for level in ("field", "pixel"):
        for metric in ("r2", "rmse"):
            pm, ps = PAPER[level][metric]
            own = s[f"{level}_{metric}_mean"]
            checks.append({"Ebene": level, "Metrik": metric.upper(), "eigen": round(own, 3), "Paper": pm,
                           "|Δ|": round(abs(own - pm), 3), "Paper-Std": ps,
                           "innerhalb 1 Std": "ja" if abs(own - pm) <= ps else "nein"})
    checks = pd.DataFrame(checks)

    # per-fold tables
    per_fold = {label: s["per_fold"] for label, s in summaries.items()}

    # scatter: field means, coloured by year
    fields = field_level(preds_all["LSTM"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for year, g in fields.groupby("year"):
        ax.scatter(g["target"], g["pred"], s=18, alpha=0.8, label=str(year))
    lo = min(fields["target"].min(), fields["pred"].min())
    hi = max(fields["target"].max(), fields["pred"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Feldmittel beobachtet (t/ha)")
    ax.set_ylabel("Feldmittel vorhergesagt (t/ha)")
    ax.set_title("LSTM, S2, GER-R, CV10 (Testfolds)")
    ax.legend(title="Jahr")
    fig.tight_layout()
    fig_path = fig_dir / "step1_field_scatter.png"
    fig.savefig(fig_path, dpi=150)

    n_fields = summaries["LSTM"]["pooled"]["n_fields"]
    n_pixels = summaries["LSTM"]["pooled"]["n_pixels"]
    lines = [
        "# Ergebnisse Schritt 1 – LSTM, Sentinel-2, GER-R, CV10",
        "",
        f"Generiert von `scripts/04_report.py`. Testvorhersagen: {n_fields} Feld-Jahre, {n_pixels:,} Pixel.",
        "",
        "## Eigene Werte und Paperwerte",
        "",
        table.to_markdown(index=False),
        "",
        "## Liegen die eigenen LSTM-Mittelwerte innerhalb einer Fold-Std der Paperwerte?",
        "",
        checks.to_markdown(index=False),
        "",
        f"Alle vier innerhalb einer Fold-Std: **{'ja' if (checks['innerhalb 1 Std'] == 'ja').all() else 'nein'}**.",
        "",
        "## Abweichungen vom Paper",
        "",
        *[f"- {d}" for d in DEVIATIONS],
        "",
        "## Feldmittel: Vorhersage gegen Beobachtung",
        "",
        f"![scatter](figures/{fig_path.name})",
        "",
    ]
    for label, pf in per_fold.items():
        lines += [f"## Werte pro Fold – {label}", "", pf.to_markdown(index=False, floatfmt=".3f"), ""]
    out = docs / "results_step1.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False))
    print(checks.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
