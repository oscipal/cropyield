"""Short results summary of all models: docs/results_summary.md (the full tables are in docs/results_lstm.md).

    python scripts/LSTM/10_summary.py

Test field R2 of every model (ensemble mean for the ensembles), per splitting method (LOYO, LORO):
1. per split (scope / crop); the world single-crop splits for corn, rapeseed and soybean are left out, they
   hold the same fields as the south_america / germany ones;
2. per country x crop, from the single-crop splits (germany/<method>/<crop>, south_america/<method>/<crop>),
   which are closest to the paper's per country x crop models, next to the paper's LSTM (S2) value.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("report", HERE / "03_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
from src.config import REPO_ROOT, load_paths  # noqa: E402  (path set up by 03_report)

MODELS = {  # run name -> column label
    "mean": "Mean",
    "mean_crop_country": "Mean crop×country",
    "lstm_s2": "LSTM",
    "lstm_s2_adm": "LSTM + ADM",
    "lstm_v2_s2": "LSTM v2",
    "lstm_patch_s2": "Patch LSTM",
    "lstm_patch_v3_s2": "Patch LSTM v3",
}
DESCRIPTIONS = [
    ("Mean", "`mean`", "Baseline: every test pixel gets the mean yield of the training pixels."),
    ("Mean crop×country", "`mean_crop_country`",
     "Baseline: the training mean yield of the pixel's crop × country."),
    ("LSTM", "`lstm_s2`", "The paper's pixel LSTM: 12 Sentinel-2 bands × 24 time steps, 2 LSTM layers (128) "
     "+ FC head, one model, early stopping on val MSE."),
    ("LSTM + ADM", "`lstm_s2_adm`", "Same LSTM with the paper's ADM input fusion: + 4 weather bands per time "
     "step, + soil (48) and topography (5) repeated at every time step."),
    ("LSTM v2", "`lstm_v2_s2`", "LSTM with NDVI/NDRE/NDMI, an observed-time-step channel, one-hot crop and "
     "country, yield z-scored per crop × country, max. 500 pixels per field and epoch, dropout + weight decay, "
     "and a 5-model ensemble."),
    ("Patch LSTM", "`lstm_patch_s2`", "LSTM v2, but each pixel is seen with its 5 × 5 neighbourhood: a small "
     "CNN per time step, then the LSTM. 5-model ensemble."),
    ("Patch LSTM v3", "`lstm_patch_v3_s2`", "Patch LSTM + training augmentation against season shifts: the "
     "time series randomly shifted by ±1 step, observed steps dropped with p = 0.1. 5-model ensemble."),
]
DROPPED = ("Tried and not kept: a field-image model (temporal encoder + U-Net over the whole field; not better "
           "than the patch LSTM on 4 splits), and adding weather, soil + topography or spectral noise to the "
           "patch LSTM (no gain on val in a 2-member screening). Time augmentation was the only change that "
           "helped on val (median subset field R² 0.33 vs 0.08), so it became Patch LSTM v3.")
METHODS = {"loyo": "LOYO (leave one year out)", "loro": "LORO (leave one region out)"}
SCOPES = {"germany": "Germany", "south_america": "South America", "world": "World"}
DUPLICATES = {("world", "corn"), ("world", "rapeseed"), ("world", "soybean")}


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = load_paths()["work_dir"] / "lstm" / "runs"
    splits, subsets = [], []
    for name in MODELS:
        for r in report.load_runs(runs / name):
            scope, method, crop = r["split"].split("/")
            base = {"model": name, "scope": scope, "method": method, "crop": crop}
            test = r["metrics"]["test"]
            splits.append({**base, "field_r2": test["all"]["field_r2"], "n_fields": test["all"]["n_fields"]})
            for g, m in test.items():
                if g != "all":
                    country, gcrop = g.split("|")
                    subsets.append({**base, "subset": f"{report.ABBR[country]}-{report.ABBR[gcrop]}",
                                    "country": country, "subset_crop": gcrop, **m})
    return pd.DataFrame(splits), pd.DataFrame(subsets)


def table(df: pd.DataFrame, index: list[str], extra: dict | None = None, order: list | None = None) -> list[str]:
    """Markdown table, one row per index (in `order` if given), a column per model; best model per row in bold."""
    piv = df.pivot_table(index=index, columns="model", values="field_r2", aggfunc="first")
    if order is not None:
        piv = piv.reindex(order)
    piv = piv[[m for m in MODELS if m in piv]]
    head = list(index) + [MODELS[m] for m in piv.columns] + list(extra or {})
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(index) + "---:|" * (len(head) - len(index))]
    for key, row in piv.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        best = row.max()
        cells = [f"**{v:.2f}**" if v == best else report.fmt(v) for v in row]
        out.append("| " + " | ".join([*map(str, key), *cells, *[f(key) for f in (extra or {}).values()]]) + " |")
    return out


def main() -> None:
    paper = yaml.safe_load((HERE / "paper_lstm_results.yaml").read_text())["s2"]
    sp, sub = collect()
    sp = sp[[(s, c) not in DUPLICATES for s, c in zip(sp["scope"], sp["crop"])]]
    single = sub[sub["scope"].ne("world") & sub["crop"].ne("all")]  # single-crop splits only

    L = ["# Results summary", "",
         "Generated by `scripts/LSTM/10_summary.py`. All numbers are **test field R²** (mean prediction vs. mean "
         "yield per field-year); for ensembles the prediction is the mean of the 5 members. The best model per "
         "row is in bold. Full tables (RMSE, pixel level, single members, training) are in "
         "`docs/results_lstm.md`; the splits are described in `docs/splits.md`.", "",
         "## Models", "", "| model | run | what it does |", "|---|---|---|"]
    L += [f"| {a} | {b} | {c} |" for a, b, c in DESCRIPTIONS]
    L += ["", DROPPED, ""]

    L += ["## Overview", "",
          "Median test field R² over the country × crop subsets of the single-crop splits (9 per method), next to "
          "the paper's LSTM (S2) on the same subsets. The paper averages many held-out years/regions per subset, "
          "we evaluate one held-out set, so the paper column is a reference, not the same experiment.", "",
          "| model | " + " | ".join(k.upper() for k in METHODS) + " | both | best in subsets |", "|---|---:|---:|---:|---:|"]
    best = single.loc[single.groupby(["method", "scope", "crop", "subset"])["field_r2"].idxmax(), "model"]
    for m, label in MODELS.items():
        d = single[single["model"] == m]
        med = [d.loc[d["method"] == k, "field_r2"].median() for k in METHODS]
        L.append(f"| {label} | " + " | ".join(report.fmt(v) for v in med)
                 + f" | {report.fmt(d['field_r2'].median())} | {int((best == m).sum())} / {len(best)} |")
    pk = single.drop_duplicates(["method", "subset"])
    pv = [paper[k]["field"][s]["r2"][0] for k, s in zip(pk["method"], pk["subset"])]
    pk = pk.assign(paper=pv)
    L.append("| *paper LSTM (S2)* | " + " | ".join(report.fmt(pk.loc[pk["method"] == k, "paper"].median())
                                                   for k in METHODS)
             + f" | {report.fmt(pk['paper'].median())} | – |")
    L.append("")

    for k, title in METHODS.items():
        L += [f"## {title}", "", "### Per split", "",
              "Whole test set of each split. R² on multi-crop or multi-country splits is inflated by the yield "
              "differences between crops and countries; the per country × crop table below is the fairer "
              "comparison.", ""]
        d = sp[sp["method"] == k].assign(
            region=lambda x: x["scope"].map(SCOPES),
            crop=lambda x: x["crop"].replace({"all": "all crops"}))
        order = {s: i for i, s in enumerate(SCOPES.values())}
        d = d.assign(_o=d["region"].map(order) * 10 + d["crop"].eq("all crops").map({True: 0, False: 1}))
        d = d.sort_values(["_o", "crop"])
        rows = d.drop_duplicates(["region", "crop"])[["region", "crop"]].apply(tuple, axis=1).tolist()
        n = d.drop_duplicates(["region", "crop"]).set_index(["region", "crop"])["n_fields"]
        L += table(d, ["region", "crop"], {"test fields": lambda key, n=n: str(int(n[key]))}, rows) + [""]

        L += ["### Per country × crop", "",
              "From the single-crop splits (Germany / South America, one model per crop), with the paper's LSTM "
              "(S2) as mean ± std over its folds.", ""]
        d = single[single["method"] == k]
        n = d.drop_duplicates("subset").set_index("subset")["n_fields"]
        L += table(d, ["subset"], {
            "test fields": lambda key, n=n: str(int(n[key[0]])),
            "paper LSTM": lambda key, k=k: report.paper_cell({"s2": paper}, "s2", k, "field", key[0])[0]})
        L.append("")

    L += ["## Notes", "",
          "- Patch LSTM v3 has the best median subset R² overall (0.31), mainly from LORO; under LOYO it is level "
          "with LSTM v2 and the paper. The gain over Patch LSTM is not uniform, and many test subsets have few "
          "fields (see `test fields`), so differences below ~0.05 are noise.",
          "- Negative R² means worse than predicting the test set's own mean yield. Often it is a level problem: "
          "the held-out year or region is predicted too high or too low overall while the fields are still ranked "
          "sensibly (e.g. Brazilian wheat under LOYO). German wheat in its single-crop LOYO split is a real "
          "failure (fields not ranked either).",
          "- On whole multi-crop splits the crop × country mean is already a strong baseline (Germany: best "
          "model), because most of the variance is between crops; this is why the per country × crop tables "
          "are the ones to compare.",
          "- Adding weather, soil and topography (LSTM + ADM, and the screened Patch LSTM variant) did not help.",
          ""]
    out = REPO_ROOT / "docs" / "results_summary.md"
    out.write_text("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
