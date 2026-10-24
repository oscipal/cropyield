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
    "lstm_patch_v3_w10_s2": "Patch LSTM v3 (10-day)",
    "convlstm_s2": "ConvLSTM (10-day)",
}
DENSE = ("lstm_patch_v3_s2", "lstm_patch_v3_w10_s2", "convlstm_s2")  # the ConvLSTM / 10-day S2 comparison
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
    ("Patch LSTM v3 (10-day)", "`lstm_patch_v3_w10_s2`", "Patch LSTM v3 on a denser Sentinel-2 time series "
     "rebuilt from the raw scenes: 51 steps of 10 days (best scene per window) instead of 24 of 30 days, time "
     "shift ±3 steps (= 30 days). Pooled all-crop splits only. 5-model ensemble."),
    ("ConvLSTM (10-day)", "`convlstm_s2`", "Convolutional LSTM on the 5 × 5 neighbourhood: the LSTM gates are "
     "3 × 3 convolutions, so space and time are mixed in one layer (1 layer, 32 channels); on the 10-day "
     "Sentinel-2 series, otherwise inputs, training and augmentation as Patch LSTM v3. Pooled all-crop splits "
     "only. 5-model ensemble."),
]
POOLED = ("europe", "south_america", "world")  # scopes of the pooled all-crop splits (ConvLSTM comparison)
DROPPED = ("Tried and not kept: a field-image model (temporal encoder + U-Net over the whole field; not better "
           "than the patch LSTM on 4 splits), and adding weather, soil + topography or spectral noise to the "
           "patch LSTM (no gain on val in a 2-member screening). Time augmentation was the only change that "
           "helped on val (median subset field R² 0.33 vs 0.08), so it became Patch LSTM v3. ConvLSTM variants "
           "(9 × 9 patch, 64 channels, 2 layers) and 15/30-day time series were screened on val and not kept "
           "(`docs/convlstm_screening.md`).")
METHODS = {"loyo": "LOYO (leave one year out)", "loro": "LORO (leave one region out)"}
SCOPES = {"germany": "Germany", "europe": "Europe", "south_america": "South America", "world": "World"}
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


def dense_section(sp: pd.DataFrame, sub: pd.DataFrame) -> list[str]:
    """Patch LSTM v3 on the 30-day cube vs. on the 10-day cube vs. ConvLSTM on the 10-day cube, pooled splits."""
    p30, p10, conv = DENSE
    L = ["## ConvLSTM and the 10-day Sentinel-2 series", "",
         "The six pooled all-crop splits. Patch LSTM v3 on the 10-day series separates the gain from the denser "
         "time series (Patch LSTM v3 → Patch LSTM v3 (10-day)) from the gain from the model (Patch LSTM v3 "
         "(10-day) → ConvLSTM (10-day)). Columns: test field R² of the whole split, then the median over its "
         "country × crop subsets.", "",
         "| split | " + " | ".join(MODELS[m] for m in DENSE) + " | " + " | ".join(f"{MODELS[m]}, median subset"
                                                                      for m in DENSE) + " | subsets |",
         "|---|" + "---:|" * (2 * len(DENSE) + 1)]
    s = sp[sp["model"].isin(DENSE) & sp["crop"].eq("all") & sp["scope"].isin(POOLED)]
    d = sub[sub["model"].isin(DENSE) & sub["crop"].eq("all") & sub["scope"].isin(POOLED)]
    for scope in POOLED:
        for k in METHODS:
            whole = s[(s["scope"] == scope) & (s["method"] == k)].set_index("model")["field_r2"]
            g = d[(d["scope"] == scope) & (d["method"] == k)]
            med = g.groupby("model")["field_r2"].median()
            L.append(f"| {scope}/{k}/all | " + " | ".join(report.fmt(whole.get(m)) for m in DENSE) + " | "
                     + " | ".join(report.fmt(med.get(m)) for m in DENSE) + f" | {g['subset'].nunique()} |")
    piv = d.pivot_table(index=["scope", "method", "subset"], columns="model", values="field_r2")
    L += ["", f"Over all {len(piv)} subsets: median field R² "
          + ", ".join(f"{MODELS[m]} {report.fmt(piv[m].median())}" for m in DENSE) + "."]
    for a, b in ((p10, p30), (conv, p10), (conv, p30)):
        diff = piv[a] - piv[b]
        L.append(f"- {MODELS[a]} vs. {MODELS[b]}: higher in {int((diff > 0).sum())} / {len(diff)} subsets, "
                 f"median difference {diff.median():+.2f}.")
    return L + [""]


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Per split results (without the duplicated world single-crop splits), per subset results, the subsets of the
    single-crop splits, and the paper's LSTM (S2) values."""
    paper = yaml.safe_load((HERE / "paper_lstm_results.yaml").read_text())["s2"]
    sp, sub = collect()
    sp = sp[[(s, c) not in DUPLICATES for s, c in zip(sp["scope"], sp["crop"])]]
    single = sub[sub["scope"].ne("world") & sub["crop"].ne("all")]  # single-crop splits only
    return sp, sub, single, paper


def overview(single: pd.DataFrame, paper: dict) -> pd.DataFrame:
    """Median test field R2 over the single-crop subsets per model and method, how often it is the best model, and
    the paper's LSTM (S2) on the same subsets (best = NaN)."""
    best = single.loc[single.groupby(["method", "scope", "crop", "subset"])["field_r2"].idxmax(), "model"]
    rows = []
    for m, label in MODELS.items():
        d = single[single["model"] == m]
        if d.empty:  # trained on the pooled splits only
            continue
        rows.append({"model": label, **{k: d.loc[d["method"] == k, "field_r2"].median() for k in METHODS},
                     "both": d["field_r2"].median(), "best": int((best == m).sum()), "subsets": len(best)})
    pk = single.drop_duplicates(["method", "subset"])
    pk = pk.assign(paper=[paper.get(k, {}).get("field", {}).get(s, {"r2": [float("nan")]})["r2"][0]
                          for k, s in zip(pk["method"], pk["subset"])])
    rows.append({"model": "paper LSTM (S2)", **{k: pk.loc[pk["method"] == k, "paper"].median() for k in METHODS},
                 "both": pk["paper"].median(), "best": float("nan"), "subsets": len(best)})
    return pd.DataFrame(rows)


def main() -> None:
    sp, sub, single, paper = load()

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
          "the paper's LSTM (S2) on the same subsets (models trained on the pooled splits only are compared in "
          "\"ConvLSTM and the 10-day Sentinel-2 series\"). The paper averages many held-out years/regions per subset, "
          "we evaluate one held-out set, so the paper column is a reference, not the same experiment.", "",
          "| model | " + " | ".join(k.upper() for k in METHODS) + " | both | best in subsets |", "|---|---:|---:|---:|---:|"]
    for _, r in overview(single, paper).iterrows():
        paper_row = r["model"].startswith("paper")
        L.append((f"| *{r['model']}* | " if paper_row else f"| {r['model']} | ")
                 + " | ".join(report.fmt(r[k]) for k in [*METHODS, "both"])
                 + (" | – |" if paper_row else f" | {int(r['best'])} / {int(r['subsets'])} |"))
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

        L += ["### Per country × crop from the pooled splits", "",
              "Subsets of the all-crop splits of Europe (Germany + Switzerland), South America and World (one "
              "model per split, crop and country as inputs); the only table with ConvLSTM values. Swiss subsets "
              "have very few test fields; no paper values exist for them.", ""]
        d = sub[(sub["method"] == k) & sub["crop"].eq("all") & sub["scope"].isin(POOLED)].assign(
            region=lambda x: x["scope"].map(SCOPES))
        d = d.assign(_o=d["scope"].map({s: i for i, s in enumerate(POOLED)})).sort_values(["_o", "subset"])
        rows = d.drop_duplicates(["region", "subset"])[["region", "subset"]].apply(tuple, axis=1).tolist()
        n = d.drop_duplicates(["region", "subset"]).set_index(["region", "subset"])["n_fields"]
        L += table(d, ["region", "subset"], {"test fields": lambda key, n=n: str(int(n[key]))}, rows) + [""]

    L += dense_section(sp, sub)

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
          "- ConvLSTM: on the same 10-day series it is level with Patch LSTM v3 (higher in 16 of 36 pooled "
          "subsets, median difference 0.00); the model itself brings no gain. The 10-day series helps the patch "
          "LSTM a little (median +0.04, 20 of 36 subsets) and lifts the whole-split R² on 5 of 6 splits, but "
          "unevenly: clear gains in South America LORO (median subset 0.45 → 0.72) and Europe LOYO, losses in "
          "World LOYO (Brazilian soybean and wheat, German rapeseed). Medians over 36 subsets differ by a few "
          "hundredths, within the noise of single held-out sets.",
          "- The pooled-split tables mix scopes with very different subset sizes; the Swiss subsets have 2–10 "
          "test fields, and Swiss `winter_cereal` is barley and triticale together (per the dataset README).",
          "- The single-crop `world/*` splits and the pre-Switzerland `europe`/`world` runs are not regenerated "
          "(old suite, `runs/_old_suite/`); the ConvLSTM and 10-day models exist on the pooled splits only. Full "
          "tables: `docs/results_convlstm.md`.",
          ""]
    out = REPO_ROOT / "docs" / "results_summary.md"
    out.write_text("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
