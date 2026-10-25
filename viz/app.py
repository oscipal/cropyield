"""YieldSAT / SwissYield visualizer: browse inputs, runs, predictions and splits.

    bash viz/run.sh        # then on your laptop: ssh -L 8501:localhost:8501 pf-pc37.ethz.ch, open localhost:8501
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crop yield viewer", page_icon="🌾", layout="wide")

import data  # noqa: E402

st.title("Crop yield viewer")
st.caption(f"Reads derived files in `{data.WORK}` only. Local use only: the data holds exact field locations.")

fields, runs = data.fields_table(), data.list_runs()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Field-years", f"{len(fields):,}")
c2.metric("Countries", fields["country"].nunique())
c3.metric("Run configs", runs["config"].nunique())
c4.metric("Finished split runs", f"{int(runs['done'].sum())} / {len(runs)}")

st.markdown("""
| Page | What it shows |
|---|---|
| **Fields** | Map of all field-years; per field the yield map, Sentinel-2 RGB/NDVI per time step, weather, soil and terrain |
| **Runs** | Test/val metrics of every config × split, per-country×crop breakdown, training curves |
| **Predictions** | Target vs prediction vs error maps per field, ensemble spread, field-level scatter |
| **Splits** | Map of a split's train/val/test fields and their counts |
| **Configure run** | Start from an existing model, choose inputs, time axis, training pixels, model size and splits; check on a few pixels and save a config to `configs/custom/` |
""")

# ---------------------------------------------------------------- results
res = data.results_summary()
st.header("Results")
if res is None:
    st.warning("scripts/LSTM/10_summary.py did not load; see the Runs page.")
else:
    done = runs[runs["done"]]
    newest = max((Path(p) / "metrics.json").stat().st_mtime for p in done["path"]) if len(done) else None
    st.caption("Test field R² (mean prediction vs. mean yield per field-year; ensembles: mean of the members), "
               "computed live from the runs' `metrics.json` with the code of `scripts/LSTM/10_summary.py`, so it "
               "matches `docs/results_summary.md` after that script is rerun."
               + (f" Newest finished run: {pd.Timestamp(newest, unit='s', tz='UTC').tz_convert('Europe/Zurich'):%d.%m.%Y %H:%M}."
                  if newest else ""))
    models, methods = res["models"], res["methods"]

    # The overview of 10_summary.py is a median over the crop-specific splits, which are not shown (user decision
    # 2026-10-25): only the pooled all-crop splits below.
    st.subheader("Per split")
    sp = res["splits"]
    sp = sp[sp["crop"].eq("all") & sp["scope"].isin(data.SHOWN_SCOPES)].assign(
        region=lambda x: x["scope"].map(data.SCOPE_LABELS), crop="all crops")
    for tab, (k, title) in zip(st.tabs(list(methods.values())), methods.items()):
        d = sp[sp["method"] == k]
        piv = d.pivot_table(index=["region", "crop"], columns="model", values="field_r2", aggfunc="first")
        piv = piv[[m for m in models if m in piv]].rename(columns=models)
        order = {r: i for i, r in enumerate(data.SCOPE_LABELS.values())}
        piv = piv.iloc[sorted(range(len(piv)), key=lambda i: (order[piv.index[i][0]],
                                                               piv.index[i][1] != "all crops", piv.index[i][1]))]
        piv["test fields"] = d.drop_duplicates(["region", "crop"]).set_index(["region", "crop"])["n_fields"]
        cols = [c for c in piv.columns if c != "test fields"]
        tab.dataframe(piv.style.format("{:.2f}", subset=cols, na_rep="–")
                      .background_gradient(subset=cols, cmap="RdYlGn", vmin=-0.3, vmax=0.9, axis=None)
                      .highlight_max(subset=cols, axis=1, props="font-weight: bold; text-decoration: underline"),
                      width="stretch")
    st.caption("Whole test set of each pooled all-crop split (crop-specific splits are not shown); the best model "
               "per row is underlined. R² on multi-crop or multi-country splits is inflated by the yield differences "
               "between crops and countries. Europe (old) = Germany only, the Europe of the split suite before "
               "Switzerland was added; models run before that have only those rows.")

    st.subheader("10-day Sentinel-2 series and ConvLSTM (pooled all-crop splits)")
    dense = [m for m in res["dense"] if m in models]
    sub = res["subsets"]
    d = sub[sub["model"].isin(dense) & sub["crop"].eq("all") & sub["scope"].isin(res["pooled"])]
    if len(d):
        med = d.groupby(["scope", "method", "model"])["field_r2"].median().unstack("model")
        med = med[[m for m in dense if m in med]].rename(columns=models)
        med.index = [f"{s}/{k}/all" for s, k in med.index]
        piv = d.pivot_table(index=["scope", "method", "subset"], columns="model", values="field_r2")
        med.loc["all subsets"] = piv.median().rename(models)
        st.dataframe(med.style.format("{:.2f}", na_rep="–")
                     .background_gradient(cmap="RdYlGn", vmin=-0.3, vmax=0.9, axis=None)
                     .highlight_max(axis=1, props="font-weight: bold; text-decoration: underline"), width="stretch")
        st.caption(f"Median test field R² over the country × crop subsets of each pooled split "
                   f"({piv.index.nunique()} subsets in all). Patch LSTM v3 → (10-day) is the gain from the denser "
                   f"series, (10-day) → ConvLSTM the gain from the model.")

with st.expander("Inventory: field-years per country × crop, runs per config"):
    st.dataframe(fields.pivot_table(index="country", columns="crop", values="field", aggfunc="count", fill_value=0),
                 width="stretch")
    st.dataframe(runs.groupby("config").agg(splits=("split", "size"), finished=("done", "sum")), width="stretch")
