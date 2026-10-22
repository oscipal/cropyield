"""YieldSAT / SwissYield visualizer: browse inputs, runs, predictions and splits.

    bash viz/run.sh        # then on your laptop: ssh -L 8501:localhost:8501 pf-pc37.ethz.ch, open localhost:8501
"""
from __future__ import annotations

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
""")

st.subheader("Field-years per country and crop")
st.dataframe(fields.pivot_table(index="country", columns="crop", values="field", aggfunc="count", fill_value=0),
             width="stretch")
st.subheader("Runs")
st.dataframe(runs.groupby("config").agg(splits=("split", "size"), finished=("done", "sum")), width="stretch")
