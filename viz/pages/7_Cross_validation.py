"""Cross-validation over years: pooled ensemble field scores of the configs trained on the folds world/cv<year>/all
(src/cv.py; doc: scripts/LSTM/22_cv_report.py)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import data
import plots
from src import cv
from src.splitting import CV_MIN_FIELDS, CV_YEARS

st.set_page_config(page_title="Cross-validation", page_icon="🔁", layout="wide")
st.title("Cross-validation over years")
YEARS = CV_YEARS["world"]
st.caption(f"World model, all crops. One fold per test year ({', '.join(YEARS)}); each fold's val year (early "
           "stopping) is the next of these years, cyclic. 2016, 2017, 2024 and later years are always train. Fold "
           "2021 = the World LOYO split, so an existing World LOYO run counts as that fold. Ensemble field scores; "
           f"only country × crop × year cells with ≥ {CV_MIN_FIELDS} fields count. **Group** = country × crop with "
           "all its CV years pooled (includes the between-year variation); **cell** = one group in one year.")

cfgs = data.cv_configs()
if cfgs.empty:
    st.info("No cross-validation run yet (no world__cv<year>__all folder).")
    st.stop()

c1, c2, c3 = st.columns([3, 1, 2])
shown = c1.multiselect("Setups", list(cfgs["config"]), default=list(cfgs["config"]), key="cv_cfgs")
role = c2.segmented_control("Set", ["test", "val"], default="test", key="cv_role") or "test"
if not shown:
    st.stop()
ref = c3.selectbox("Compare with", shown, key="cv_ref")

prog = cfgs[cfgs["config"].isin(shown)].assign(
    **{"folds done": lambda d: d["done"].map(lambda x: f"{len(x)} / {len(YEARS)}: {', '.join(x) or '–'}"),
       "folds started": lambda d: d["started"].map(", ".join)})[["config", "folds done", "folds started"]]
st.dataframe(prog, hide_index=True, width="content")

F = {c: data.cv_field_preds(c, role) for c in shown}
G, C = {}, {}
for c in shown:
    G[c], C[c] = cv.scores(F[c])

# ---------------------------------------------------------------- overview
rows = []
for c in shown:
    s = cv.summary(F[c])
    row = {"setup": c, "folds": s["folds"], "median group R²": s["group_r2"], "median cell R²": s["cell_r2"],
           "median group rRMSE %": s["group_rrmse"], "pooled rRMSE %": s["pooled_rrmse"]}
    for tab, keys, met, name in ((G, ["country", "crop"], "r2", "group R² vs ref"),
                                 (C, ["country", "crop", "year"], "r2", "cell R² vs ref"),
                                 (C, ["country", "crop", "year"], "rrmse", "cell rRMSE vs ref")):
        if c == ref:
            row[name] = "–"
        else:
            d, nb, n = cv.paired(tab[c], tab[ref], met, keys)
            row[name] = "–" if not n else f"{d:+.{2 if met == 'r2' else 1}f} ({nb} / {n})"
    rows.append(row)
st.subheader(f"Overview ({role})")
st.dataframe(pd.DataFrame(rows).style.format({"median group R²": "{:+.2f}", "median cell R²": "{:+.2f}",
                                               "median group rRMSE %": "{:.1f}", "pooled rRMSE %": "{:.1f}"},
                                              na_rep="–"), hide_index=True, width="stretch")
st.caption(f"\"vs ref\": median over the groups (cells) both setups have of the difference to {ref}, with the number "
           "where the setup is better (higher R², lower rRMSE). Compare setups only on the same folds: a setup with "
           "fewer finished folds is scored on fewer years.")

# ---------------------------------------------------------------- per cell
st.subheader("Per year")
MET = {"r2": "R²", "rrmse": "rRMSE (%)", "bias": "bias (%): mean prediction / mean observed − 1"}
met = st.segmented_control("Score", list(MET), default="r2", format_func=lambda m: MET[m].split(":")[0],
                           key="cv_met") or "r2"
cells = pd.concat([C[c].assign(setup=c) for c in shown if len(C[c])], ignore_index=True) if any(
    len(C[c]) for c in shown) else pd.DataFrame()
if cells.empty:
    st.info("No scored cells yet.")
else:
    cells["group"] = cells["country"] + " " + cells["crop"]
    y = cells[met].clip(lower=-2) if met == "r2" else cells[met]
    fig = px.scatter(cells.assign(value=y), x="year", y="value", color="setup", facet_col="group", facet_col_wrap=4,
                     hover_data={"n_fields": True, met: ":.2f", "value": False},
                     color_discrete_sequence=plots.OKABE_ITO, category_orders={"year": YEARS, "setup": shown})
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(marker=dict(size=9, opacity=0.85))
    fig.update_yaxes(title_text="", matches=None if met == "rrmse" else "y")
    if met == "bias":
        fig.add_hline(y=0, line=dict(color="#888", dash="dot"))
    fig.update_layout(height=260 * ((cells["group"].nunique() + 3) // 4), margin=dict(l=0, r=0, t=30, b=0),
                      legend=dict(orientation="h", y=-0.08))
    st.plotly_chart(fig, width="stretch", key="cv_cells")
    st.caption(f"{MET[met]} per country × crop and year ({role} predictions of the fold that held that year out). "
               + ("R² below −2 is drawn at −2. " if met == "r2" else "")
               + "Hover for the number of fields.")

# ---------------------------------------------------------------- per group
st.subheader("Per group (all CV years pooled)")
g = pd.concat([G[c].assign(setup=c) for c in shown if len(G[c])], ignore_index=True) if any(
    len(G[c]) for c in shown) else pd.DataFrame()
if not g.empty:
    g["group"] = g["country"] + " " + g["crop"]
    t = g.pivot_table(index="group", columns="setup", values=["r2", "rrmse"])
    t.columns = [f"{s} {'R²' if m == 'r2' else 'rRMSE %'}" for m, s in t.columns]
    n = g[g["setup"] == ref].set_index("group")[["n_years", "n_fields"]].rename(
        columns={"n_years": "years", "n_fields": "fields"})
    t = n.join(t, how="right")
    st.dataframe(t.reset_index().style.format({c: ("{:+.2f}" if c.endswith("R²") else "{:.1f}") for c in t.columns},
                                              na_rep="–"), hide_index=True, width="stretch")
    st.caption(f"Years = CV years with ≥ {CV_MIN_FIELDS} fields of the group, fields = their total (in {ref}). A group's R² is computed over "
               "all its years together, so it also rewards getting the level of each year right.")
