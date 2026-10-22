"""Per-field target / prediction / error maps of a run, ensemble spread and the field-level scatter."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import data
import plots

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")
st.title("Predictions")

runs = data.list_runs().query("done")
with st.sidebar:
    st.header("Run")
    cfg = st.selectbox("Config", sorted(runs["config"].unique()),
                       index=sorted(runs["config"].unique()).index("lstm_patch_v3_s2")
                       if "lstm_patch_v3_s2" in set(runs["config"]) else 0)
    r = runs[runs["config"] == cfg]
    split = st.selectbox("Split", sorted(r["split"]))
    run = r[r["split"] == split].iloc[0]
    role = st.radio("Set", ["test", "val"] if run["has_val"] else ["test"], horizontal=True)

with st.spinner("Loading predictions"):
    fl = data.field_level(run["path"], role)
    p = data.preds(run["path"], role)

# ---------------------------------------------------------------- field-level scatter
c1, c2 = st.columns([3, 2])
with c1:
    color = st.radio("Colour by", ["country", "crop", "year"], horizontal=True)
    fig = go.Figure()
    for i, (k, g) in enumerate(fl.groupby(color)):
        fig.add_trace(go.Scatter(x=g["target"], y=g["pred"], mode="markers", name=str(k),
                                 marker=dict(color=plots.OKABE_ITO[i % 8], size=7, opacity=0.8),
                                 customdata=g["field"], hovertemplate="%{customdata}<br>target %{x:.2f}"
                                 "<br>pred %{y:.2f}<extra></extra>"))
    lo, hi = float(np.nanmin(fl[["target", "pred"]])), float(np.nanmax(fl[["target", "pred"]]))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#999", dash="dash"),
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=f"Field mean yield, {role} (click a point to open the field)", height=440,
                      xaxis_title="target t/ha", yaxis_title="prediction t/ha", margin=dict(l=0, r=0, t=36, b=0))
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
                         key=f"scatter_{cfg}_{split}_{role}_{color}")
    pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
    if pts and pts[0].get("customdata"):
        st.session_state["pred_field"] = pts[0]["customdata"]
with c2:
    st.markdown("**Fields by absolute error of the field mean**")
    st.dataframe(fl[["field", "country", "crop", "target", "pred", "error"]].style.format(
        {"target": "{:.2f}", "pred": "{:.2f}", "error": "{:+.2f}"}), hide_index=True, height=440)

# ---------------------------------------------------------------- field maps
options = fl["field"].tolist()
if st.session_state.get("pred_field") not in options:
    st.session_state["pred_field"] = options[0]
field = st.selectbox("Field-year (sorted by absolute error)", options,
                     index=options.index(st.session_state["pred_field"]))
st.session_state["pred_field"] = field
fp = p[p["field"] == field]
info = fl.set_index("field").loc[field]
st.caption(f"{info['country']} · {info['crop']} · {info['year']} · {int(info['n_pixels']):,} pixels · "
           f"mean target {info['target']:.2f}, pred {info['pred']:.2f} t/ha")
if st.button("Open inputs of this field on the Fields page"):
    st.session_state["field"] = field
    st.switch_page("pages/1_Fields.py")

r0, c0 = int(fp["row"].min()), int(fp["col"].min())
lr, lc = fp["row"].to_numpy(int) - r0, fp["col"].to_numpy(int) - c0
h, w = int(lr.max()) + 1, int(lc.max()) + 1
img = lambda v: data.to_image(np.asarray(v, dtype=np.float32), lr, lc, h, w)
vmin, vmax = np.nanpercentile(np.r_[fp["target"], fp["pred"]], [2, 98])
err = fp["pred"] - fp["target"]
emax = float(np.nanpercentile(np.abs(err), 98)) or 1.0
cols = st.columns(4 if "pred_std" in fp.columns else 3)
cols[0].plotly_chart(plots.image(img(fp["target"]), "Target", zmin=vmin, zmax=vmax, unit="t/ha"), width="stretch")
cols[1].plotly_chart(plots.image(img(fp["pred"]), "Prediction", zmin=vmin, zmax=vmax, unit="t/ha"), width="stretch")
cols[2].plotly_chart(plots.image(img(err), "Error (pred − target)", colorscale="RdBu", zmin=-emax, zmax=emax,
                                 zmid=0, unit="t/ha"), width="stretch")
if "pred_std" in fp.columns:
    cols[3].plotly_chart(plots.image(img(fp["pred_std"]), "Ensemble std", colorscale="Magma", unit="t/ha"),
                         width="stretch")

with np.errstate(all="ignore"):
    corr = np.corrcoef(fp["target"], fp["pred"])[0, 1] if len(fp) > 2 else np.nan
rmse = float(np.sqrt(np.mean(err ** 2)))
c1, c2, c3 = st.columns(3)
c1.metric("Pixel RMSE (field)", f"{rmse:.2f} t/ha")
c2.metric("Pixel correlation (field)", f"{corr:.2f}")
c3.metric("Field-mean error", f"{info['error']:+.2f} t/ha")
s = fp.sample(min(len(fp), 5000), random_state=0)
fig = go.Figure(go.Scattergl(x=s["target"], y=s["pred"], mode="markers", marker=dict(size=3, opacity=0.4,
                                                                                     color="#0072B2")))
fig.add_trace(go.Scatter(x=[vmin, vmax], y=[vmin, vmax], mode="lines", line=dict(color="#999", dash="dash"),
                         hoverinfo="skip"))
fig.update_layout(title="Pixels of this field (up to 5000)", xaxis_title="target t/ha", yaxis_title="pred t/ha",
                  height=380, showlegend=False, margin=dict(l=0, r=0, t=36, b=0))
st.plotly_chart(fig, width="stretch")
