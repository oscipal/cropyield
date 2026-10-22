"""Results of one model per split (top menu: model family, then variant), and a comparison of several models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import plots

st.set_page_config(page_title="Runs", page_icon="📊", layout="wide")
st.title("Runs")

runs, m = data.list_runs(), data.metrics_long()
if m.empty:
    st.warning("No finished run (metrics.json) found.")
    st.stop()
families = data.model_families()
METRICS = ["field_r2", "field_rmse", "pixel_r2", "pixel_rmse"]

# ---------------------------------------------------------------- model menu
fam = st.segmented_control("Model", list(families), default="Patch LSTM" if "Patch LSTM" in families
                           else list(families)[0], key="runs_family") or list(families)[0]
configs = families[fam]
if note := data.FAMILY_NOTES.get(fam):
    st.caption(note)
c1, c2, c3 = st.columns([2, 1, 1])
cfg = c1.selectbox("Variant", configs, key=f"runs_cfg_{fam}",
                   format_func=lambda c: c + ("  · screen" if data.config_info(c)["screen"] else "")) \
    if len(configs) > 1 else configs[0]
role = c2.radio("Set", ["test", "val"], horizontal=True, key="runs_role")
metric = c3.selectbox("Metric", METRICS, key="runs_metric")
higher_better = metric.endswith("r2")

info = data.config_info(cfg)
st.markdown(f"**`{cfg}`** — {info['desc'] or info['header'] or 'no description'}")
if info["screen"]:
    st.caption("Screening variant: val only matters, 2 members, 4 pooled splits (docs/convlstm_screening.md).")
if info["path"]:
    with st.expander(f"Config `{info['path']}`"):
        st.code(info["yaml"], language="yaml")

mine = m[(m["config"] == cfg) & (m["set"] == role)]
overall = mine[mine["group"] == "all"].sort_values("split")
base = m[(m["config"] == "mean") & (m["set"] == role) & (m["group"] == "all")].set_index("split")[metric]
cfg_runs = runs[runs["config"] == cfg]

# ---------------------------------------------------------------- header metrics
c = st.columns(4)
c[0].metric("Finished splits", f"{int(cfg_runs['done'].sum())} / {len(cfg_runs)}")
if len(overall):
    med = overall[metric].median()
    d_base = (overall.set_index("split")[metric] - base).dropna()
    c[1].metric(f"Median {role} {metric}", f"{med:.3f}")
    if cfg != "mean" and len(d_base):
        c[2].metric(f"Median Δ vs mean baseline", f"{d_base.median():+.3f}",
                    help="Same splits; the mean baseline predicts the training mean yield.")
    c[3].metric("Fields (sum over splits)", f"{int(overall['n_fields'].sum()):,}")
else:
    st.info(f"No {role} metrics for this config.")
    st.stop()

# ---------------------------------------------------------------- per split
st.subheader(f"Results per split ({role})")
tbl = overall.set_index("split")[["field_r2", "field_rmse", "pixel_r2", "pixel_rmse", "n_fields", "n_pixels",
                                  "member_std_field_r2", "best_epoch"]].copy()
tbl.insert(1, "Δ field_r2 vs mean", tbl["field_r2"] - m[(m["config"] == "mean") & (m["set"] == role) &
                                                          (m["group"] == "all")].set_index("split")["field_r2"])
missing = cfg_runs[~cfg_runs["done"]]["split"]
for s in missing:
    tbl.loc[s] = np.nan
tbl = tbl.reset_index()
styled = (tbl.style.format({c: "{:.3f}" for c in ["field_r2", "Δ field_r2 vs mean", "field_rmse", "pixel_r2",
                                                   "pixel_rmse", "member_std_field_r2"]} |
                           {"n_fields": "{:,.0f}", "n_pixels": "{:,.0f}", "best_epoch": "{:.0f}"}, na_rep="–")
          .background_gradient(subset=["field_r2", "pixel_r2"], cmap="RdYlGn", vmin=-0.2, vmax=0.8)
          .background_gradient(subset=["Δ field_r2 vs mean"], cmap="RdBu", vmin=-0.5, vmax=0.5))
ev = st.dataframe(styled, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                  key=f"runs_tbl_{cfg}_{role}")
if len(missing):
    st.caption(f"– = no metrics.json yet (running or unfinished): {', '.join(missing)}")

fig = go.Figure(go.Bar(y=overall["split"], x=overall[metric], orientation="h", marker_color="#0072B2", name=cfg,
                       error_x=dict(type="data", array=overall["member_std_field_r2"], visible=metric == "field_r2"),
                       hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
if cfg != "mean":
    b = base.reindex(overall["split"])
    fig.add_scatter(y=overall["split"], x=b, mode="markers", name="mean baseline",
                    marker=dict(symbol="line-ns", size=16, line=dict(width=3, color="#555")))
fig.update_layout(height=max(260, 28 * len(overall) + 80), margin=dict(l=0, r=0, t=10, b=0), xaxis_title=metric,
                  yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- selected split
rows = ev.selection.rows if ev and ev.selection else []
split = tbl.iloc[rows[0]]["split"] if rows else st.selectbox("Split details", overall["split"], key="runs_split")
if split not in set(overall["split"]):
    st.info("This split has no metrics yet.")
    st.stop()
st.subheader(f"{split}")
path = overall.set_index("split").at[split, "path"]
c1, c2 = st.columns([3, 2])
with c1:
    g = mine[(mine["split"] == split) & (mine["group"] != "all")]
    if len(g):
        st.markdown("**Per country × crop**")
        st.dataframe(g.set_index("group")[["field_r2", "field_rmse", "pixel_r2", "pixel_rmse", "n_fields"]]
                     .style.format("{:.3f}", subset=["field_r2", "field_rmse", "pixel_r2", "pixel_rmse"])
                     .background_gradient(subset=["field_r2"], cmap="RdYlGn", vmin=-0.2, vmax=0.8), width="stretch")
with c2:
    if st.button("Open predictions of this split ▶", type="primary"):
        st.session_state["goto_pred"] = {"config": cfg, "split": split, "role": role}
        st.switch_page("pages/3_Predictions.py")
r = cfg_runs[cfg_runs["split"] == split].iloc[0]
if r["has_history"]:
    h = data.history(path)
    if "member" not in h.columns:
        h["member"] = 0
    ycols = [c for c in h.columns if c not in ("member", "epoch", "epoch_time_s")]
    cols = st.columns(len(ycols))
    for c, y in zip(cols, ycols):
        fig = px.line(h, x="epoch", y=y, color=h["member"].astype(str), markers=True,
                      color_discrete_sequence=plots.OKABE_ITO, labels={"color": "member"})
        fig.update_layout(title=y, height=280, margin=dict(l=0, r=0, t=36, b=0), showlegend=(y == ycols[0]))
        c.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- comparison
with st.expander("Compare several models"):
    main = [v[0] for v in families.values()]
    sel = st.multiselect("Configs", sorted(m["config"].unique()), default=[c for c in main if c in set(m["config"])],
                         key="cmp_configs")
    cm = m[m["config"].isin(sel) & (m["set"] == role) & (m["group"] == "all")]
    if len(cm):
        pivot = cm.pivot_table(index="split", columns="config", values=metric)[[c for c in sel if c in set(cm["config"])]]
        ps = pivot.style.format("{:.3f}", na_rep="–")
        st.dataframe((ps.highlight_max if higher_better else ps.highlight_min)(axis=1, color="#56B4E955"),
                     width="stretch")
        fig = px.bar(cm, x="split", y=metric, color="config", barmode="group", color_discrete_sequence=plots.OKABE_ITO)
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.35))
        st.plotly_chart(fig, width="stretch")
