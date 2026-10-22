"""Predictions of one run: map of the val / test fields (train grey), field-level scatter and error ranking, and the
target / prediction / error maps of the selected field."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import data
import panels
import plots

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")
st.title("Predictions")

runs = data.list_runs().query("done")
families = data.model_families()

# jump from the Runs page
if goto := st.session_state.pop("goto_pred", None):
    st.session_state["pred_family"] = data.family_of(goto["config"])
    st.session_state[f"pred_cfg_{st.session_state['pred_family']}"] = goto["config"]
    st.session_state["pred_split"] = goto["split"]
    st.session_state["pred_role"] = goto["role"]

with st.sidebar:
    st.header("Run")
    fams = [f for f, cs in families.items() if set(cs) & set(runs["config"])]
    if st.session_state.get("pred_family") not in fams:
        st.session_state["pred_family"] = "Patch LSTM" if "Patch LSTM" in fams else fams[0]
    fam = st.selectbox("Model", fams, key="pred_family")
    cfgs = [c for c in families[fam] if c in set(runs["config"])]
    cfg = st.selectbox("Variant", cfgs, key=f"pred_cfg_{fam}")
    st.caption(data.config_info(cfg)["desc"] or data.config_info(cfg)["header"])
    r = runs[runs["config"] == cfg]
    splits = sorted(r["split"])
    if st.session_state.get("pred_split") not in splits:
        st.session_state["pred_split"] = splits[0]
    split = st.selectbox("Split", splits, key="pred_split")
    run = r[r["split"] == split].iloc[0]
    roles = ["test", "val"] if run["has_val"] else ["test"]
    if st.session_state.get("pred_role") not in roles:
        st.session_state["pred_role"] = "test"
    role = st.radio("Set", roles, horizontal=True, key="pred_role")
    color_by = st.radio("Map colour", ["set", "field-mean error", "absolute error", "target yield"],
                        key="pred_color")

scope, method, scrop = split.split("/")
members, _ = data.split_table(scope, method, scrop)
with st.spinner("Loading predictions"):
    fl = data.field_level(run["path"], role)
fl["abs_error"] = fl["error"].abs()

# ---------------------------------------------------------------- year
role_years = sorted(members[members["set"] == role]["year"].unique())
if method == "loyo":
    year = role_years[0] if len(role_years) == 1 else None
    other = members[members["set"] == ("val" if role == "test" else "test")]["year"].unique()
    st.markdown(f"**LOYO: {role} year {', '.join(role_years)}**"
                + (f" · {'val' if role == 'test' else 'test'} year {', '.join(sorted(other))}" if len(other) else ""))
else:
    year = panels.year_chooser(role_years, key=f"pred_year_{split}_{role}")

# ---------------------------------------------------------------- map
shapes_all = data.field_shapes()
split_shapes = shapes_all.drop(columns=["year"]).merge(members[["field", "set", "year"]], on="field")
eval_sets = ["test", "val"] if color_by == "set" else [role]
shown = split_shapes[split_shapes["set"].isin(eval_sets)]
train = split_shapes[~split_shapes["set"].isin(eval_sets)]
if year is not None:
    shown = shown[shown["year"] == year] if method != "loyo" else shown
    train = train[train["year"] == year] if method != "loyo" else train
train = train.drop_duplicates("physical_field")
shown = shown.merge(fl[["field", "error", "abs_error", "target", "pred"]], on="field", how="left")
for c in ("error", "abs_error", "target", "pred"):
    shown[c] = shown[c].round(2)

if st.session_state.get("pred_field") not in set(fl["field"]):
    st.session_state["pred_field"] = fl["field"].iloc[0]
if year is not None and method != "loyo":
    in_year = fl[fl["year"] == year]["field"]
    if len(in_year) and st.session_state["pred_field"] not in set(in_year):
        st.session_state["pred_field"] = in_year.iloc[0]

if len(split_shapes):
    if color_by == "set":
        col, colors, cmap, vr, cap = "set", plots.SET_COLORS, None, None, None
        legend = {k: v for k, v in plots.SET_COLORS.items() if k != "train"} | {"train": "#9e9e9e"}
    elif color_by == "field-mean error":
        e = float(np.nanpercentile(fl["abs_error"], 95)) or 1.0
        col, colors, cmap, vr, cap, legend = "error", None, "rdbu", (-e, e), "pred − target (t/ha)", {"train": "#9e9e9e"}
    elif color_by == "absolute error":
        col, colors, cmap, vr, cap, legend = "abs_error", None, "viridis", None, "|pred − target| (t/ha)", \
            {"train": "#9e9e9e"}
    else:
        col, colors, cmap, vr, cap, legend = "target", None, "viridis", None, "field mean yield (t/ha)", \
            {"train": "#9e9e9e"}
    missing = len(members[members["set"].isin(eval_sets)]) - len(split_shapes[split_shapes["set"].isin(eval_sets)])
    st.caption(plots.legend(legend) + (f" · {missing} val/test field-years without outline"
                                       if missing else ""), unsafe_allow_html=True)
    clicked = panels.persistent_map(
        lambda: plots.field_map(shown, col, colors=colors, cmap=cmap or "viridis",
                                     selected=st.session_state["pred_field"], grey=train, vrange=vr, caption=cap,
                                     tooltip=("field", "set", "crop", "year", "target", "pred", "error"),
                                     bounds=split_shapes.total_bounds),
        key="pred_map", reset_token=split)
    if clicked:
        s = members.set_index("field")["set"].get(clicked)
        if s in ("val", "test"):
            if s != role and s in roles:
                st.session_state["pred_role"] = s
            st.session_state["pred_field"] = clicked
            st.rerun()

# ---------------------------------------------------------------- field-level scatter + ranking
flv = fl if year is None or method == "loyo" else fl[fl["year"] == year]
c1, c2 = st.columns([3, 2])
with c1:
    color = st.radio("Scatter colour", ["country", "crop", "year"], horizontal=True, key="pred_scatter_color")
    fig = go.Figure()
    for i, (k, g) in enumerate(flv.groupby(color)):
        fig.add_trace(go.Scatter(x=g["target"], y=g["pred"], mode="markers", name=str(k),
                                 marker=dict(color=plots.OKABE_ITO[i % 8], size=7, opacity=0.8),
                                 customdata=g["field"], hovertemplate="%{customdata}<br>target %{x:.2f}"
                                 "<br>pred %{y:.2f}<extra></extra>"))
    if len(flv):
        lo, hi = float(np.nanmin(flv[["target", "pred"]])), float(np.nanmax(flv[["target", "pred"]]))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#999", dash="dash"),
                                 showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=f"Field mean yield, {role} (click a point to select the field)", height=420,
                      xaxis_title="target t/ha", yaxis_title="prediction t/ha", margin=dict(l=0, r=0, t=36, b=0))
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
                         key=f"scatter_{cfg}_{split}_{role}_{color}_{year}")
    pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
    if pts and pts[0].get("customdata") and pts[0]["customdata"] != st.session_state.get("_pred_scatter_click"):
        st.session_state["_pred_scatter_click"] = st.session_state["pred_field"] = pts[0]["customdata"]
        st.rerun()
with c2:
    st.markdown("**Fields by absolute error of the field mean** (click a row)")
    rank = flv[["field", "country", "crop", "year", "target", "pred", "error"]].reset_index(drop=True)
    ev = st.dataframe(rank.style.format({"target": "{:.2f}", "pred": "{:.2f}", "error": "{:+.2f}"}), hide_index=True,
                      height=420, on_select="rerun", selection_mode="single-row",
                      key=f"rank_{cfg}_{split}_{role}_{year}")
    rows = ev.selection.rows if ev and ev.selection else []
    if rows and rank.at[rows[0], "field"] != st.session_state.get("_pred_rank_click"):
        st.session_state["_pred_rank_click"] = st.session_state["pred_field"] = rank.at[rows[0], "field"]
        st.rerun()

# ---------------------------------------------------------------- selected field
options = fl["field"].tolist()
field = st.selectbox("Field-year (sorted by absolute error)", options,
                     index=options.index(st.session_state["pred_field"]))
st.session_state["pred_field"] = field
info = fl.set_index("field").loc[field]
st.caption(f"{info['country']} · {info['crop']} · {info['year']} · {int(info['n_pixels']):,} pixels · {role} of "
           f"`{split}` · `{cfg}`")
if st.button("Open inputs of this field on the Fields page"):
    st.session_state["goto_field"] = field
    st.switch_page("pages/1_Fields.py")
panels.prediction_panel(run["path"], role, field)
