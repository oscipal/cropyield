"""Building blocks shared by the Fields and Predictions pages: year chooser, map with a persisted view, prediction
panel of one field."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

import data
import plots


def year_chooser(years: list, key: str, allow_all: bool = True, label: str = "Year") -> str | None:
    """Prev / next buttons and a slider over the years; returns the year or None for "all"."""
    opts = (["all"] if allow_all else []) + [str(y) for y in sorted(years)]
    if st.session_state.get(key) not in opts:
        st.session_state[key] = opts[1] if allow_all and len(opts) > 1 else opts[0]

    def shift(d):
        i = opts.index(st.session_state[key])
        st.session_state[key] = opts[min(max(i + d, 0), len(opts) - 1)]

    c1, c2, c3 = st.columns([1, 12, 1], vertical_alignment="bottom")
    c1.button("◀", key=f"{key}_prev", on_click=shift, args=(-1,), width="stretch")
    if len(opts) > 1:
        c2.select_slider(label, options=opts, key=key)
    else:
        c2.markdown(f"**{label}: {opts[0]}**")
    c3.button("▶", key=f"{key}_next", on_click=shift, args=(1,), width="stretch")
    return None if st.session_state[key] == "all" else st.session_state[key]


def persistent_map(build, key: str, reset_token, height: int = 480) -> str | None:
    """Show the folium map from build() -> (base map, polygon layer). Pan / zoom stay on the client: the view is not
    sent back (no rerun while navigating), and the polygons are swapped in place, so the view is kept across reruns
    until reset_token changes (it is part of the key). Returns the field of a newly clicked polygon (None otherwise)."""
    m, fg = build()
    out = st_folium(m, feature_group_to_add=fg, height=height, use_container_width=True,
                    key=f"{key}_{abs(hash(reset_token))}", returned_objects=["last_active_drawing"]) or {}
    clicked = ((out.get("last_active_drawing") or {}).get("properties") or {}).get("field")
    if clicked and clicked != st.session_state.get(f"{key}_click"):
        st.session_state[f"{key}_click"] = clicked
        return clicked
    return None


def prediction_panel(path: str, role: str, field: str) -> None:
    """Target / prediction / error / ensemble std maps, metrics and pixel scatter of one field of a run."""
    p = data.preds(path, role)
    fp = p[p["field"] == field]
    if fp.empty:
        st.info("No prediction for this field in this run and set.")
        return
    if "row" in fp.columns:
        r0, c0 = int(fp["row"].min()), int(fp["col"].min())
        lr, lc = fp["row"].to_numpy(int) - r0, fp["col"].to_numpy(int) - c0
        h, w = int(lr.max()) + 1, int(lc.max()) + 1
    else:   # baselines store only `pixel` (row in lstm/data/meta.parquet): place it via the field's inputs
        inp = data.field_inputs(field, "data")
        pos = fp["pixel"].to_numpy(int) - int(data.field_index().at[field, "start"])
        lr, lc, h, w = inp["lr"][pos], inp["lc"][pos], inp["h"], inp["w"]
    img = lambda v: data.to_image(np.asarray(v, dtype=np.float32), lr, lc, h, w)
    vmin, vmax = np.nanpercentile(np.r_[fp["target"], fp["pred"]], [2, 98])
    err = fp["pred"] - fp["target"]
    emax = float(np.nanpercentile(np.abs(err), 98)) or 1.0
    cols = st.columns(4 if "pred_std" in fp.columns else 3)
    cols[0].plotly_chart(plots.image(img(fp["target"]), "Target", zmin=vmin, zmax=vmax, unit="t/ha"), width="stretch")
    cols[1].plotly_chart(plots.image(img(fp["pred"]), "Prediction", zmin=vmin, zmax=vmax, unit="t/ha"),
                         width="stretch")
    cols[2].plotly_chart(plots.image(img(err), "Error (pred − target)", colorscale="RdBu_r", zmin=-emax, zmax=emax,
                                     zmid=0, unit="t/ha"), width="stretch")
    if "pred_std" in fp.columns:
        cols[3].plotly_chart(plots.image(img(fp["pred_std"]), "Ensemble std", colorscale="Magma", unit="t/ha"),
                             width="stretch")
    with np.errstate(all="ignore"):
        corr = np.corrcoef(fp["target"], fp["pred"])[0, 1] if len(fp) > 2 else np.nan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean target", f"{fp['target'].mean():.2f} t/ha")
    c2.metric("Mean prediction", f"{fp['pred'].mean():.2f} t/ha", f"{fp['pred'].mean() - fp['target'].mean():+.2f}",
              delta_color="off")
    c3.metric("Pixel RMSE (field)", f"{float(np.sqrt(np.mean(err ** 2))):.2f} t/ha")
    c4.metric("Pixel correlation (field)", f"{corr:.2f}")
    s = fp.sample(min(len(fp), 5000), random_state=0)
    fig = go.Figure(go.Scattergl(x=s["target"], y=s["pred"], mode="markers",
                                 marker=dict(size=3, opacity=0.4, color="#0072B2")))
    fig.add_trace(go.Scatter(x=[vmin, vmax], y=[vmin, vmax], mode="lines", line=dict(color="#999", dash="dash"),
                             hoverinfo="skip"))
    fig.update_layout(title="Pixels of this field (up to 5000)", xaxis_title="target t/ha", yaxis_title="pred t/ha",
                      height=360, showlegend=False, margin=dict(l=0, r=0, t=36, b=0))
    st.plotly_chart(fig, width="stretch")
