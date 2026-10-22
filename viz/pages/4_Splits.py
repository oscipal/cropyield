"""Map and counts of one split of the split suite (train / val / test fields)."""
from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

import data
import plots

st.set_page_config(page_title="Splits", page_icon="🧩", layout="wide")
st.title("Splits")

groups = data.split_groups()
with st.sidebar:
    st.header("Split")
    scope = st.selectbox("Scope", sorted(groups["scope"].unique()),
                         index=sorted(groups["scope"].unique()).index("world") if "world" in set(groups["scope"]) else 0)
    g = groups[groups["scope"] == scope]
    method = st.selectbox("Method", sorted(g["method"].unique()))
    crops = sorted(g[g["method"] == method]["crop"])
    crop = st.selectbox("Crop", crops, index=crops.index("all") if "all" in crops else 0)

t, attrs = data.split_table(scope, method, crop)
c = st.columns(3)
for col, s in zip(c, ("train", "val", "test")):
    sub = t[t["set"] == s]
    col.metric(s, f"{len(sub):,} field-years", f"{int(sub['n_pixels'].sum()):,} pixels", delta_color="off")

shapes = data.field_shapes().merge(t[["field", "set"]], on="field")
if len(shapes):
    st.caption(" · ".join(f"<span style='color:{v}'>■</span> {k}" for k, v in plots.SET_COLORS.items())
               + (f" · {len(t) - len(shapes)} field-years without a polygon" if len(t) > len(shapes) else ""),
               unsafe_allow_html=True)
    m, fg = plots.field_map(shapes, "set", colors=plots.SET_COLORS, tooltip=("field", "set", "crop", "year", "region"))
    st_folium(m, feature_group_to_add=fg, height=520, use_container_width=True, returned_objects=[],
              key=f"splitmap_{scope}_{method}_{crop}")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Field-years per set × country / crop**")
    st.dataframe(t.pivot_table(index=["country", "crop"], columns="set", values="field", aggfunc="count",
                               fill_value=0)[["train", "val", "test"]], width="stretch")
with c2:
    by = "year" if method == "loyo" else "region"
    st.markdown(f"**Field-years per set × {by}**")
    st.dataframe(t.pivot_table(index=by, columns="set", values="field", aggfunc="count", fill_value=0)
                 .reindex(columns=["train", "val", "test"], fill_value=0), width="stretch")
with st.expander("Split attributes"):
    st.json(attrs)
