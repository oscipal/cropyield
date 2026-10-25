"""Results explorer: pick a model, step through the setups tried for it (in the order they were run), see how the
score developed (click a point to open that setup) and a filterable table per region / method / country / crop."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import plots

st.set_page_config(page_title="Results", page_icon="📈", layout="wide")
st.title("Results")

res, S = data.results_long(), data.setups()
if res.empty:
    st.warning("No finished run (metrics.json) found.")
    st.stop()

ARCHS = ["ConvLSTM", "Patch LSTM", "LSTM (pixel)", "Field image", "Baselines"]
REGIONS = data.SCOPE_LABELS
METRICS = {"field_r2": "field R²", "field_rmse": "field RMSE (t/ha)", "pixel_r2": "pixel R²",
           "pixel_rmse": "pixel RMSE (t/ha)"}

# ---------------------------------------------------------------- model and score
archs = [a for a in ARCHS if a in set(S["architecture"])]
arch = st.segmented_control("Model", archs, default="ConvLSTM" if "ConvLSTM" in archs else archs[0],
                            key="res_arch") or archs[0]
S_arch = S[S["architecture"] == arch].reset_index(drop=True)

c1, c2, c3 = st.columns([1, 1.3, 1.2])
role = c1.segmented_control("Set", ["val", "test"], default="val", key="res_role") or "val"
kind = c2.segmented_control("Score of", ["single", "ensemble"], default="single", key="res_kind",
                            format_func={"single": "one model (member mean)", "ensemble": "ensemble"}.get) or "single"
metric = c3.selectbox("Metric", list(METRICS), format_func=METRICS.get, key="res_metric")
higher = metric.endswith("r2")
st.caption("Choices between setups are made on **val**; test is for the final record. *One model* = the members' "
           "average score, comparable between 2-member screenings and 5-member final runs.")

# ---------------------------------------------------------------- filters
with st.container(border=True):
    f1, f2 = st.columns([1, 3])
    methods = f1.pills("Method", ["loyo", "loro"], selection_mode="multi", default=["loyo", "loro"],
                       format_func=str.upper, key="res_methods")
    present_scopes = [s for s in REGIONS if s in set(res["scope"])]
    scopes = f2.pills("Region (split)", present_scopes, selection_mode="multi", default=present_scopes,
                      format_func=REGIONS.get, key="res_scopes")
    f4, f5, f6 = st.columns([2, 2, 1.4])
    countries = sorted(res["country"].dropna().unique())
    sel_countries = f4.pills("Country", countries, selection_mode="multi", default=countries, key="res_countries")
    crops = sorted(res["crop"].dropna().unique())
    sel_crops = f5.pills("Crop", crops, selection_mode="multi", default=crops, key="res_crops")
    min_fields = f6.number_input("Min. fields per subset", 1, 100, 5, key="res_minf",
                                 help="Country × crop subsets with fewer fields are left out (R² is unstable).")
    f7, f8 = st.columns([3, 1.4])
    exps = list(dict.fromkeys(S_arch["experiment"]))
    sel_exps = f7.pills("Experiment", exps, selection_mode="multi", default=exps, key=f"res_exps_{arch}")
    common = f8.toggle("Same splits for all setups", value=True, key="res_common",
                       help="Score every setup on the splits that all shown setups were trained on, so the points "
                            "are comparable. Off: each setup on all of its own splits.")

shown = S_arch[S_arch["experiment"].isin(sel_exps or [])].reset_index(drop=True)
if shown.empty:
    st.info("No setup left with these filters.")
    st.stop()


def filt(d: pd.DataFrame, subsets: bool) -> pd.DataFrame:
    d = d[(d["kind"] == kind) & (d["set"] == role) & d["method"].isin(methods or []) & d["scope"].isin(scopes or [])]
    if subsets:
        d = d[(d["group"] != "all") & d["country"].isin(sel_countries or []) & d["crop"].isin(sel_crops or [])
              & (d["n_fields"] >= min_fields)]
    else:
        d = d[d["group"] == "all"]
    return d


sub = filt(res[res["config"].isin(shown["config"])], subsets=True)
tot = filt(res[res["config"].isin(shown["config"])], subsets=False)
if common and len(sub):
    have = sub.groupby("config")["split"].apply(set)
    done = shown.loc[shown["complete"], "config"]          # running setups do not shrink the common splits
    keep = set.intersection(*[have.get(c, set()) for c in (done if len(done) else shown["config"])])
    sub, tot = sub[sub["split"].isin(keep)], tot[tot["split"].isin(keep)]
    no_common = len(keep) == 0
else:
    keep, no_common = set(sub["split"]), False
if no_common:
    st.warning("The shown setups have no split in common under these filters. Narrow the experiments, or turn off "
               "'Same splits for all setups'.")

# ---------------------------------------------------------------- score development
score = sub.groupby("config")[metric].median()
n_sub = sub.groupby("config").size()
shown["score"] = shown["config"].map(score)
shown["n_subsets"] = shown["config"].map(n_sub).fillna(0).astype(int)
shown["label"] = [f"{i + 1}. {c}" for i, c in enumerate(shown["config"])]
best = shown["score"].cummax() if higher else shown["score"].cummin()

cfg_key = f"res_sel_{arch}"      # not a widget key: widget state is dropped while another model is shown
if st.session_state.get(cfg_key) not in set(shown["config"]):
    ok = shown.dropna(subset=["score"])
    st.session_state[cfg_key] = (ok.loc[ok["score"].idxmax() if higher else ok["score"].idxmin(), "config"]
                                 if len(ok) else shown["config"].iloc[-1])

st.subheader("How the score developed")
colors = {e: plots.OKABE_ITO[i % len(plots.OKABE_ITO)] for i, e in enumerate(exps)}
fig = go.Figure()
fig.add_scatter(x=shown["label"], y=shown["score"], mode="lines", line=dict(color="#bbb", width=1),
                hoverinfo="skip", showlegend=False, connectgaps=True)
fig.add_scatter(x=shown["label"], y=best, mode="lines", line=dict(color="#555", width=1.5, dash="dot", shape="hv"),
                name="best so far", hoverinfo="skip")
for e in exps:
    s = shown[shown["experiment"] == e]
    if s.empty:
        continue
    fig.add_scatter(
        x=s["label"], y=s["score"], mode="markers", name=e,
        marker=dict(size=12, color=np.where(s["complete"], colors[e], "white"), symbol="circle",
                    line=dict(width=np.where(s["complete"], 1, 2.5), color=np.where(s["complete"], "white", colors[e]))),
        customdata=np.c_[s["config"], s["n_subsets"], s["n_members"],
                         pd.to_datetime(s["first"], unit="s").dt.strftime("%d.%m.%Y"), s["desc"].str.slice(0, 120),
                         np.where(s["complete"], "", "<br><i>still running: " + s["splits"].astype(str) + " / "
                                  + s["planned"].astype(str) + " splits done</i>")],
        hovertemplate="<b>%{customdata[0]}</b><br>" + METRICS[metric] + " %{y:.3f} (median of %{customdata[1]} "
                      "subsets)<br>%{customdata[2]} members, first finished %{customdata[3]}<br>%{customdata[4]}"
                      "%{customdata[5]}<extra></extra>")
cur = shown[shown["config"] == st.session_state[cfg_key]]
fig.add_scatter(x=cur["label"], y=cur["score"], mode="markers", name="selected", hoverinfo="skip",
                marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=3, color="#D55E00")))
fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=f"median {METRICS[metric]} ({role})",
                  xaxis=dict(tickangle=-35), legend=dict(orientation="h", y=1.12), clickmode="event+select")
ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key=f"res_plot_{arch}")
pts = (ev.selection.points if ev and ev.selection else []) or []
clicked = next((p["customdata"][0] for p in pts if p.get("customdata")), None)
if clicked and clicked != st.session_state.get(f"res_last_click_{arch}"):
    st.session_state[f"res_last_click_{arch}"] = clicked
    st.session_state[cfg_key] = clicked
    st.rerun()
splits_txt = ", ".join(sorted(keep)) if len(keep) <= 8 else f"{len(keep)} splits"
st.caption(f"Each point: median {METRICS[metric]} over the country × crop subsets that pass the filters "
           f"({'splits common to all shown setups: ' if common else 'splits: '}{splits_txt or '–'}), in the order "
           f"the setups were first run. Click a point to open that setup; hollow = still running (scored on the "
           f"splits it has so far). The member spread of a screening is ~0.1, so smaller steps are noise.")

# ---------------------------------------------------------------- selected setup
order = list(shown["config"])


def step(d: int) -> None:
    i = order.index(st.session_state[cfg_key])
    st.session_state[cfg_key] = order[(i + d) % len(order)]


b1, b2, b3 = st.columns([0.35, 4, 0.35])
b1.button("◀", on_click=step, args=(-1,), width="stretch", help="previous setup")
cfg = b2.selectbox("Setup", order, index=order.index(st.session_state[cfg_key]), label_visibility="collapsed",
                   key=f"res_box_{arch}_{st.session_state[cfg_key]}",
                   format_func=lambda c: f"{order.index(c) + 1}. {c}  ·  {shown.set_index('config').at[c, 'experiment']}")
if cfg != st.session_state[cfg_key]:
    st.session_state[cfg_key] = cfg
    st.rerun()
b3.button("▶", on_click=step, args=(1,), width="stretch", help="next setup")
row = shown.set_index("config").loc[cfg]
i = order.index(cfg)
prev = order[i - 1] if i > 0 else None
others = [c for c in order if c != cfg]
cmp_cfg = st.selectbox("Compare with", [None] + others, index=(others.index(prev) + 1) if prev else 0,
                       format_func=lambda c: "– nothing –" if c is None else c, key=f"res_cmp_{arch}_{cfg}")

info = data.config_info(cfg)
with st.container(border=True):
    l, r = st.columns([3, 2])
    l.markdown(f"### `{cfg}`\n{info['desc'] or info['header'] or 'no description'}")
    l.caption(f"{row['experiment']} · {int(row['n_members'])} member(s) · trained on {int(row['splits'])} splits"
              + ("" if row["complete"] else f" (running, {int(row['planned'])} planned)") + " · "
              f"first finished {pd.to_datetime(row['first'], unit='s'):%d.%m.%Y %H:%M}"
              + (f" · config `{info['path']}`" if info["path"] else ""))
    m1, m2 = r.columns(2)
    m1.metric(f"median {METRICS[metric]}", "–" if pd.isna(row["score"]) else f"{row['score']:.3f}",
              f"{row['score'] - shown.set_index('config').at[cmp_cfg, 'score']:+.3f} vs {cmp_cfg}"
              if cmp_cfg and pd.notna(row["score"]) and pd.notna(shown.set_index('config').at[cmp_cfg, 'score'])
              else None, delta_color="normal" if higher else "inverse")
    m2.metric("subsets", int(row["n_subsets"]))
    if cmp_cfg:
        a, b = data.flat_config(cfg), data.flat_config(cmp_cfg)
        diff = [(k, a.get(k, "–"), b.get(k, "–")) for k in sorted(set(a) | set(b))
                if a.get(k) != b.get(k) and k not in ("splits",)]
        if diff:
            st.markdown(f"**What differs from `{cmp_cfg}`**")
            st.dataframe(pd.DataFrame(diff, columns=["setting", cfg, cmp_cfg]), hide_index=True, width="stretch")
        elif a and b:
            st.caption(f"Same settings as `{cmp_cfg}` (apart from the splits).")
    if info["yaml"]:
        with st.expander("Full config"):
            st.code(info["yaml"], language="yaml")

# ---------------------------------------------------------------- table
st.subheader("Scores")
view = st.segmented_control(
    "Table", ["subsets", "splits", "summary", "all setups"], default="subsets", key="res_view",
    format_func={"subsets": "per country × crop", "splits": "per split (whole set)",
                 "summary": "median per region × method", "all setups": "all shown setups side by side"}.get) \
    or "subsets"
pick = [cfg] + ([cmp_cfg] if cmp_cfg else [])


def tidy(d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(region=d["scope"].map(REGIONS), method=d["method"].str.upper())


if view == "all setups":
    d = tidy(sub)
    t = d.pivot_table(index=["region", "method", "country", "crop"], columns="config", values=metric,
                      aggfunc="first")
    t = t[[c for c in order if c in t.columns]]
    t.insert(0, "fields", d.groupby(["region", "method", "country", "crop"])["n_fields"].max())
    cols = [c for c in t.columns if c != "fields"]
    t.loc[("all", "", "", "median"), cols] = t[cols].median()
else:
    if view == "splits":
        d = tidy(tot[tot["config"].isin(pick)])
        idx = ["region", "method"]
    else:
        d = tidy(sub[sub["config"].isin(pick)])
        idx = ["region", "method", "country", "crop"]
    if view == "summary":
        g = d.groupby(["region", "method", "config"])
        t = g[metric].median().unstack("config")
        t.insert(0, "subsets", g.size().groupby(level=[0, 1]).max())
    else:
        t = d.pivot_table(index=idx, columns="config", values=metric, aggfunc="first")
        t.insert(0, "fields", d.groupby(idx)["n_fields"].max())
    t = t[[c for c in t.columns if c in ("fields", "subsets")] + [c for c in pick if c in t.columns]]
    if cmp_cfg and cfg in t.columns and cmp_cfg in t.columns:
        t["Δ"] = t[cfg] - t[cmp_cfg]
    cols = [c for c in pick if c in t.columns]
if t.empty:
    st.info("Nothing to show with these filters.")
else:
    t = t.reset_index()
    vmin, vmax = (-0.3, 0.9) if higher else (None, None)
    sty = t.style.format({c: "{:.2f}" for c in cols + (["Δ"] if "Δ" in t else [])}, na_rep="–") \
        .format({c: "{:,.0f}" for c in ("fields", "subsets") if c in t}, na_rep="–")
    if higher:
        sty = sty.background_gradient(subset=cols, cmap="RdYlGn", vmin=vmin, vmax=vmax, axis=None)
    if "Δ" in t:
        sty = sty.background_gradient(subset=["Δ"], cmap="RdBu" if higher else "RdBu_r", vmin=-0.5, vmax=0.5)
    st.dataframe(sty, hide_index=True, width="stretch", height=min(38 * (len(t) + 1), 640))
    st.download_button("Download table (CSV)", t.to_csv(index=False).encode(), f"results_{arch}_{role}.csv",
                       "text/csv")
st.caption(f"{role} set, {'one model (member mean)' if kind == 'single' else 'ensemble'}. Per-subset R² compares "
           f"field means within one country × crop; on pooled splits the whole-set R² is inflated by yield "
           f"differences between crops and countries. Subsets with few fields are noisy; Swiss subsets have 6–19 "
           f"val fields.")
