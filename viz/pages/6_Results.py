"""Results explorer: pick a model, step through the setups tried for it (in the order they were run), see how the
score developed (click a point to open that setup) and a filterable table per region / method / country / crop, on
all val / test fields or on new / seen fields only (recomputed from the runs' predictions, data.results_fieldset)."""
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
METRICS = {"field_r2": "field R²", "field_rmse": "field RMSE (t/ha)", "field_rrmse": "field rRMSE (%)",
           "pixel_r2": "pixel R²", "pixel_rmse": "pixel RMSE (t/ha)"}
FMT = {"field_rrmse": ".1f"}                 # number format per metric (default .3f)

# ---------------------------------------------------------------- model and score
archs = [a for a in ARCHS if a in set(S["architecture"])]
arch = st.segmented_control("Model", archs, default="ConvLSTM" if "ConvLSTM" in archs else archs[0],
                            key="res_arch") or archs[0]
S_arch = S[S["architecture"] == arch].reset_index(drop=True)

c1, c2, c3, c4 = st.columns([1, 1, 1.6, 1.3])
method = c1.segmented_control("Method", ["loyo", "loro"], default="loyo", format_func=str.upper,
                              key="res_method") or "loyo"
methods = [method]
role = c2.segmented_control("Set", ["val", "test"], default="val", key="res_role") or "val"
fset = c3.segmented_control("Fields", list(data.FIELD_SETS), default="all", key="res_fset",
                            format_func=data.FIELD_SETS.get) or "all"
metric = c4.selectbox("Metric", list(METRICS), format_func=METRICS.get, key="res_metric",
                      help="R²: higher is better. RMSE (t/ha) and rRMSE (RMSE in % of the subset's mean observed "
                           "field yield): lower is better.")
avail = set(res.loc[res["config"].isin(S_arch["config"]) & (res["method"] == method), "scope"])
present_scopes = [s for s in REGIONS if s in avail] or [s for s in REGIONS if s in set(res["scope"])]
# one choice: World holds the same fields as Europe + South America, so mixing it with them would count groups twice
region_opts = {}
for eu in ("europe", "germany"):
    if eu in present_scopes and "south_america" in present_scopes:
        region_opts[f"{eu}+south_america"] = f"{REGIONS[eu]} + South America"
region_opts |= {s: REGIONS[s] for s in present_scopes}
region = st.segmented_control("Region (split)", list(region_opts), default=next(iter(region_opts)),
                              format_func=region_opts.get, key=f"res_region_{arch}_{method}") \
    or next(iter(region_opts))
scopes = region.split("+")
min_fields = 5   # country x crop subsets with fewer fields are left out (R² is unstable)

KIND_LABELS = {"single": "one model (member mean)", "ensemble": "ensemble"}
with st.expander("More options"):
    n_mem = int(S_arch["n_members"].max() or 1)
    kinds = ["single", "ensemble"] + [f"m{i}" for i in range(n_mem)]
    kind = st.segmented_control(
        "Score of", kinds, default="single", key="res_kind",
        format_func=lambda k: KIND_LABELS.get(k) or f"model {int(k[1:]) + 1}",
        help="One model = the members' average score; ensemble = the averaged prediction; model i = member i alone "
             "(its own random seed). Setups with fewer members have no point for the higher model numbers.") \
        or "single"
    o3, o4 = st.columns([3, 1.4])
    exps = list(dict.fromkeys(S_arch["experiment"]))
    sel_exps = o3.pills("Experiment", exps, selection_mode="multi", default=exps, key=f"res_exps_{arch}")
    common = o4.toggle("Same splits for all setups", value=True, key="res_common",
                       help="Score every setup on the splits that all shown setups were trained on, so the points "
                            "are comparable. Off: each setup on all of its own splits.")
higher = metric.endswith("r2")
yfmt = FMT.get(metric, ".3f")
st.caption("Every model is trained on all crops and countries of a region together; the score is the median over its "
           "country × crop subsets (≥ 5 fields). Choices between setups are made on **val**; test is for the final "
           "record. *One model* = the members' average score, comparable between 2-member screenings and 5-member "
           "final runs.")
if fset != "all":
    st.caption("*New* = val / test field-years whose physical field (same land, other years) has no field-year in "
               "the split's train set; *seen* = it has one (scripts/loyo_eval.py). Scores are recomputed from the "
               "runs' predictions on that set; country × crop subsets with < 5 fields in it are dropped. The new-field "
               "sets are small (few subsets, Europe almost none; on LORO splits every field is new), so they are for "
               "looking, not for choosing a model.")

shown = S_arch[S_arch["experiment"].isin(sel_exps or [])].reset_index(drop=True)
shown["final"] = shown["n_members"] >= 5              # final runs: 5 members; screenings: 2 (choice on val only)
shown["run_type"] = [("baseline (no training)" if a == "Baselines" else
                      f"{'final run' if f else 'screening'} ({int(n)} model{'s' if n != 1 else ''})")
                     for a, f, n in zip(shown["architecture"], shown["final"], shown["n_members"])]
if shown.empty:
    st.info("No setup left with these filters.")
    st.stop()


def filt(d: pd.DataFrame, subsets: bool, k: str | None = None) -> pd.DataFrame:
    d = d[(d["kind"] == (k or kind)) & (d["set"] == role) & d["method"].isin(methods or []) & d["scope"].isin(scopes or [])]
    if subsets:
        d = d[(d["group"] != "all") & (d["n_fields"] >= min_fields)]
    else:
        d = d[d["group"] == "all"]
    return d


BASELINE = "peak_ndvi"    # drawn as a horizontal line in both charts, scored under the same filters
if fset == "all":
    base = res
else:                                                      # recomputed from preds_<role>.parquet, first load is slow
    runs = data.list_runs().query("done")
    need = runs[runs["config"].isin(shown["config"]) & runs["method"].isin(methods or [])
                & runs["scope"].isin(scopes or [])]
    with st.spinner(f"Scoring {len(need)} runs on {data.FIELD_SETS[fset]} {role} fields ..."):
        base = data.results_fieldset(fset, role, tuple(dict.fromkeys([*shown["config"], BASELINE])),
                                     tuple(sorted(set(need["split"]))))
    no_preds = [c for c in shown["config"] if c in set(need["config"]) and c not in set(base.get("config", []))]
    if no_preds:
        st.caption(f"Not scored (no saved {role} predictions, or no subset with ≥ 5 {fset} fields): "
                   + ", ".join(no_preds) + ".")
    if base.empty:
        st.info(f"No shown run has saved {role} predictions with a subset of ≥ 5 {fset} fields under these filters.")
        st.stop()
sub = filt(base[base["config"].isin(shown["config"])], subsets=True)
tot = filt(base[base["config"].isin(shown["config"])], subsets=False)
if common and len(sub):
    have = sub.groupby("config")["split"].apply(set)
    done, pool = shown.loc[shown["complete"], "config"], shown["config"]   # running setups do not shrink the splits
    # nor do setups without a run under these filters (e.g. LOYO-only screenings on LORO, baselines without preds)
    done, pool = done[done.isin(have.index)], pool[pool.isin(have.index)]
    keep = set.intersection(*[have.get(c, set()) for c in (done if len(done) else pool)])
    sub, tot = sub[sub["split"].isin(keep)], tot[tot["split"].isin(keep)]
    no_common = len(keep) == 0
else:
    keep, no_common = set(sub["split"]), False
if no_common:
    st.warning("The shown setups have no split in common under these filters. Narrow the experiments, or turn off "
               "'Same splits for all setups'.")

# peak-NDVI baseline under the same set, method, regions, field set and splits (no members: single = ensemble)
bl_sub = filt(base[base["config"] == BASELINE], subsets=True, k="ensemble")   # one fit, no seeds
bl_sub = bl_sub[bl_sub["split"].isin(keep)]
bl_missing = sorted(set(keep) - set(data.list_runs().query("done & config == @BASELINE")["split"]))
if bl_missing:                       # a line over only part of the splits would not be comparable
    bl_sub = bl_sub.iloc[:0]


def bl_line(f: go.Figure, d: pd.DataFrame) -> bool:
    """Dashed horizontal line at the baseline's median over the subsets of d; False if there is none."""
    if d.empty or pd.isna(d[metric].median()):
        return False
    f.add_hline(y=float(d[metric].median()), line=dict(color="#333", width=1.5, dash="dash"),
                annotation_text="peak-NDVI baseline", annotation_position="bottom left",
                annotation_font=dict(size=11, color="#333"))
    return True


def bl_note(drawn: bool, where: str) -> str:
    if drawn:
        return f"; dark dashed line = the peak-NDVI baseline ({BASELINE}), median over {where} under the same filters"
    why = f"no run on {', '.join(bl_missing)}" if bl_missing else "no subset under these filters"
    return f"; no peak-NDVI baseline line ({why})"


# ---------------------------------------------------------------- score development
score = sub.groupby("config")[metric].median()
n_sub = sub.groupby("config").size()
shown["score"] = shown["config"].map(score)
shown["n_subsets"] = shown["config"].map(n_sub).fillna(0).astype(int)
if (shown["n_subsets"] > 0).any():   # setups without a run under these filters (e.g. LOYO screenings on World) drop out
    shown = shown[shown["n_subsets"] > 0].reset_index(drop=True)
tt = data.train_times()
tt = tt[tt["split"].isin(keep)].groupby("config")["train_s"].sum()   # same splits as the score


def fmt_time(s: float) -> str:
    return "–" if pd.isna(s) else f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min"


shown["train_txt"] = [f"{fmt_time(tt.get(c))} training on {len(keep)} split(s)" for c in shown["config"]]
shown["label"] = [f"{i + 1}. {c}" for i, c in enumerate(shown["config"])]
best = shown["score"].cummax() if higher else shown["score"].cummin()

cfg_key = f"res_sel_{arch}"      # not a widget key: widget state is dropped while another model is shown
if st.session_state.get(cfg_key) not in set(shown["config"]):
    ok = shown.dropna(subset=["score"])
    st.session_state[cfg_key] = (ok.loc[ok["score"].idxmax() if higher else ok["score"].idxmin(), "config"]
                                 if len(ok) else shown["config"].iloc[-1])

STAR = dict(symbol="star", size=20, color="#F0E442", line=dict(width=1.5, color="#333"))


def best_of(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """The row(s) with the best value of col (highest R², lowest RMSE / rRMSE): drawn as a star in every chart."""
    v = d[col].dropna()
    return d.iloc[0:0] if v.empty else d[d[col] == (v.max() if higher else v.min())]


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
        marker=dict(size=12, color=np.where(s["complete"], colors[e], "white"),
                    symbol=np.where(s["final"], "circle", "diamond"),
                    line=dict(width=np.where(s["complete"], 1, 2.5), color=np.where(s["complete"], "white", colors[e]))),
        customdata=np.c_[s["config"], s["n_subsets"], s["n_members"],
                         pd.to_datetime(s["first"], unit="s").dt.strftime("%d.%m.%Y"), s["desc"].str.slice(0, 120),
                         np.where(s["complete"], "", "<br><i>still running: " + s["splits"].astype(str) + " / "
                                  + s["planned"].astype(str) + " splits done</i>"), s["train_txt"], s["run_type"]],
        hovertemplate="<b>%{customdata[0]}</b><br>" + METRICS[metric] + f" %{{y:{yfmt}}} (median of "
                      "%{customdata[1]} subsets)<br>%{customdata[7]}, first finished %{customdata[3]}<br>"
                      "%{customdata[6]}<br>%{customdata[4]}%{customdata[5]}<extra></extra>")
cur = shown[shown["config"] == st.session_state[cfg_key]]
b = best_of(shown, "score")
fig.add_scatter(x=b["label"], y=b["score"], mode="markers", name="best", marker=STAR, hoverinfo="skip")
fig.add_scatter(x=cur["label"], y=cur["score"], mode="markers", name="selected", hoverinfo="skip",
                marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=3, color="#D55E00")))
bl_top = bl_line(fig, bl_sub)
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
           f"the setups were first run. ◆ = screening (2 models, for choosing on val), ● = final run (5 models), "
           f"★ = the best setup. "
           f"Hover shows the training time (sum of the epoch times of all members on "
           f"these splits, without data loading). Click a point to open that setup; hollow = still running "
           f"(scored on the "
           f"splits it has so far){bl_note(bl_top, 'its subsets on the same splits')}. The member spread of a "
           f"screening is ~0.1, so smaller steps are noise.")

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
    l.caption(f"{row['experiment']} · {row['run_type']} · trained on {int(row['splits'])} splits"
              + ("" if row["complete"] else f" (running, {int(row['planned'])} planned)") + " · "
              f"first finished {pd.to_datetime(row['first'], unit='s'):%d.%m.%Y %H:%M}"
              + (f" · config `{info['path']}`" if info["path"] else ""))
    m1, m2 = r.columns(2)
    m1.metric(f"median {METRICS[metric]}", "–" if pd.isna(row["score"]) else f"{row['score']:{yfmt}}",
              f"{row['score'] - shown.set_index('config').at[cmp_cfg, 'score']:+{yfmt}} vs {cmp_cfg}"
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
    bullets = data.config_bullets(cfg)
    if bullets:
        st.markdown("\n".join(f"- {x}" for x in bullets))
    if info["yaml"]:
        with st.expander("Full config"):
            st.code(info["yaml"], language="yaml")

# ---------------------------------------------------------------- table
st.subheader("Scores")
pick = [cfg] + ([cmp_cfg] if cmp_cfg else [])
# columns per metric: (short name, header unit, number format, colour scale: values, Δ ±range; None = no colour)
TABLE = {"field_r2": ("R²", "", "{:.2f}", (-0.3, 0.9), 0.3), "field_rmse": ("RMSE", " (t/ha)", "{:.2f}", None, 0.5),
         "field_rrmse": ("rRMSE", " (%)", "{:.1f}", (5, 40), 10)}
if metric not in TABLE:                                     # a pixel metric chosen at the top: show it first
    TABLE = {metric: (METRICS[metric].split(" (")[0], " (t/ha)" if "rmse" in metric else "", "{:.2f}",
                      (-0.3, 0.9) if metric.endswith("r2") else None, 0.3 if metric.endswith("r2") else 0.5),
             **TABLE}
idx = ["region", "method", "country", "crop"]
d = sub[sub["config"].isin(pick)]
d = d.assign(region=d["scope"].map(REGIONS), method=d["method"].str.upper())
w = d.set_index(idx + ["config"])[list(TABLE)]
w = w[~w.index.duplicated()].unstack("config")
t = pd.DataFrame(index=w.index)
t["fields"] = d.groupby(idx)["n_fields"].max()
head = lambda m, c: f"{TABLE[m][0]}{TABLE[m][1]} · {c}"
for m, (short, *_) in TABLE.items():
    for c in pick:
        t[head(m, c)] = w[(m, c)] if (m, c) in w.columns else np.nan
    if cmp_cfg:
        t[f"Δ {short}"] = t[head(m, cfg)] - t[head(m, cmp_cfg)]
vals = [c for c in t.columns if c != "fields"]
if not t.empty:
    t.loc[("all", "", "", "median"), vals] = t[vals].median()
if t.empty:
    st.info("Nothing to show with these filters.")
else:
    t = t.reset_index()
    sty = t.style.format({"fields": "{:,.0f}"}, na_rep="–")
    for m, (short, unit, fmt, scale, dmax) in TABLE.items():
        hi = m.endswith("r2")
        cs = [head(m, c) for c in pick]
        sty = sty.format({c: fmt for c in cs + [f"Δ {short}"] if c in t}, na_rep="–")
        if scale:
            sty = sty.background_gradient(subset=cs, cmap="RdYlGn" if hi else "RdYlGn_r", vmin=scale[0],
                                          vmax=scale[1], axis=None)
        if f"Δ {short}" in t:                                # Δ = setup − compare; green = better
            sty = sty.background_gradient(subset=[f"Δ {short}"], cmap="RdYlGn" if hi else "RdYlGn_r", vmin=-dmax,
                                          vmax=dmax)
    st.dataframe(sty, hide_index=True, width="stretch", height=min(38 * (len(t) + 1), 640))
    st.download_button("Download table (CSV)", t.to_csv(index=False).encode(), f"results_{arch}_{role}_{fset}.csv",
                       "text/csv")
kind_txt = KIND_LABELS.get(kind) or f"model {int(kind[1:]) + 1} alone"
st.caption(f"{role} set, {kind_txt}"
           f"{'' if fset == 'all' else ', ' + data.FIELD_SETS[fset] + ' fields'}. Per-subset R² compares "
           f"field means within one country × crop; on pooled splits the whole-set R² is inflated by yield "
           f"differences between crops and countries. RMSE in t/ha on field means; rRMSE = that RMSE in % of the "
           f"subset's mean observed field yield (lower is better; one model = member-mean RMSE / mean). "
           f"Δ = {cfg} − {cmp_cfg or 'compare setup'}, green = better. Subsets with few fields "
           f"are noisy; Swiss subsets have 6–19 val fields.")


# ---------------------------------------------------------------- one area × crop over the setups
st.subheader("Score of one area and crop over the setups")
CONTINENT = {"Germany": "Europe", "Switzerland": "Europe", "Argentina": "South America", "Brazil": "South America",
             "Uruguay": "South America"}
d = sub.assign(continent=sub["country"].map(CONTINENT))   # same model, set and filters as above
countries = sorted(d["country"].dropna().unique())
if not countries:
    st.info("Nothing to show with these filters.")
    st.stop()
areas = ["all"] + [c for c in ("Europe", "South America") if c in set(d["continent"])] + countries
area = st.pills("Area", areas, default="all", key="res_tl_area",
                format_func=lambda a: "all countries" if a == "all" else a) or "all"
in_area = d if area == "all" else d[(d["continent"] == area) | (d["country"] == area)]
crops = ["all"] + sorted(in_area["crop"].dropna().unique())
crop = st.pills("Crop", crops, default="all", key=f"res_tl_crop_{area}",
                format_func=lambda c: "all crops" if c == "all" else c) or "all"
sel = in_area if crop == "all" else in_area[in_area["crop"] == crop]

g = sel.groupby("config").agg(area_score=(metric, "median"), n_sub=(metric, "size"), n_fields=("n_fields", "sum"))
tl = shown.join(g, on="config")
what = f"{'all countries' if area == 'all' else area}, {'all crops' if crop == 'all' else crop}"
fig2 = go.Figure()
fig2.add_scatter(x=tl["label"], y=tl["score"], mode="lines", name="median over all subsets (plot above)",
                 line=dict(color="#bbb", width=1.5, dash="dash"), hoverinfo="skip", connectgaps=True)
fig2.add_scatter(x=tl["label"], y=tl["area_score"], mode="lines", line=dict(color="#888", width=1),
                 hoverinfo="skip", showlegend=False, connectgaps=True)
for e in exps:
    s = tl[(tl["experiment"] == e) & tl["area_score"].notna()]
    if s.empty:
        continue
    fig2.add_scatter(x=s["label"], y=s["area_score"], mode="markers", name=e,
                     marker=dict(size=11, color=colors[e], line=dict(width=1, color="white")),
                     customdata=np.c_[s["config"], s["n_sub"].astype(int), s["n_fields"].astype(int),
                                      s["train_txt"]],
                     hovertemplate="<b>%{customdata[0]}</b><br>" + METRICS[metric] + f" %{{y:{yfmt}}} (median of "
                                   "%{customdata[1]} subsets, %{customdata[2]} fields)<br>%{customdata[3]}"
                                   "<extra></extra>")
cur2 = tl[tl["config"] == cfg]
b2 = best_of(tl, "area_score")
fig2.add_scatter(x=b2["label"], y=b2["area_score"], mode="markers", name="best", marker=STAR, hoverinfo="skip")
fig2.add_scatter(x=cur2["label"], y=cur2["area_score"], mode="markers", name="selected", hoverinfo="skip",
                 marker=dict(size=21, color="rgba(0,0,0,0)", line=dict(width=3, color="#D55E00")))
bsel = bl_sub.assign(continent=bl_sub["country"].map(CONTINENT))
bsel = bsel if area == "all" else bsel[(bsel["continent"] == area) | (bsel["country"] == area)]
bsel = bsel if crop == "all" else bsel[bsel["crop"] == crop]
bl_bottom = bl_line(fig2, bsel)
fig2.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=f"{METRICS[metric]}, {what} ({role})",
                   xaxis=dict(tickangle=-35), legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig2, width="stretch", key=f"res_tl_plot_{arch}")
st.caption(f"{what}: median {METRICS[metric]} over the matching country × crop subsets (one subset = its own score), "
           f"for the model, set and filters chosen at the top, in the same order as the plot above; grey dashed = the "
           f"median over all subsets{bl_note(bl_bottom, 'the same subsets')}. Single subsets with few fields jump a "
           f"lot between setups.")

# ---------------------------------------------------------------- score per crop
st.subheader("Score per crop")
pc_countries = sorted(sub["country"].dropna().unique())
pc_conts = [c for c in ("Europe", "South America") if c in set(sub["country"].map(CONTINENT))]
pc_country = st.pills("Country", ["all"] + pc_conts + pc_countries, default="all", key="res_pc_country",
                      format_func=lambda c: "all countries" if c == "all" else c) or "all"


def in_pc(d: pd.DataFrame) -> pd.DataFrame:
    return d if pc_country == "all" else d[(d["country"] == pc_country) | (d["country"].map(CONTINENT) == pc_country)]


pc_sub, pc_bl = in_pc(sub), in_pc(bl_sub)
pc = pc_sub.groupby(["config", "crop"]).agg(crop_score=(metric, "median"), n_sub=(metric, "size"),
                                         n_fields=("n_fields", "sum")).reset_index()
crop_order = sorted(pc["crop"].unique())
bl_pc = pc_bl.groupby("crop")[metric].median().reindex(crop_order) if len(pc_bl) else pd.Series(dtype=float)
fig3 = go.Figure()
for c in shown["config"]:                     # every shown setup as a thin line, the selected / compare ones on top
    if c in (cfg, cmp_cfg):
        continue
    s = pc[pc["config"] == c].set_index("crop").reindex(crop_order)
    fig3.add_scatter(x=crop_order, y=s["crop_score"], mode="lines", line=dict(color="#ccc", width=1),
                     name="other setups", legendgroup="other", showlegend=False, connectgaps=False,
                     hovertemplate=f"{c}<br>%{{x}}: %{{y:{yfmt}}}<extra></extra>")
fig3.add_scatter(x=[None], y=[None], mode="lines", line=dict(color="#ccc", width=1), name="other setups",
                 legendgroup="other")
for c, col in ((cmp_cfg, "#0072B2"), (cfg, "#D55E00")):
    if not c:
        continue
    s = pc[pc["config"] == c].set_index("crop").reindex(crop_order)
    fig3.add_scatter(x=crop_order, y=s["crop_score"], mode="lines+markers", name=c, line=dict(color=col, width=3),
                     marker=dict(size=10), customdata=np.c_[s["n_sub"].fillna(0).astype(int),
                                                            s["n_fields"].fillna(0).astype(int)],
                     hovertemplate=f"<b>{c}</b><br>%{{x}}: " + METRICS[metric] + f" %{{y:{yfmt}}} (median of "
                                   "%{customdata[0]} country subsets, %{customdata[1]} fields)<extra></extra>")
b3 = pd.concat([best_of(g, "crop_score") for _, g in pc.groupby("crop")]) if len(pc) else pc
fig3.add_scatter(x=b3["crop"], y=b3["crop_score"], mode="markers", name="best setup per crop", marker=STAR,
                 customdata=b3[["config"]].to_numpy(),
                 hovertemplate="best on %{x}: <b>%{customdata[0]}</b><br>" + METRICS[metric]
                               + f" %{{y:{yfmt}}}<extra></extra>")
if bl_pc.notna().any():
    fig3.add_scatter(x=crop_order, y=bl_pc, mode="lines+markers", name="peak-NDVI baseline",
                     line=dict(color="#333", width=1.5, dash="dash"), marker=dict(size=6),
                     hovertemplate="peak-NDVI baseline<br>%{x}: %{y:" + yfmt + "}<extra></extra>")
fig3.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=f"{METRICS[metric]} ({role})",
                   xaxis_title="crop", legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig3, width="stretch", key=f"res_crop_plot_{arch}")
st.caption(f"Per crop: {METRICS[metric]} of the chosen country's subset of that crop; with *all countries*, "
           f"*Europe* or *South America* the median over that crop's country subsets there (a crop grown in one "
           f"country only is that country's subset). "
           f"Orange = the "
           f"selected setup, blue = the compare setup, grey = the other shown setups, ★ = the best setup on each "
           f"crop, dashed = the peak-NDVI "
           f"baseline; same model, set and filters as at the top.")
