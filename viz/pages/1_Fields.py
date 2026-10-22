"""Field map and per-field inputs: yield map, Sentinel-2 per time step, weather, soil and terrain."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data
import panels
import plots
from timeline import timeline

st.set_page_config(page_title="Fields", page_icon="🗺️", layout="wide")
st.title("Fields")

DEFAULT_SPLIT = ("world", "loro", "all")
fields = data.fields_table()
by_field = fields.set_index("field")

# jump from another page (Predictions: "open inputs of this field")
if goto := st.session_state.pop("goto_field", None):
    st.session_state["field"] = goto
    st.session_state["fields_year"] = str(by_field.at[goto, "year"])
    st.session_state["fields_country"], st.session_state["fields_crop"] = [], []

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Filter")
    countries = st.multiselect("Country", sorted(fields["country"].unique()), key="fields_country")
    crops = st.multiselect("Crop type", sorted(fields["crop"].unique()), key="fields_crop")
    st.header("Map colour")
    color_by = st.radio("Colour by", ["split", "crop type", "yield"], horizontal=True, key="fields_color")
    if color_by == "split":
        groups = data.split_groups()
        scopes = sorted(groups["scope"].unique())
        scope = st.selectbox("Scope", scopes, index=scopes.index(DEFAULT_SPLIT[0]), key="fields_scope")
        methods = sorted(groups[groups["scope"] == scope]["method"].unique())
        method = st.selectbox("Method", methods, index=methods.index(DEFAULT_SPLIT[1])
                              if DEFAULT_SPLIT[1] in methods else 0, key="fields_method")
        scrops = sorted(groups[(groups["scope"] == scope) & (groups["method"] == method)]["crop"])
        scrop = st.selectbox("Split crop", scrops, index=scrops.index("all") if "all" in scrops else 0,
                             key="fields_scrop")
        split_name = f"{scope}/{method}/{scrop}"

sub_cc = fields
if countries:
    sub_cc = sub_cc[sub_cc["country"].isin(countries)]
if crops:
    sub_cc = sub_cc[sub_cc["crop"].isin(crops)]
if sub_cc.empty:
    st.warning("No field-year matches the filter.")
    st.stop()

# ---------------------------------------------------------------- year + map
year = panels.year_chooser(sorted(sub_cc["year"].unique()), key="fields_year")
sub = sub_cc if year is None else sub_cc[sub_cc["year"] == year]
if sub.empty:
    st.warning("No field-year in this year.")
    st.stop()

prev = st.session_state.get("field")
if prev not in set(sub["field"]):
    same = sub[sub["physical_field"] == by_field.at[prev, "physical_field"]] if prev in by_field.index else sub.iloc[:0]
    st.session_state["field"] = (same if len(same) else sub)["field"].iloc[0]

shapes_all = data.field_shapes()
shapes_cc = shapes_all[shapes_all["field"].isin(sub_cc["field"])]
shapes = shapes_cc[shapes_cc["field"].isin(sub["field"])].copy()
n_missing = len(sub) - len(shapes)

if color_by == "split":
    member = data.split_membership()
    member = member[member["split"] == split_name].set_index("field")["set"]
    shapes["set"] = shapes["field"].map(member).fillna("not in split")
    col, colors, legend = "set", plots.SET_COLORS_MAP, plots.SET_COLORS_MAP
elif color_by == "crop type":
    col, colors = "crop", plots.CROP_COLORS
    legend = {k: v for k, v in colors.items() if k in set(shapes_cc["crop"])}
else:
    col, colors, legend = "yield_mean", None, None
tooltip = ("field", "crop", "year", "yield_mean") + (("set",) if color_by == "split" else ())

if len(shapes_cc):
    st.caption((plots.legend(legend) + " · " if legend else "")
               + f"{len(sub):,} field-years" + (f" · {n_missing} without outline (e.g. Switzerland), pick them "
                                                f"below" if n_missing else ""), unsafe_allow_html=True)
    vrange = tuple(np.nanpercentile(shapes_cc["yield_mean"], [2, 98])) if col == "yield_mean" else None
    clicked = panels.persistent_map(
        lambda view: plots.field_map(shapes, col, colors=colors, selected=st.session_state["field"], tooltip=tooltip,
                                     bounds=shapes_cc.total_bounds, view=view, vrange=vrange,
                                     caption="field mean yield (t/ha)"),
        key="fields_map", reset_token=(tuple(countries), tuple(crops)))
    if clicked and clicked in set(sub["field"]):
        st.session_state["field"] = clicked
        st.rerun()

options = sub["field"].tolist()
field = st.selectbox("Field-year", options, index=options.index(st.session_state["field"]))
st.session_state["field"] = field
row = by_field.loc[field]
dates = data.field_dates().loc[field]
cols = st.columns(7)
for c, k in zip(cols, ["country", "crop", "year", "farm", "region", "n_pixels"]):
    c.metric(k, f"{row[k]:,}" if isinstance(row[k], (int, np.integer)) else str(row[k]))
if color_by == "split":
    cols[6].metric(f"set in {split_name}", member.get(field, "not in split"))
st.caption(f"Mean yield {row['yield_mean']:.2f} t/ha · sown {dates['seeding_date']:%d %b %Y} "
           f"({dates['seeding_date_type']}) · harvested {dates['harvesting_date']:%d %b %Y} · physical field "
           f"`{row['physical_field']}`")

# ---------------------------------------------------------------- field inputs
tab_y, tab_s2, tab_w, tab_st = st.tabs(["Yield", "Sentinel-2", "Weather", "Soil & terrain"])
base = data.field_inputs(field, "data")
img = lambda v: data.to_image(v, base["lr"], base["lc"], base["h"], base["w"])
SOW, HARVEST = pd.Timestamp(dates["seeding_date"]), pd.Timestamp(dates["harvesting_date"])


def mark_season(fig, x0, x1, **kw):
    """Grey out what lies outside sowing..harvest and draw both dates."""
    if x0 < SOW:
        fig.add_vrect(x0=x0, x1=SOW, fillcolor="#999", opacity=0.12, line_width=0, **kw)
    if x1 > HARVEST:
        fig.add_vrect(x0=HARVEST, x1=x1, fillcolor="#999", opacity=0.12, line_width=0, **kw)
    for d, c in ((SOW, "#009E73"), (HARVEST, "#D55E00")):
        fig.add_vline(x=d, line=dict(color=c, dash="dash", width=1.5), **kw)


with tab_y:
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(plots.image(img(base["target"]), "Yield (target)", unit="t/ha", height=480), width="stretch")
    c2.plotly_chart(go.Figure(go.Histogram(x=base["target"], nbinsx=50, marker_color="#0072B2"))
                    .update_layout(title="Pixel yield distribution", xaxis_title="t/ha", height=320,
                                   margin=dict(l=0, r=0, t=36, b=0)), width="stretch")

with tab_s2:
    grid_label = st.radio("Time grid", list(data.GRIDS), horizontal=True, key="s2_grid",
                          help="30-day (lstm/data): the YieldSAT cube, 24 windows from 1 Jan of the year before "
                               "harvest. 10/15-day and '30-day S2 only': rebuilt from the raw scenes, from day 210.")
    grid = data.GRIDS[grid_label]
    inp = base if grid == "data" else data.field_inputs(field, grid)
    sd = data.step_dates(field, grid)
    nd = data.ndvi(inp)
    valid = np.isfinite(data.band(inp, "B04")).mean(axis=0)
    n = len(valid)
    if st.session_state.get("s2_for") != (field, grid) or not 0 <= st.session_state.get("s2_step", -1) < n:
        st.session_state["s2_for"] = (field, grid)
        with np.errstate(all="ignore"):
            score = np.where(valid > 0.5, np.nan_to_num(np.nanmedian(nd, axis=0), nan=-1), -2)
        st.session_state["s2_step"] = int(np.argmax(score))
    step = st.session_state["s2_step"]

    scenes = data.scene_windows() if grid == "data" else None
    hover, color = [], []
    for k in range(n):
        s = sd.iloc[k]
        season = "in season" if s["in_season"] > 0 else "out of season"
        txt = f"step {k}: {s['start']:%d %b %Y} – {s['end']:%d %b %Y} ({season})\nvalid pixels {valid[k]:.0%}"
        if scenes is not None and (field, k) in scenes.index:
            w = scenes.loc[(field, k)]
            if pd.notna(w["chosen_date"]):
                txt += (f"\nscene {pd.Timestamp(w['chosen_date']):%d %b %Y} of {int(w['n_scenes'])} in window"
                        f"\nclass: {w['class']}" + (f" ({w['dominant_cause']})" if w["class"] != "valid"
                                                     and pd.notna(w["dominant_cause"]) else "")
                        + (f"\nbest scene in window: {w['best_valid']:.0%} valid" if w["best_valid"] > valid[k] + 0.05
                           else ""))
            else:
                txt += "\nno scene in this window"
        hover.append(txt)
        color.append("#009E73" if valid[k] >= 0.5 else "#E69F00" if valid[k] >= 0.05 else "#999999")
    info = data.grid_info(grid)
    start_day, wdays = info.get("start_day", 0), info.get("window_days", 30)
    pos = lambda d: (int(d) - start_day) / wdays
    clicked = timeline(valid, sd["in_season"] > 0, [f"{d:%b %y}" for d in sd["start"]], hover, color, step,
                       sow_x=pos(dates["sow_day"]), harvest_x=pos(dates["harvest_day"]),
                       sow_label=f"{SOW:%d %b %y}", harvest_label=f"{HARVEST:%d %b %y}", key=f"tl_{grid}")
    if clicked is not None and clicked != step:
        st.session_state["s2_step"] = int(clicked)
        st.rerun()

    def move(d):
        st.session_state["s2_step"] = int(min(max(st.session_state["s2_step"] + d, 0), n - 1))

    b1, b2, b3 = st.columns([1, 8, 1], vertical_alignment="center")
    b1.button("◀ step", on_click=move, args=(-1,), width="stretch")
    s = sd.iloc[step]
    b2.markdown(f"**Step {step}: {s['start']:%d %b %Y} – {s['end']:%d %b %Y}** · {valid[step]:.0%} valid pixels"
                + ("" if s["in_season"] > 0 else " · *out of season*"))
    b3.button("step ▶", on_click=move, args=(1,), width="stretch")

    rgb = np.stack([data.band(inp, b)[:, step] for b in ("B04", "B03", "B02")], axis=-1)
    c1, c2 = st.columns(2)
    c1.plotly_chart(plots.rgb(img(rgb), "RGB (B04/B03/B02, 2–98 % stretch)"), width="stretch")
    c2.plotly_chart(plots.image(img(nd[:, step]), "NDVI", colorscale="RdYlGn", zmin=-0.2, zmax=1), width="stretch")
    with np.errstate(all="ignore"):
        q = np.nanpercentile(nd, [10, 50, 90], axis=0)
    mid = sd["start"] + (sd["end"] - sd["start"]) / 2
    fig = plots.lines(mid, {"NDVI median": q[1]}, "Field NDVI (band: pixel 10–90 %), bars: valid pixels",
                      bands={"NDVI median": (q[0], q[2])})
    fig.add_bar(x=mid, y=valid, name="valid pixels", yaxis="y2", marker_color="#999999", opacity=0.3)
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 1], showgrid=False, title="valid"))
    mark_season(fig, sd["start"].min(), sd["end"].max())
    st.plotly_chart(fig, width="stretch")
    s2_bands = [b for b in inp["dyn_bands"] if b.startswith("B")]
    sel = st.multiselect("Bands (field mean)", s2_bands, default=["B02", "B04", "B08", "B11"])
    if sel:
        fig = plots.lines(mid, {b: np.nanmean(data.band(inp, b), axis=0) for b in sel}, "Field mean reflectance (DN)")
        mark_season(fig, sd["start"].min(), sd["end"].max())
        st.plotly_chart(fig, width="stretch")

with tab_w:
    PERIODS = {"daily": None, "10-day": "data_w10", "15-day": "data_w15", "30-day": "data",
               "YieldSAT (between scenes)": "yieldsat"}
    period = st.segmented_control("Period", list(PERIODS), default="30-day", key="w_period") or "30-day"
    g = PERIODS[period]
    if g == "yieldsat":
        st.caption("The 4 weather bands the models get today (lstm/data/dyn.npy): YieldSAT sums over the days between "
                   "consecutive *selected scenes* (both scene days counted), in Kelvin and metres. A value mostly "
                   "reflects the length of the gap, not the weather.")
        sd30 = data.step_dates(field, "data")
        x = sd30["start"] + (sd30["end"] - sd30["start"]) / 2
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Σ temperature between selected scenes (K·days)",
                                            "Σ precipitation between selected scenes (m)"))
        for i, b in enumerate(("temp_min", "temp_mean", "temp_max")):
            fig.add_scatter(x=x, y=np.nanmean(data.band(base, b), axis=0), name=b, mode="lines+markers",
                            line=dict(color=plots.OKABE_ITO[i]), row=1, col=1)
        fig.add_bar(x=x, y=np.nanmean(data.band(base, "total_prec"), axis=0), name="total_prec",
                    marker_color="#56B4E9", row=2, col=1)
        mark_season(fig, sd30["start"].min(), sd30["end"].max(), row="all", col=1)
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig, width="stretch")
    else:
        if g is None:
            w = data.daily_weather(field)
            x, xw = w["date"], None
        else:
            w = data.field_weather(field, g)
            if w is None:
                st.warning("Weather features missing: run `python scripts/LSTM/13_build_weather_grid.py`.")
                st.stop()
            x = w["start"] + (w["end"] - w["start"]) / 2
        has_gdd = g is not None
        rows = 3 if has_gdd else 2
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            specs=[[{}]] + [[{"secondary_y": True}]] * (rows - 1),
                            subplot_titles=("Temperature (°C): min–max band, mean line",
                                            "Precipitation (mm), line: cumulative since sowing",
                                            "Growing degree days, line: cumulative since sowing")[:rows])
        steps = w["step"] if "step" in w else None
        fig.add_scatter(x=x, y=w["tmax_c"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                        row=1, col=1)
        fig.add_scatter(x=x, y=w["tmin_c"], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(213,94,0,0.18)", name="min–max", row=1, col=1,
                        customdata=steps, hovertemplate="%{x|%d %b %Y}: min %{y:.1f} °C<extra></extra>")
        fig.add_scatter(x=x, y=w["tmean_c"], mode="lines+markers" if g else "lines", name="mean",
                        line=dict(color="#D55E00", width=2), marker=dict(size=5), row=1, col=1, customdata=steps,
                        hovertemplate="%{x|%d %b %Y}: mean %{y:.1f} °C<extra></extra>")
        fig.add_hline(y=0, line=dict(color="#56B4E9", width=1, dash="dot"), row=1, col=1)
        fig.add_bar(x=x, y=w["prec_mm"], name="precipitation", marker_color="#56B4E9", row=2, col=1,
                    customdata=steps)
        fig.add_scatter(x=x, y=w["prec_cum_mm"], name="cumulative", line=dict(color="#0072B2"), row=2, col=1,
                        secondary_y=True)
        if has_gdd:
            fig.add_bar(x=x, y=w["gdd"], name="GDD", marker_color="#E69F00", row=3, col=1, customdata=steps)
            fig.add_scatter(x=x, y=w["gdd_cum"], name="GDD cumulative", line=dict(color="#a06a00"), row=3, col=1,
                            secondary_y=True)
        xs = pd.to_datetime(x)
        mark_season(fig, xs.min(), xs.max(), row="all", col=1)
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=rows, col=1)
        fig.update_layout(height=260 * rows + 60, margin=dict(l=0, r=0, t=40, b=0), bargap=0.15,
                          legend=dict(orientation="h", y=-0.12))
        ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
                             key=f"weather_{period}_{field}")
        if g is not None:
            st.caption("Click a window to show the same step in the Sentinel-2 tab." +
                       ("" if g == "data" else " (switches its time grid to match)"))
            pts = (ev or {}).get("selection", {}).get("points", []) if ev else []
            pts = [p for p in pts if p.get("customdata") is not None]
            if pts:
                label = {v: k for k, v in data.GRIDS.items()}[g]
                st.session_state["s2_grid"] = label
                st.session_state["s2_for"] = (field, g)
                st.session_state["s2_step"] = int(pts[0]["customdata"])
                st.success(f"Step {int(pts[0]['customdata'])} selected in the Sentinel-2 tab ({label}).")
            with st.expander("More: frost / heat days, humidity, radiation"):
                cols = st.columns(3)
                f2 = go.Figure([go.Bar(x=x, y=w["frost_days"], name="frost (Tmin < 0 °C)", marker_color="#56B4E9"),
                                go.Bar(x=x, y=w["heat_days"], name="heat (Tmax > 30 °C)", marker_color="#D55E00")])
                f2.update_layout(title="Days per window", height=300, margin=dict(l=0, r=0, t=36, b=0),
                                 legend=dict(orientation="h", y=-0.2))
                cols[0].plotly_chart(f2, width="stretch")
                for c, f, t in ((cols[1], "rh_mean", "Relative humidity (%)"),
                                (cols[2], "srad", "Radiation (S_rad, likely Wh m⁻² d⁻¹)")):
                    if w[f].notna().any():
                        c.plotly_chart(plots.lines(x, {f: w[f]}, t, height=300), width="stretch")
                    else:
                        c.info(f"{t}: not in this field's weather CSV")

with tab_st:
    st.caption("SoilGrids 2.0 (250 m, upsampled to 10 m with cubic splines, so expect smooth gradients rather than "
               "detail) and SRTM terrain (30 m).")
    depth = st.segmented_control("Depth (cm)", data.DEPTHS, default="0-5", key="soil_depth") or "0-5"
    sand, silt, clay = (data.soil_values(base, p, depth) for p in ("sand", "silt", "clay"))
    cls = data.texture_class(sand, silt, clay)
    present = [i for i in range(len(data.TEXTURE)) if (cls == i).any()]
    k = len(data.TEXTURE)
    scale = [[v, plots.TEXTURE_COLORS[i]] for i in range(k) for v in (i / k, (i + 1) / k)]
    c1, c2 = st.columns([3, 2])
    with c1:
        z = np.where(cls >= 0, cls, np.nan).astype(float)
        fig = go.Figure(go.Heatmap(z=img(z), colorscale=scale, zmin=-0.5, zmax=k - 0.5, showscale=False,
                                   customdata=np.array(data.TEXTURE + ["-"])[np.nan_to_num(img(z), nan=k).astype(int)],
                                   hovertemplate="%{customdata}<extra></extra>"))
        st.plotly_chart(plots._image_layout(fig, f"USDA soil texture class, {depth} cm", 460), width="stretch")
        shares = {data.TEXTURE[i]: (cls == i).mean() for i in present}
        st.markdown(" · ".join(f"<span style='color:{plots.TEXTURE_COLORS[i]}'>■</span> {data.TEXTURE[i]} "
                               f"{shares[data.TEXTURE[i]]:.0%}" for i in present), unsafe_allow_html=True)
    with c2:
        # background: the USDA classes on a 2 % grid of the triangle
        gs = np.arange(0, 101, 2)
        gc, gsa = np.meshgrid(gs, gs)
        gsi = 100 - gc - gsa
        ok = gsi >= 0
        gcl = data.texture_class(gsa[ok], gsi[ok], gc[ok])
        fig = go.Figure(go.Scatterternary(a=gc[ok], b=gsa[ok], c=gsi[ok], mode="markers", hoverinfo="skip",
                                          marker=dict(size=6, symbol="square", opacity=0.35,
                                                      color=[plots.TEXTURE_COLORS[i] for i in gcl]),
                                          showlegend=False))
        t = pd.DataFrame({"clay": clay, "sand": sand, "silt": silt}).dropna().round(0).value_counts().reset_index()
        tc = data.texture_class(t["sand"], t["silt"], t["clay"])
        fig.add_trace(go.Scatterternary(a=t["clay"], b=t["sand"], c=t["silt"], mode="markers", showlegend=False,
                                        marker=dict(size=np.clip(4 + 14 * np.sqrt(t["count"] / t["count"].max()), 4, 18),
                                                    color=[plots.TEXTURE_COLORS[i] for i in tc],
                                                    line=dict(color="#000", width=1)),
                                        customdata=np.c_[t["count"], np.array(data.TEXTURE + ["-"])[tc]],
                                        hovertemplate="clay %{a}%, sand %{b}%, silt %{c}%<br>%{customdata[1]}"
                                                      "<br>%{customdata[0]} pixels<extra></extra>"))
        fig.update_layout(ternary=dict(aaxis_title="clay %", baxis_title="sand %", caxis_title="silt %"),
                          title="Texture triangle (field pixels)", height=460, margin=dict(l=30, r=30, t=50, b=30))
        st.plotly_chart(fig, width="stretch")

    unc = st.toggle("Show SoilGrids uncertainty instead of values", key="soil_unc")
    props = ["phh2o", "soc", "cec", "nitrogen", "cfvo"]
    cols = st.columns(len(props))
    for c, p in zip(cols, props):
        name, unit, _ = data.SOIL[p]
        c.plotly_chart(plots.image(img(data.soil_values(base, p, depth, unc)),
                                   f"{name}{' (uncertainty)' if unc else ''}", unit=unit, height=300),
                       width="stretch")

    with st.expander("Depth profile and numbers"):
        prof = pd.DataFrame({p: [np.nanmean(data.soil_values(base, p, d)) for d in data.DEPTHS] for p in data.SOIL},
                            index=data.DEPTHS)
        c1, c2 = st.columns(2)
        fig = go.Figure([go.Bar(y=prof.index, x=prof[p], name=data.SOIL[p][0], orientation="h",
                                marker_color=col) for p, col in (("sand", "#d9b65e"), ("silt", "#c6a78a"),
                                                                  ("clay", "#9b5b3b"))])
        fig.update_layout(barmode="stack", title="Texture by depth (%)", yaxis=dict(autorange="reversed",
                          title="depth (cm)"), height=320, margin=dict(l=0, r=0, t=36, b=0))
        c1.plotly_chart(fig, width="stretch")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(y=prof.index, x=prof["phh2o"], name="pH", mode="lines+markers", line=dict(color="#0072B2"))
        fig.add_scatter(y=prof.index, x=prof["soc"], name="organic C (g/kg)", mode="lines+markers",
                        line=dict(color="#009E73"), xaxis="x2")
        fig.update_layout(title="pH and organic carbon by depth", height=320, margin=dict(l=0, r=0, t=36, b=0),
                          yaxis=dict(autorange="reversed", title="depth (cm)"),
                          xaxis2=dict(overlaying="x", side="top"), legend=dict(orientation="h", y=-0.2))
        c2.plotly_chart(fig, width="stretch")
        st.dataframe(prof.rename(columns={p: f"{v[0]} ({v[1]})" if v[1] else v[0] for p, v in data.SOIL.items()})
                     .T.style.format("{:.3g}"), width="stretch")

    st.subheader("Terrain")
    cols = st.columns(len(data.TERRAIN))
    names = base["static_bands"]
    for c, (t, (name, unit)) in zip(cols, data.TERRAIN.items()):
        c.plotly_chart(plots.image(img(base["static"][:, names.index(t)]), name, colorscale="Cividis", unit=unit,
                                   height=280), width="stretch")
