"""Plotly figures and folium maps for the visualizer."""
from __future__ import annotations

import folium
import numpy as np
import plotly.graph_objects as go
from branca.element import MacroElement, Template

# Okabe-Ito, as in scripts/swissyield/swiss_lib.py
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
SET_COLORS = {"train": "#0072B2", "val": "#E69F00", "test": "#D55E00"}


def image(img: np.ndarray, title: str = "", colorscale: str = "Viridis", zmin=None, zmax=None,
          zmid=None, unit: str = "", height: int = 380) -> go.Figure:
    """Single-band field image (10 m pixels, north up), NaN transparent."""
    fig = go.Figure(go.Heatmap(z=img, colorscale=colorscale, zmin=zmin, zmax=zmax, zmid=zmid,
                               colorbar=dict(title=unit, thickness=12),
                               hovertemplate="row %{y}, col %{x}<br>%{z:.3f}<extra></extra>"))
    return _image_layout(fig, title, height)


def rgb(img: np.ndarray, title: str = "", height: int = 380) -> go.Figure:
    """(h, w, 3) float image with NaN, shown with a 2-98 percentile stretch over the whole field."""
    finite = img[np.isfinite(img).all(axis=-1)]
    if len(finite):
        lo, hi = np.percentile(finite, 2, axis=0), np.percentile(finite, 98, axis=0)
        x = np.clip((img - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    else:
        x = np.zeros_like(img)
    a = np.where(np.isfinite(img).all(axis=-1, keepdims=True), 255, 0)
    u8 = np.concatenate([np.nan_to_num(x * 255), a], axis=-1).astype(np.uint8)
    fig = go.Figure(go.Image(z=u8, colormodel="rgba", hoverinfo="skip"))
    return _image_layout(fig, title, height)


def _image_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, showticklabels=False, showgrid=False)
    fig.update_xaxes(showticklabels=False, showgrid=False, constrain="domain")
    fig.update_layout(title=title, height=height, margin=dict(l=0, r=0, t=36, b=0),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def lines(x, series: dict[str, np.ndarray], title: str = "", ytitle: str = "", height: int = 320,
          bands: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> go.Figure:
    """Line chart of several series over the time steps; optional (low, high) bands per series."""
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        c = OKABE_ITO[i % len(OKABE_ITO)]
        if bands and name in bands:
            lo, hi = bands[name]
            fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1], y=list(hi) + list(lo)[::-1], fill="toself",
                                     fillcolor=c, opacity=0.18, line=dict(width=0), hoverinfo="skip",
                                     showlegend=False))
        fig.add_trace(go.Scatter(x=list(x), y=y, name=name, mode="lines+markers", line=dict(color=c, width=2),
                                 marker=dict(size=5), connectgaps=False))
    fig.update_layout(title=title, yaxis_title=ytitle, height=height, margin=dict(l=0, r=0, t=36, b=0),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.2))
    return fig


DOT_MAX_ZOOM = 13   # below this zoom every field also gets a fixed-size dot, so small fields stay visible


class _DotToggle(MacroElement):
    """Shows the field dots (circle markers of GeoJSON points) below DOT_MAX_ZOOM only; also for layers added later
    (st_folium's feature_group_to_add)."""
    _template = Template("""
    {% macro script(this, kwargs) %}
    (function(map) {
      const isDot = (l) => l instanceof L.CircleMarker && l.feature && l.feature.geometry.type === "Point";
      const apply = (l) => { if (isDot(l) && l._path) l._path.style.display =
                               map.getZoom() < {{ this.max_zoom }} ? "" : "none"; };
      map.on("zoomend", () => map.eachLayer(apply));
      map.on("layeradd", (e) => apply(e.layer));
    })({{ this._parent.get_name() }});
    {% endmacro %}""")

    def __init__(self, max_zoom: int):
        super().__init__()
        self.max_zoom = max_zoom


def field_map(gdf, color_col: str, colors: dict | None = None, cmap: str = "viridis",
              selected: str | None = None, tooltip=("field", "crop", "year", "yield_mean"), grey=None,
              bounds=None, vrange: tuple[float, float] | None = None,
              caption: str | None = None) -> tuple[folium.Map, folium.FeatureGroup]:
    """Folium map of field polygons, coloured by a category (colors dict) or a continuous column (cmap: "viridis" or
    "rdbu" for a diverging range around 0). grey: polygons drawn underneath in grey without tooltip. The map fits
    `bounds` (minx, miny, maxx, maxy), else the data.
    Every field also gets a dot of fixed screen size at its centre, shown below DOT_MAX_ZOOM.
    Returns (base map, polygon layer): pass the layer to st_folium(feature_group_to_add=...) so that a change of the
    polygons (selection, year) updates the layer in place instead of rebuilding the map and losing pan / zoom."""
    import branca.colormap as cm
    m = folium.Map(tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri World Imagery", name="Satellite").add_to(m)
    fg = folium.FeatureGroup(name="fields")
    if grey is not None and len(grey):
        folium.GeoJson(grey[["geometry"]].to_json(), name="train (grey)",
                       style_function=lambda f: {"fillColor": "#9e9e9e", "color": "#777777", "weight": 0.4,
                                                 "fillOpacity": 0.45}).add_to(fg)
    if colors is None and len(gdf):
        vals = gdf[color_col].astype(float)
        if vrange is None:
            vrange = (float(np.nanpercentile(vals, 2)), float(np.nanpercentile(vals, 98)))
        ramp = (["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"] if cmap == "rdbu"
                else ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"])
        scale = cm.LinearColormap(ramp, vmin=vrange[0], vmax=max(vrange[1], vrange[0] + 1e-6),
                                  caption=caption or color_col)
        scale.add_to(m)
        color_of = lambda v: scale(min(max(v, scale.vmin), scale.vmax)) if v is not None and np.isfinite(v) \
            else "#999999"
    else:
        color_of = lambda v: (colors or {}).get(v, "#999999")
    if len(gdf):
        cols = [c for c in tooltip if c in gdf.columns]
        props = gdf[list(dict.fromkeys(cols + [color_col, "field", "geometry"]))]
        folium.GeoJson(
            props.to_json(), name="fields",
            style_function=lambda f: {"fillColor": color_of(f["properties"][color_col]), "color":
                                      "#ff00ff" if f["properties"]["field"] == selected else "#333333",
                                      "weight": 3 if f["properties"]["field"] == selected else 0.6,
                                      "fillOpacity": 0.8},
            highlight_function=lambda f: {"weight": 3, "color": "#ff00ff"},
            tooltip=folium.GeoJsonTooltip(fields=cols)).add_to(fg)
        dots = props.copy()
        dots["geometry"] = dots.geometry.representative_point()
        folium.GeoJson(
            dots.to_json(), name="field dots", marker=folium.CircleMarker(radius=5),
            style_function=lambda f: {"fillColor": color_of(f["properties"][color_col]), "fillOpacity": 0.9,
                                      "color": "#ff00ff" if f["properties"]["field"] == selected else "#333333",
                                      "weight": 3 if f["properties"]["field"] == selected else 0.8,
                                      "radius": 8 if f["properties"]["field"] == selected else 5},
            tooltip=folium.GeoJsonTooltip(fields=cols)).add_to(fg)
    folium.LayerControl(collapsed=True).add_to(m)
    _DotToggle(DOT_MAX_ZOOM).add_to(m)
    b = bounds if bounds is not None else (gdf.total_bounds if len(gdf) else grey.total_bounds)
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    return m, fg


def legend(colors: dict) -> str:
    return " · ".join(f"<span style='color:{v}'>■</span> {k}" for k, v in colors.items())


# fixed colours so a crop / texture class keeps its colour across fields and filters
CROP_COLORS = {"wheat": "#E69F00", "winter_cereal": "#F0E442", "rapeseed": "#CC79A7", "corn": "#D55E00",
               "soybean": "#009E73"}
TEXTURE_COLORS = ["#f6e8a6", "#ecd27f", "#d9b65e", "#a5885a", "#c6a78a", "#d8c7b5", "#c2703d", "#9b5b3b",
                  "#7e6a5c", "#b5452e", "#6d5a7a", "#4a3328"]
