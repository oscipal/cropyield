"""Time-step chooser for the Sentinel-2 tab: one bar per step (height = share of valid pixels, hover = window, scene and
validity), ticks for the in-season steps, sowing / harvest markers. Click a bar or tick, or use the arrow keys."""
from __future__ import annotations

import json

import streamlit as st

CSS = """
.tl { font: 12px var(--st-font, sans-serif); color: var(--st-text-color, #333); user-select: none; outline: none; }
.tl svg { width: 100%; height: 118px; display: block; overflow: visible; }
.tl .hit { fill: transparent; cursor: pointer; }
.tl .hit:hover { fill: rgba(127,127,127,0.12); }
.tl .sel { fill: rgba(255,0,255,0.13); stroke: #ff00ff; stroke-width: 1.5; }
.tl .tip { position: absolute; pointer-events: none; background: var(--st-background-color, #fff);
           border: 1px solid #999; border-radius: 4px; padding: 4px 7px; white-space: pre; display: none;
           box-shadow: 0 2px 6px rgba(0,0,0,0.2); z-index: 10; }
.tl .wrap { position: relative; }
"""

JS = """
export default function(component) {
  const { data, setTriggerValue, parentElement } = component;
  parentElement.innerHTML = "";
  const d = data;
  const n = d.valid.length;
  const root = document.createElement("div"); root.className = "tl"; root.tabIndex = 0;
  const wrap = document.createElement("div"); wrap.className = "wrap"; root.appendChild(wrap);
  const tip = document.createElement("div"); tip.className = "tip"; wrap.appendChild(tip);
  const W = 1000, top = 18, barH = 56, trackY = top + barH + 14;
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} 118`); svg.setAttribute("preserveAspectRatio", "none");
  wrap.appendChild(svg);
  const x = (i) => (i + 0.5) * W / n, bw = Math.max(2, W / n * 0.7);
  const el = (tag, attrs) => { const e = document.createElementNS(svgNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
  // season band on the track
  if (d.sow_x !== null && d.harvest_x !== null) {
    el("rect", { x: d.sow_x * W / n, y: trackY - 3, width: Math.max(1, (d.harvest_x - d.sow_x) * W / n),
                 height: 6, fill: "#009E73", opacity: 0.35, rx: 3 });
  }
  el("line", { x1: 0, x2: W, y1: trackY, y2: trackY, stroke: "#999", "stroke-width": 1 });
  for (let i = 0; i < n; i++) {
    if (i === d.value) el("rect", { x: i * W / n, y: 2, width: W / n, height: trackY + 10, class: "sel", rx: 2 });
    const v = d.valid[i];
    if (v > 0) {
      const h = Math.max(2, v * barH);
      el("rect", { x: x(i) - bw / 2, y: top + barH - h, width: bw, height: h, fill: d.color[i], rx: 1 });
    } else if (d.in_season[i]) {
      el("rect", { x: x(i) - bw / 2, y: top + barH - 2, width: bw, height: 2, fill: "#bbbbbb" });
    }
    if (d.in_season[i]) el("line", { x1: x(i), x2: x(i), y1: trackY - 7, y2: trackY + 7, stroke: "#333",
                                     "stroke-width": 1.5 });
    else el("line", { x1: x(i), x2: x(i), y1: trackY - 2, y2: trackY + 2, stroke: "#bbb", "stroke-width": 1 });
    if (d.label_every && i % d.label_every === 0) {
      const t = el("text", { x: x(i), y: trackY + 22, "text-anchor": "middle", "font-size": 11, fill: "#777" });
      t.textContent = d.short[i];
    }
    const hit = el("rect", { x: i * W / n, y: 0, width: W / n, height: trackY + 12, class: "hit" });
    hit.addEventListener("click", () => setTriggerValue("step", i));
    hit.addEventListener("mousemove", (e) => {
      tip.textContent = d.hover[i]; tip.style.display = "block";
      const r = wrap.getBoundingClientRect();
      let left = e.clientX - r.left + 12; if (left > r.width - 260) left = e.clientX - r.left - 270;
      tip.style.left = left + "px"; tip.style.top = "0px";
    });
    hit.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  }
  for (const [key, label, col] of [["sow_x", "sowing", "#009E73"], ["harvest_x", "harvest", "#D55E00"]]) {
    if (d[key] === null) continue;
    const xx = d[key] * W / n;
    el("line", { x1: xx, x2: xx, y1: top - 4, y2: trackY + 9, stroke: col, "stroke-width": 2,
                 "stroke-dasharray": "4 3" });
    const t = el("text", { x: xx, y: top - 7, "text-anchor": "middle", "font-size": 11, fill: col });
    t.textContent = label + " " + d[key + "_label"];
  }
  root.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" && d.value < n - 1) setTriggerValue("step", d.value + 1);
    if (e.key === "ArrowLeft" && d.value > 0) setTriggerValue("step", d.value - 1);
  });
  parentElement.appendChild(root);
}
"""

_component = st.components.v2.component("s2_timeline", css=CSS, js=JS)


def timeline(valid, in_season, short, hover, color, value: int, sow_x=None, harvest_x=None, sow_label="",
             harvest_label="", key: str = "timeline") -> int | None:
    """Render the timeline; returns the clicked step (or None if nothing was clicked in this run).
    sow_x / harvest_x: position in step units (step index + fraction of the window)."""
    n = len(valid)
    data = {"valid": [float(v) for v in valid], "in_season": [bool(s) for s in in_season], "short": list(short),
            "hover": list(hover), "color": list(color), "value": int(value),
            "sow_x": None if sow_x is None else float(sow_x), "harvest_x": None if harvest_x is None else float(harvest_x),
            "sow_x_label": sow_label, "harvest_x_label": harvest_label, "label_every": max(1, round(n / 12))}
    json.dumps(data)   # fail early on non-serialisable values
    res = _component(data=data, key=key, on_step_change=lambda: None)
    return getattr(res, "step", None)
