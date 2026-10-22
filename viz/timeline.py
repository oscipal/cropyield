"""Time-step slider for the Sentinel-2 tab, full width, on a day axis (one SVG drawn in pixels at the actual width):
    day strip   one small grey bar per day with a Sentinel-2 scene (height = share of valid pixels; darker = the scene
                the 30-day cube uses for its window); hover for the date, valid share and what covers the rest
    slider      track with small ticks at the window limits, tall ticks at sowing and harvest, a thumb on the chosen
                window; click or drag on it, click a scene bar, or use the arrow keys
    labels      months
All styling is inline (attributes), nothing depends on a stylesheet."""
from __future__ import annotations

import json

import streamlit as st

JS = """
export default function(component) {
  const { data: d, setTriggerValue, parentElement } = component;
  const NS = "http://www.w3.org/2000/svg";
  const RED = "#ff4b4b", GREY = "#d0d0d0", DARK = "#8c8c8c";
  const H = 92, PAD = 10;                                     // height, side padding (room for the thumb)
  const Y = { stripBottom: 28, track: 52, label: 84 };
  const total = d.n * d.wdays;

  parentElement.innerHTML = "";
  const root = document.createElement("div");
  root.tabIndex = 0;
  Object.assign(root.style, { position: "relative", width: "100%", height: H + "px", outline: "none",
                              userSelect: "none", fontFamily: "sans-serif" });
  parentElement.appendChild(root);
  const tip = document.createElement("div");
  Object.assign(tip.style, { position: "absolute", display: "none", pointerEvents: "none", zIndex: 10,
    background: "#fff", color: "#222", border: "1px solid #999", borderRadius: "4px", padding: "4px 7px",
    font: "12px sans-serif", whiteSpace: "pre", boxShadow: "0 2px 6px rgba(0,0,0,0.2)" });

  let value = d.value, dragging = false;
  // the rerun after a change rebuilds the component: keep the keyboard focus if the user just used it
  const commit = (k) => { k = Math.max(0, Math.min(d.n - 1, k)); globalThis.__s2tlUsed = Date.now();
    if (k !== d.value) setTriggerValue("step", k); };

  function draw() {
    const W = Math.max(200, root.clientWidth);
    const x = (day) => PAD + (day - d.start) / total * (W - 2 * PAD);
    const stepAt = (px) => Math.floor((px - PAD) / (W - 2 * PAD) * d.n);
    root.innerHTML = ""; root.appendChild(tip);
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", W); svg.setAttribute("height", H); svg.style.display = "block";
    root.appendChild(svg);
    const el = (tag, a, parent = svg) => { const e = document.createElementNS(NS, tag);
      for (const k in a) e.setAttribute(k, a[k]); parent.appendChild(e); return e; };
    const showTip = (ev, text) => { tip.textContent = text; tip.style.display = "block";
      const r = root.getBoundingClientRect(); let left = ev.clientX - r.left + 12;
      if (left + 260 > r.width) left = ev.clientX - r.left - 272;
      tip.style.left = Math.max(0, left) + "px"; tip.style.top = (ev.clientY - r.top + 14) + "px"; };
    const hideTip = () => { tip.style.display = "none"; };

    // day strip: one bar per scene day
    const bw = Math.max(2, (W - 2 * PAD) / total);
    for (const [day, valid, chosen, text] of d.scenes) {
      const cx = x(day + 0.5), h = 3 + Math.round(24 * valid);
      const bar = el("rect", { x: cx - bw / 2, y: Y.stripBottom - h, width: bw, height: h,
                               fill: chosen ? DARK : GREY });
      const hit = el("rect", { x: cx - 4, y: 0, width: 8, height: Y.stripBottom + 2, fill: "transparent",
                               style: "cursor:pointer" });
      hit.addEventListener("mousemove", (ev) => { bar.setAttribute("fill", "#333"); showTip(ev, text); });
      hit.addEventListener("mouseleave", () => { bar.setAttribute("fill", chosen ? DARK : GREY); hideTip(); });
      hit.addEventListener("click", () => commit(Math.floor((day - d.start) / d.wdays)));
    }

    // slider: track, window ticks, season lines, thumb
    const g = el("g", { style: "cursor:pointer" });
    el("rect", { x: 0, y: Y.track - 14, width: W, height: 28, fill: "transparent" }, g);   // hit area
    el("rect", { x: PAD, y: Y.track - 2, width: W - 2 * PAD, height: 4, rx: 2, fill: "#dcdcdc" }, g);
    for (let k = 0; k <= d.n; k++) {
      const tx = x(d.start + k * d.wdays);
      el("line", { x1: tx, x2: tx, y1: Y.track - 6, y2: Y.track + 6, stroke: "#9a9a9a", "stroke-width": 1 }, g);
    }
    for (const [day, label, col] of d.season) {
      const sx = x(day);
      el("line", { x1: sx, x2: sx, y1: Y.track - 16, y2: Y.track + 16, stroke: col, "stroke-width": 2.5 }, g);
      const t = el("text", { x: sx + 4, y: Y.track - 9, fill: col, "font-size": 11 }, g);
      t.textContent = label;
    }
    const band = el("rect", { y: Y.track - 4, height: 8, rx: 2, fill: RED, opacity: 0.35 }, g);
    const thumb = el("circle", { cy: Y.track, r: 7, fill: RED, stroke: "#fff", "stroke-width": 2 }, g);
    const place = (k) => { const a = x(d.start + k * d.wdays), b = x(d.start + (k + 1) * d.wdays);
      band.setAttribute("x", a); band.setAttribute("width", Math.max(2, b - a));
      thumb.setAttribute("cx", (a + b) / 2); };
    place(value);
    const local = (ev) => ev.clientX - svg.getBoundingClientRect().left;
    g.addEventListener("pointerdown", (ev) => { dragging = true; g.setPointerCapture(ev.pointerId); root.focus();
      value = Math.max(0, Math.min(d.n - 1, stepAt(local(ev)))); place(value); });
    g.addEventListener("pointermove", (ev) => {
      const k = Math.max(0, Math.min(d.n - 1, stepAt(local(ev))));
      if (dragging && k !== value) { value = k; place(value); }
      showTip(ev, d.windows[k]); });
    g.addEventListener("pointerup", () => { if (dragging) { dragging = false; commit(value); } });
    g.addEventListener("mouseleave", hideTip);

    // month labels
    for (const [day, text] of d.months) {
      const mx = x(day);
      el("line", { x1: mx, x2: mx, y1: Y.track + 9, y2: Y.track + 14, stroke: "#bbb", "stroke-width": 1 });
      const t = el("text", { x: mx, y: Y.label, fill: "#888", "font-size": 11, "text-anchor": "middle" });
      t.textContent = text;
    }
  }

  root.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowRight") { ev.preventDefault(); commit(d.value + 1); }
    if (ev.key === "ArrowLeft") { ev.preventDefault(); commit(d.value - 1); }
  });
  draw();
  let lastW = root.clientWidth;
  new ResizeObserver(() => { if (root.clientWidth !== lastW) { lastW = root.clientWidth; draw(); } }).observe(root);
  if (Date.now() - (globalThis.__s2tlUsed || 0) < 5000) root.focus();
}
"""

_component = st.components.v2.component("s2_timeline", js=JS)


def timeline(n: int, start_day: int, window_days: int, value: int, windows: list[str], scenes: list, months: list,
             season: list, key: str = "timeline") -> int | None:
    """Render the slider; returns the newly chosen step (or None if nothing changed in this run).
    Days in the cube convention (src/season.py). windows: hover text per step; scenes: [(day, valid share, chosen,
    hover text)]; months: [(day, label)]; season: [(day, label, colour)] for sowing / harvest."""
    data = {"n": int(n), "start": int(start_day), "wdays": int(window_days), "value": int(value),
            "windows": list(windows), "scenes": [[int(a), float(b), bool(c), str(t)] for a, b, c, t in scenes],
            "months": [[int(a), str(t)] for a, t in months], "season": [[float(a), str(t), c] for a, t, c in season]}
    json.dumps(data)   # fail early on non-serialisable values
    res = _component(data=data, key=key, on_step_change=lambda: None)
    return getattr(res, "step", None)
