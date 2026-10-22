"""Time-step slider for the Sentinel-2 tab, full width, on a day axis:
    day strip   one small bar per day with a Sentinel-2 scene (height = share of valid pixels; darker = the scene the
                cube uses for its window); hover for the date, valid share and what covers the rest
    track       the slider: small ticks at the window limits, tall ticks at sowing and harvest, the chosen window
                shaded; click or drag on it (or on the strip) to choose a window, or use the arrow keys
    labels      months"""
from __future__ import annotations

import json

import streamlit as st

CSS = """
.tl { font: 12px var(--st-font, sans-serif); color: var(--st-text-color, #333); user-select: none; outline: none;
      position: relative; padding: 0 2px; }
.tl .strip { position: relative; height: 26px; }
.tl .scene { position: absolute; bottom: 0; background: #cfcfcf; border-radius: 1px 1px 0 0; pointer-events: none; }
.tl .scene.chosen { background: #8f8f8f; }
.tl .scene.hl { background: #333; }
.tl .shit { position: absolute; top: 0; bottom: 0; width: 7px; margin-left: -3.5px; cursor: pointer; }
.tl .track { position: relative; height: 36px; cursor: pointer; touch-action: none; }
.tl .line { position: absolute; left: 0; right: 0; top: 17px; height: 3px; background: #c8c8c8; border-radius: 2px; }
.tl .tick { position: absolute; top: 13px; height: 11px; width: 1px; background: #9a9a9a; }
.tl .season { position: absolute; top: 2px; height: 33px; width: 2px; margin-left: -1px; }
.tl .slabel { position: absolute; top: -1px; font-size: 10px; white-space: nowrap; transform: translateX(4px); }
.tl .sel { position: absolute; top: 9px; height: 19px; background: rgba(255,75,75,0.22);
           border-left: 2px solid var(--st-primary-color, #ff4b4b); border-right: 2px solid var(--st-primary-color, #ff4b4b);
           border-radius: 2px; box-sizing: border-box; pointer-events: none; }
.tl .labels { position: relative; height: 14px; color: #888; font-size: 11px; }
.tl .labels span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
.tl .tip { position: absolute; pointer-events: none; background: var(--st-background-color, #fff); z-index: 10;
           border: 1px solid #999; border-radius: 4px; padding: 4px 7px; white-space: pre; display: none;
           box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-size: 12px; }
"""

JS = """
export default function(component) {
  const { data: d, setTriggerValue, parentElement } = component;
  const doc = parentElement.getRootNode();
  const hadFocus = !!(doc.activeElement && parentElement.contains(doc.activeElement));
  parentElement.innerHTML = "";
  const total = d.n * d.wdays;
  const pct = (day) => 100 * (day - d.start) / total;                  // % of the width for a day
  const div = (cls, parent) => { const e = document.createElement("div"); e.className = cls;
    parent.appendChild(e); return e; };
  const root = div("tl", parentElement); root.tabIndex = 0;
  const tip = div("tip", root);
  const showTip = (e, text) => {
    tip.textContent = text; tip.style.display = "block";
    const r = root.getBoundingClientRect();
    let left = e.clientX - r.left + 12; if (left > r.width - 280) left = e.clientX - r.left - 290;
    tip.style.left = Math.max(0, left) + "px"; tip.style.top = (e.clientY - r.top + 16) + "px";
  };
  const hideTip = () => { tip.style.display = "none"; };

  // day strip
  const strip = div("strip", root);
  const bw = `max(2px, ${100 / total}%)`;
  for (const [day, valid, chosen, text] of d.scenes) {
    const b = div("scene" + (chosen ? " chosen" : ""), strip);
    b.style.left = `calc(${pct(day + 0.5)}% - ${bw} / 2)`; b.style.width = bw;
    b.style.height = `${4 + Math.round(22 * valid)}px`;
    const h = div("shit", strip); h.style.left = `${pct(day + 0.5)}%`;
    h.addEventListener("mousemove", (e) => { b.classList.add("hl"); showTip(e, text); });
    h.addEventListener("mouseleave", () => { b.classList.remove("hl"); hideTip(); });
    h.addEventListener("click", () => setTriggerValue("step", Math.floor((day - d.start) / d.wdays)));
  }

  // track
  const track = div("track", root);
  div("line", track);
  for (let k = 0; k <= d.n; k++) div("tick", track).style.left = `${pct(d.start + k * d.wdays)}%`;
  for (const [day, label, col] of d.season) {
    const s = div("season", track); s.style.left = `${pct(day)}%`; s.style.background = col;
    const t = div("slabel", track); t.style.left = `${pct(day)}%`; t.style.color = col; t.textContent = label;
  }
  const sel = div("sel", track);
  const place = (k) => { sel.style.left = `${pct(d.start + k * d.wdays)}%`; sel.style.width = `${100 / d.n}%`; };
  let cur = d.value; place(cur);
  const stepAt = (e) => { const r = track.getBoundingClientRect();
    return Math.min(d.n - 1, Math.max(0, Math.floor((e.clientX - r.left) / r.width * d.n))); };
  let dragging = false;
  track.addEventListener("pointerdown", (e) => { dragging = true; track.setPointerCapture(e.pointerId);
    cur = stepAt(e); place(cur); root.focus(); });
  track.addEventListener("pointermove", (e) => {
    const k = stepAt(e);
    if (dragging && k !== cur) { cur = k; place(cur); }
    showTip(e, d.windows[k]);
  });
  track.addEventListener("pointerup", () => { if (dragging) { dragging = false;
    if (cur !== d.value) setTriggerValue("step", cur); } });
  track.addEventListener("mouseleave", hideTip);
  root.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" && d.value < d.n - 1) { e.preventDefault(); setTriggerValue("step", d.value + 1); }
    if (e.key === "ArrowLeft" && d.value > 0) { e.preventDefault(); setTriggerValue("step", d.value - 1); }
  });

  // month labels
  const labels = div("labels", root);
  for (const [day, text] of d.months) {
    const s = document.createElement("span"); s.style.left = `${pct(day)}%`; s.textContent = text;
    labels.appendChild(s);
  }
  if (hadFocus) root.focus();
}
"""

_component = st.components.v2.component("s2_timeline", css=CSS, js=JS)


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
