"""SoilGrids 2.0 soil properties in physical units, with depth selection.

The 8 properties x 6 depths in lstm/data/static.npy (columns `<prop>_<top>-<bottom>`) are SoilGrids "mapped units"
(scaled integers); they equal the raw soil/*.tif bands (band 2 k = value, 2 k + 1 = uncertainty of depth k, depths
in the order below). SoilGrids is 250 m, so values are near-constant within a field.

Spec (the part after "soil:" in the loader's feature list):
    "" or None          all properties, 0-30 cm
    "0-30cm"            all properties, thickness-weighted mean over the layers overlapping 0-30 cm
    "clay,soc@0-30cm"   selected properties at that depth
    "clay@0-5cm"        a single standard layer
Feature names: `soil_<prop>_<top>-<bottom>cm`, e.g. soil_clay_0-30cm.

The uncertainty layers are not offered: their unit is undocumented in YieldSAT (SoilGrids' index is the 90 %
interval width / median x 10) and they have extreme tails (around 3,400 for every property in some German pixels).
"""
from __future__ import annotations

import numpy as np

# property -> (factor from mapped units, physical unit, description)
PROPS = {
    "clay": (0.1, "%", "clay content"),
    "sand": (0.1, "%", "sand content"),
    "silt": (0.1, "%", "silt content"),
    "soc": (0.1, "g/kg", "soil organic carbon"),
    "nitrogen": (0.01, "g/kg", "total nitrogen"),
    "phh2o": (0.1, "pH", "pH in water"),
    "cec": (0.1, "cmol(c)/kg", "cation exchange capacity"),
    "cfvo": (0.1, "vol %", "coarse fragments"),
}
LAYERS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 100), (100, 200)]
DEFAULT_DEPTH = (0, 30)


def parse(spec: str | None) -> tuple[list[str], tuple[int, int]]:
    """'clay,soc@0-30cm' -> (['clay', 'soc'], (0, 30)); '0-30cm' -> (all, (0, 30)); '' -> (all, default)."""
    spec = (spec or "").replace(" ", "")
    props, depth = list(PROPS), DEFAULT_DEPTH
    if "@" in spec:
        p, d = spec.split("@", 1)
        props, depth = p.split(","), _depth(d)
    elif spec and spec[0].isdigit():
        depth = _depth(spec)
    elif spec:
        props = spec.split(",")
    unknown = [p for p in props if p not in PROPS]
    if unknown:
        raise KeyError(f"unknown soil properties {unknown}; known: {list(PROPS)}")
    return props, depth


def _depth(s: str) -> tuple[int, int]:
    top, bottom = s.removesuffix("cm").split("-")
    top, bottom = int(top), int(bottom)
    if not 0 <= top < bottom <= LAYERS[-1][1]:
        raise ValueError(f"depth {s}: need 0 <= top < bottom <= {LAYERS[-1][1]} cm")
    return top, bottom


def layer_weights(depth: tuple[int, int]) -> dict[str, float]:
    """Thickness weights of the standard layers overlapping `depth`, keyed by the column suffix ('0-5', ...)."""
    top, bottom = depth
    w = {f"{a}-{b}": max(0, min(b, bottom) - max(a, top)) for a, b in LAYERS}
    total = sum(w.values())
    return {k: v / total for k, v in w.items() if v > 0}


def columns(spec: str | None, static_names: list[str]) -> list[tuple[str, dict[int, float], str]]:
    """[(feature name, {static column: weight incl. unit factor}, unit)] for a soil spec."""
    props, depth = parse(spec)
    w = layer_weights(depth)
    out = []
    for p in props:
        factor, unit, _ = PROPS[p]
        cols = {static_names.index(f"{p}_{k}"): wk * factor for k, wk in w.items()}
        out.append((f"soil_{p}_{depth[0]}-{depth[1]}cm", cols, unit))
    return out


def compute(static: np.ndarray, rows: np.ndarray | slice, cols: list[tuple[str, dict[int, float], str]]
            ) -> np.ndarray:
    """(len(rows), len(cols)) float32 soil features for the given static.npy rows."""
    need = sorted({c for _, cw, _ in cols for c in cw})
    x = np.asarray(static[rows][:, need], dtype=np.float64)
    pos = {c: i for i, c in enumerate(need)}
    return np.stack([sum(x[:, pos[c]] * w for c, w in cw.items()) for _, cw, _ in cols], axis=1).astype(np.float32)
