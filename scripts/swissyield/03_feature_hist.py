"""Histograms of the model inputs: SwissYield vs YieldSAT, on random labelled-pixel samples.

Figures: s2_bands.png (12 bands, all valid values over the 24 steps), ndvi.png, weather.png,
static_soil.png (8 soil properties at 5-15 cm), static_terrain.png (all other static bands),
valid_fraction.png (share of pixels with an observation per time step, and valid steps per pixel).
Table: tables/feature_shift.csv (per feature: Swiss median/IQR vs YieldSAT Germany wheat and standardised shift).
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swiss_lib import COLORS, DYNAMIC, S2, TABLES, WEATHER, clip_range, draw_hists, ndvi, sample_group, save

N = 200_000
NAMES = ["Swiss DUP1", "Swiss DUP2", "YieldSAT Germany wheat", "YieldSAT wheat"]


def grid(n: int, cols: int = 4):
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows))
    return fig, axs.ravel()


def hist_grid(series_by_feature: dict[str, dict[str, np.ndarray]], fname: str, nbins: int = 70) -> None:
    fig, axs = grid(len(series_by_feature))
    for ax, (feat, series) in zip(axs, series_by_feature.items()):
        lo, hi = clip_range(list(series.values()))
        draw_hists(ax, series, np.linspace(lo, hi, nbins))
        ax.set_title(feat, fontsize=9)
    for ax in axs[len(series_by_feature):]:
        ax.axis("off")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, fname)


def main() -> None:
    S = {n: sample_group(n, N) for n in NAMES}
    for n, s in S.items():
        print(f"{n}: {len(s['meta']):,} sampled pixels")

    def valid(a):
        return a[np.isfinite(a)]

    hist_grid({b: {n: valid(S[n]["dyn"][..., DYNAMIC.index(b)]) for n in NAMES} for b in S2}, "s2_bands.png")
    hist_grid({"NDVI (all steps)": {n: valid(ndvi(S[n]["dyn"])) for n in NAMES},
               "NDVI max over season": {n: valid(np.nanmax(np.where(np.isfinite(ndvi(S[n]["dyn"])), ndvi(S[n]["dyn"]), -9), axis=1)) for n in NAMES}},
              "ndvi.png")
    hist_grid({w: {n: valid(S[n]["dyn"][..., DYNAMIC.index(w)]) for n in NAMES} for w in WEATHER}, "weather.png")

    # static bands: use the names common to both datasets, one value per pixel
    names = {n: S[n]["static_names"] for n in NAMES}
    common = [b for b in names["Swiss DUP1"] if b in names["YieldSAT wheat"]]
    col = lambda n, b: S[n]["static"][:, names[n].index(b)]  # noqa: E731
    soil = [b for b in common if re.search(r"_5-15$", b)]
    other = [b for b in common if not re.search(r"_\d+-\d+(_uncertainty)?$", b)]
    hist_grid({b: {n: valid(col(n, b)) for n in NAMES} for b in soil}, "static_soil.png")
    hist_grid({b: {n: valid(col(n, b)) for n in NAMES} for b in other}, "static_terrain.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for n in NAMES:
        frac = np.isfinite(S[n]["dyn"][..., DYNAMIC.index("B08")]).mean(axis=0)
        axs[0].plot(range(24), frac, label=n, color=COLORS.get(n))
    axs[0].set(xlabel="time step (30 d from 1 Jan)", ylabel="fraction of pixels with S2 value", title="S2 availability")
    draw_hists(axs[1], {n: np.isfinite(S[n]["dyn"][..., DYNAMIC.index("B08")]).sum(1) for n in NAMES}, np.arange(0, 26) - 0.5)
    axs[1].set(xlabel="valid time steps per pixel", title="valid steps")
    axs[0].legend(fontsize=8)
    save(fig, "valid_fraction.png")

    # numeric shift of every feature, Swiss (DUP1, DUP2) vs YieldSAT Germany wheat
    def static_col(s, b):
        return valid(s["static"][:, s["static_names"].index(b)])

    feats = {f"dyn:{b}": (lambda s, i=i: valid(s["dyn"][..., i])) for i, b in enumerate(DYNAMIC)}
    feats.update({f"static:{b}": (lambda s, b=b: static_col(s, b)) for b in common})
    ref = S["YieldSAT Germany wheat"]
    rows = []
    for f, get in feats.items():
        r = np.percentile(get(ref), [25, 50, 75])
        for n in ("Swiss DUP1", "Swiss DUP2"):
            x = np.percentile(get(S[n]), [25, 50, 75])
            iqr = max(r[2] - r[0], 1e-9)
            rows.append({"feature": f, "group": n, "median": x[1], "ref_median": r[1], "iqr": x[2] - x[0],
                         "ref_iqr": r[2] - r[0], "shift_in_ref_iqr": (x[1] - r[1]) / iqr})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "feature_shift.csv", index=False)
    print("largest median shifts vs YieldSAT Germany wheat (in units of its IQR):")
    print(out.reindex(out["shift_in_ref_iqr"].abs().sort_values(ascending=False).index).head(25).round(2).to_string())


if __name__ == "__main__":
    main()
