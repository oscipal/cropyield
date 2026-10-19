"""Histograms of the yield target: SwissYield (all / DUP1 / DUP2) against YieldSAT.

Figures (<work_dir>/swissyield/figures):
  target_pixel.png        pixel-level yield, Swiss sources vs YieldSAT groups (density)
  target_field.png        field-year mean yield
  target_within_field.png within-field std and coefficient of variation
  target_by_year.png      Swiss yield per harvest year (small multiples, Germany wheat as reference)
  target_by_crop.png      Swiss yield per crop label
Table: tables/target_quantiles.csv
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swiss_lib import COLORS, TABLES, YS_SELECT, draw_hists, save, swiss_meta, yieldsat_meta


def groups() -> dict[str, pd.DataFrame]:
    s, y = swiss_meta(), yieldsat_meta()
    s = s[s["target"].notna()]
    out = {"Swiss all": s, "Swiss DUP1": s[s["source"] == "DUP1"], "Swiss DUP2": s[s["source"] == "DUP2"]}
    for name, sel in YS_SELECT.items():
        out[name] = y[sel(y).to_numpy()]
    return out


def field_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("field_year")["target"]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(), "n": g.size()}).assign(cv=lambda d: d["std"] / d["mean"])


def main() -> None:
    G = groups()
    rows = []
    for name, df in G.items():
        q = df["target"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        rows.append({"group": name, "pixels": len(df), "field_years": df["field_year"].nunique(),
                     "mean": df["target"].mean(), "std": df["target"].std(), **{f"q{int(k*100):02d}": v for k, v in q.items()}})
    q = pd.DataFrame(rows).set_index("group")
    q.to_csv(TABLES / "target_quantiles.csv")
    print(q.round(2).to_string())

    bins = np.linspace(0, 16, 81)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, log in zip(axs, (False, True)):
        draw_hists(ax, {k: v["target"].to_numpy() for k, v in G.items()}, bins)
        ax.set(xlabel="yield (t/ha)", ylabel="density", title="pixel yield" + (" (log y)" if log else ""))
        if log:
            ax.set_yscale("log")
    axs[0].legend(fontsize=8)
    save(fig, "target_pixel.png")

    fs = {k: field_stats(v) for k, v in G.items()}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    draw_hists(axs[0], {k: v["mean"] for k, v in fs.items()}, np.linspace(0, 14, 43))
    axs[0].set(xlabel="field-year mean yield (t/ha)", ylabel="density", title="field means")
    axs[0].legend(fontsize=8)
    draw_hists(axs[1], {k: v["std"] for k, v in fs.items()}, np.linspace(0, 4, 41))
    axs[1].set(xlabel="within-field std (t/ha)", title="within-field std")
    draw_hists(axs[2], {k: v["cv"] for k, v in fs.items()}, np.linspace(0, 0.8, 41))
    axs[2].set(xlabel="within-field CV", title="within-field coefficient of variation")
    save(fig, "target_field.png")

    # share of total variance that lies between field-years (rest is within-field)
    print("\nvariance decomposition (between field-years / total):")
    for k, df in G.items():
        tot = df["target"].var()
        between = df.groupby("field_year")["target"].transform("mean").var()
        print(f"  {k:24s} {between / tot:.2f}")

    sw = G["Swiss all"]
    years = sorted(sw["year"].unique())
    ref = G["YieldSAT Germany wheat"]["target"].to_numpy()
    fig, axs = plt.subplots(2, (len(years) + 1) // 2, figsize=(3.4 * ((len(years) + 1) // 2), 6), sharex=True)
    for ax, yr in zip(axs.ravel(), years):
        d = sw[sw["year"] == yr]
        draw_hists(ax, {"YieldSAT Germany wheat": ref}, bins, alpha=0.6)
        for src in ("DUP1", "DUP2"):
            a = d.loc[d["source"] == src, "target"].to_numpy()
            if len(a) > 100:
                draw_hists(ax, {f"Swiss {src}": a}, bins)
        ax.set_title(f"{yr} (n={d['field_year'].nunique()} field-years)", fontsize=9)
    for ax in axs.ravel()[len(years):]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=7)
    fig.supxlabel("yield (t/ha)")
    save(fig, "target_by_year.png")

    crops = sorted(sw["crop"].unique())
    fig, axs = plt.subplots(1, len(crops), figsize=(5 * len(crops), 4), squeeze=False)
    for ax, c in zip(axs[0], crops):
        d = sw[sw["crop"] == c]
        draw_hists(ax, {"YieldSAT Germany wheat": ref}, bins, alpha=0.6)
        for src in ("DUP1", "DUP2"):
            a = d.loc[d["source"] == src, "target"].to_numpy()
            if len(a) > 100:
                draw_hists(ax, {f"Swiss {src}": a}, bins)
        ax.set(title=f"{c} (n={d['field_year'].nunique()} field-years)", xlabel="yield (t/ha)")
    axs[0, 0].legend(fontsize=8)
    save(fig, "target_by_crop.png")


if __name__ == "__main__":
    main()
