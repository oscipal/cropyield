"""Seasonal profiles and yield relationships, SwissYield vs YieldSAT Germany wheat.

Figures: seasonal_profiles.png (field-mean NDVI, valid-pixel share, B11, temperature, precipitation per time step),
corr_by_step.png (correlation of NDVI with yield per step: between field-years and within fields),
field_scatter.png (field mean yield vs peak NDVI and vs year), 
Tables: tables/swiss_static_vs_yield.csv; report/05_relations.md
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swiss_lib import COLORS, DYNAMIC, TABLES, Report, field_aggregates, ndvi, sample_group, save

REF = "YieldSAT Germany wheat"
N = 200_000


def groups_agg() -> dict[str, dict]:
    sw, ys = field_aggregates("Swiss all"), field_aggregates(REF)
    out = {REF: ys}
    for src in ("DUP1", "DUP2"):
        k = (sw["table"]["source"] == src).to_numpy()
        out[f"Swiss {src}"] = {"table": sw["table"][k].reset_index(drop=True), "dyn_mean": sw["dyn_mean"][k], "dyn_n": sw["dyn_n"][k],
                               "static_mean": sw["static_mean"][k], "static_names": sw["static_names"]}
    return out


def corr(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 5 and x[m].std() > 0 else np.nan


def main() -> None:
    r = Report("05_relations", "Seasonal profiles and yield relationships")
    A = groups_agg()
    bi = {b: DYNAMIC.index(b) for b in DYNAMIC}

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    steps = np.arange(24)
    for name, a in A.items():
        nd = ndvi(a["dyn_mean"])
        c = COLORS[name]
        for ax, y, t in ((axs[0, 0], nd, "NDVI of field mean"), (axs[0, 2], a["dyn_mean"][..., bi["B11"]], "B11 (field mean)"),
                         (axs[1, 0], a["dyn_mean"][..., bi["temp_mean"]], "temp_mean per step"),
                         (axs[1, 1], a["dyn_mean"][..., bi["total_prec"]], "total_prec per step")):
            m, s = np.nanmean(y, 0), np.nanstd(y, 0)
            ax.plot(steps, m, color=c, label=name)
            ax.fill_between(steps, m - s, m + s, color=c, alpha=0.15)
            ax.set_title(t, fontsize=10)
        frac = (a["dyn_n"][..., bi["B08"]] / a["table"]["n_pixels"].to_numpy()[:, None]).mean(0)
        axs[0, 1].plot(steps, frac, color=c, label=name)
    axs[0, 1].set_title("share of pixels with an S2 value", fontsize=10)
    axs[1, 2].axis("off")
    axs[0, 0].legend(fontsize=8)
    for ax in axs.ravel():
        ax.set_xlabel("time step (30 d from 1 Jan of seeding year)", fontsize=8)
    save(fig, "seasonal_profiles.png")

    # correlations between field-years and within fields
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for name, a in A.items():
        nd = ndvi(a["dyn_mean"])
        ybar = a["table"]["yield_mean"].to_numpy()
        between = [corr(nd[:, k], ybar) for k in steps]
        axs[0].plot(steps, between, color=COLORS[name], label=name, marker=".")
        rows += [{"group": name, "level": "between field-years", "step": k, "r_ndvi_yield": v} for k, v in enumerate(between)]
    axs[0].set(title="NDVI vs yield, between field-years (field means)", xlabel="time step", ylabel="Pearson r")
    for name in ("Swiss DUP1", "Swiss DUP2", REF):
        s = sample_group(name.replace("Swiss ", "Swiss ") if name != REF else REF, N)
        m, nd = s["meta"].reset_index(drop=True), ndvi(s["dyn"])
        y = m["target"].to_numpy(float)
        g = m["field_year"].to_numpy()
        yd = y - pd.Series(y).groupby(g).transform("mean").to_numpy()
        within = []
        for k in steps:
            x = nd[:, k]
            xd = x - pd.Series(x).groupby(g).transform("mean").to_numpy()
            within.append(corr(xd, yd))
        axs[1].plot(steps, within, color=COLORS[name], label=name, marker=".")
        rows += [{"group": name, "level": "within field", "step": k, "r_ndvi_yield": v} for k, v in enumerate(within)]
    axs[1].set(title="NDVI vs yield, within fields (both demeaned per field-year, pixel sample)", xlabel="time step")
    axs[0].legend(fontsize=8)
    save(fig, "corr_by_step.png")
    cor = pd.DataFrame(rows)
    r.h("NDVI-yield correlation per time step")
    r.table(cor.pivot_table(index="step", columns=["level", "group"], values="r_ndvi_yield").round(2))
    r.p("Figure: `figures/corr_by_step.png`. Steps without observations are NaN.")

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    peak_rows = []
    for name, a in A.items():
        nd = ndvi(a["dyn_mean"])
        peak = np.nanmax(nd, axis=1)
        t = a["table"]
        axs[0].scatter(peak, t["yield_mean"], s=14, color=COLORS[name], label=name, alpha=0.7)
        axs[1].scatter(t["yield_std"], t["yield_mean"], s=14, color=COLORS[name], alpha=0.7)
        peak_rows.append({"group": name, "field_years": len(t), "r(peak NDVI, field yield)": corr(peak, t["yield_mean"].to_numpy()),
                          "r(within-std, yield)": corr(t["yield_std"].to_numpy(), t["yield_mean"].to_numpy())})
    axs[0].set(xlabel="peak NDVI (field mean over steps)", ylabel="field mean yield (t/ha)")
    axs[1].set(xlabel="within-field yield std (t/ha)", ylabel="field mean yield (t/ha)")
    axs[0].legend(fontsize=8)
    sw = A["Swiss DUP1"]["table"].assign(src="DUP1")
    sw = pd.concat([sw, A["Swiss DUP2"]["table"].assign(src="DUP2")])
    for src, d in sw.groupby("src"):
        for yr, dd in d.groupby("year"):
            axs[2].scatter([int(yr)] * len(dd), dd["yield_mean"], color=COLORS[f"Swiss {src}"], s=14, alpha=0.7)
    axs[2].set(xlabel="harvest year", ylabel="field mean yield (t/ha)", title="Swiss field means by year")
    save(fig, "field_scatter.png")
    r.h("Field-level relations")
    r.table(pd.DataFrame(peak_rows).round(2))
    r.table(sw.groupby(["src", "year"])["yield_mean"].agg(["count", "mean", "std", "min", "max"]).round(2))
    r.p("By crop (DUP1 mixes wheat and winter_cereal = barley/triticale):")
    r.table(sw.groupby(["src", "crop"])["yield_mean"].agg(["count", "mean", "std"]).round(2))

    # which static layers explain Swiss field means (field level; many layers are constant within a farm)
    a = pd.concat([pd.DataFrame(A[g]["static_mean"], columns=A[g]["static_names"]).assign(y=A[g]["table"]["yield_mean"].to_numpy(), src=g)
                   for g in ("Swiss DUP1", "Swiss DUP2")])
    rows = [{"band": b, "r_all": corr(a[b].to_numpy(), a["y"].to_numpy()), "r_DUP1": corr(a[a.src == "Swiss DUP1"][b].to_numpy(), a[a.src == "Swiss DUP1"]["y"].to_numpy()),
             "r_DUP2": corr(a[a.src == "Swiss DUP2"][b].to_numpy(), a[a.src == "Swiss DUP2"]["y"].to_numpy()), "n_unique": a[b].nunique()}
            for b in A["Swiss DUP1"]["static_names"]]
    st = pd.DataFrame(rows).sort_values("r_all", key=abs, ascending=False)
    st.to_csv(TABLES / "swiss_static_vs_yield.csv", index=False)
    r.h("Static layers vs field-mean yield (top 15 by |r|)")
    r.table(st.head(15).round(2))
    r.p("Only 73 field-years, many of them neighbouring fields of one farm (DUP1) or the same field in several years: treat these correlations as descriptive. `n_unique` = distinct field means.")
    r.save()


if __name__ == "__main__":
    main()
