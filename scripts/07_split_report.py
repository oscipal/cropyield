"""Write docs/splits.md: dataset statistics and the composition of every split in the split suite.

Reads the splits back from the zarr store (paths.yaml: split_suite), so the report shows what is stored.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_paths  # noqa: E402
from src.splitting import ROLES, SCOPES, SIZE_WEIGHT, TARGET, subset  # noqa: E402

ABBR = {"Germany": "DE", "Argentina": "AR", "Brazil": "BR", "Uruguay": "UY"}
METHOD_NAMES = {"loyo": "leave one year out", "loro": "leave regions out"}


def counts(s: pd.Series, sort_index: bool = False) -> str:
    vc = s.value_counts()
    vc = vc.sort_index() if sort_index else vc
    return ", ".join(f"{k} {v}" for k, v in vc.items())


def yields(df: pd.DataFrame) -> str:
    g = df.groupby(["crop", "country"])["yield_mean"].agg(["mean", "std", "size"])
    return "; ".join(f"{crop} {ABBR[c]} {r['mean']:.2f} ± {0 if r['size'] < 2 else r['std']:.2f}"
                     for (crop, c), r in g.iterrows())


def dataset_tables(fields: pd.DataFrame) -> list[str]:
    rows = []
    for (country, crop), g in fields.groupby(["country", "crop"]):
        rows.append({"country": country, "crop": crop, "field-years": len(g),
                     "physical fields": g["physical_field"].nunique(), "farms": g["farm"].nunique(),
                     "regions": g["region"].nunique(), "pixels": f"{g['n_pixels'].sum():,}",
                     "years": f"{g['year'].min()}–{g['year'].max()}",
                     "yield t/ha (mean ± sd)": f"{g['yield_mean'].mean():.2f} ± {g['yield_mean'].std():.2f}"})
    for country, g in fields.groupby("country"):
        rows.append({"country": f"**{country}**", "crop": "**all**", "field-years": len(g),
                     "physical fields": g["physical_field"].nunique(), "farms": g["farm"].nunique(),
                     "regions": g["region"].nunique(), "pixels": f"{g['n_pixels'].sum():,}",
                     "years": f"{g['year'].min()}–{g['year'].max()}", "yield t/ha (mean ± sd)": "–"})
    per_year = pd.crosstab([fields["country"], fields["crop"]], fields["year"], margins=True,
                           margins_name="total").reset_index()
    return ["### Field-years, fields and farms", "", pd.DataFrame(rows).to_markdown(index=False), "",
            "### Field-years per year", "", per_year.to_markdown(index=False), ""]


def split_table(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, code in ROLES.items():
        s = sub[sub["split"] == code]
        rows.append({
            "set": name,
            "field-years": f"{len(s)} ({len(s) / len(sub):.0%})",
            "physical fields": s["physical_field"].nunique(),
            "regions": s["region"].nunique(),
            "pixels": f"{s['n_pixels'].sum():,} ({s['n_pixels'].sum() / sub['n_pixels'].sum():.0%})",
            "countries": counts(s["country"].map(ABBR)),
            "crops": counts(s["crop"]),
            "years": counts(s["year"], sort_index=True),
            "yield t/ha (mean ± sd)": yields(s),
        })
    return pd.DataFrame(rows)


def main() -> None:
    paths = load_paths()
    fields = pd.read_parquet(paths["work_dir"] / "fields.parquet")
    store = paths["split_suite"]

    overview, sections = [], []
    for scope, countries in SCOPES.items():
        crops = sorted(subset(fields, scope, "all")["crop"].unique()) + ["all"]
        title = scope.replace("_", " ").title()
        sections += [f"## {title}" + ("" if countries == [title] else f" ({', '.join(countries)})"), ""]
        for method in METHOD_NAMES:
            for crop in crops:
                group = f"{scope}/{method}/{crop}"
                ds = xr.open_zarr(store, group=group, consolidated=False).load()
                stored = pd.DataFrame({"field": [str(f) for f in ds["field"].values],
                                       "split": ds["split"].values})
                sub = stored.merge(fields, on="field", how="left", validate="one_to_one")
                assert sub["country"].notna().all(), f"{group}: fields missing from fields.parquet"
                a = ds.attrs
                n = {k: int((sub["split"] == v).sum()) for k, v in ROLES.items()}
                if method == "loyo":
                    held_test, held_val = ", ".join(a["test_years"]), ", ".join(a["val_years"])
                else:
                    held_test, held_val = (f"{len(r)} region{'s' * (len(r) != 1)}"
                                           for r in (a["test_regions"], a["val_regions"]))
                overview.append({"split": f"`{group}`", "field-years": len(sub),
                                 "test held out": held_test, "val held out": held_val,
                                 **{k: f"{n[k]} ({n[k] / len(sub):.0%})" for k in ROLES},
                                 "score": f"{a['score']:.2f}"})

                lines = [f"### `{group}`", ""]
                if method == "loyo":
                    lines += [f"Test year **{held_test}**, val year **{held_val}**, train all other years.", ""]
                else:
                    lines += [f"Test regions ({len(a['test_regions'])}): {', '.join(a['test_regions'])}", "",
                              f"Val regions ({len(a['val_regions'])}): {', '.join(a['val_regions'])}", ""]
                comps = ", ".join(f"{k} {a[k]:.2f}" for k in sorted(a) if k.startswith(("val_", "test_"))
                                  and k not in ("val_years", "val_regions", "test_years", "test_regions"))
                lines += [split_table(sub).to_markdown(index=False), "",
                          f"Score {a['score']:.3f} ({comps}).", ""]
                sections += lines

    doc = [
        "# Data splits",
        "",
        "Generated by `scripts/07_split_report.py` from the split store "
        f"`{store}` and the field table `{paths['work_dir'] / 'fields.parquet'}`.",
        "",
        "## How the splits are built",
        "",
        "- **Scopes:** `germany` (DE), `south_america` (AR, BR, UY), `world` (all four).",
        "- **Crops:** one split per crop in the scope, plus `all` (crops pooled).",
        "- **Methods:** `loyo` holds out one year as test and another year as val; train is all other years. "
        "`loro` holds out sets of whole regions as test and val (target "
        f"{1 - 2 * TARGET:.0%} / {TARGET:.0%} / {TARGET:.0%} of field-years).",
        "- **One split per scope × method × crop** (no rotation over folds): "
        f"{len(overview)} splits.",
        "- **Unit:** the field-year. Pixels of a field are never split between sets, since neighbouring "
        "10 m pixels are nearly identical.",
        "- **Physical field:** field IDs are per field-year, so the same piece of land recurs across years "
        "under different IDs. Field-years whose pixel footprints overlap (IoU > 0.5 on a 20 m grid) form one "
        "physical field. This includes double cropping (two crops on the same land in one season, common in "
        "Brazil) and a few duplicated records.",
        "- **Region:** the farm, except that farms are merged when they share a physical field or when any of "
        "their fields lie within 10 km of each other (edge to edge, on the field footprints). Fields of "
        "different regions are therefore more than 10 km apart, and a held-out region is never next to a "
        "training region. In `loro`, every region and therefore every physical field is entirely in one set. "
        "In `loyo`, the same physical field appears in train in other years by design.",
        "- **Choosing the split:** among all year pairs (`loyo`) or 20,000 random region assignments "
        "(`loro`, seed 42), the split with the lowest score is kept. Score, summed over val and test: "
        f"{SIZE_WEIGHT:g} × |share of field-years − {TARGET:.0%}| + total-variation distance of the "
        "crop × country mix + total-variation distance of the year mix (`loro` only) + KS distance of the "
        "field-mean yield standardised per crop and country. Train must contain every crop. "
        "Lower is better.",
        "",
        "Load a split:",
        "",
        "```python",
        "from src.splits import load_split",
        f's = load_split("{store}", "south_america", "loro", "soybean")',
        's["train_fields"], s["val_fields"], s["test_fields"], s["attrs"]',
        "```",
        "",
        "## Dataset",
        "",
        *dataset_tables(fields),
        "## All splits",
        "",
        "Counts are field-years (share of the split's subset).",
        "",
        pd.DataFrame(overview).to_markdown(index=False),
        "",
        "Notes:",
        "",
        "- `world/rapeseed` equals `germany/rapeseed` and `world/corn` equals `south_america/corn` in content.",
        "- Germany has only 6 farms (one holds about half the fields), so its `loro` splits cannot hit "
        "70/15/15 exactly and the held-out farms differ in yield level from the rest.",
        "- Years differ a lot in size, so some `loyo` test or val sets are small "
        "(e.g. `south_america/loyo/corn`).",
        "",
        *sections,
    ]
    out = paths["docs_dir"] / "splits.md"
    out.write_text("\n".join(doc), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
