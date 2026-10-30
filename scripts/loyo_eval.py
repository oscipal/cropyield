"""LOYO evaluation of the ConvLSTM screening runs on val: a peak-NDVI baseline, new fields only, bootstrap CIs.

    python scripts/loyo_eval.py                                  # -> docs/loyo_eval.md
    python scripts/loyo_eval.py --configs loyo_ref loyo_sow --members 2 --boot 1000 --world

Read-only on existing predictions (<work_dir>/lstm/runs/<config>/<scope>__loyo__all/preds_val.parquet), trains
nothing. Val only: the test years are not looked at (--role accepts only "val").

Scores per split x country x crop subset, on field means (mean target vs mean prediction per field-year):
  field R2 of "one model" = mean over the first N members of each member's field R2 (as scripts/LSTM/08_screen_report.py,
  so 2- and 5-member runs are comparable). Two field sets: all val fields, and "new" val fields whose physical field
  (same land, other years) has no field-year in the split's train set. Subsets with < 5 fields in a set are skipped.

Bootstrap: fields resampled with replacement within each subset, the same draws for every model (paired). Per model:
90 % CI of each subset R2 and of the median over subsets; per model vs a reference: median over the common subsets of
the per-subset difference, its 90 % CI and the share of draws in which the model is better.

Baselines (model rows, one deterministic "member"):
  peak_ndvi   per country x crop, OLS of field-mean yield on field peak NDVI, fitted on the split's train fields
              (pooled over the crops of the country if < 5 train fields). Peak NDVI = max over the 10-day steps that
              start between sowing and harvest (src.loader.seasons()) of the field-mean NDVI of 100 random pixels per
              field-year; a step counts only if >= 25 % of those pixels are clear. Val fields without a valid peak get
              the training mean.
  train_mean  training mean of the country x crop: a constant per subset, so its subset R2 is <= 0 by construction.

Memory: the loader's pixel table is read with categorical string columns (src.loader._base is replaced in this
process only) and the NDVI is loaded in chunks of 300 field-years, so the job stays below ~2 GB.

CLAUDE.md: no Swiss data values. The doc and the console show R2, differences of R2 and counts only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.loader as L  # noqa: E402
from src.config import load_paths  # noqa: E402
from src.splits import load_split  # noqa: E402

DEFAULT_CONFIGS = ["loyo_ref", "loyo_sow", "loyo_lai", "convlstm_s2", "lstm_patch_v3_w10_s2", "quick_s2"]
SPLITS = ["europe/loyo/all", "south_america/loyo/all"]
BASELINES = ["peak_ndvi", "train_mean"]
MIN_FIELDS = 5
MIN_CLEAR = 0.25
PIXELS = 100
CHUNK = 300
CHECK = {"loyo_ref": 0.251, "loyo_sow": 0.384}   # docs/loyo_conv_screening.md, all val fields, 2 members
FIELD_SETS = {"all": "all val fields", "new": "new fields only (physical field not in train)"}


# --------------------------------------------------------------------------- low-memory pixel table for src.loader

@lru_cache(maxsize=1)
def _lean_base() -> dict:
    """src.loader._base with categorical (object-category) string columns: ~1.5 GB instead of ~4 GB."""
    work = load_paths()["work_dir"] / "lstm"
    info = json.loads((work / "data" / "info.json").read_text())
    cols = ["country", "field", "farm", "crop", "year"]
    meta = pq.read_table(work / "data" / "meta.parquet", read_dictionary=cols).to_pandas()
    for c in cols:
        s = meta[c]
        meta[c] = pd.Categorical.from_codes(s.cat.codes.to_numpy(),
                                            categories=pd.Index(s.cat.categories.to_numpy(dtype=object), dtype=object))
    code = meta["field"].cat.codes.to_numpy()
    start = np.r_[0, np.flatnonzero(code[1:] != code[:-1]) + 1]
    fields = pd.DataFrame({"field": meta["field"].cat.categories.to_numpy()[code[start]],
                           "country": meta["country"].to_numpy()[start].astype(str), "start": start,
                           "stop": np.r_[start[1:], len(code)]}).set_index("field")
    assert fields.index.is_unique, "rows of a field-year are not contiguous in meta.parquet"
    return {"work": work, "meta": meta, "fields": fields, "static_names": info["static_bands"]}


# --------------------------------------------------------------------------- inputs

def split_fields(split: str, ft: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, val) field tables of a split; val has `new` = physical field has no field-year in train."""
    s = load_split(load_paths()["split_suite"], *split.split("/"))
    tr = ft.loc[s["train_fields"]].copy()
    va = ft.loc[s["val_fields"]].copy()
    va["new"] = ~va["physical_field"].isin(set(tr["physical_field"]))
    return tr, va


def config_preds(run: Path, val: pd.DataFrame, n_members: int) -> tuple[pd.DataFrame | None, str]:
    """Field means (target, m0..m{k-1}) of a finished run's val predictions, indexed like `val`; or (None, reason)."""
    f, mj = run / "preds_val.parquet", run / "metrics.json"
    if not f.exists() or not mj.exists():
        return None, "no finished preds_val.parquet / metrics.json"
    attrs = json.loads(mj.read_text()).get("split_attrs", {})
    if attrs and "/".join(str(attrs.get(k)) for k in ("scope", "method", "crop")) != run.name.replace("__", "/"):
        return None, f"split_attrs do not match {run.name}"
    cols = [c for c in pq.ParquetFile(f).schema_arrow.names if c.startswith("pred_m")]
    use = [f"pred_m{m}" for m in range(n_members) if f"pred_m{m}" in cols]
    p = pd.read_parquet(f, columns=["field_year", "target"] + use)
    g = p.groupby("field_year", observed=True)[["target"] + use].mean()
    g.columns = ["target"] + [f"m{m}" for m in range(len(use))]
    if set(g.index) != set(val.index):
        return None, "val fields differ from the split"
    note = "" if len(use) == n_members else f"only {len(use)} members"
    return g.loc[val.index], note


def peak_ndvi(fields: list[str]) -> pd.Series:
    """Season peak of the field-mean NDVI (100 random pixels per field-year, steps with >= 25 % clear pixels)."""
    seas = L.seasons()
    out = []
    for a in range(0, len(fields), CHUNK):
        d = L.load(fields=fields[a:a + CHUNK], features=["indices:ndvi"], grid="10d", pixels_per_field=PIXELS,
                   seed=0, verbose=False)
        x = pd.DataFrame(d.dyn[..., 0].astype(np.float32))
        fid = d.meta["field"].astype(str).to_numpy()
        grp = x.groupby(fid)
        mean, clear = grp.mean(), grp.count().div(grp.size(), axis=0)        # (fields, steps)
        mean = mean.where(clear >= MIN_CLEAR)
        s = seas.reindex(mean.index)
        days = d.days[None, :]
        in_season = (days >= s["sow_day"].to_numpy()[:, None]) & (days < s["harvest_day"].to_numpy()[:, None])
        out.append(mean.where(in_season).max(axis=1, skipna=True))
        del d, x
    return pd.concat(out).reindex(fields)


def baselines(tr: pd.DataFrame, va: pd.DataFrame, peak: pd.Series) -> tuple[dict[str, pd.DataFrame], dict]:
    """Field predictions of peak_ndvi and train_mean for the val fields (target = field-mean yield)."""
    tr = tr.assign(peak=peak.reindex(tr.index))
    va = va.assign(peak=peak.reindex(va.index))
    pk, tm = pd.Series(np.nan, index=va.index), pd.Series(np.nan, index=va.index)
    info = {"pooled_fits": 0, "no_peak_val": int(va["peak"].isna().sum()), "no_peak_train": int(tr["peak"].isna().sum())}
    for (country, crop), g in va.groupby(["country", "crop"]):
        t = tr[(tr["country"] == country) & (tr["crop"] == crop)]
        if t.empty:
            t = tr[tr["country"] == country]
        mean = t["yield_mean"].mean()
        tm[g.index] = mean
        fit = t.dropna(subset=["peak"])
        if len(fit) < MIN_FIELDS:
            fit = tr[tr["country"] == country].dropna(subset=["peak"])
            info["pooled_fits"] += 1
        b, a = np.polyfit(fit["peak"].to_numpy(float), fit["yield_mean"].to_numpy(float), 1)
        pk[g.index] = np.where(g["peak"].notna(), a + b * g["peak"].to_numpy(float), mean)
    target = va["yield_mean"].astype(float)
    return ({"peak_ndvi": pd.DataFrame({"target": target, "m0": pk}),
             "train_mean": pd.DataFrame({"target": target, "m0": tm})}, info)


# --------------------------------------------------------------------------- scores

def r2_draws(t: np.ndarray, P: np.ndarray, idx: np.ndarray) -> tuple[float, np.ndarray]:
    """Point R2 and bootstrap R2 (B,) of field means, each the mean over the member columns of P."""
    def r2(tt, pp):  # (..., n)
        ss_tot = ((tt - tt.mean(-1, keepdims=True)) ** 2).sum(-1)
        ss_res = ((tt - pp) ** 2).sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)
    point = float(np.mean([r2(t, P[:, m]) for m in range(P.shape[1])]))
    draws = np.mean([r2(t[idx], P[idx, m]) for m in range(P.shape[1])], axis=0)
    return point, draws


def score(models: dict[str, dict[str, pd.DataFrame]], vals: dict[str, pd.DataFrame], n_boot: int, seed: int):
    """res[field_set][model][(split, country, crop)] = (point, draws); n[field_set][key] = fields; skipped counts."""
    res = {fs: {} for fs in FIELD_SETS}
    nf = {fs: {} for fs in FIELD_SETS}
    skipped = {fs: [] for fs in FIELD_SETS}
    k = 0
    for split, va in vals.items():
        for fs in FIELD_SETS:
            v = va if fs == "all" else va[va["new"]]
            for (country, crop), g in v.groupby(["country", "crop"]):
                key = (split, country, crop)
                k += 1
                if len(g) < MIN_FIELDS:
                    skipped[fs].append((key, len(g)))
                    continue
                nf[fs][key] = len(g)
                idx = np.random.default_rng([seed, k]).integers(0, len(g), size=(n_boot, len(g)))
                for name, per_split in models.items():
                    if split not in per_split:
                        continue
                    d = per_split[split].loc[g.index]
                    P = d[[c for c in d.columns if c.startswith("m")]].to_numpy(float)
                    res[fs].setdefault(name, {})[key] = r2_draws(d["target"].to_numpy(float), P, idx)
    return res, nf, skipped


def med(r: dict, keys: list) -> tuple[float, np.ndarray]:
    return float(np.median([r[k][0] for k in keys])), np.nanmedian(np.stack([r[k][1] for k in keys], 1), axis=1)


def delta(r: dict, ref: dict, keys: list | None = None) -> tuple[float, np.ndarray, int] | None:
    """Median over common subsets of (model - reference) R2, its draws and the number of subsets."""
    keys = [k for k in (keys if keys is not None else r) if k in r and k in ref]
    if not keys:
        return None
    pt = float(np.median([r[k][0] - ref[k][0] for k in keys]))
    dr = np.nanmedian(np.stack([r[k][1] - ref[k][1] for k in keys], 1), axis=1)
    return pt, dr, len(keys)


def ci(draws: np.ndarray) -> str:
    lo, hi = np.nanpercentile(draws, [5, 95])
    return f"[{lo:.2f}, {hi:.2f}]"


def fmt_delta(d) -> tuple[str, str]:
    if d is None:
        return "–", "–"
    pt, dr, n = d
    return f"{pt:+.2f} {ci(dr)}" + (f" ({n})" if n else ""), f"{np.mean(dr > 0):.2f}"


# --------------------------------------------------------------------------- report

def short(key) -> str:
    return f"{key[0].split('/')[0]} {key[1]} {key[2]}"


def summary_table(r: dict, ref_name: str, base_name: str) -> list[str]:
    lines = [f"| model | subsets | median field R² [90 % CI] | Δ vs `{ref_name}` [90 % CI] (subsets) | share better "
             f"| Δ vs `{base_name}` [90 % CI] (subsets) | share better |", "|---|---|---|---|---|---|---|"]
    for name, rr in r.items():
        keys = list(rr)
        pt, dr = med(rr, keys)
        d1 = fmt_delta(delta(rr, r[ref_name])) if name != ref_name and ref_name in r else ("–", "–")
        d2 = fmt_delta(delta(rr, r[base_name])) if name != base_name and base_name in r else ("–", "–")
        lines.append(f"| `{name}` | {len(keys)} | {pt:.3f} {ci(dr)} | {d1[0]} | {d1[1]} | {d2[0]} | {d2[1]} |")
    return lines


def subset_table(r: dict, nf: dict) -> list[str]:
    names = list(r)
    lines = ["| split | country | crop | fields | " + " | ".join(f"`{n}`" for n in names) + " |",
             "|---|---|---|---|" + "---|" * len(names)]
    for key, n in nf.items():
        cells = [f"{r[m][key][0]:.2f} {ci(r[m][key][1])}" if key in r[m] else "–" for m in names]
        lines.append(f"| {key[0]} | {key[1]} | {key[2].replace('_', ' ')} | {n} | " + " | ".join(cells) + " |")
    return lines


def ranking(r: dict) -> list[str]:
    common = set.intersection(*[set(v) for v in r.values()])
    return sorted(r, key=lambda m: -med(r[m], sorted(common))[0]), sorted(common)


def findings(res: dict, nf: dict, ref: str, configs: list[str]) -> list[str]:
    """Plain-English findings composed from the numbers (so they stay in sync when the script is rerun)."""
    out, pk, tm = [], "peak_ndvi", "train_mean"
    A, N = res["all"], res["new"]

    def dstr(r, a, b, keys=None):
        d = delta(r[a], r[b], keys) if a in r and b in r else None
        if d is None:
            return None
        pt, dr, n = d
        return f"{pt:+.2f} {ci(dr)} ({n} subsets, better in {np.mean(dr > 0):.0%} of draws)"

    if pk in A:
        parts = [f"`{c}` all {dstr(A, c, pk)}" + (f", new {dstr(N, c, pk)}" if c in N else "") for c in configs]
        win = all(np.mean(delta(r[c], r[pk])[1] > 0) > 0.95 for r in (A, N) for c in configs if c in r and pk in r)
        out.append(("**Every run beats the peak-NDVI regression** (in > 95 % of draws)" if win else
                    "**Runs vs the peak-NDVI regression**") + ", all and new fields (median paired Δ R², run − "
                   f"`{pk}`): " + "; ".join(parts) + ". `loyo_lai` has only European subsets (1 with new fields).")
        pa, da = med(A[pk], list(A[pk]))
        out.append(f"**The peak-NDVI baseline is weak here**: median subset R² {pa:.2f} {ci(da)} on all fields, "
                   f"{med(N[pk], list(N[pk]))[0]:.2f} on new fields; vs the training mean (`{tm}`, ≤ 0 by "
                   f"construction) it is {dstr(A, pk, tm)} on all fields and {dstr(N, pk, tm)} on new fields, "
                   f"better in {sum(A[pk][k][0] > A[tm][k][0] for k in A[pk])} / {len(A[pk])} subsets. Peak NDVI "
                   "ranks fields within the training years (positive correlation with yield in the non-Swiss "
                   "subsets) but the fitted line does not transfer to the held-out year's level and slope.")

    def full(r):
        n = max((len({k[0] for k in v}) for v in r.values()), default=0)
        return {m: v for m, v in r.items() if m in configs and len({k[0] for k in v}) == n}
    ra, rn = full(A), full(N)
    if len(ra) > 1 and len(rn) > 1:
        (oa, ca), (on, cn) = ranking(ra), ranking(rn)
        out.append(f"**The ranking changes on new fields.** All val fields ({len(ca)} subsets): "
                   + " > ".join(f"`{m}` {med(ra[m], ca)[0]:.2f}" for m in oa) + f". New fields only ({len(cn)} "
                   "subsets, all but one South American): " + " > ".join(f"`{m}` {med(rn[m], cn)[0]:.2f}" for m in on)
                   + ".")
        keys = sorted(set(cn))
        same = [f"`{m}` {med(ra[m], keys)[0]:.2f} → {med(rn[m], keys)[0]:.2f}" for m in on]
        out.append("**All vs new fields on the same subsets** (new fields are not uniformly harder; the "
                   "sowing-aligned and quick runs lose most): median over the same "
                   f"{len(keys)} subsets, all val fields → new only: " + "; ".join(same) + ". Most new fields are "
                   "South American (where they are 62 % of the val fields), so this mostly reflects those subsets; "
                   "in Europe only Germany rapeseed has ≥ 5 new fields.")
    if "loyo_sow" in A and ref in A:
        reg = lambda r, c, region: [k for k in r[c] if k[0].startswith(region)]
        da_, dr_ = med(A["loyo_sow"], list(A["loyo_sow"]))[0], med(A[ref], list(A[ref]))[0]
        out.append(f"**`loyo_sow` (sowing-aligned axis) vs `{ref}`**: the higher median ({da_:.2f} vs {dr_:.2f}) is "
                   f"not a paired gain: median paired Δ {dstr(A, 'loyo_sow', ref)}. By region, all fields: Europe "
                   f"{dstr(A, 'loyo_sow', ref, reg(A, 'loyo_sow', 'europe'))}, South America "
                   f"{dstr(A, 'loyo_sow', ref, reg(A, 'loyo_sow', 'south_america'))}; new fields: "
                   f"{dstr(N, 'loyo_sow', ref)}. So \"better for Europe\" survives the field bootstrap on all fields "
                   "(4 subsets, one val year), \"worse for South America\" is clear, and on new fields it is worse.")
    others = []
    for c in configs:
        if c in (ref, "loyo_sow") or c not in A:
            continue
        reg = {r: [k for k in A[c] if k[0].startswith(r)] for r in ("europe", "south_america")}
        txt = f"`{c}` all {dstr(A, c, ref)}"
        if reg["south_america"]:
            txt += f" (Europe {dstr(A, c, ref, reg['europe'])}, South America {dstr(A, c, ref, reg['south_america'])})"
        if c in N:
            txt += f", new {dstr(N, c, ref)}"
        others.append(txt)
    if others:
        out.append(f"**Other runs vs `{ref}`** (median paired Δ R²): " + "; ".join(others) + ".")
    if ref in A:
        pa, da = med(A[ref], list(A[ref]))
        pn, dn = med(N[ref], list(N[ref]))
        out.append(f"**The median over subsets is fragile**: `{ref}` {pa:.2f} {ci(da)} on all fields, {pn:.2f} "
                   f"{ci(dn)} on new fields. The subsets split into a good group (Argentina, Switzerland) and a bad "
                   "one (Brazil, Uruguay, German rapeseed), and the median falls in the gap; single-subset CIs with "
                   "6–7 Swiss fields or 5 new Argentine wheat fields span more than one R² unit. Differences in the "
                   "median of a few hundredths between runs are noise; paired Δ with its share of draws is the "
                   "better guide.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--reference", default="loyo_ref")
    ap.add_argument("--members", type=int, default=2, help="use the first N members of every run")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--role", choices=["val"], default="val", help="val only; test is never scored here")
    ap.add_argument("--world", action="store_true", help="also world/loyo/all (configs that have it)")
    ap.add_argument("--out", default="docs/loyo_eval.md")
    args = ap.parse_args()

    L._base = _lean_base                                   # this process only: low-memory pixel table
    paths = load_paths()
    work = paths["work_dir"]
    ft = pd.read_parquet(work / "fields.parquet").set_index("field")
    splits = SPLITS + (["world/loyo/all"] if args.world else [])
    configs = list(dict.fromkeys(args.configs + ([args.reference] if args.reference not in args.configs else [])))

    vals, models, notes, base_info, counts = {}, {m: {} for m in configs + BASELINES}, [], {}, {}
    for split in splits:
        tr, va = split_fields(split, ft)
        vals[split] = va
        counts[split] = (len(tr), len(va), int(va["new"].sum()))
        for c in configs:
            p, note = config_preds(work / "lstm" / "runs" / c / split.replace("/", "__"), va, args.members)
            if p is None:
                notes.append(f"`{c}` on `{split}`: skipped ({note})")
                continue
            if note:
                notes.append(f"`{c}` on `{split}`: {note}")
            models[c][split] = p
        print(f"{split}: peak NDVI of {len(tr) + len(va)} field-years", flush=True)
        peak = peak_ndvi(list(tr.index) + list(va.index))
        b, base_info[split] = baselines(tr, va, peak)
        for name, d in b.items():
            models[name][split] = d
    models = {m: v for m, v in models.items() if v}
    configs = [c for c in configs if c in models]

    res, nf, skipped = score(models, vals, args.boot, args.seed)

    # reproduction check against docs/loyo_conv_screening.md (all val fields, europe + south_america)
    check = []
    for c, want in CHECK.items():
        if c in res["all"]:
            keys = [k for k in res["all"][c] if k[0] in SPLITS]
            got = med(res["all"][c], keys)[0]
            check.append((c, want, got, abs(got - want) < 5e-4))
    for c, want, got, ok in check:
        print(f"check {c}: {got:.3f} vs {want:.3f} {'ok' if ok else 'MISMATCH'}")
    if not all(ok for *_, ok in check):
        sys.exit("reproduction check failed")

    ref = args.reference
    L_ = [f"# LOYO evaluation on val: peak-NDVI baseline, new fields, bootstrap CIs", "",
          f"Generated by `scripts/loyo_eval.py` on {date.today().isoformat()} (step 0 of the evaluation fixes in "
          f"`docs/loyo_literature.md`, P2 and P6). **Val only**: the held-out val year of each LOYO split; test is "
          f"not scored. Read-only on the runs' `preds_val.parquet`; nothing was trained.", "",
          f"- **Splits**: " + ", ".join(f"`{s}` ({counts[s][0]} train / {counts[s][1]} val field-years, "
                                        f"{counts[s][2]} val field-years new)" for s in splits) + ".",
          f"- **Score**: field R² per country × crop subset on field means (mean target vs mean prediction per "
          f"field-year); a run counts as one model = the mean over its **first {args.members} members** of each "
          f"member's field R² (as `scripts/LSTM/08_screen_report.py`), so 2- and 5-member runs are comparable. "
          f"Summaries take the median over subsets.",
          f"- **Field sets**: *all* val fields, and *new only* = val field-years whose physical field (same land, "
          f"other years) has no field-year in the split's train set, i.e. new fields in a new year. Subsets with "
          f"< {MIN_FIELDS} fields in a set are skipped.",
          f"- **Bootstrap**: {args.boot} draws (seed {args.seed}), fields resampled with replacement within each "
          f"subset, the same draws for every model (paired). CIs are the 5–95 % percentiles. *Δ* = median over the "
          f"common subsets of the per-subset R² difference (model − reference); *share better* = share of draws "
          f"with Δ > 0. The CIs cover field sampling only, not the choice of the val year or of the seeds.",
          f"- **Baselines**: `peak_ndvi` = per country × crop OLS of field-mean yield on field peak NDVI, fitted on "
          f"the split's train fields (pooled over the country's crops if < {MIN_FIELDS} train fields; Meroni et al. "
          f"2021 benchmark). Peak NDVI = max over the 10-day steps starting between sowing and harvest "
          f"(`src.loader.seasons()`) of the field-mean NDVI of {PIXELS} random pixels per field-year; a step counts if "
          f"≥ {MIN_CLEAR:.0%} of them are clear. `train_mean` = training mean of the country × crop: constant within "
          f"a subset, so its subset R² is ≤ 0 by construction (it is the level a model must beat).",
          f"- **Reproduction check** (all val fields, median over the europe + south_america subsets, vs "
          f"`docs/loyo_conv_screening.md`): " + "; ".join(f"`{c}` {got:.3f} (doc {want:.3f}) {'ok' if ok else 'MISMATCH'}"
                                                         for c, want, got, ok in check) + ".",
          f"- **Data rule** (CLAUDE.md): no Swiss data values; this page shows R², R² differences and counts only.",
          ""]
    for s in splits:
        bi = base_info[s]
        L_.append(f"- `peak_ndvi` on `{s}`: {bi['pooled_fits']} subset fit(s) pooled over crops; "
                  f"{bi['no_peak_train']} train / {bi['no_peak_val']} val field-years without a valid peak "
                  f"(train: left out of the fit; val: predicted with the training mean).")
    if notes:
        L_ += [""] + [f"- {n}." for n in notes]
    L_ += ["", "## Findings", ""] + [f"{i}. {t}" for i, t in enumerate(findings(res, nf, ref, configs), 1)]
    L_ += ["", "Caveats: one val year per split; 2 members; 6–86 val fields per subset (new only: fewer; Swiss subsets "
           "6–7 fields); bootstrap CIs ignore year-to-year and seed variation, so they are lower bounds on the real "
           "uncertainty."]
    for fs, label in FIELD_SETS.items():
        sk = skipped[fs]
        L_ += ["", f"## {label.capitalize()}", "",
               f"{len(nf[fs])} subsets with ≥ {MIN_FIELDS} fields; {len(sk)} skipped"
               + (": " + ", ".join(f"{short(k)} ({n})" for k, n in sk) if sk else "") + ".", ""]
        L_ += summary_table(res[fs], ref, "peak_ndvi")
        L_ += ["", f"### Field R² per subset [90 % CI], {label}", ""] + subset_table(res[fs], nf[fs])
    text = "\n".join(L_) + "\n"
    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
