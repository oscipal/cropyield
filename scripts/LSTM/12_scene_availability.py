"""How much of each field's season the Sentinel-2 cube really sees: scene availability per field-year and window.

    python scripts/LSTM/12_scene_availability.py                       # 30-day grid from day 0 (24 steps)
    python scripts/LSTM/12_scene_availability.py --limit 20            # quick test, no doc
    python scripts/LSTM/12_scene_availability.py --show Germany_DUP3_farm1_field23_wheat_2017   # one field, no output
    python scripts/LSTM/12_scene_availability.py --check-dyn 5         # chosen valid share vs lstm/data/dyn.npy

Grid and scene days as in 11_build_s2_grid.py (day = date - 1 Jan of the year before the harvest year - 1, window k
= [start_day + k * days, start_day + (k + 1) * days)). Per window:
    n_scenes          scenes in the window (n_scenes_in_season: of those, between sowing and harvest)
    chosen_*          the scene the YieldSAT rule picks (fewest SCL 8 + 9 on the raster, ties to the latest),
                      its date and the shares of the field's labelled pixels (rows/cols of meta.parquet) with
                      SCL 4/5 (chosen_valid), cloud 8/9, cirrus 10, shadow 3, snow 11, dark/unclassified 2/7,
                      nodata/defect 0/1
    best_valid        the highest SCL 4/5 share over all scenes in the window
    in_season         the window overlaps [sowing, harvest) (metadata-<field>.json: seeding_date, harvesting_date)
    class             out_of_season | no_scene | all_unusable (best < 5 %) | rule_loss (chosen < 5 %, best >= 50 %)
                      | valid (chosen >= 50 %) | partly_valid (the rest)
    dominant_cause    the largest non-valid share of the chosen scene (cloud, cirrus, shadow, snow, dark, nodata,
                      water), "none" without a scene or when the chosen scene is fully valid

Outputs: <work_dir>/scene_availability/windows_w<days>.parquet (one row per field-year x window, only shares and
counts) and docs/scene_availability.md (aggregates per country x crop x calendar month). Only counts and aggregate
shares are printed; per-field output (--show, --check-dyn) refuses Swiss fields (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_paths  # noqa: E402

_spec = importlib.util.spec_from_file_location("s2grid", Path(__file__).with_name("11_build_s2_grid.py"))
g11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g11)

USABLE, VALID = 0.05, 0.50
# non-valid SCL groups of the chosen scene; the order breaks ties for dominant_cause
CAUSES = {"cloud": g11.CLOUD, "cirrus": (10,), "shadow": (3,), "snow": (11,), "dark": (2, 7), "nodata": (0, 1),
          "water": (6,)}
CLASSES = ["out_of_season", "no_scene", "all_unusable", "rule_loss", "partly_valid", "valid"]


def season_days(field_dir: Path, jan1) -> tuple[float, float]:
    """Sowing and harvest day in the scenes() convention, NaN if a date is missing."""
    meta = json.loads((field_dir / f"metadata-{field_dir.name}.json").read_text())
    out = []
    for k in ("seeding_date", "harvesting_date"):
        v = meta.get(k)
        out.append((datetime.strptime(v, "%d.%m.%Y").date() - jan1).days - 1 if v else np.nan)
    return out[0], out[1]


def field_windows(field_dir: Path, rows: np.ndarray, cols: np.ndarray, days: int, start: int, n_times: int
                  ) -> pd.DataFrame:
    """One row per window of one field-year (columns as in the module docstring, without field/country/crop/year)."""
    jan1, sc = g11.scenes(field_dir)
    sow, harvest = season_days(field_dir, jan1)
    lo = start if np.isnan(sow) else sow
    hi = start + n_times * days if np.isnan(harvest) else harvest
    n_pix = len(rows)
    rec = {k: dict(n=0, n_season=0, best=-1.0, chosen=None) for k in range(n_times)}
    for day, _, scl_path in sc:                                   # one SCL raster in memory at a time
        if not start <= day < start + n_times * days:
            continue
        k = (day - start) // days
        with rasterio.open(scl_path) as r:
            scl = r.read(1)
        clouds = int(np.isin(scl, g11.CLOUD).sum())
        cnt = np.bincount(scl[rows, cols].ravel(), minlength=256)
        shares = {c: cnt[list(v)].sum() / n_pix for c, v in CAUSES.items()}
        valid = cnt[list(g11.CLEAR)].sum() / n_pix
        w = rec[k]
        w["n"] += 1
        w["n_season"] += int(lo <= day < hi)
        w["best"] = max(w["best"], valid)
        # same rule as g11.build_field: fewest cloud pixels, ties to the later scene (sorted by day)
        if w["chosen"] is None or (-clouds, day) >= (-w["chosen"][0], w["chosen"][1]):
            w["chosen"] = (clouds, day, valid, shares)
    out = []
    for k, w in rec.items():
        a, b = start + k * days, start + (k + 1) * days
        in_season = bool(a < hi and b > lo)                       # overlaps [sowing, harvest)
        row = dict(step=k, window_start_day=a, window_end_day=b,
                   window_start_date=jan1 + timedelta(days=a + 1), window_end_date=jan1 + timedelta(days=b + 1),
                   sow_day=sow, harvest_day=harvest, in_season=in_season, n_scenes=w["n"],
                   n_scenes_in_season=w["n_season"], chosen_date=None, chosen_valid=np.nan,
                   best_valid=np.nan, **{f"chosen_{c}": np.nan for c in CAUSES})
        cause = "none"
        if w["chosen"] is not None:
            _, day, valid, shares = w["chosen"]
            row.update(chosen_date=jan1 + timedelta(days=day + 1), chosen_valid=valid, best_valid=w["best"],
                       **{f"chosen_{c}": s for c, s in shares.items()})
            top = max(shares, key=shares.get)                     # first of equal maxima, in CAUSES order
            cause = top if shares[top] > 0 else "none"
        if not in_season:
            cls = "out_of_season"
        elif w["chosen"] is None:
            cls = "no_scene"
        elif row["best_valid"] < USABLE:
            cls = "all_unusable"
        elif row["chosen_valid"] < USABLE and row["best_valid"] >= VALID:
            cls = "rule_loss"
        elif row["chosen_valid"] >= VALID:
            cls = "valid"
        else:
            cls = "partly_valid"
        row.update(**{"class": cls, "dominant_cause": cause})
        out.append(row)
    return pd.DataFrame(out)


def _work(task):
    """Pool worker; g11._init has put rows, cols and the grid into g11._G."""
    i, d, a, b = task
    G = g11._G
    try:
        return i, field_windows(d, G["rows"][a:b], G["cols"][a:b], G["days"], G["start"], G["n_times"]), None
    except Exception as e:                                         # noqa: BLE001  (reported as a count)
        return i, None, f"{type(e).__name__}"


# --------------------------------------------------------------------------- inputs

def load_fields(paths: dict, meta_path: Path) -> pd.DataFrame:
    """field_dirs() of 11_build_s2_grid plus crop and year. To save RAM, only the first meta row of each field is
    turned into pandas (rows from viz/field_index.parquet, written by the viewer's index build); start/stop are
    taken from there and checked against the field names in meta."""
    idx = pd.read_parquet(paths["work_dir"] / "viz" / "field_index.parquet", columns=["field", "start", "end"])
    idx = idx.sort_values("start", ignore_index=True)
    cols = ["country", "field", "crop", "year"]
    t = pq.read_table(meta_path, columns=cols, read_dictionary=cols)
    n = t.num_rows
    assert (idx["start"].to_numpy()[1:] == idx["end"].to_numpy()[:-1]).all() and idx["end"].iloc[-1] == n
    first = t.take(idx["start"].to_numpy()).to_pandas()
    last = t.take(idx["end"].to_numpy() - 1).column("field").to_pylist()
    del t
    first = first.astype(str)
    assert (first["field"] == idx["field"]).all() and last == idx["field"].tolist(), "field_index out of date"
    fields = g11.field_dirs(paths, first[["country", "field"]])
    fields["start"], fields["stop"] = idx["start"].to_numpy(), idx["end"].to_numpy()
    fields["crop"], fields["year"] = first["crop"].to_numpy(), first["year"].to_numpy()
    return fields


def run(fields: pd.DataFrame, meta_path: Path, days: int, start: int, n_times: int, workers: int) -> pd.DataFrame:
    tasks = [(i, r.dir, r.start, r.stop) for i, r in fields.iterrows()]
    parts, errors, t0 = [], {}, time.time()
    with Pool(workers, g11._init, (meta_path, None, days, start, n_times)) as pool:
        for j, (i, df, err) in enumerate(pool.imap_unordered(_work, tasks, chunksize=2), 1):
            if err:
                errors[err] = errors.get(err, 0) + 1
            else:
                r = fields.loc[i]
                df.insert(0, "year", r.year)
                df.insert(0, "crop", r.crop)
                df.insert(0, "country", r.country)
                df.insert(0, "field", r.field)
                parts.append(df)
            if j % 50 == 0 or j == len(tasks):
                el = time.time() - t0
                print(f"  {j:,}/{len(tasks):,} field-years ({el:.0f} s, ETA {el / j * (len(tasks) - j):.0f} s)",
                      flush=True)
    if errors:
        print(f"field-years that failed (skipped): {errors}")
    w = pd.concat(parts, ignore_index=True)
    for c in ("window_start_date", "window_end_date", "chosen_date"):
        w[c] = pd.to_datetime(w[c])
    for c in ("country", "crop", "class", "dominant_cause"):
        w[c] = w[c].astype("category")
    return w.sort_values(["country", "field", "step"], ignore_index=True)


# --------------------------------------------------------------------------- report

def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{100 * x:.0f}"


def headline(s: pd.DataFrame) -> pd.DataFrame:
    """Per country x crop over in-season windows."""
    g = s.groupby(["country", "crop"], observed=True)
    h = pd.DataFrame({
        "field-years": g["field"].nunique(),
        "in-season windows": g.size(),
        "scenes / window": g["n_scenes"].mean().round(1),
        "no_scene %": g.apply(lambda d: (d["class"] == "no_scene").mean(), include_groups=False),
        "usable % (chosen >= 5 %)": g.apply(lambda d: (d["chosen_valid"] >= USABLE).mean(), include_groups=False),
        "valid % (chosen >= 50 %)": g.apply(lambda d: (d["chosen_valid"] >= VALID).mean(), include_groups=False),
        "best >= 5 % (perfect rule)": g.apply(lambda d: (d["best_valid"] >= USABLE).mean(), include_groups=False),
        "best >= 50 %": g.apply(lambda d: (d["best_valid"] >= VALID).mean(), include_groups=False),
        "rule_loss %": g.apply(lambda d: (d["class"] == "rule_loss").mean(), include_groups=False),
    })
    for c in h.columns[3:]:                                       # shares -> %
        h[c] = h[c].map(pct)
    return h.reset_index()


def monthly(s: pd.DataFrame) -> pd.DataFrame:
    """Per country x crop x calendar month of the window start: share of in-season windows per class (%),
    dominant cause among windows without a usable observation (chosen < 5 %), with its share."""
    s = s.assign(month=s["window_start_date"].dt.month)
    keys = ["country", "crop", "month"]
    n = s.groupby(keys, observed=True).size().rename("windows")
    cls = pd.crosstab([s[k] for k in keys], s["class"], normalize="index")
    cls = cls.reindex(columns=[c for c in CLASSES[1:]], fill_value=0.0).map(pct)
    bad = s[(s["chosen_valid"] < USABLE) & s["chosen_date"].notna()]
    cause = bad.groupby(keys, observed=True)["dominant_cause"].agg(
        lambda x: f"{x.value_counts().index[0]} ({pct(x.value_counts(normalize=True).iloc[0])} %)")
    t = pd.concat([n, cls, cause.rename("dominant cause (unusable)")], axis=1)
    t["dominant cause (unusable)"] = t["dominant cause (unusable)"].fillna("")
    t = t.reset_index()
    t["month"] = pd.to_datetime(t["month"], format="%m").dt.strftime("%b")
    return t


def md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in df.itertuples(index=False))


def write_doc(w: pd.DataFrame, path: Path, days: int, start: int, n_times: int, out_path: Path) -> None:
    s = w[w["in_season"]]
    h = headline(s)
    usable = (s["chosen_valid"] >= USABLE).mean()
    best_usable = (s["best_valid"] >= USABLE).mean()
    n_rl = int((s["class"] == "rule_loss").sum())
    n_lost = int(((s["chosen_valid"] < USABLE) & (s["best_valid"] >= USABLE)).sum())
    rl_fields = s.loc[s["class"] == "rule_loss", "field"].nunique()
    valid, best_valid = (s["chosen_valid"] >= VALID).mean(), (s["best_valid"] >= VALID).mean()
    rl_causes = s.loc[s["class"] == "rule_loss", "dominant_cause"].value_counts(normalize=True)
    causes = s.loc[(s["chosen_valid"] < USABLE) & s["chosen_date"].notna(), "dominant_cause"] \
        .value_counts(normalize=True)
    cmd = (f"python scripts/LSTM/12_scene_availability.py --window-days {days} --start-day {start} "
           f"--n-times {n_times}")
    text = f"""# Sentinel-2 scene availability per season

Written by `scripts/LSTM/12_scene_availability.py`. Grid: {days}-day windows from day {start}, {n_times} steps
(day 0 = 1 January of the year before the harvest year, as the Sentinel-2 cube in `lstm/data`). Per-window data:
`{out_path}`.

Reproduce:

```bash
source /scratch2/oscipal/cropyield/bin/activate
{cmd}
```

## How to read

- A window is **in season** when it overlaps sowing..harvest (`seeding_date`, `harvesting_date` of the field's
  metadata). Only in-season windows are counted below.
- **valid share** of a scene = share of the field's labelled pixels with SCL 4 or 5 (vegetation, bare soil); only
  these pixels get values in the cube. The **chosen** scene is the one the YieldSAT rule puts in the cube: fewest
  SCL 8 + 9 (cloud) pixels on the whole raster, ties to the latest. **best** = the highest valid share of any scene
  in the window.
- Classes: **no_scene** (no Sentinel-2 scene in the window), **all_unusable** (best < 5 %), **rule_loss** (chosen
  < 5 % although best >= 50 %: the rule threw away a good scene), **valid** (chosen >= 50 %), **partly_valid** (the
  rest: chosen 5–50 %, or chosen < 5 % with best 5–50 %).
- **usable** = the chosen scene has >= 5 % valid pixels on the field, i.e. the model sees something in that window.
- **dominant cause** = the largest non-valid SCL group on the field in the chosen scene: cloud (8, 9), cirrus (10),
  shadow (3), snow (11), dark/unclassified (2, 7; mostly haze and low sun), nodata (0, 1), water (6).
- Shares are % of in-season windows. All numbers are aggregates; Switzerland is included only as aggregates.

## Headline

{usable * 100:.0f} % of {len(s):,} in-season windows ({s['field'].nunique():,} field-years) have a usable
observation (chosen >= 5 %); with a perfect selection rule it would be {best_usable * 100:.0f} %
(best >= 5 %).

The rule loses a usable scene (chosen < 5 %, best >= 5 %) in {n_lost:,} windows ({100 * n_lost / len(s):.2f} %);
the strict rule_loss class (best >= 50 %) has {n_rl:,} windows ({100 * n_rl / len(s):.2f} %) in {rl_fields:,}
field-years.

In the rule_loss windows the chosen scene is mostly lost to {", ".join(f"{k} {100 * v:.0f} %" for k, v in rl_causes.items() if v >= 0.005)}
(share of rule_loss windows by dominant cause of the chosen scene). The rule counts only SCL 8 + 9 on the whole
raster, so a scene with cirrus (10), haze (2, 7) or snow over the field, or clouds only over the field, can win.
A "most SCL 4/5 pixels on the field" rule (a variant of `11_build_s2_grid.py`) would raise the usable share from
{usable * 100:.0f} % to {best_usable * 100:.0f} % and the valid share (>= 50 %) from {valid * 100:.0f} % to
{best_valid * 100:.0f} %.

Dominant cause in windows without a usable chosen scene: {", ".join(f"{k} {100 * v:.0f} %" for k, v in causes.items() if v >= 0.005)}.

{md(h)}
## Per country × crop × month of the window start

Columns are % of the in-season windows that start in that month.

{md(monthly(s))}"""
    path.write_text(text)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- checks (non-Swiss fields only)

def show(fields, meta_path, name, days, start, n_times):
    r = fields[fields["field"] == name]
    if r.empty:
        raise SystemExit(f"unknown field {name}")
    r = r.iloc[0]
    if r.country == "Switzerland":
        raise SystemExit("per-field output is not allowed for Swiss fields (CLAUDE.md)")
    m = pd.read_parquet(meta_path, columns=["row", "col"]).iloc[r.start:r.stop]
    w = field_windows(r.dir, m["row"].to_numpy(), m["col"].to_numpy(), days, start, n_times)
    cols = ["step", "window_start_date", "in_season", "n_scenes", "chosen_date", "chosen_valid", "best_valid",
            "chosen_cloud", "chosen_snow", "chosen_dark", "class", "dominant_cause"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(w[cols].to_string(index=False))


def check_dyn(fields, meta_path, data_dir, n, seed=0):
    """chosen_valid (30-day grid from day 0) vs share of finite B04 per window in lstm/data/dyn.npy."""
    ref = np.load(data_dir / "dyn.npy", mmap_mode="r")
    b = json.loads((data_dir / "info.json").read_text())["dynamic_bands"].index("B04")
    pick = fields[fields["country"] != "Switzerland"].groupby("country").sample(n, random_state=seed)
    m = pd.read_parquet(meta_path, columns=["row", "col"])
    worst = 0.0
    for r in pick.itertuples():
        w = field_windows(r.dir, m["row"].to_numpy()[r.start:r.stop], m["col"].to_numpy()[r.start:r.stop],
                          30, 0, 24)
        fin = np.isfinite(np.asarray(ref[r.start:r.stop, :, b])).mean(0)
        got = w["chosen_valid"].fillna(0).to_numpy()
        d = np.abs(got - fin).max()
        worst = max(worst, d)
        print(f"  {r.country:10s} {r.field:50s} windows with a scene {w['chosen_date'].notna().sum():2d}  "
              f"max |chosen_valid - finite B04| {d:.5f}")
    print(f"check-dyn: worst difference {worst:.5f}")


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--start-day", type=int, default=0)
    ap.add_argument("--n-times", type=int, default=None, help=f"default: up to day {g11.END_DAY}")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N field-years only (test; no doc)")
    ap.add_argument("--show", default=None, help="print the windows of one non-Swiss field, no output")
    ap.add_argument("--check-dyn", type=int, default=0, help="compare N non-Swiss field-years per country "
                                                             "with lstm/data/dyn.npy (30-day grid)")
    args = ap.parse_args()
    days, start = args.window_days, args.start_day
    n_times = args.n_times or (g11.END_DAY - start) // days

    paths = load_paths()
    data_dir = paths["work_dir"] / "lstm" / "data"
    meta_path = data_dir / "meta.parquet"
    fields = load_fields(paths, meta_path)
    print(f"{len(fields):,} field-years; grid: start day {start}, {days}-day windows, {n_times} steps", flush=True)

    if args.show:
        return show(fields, meta_path, args.show, days, start, n_times)
    if args.check_dyn:
        return check_dyn(fields, meta_path, data_dir, args.check_dyn)
    if args.limit:
        fields = fields.iloc[:args.limit]

    t0 = time.time()
    w = run(fields, meta_path, days, start, n_times, args.workers)
    print(f"done in {time.time() - t0:.0f} s: {len(w):,} windows, {w['field'].nunique():,} field-years")
    print("missing sowing / harvest date (field-years):",
          int(w.groupby("field")["sow_day"].first().isna().sum()),
          int(w.groupby("field")["harvest_day"].first().isna().sum()))
    print("class counts:", w["class"].value_counts().to_dict())

    out_dir = paths["work_dir"] / "scene_availability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"windows_w{days}_limit{args.limit}.parquet" if args.limit else f"windows_w{days}.parquet")
    if start != 0 and not args.limit:
        out_path = out_dir / f"windows_w{days}_s{start}.parquet"
    w.to_parquet(out_path, index=False)
    print(f"wrote {out_path}")
    if not args.limit:
        doc = paths["docs_dir"] / ("scene_availability.md" if (days, start) == (30, 0)
                                   else f"scene_availability_w{days}_s{start}.md")
        write_doc(w, doc, days, start, n_times, out_path)


if __name__ == "__main__":
    main()
