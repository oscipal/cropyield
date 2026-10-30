"""Sentinel-2 cube on a thermal-time (GDD) axis: windows of equal growing degree days instead of calendar days
(world variant wv_gdd, `inputs.grid: gdd`; literature P1 / P3 in docs/loyo_literature.md: Desloires 2023).

    python scripts/LSTM/20_build_gdd_grid.py        # -> <work_dir>/lstm/data_gdd/

Per field-year: GDD accumulate from a fixed start day per hemisphere x crop, GDD = sum over days of
max(0, Tmean - Tbase) with the ERA5-Land daily mean 2 m temperature of the field's cell (lstm/daily/era5land.npy,
scripts/LSTM/16_build_era5land_grid.py --grids daily); Tbase 0 degC for wheat / winter cereal / rapeseed, 10 degC for
corn / soybean (as src/features.py). Window k holds the scenes whose accumulated GDD lies in [k W, (k + 1) W),
N_STEPS windows; per window the scene with the largest clear share on the field (ties: the later one) from the scene
store (scripts/LSTM/19_build_s2_scenes.py: scenes with >= 5 % clear pixels, clear pixels only). Scenes before the
start day or beyond the last window are not used.

Rule per hemisphere x crop, from the reported dates of the non-Swiss field-years only (CLAUDE.md: Swiss data are
not inspected; Swiss wheat and winter cereal use the European wheat rule, which comes from the German fields):
    start = 5th percentile of the sowing day minus 30 days (pre-season context), rounded down to 10 days, >= 210
    W     = 95th percentile of the GDD between start and harvest / (N_STEPS - 1), so that the last window still
            contains the harvest of 95 % of the field-years
The sowing dates enter only through this percentile, so the axis does not depend on a field's own (possibly
estimated) sowing date.

Outputs in <work_dir>/lstm/data_gdd/ (pixel rows as lstm/data/meta.parquet): dyn.npy uint16 (pixels, N_STEPS, 12),
65535 = missing; info.json (n_times, window_days 1, start_day 0 so that the loader's step index is the window index;
the rules); meta.parquet / static.npy symlinks. Only counts and the rules are printed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_paths  # noqa: E402
from src.loader import seasons  # noqa: E402

N_STEPS = 17                      # as the 30-day grid of the base model (days 210-720)
TBASE = {"wheat": 0.0, "winter_cereal": 0.0, "rapeseed": 0.0, "corn": 10.0, "soybean": 10.0}
HEMI = {"Germany": "europe", "Switzerland": "europe", "Argentina": "south_america", "Brazil": "south_america",
        "Uruguay": "south_america"}
EXCLUDED = ("Switzerland",)       # not used to derive the rules (CLAUDE.md)
MISSING = np.uint16(65535)


def main() -> None:
    work = load_paths()["work_dir"]
    lstm = work / "lstm"
    sdir, out = lstm / "scenes", lstm / "data_gdd"
    out.mkdir(exist_ok=True)
    sinfo = json.loads((sdir / "info.json").read_text())
    values = np.memmap(sdir / "values.u16", dtype=np.uint16, mode="r", shape=tuple(sinfo["shape"]))
    fy = pd.read_parquet(sdir / "fields.parquet").set_index("field")
    sc = pd.read_parquet(sdir / "scenes.parquet")
    meta = pd.read_parquet(lstm / "data" / "meta.parquet", columns=["field", "country", "crop"])
    first = meta.groupby("field", sort=False, observed=True).first()
    fy = fy.join(first[["country", "crop"]].astype(str))
    fy["hemi"] = fy["country"].map(HEMI)
    s = seasons().reindex(fy.index)
    dj = json.loads((lstm / "daily" / "era5land.json").read_text())
    t2m = np.load(lstm / "daily" / "era5land.npy")[..., dj["features"].index("t2m_c")]   # (n_fy, days)
    drow = pd.Series(np.arange(len(dj["fields"])), index=dj["fields"]).reindex(fy.index).to_numpy()
    d0 = dj["start_day"]
    n_days = t2m.shape[1]

    def cum_gdd(field: str, start: int) -> np.ndarray:
        """Accumulated GDD at the end of every day of the daily axis (0 before `start`; missing days add 0)."""
        g = np.maximum(np.nan_to_num(t2m[drow[fy.index.get_loc(field)]], nan=0.0) - TBASE[fy.at[field, "crop"]], 0.0)
        g[: max(start - d0, 0)] = 0.0
        return np.cumsum(g)

    # rules per hemisphere x crop from the non-Swiss field-years
    rules = {}
    for (hemi, crop), g in fy.groupby(["hemi", "crop"]):
        ref = g[~g["country"].isin(EXCLUDED)]
        if ref.empty:
            continue
        sow, harv = s.loc[ref.index, "sow_day"].dropna(), s.loc[ref.index, "harvest_day"]
        start = max(d0, int(np.floor((np.percentile(sow, 5) - 30) / 10) * 10))
        at_harvest = [cum_gdd(f, start)[int(np.clip(harv[f] - d0, 0, n_days - 1))] for f in ref.index
                      if pd.notna(harv[f])]
        w = float(np.percentile(at_harvest, 95)) / (N_STEPS - 1)
        rules[f"{hemi}|{crop}"] = {"start_day": start, "gdd_per_window": round(w, 1), "tbase_c": TBASE[crop],
                                   "n_reference_field_years": int(len(ref))}
    for f_hemi, crop in (("europe", "winter_cereal"),):           # Swiss only: use the European wheat rule
        rules.setdefault(f"{f_hemi}|{crop}", {**rules[f"{f_hemi}|wheat"], "from": f"{f_hemi}|wheat"})
    for k, r in rules.items():
        print(f"  {k:28s} start day {r['start_day']}, {r['gdd_per_window']} GDD per window (Tbase {r['tbase_c']}), "
              f"{r.get('from', str(r['n_reference_field_years']) + ' reference field-years')}", flush=True)

    n_pix = int(fy["stop"].max())
    dyn = np.lib.format.open_memmap(out / "dyn.npy", mode="w+", dtype=np.uint16, shape=(n_pix, N_STEPS, 12))
    filled, n_windows, used = 0, 0, 0
    by_field = dict(tuple(sc.groupby("field")))
    for field, r in fy.iterrows():
        rule = rules[f"{r['hemi']}|{r['crop']}"]
        P = r["stop"] - r["start"]
        block = np.full((P, N_STEPS, 12), MISSING, dtype=np.uint16)
        g = by_field.get(field)
        if g is not None and r["n_scenes"]:
            cg = cum_gdd(field, rule["start_day"])
            day = g["day"].to_numpy()
            ok = day >= rule["start_day"]
            k = np.full(len(day), -1)
            k[ok] = np.floor(cg[np.clip(day[ok] - d0, 0, n_days - 1)] / rule["gdd_per_window"]).astype(int)
            k[k >= N_STEPS] = -1
            cube = np.asarray(values[r["offset"]:r["offset"] + P * r["n_scenes"]]).reshape(P, r["n_scenes"], 12)
            for w in np.unique(k[k >= 0]):
                cand = np.flatnonzero(k == w)
                best = cand[np.lexsort((g["day"].to_numpy()[cand], g["clear_share"].to_numpy()[cand]))[-1]]
                block[:, w] = cube[:, g["k"].to_numpy()[best]]
                filled += 1
            used += int((k >= 0).sum())
        n_windows += N_STEPS
        dyn[r["start"]:r["stop"]] = block
    dyn.flush()
    for name in ("meta.parquet", "static.npy"):
        link = out / name
        if not link.exists():
            link.symlink_to(lstm / "data" / name)
    (out / "info.json").write_text(json.dumps({
        "dynamic_bands": sinfo["bands"], "n_pixels": n_pix, "n_times": N_STEPS, "window_days": 1, "start_day": 0,
        "missing_value": int(MISSING), "axis": "gdd", "rules": rules, "tbase_c": TBASE,
        "source": "scripts/LSTM/20_build_gdd_grid.py"}, indent=1))
    print(f"done: {len(fy)} field-years, {filled:,} of {n_windows:,} windows filled ({filled / n_windows:.0%}), "
          f"{used:,} scenes on the axis", flush=True)


if __name__ == "__main__":
    main()
