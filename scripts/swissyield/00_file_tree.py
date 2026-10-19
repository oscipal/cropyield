"""File-tree profile of the SwissYield folder (names, counts, sizes only: no file is opened)."""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from swiss_lib import PATHS, RAW, Report


def main() -> None:
    root = PATHS["swissyield"]
    r = Report("00_file_tree", "SwissYield file tree")
    r.p(f"Root: `{root}`\n\nLayout: `Raw/Switzerland/<field_shared_name>/{{metadata-*.json, s2_images, scl_masks, dem, soil, "
        "weather, yield_masks}`, `Preprocessed/Switzerland[_DUP1|_DUP2]/merge_s2-soil-dem-weather-coords.nc`. "
        "field_shared_name = `Switzerland_<DUP1|DUP2>_<farm>_field<n>_<crop>_<year>`.")

    r.h("Preprocessed files")
    r.table(pd.DataFrame([{"file": str(p.relative_to(root)), "GB": p.stat().st_size / 1e9}
                          for p in sorted((root / "Preprocessed").rglob("*.nc"))]))

    fields = sorted(p for p in RAW.iterdir() if p.is_dir())
    rows, kinds = [], Counter()
    for f in fields:
        m = re.match(r"Switzerland_(DUP\d)_(farm\d)_field(\d+)_(.+)_(\d{4})$", f.name)
        files = [p for p in f.rglob("*") if p.is_file()]
        sub = Counter(p.parent.name if p.parent != f else "." for p in files)
        for p in files:
            kinds[(p.parent.name if p.parent != f else ".", re.sub(r"\d+", "#", p.stem.split("-")[0]), p.suffix)] += 1
        s2 = sorted(p.stem[-8:] for p in (f / "s2_images").glob("*.tif"))
        rows.append({"field": f.name, "source": m[1], "farm": m[2], "crop": m[4], "year": int(m[5]),
                     "s2_scenes": sub["s2_images"], "scl_masks": sub["scl_masks"], "dem_files": sub["dem"],
                     "soil_files": sub["soil"], "first_scene": s2[0] if s2 else "", "last_scene": s2[-1] if s2 else "",
                     "MB": sum(p.stat().st_size for p in files) / 1e6})
    t = pd.DataFrame(rows)
    t.to_csv(PATHS["work_dir"] / "swissyield" / "tables" / "raw_file_tree.csv", index=False)
    r.h(f"Raw: {len(t)} field-years, {int(t.s2_scenes.sum())} S2 scenes, {t.MB.sum() / 1e3:.1f} GB")
    r.table(t.groupby(["source", "farm", "crop"]).agg(field_years=("field", "size"), scenes=("s2_scenes", "sum"),
                                                       MB=("MB", "sum")))
    r.table(t.groupby(["source", "year"]).size().unstack(fill_value=0))
    r.h("File kinds per field-year (folder, name pattern, extension, total count)")
    r.table(pd.DataFrame([{"folder": a, "pattern": b, "ext": c, "count": n} for (a, b, c), n in sorted(kinds.items())]))
    r.h("Consistency")
    r.p(f"scenes == SCL masks everywhere: {bool((t.s2_scenes == t.scl_masks).all())}; "
        f"dem files per field: {sorted(t.dem_files.unique())}; soil files per field: {sorted(t.soil_files.unique())}")
    r.h("Scenes per field-year")
    r.table(t.groupby("source")["s2_scenes"].describe())
    r.save()


if __name__ == "__main__":
    main()
