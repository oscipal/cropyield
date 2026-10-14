"""Small QGIS example: one split, one year -> work_dir/split_example_<scope>_<method>_<crop>_<year>.gpkg.

Reads split_shapes.gpkg (scripts/09_export_split_shapes.py), keeps the field-years of one year and copies the
layer with its style. Usage: python scripts/10_export_split_example.py [scope method crop year]
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_paths  # noqa: E402


def main() -> None:
    scope, method, crop, year = (sys.argv[1:5] + ["world", "loro", "all", "2018"])[:4] if len(sys.argv) > 4 \
        else ("world", "loro", "all", "2018")
    work = load_paths()["work_dir"]
    name = f"{scope}_{method}_{crop}"
    g = gpd.read_file(work / "split_shapes.gpkg", layer=name)
    g = g[g["year"].astype(str) == str(year)]
    out = work / f"split_example_{name}_{year}.gpkg"
    out.unlink(missing_ok=True)
    g.to_file(out, layer=name, driver="GPKG")
    src = sqlite3.connect(work / "split_shapes.gpkg")
    row = src.execute("SELECT f_geometry_column, styleName, styleQML, description FROM layer_styles "
                      "WHERE f_table_name=?", (name,)).fetchone()
    con = sqlite3.connect(out)
    con.execute("""CREATE TABLE layer_styles (id INTEGER PRIMARY KEY AUTOINCREMENT, f_table_catalog TEXT,
        f_table_schema TEXT, f_table_name TEXT, f_geometry_column TEXT, styleName TEXT, styleQML TEXT,
        styleSLD TEXT, useAsDefault INTEGER, description TEXT, owner TEXT, ui TEXT,
        update_time DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    con.execute("INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, f_geometry_column, "
                "styleName, styleQML, styleSLD, useAsDefault, description, owner, ui) "
                "VALUES ('', '', ?, ?, ?, ?, '', 1, ?, '', '')", (name, *row[:3], row[3]))
    con.commit()
    print(g.groupby(["set", "crop"]).size().unstack(fill_value=0))
    print(f"wrote {out} ({len(g)} field-years)")


if __name__ == "__main__":
    main()
