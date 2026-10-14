"""Export the split suite as a GeoPackage for QGIS: work_dir/split_shapes.gpkg.

One layer per split (scope_method_crop, 24 layers). Each holds the field-years that belong to that split with
a ``set`` column (train / val / test) plus crop, year, region, farm, yield_mean. Footprints come from
field_shapes.gpkg (scripts/08_export_field_shapes.py), simplified to ~15 m.

Each layer gets a default QGIS style (stored in the gpkg, loaded automatically): colour = set, hatch pattern =
crop.
"""
from __future__ import annotations

import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_paths  # noqa: E402
from src.splits import load_split  # noqa: E402

SIMPLIFY_DEG = 1.5e-4
SET_COLOUR = {"train": "76,120,168", "val": "242,169,0", "test": "214,39,40"}
CROP_BRUSH = {"corn": "solid", "soybean": "b_diagonal", "wheat": "f_diagonal", "rapeseed": "cross"}


def split_keys(path: Path) -> list[tuple[str, str, str]]:
    root = zarr.open_group(str(path), mode="r")
    return sorted((s, m, c) for s, sg in root.groups() for m, mg in sg.groups() for c, _ in mg.groups())


def qml() -> str:
    rules, symbols = [], []
    for i, (key, (sset, crop)) in enumerate(((f"{s}_{c}", (s, c)) for s in SET_COLOUR for c in CROP_BRUSH)):
        rgb = SET_COLOUR[sset]
        rules.append(f'<rule filter="&quot;set&quot; = \'{sset}\' AND &quot;crop&quot; = \'{crop}\'" '
                     f'key="{{{uuid.uuid4()}}}" symbol="{i}" label="{sset} / {crop}"/>')
        symbols.append(
            f'<symbol type="fill" name="{i}" alpha="1" clip_to_extent="1" force_rhr="0"><layer class="SimpleFill" '
            f'enabled="1" pass="0" locked="0"><Option type="Map">'
            f'<Option name="color" value="{rgb},255" type="QString"/>'
            f'<Option name="style" value="{CROP_BRUSH[crop]}" type="QString"/>'
            f'<Option name="outline_color" value="{rgb},255" type="QString"/>'
            f'<Option name="outline_style" value="solid" type="QString"/>'
            f'<Option name="outline_width" value="0.5" type="QString"/>'
            f'<Option name="outline_width_unit" value="MM" type="QString"/>'
            f'</Option></layer></symbol>')
    return ('<!DOCTYPE qgis><qgis version="3.34.0"><renderer-v2 type="RuleRenderer" symbollevels="0" '
            'enableorderby="0" forceraster="0"><rules key="{' + str(uuid.uuid4()) + '}">' + "".join(rules) +
            '</rules><symbols>' + "".join(symbols) + '</symbols></renderer-v2></qgis>')


def embed_style(gpkg: Path, layers: list[str]) -> None:
    style = qml()
    con = sqlite3.connect(gpkg)
    con.execute("""CREATE TABLE IF NOT EXISTS layer_styles (id INTEGER PRIMARY KEY AUTOINCREMENT,
        f_table_catalog TEXT, f_table_schema TEXT, f_table_name TEXT, f_geometry_column TEXT, styleName TEXT,
        styleQML TEXT, styleSLD TEXT, useAsDefault INTEGER, description TEXT, owner TEXT, ui TEXT,
        update_time DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    for name in layers:
        geom = con.execute("SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?", (name,)).fetchone()[0]
        con.execute("""INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, f_geometry_column,
            styleName, styleQML, styleSLD, useAsDefault, description, owner, ui, update_time)
            VALUES ('', '', ?, ?, 'set_colour_crop_hatch', ?, '', 1, 'colour = set, hatch = crop', '', '', ?)""",
                    (name, geom, style, now))
    con.commit()
    con.close()


def main() -> None:
    paths = load_paths()
    work = paths["work_dir"]
    shapes = gpd.read_file(work / "field_shapes.gpkg", layer="all")
    shapes["geometry"] = shapes.geometry.simplify(SIMPLIFY_DEG, preserve_topology=True)
    shapes = shapes.set_index("field")[["geometry", "country", "crop", "year", "farm", "region", "yield_mean"]]
    out = work / "split_shapes.gpkg"
    out.unlink(missing_ok=True)
    layers = []
    for scope, method, crop in split_keys(paths["split_suite"]):
        s = load_split(paths["split_suite"], scope, method, crop)
        parts = [shapes.loc[s[f"{k}_fields"]].assign(set=k) for k in ("train", "val", "test")]
        g = gpd.GeoDataFrame(gpd.pd.concat(parts).rename_axis("field").reset_index(), crs=4326)
        name = f"{scope}_{method}_{crop}"
        g.to_file(out, layer=name, driver="GPKG")
        layers.append(name)
        print(name, g["set"].value_counts().to_dict())
    embed_style(out, layers)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
