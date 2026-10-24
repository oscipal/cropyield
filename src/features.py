"""Registry of the loader's input features: one place that says what a feature is, its unit and where it comes from.

A feature list is a list of group specs, each `group` or `group:argument`:

    s2                      12 Sentinel-2 bands, surface reflectance (DN / 10000)            dynamic
    s2:B04,B08              selected bands
    indices                 ndvi, ndre, ndmi from the reflectances; indices:ndvi           dynamic
    observed                1 where the pixel has a clear S2 observation in the window      dynamic
    weather                 default weather set (WEATHER_DEFAULT); weather:tmean_c,prec_mm;  dynamic, per field
                            weather:all
    terrain                 6 terrain features (scripts/LSTM/14_build_terrain.py); terrain:slope,elevation
    soil                    8 SoilGrids properties 0-30 cm in physical units; soil:<spec> (src/soil.py)
    coords                  coord_x / y / z as delivered (location; encoding undocumented; leaks the region under LORO)
    crop, country           one-hot (categories of the whole dataset, so train / val / test agree)

Dynamic features have one value per time step of the grid, static ones one per pixel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src import soil

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
INDICES = {"ndvi": ("B08", "B04"), "ndre": ("B08", "B05"), "ndmi": ("B08", "B11")}  # (a - b) / (a + b)
WEATHER_UNITS = {"tmean_c": "degC", "tmax_c": "degC", "tmin_c": "degC", "prec_mm": "mm", "gdd": "degC d",
                 "gdd_cum": "degC d", "prec_cum_mm": "mm", "frost_days": "d", "heat_days": "d", "rh_mean": "%",
                 "srad": "as delivered (likely Wh m-2 d-1)", "vpd_kpa": "kPa", "season_frac": "0-1"}
# rh_mean, srad, vpd_kpa are missing for all Brazilian and most Argentine fields; season_frac is a mask
WEATHER_DEFAULT = ["tmean_c", "tmax_c", "tmin_c", "prec_mm", "gdd", "gdd_cum", "prec_cum_mm", "frost_days",
                   "heat_days"]
TERRAIN_UNITS = {"elevation": "m", "rel_elevation": "m", "slope": "deg", "aspect_sin": "-1..1 (east +)",
                 "aspect_cos": "-1..1 (north +)", "curvature": "1/km (convex +)"}
COORDS = ["coord_x", "coord_y", "coord_z"]

# group -> (source, what it is, known issues); used by scripts/features_doc.py for docs/features.md
GROUP_DOCS = {
    "s2": ("`<work>/lstm/data_w<days>/dyn.npy`, `scripts/LSTM/11_build_s2_grid.py` from the raw scenes",
           "Sentinel-2 L2A surface reflectance (DN / 10000) of the best scene per window (fewest SCL 8 + 9 cloud "
           "pixels on the field raster, ties to the latest); only SCL 4 / 5 (vegetation, bare soil) pixels kept. "
           "B10 is not part of L2A.",
           "Windows without a scene or with the pixel clouded are NaN. Lower-resolution bands (20 m, 60 m) are "
           "upsampled to 10 m in the delivered rasters."),
    "indices": ("computed by the loader from the reflectances",
                "Normalised differences (a - b) / (a + b): NDVI = (B08 - B04), NDRE = (B08 - B05), NDMI = (B08 - "
                "B11); clipped to [-1, 1].", "NaN where the S2 bands are NaN."),
    "observed": ("computed by the loader", "1 where the pixel has a clear S2 observation in the window, else 0.",
                 "Never NaN; models should get it whenever NaN inputs are filled."),
    "weather": ("`<work>/lstm/data_w<days>/weather.npy`, `scripts/LSTM/13_build_weather_grid.py` from the daily "
                "ERA5 CSVs", "Daily ERA5 aggregated over the same calendar windows as S2: means of temperature, "
                "sums of precipitation and GDD (sum of max(0, Tmean - Tbase), Tbase 0 °C for wheat / winter "
                "cereal / rapeseed, 10 °C for corn / soybean), cumulative GDD / precipitation since sowing at the "
                "window end, frost days (Tmin < 0 °C), heat days (Tmax > 30 °C). One value per field, the same "
                "for all its pixels.",
                "NaN outside the CSV record (sowing - 60 d .. harvest + 60 d); cumulative values NaN before "
                "sowing. rh_mean, srad, vpd_kpa are missing for all Brazilian and most Argentine fields (not in "
                "the default set). Replaces the delivered weather bands, which were sums between selected scenes "
                "in K and m (mostly the gap length)."),
    "terrain": ("`<work>/lstm/terrain.npy`, `scripts/LSTM/14_build_terrain.py` from the raw `dem/dem-*.tif`",
                "Elevation from the raw SRTM DEM; slope, aspect and curvature from a Gaussian-weighted local "
                "quadratic fit (σ 1.5 px ≈ 30 m) per pixel; rel_elevation = elevation minus the field median.",
                "Replaces the delivered slope (unknown unit), curvature (clipped, integers in Brazil) and TWI "
                "(missing for most pixels; not recomputed, needs the upstream catchment). The delivered elevation "
                "is 0 for 408 Brazilian pixels; here it is the raw DEM value. The Brazilian DEM is integer metres "
                "in 30 m blocks, so its slopes are coarser."),
    "soil": ("`<work>/lstm/data/static.npy` (= raw `soil/*.tif`), converted by `src/soil.py`",
             "SoilGrids 2.0 at 250 m: 8 properties at 6 standard depths, converted from mapped units, averaged "
             "over the requested depth weighted by layer thickness (default 0-30 cm).",
             "Near-constant within a field (250 m). Uncertainty layers are not offered (unit undocumented, "
             "extreme tails)."),
    "coords": ("`<work>/lstm/data/static.npy`", "Pixel location as delivered (coord_x / y / z, values in [-1, 1]).",
               "Encoding undocumented and not a plain unit sphere (Brazil and Uruguay, both southern hemisphere, "
               "have opposite signs of coord_z). Identifies the region: leaks it under LORO splits; opt-in only."),
    "crop": ("`meta.parquet`", "One-hot crop over all crops in the dataset.",
             "Swiss winter_cereal is barley and triticale together (per the dataset README)."),
    "country": ("`meta.parquet`", "One-hot country over all countries in the dataset.",
                "Under LORO a held-out country's column is all zeros in training."),
}
INDEX_DOCS = {"ndvi": "vegetation greenness", "ndre": "red-edge chlorophyll", "ndmi": "canopy water content"}
WEATHER_DOCS = {"tmean_c": "mean daily mean temperature", "tmax_c": "mean daily maximum temperature",
                "tmin_c": "mean daily minimum temperature", "prec_mm": "precipitation sum",
                "gdd": "growing degree days in the window", "gdd_cum": "GDD since sowing (window end)",
                "prec_cum_mm": "precipitation since sowing (window end)", "frost_days": "days with Tmin < 0 °C",
                "heat_days": "days with Tmax > 30 °C", "rh_mean": "mean relative humidity",
                "srad": "mean daily solar radiation", "vpd_kpa": "mean vapour pressure deficit",
                "season_frac": "share of the window between sowing and harvest"}
TERRAIN_DOCS = {"elevation": "height above sea level", "rel_elevation": "height relative to the field median "
                "(low / high spots)", "slope": "steepness", "aspect_sin": "east (+1) / west (-1) facing",
                "aspect_cos": "north (+1) / south (-1) facing; both 0 on flat ground (< 0.1°)",
                "curvature": "convex (crest, +) / concave (hollow, -), minus the Laplacian"}
CATEGORICAL = ("crop", "country")
DYNAMIC_GROUPS = ("s2", "indices", "observed", "weather")
STATIC_GROUPS = ("terrain", "soil", "coords") + CATEGORICAL


@dataclass
class Group:
    """One resolved group spec: the feature names it produces and how to compute them."""
    group: str
    names: list[str]
    units: list[str]
    dynamic: bool
    arg: object = None           # group-specific: band names, weather columns, soil columns, categories
    extra: dict = field(default_factory=dict)


def _pick(arg: str | None, available: list[str], default: list[str], what: str) -> list[str]:
    if not arg:
        return list(default)
    if arg == "all":
        return list(available)
    names = [a.strip() for a in arg.split(",")]
    unknown = [n for n in names if n not in available]
    if unknown:
        raise KeyError(f"unknown {what} {unknown}; available: {available}")
    return names


def resolve(features: list[str], static_names: list[str], categories: dict[str, list[str]]) -> list[Group]:
    """Feature specs -> resolved groups, in the given order; duplicate feature names are an error."""
    out: list[Group] = []
    for spec in features:
        g, _, arg = spec.partition(":")
        arg = arg or None
        if g == "s2":
            b = _pick(arg, S2_BANDS, S2_BANDS, "S2 bands")
            out.append(Group(g, b, ["reflectance"] * len(b), True, b))
        elif g == "indices":
            i = _pick(arg, list(INDICES), list(INDICES), "indices")
            out.append(Group(g, i, ["-1..1"] * len(i), True, i))
        elif g == "observed":
            out.append(Group(g, ["observed"], ["0/1"], True))
        elif g == "weather":
            w = _pick(arg, list(WEATHER_UNITS), WEATHER_DEFAULT, "weather features")
            out.append(Group(g, w, [WEATHER_UNITS[n] for n in w], True, w))
        elif g == "terrain":
            t = _pick(arg, list(TERRAIN_UNITS), list(TERRAIN_UNITS), "terrain features")
            out.append(Group(g, t, [TERRAIN_UNITS[n] for n in t], False, t))
        elif g == "soil":
            cols = soil.columns(arg, static_names)
            out.append(Group(g, [c[0] for c in cols], [c[2] for c in cols], False, cols))
        elif g == "coords":
            out.append(Group(g, list(COORDS), ["as delivered"] * 3, False, [static_names.index(c) for c in COORDS]))
        elif g in CATEGORICAL:
            cats = categories[g]
            out.append(Group(g, [f"{g}={c}" for c in cats], ["one-hot"] * len(cats), False, cats))
        else:
            raise KeyError(f"unknown feature group {g!r}; known: {list(DYNAMIC_GROUPS + STATIC_GROUPS)}")
    names = [n for grp in out for n in grp.names]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"features listed twice: {dup}")
    return out
