"""Lazy access to the preprocessed YieldSAT NetCDF.

The German file is ~7 GB float32, so nothing here loads the full feature array.
Features are read in contiguous pixel blocks and filtered in memory.
"""
from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import xarray as xr

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
CATEGORICAL_VARS = [
    "crop",
    "year",
    "farm_identifier",
    "field_shared_name",
    "seeding_date_type",
    "seeding_date",
    "harvesting_date",
]
PIXEL_VARS = ["row", "col", "target"]
RENAME = {"farm_identifier": "farm", "field_shared_name": "field"}
FEATURE_VAR = "sample"

_MAPPING_KEYS = ("mapping", "categories", "labels", "classes", "encoding", "codes", "dictionary", "values")
_BAND_NAME_KEYS = ("bands", "band_names", "features", "feature_names", "channels", "variables")


# --------------------------------------------------------------------------- opening

def open_dataset(path: str | Path) -> xr.Dataset:
    """Open lazily (no dask needed: the netCDF backend only reads indexed slices)."""
    return xr.open_dataset(path, cache=False)


def pixel_dim(ds: xr.Dataset) -> str:
    return ds["target"].dims[0]


def iter_blocks(n: int, block: int) -> Iterator[tuple[int, int]]:
    for start in range(0, n, block):
        yield start, min(start + block, n)


# --------------------------------------------------------------------------- categorical decoding

def _to_str(v) -> str:
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8")
    return str(v)


def _is_int(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, (int, np.integer)):
        return True
    if isinstance(x, (str, bytes, np.str_, np.bytes_)):
        s = _to_str(x).strip()
        return s.lstrip("-").isdigit()
    return False


def _is_range(xs) -> bool:
    vals = sorted(int(_to_str(x)) for x in xs)
    return vals == list(range(vals[0], vals[0] + len(vals))) and vals[0] in (0, 1)


def _parse_mapping(raw) -> dict[int, str] | None:
    """Normalise a mapping attribute to {code: label}. Accepts code->label and label->code."""
    if isinstance(raw, (str, bytes, np.str_, np.bytes_)):
        s = _to_str(raw)
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(s)
                break
            except (ValueError, SyntaxError, TypeError):
                continue
        else:
            return None
        if isinstance(obj, (str, bytes)):
            return None
        return _parse_mapping(obj)
    if isinstance(raw, (list, tuple, np.ndarray)):
        return {i: _to_str(v) for i, v in enumerate(raw)}
    if not isinstance(raw, dict) or not raw:
        return None

    keys, vals = list(raw.keys()), list(raw.values())
    keys_int, vals_int = all(_is_int(k) for k in keys), all(_is_int(v) for v in vals)
    if keys_int and vals_int:
        # Both sides integer (e.g. year codes). The side that is a 0/1-based range is the code.
        if _is_range(vals) and not _is_range(keys):
            return {int(_to_str(v)): _to_str(k) for k, v in raw.items()}
        return {int(_to_str(k)): _to_str(v) for k, v in raw.items()}
    if keys_int:
        return {int(_to_str(k)): _to_str(v) for k, v in raw.items()}
    if vals_int:
        return {int(_to_str(v)): _to_str(k) for k, v in raw.items()}
    return None


def get_mapping(ds: xr.Dataset, var: str) -> dict[int, str]:
    """Find the code->label mapping of a categorical variable in its attrs (or the global attrs)."""
    attrs = ds[var].attrs
    for key in _MAPPING_KEYS:
        if key in attrs and (m := _parse_mapping(attrs[key])):
            return m
    if "flag_values" in attrs and "flag_meanings" in attrs:
        meanings = _to_str(attrs["flag_meanings"]).split()
        return {int(v): m for v, m in zip(np.atleast_1d(attrs["flag_values"]), meanings)}
    # attrs themselves are the mapping (a single entry only if its key is a code, e.g. {'0': 'soybean'})
    if (len(attrs) > 1 or all(_is_int(k) for k in attrs)) and (m := _parse_mapping(dict(attrs))):
        return m
    int_valued = {k: v for k, v in attrs.items() if _is_int(v) and not isinstance(v, str)}
    if len(int_valued) > 1 and (m := _parse_mapping(int_valued)):
        return m
    for key in (f"{var}_mapping", f"{var}_categories", f"{var}_labels", var):
        if key in ds.attrs and (m := _parse_mapping(ds.attrs[key])):
            return m
    raise KeyError(f"No mapping found for '{var}'. Variable attrs: {list(attrs)}")


def decode(ds: xr.Dataset, var: str, codes: np.ndarray) -> np.ndarray:
    mapping = get_mapping(ds, var)
    labels = pd.Series(codes).map(mapping)
    missing = labels.isna() & pd.Series(codes).notna()
    if missing.any():
        unknown = np.unique(np.asarray(codes)[missing.to_numpy()])[:10]
        warnings.warn(f"{var}: {missing.sum()} values without mapping, e.g. {unknown}")
    return labels.to_numpy()


def load_meta(ds: xr.Dataset) -> pd.DataFrame:
    """Pixel-level metadata table (small: all 1-D variables over the pixel dimension)."""
    pdim = pixel_dim(ds)
    cols: dict[str, np.ndarray] = {}
    for var in CATEGORICAL_VARS:
        if var not in ds.variables:
            warnings.warn(f"'{var}' not in dataset")
            continue
        if ds[var].dims != (pdim,):
            warnings.warn(f"'{var}' has dims {ds[var].dims}, expected ({pdim},); skipped")
            continue
        codes = ds[var].values
        name = RENAME.get(var, var)
        cols[f"{name}_code"] = codes
        try:
            cols[name] = decode(ds, var, codes)
        except KeyError as e:
            warnings.warn(str(e))
            cols[name] = codes
    for var in PIXEL_VARS:
        if var in ds.variables and ds[var].dims == (pdim,):
            cols[var] = ds[var].values
    meta = pd.DataFrame(cols)
    meta.index.name = "pixel"
    if "field" in meta and "year" in meta:
        meta["field_year"] = meta["field"].astype(str) + "_" + meta["year"].astype(str)
    return meta


# --------------------------------------------------------------------------- feature access

def _decode_names(values) -> list[str]:
    return [_to_str(v) for v in np.asarray(values).ravel()]


class FeatureReader:
    """Uniform block reader for features stored either

    * stacked: one variable ``sample`` with dims (pixel, time, band) in any order, band names in a
      coordinate or attribute, or
    * separate: one variable per band with dims (pixel, time) or (pixel,).
    """

    def __init__(self, ds: xr.Dataset, var: str = FEATURE_VAR):
        self.ds = ds
        self.pdim = pixel_dim(ds)
        self.n_pixels = ds.sizes[self.pdim]
        if var in ds.data_vars and ds[var].ndim == 3:
            self.mode = "stacked"
            self.da = ds[var]
            self.bdim, self.names = self._find_band_dim(ds, self.da)
            self.tdim = next(d for d in self.da.dims if d not in (self.pdim, self.bdim))
        else:
            self.mode = "separate"
            skip = set(CATEGORICAL_VARS) | set(PIXEL_VARS) | {"times"}
            self.names = [
                v for v in ds.data_vars
                if v not in skip and ds[v].dims and ds[v].dims[0] == self.pdim and ds[v].ndim in (1, 2)
            ]
            two_d = [v for v in self.names if ds[v].ndim == 2]
            self.tdim = ds[two_d[0]].dims[1] if two_d else None
        self.n_times = ds.sizes[self.tdim] if self.tdim else 1

    def _find_band_dim(self, ds: xr.Dataset, da: xr.DataArray) -> tuple[str, list[str]]:
        other = [d for d in da.dims if d != self.pdim]
        for d in other:
            if d in ds.coords and ds[d].dtype.kind in "OSU":
                return d, _decode_names(ds[d].values)
        for source in (da.attrs, ds.attrs):
            for key in _BAND_NAME_KEYS:
                if key in source:
                    raw = source[key]
                    names = _parse_mapping(raw)
                    names = [names[k] for k in sorted(names)] if names else _to_str(raw).replace(",", " ").split()
                    for d in other:
                        if ds.sizes[d] == len(names):
                            return d, names
        raise KeyError(f"Cannot identify band names for '{da.name}' with dims {da.dims}")

    def __repr__(self) -> str:
        return f"FeatureReader(mode={self.mode}, n_pixels={self.n_pixels}, n_times={self.n_times}, n_features={len(self.names)})"

    def read(self, names: list[str], start: int, stop: int) -> np.ndarray:
        """Return float32 array (n, time, len(names)) for pixels [start, stop)."""
        missing = [n for n in names if n not in self.names]
        if missing:
            raise KeyError(f"Features not found: {missing}")
        sl = {self.pdim: slice(start, stop)}
        if self.mode == "stacked":
            idx = [self.names.index(n) for n in names]
            order = np.argsort(idx)
            sorted_idx = [idx[i] for i in order]
            arr = self.da.isel({**sl, self.bdim: sorted_idx}).transpose(self.pdim, self.tdim, self.bdim).values
            arr = arr[..., np.argsort(order)]
        else:
            parts = []
            for n in names:
                a = self.ds[n].isel(sl).values
                parts.append(np.repeat(a[:, None], self.n_times, axis=1) if a.ndim == 1 else a)
            arr = np.stack(parts, axis=-1)
        return arr.astype(np.float32, copy=False)

    def extract(self, names: list[str], mask: np.ndarray, out: np.ndarray | None = None,
                block: int = 100_000, verbose: bool = True) -> np.ndarray:
        """Read only the pixels where ``mask`` is True. ``out`` may be a pre-allocated memmap."""
        n_sel = int(mask.sum())
        if out is None:
            out = np.empty((n_sel, self.n_times, len(names)), dtype=np.float32)
        pos = 0
        for start, stop in iter_blocks(self.n_pixels, block):
            m = mask[start:stop]
            if not m.any():
                continue
            arr = self.read(names, start, stop)[m]
            out[pos:pos + len(arr)] = arr
            pos += len(arr)
            if verbose:
                print(f"  read pixels {start:,}-{stop:,}: {pos:,}/{n_sel:,} selected", flush=True)
        assert pos == n_sel
        return out
