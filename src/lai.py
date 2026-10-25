"""Leaf area index from Sentinel-2 surface reflectance with the SNAP / S2ToolBox biophysical processor network.

The network (8 bands + 3 angle cosines -> 5 tansig neurons -> 1 linear) is the one of the ESA S2ToolBox
"Biophysical Processor" (Weiss & Baret, ATBD v1.1, 2016), trained on PROSAIL simulations. Weights and
normalisation ranges copied from Sentinel Hub's script
https://github.com/sentinel-hub/custom-scripts/blob/main/sentinel-2/lai/script.js (fetched 2026-10-25).

Angles are not in the data, so:
- sun zenith from date and latitude at 10:30 local solar time (Sentinel-2 descending node);
- view zenith 5° (near nadir; S2 swath gives 0-11°), relative azimuth cos = 0 (middle of the training range).
`scripts/check_lai.py` checks the reference spectra and how much these choices move LAI.
"""
from __future__ import annotations

import numpy as np

BANDS = ["B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
BAND_RANGE = np.array([[0, 0.253061520471542], [0, 0.290393577911328], [0, 0.305398915248555],
                       [0.006637972542253, 0.608900395797889], [0.013972727018939, 0.753827384322927],
                       [0.026690138082061, 0.782011770669178], [0.016388074192258, 0.493761397883092],
                       [0, 0.493025984460231]], dtype=np.float32)
VIEW_ZEN_RANGE = (0.918595400582046, 1.0)        # cos
SUN_ZEN_RANGE = (0.342022871159208, 0.936206429175402)
LAI_RANGE = (0.000319182538301, 14.4675094548151)
# hidden layer: rows = neurons; columns = bias, 8 bands, view zenith, sun zenith, relative azimuth
W1 = np.array([
    [4.96238030555279, -0.023406878966470, 0.921655164636366, 0.135576544080099, -1.938331472397950,
     -3.342495816122680, 0.902277648009576, 0.205363538258614, -0.040607844721716, -0.083196409727092,
     0.260029270773809, 0.284761567218845],
    [1.416008443981500, -0.132555480856684, -0.139574837333540, -1.014606016898920, -1.330890038649270,
     0.031730624503341, -1.433583541317050, -0.959637898574699, 1.133115706551000, 0.216603876541632,
     0.410652303762839, 0.064760155543506],
    [1.075897047213310, 0.086015977724868, 0.616648776881434, 0.678003876446556, 0.141102398644968,
     -0.096682206883546, -1.128832638862200, 0.302189102741375, 0.434494937299725, -0.021903699490589,
     -0.228492476802263, -0.039460537589826],
    [1.533988264655420, -0.109366593670404, -0.071046262972729, 0.064582411478320, 2.906325236823160,
     -0.673873108979163, -3.838051868280840, 1.695979344531530, 0.046950296081713, -0.049709652688365,
     0.021829545430994, 0.057483827104091],
    [3.024115930757230, -0.089939416159969, 0.175395483106147, -0.081847329172620, 2.219895367487790,
     1.713873975136850, 0.713069186099534, 0.138970813499201, -0.060771761518025, 0.124263341255473,
     0.210086140404351, -0.183878138700341]], dtype=np.float64)
W2 = np.array([1.096963107077220, -1.500135489728730, -0.096283269121503, -0.194935930577094,
               -0.352305895755591, 0.075107415847473], dtype=np.float64)
VIEW_ZENITH_DEG = 5.0
COS_REL_AZIMUTH = 0.0
OVERPASS_SOLAR_HOUR = 10.5
LAI_MAX = 8.0          # SNAP flags LAI > 8 as out of range; clipped here


def _norm(x, lo, hi):
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def cos_sun_zenith(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """cos of the solar zenith at OVERPASS_SOLAR_HOUR local solar time; lat (...,), doy broadcastable."""
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284.0 + doy) / 365.0)
    h = np.deg2rad((OVERPASS_SOLAR_HOUR - 12.0) * 15.0)
    lat = np.deg2rad(lat_deg)
    return np.clip(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(h), 0.0, 1.0)


def snap_lai(refl: np.ndarray, band_index: list[int], cos_sz: np.ndarray, cos_vz: float | None = None,
             cos_raz: float = COS_REL_AZIMUTH) -> np.ndarray:
    """LAI (m2 m-2) for reflectances refl (..., n_bands) with the SNAP bands at band_index; cos_sz broadcastable
    to refl[..., 0]. NaN where any band is NaN; clipped to [0, LAI_MAX]."""
    cos_vz = np.cos(np.deg2rad(VIEW_ZENITH_DEG)) if cos_vz is None else cos_vz
    b = refl[..., band_index].astype(np.float32)
    x = _norm(b, BAND_RANGE[:, 0], BAND_RANGE[:, 1])
    vz = np.float32(_norm(cos_vz, *VIEW_ZEN_RANGE))
    sz = _norm(np.broadcast_to(cos_sz, b.shape[:-1]).astype(np.float32), *SUN_ZEN_RANGE)
    w = W1.astype(np.float32)
    s = (w[:, 0] + x @ w[:, 1:9].T + vz * w[:, 9] + sz[..., None] * w[:, 10] + np.float32(cos_raz) * w[:, 11])
    n = np.tanh(s)                                    # tansig(x) = 2 / (1 + e^-2x) - 1 = tanh(x)
    out = W2[0] + n @ W2[1:].astype(np.float32)
    lai = 0.5 * (out + 1.0) * (LAI_RANGE[1] - LAI_RANGE[0]) + LAI_RANGE[0]
    return np.clip(lai, 0.0, LAI_MAX).astype(np.float32)
