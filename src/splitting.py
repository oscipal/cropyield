"""Single train/val/test splits chosen to be representative of their subset.

Unit of assignment is the field-year (never pixels inside a field). Roles: 0 train, 1 val, 2 test.

* loyo: test = one year, val = another year, train = the rest. Exhaustive over all year pairs.
* loro: test and val are sets of whole regions (farms, merged where they share a physical field),
  so every year of a physical field stays on one side. Seeded random search.
* cv (cross-validation over years, cv_folds): one split per year of CV_YEARS[scope], group <scope>/cv<year>/all;
  test = that year, val = the next CV year (cyclic), train = all other years. Every CV year is test once and val once.

Both minimise, summed over val and test,
    SIZE_WEIGHT * |field-year share - target|
    + TV distance of the crop x country mix vs the whole subset
    + TV distance of the year mix (loro only; in loyo the year is the held-out factor)
    + KS distance of field-mean yield standardised per crop and country,
subject to train containing every crop of the subset.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

ROLES = {"train": 0, "val": 1, "test": 2}
SCOPES = {
    "germany": ["Germany"],  # legacy: the original suite without Switzerland
    "europe": ["Germany", "Switzerland"],
    "south_america": ["Argentina", "Brazil", "Uruguay"],
    "world": ["Germany", "Switzerland", "Argentina", "Brazil", "Uruguay"],
}
MIN_REGIONS = 3  # loro needs a train, a val and a test region; crops with fewer regions cannot be split
TARGET = 0.15
SIZE_WEIGHT = 2.0
# Cross-validation over years (user, 2026-10-30): only years with enough data are test (and val) years: more than 200
# field-years and >= 6 country x crop groups with >= CV_MIN_FIELDS fields (Switzerland not counted). 2016 (Germany
# only), 2017 and 2024 stay in the training set of every fold.
CV_YEARS = {"world": ["2018", "2019", "2020", "2021", "2022", "2023"]}
CV_MIN_FIELDS = 10  # a country x crop x year is scored only with at least this many field-years


def subset(fields: pd.DataFrame, scope: str, crop: str, method: str | None = None) -> pd.DataFrame:
    """Field-years of a scope and crop. For loro, crops present in fewer than MIN_REGIONS regions of the scope
    (Swiss barley/triticale, which exist on one farm only) are left out: they cannot be split by region."""
    sub = fields[fields["country"].isin(SCOPES[scope])]
    if method == "loro":
        n_regions = sub.groupby("crop")["region"].nunique()
        sub = sub[sub["crop"].isin(n_regions[n_regions >= MIN_REGIONS].index)]
    if crop != "all":
        sub = sub[sub["crop"] == crop]
    return sub.reset_index(drop=True)


class Scorer:
    """Vectorised representativeness score for a boolean field mask."""

    def __init__(self, sub: pd.DataFrame, use_year: bool):
        self.n = len(sub)
        self.use_year = use_year
        self.mix = pd.factorize(sub["crop"] + "|" + sub["country"])[0]
        self.year = pd.factorize(sub["year"])[0]
        self.crop = pd.factorize(sub["crop"])[0]
        self.mix_full = np.bincount(self.mix) / self.n
        self.year_full = np.bincount(self.year) / self.n
        # standardise per crop and country: yield levels differ strongly between countries (DE vs SA wheat)
        grp = sub.groupby(["crop", "country"])["yield_mean"]
        z = (sub["yield_mean"] - grp.transform("mean")) / grp.transform("std").fillna(1.0).replace(0, 1.0)
        self.order = np.argsort(z.to_numpy(), kind="stable")
        self.ecdf_full = np.arange(1, self.n + 1) / self.n

    def components(self, mask: np.ndarray) -> dict:
        k = int(mask.sum())
        mix = np.bincount(self.mix[mask], minlength=len(self.mix_full)) / k
        out = {"size": abs(k / self.n - TARGET),
               "mix_tv": 0.5 * float(np.abs(mix - self.mix_full).sum()),
               "yield_ks": float(np.abs(np.cumsum(mask[self.order]) / k - self.ecdf_full).max())}
        if self.use_year:
            yr = np.bincount(self.year[mask], minlength=len(self.year_full)) / k
            out["year_tv"] = 0.5 * float(np.abs(yr - self.year_full).sum())
        return out

    @staticmethod
    def total(c: dict) -> float:
        return SIZE_WEIGHT * c["size"] + sum(v for key, v in c.items() if key != "size")

    def train_has_all_crops(self, train: np.ndarray) -> bool:
        return len(np.unique(self.crop[train])) == len(np.unique(self.crop))

    def evaluate(self, role: np.ndarray) -> tuple[float, dict] | None:
        val, test = role == 1, role == 2
        if not val.any() or not test.any() or not self.train_has_all_crops(role == 0):
            return None
        cv, ct = self.components(val), self.components(test)
        comps = {**{f"val_{k}": v for k, v in cv.items()}, **{f"test_{k}": v for k, v in ct.items()}}
        return self.total(cv) + self.total(ct), comps


def loyo_split(sub: pd.DataFrame) -> tuple[np.ndarray, dict]:
    scorer = Scorer(sub, use_year=False)
    years = sorted(sub["year"].unique())
    year = sub["year"].to_numpy()
    best = None
    for test_y, val_y in itertools.permutations(years, 2):
        role = np.where(year == test_y, 2, np.where(year == val_y, 1, 0)).astype(np.int8)
        res = scorer.evaluate(role)
        if res and (best is None or res[0] < best[0]):
            best = (res[0], res[1], role, test_y, val_y)
    if best is None:
        raise ValueError("no valid LOYO split (train would miss a crop for every year pair)")
    score, comps, role, test_y, val_y = best
    return role, {"method": "loyo", "test_years": [test_y], "val_years": [val_y],
                  "score": score, **comps, "n_candidates": len(years) * (len(years) - 1)}


def loro_split(sub: pd.DataFrame, n_trials: int = 20_000, seed: int = 42) -> tuple[np.ndarray, dict]:
    scorer = Scorer(sub, use_year=True)
    reg_codes, reg_names = pd.factorize(sub["region"])
    if len(reg_names) < 3:
        raise ValueError(f"only {len(reg_names)} regions; need at least 3")
    sizes = np.bincount(reg_codes)
    goal = TARGET * len(sub)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_trials):
        reg_role = np.zeros(len(reg_names), dtype=np.int8)
        filled = {1: 0, 2: 0}
        for r in rng.permutation(len(reg_names)):
            # put the region where it brings a set closest to its target; else train
            gains = {s: abs(filled[s] - goal) - abs(filled[s] + sizes[r] - goal) for s in (2, 1)}
            s = max(gains, key=gains.get)
            if gains[s] > 0:
                reg_role[r] = s
                filled[s] += sizes[r]
        role = reg_role[reg_codes]
        res = scorer.evaluate(role)
        if res and (best is None or res[0] < best[0]):
            best = (res[0], res[1], role, reg_role)
    if best is None:
        raise ValueError("no valid LORO split found")
    score, comps, role, reg_role = best
    return role, {"method": "loro", "test_regions": sorted(reg_names[reg_role == 2].tolist()),
                  "val_regions": sorted(reg_names[reg_role == 1].tolist()),
                  "score": score, **comps, "n_trials": n_trials, "seed": seed}


def cv_folds(sub: pd.DataFrame, years: list[str]) -> list[tuple[str, np.ndarray, dict]]:
    """Cross-validation over years: (method name "cv<year>", role, info) per test year; val = the next year of
    `years` (cyclic), train = every other year of sub (also the years outside `years`)."""
    year = sub["year"].astype(str).to_numpy()
    scorer = Scorer(sub, use_year=False)
    out = []
    for i, test_y in enumerate(years):
        val_y = years[(i + 1) % len(years)]
        role = np.where(year == test_y, 2, np.where(year == val_y, 1, 0)).astype(np.int8)
        res = scorer.evaluate(role)
        if res is None:
            raise ValueError(f"cv fold {test_y}: train misses a crop or val / test is empty")
        out.append((f"cv{test_y}", role, {"method": "cv", "test_years": [test_y], "val_years": [val_y],
                                          "cv_years": list(years), "score": res[0], **res[1]}))
    return out
