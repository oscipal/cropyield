"""Sowing and harvest dates of a field-year from its raw metadata-<field>.json, on the day axis of the S2 cubes.

Day convention as scripts/LSTM/11_build_s2_grid.py:scenes(): day = (date - 1 January of the year before the harvest
year).days - 1, so 30-day window k covers days [30 k, 30 (k + 1)). The harvest year comes from harvesting_date (the
label `year` is off by one in a few field-years).
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path


def _parse(s: str | None) -> date | None:
    return datetime.strptime(s, "%d.%m.%Y").date() if s else None


def day_of(d: date, jan1: date) -> int:
    return (d - jan1).days - 1


def read_season(field_dir: Path) -> dict:
    """seeding/harvest dates, seeding_date_type, day 0 (jan1) and sow_day/harvest_day of one field-year folder."""
    meta = json.loads((field_dir / f"metadata-{field_dir.name}.json").read_text())
    sow, harvest = _parse(meta.get("seeding_date")), _parse(meta.get("harvesting_date"))
    jan1 = date((harvest.year if harvest else int(meta["year"])) - 1, 1, 1)
    return {"seeding_date": sow, "harvesting_date": harvest, "seeding_date_type": meta.get("seeding_date_type"),
            "jan1": jan1, "sow_day": day_of(sow, jan1) if sow else None,
            "harvest_day": day_of(harvest, jan1) if harvest else None}
