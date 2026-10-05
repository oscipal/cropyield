"""Configuration loading. Paths live only in configs/paths.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PATHS_FILE = REPO_ROOT / "configs" / "paths.yaml"


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paths(path: str | Path = PATHS_FILE) -> dict:
    """Return paths.yaml with all entries converted to absolute Paths."""
    cfg = load_yaml(path)
    out = {k: Path(v) for k, v in cfg["data"].items()}
    out["work_dir"] = Path(cfg["work_dir"])
    out["splits_dir"] = REPO_ROOT / cfg["splits_dir"]
    out["docs_dir"] = REPO_ROOT / cfg["docs_dir"]
    for key in ("preprocessed", "raw"):
        if out["work_dir"].resolve().is_relative_to(out[key].resolve()):
            raise ValueError(f"work_dir must not be inside the data folder {out[key]}")
    return out
