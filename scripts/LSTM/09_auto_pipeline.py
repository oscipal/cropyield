"""Unattended rounds 2 and final of .claude/plans/patch_lstm_improvements.md, after the round-1 screening job.

    python scripts/LSTM/09_auto_pipeline.py --wait-pid <pid of patch_screen.sh>

1. wait for round 1; keep a variant if its median val subset field R2 beats the reference (lstm_patch_s2, first
   2 members) by more than max(reference spread, own spread); offset share is printed as a diagnostic only
   (it is a fraction of the error and can rise while the error falls);
2. combine the kept variants into configs/patch_screen/patch_combined.yaml and screen it the same way
   (skipped if only one variant was kept);
3. best of kept variants and combined (by val median field R2) -> configs/lstm_patch_v3_s2.yaml, 5 members,
   all 24 splits; stops here if nothing was kept;
4. 03_report.py over all models.
Decisions use val only. Everything is printed with '#####' markers.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import load_paths  # noqa: E402

spec = importlib.util.spec_from_file_location("screen", ROOT / "scripts/LSTM/08_screen_report.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)

REF = "lstm_patch_s2"
SPLITS = screen.DEFAULT_SPLITS
VARIANTS = ["patch_weather", "patch_soil_dem", "patch_aug_time", "patch_aug_spec"]
ALL_CONFIGS = ["lstm_s2", "lstm_s2_adm", "lstm_v2_s2", "lstm_patch_s2", "lstm_patch_v3_s2", "baseline_mean",
               "baseline_mean_crop"]


def say(msg: str) -> None:
    print(f"##### {msg}", flush=True)


def summary(name: str) -> dict:
    df = screen.collect(load_paths()["work_dir"] / "lstm" / "runs" / name, SPLITS, "val", 2)
    g = df.groupby(["split", "subset"])
    r, sp = g["field_r2"].mean(), g["field_r2"].std(ddof=0).fillna(0.0)
    return {"r2": float(r.median()), "spread": float(sp.median()), "offset": float(g["offset_share"].mean().median())}


def run_train(cfg_path: Path, splits: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / "scripts/LSTM/04_train_v2.py"), "--config", str(cfg_path), "--no-wandb"]
    if splits:
        cmd += ["--splits", *splits]
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True)
    args = ap.parse_args()
    while Path(f"/proc/{args.wait_pid}").exists():
        time.sleep(60)

    ref = summary(REF)
    say(f"reference {ref}")
    kept, stats = [], {}
    for v in VARIANTS:
        try:
            stats[v] = summary(v)
        except Exception as e:  # missing splits
            say(f"{v}: no usable results ({e})")
            continue
        s = stats[v]
        ok = s["r2"] > ref["r2"] + max(ref["spread"], s["spread"])
        say(f"{v}: {s} -> {'keep' if ok else 'drop'}")
        if ok:
            kept.append(v)
    if not kept:
        say("no variant beat the reference; stopping, nothing to combine")
        return

    cfgs = {v: yaml.safe_load((ROOT / f"configs/patch_screen/{v}.yaml").read_text()) for v in kept}
    best_name = max(kept, key=lambda v: stats[v]["r2"])
    if len(kept) > 1:
        comb = yaml.safe_load((ROOT / "configs/lstm_patch_s2.yaml").read_text())
        comb["ensemble"]["n_members"] = 2
        for v in kept:
            c = cfgs[v]
            comb["inputs"]["dynamic"] = c["inputs"]["dynamic"] if v == "patch_weather" else comb["inputs"]["dynamic"]
            comb["inputs"]["static"] = c["inputs"]["static"] if v == "patch_soil_dem" else comb["inputs"]["static"]
            if c["train"].get("augment"):
                comb["train"].setdefault("augment", {}).update(c["train"]["augment"])
        comb["name"] = "patch_combined"
        path = ROOT / "configs/patch_screen/patch_combined.yaml"
        path.write_text(f"# Screening: lstm_patch_s2 + {', '.join(kept)} (written by 09_auto_pipeline.py)\n"
                        + yaml.safe_dump(comb, sort_keys=False))
        say("variant patch_combined")
        run_train(path, SPLITS)
        stats["patch_combined"] = summary("patch_combined")
        cfgs["patch_combined"] = comb
        say(f"patch_combined: {stats['patch_combined']}")
        cand = kept + ["patch_combined"]
        best_name = max(cand, key=lambda v: stats[v]["r2"])
    if stats[best_name]["r2"] <= ref["r2"] + max(ref["spread"], stats[best_name]["spread"]):
        say("best candidate does not beat the reference; stopping")
        return

    final = cfgs[best_name]
    final["name"], final["ensemble"]["n_members"] = "lstm_patch_v3_s2", 5
    fpath = ROOT / "configs/lstm_patch_v3_s2.yaml"
    fpath.write_text(f"# Patch LSTM v3: lstm_patch_s2 + the changes of screening config {best_name} "
                     f"(chosen on val by 09_auto_pipeline.py)\n" + yaml.safe_dump(final, sort_keys=False))
    say(f"final config from {best_name}")
    run_train(fpath)
    cmd = [sys.executable, str(ROOT / "scripts/LSTM/03_report.py"), "--configs",
           *[str(ROOT / "configs" / f"{c}.yaml") for c in ALL_CONFIGS]]
    subprocess.run(cmd, check=True, cwd=ROOT)
    say("pipeline done")


if __name__ == "__main__":
    main()
