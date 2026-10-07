"""Collect the LSTM test results of all splits into docs/results_lstm.md, next to the paper's LSTM values."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from src.config import REPO_ROOT, load_paths, load_yaml  # noqa: E402

ABBR = {"Argentina": "ARG", "Brazil": "BRA", "Germany": "GER", "Uruguay": "URG",
        "corn": "C", "rapeseed": "R", "soybean": "S", "wheat": "W"}
SCOPE_ORDER = {"germany": 0, "south_america": 1, "world": 2}
SINGLE = " (single)"


def load_runs(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(run_dir.glob("*/metrics.json"))]


def fmt(x, nd=2):
    return "–" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def paper_cell(paper, inp, method, level, subset):
    try:
        v = paper[inp][method][level][subset]
    except KeyError:
        return "–", "–"
    return f"{v['r2'][0]:.2f} ± {v['r2'][1]:.2f}", f"{v['rmse'][0]:.2f} ± {v['rmse'][1]:.2f}"


def summary(gr: pd.DataFrame, cfgs: list[dict], paper: dict, inp_key: dict) -> list[str]:
    """Summary over the country x crop subsets of all splits (world single-crop splits are left out: they
    equal the south_america / germany splits)."""
    g = gr[~(gr["scope"].eq("world") & gr["crop"].ne("all"))].copy()
    paper_inp = {c["name"]: c.get("paper_inputs") or inp_key.get(c["name"], "s2") for c in cfgs}
    variants = []
    for c in cfgs:
        variants += [n for n in (f"{c['name']}{SINGLE}", c["name"]) if n in set(g["config"])]
    base = cfgs[0]["name"]
    keys = ["split", "subset"]
    b = g[g["config"] == base].set_index(keys)

    def paper_val(row, inp, level):
        return paper[inp][row["method"]][level][row["subset"]]["r2"][0]

    rows = []
    for v in variants:
        d = g[g["config"] == v].set_index(keys)
        inp = paper_inp[v.removesuffix(SINGLE)]
        d["paper_r2"] = [paper_val(r, inp, "field") for _, r in d.reset_index().iterrows()]
        mixed = d.reset_index()["crop"].eq("all").to_numpy()
        row = {"variant": f"`{v}`", "n": len(d),
               "median field R²": fmt(d["field_r2"].median()),
               "median field R² (all-crop splits)": fmt(d["field_r2"][mixed].median()),
               "median field R² (single-crop splits)": fmt(d["field_r2"][~mixed].median()),
               "median pixel R²": fmt(d["pixel_r2"].median()),
               "mean field RMSE": fmt(d["field_rmse"].mean())}
        row[f"field R² > `{base}`"] = "–" if v == base else \
            f"{int((d['field_r2'] > b['field_r2'].reindex(d.index)).sum())} / {len(d)}"
        row["field R² ≥ paper"] = f"{int((d['field_r2'] >= d['paper_r2']).sum())} / {len(d)}"
        rows.append(row)
    p = g[g["config"] == base].reset_index(drop=True)
    pr = [paper_val(r, "s2", "field") for _, r in p.iterrows()]
    rows.append({"variant": "paper LSTM, S2 (same subsets)", "n": len(p),
                 "median field R²": fmt(pd.Series(pr).median()),
                 "median field R² (all-crop splits)": fmt(pd.Series(pr)[p["crop"].eq("all")].median()),
                 "median field R² (single-crop splits)": fmt(pd.Series(pr)[p["crop"].ne("all")].median()),
                 "median pixel R²": fmt(pd.Series([paper["s2"][r["method"]]["pixel"][r["subset"]]["r2"][0]
                                                   for _, r in p.iterrows()]).median()),
                 "mean field RMSE": fmt(pd.Series([paper["s2"][r["method"]]["field"][r["subset"]]["rmse"][0]
                                                   for _, r in p.iterrows()]).mean()),
                 f"field R² > `{base}`": "–", "field R² ≥ paper": "–"})
    L = ["## Summary", "",
         "Test metrics per country × crop subset of every split (the `world` single-crop splits are left out, "
         "they equal the `south_america` / `germany` ones), aggregated per variant. Counts compare the same "
         "split and subset. The paper row uses the paper's LOYO/LORO mean for each of the same subsets; "
         "paper values are averages over many held-out years/regions, ours are single held-out sets.", "",
         pd.DataFrame(rows).to_markdown(index=False), ""]
    ens = [c["name"] for c in cfgs if f"{c['name']}{SINGLE}" in set(g["config"])]
    for n in ens:
        e = g[g["config"] == n].set_index(keys)["field_r2"]
        s1 = g[g["config"] == f"{n}{SINGLE}"].set_index(keys)["field_r2"].reindex(e.index)
        L += [f"Ensemble vs. single model (`{n}`): field R² higher for the ensemble in {int((e > s1).sum())} / "
              f"{len(e)} subsets, median gain {fmt((e - s1).median())}.", ""]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[str(REPO_ROOT / "configs" / f"{n}.yaml")
                                                     for n in ("lstm_s2", "lstm_s2_adm", "lstm_v2_s2")])
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "results_lstm.md"))
    args = ap.parse_args()

    paths = load_paths()
    paper = yaml.safe_load((HERE / "paper_lstm_results.yaml").read_text())
    cfgs = [load_yaml(c) for c in args.configs]
    inp_key = {"lstm_s2": "s2", "lstm_s2_adm": "s2_adm"}

    overall, groups = [], []
    for cfg in cfgs:
        for r in load_runs(paths["work_dir"] / "lstm" / "runs" / cfg["name"]):
            scope, method, crop = r["split"].split("/")
            base = {"config": cfg["name"], "split": r["split"], "scope": scope, "method": method, "crop": crop}
            m = r["metrics"]["test"]["all"]
            overall.append({**base, **m, "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"],
                            "n_train_fields": r["n_fields"]["train"]})
            sources = [(cfg["name"], r["metrics"]["test"])]
            if "metrics_member_mean" in r:  # ensemble run: also keep the average single member
                sources.append((f"{cfg['name']}{SINGLE}", r["metrics_member_mean"]["test"]))
            for name, ms in sources:
                for g, gm in ms.items():
                    if g == "all":
                        continue
                    country, gcrop = g.split("|")
                    groups.append({**base, "config": name, "subset": f"{ABBR[country]}-{ABBR[gcrop]}",
                                   "n_fields": r["metrics"]["test"][g]["n_fields"], **gm})
    if not overall:
        raise SystemExit("no finished runs found")
    ov, gr = pd.DataFrame(overall), pd.DataFrame(groups)
    key = lambda d: d.assign(_s=d["scope"].map(SCOPE_ORDER), _c=d["crop"].eq("all"))  # noqa: E731
    ov = key(ov).sort_values(["_s", "method", "_c", "crop"]).drop(columns=["_s", "_c"])
    gr = key(gr).sort_values(["_s", "method", "_c", "crop", "subset"]).drop(columns=["_s", "_c"])

    L = ["# LSTM results on the split suite", "",
         "Generated by `scripts/LSTM/03_report.py` from `<work_dir>/lstm/runs`. One model per split "
         "(`docs/splits.md`), evaluated on the split's test set.", "",
         "## Setup", "",
         "Model and training as the LSTM of the YieldSAT benchmark (paper ref. [36], Pathak et al., IGARSS 2023): "
         "2 stacked LSTM layers (128 hidden units), last hidden state, FC 128 → BatchNorm → ReLU → FC 1; "
         "MSE loss, Adam lr 1e-3, batch 1024, at most 50 epochs, early stopping after 8 epochs without "
         "improvement of the val MSE; the weights of the best val epoch are evaluated. Pixels are independent samples.",
         "",
         "- **Inputs:** `lstm_s2` = 12 S2 bands × 24 time steps. `lstm_s2_adm` = input fusion: S2 + 4 weather "
         "bands per time step, plus 48 soil (8 properties × 6 depths, no uncertainty layers) and 5 topography bands "
         "repeated at every time step (69 features). Coordinates are not used.",
         "- **Normalisation (not specified in the paper):** z-score per band with mean/std of the split's "
         "training pixels (valid time steps only); missing time steps outside the season (NaN) are then set to 0.",
         "- **`lstm_v2_s2`** (S2 inputs only; not run with ADM) adds to the paper LSTM, all switchable in "
         "`configs/lstm_v2_s2.yaml` (code: `scripts/LSTM/lstm_v2.py`, `04_train_v2.py`): "
         "(1) NDVI, NDRE and NDMI per time step, computed from the raw bands; "
         "(2) a 0/1 channel marking observed time steps; "
         "(3) one-hot crop and country, repeated over time; "
         "(4) the yield z-scored per crop × country with training statistics, predictions transformed back to t/ha; "
         "(5) field-balanced training: per epoch at most 500 random pixels per field, early stopping on the "
         "val MSE averaged per field (patience 10, at most 100 epochs); "
         "(6) dropout 0.2 (between the LSTM layers and in the head), AdamW with weight decay 1e-4; "
         "(7) Deep Ensemble of 5 members (different seeds), prediction = mean of the members. "
         "`lstm_v2_s2` columns are the ensemble; `lstm_v2_s2 (single)` is the average of the 5 members' metrics, "
         "i.e. the expected result of one v2 model.",
         "- **Validation:** the split's val set (held-out year or regions) is used for early stopping, the test "
         "set only for the final evaluation.",
         "- **Metrics:** pixel level over all test pixels; field level compares the mean prediction with the mean "
         "target per field-year. R² on pooled multi-crop or multi-country test sets is inflated by the yield "
         "differences between crops and countries; the per country × crop rows are the comparable ones.",
         "- **Paper values** are mean ± std over all folds of a per country × crop model (LOYO: every year held "
         "out once; LORO: every region). Ours are single held-out year/region sets, and models of the "
         "south_america/world/all splits are trained on several countries and crops. Treat the comparison as "
         "a sanity check, not a replication of the same experiment.", ""]

    L += summary(gr, cfgs, paper, inp_key)
    L += ["## Test results per split", ""]
    for level in ("field", "pixel"):
        t = ov.pivot_table(index="split", columns="config", values=[f"{level}_r2", f"{level}_rmse"], sort=False)
        rows = []
        for split in ov["split"].unique():
            row = {"split": f"`{split}`"}
            for cfg in cfgs:
                n = cfg["name"]
                for met in ("r2", "rmse"):
                    col = (f"{level}_{met}", n)
                    row[f"{n} {'R²' if met == 'r2' else 'RMSE'}"] = fmt(t.loc[split, col]) if col in t else "–"
            rows.append(row)
        L += [f"### {level.capitalize()} level", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    info = ov[["config", "split", "n_fields", "n_pixels", "n_train_fields", "best_epoch", "epochs_run"]]
    L += ["### Training", "", info.to_markdown(index=False), ""]

    L += ["## Test results per country × crop, next to the paper", "",
          "Paper columns are the LSTM rows of appendix Tab. 15–18 (LOYO / LORO), same input set.", ""]
    for level in ("field", "pixel"):
        rows = []
        for (split, subset), g in gr.groupby(["split", "subset"], sort=False):
            method = split.split("/")[1]
            row = {"split": f"`{split}`", "subset": subset, "test fields": int(g["n_fields"].iloc[0])}
            for cfg in cfgs:
                n, inp = cfg["name"], cfg.get("paper_inputs") or inp_key.get(cfg["name"], "s2")
                gg = g[g["config"] == n]
                row[f"{n} R²"] = fmt(gg[f"{level}_r2"].iloc[0]) if len(gg) else "–"
                row[f"{n} RMSE"] = fmt(gg[f"{level}_rmse"].iloc[0]) if len(gg) else "–"
                pr2, prmse = paper_cell(paper, inp, method, level, subset)
                row[f"paper {inp} R²"], row[f"paper {inp} RMSE"] = pr2, prmse
            rows.append(row)
        L += [f"### {level.capitalize()} level", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    Path(args.out).write_text("\n".join(L))
    print(f"wrote {args.out} ({len(ov)} runs)")


if __name__ == "__main__":
    main()
