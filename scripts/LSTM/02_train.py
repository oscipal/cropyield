"""Train the pixel LSTM once per split of the split suite and evaluate it on the split's test set.

Per split: z-score every band with mean/std of the split's training pixels (missing time steps -> fill
value), train with early stopping on the split's val set, keep the best epoch, predict val and test.

Outputs in <work_dir>/lstm/runs/<config name>/<scope>__<method>__<crop>/:
    preds_test.parquet, preds_val.parquet   pixel predictions with country, crop, year, field
    history.csv                             train/val MSE per epoch
    metrics.json                            test/val metrics overall and per country x crop, normalisation stats
    model.pt                                best weights
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
# set before torch initialises CUDA: avoids fragmentation between training and evaluation batches
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from lstm_lib import PaperLSTM, band_stats, expand_bands, normalized  # noqa: E402
from src.config import REPO_ROOT, load_paths, load_yaml  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.splits import load_split  # noqa: E402

ROLE_NAMES = {0: "train", 1: "val", 2: "test"}


def all_split_names(store: Path) -> list[str]:
    import zarr
    root = zarr.open_group(store, mode="r")
    return [f"{s}/{m}/{c}" for s in sorted(root.group_keys()) for m in sorted(root[s].group_keys())
            for c in sorted(root[s][m].group_keys())]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, Xd, Xs, batch_size, device) -> np.ndarray:
    model.eval()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    out = []
    for i in range(0, len(Xd), batch_size):
        xd = Xd[i:i + batch_size].to(device, non_blocking=True).float()
        xs = Xs[i:i + batch_size].to(device, non_blocking=True).float()
        out.append(model(xd, xs).cpu())
    return torch.cat(out).numpy()


def group_metrics(preds: pd.DataFrame) -> dict:
    out = {"all": compute_metrics(preds)}
    for (country, crop), g in preds.groupby(["country", "crop"], observed=True):
        out[f"{country}|{crop}"] = compute_metrics(g)
    return out


def run_split(split_name, cfg, data, out_dir, device, use_wandb):
    dyn, stat, meta, dyn_cols, stat_cols = data
    scope, method, crop = split_name.split("/")
    s = load_split(load_paths()["split_suite"], scope, method, crop)
    role_of = {f: r for r, key in ((0, "train_fields"), (1, "val_fields"), (2, "test_fields")) for f in s[key]}
    role = meta["field"].map(role_of).to_numpy()
    idx = {r: np.flatnonzero(role == r) for r in (0, 1, 2)}
    n_missing = len(role_of) - meta.loc[~pd.isna(role), "field"].nunique()
    if n_missing:
        raise ValueError(f"{split_name}: {n_missing} split fields have no pixels")
    print(f"\n=== {split_name}: train {len(idx[0]):,} / val {len(idx[1]):,} / test {len(idx[2]):,} pixels",
          flush=True)

    # normalisation statistics from the training pixels only
    t0 = time.time()
    d_mean, d_std = band_stats(dyn, dyn_cols, idx[0])
    s_mean, s_std = band_stats(stat, stat_cols, idx[0]) if stat_cols else (np.zeros(0, np.float32),) * 2
    fill = float(cfg["inputs"]["fill_value"])
    X = {}
    for r in (0, 1, 2):
        xd = normalized(dyn, dyn_cols, idx[r], d_mean, d_std, fill)
        xs = normalized(stat, stat_cols, idx[r], s_mean, s_std, fill) if stat_cols \
            else torch.zeros((len(idx[r]), 0), dtype=torch.float16)
        X[r] = (xd, xs)
    y_train = torch.from_numpy(meta["target"].to_numpy(np.float32)[idx[0]])
    y_val = meta["target"].to_numpy(np.float32)[idx[1]]
    print(f"  normalised in {time.time() - t0:.0f} s", flush=True)

    tc, mc = cfg["train"], cfg["model"]
    gb = sum(t.element_size() * t.nelement() for t in X[0]) / 1e9
    if device.startswith("cuda") and gb <= tc["gpu_data_max_gb"]:
        Xd_tr, Xs_tr, y_tr = X[0][0].to(device), X[0][1].to(device), y_train.to(device)
    else:
        Xd_tr, Xs_tr, y_tr = X[0][0], X[0][1], y_train
    print(f"  training data {gb:.1f} GB on {Xd_tr.device}", flush=True)

    set_seed(cfg["seed"])
    model = PaperLSTM(len(dyn_cols) + len(stat_cols), mc["hidden_size"], mc["num_layers"], mc["head_size"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(cfg["seed"])

    run = None
    if use_wandb:
        import wandb
        run = wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], mode=cfg["wandb"]["mode"],
                         group=cfg["name"], name=f"{cfg['name']}_{split_name.replace('/', '_')}",
                         config={**cfg, "split": split_name, "n_train_pixels": len(idx[0])}, reinit=True)

    history, best = [], {"val_mse": np.inf, "epoch": -1, "state": None}
    bad = 0
    for epoch in range(tc["max_epochs"]):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        perm = torch.randperm(len(y_tr), generator=gen).to(Xd_tr.device)
        for i in range(0, len(perm), tc["batch_size"]):
            b = perm[i:i + tc["batch_size"]]
            if len(b) < 2:  # BatchNorm needs more than one sample
                continue
            xd = Xd_tr[b].to(device, non_blocking=True).float()
            xs = Xs_tr[b].to(device, non_blocking=True).float()
            yb = y_tr[b].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xd, xs), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            n += len(b)
        val_mse = float(np.mean((predict(model, *X[1], tc["eval_batch_size"], device) - y_val) ** 2))
        rec = {"epoch": epoch, "train_mse": total / n, "val_mse": val_mse, "epoch_time_s": time.time() - t0}
        history.append(rec)
        if run:
            run.log(rec)
        improved = val_mse < best["val_mse"]
        print(f"  epoch {epoch:2d} train_mse {total / n:.4f} val_mse {val_mse:.4f} "
              f"({rec['epoch_time_s']:.0f} s){' *' if improved else ''}", flush=True)
        if improved:
            best = {"val_mse": val_mse, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            bad = 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best["state"])
    del Xd_tr, Xs_tr, y_tr
    torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["pixel", "country", "field", "farm", "crop", "year", "row", "col", "target"]
    metrics = {}
    for r in (1, 2):
        p = meta.iloc[idx[r]][cols].reset_index(drop=True)
        p["pred"] = predict(model, *X[r], tc["eval_batch_size"], device)
        p["field_year"] = p["field"]  # field names are per field-year
        p.to_parquet(out_dir / f"preds_{ROLE_NAMES[r]}.parquet", index=False)
        metrics[ROLE_NAMES[r]] = group_metrics(p)
    torch.save(model.state_dict(), out_dir / "model.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    summary = {"split": split_name, "config": cfg["name"], "split_attrs": s["attrs"],
               "best_epoch": best["epoch"], "best_val_mse": best["val_mse"], "epochs_run": len(history),
               "n_pixels": {ROLE_NAMES[r]: int(len(idx[r])) for r in idx},
               "n_fields": {ROLE_NAMES[r]: int(meta["field"].iloc[idx[r]].nunique()) for r in idx},
               "metrics": metrics,
               "norm": {"dynamic_mean": d_mean.tolist(), "dynamic_std": d_std.tolist(),
                        "static_mean": s_mean.tolist(), "static_std": s_std.tolist()}}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1, default=str))

    m = metrics["test"]["all"]
    print(f"  TEST {split_name}: field R2 {m['field_r2']:.3f} RMSE {m['field_rmse']:.3f} | "
          f"pixel R2 {m['pixel_r2']:.3f} RMSE {m['pixel_rmse']:.3f} (best epoch {best['epoch']})", flush=True)
    if run:
        run.summary.update({f"test/{k}": v for k, v in m.items()} | {"best_epoch": best["epoch"]})
        run.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "lstm_s2.yaml"))
    ap.add_argument("--splits", nargs="*", default=None, help="override config, e.g. germany/loyo/rapeseed")
    ap.add_argument("--overwrite", action="store_true", help="retrain splits that already have metrics.json")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = load_paths()
    data_dir = paths["work_dir"] / "lstm" / "data"
    run_root = paths["work_dir"] / "lstm" / "runs" / cfg["name"]
    torch.backends.cudnn.benchmark = True

    info = json.loads((data_dir / "info.json").read_text())
    dyn_bands = expand_bands(cfg["inputs"]["dynamic"], info["dynamic_bands"])
    stat_bands = expand_bands(cfg["inputs"]["static"], info["static_bands"])
    dyn_cols = [info["dynamic_bands"].index(b) for b in dyn_bands]
    stat_cols = [info["static_bands"].index(b) for b in stat_bands]
    dyn = np.load(data_dir / "dyn.npy", mmap_mode="r")
    stat = np.load(data_dir / "static.npy", mmap_mode="r")
    meta = pd.read_parquet(data_dir / "meta.parquet")
    for c in ("country", "crop", "year", "farm"):
        meta[c] = meta[c].astype("category")
    print(f"{cfg['name']}: {len(meta):,} pixels, dynamic {dyn_bands}, static {len(stat_bands)} bands, "
          f"device {args.device}")

    splits = args.splits or (all_split_names(paths["split_suite"]) if cfg["splits"] == "all" else cfg["splits"])
    use_wandb = not args.no_wandb and cfg["wandb"]["mode"] != "disabled"
    failed = []
    for name in splits:
        out_dir = run_root / name.replace("/", "__")
        if (out_dir / "metrics.json").exists() and not args.overwrite:
            print(f"skip {name} (done)")
            continue
        try:
            run_split(name, cfg, (dyn, stat, meta, dyn_cols, stat_cols), out_dir, args.device, use_wandb)
        except Exception:  # keep going with the other splits, report at the end
            traceback.print_exc()
            print(f"FAILED {name}", flush=True)
            failed.append(name)
            if use_wandb:
                import wandb
                if wandb.run is not None:
                    wandb.run.finish(exit_code=1)
        torch.cuda.empty_cache()
    if failed:
        raise SystemExit(f"failed splits: {failed}")


if __name__ == "__main__":
    main()
