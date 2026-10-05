"""Train LSTM (S2 only, GER-R, CV10) or baseline B0 on the saved splits; one W&B run per fold."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT, load_paths, load_yaml  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.models import LSTMRegressor  # noqa: E402
from src.preprocess import band_stats, normalize  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, X: torch.Tensor, batch_size: int, device) -> np.ndarray:
    model.eval()
    out = [model(X[i:i + batch_size].to(device)).cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def train_lstm(X, meta, idx_fit, idx_val, idx_test, cfg, seed, device, log):
    mean, std = band_stats(X, idx_fit)
    Xf, Xv, Xt = (torch.from_numpy(normalize(X, i, mean, std)) for i in (idx_fit, idx_val, idx_test))
    yf = torch.from_numpy(meta["target"].to_numpy(np.float32)[idx_fit])
    yv = meta["target"].to_numpy(np.float32)[idx_val]

    tc = cfg["train"]
    model = LSTMRegressor(X.shape[-1], cfg["model"]["hidden_size"], cfg["model"]["num_layers"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)

    best_val, best_state, best_epoch, bad = np.inf, None, -1, 0
    for epoch in range(tc["max_epochs"]):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        perm = torch.randperm(len(Xf), generator=gen)
        for i in range(0, len(perm), tc["batch_size"]):
            b = perm[i:i + tc["batch_size"]]
            xb, yb = Xf[b].to(device), yf[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            n += len(b)
        val_mse = float(np.mean((predict(model, Xv, tc["eval_batch_size"], device) - yv) ** 2))
        log({"epoch": epoch, "train_mse": total / n, "val_mse": val_mse, "epoch_time_s": time.time() - t0})
        print(f"    epoch {epoch:2d} train_mse {total / n:.4f} val_mse {val_mse:.4f}", flush=True)
        if val_mse < best_val:
            best_val, best_state, best_epoch, bad = val_mse, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    pred = predict(model, Xt, tc["eval_batch_size"], device)
    extra = {"best_epoch": best_epoch, "best_val_mse": best_val, "epochs_run": epoch + 1}
    return pred, extra, {"band_mean": mean.tolist(), "band_std": std.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "ger_r_lstm.yaml"))
    ap.add_argument("--model", choices=["lstm", "b0"], default="lstm")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = load_paths()
    data_dir = paths["work_dir"] / cfg["data"]["subdir"]
    experiment = f"{args.model}_{cfg['name']}"
    pred_dir = data_dir / "preds" / experiment
    pred_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / "X.npy", mmap_mode="r")
    meta = pd.read_parquet(data_dir / "meta.parquet")
    splits = json.loads((paths["splits_dir"] / cfg["splits"]["file"]).read_text())
    folds = splits["folds"] if args.folds is None else [f for f in splits["folds"] if f["fold"] in args.folds]
    print(f"{experiment}: X {X.shape}, {len(folds)} folds, device {args.device}")

    for fold in folds:
        k = fold["fold"]
        seed = cfg["seed"] + k
        set_seed(seed)
        idx_fit = np.flatnonzero(meta["field"].isin(fold["train_fields"]).to_numpy())
        idx_val = np.flatnonzero(meta["field"].isin(fold["val_fields"]).to_numpy())
        idx_test = np.flatnonzero(meta["field"].isin(fold["test_fields"]).to_numpy())
        print(f"  fold {k}: fit {len(idx_fit):,} / val {len(idx_val):,} / test {len(idx_test):,} pixels")

        run = None
        if not args.no_wandb and cfg["wandb"]["mode"] != "disabled":
            import wandb
            run = wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"],
                             mode=cfg["wandb"]["mode"], group=experiment, name=f"{experiment}_fold{k}",
                             config={**cfg, "model_type": args.model, "fold": k, "fold_seed": seed},
                             reinit=True)
        log = run.log if run else (lambda d: None)

        if args.model == "b0":
            # constant prediction: mean of all training pixels of the outer fold (fit + inner val)
            y_train = meta["target"].to_numpy()[np.concatenate([idx_fit, idx_val])]
            pred = np.full(len(idx_test), float(y_train.mean()), dtype=np.float32)
            extra, norm = {"train_mean": float(y_train.mean())}, {}
        else:
            pred, extra, norm = train_lstm(X, meta, idx_fit, idx_val, idx_test, cfg, seed, args.device, log)

        out = meta.iloc[idx_test][["pixel", "field", "field_year", "farm", "year", "row", "col", "target"]].copy()
        out["pred"] = pred
        out["fold"] = k
        out.to_parquet(pred_dir / f"fold{k}.parquet", index=False)
        (pred_dir / f"fold{k}.json").write_text(json.dumps({**extra, **norm}, indent=1))

        m = compute_metrics(out)
        print(f"  fold {k}: " + ", ".join(f"{a} {b:.4f}" if isinstance(b, float) else f"{a} {b}"
                                          for a, b in m.items()))
        if run:
            run.summary.update({f"test/{a}": b for a, b in m.items()} | extra)
            run.finish()


if __name__ == "__main__":
    main()
