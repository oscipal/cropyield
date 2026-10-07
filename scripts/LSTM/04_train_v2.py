"""Train LSTM v2 (see lstm_v2.py) once per split, optionally as a Deep Ensemble, and evaluate on test.

Outputs in <work_dir>/lstm/runs/<config name>/<scope>__<method>__<crop>/ as for 02_train.py:
    preds_test.parquet, preds_val.parquet   pred = ensemble mean (t/ha), pred_std, pred_m<i> per member
    history.csv                             per member and epoch: train loss, val MSE, field-balanced val MSE
    metrics.json                            "metrics": ensemble; "metrics_members": every member;
                                            "metrics_member_mean": mean over members (single-model level)
    model_m<i>.pt
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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from lstm_v2 import (FeatureSpec, LSTMv2, PatchLSTM, TargetScaler, field_balanced_mse,  # noqa: E402
                     gather_patches, sample_per_field)
from src.config import REPO_ROOT, load_paths, load_yaml  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.splits import load_split  # noqa: E402

ROLE_NAMES = {0: "train", 1: "val", 2: "test"}
METRIC_KEYS = ("field_r2", "field_rmse", "pixel_r2", "pixel_rmse")


def all_split_names(store: Path) -> list[str]:
    import zarr
    root = zarr.open_group(store, mode="r")
    return [f"{s}/{m}/{c}" for s in sorted(root.group_keys()) for m in sorted(root[s].group_keys())
            for c in sorted(root[s][m].group_keys())]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch(inputs, b: torch.Tensor, device):
    """Model inputs for the rows b of one set. inputs = (Xd, Xs, nbr); nbr is None for pixel models, else the
    set's local neighbour table, and the dynamic input becomes the patch (B, k*k, T, F + 1)."""
    Xd, Xs, nbr = inputs
    b = b.to(Xd.device)
    xd = Xd[b] if nbr is None else gather_patches(Xd, nbr[b])
    return xd.to(device, non_blocking=True).float(), Xs[b].to(device, non_blocking=True).float()


@torch.no_grad()
def predict(model, inputs, batch_size, device) -> np.ndarray:
    model.eval()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    n = len(inputs[0])
    out = [model(*batch(inputs, torch.arange(i, min(i + batch_size, n)), device)).cpu()
           for i in range(0, n, batch_size)]
    return torch.cat(out).numpy()


def build_model(cfg: dict, n_dyn: int, n_static: int) -> torch.nn.Module:
    mc = cfg["model"]
    if mc.get("type", "lstm") == "patch":
        return PatchLSTM(n_dyn, n_static, mc["patch_size"], mc["conv_channels"], mc["hidden_size"],
                         mc["num_layers"], mc["head_size"], mc.get("dropout", 0.0))
    return LSTMv2(n_dyn + n_static, mc["hidden_size"], mc["num_layers"], mc["head_size"], mc.get("dropout", 0.0))


def local_neighbours(nbr_all: np.ndarray, idx: np.ndarray, n_total: int) -> torch.Tensor:
    """Neighbour table of one set with row indices into that set (-1 = missing). Neighbours never leave a
    field-year, so every existing neighbour of a pixel in the set is in the set too."""
    loc = np.full(n_total, -1, dtype=np.int32)
    loc[idx] = np.arange(len(idx), dtype=np.int32)
    nb = np.asarray(nbr_all[idx])
    out = np.where(nb >= 0, loc[np.maximum(nb, 0)], -1).astype(np.int32)
    assert ((out >= 0) == (nb >= 0)).all(), "a neighbour lies outside its pixel's set"
    return torch.from_numpy(out).long()


def group_metrics(preds: pd.DataFrame, col: str = "pred") -> dict:
    p = preds.assign(pred=preds[col])
    out = {"all": compute_metrics(p)}
    for (country, crop), g in p.groupby(["country", "crop"], observed=True):
        out[f"{country}|{crop}"] = compute_metrics(g)
    return out


def train_member(member, X, y_z, y_true, codes, scal, cfg, device, run):
    """Train one model; early stopping on the field-balanced val MSE in t/ha. Returns model and history."""
    tc, mc = cfg["train"], cfg["model"]
    seed = cfg["seed"] + 1000 * member
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(cfg, X[0][0].shape[-1], X[0][1].shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    loss_fn = torch.nn.MSELoss()
    y_tr = torch.from_numpy(y_z)

    history, best, bad = [], {"score": np.inf, "epoch": -1, "state": None}, 0
    for epoch in range(tc["max_epochs"]):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        pos = torch.from_numpy(sample_per_field(codes[0], tc.get("pixels_per_field"), rng))
        for i in range(0, len(pos), tc["batch_size"]):
            b = pos[i:i + tc["batch_size"]]
            if len(b) < 2:  # BatchNorm needs more than one sample
                continue
            xd, xs = batch(X[0], b, device)
            yb = y_tr[b].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xd, xs), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
            n += len(b)
        pred = predict(model, X[1], tc["eval_batch_size"], device) * scal[1][1] + scal[1][0]
        err2 = (pred - y_true[1]) ** 2
        rec = {"member": member, "epoch": epoch, "train_loss": total / n, "val_mse": float(err2.mean()),
               "val_field_mse": field_balanced_mse(err2, codes[1]), "epoch_time_s": time.time() - t0}
        history.append(rec)
        if run:
            run.log({f"m{member}/{k}": v for k, v in rec.items() if k != "member"})
        improved = rec["val_field_mse"] < best["score"]
        print(f"  m{member} epoch {epoch:2d} train_loss {rec['train_loss']:.4f} val_mse {rec['val_mse']:.4f} "
              f"val_field_mse {rec['val_field_mse']:.4f} ({rec['epoch_time_s']:.0f} s){' *' if improved else ''}",
              flush=True)
        if improved:
            best, bad = {"score": rec["val_field_mse"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}, 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best["state"])
    return model, history, best


def run_split(split_name, cfg, data, out_dir, device, use_wandb):
    dyn, stat, meta, info = data
    tc = cfg["train"]
    scope, method, crop = split_name.split("/")
    s = load_split(load_paths()["split_suite"], scope, method, crop)
    role_of = {f: r for r, key in ((0, "train_fields"), (1, "val_fields"), (2, "test_fields")) for f in s[key]}
    role = meta["field"].map(role_of).to_numpy()
    idx = {r: np.flatnonzero(role == r) for r in (0, 1, 2)}
    if len(role_of) != meta.loc[~pd.isna(role), "field"].nunique():
        raise ValueError(f"{split_name}: some split fields have no pixels")
    print(f"\n=== {split_name}: train {len(idx[0]):,} / val {len(idx[1]):,} / test {len(idx[2]):,} pixels",
          flush=True)

    t0 = time.time()
    spec = FeatureSpec(cfg["inputs"], info).fit(dyn, stat, idx[0], meta)
    X = {r: spec.transform(dyn, stat, idx[r], meta) for r in (0, 1, 2)}
    if cfg["model"].get("type", "lstm") == "patch":
        nbr_all = np.load(info["data_dir"] / f"nbr_k{cfg['model']['patch_size']}.npy", mmap_mode="r")
        X = {r: (*X[r], local_neighbours(nbr_all, idx[r], len(meta))) for r in X}
    else:
        X = {r: (*X[r], None) for r in X}
    tscaler = TargetScaler(cfg.get("target", {}).get("standardize_by")).fit(meta, idx[0])
    scal = {r: tscaler.params(meta, idx[r]) for r in (0, 1, 2)}
    y_true = {r: meta["target"].to_numpy(np.float32)[idx[r]] for r in (0, 1, 2)}
    y_z = ((y_true[0] - scal[0][0]) / scal[0][1]).astype(np.float32)
    codes = {r: pd.factorize(meta["field"].iloc[idx[r]])[0] for r in (0, 1, 2)}
    print(f"  features: {len(spec.dyn_names)} dynamic {spec.dyn_names}, {X[0][1].shape[1]} static; "
          f"prepared in {time.time() - t0:.0f} s", flush=True)

    gb = sum(t.element_size() * t.nelement() for t in X[0] if t is not None) / 1e9
    if str(device).startswith("cuda") and gb <= tc["gpu_data_max_gb"]:
        X[0] = tuple(t.to(device) if t is not None else None for t in X[0])
    print(f"  training data {gb:.1f} GB on {X[0][0].device}", flush=True)

    run = None
    if use_wandb:
        import wandb
        run = wandb.init(project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], mode=cfg["wandb"]["mode"],
                         group=cfg["name"], name=f"{cfg['name']}_{split_name.replace('/', '_')}",
                         config={**cfg, "split": split_name, "n_train_pixels": len(idx[0])}, reinit=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["pixel", "country", "field", "farm", "crop", "year", "row", "col", "target"]
    preds = {r: meta.iloc[idx[r]][cols].reset_index(drop=True) for r in (1, 2)}
    history, bests = [], []
    n_members = cfg.get("ensemble", {}).get("n_members", 1)
    for m in range(n_members):
        model, hist, best = train_member(m, X, y_z, y_true, codes, scal, cfg, device, run)
        history += hist
        bests.append({"member": m, "best_epoch": best["epoch"], "best_val_field_mse": best["score"],
                      "epochs_run": len(hist)})
        torch.save(model.state_dict(), out_dir / f"model_m{m}.pt")
        for r in (1, 2):
            preds[r][f"pred_m{m}"] = predict(model, X[r], tc["eval_batch_size"], device) * scal[r][1] + scal[r][0]
        del model
    del X
    torch.cuda.empty_cache()

    member_cols = [f"pred_m{m}" for m in range(n_members)]
    metrics, metrics_members = {}, {}
    for r in (1, 2):
        p = preds[r]
        p["pred"] = p[member_cols].mean(axis=1)
        p["pred_std"] = p[member_cols].std(axis=1, ddof=0)
        p["field_year"] = p["field"]  # field names are per field-year
        p.to_parquet(out_dir / f"preds_{ROLE_NAMES[r]}.parquet", index=False)
        metrics[ROLE_NAMES[r]] = group_metrics(p)
        metrics_members[ROLE_NAMES[r]] = [group_metrics(p, c) for c in member_cols]
    member_mean = {role: {g: {k: float(np.mean([mm[g][k] for mm in ms])) for k in METRIC_KEYS}
                          for g in ms[0]} for role, ms in metrics_members.items()}
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    summary = {"split": split_name, "config": cfg["name"], "split_attrs": s["attrs"],
               "best_epoch": int(np.median([b["best_epoch"] for b in bests])), "members": bests,
               "epochs_run": int(np.sum([b["epochs_run"] for b in bests])),
               "n_pixels": {ROLE_NAMES[r]: int(len(idx[r])) for r in idx},
               "n_fields": {ROLE_NAMES[r]: int(meta["field"].iloc[idx[r]].nunique()) for r in idx},
               "metrics": metrics, "metrics_member_mean": member_mean, "metrics_members": metrics_members,
               "features": spec.state(), "target_scaler": tscaler.state()}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1, default=str))

    m, mm = metrics["test"]["all"], member_mean["test"]["all"]
    print(f"  TEST {split_name}: ensemble field R2 {m['field_r2']:.3f} RMSE {m['field_rmse']:.3f} | "
          f"pixel R2 {m['pixel_r2']:.3f} RMSE {m['pixel_rmse']:.3f} || member mean field R2 {mm['field_r2']:.3f} "
          f"pixel R2 {mm['pixel_r2']:.3f}", flush=True)
    if run:
        run.summary.update({f"test/{k}": v for k, v in m.items()} |
                           {f"test_member_mean/{k}": v for k, v in mm.items()})
        run.finish()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "lstm_v2_s2.yaml"))
    ap.add_argument("--splits", nargs="*", default=None, help="override config, e.g. germany/loyo/rapeseed")
    ap.add_argument("--overwrite", action="store_true", help="retrain splits that already have metrics.json")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-epochs", type=int, default=None, help="override (for quick tests)")
    ap.add_argument("--members", type=int, default=None, help="override ensemble size (for quick tests)")
    ap.add_argument("--run-name", default=None, help="override output folder name (for quick tests)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.max_epochs:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.members:
        cfg.setdefault("ensemble", {})["n_members"] = args.members
    if args.run_name:
        cfg["name"] = args.run_name
    paths = load_paths()
    data_dir = paths["work_dir"] / "lstm" / "data"
    run_root = paths["work_dir"] / "lstm" / "runs" / cfg["name"]
    torch.backends.cudnn.benchmark = True

    info = json.loads((data_dir / "info.json").read_text())
    info["data_dir"] = data_dir
    dyn = np.load(data_dir / "dyn.npy", mmap_mode="r")
    stat = np.load(data_dir / "static.npy", mmap_mode="r")
    meta = pd.read_parquet(data_dir / "meta.parquet")
    for c in ("country", "crop", "year", "farm"):
        meta[c] = meta[c].astype("category")
    print(f"{cfg['name']}: {len(meta):,} pixels, device {args.device}")

    splits = args.splits or (all_split_names(paths["split_suite"]) if cfg["splits"] == "all" else cfg["splits"])
    use_wandb = not args.no_wandb and cfg["wandb"]["mode"] != "disabled"
    failed = []
    for name in splits:
        out_dir = run_root / name.replace("/", "__")
        if (out_dir / "metrics.json").exists() and not args.overwrite:
            print(f"skip {name} (done)")
            continue
        try:
            run_split(name, cfg, (dyn, stat, meta, info), out_dir, args.device, use_wandb)
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
