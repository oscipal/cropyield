"""Train the field-image model (field_image.py) once per split, optionally as a Deep Ensemble, and evaluate on test.

Inputs, target scaling, split handling, command line and outputs are those of LSTM v2 (04_train_v2.py), so
03_report.py reads the runs like any other model. What differs:

* training samples are random crops (train.crop_size) of the training fields, field-balanced: every field gives
  train.crops_per_field crops per epoch, re-drawn every epoch, with random 90 degree rotations and flips;
* the loss is the MSE over the labelled pixels of a crop (standardised target), optionally plus
  train.field_mean_weight x the squared error of the crop's mean prediction;
* val/test are predicted per whole field (padded), larger fields in overlapping tiles (predict.tile,
  predict.overlap) whose predictions are averaged.

Outputs in <work_dir>/lstm/runs/<config name>/<scope>__<method>__<crop>/, layout as in 04_train_v2.py.
"""
from __future__ import annotations

import copy
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
v2 = importlib.import_module("04_train_v2")  # also puts the repository root on sys.path
from field_image import FieldImageModel, FieldSet, collate  # noqa: E402
from lstm_v2 import field_balanced_mse  # noqa: E402


def build_model(cfg: dict, n_dyn: int, n_static: int, n_steps: int, obs_index: int | None) -> FieldImageModel:
    mc = cfg["model"]
    return FieldImageModel(n_dyn, n_static, n_steps, dict(mc["encoder"]), mc["embed"], mc["unet_channels"],
                           mc.get("dropout", 0.0), obs_index)


def to_device(Xd, Xs, pos: np.ndarray, device):
    p = torch.from_numpy(pos).to(Xd.device)
    return Xd[p].to(device, non_blocking=True).float(), Xs[p].to(device, non_blocking=True).float()


@torch.no_grad()
def predict(model, X, fields: FieldSet, cfg: dict, device) -> np.ndarray:
    """z-scored prediction for every pixel of a set (in the set's order): embeddings for all pixels first, then
    the U-Net per field or tile; overlapping tiles are averaged."""
    model.eval()
    tc, pc = cfg["train"], cfg["predict"]
    Xd, Xs = X
    amp = tc.get("amp", False) and str(device).startswith("cuda")
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        emb = torch.cat([model.encode(*to_device(Xd, Xs, np.arange(i, min(i + pc["batch_pixels"], len(Xd))), device))
                         .float().cpu() for i in range(0, len(Xd), pc["batch_pixels"])])
        total, count = torch.zeros(len(Xd)), torch.zeros(len(Xd))
        for f in range(len(fields)):
            for pos, r, c, h, w in fields.tiles(f, model.unet.multiple, pc["tile"], pc["overlap"]):
                p = torch.from_numpy(pos)
                zero = torch.zeros(len(pos), dtype=torch.long, device=device)
                out = model.spatial(emb[p].to(device), Xs[p.to(Xs.device)].to(device).float(), zero,
                                    torch.from_numpy(r).long().to(device), torch.from_numpy(c).long().to(device),
                                    1, h, w)
                total[p] += out.float().cpu()
                count[p] += 1
    assert (count > 0).all(), "some pixels were not predicted"
    return (total / count).numpy()


def train_member(member, X, fields, y_z, y_true, codes, scal, cfg, device, run):
    """Train one model; early stopping on the field-balanced val MSE in t/ha. Returns model, history, best."""
    tc, pc = cfg["train"], cfg["predict"]
    seed = cfg["seed"] + 1000 * member
    v2.set_seed(seed)
    rng = np.random.default_rng(seed)
    Xd, Xs = X[0]
    obs_index = Xd.shape[-1] - 1 if cfg["inputs"].get("mask_channel") else None
    model = build_model(cfg, Xd.shape[-1], Xs.shape[-1], Xd.shape[1], obs_index).to(device)
    if member == 0:
        print(f"  model: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    if tc.get("scheduler") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["max_epochs"])
    elif tc.get("scheduler") == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=tc.get("plateau_patience", 5))
    else:
        sched = None
    amp = tc.get("amp", False) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    y_tr = torch.from_numpy(y_z)
    size, bs, w_mean = tc["crop_size"], tc["batch_size"], tc.get("field_mean_weight", 0.0)

    history, best, bad = [], {"score": np.inf, "epoch": -1, "state": None}, 0
    for epoch in range(tc["max_epochs"]):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        order = rng.permutation(np.repeat(np.arange(len(fields[0])), tc.get("crops_per_field", 1)))
        for i in range(0, len(order), bs):
            crops = [fields[0].crop(f, size, rng, tc.get("augment", True)) for f in order[i:i + bs]]
            pos, img, r, c = collate(crops)
            xd, xs = to_device(Xd, Xs, pos, device)
            img_t = torch.from_numpy(img).long().to(device)
            yb = y_tr[torch.from_numpy(pos)].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                pred = model(xd, xs, img_t, torch.from_numpy(r).long().to(device),
                             torch.from_numpy(c).long().to(device), len(crops), size, size,
                             tc.get("encoder_chunk"), tc.get("checkpoint_encoder", False)).float()
            loss = torch.mean((pred - yb) ** 2)
            if w_mean:
                k = torch.zeros(len(crops), device=device).index_add_(0, img_t, torch.ones_like(yb))
                err = torch.zeros(len(crops), device=device).index_add_(0, img_t, pred - yb) / k
                loss = loss + w_mean * torch.mean(err ** 2)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(crops)
            n += len(crops)
        pred = predict(model, X[1], fields[1], cfg, device) * scal[1][1] + scal[1][0]
        err2 = (pred - y_true[1]) ** 2
        rec = {"member": member, "epoch": epoch, "train_loss": total / n, "val_mse": float(err2.mean()),
               "val_field_mse": field_balanced_mse(err2, codes[1]), "lr": opt.param_groups[0]["lr"],
               "epoch_time_s": time.time() - t0}
        history.append(rec)
        if run:
            run.log({f"m{member}/{k}": v for k, v in rec.items() if k != "member"})
        improved = rec["val_field_mse"] < best["score"]
        print(f"  m{member} epoch {epoch:3d} train_loss {rec['train_loss']:.4f} val_mse {rec['val_mse']:.4f} "
              f"val_field_mse {rec['val_field_mse']:.4f} lr {rec['lr']:.1e} ({rec['epoch_time_s']:.0f} s)"
              f"{' *' if improved else ''}", flush=True)
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(rec["val_field_mse"])
        elif sched:
            sched.step()
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
    torch.backends.cudnn.benchmark = False  # image and pixel counts change with every batch
    prep = v2.prepare_split(split_name, cfg, data)
    idx, X, scal, codes = prep["idx"], prep["X"], prep["scal"], prep["codes"]
    rows, cols = meta["row"].to_numpy(), meta["col"].to_numpy()
    fields = {r: FieldSet(rows[idx[r]], cols[idx[r]], codes[r]) for r in (0, 1, 2)}
    sides = np.maximum(fields[0].h, fields[0].w)
    print(f"  train fields {len(fields[0])}: median side {np.median(sides):.0f} px, "
          f"{(sides <= tc['crop_size']).mean():.0%} fit in one crop of {tc['crop_size']}", flush=True)

    gb = sum(t.element_size() * t.nelement() for t in X[0]) / 1e9
    if str(device).startswith("cuda") and gb <= tc["gpu_data_max_gb"]:
        X[0] = tuple(t.to(device) for t in X[0])
    print(f"  training data {gb:.1f} GB on {X[0][0].device}", flush=True)

    run = v2.init_wandb(cfg, split_name, len(idx[0])) if use_wandb else None
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = v2.pred_frames(meta, idx)
    history, bests = [], []
    for m in range(cfg.get("ensemble", {}).get("n_members", 1)):
        model, hist, best = train_member(m, X, fields, prep["y_z"], prep["y_true"], codes, scal, cfg, device, run)
        history += hist
        bests.append({"member": m, "best_epoch": best["epoch"], "best_val_field_mse": best["score"],
                      "epochs_run": len(hist)})
        torch.save(model.state_dict(), out_dir / f"model_m{m}.pt")
        for r in (1, 2):
            preds[r][f"pred_m{m}"] = predict(model, X[r], fields[r], cfg, device) * scal[r][1] + scal[r][0]
        del model
    del X, prep["X"]
    torch.cuda.empty_cache()
    v2.save_results(split_name, cfg, prep, meta, preds, history, bests, out_dir, run)


if __name__ == "__main__":
    v2.main(run_split, "field_image_s2.yaml")
