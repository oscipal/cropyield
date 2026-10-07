"""Checks for the field-image model (field_image.py); no training.

    python scripts/LSTM/check_field_image.py              data and shape checks on the CPU
    python scripts/LSTM/check_field_image.py --gpu        + peak GPU memory of a training step at S = 128 and 256

1. Field images from meta.parquet: every pixel gets its own grid cell, scatter -> gather returns the pixels in
   their original order with their labels, and the cell is (row - row_min, col - col_min) of the field.
2. Crops: never empty, inside the window, augmentation is a bijection of the window, small fields stay whole.
3. Inference tiles: cover every pixel of every field, also fields larger than one tile.
4. Model: output shape, parameter count, a training step with a backward pass, encoder chunking/checkpointing
   gives the same result as one call, predict() of 07 equals a direct forward pass of the whole field.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from field_image import FieldImageModel, FieldSet, collate  # noqa: E402
from src.config import REPO_ROOT, load_paths, load_yaml  # noqa: E402


def check_real_fields() -> FieldSet:
    meta = pd.read_parquet(load_paths()["work_dir"] / "lstm" / "data" / "meta.parquet",
                           columns=["field", "row", "col", "target"])
    codes = pd.factorize(meta["field"])[0]
    fs = FieldSet(meta["row"].to_numpy(), meta["col"].to_numpy(), codes)
    y = meta["target"].to_numpy()
    rows, cols = meta["row"].to_numpy().astype(int), meta["col"].to_numpy().astype(int)
    for f in range(len(fs)):
        a, b = fs.start[f], fs.end[f]
        lr, lc = fs.lr[a:b], fs.lc[a:b]
        assert (lr == rows[a:b] - rows[a:b].min()).all() and (lc == cols[a:b] - cols[a:b].min()).all()
        grid = np.full((fs.h[f], fs.w[f]), np.nan)
        owner = np.full((fs.h[f], fs.w[f]), -1)
        owner[lr, lc] = np.arange(b - a)
        assert (owner >= 0).sum() == b - a, f"field {f}: two pixels share a grid cell"
        grid[lr, lc] = y[a:b]
        assert np.array_equal(grid[lr, lc], y[a:b]) and (owner[lr, lc] == np.arange(b - a)).all()
    sides = np.maximum(fs.h, fs.w)
    print(f"1. {len(fs)} field images, {len(y):,} pixels: grid cells unique, scatter -> gather keeps order and "
          f"labels. Bounding box side median {np.median(sides):.0f}, max {sides.max()} px; "
          f"labelled share median {np.median((fs.end - fs.start) / (fs.h * fs.w)):.2f}")
    return fs


def check_crops(fs: FieldSet, size: int = 128, n: int = 3000) -> None:
    rng = np.random.default_rng(0)
    for f in rng.integers(len(fs), size=n):
        for aug in (False, True):
            pos, r, c = fs.crop(f, size, rng, augment=aug)
            assert len(pos) > 0
            assert (r >= 0).all() and (r < size).all() and (c >= 0).all() and (c < size).all()
            assert len(set(zip(r.tolist(), c.tolist()))) == len(pos), "two pixels in one cell"
            if max(fs.h[f], fs.w[f]) <= size:
                assert len(pos) == fs.end[f] - fs.start[f], "a field that fits must stay whole"
            if not aug:  # without augmentation the window is a shifted copy of the field
                a = fs.start[f]
                d = np.c_[r - fs.lr[pos], c - fs.lc[pos]]
                assert (d == d[0]).all()
                assert a <= pos.min() and pos.max() < fs.end[f]
    print(f"2. {n} random fields x (plain, augmented) crops of {size}: non-empty, inside the window, one pixel per "
          f"cell, small fields whole")


def check_tiles(fs: FieldSet, multiple: int = 8, tile: int = 256, overlap: int = 32) -> None:
    n_tiled = 0
    for f in range(len(fs)):
        n = fs.end[f] - fs.start[f]
        seen = np.zeros(n, dtype=int)
        k = 0
        for pos, r, c, h, w in fs.tiles(f, multiple, tile, overlap):
            assert h % multiple == 0 and w % multiple == 0 and h <= max(tile, 1) and w <= max(tile, 1)
            assert (r < h).all() and (c < w).all() and (r >= 0).all() and (c >= 0).all()
            seen[pos - fs.start[f]] += 1
            k += 1
        assert (seen > 0).all(), f"field {f}: pixels not covered"
        n_tiled += k > 1
    print(f"3. tiles ({tile}, overlap {overlap}) cover every pixel of all {len(fs)} fields; {n_tiled} fields "
          f"need more than one tile")


def small_model(n_dyn, n_static, enc_type="lstm"):
    enc = {"type": enc_type, "hidden_size": 32, "num_layers": 2, "heads": 4, "dropout": 0.1}
    return FieldImageModel(n_dyn, n_static, 24, enc, embed=16, unet_channels=[8, 16, 32], dropout=0.1,
                           obs_index=n_dyn - 1)


def synthetic_fields(rng, sizes):
    """Fields as random blobs inside bounding boxes of the given sizes."""
    rows, cols, codes = [], [], []
    for i, (h, w) in enumerate(sizes):
        m = rng.random((h, w)) < 0.6
        m[0, 0] = m[h - 1, w - 1] = True
        r, c = np.nonzero(m)
        rows.append(r + 1000)
        cols.append(c + 50)
        codes.append(np.full(len(r), i))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(codes)


def check_model() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n_dyn, n_static = 16, 6
    rows, cols, codes = synthetic_fields(rng, [(20, 30), (70, 40), (300, 90)])
    fs = FieldSet(rows, cols, codes)
    n = len(rows)
    xd = torch.randn(n, 24, n_dyn)
    xd[..., -1] = (torch.rand(n, 24) < 0.7).float()
    xs = torch.randn(n, n_static)
    y = torch.randn(n)

    for enc_type in ("lstm", "transformer"):
        model = small_model(n_dyn, n_static, enc_type)
        crops = [fs.crop(f, 64, rng, augment=True) for f in (0, 1, 2, 2)]
        pos, img, r, c = collate(crops)
        p = torch.from_numpy(pos)
        model.train()
        out = model(xd[p], xs[p], torch.from_numpy(img), torch.from_numpy(r).long(), torch.from_numpy(c).long(),
                    len(crops), 64, 64, chunk=100, use_checkpoint=True)
        assert out.shape == (len(pos),)
        torch.mean((out - y[p]) ** 2).backward()
        assert all(q.grad is not None for q in model.parameters() if q.requires_grad), "unused parameters"
        model.eval()
        with torch.no_grad():
            e1 = model.encode(xd[:500], xs[:500])
            e2 = model.encode(xd[:500], xs[:500], chunk=77, use_checkpoint=True)
        assert torch.allclose(e1, e2, atol=1e-5)
        print(f"4. {enc_type}: forward/backward on {len(crops)} crops ({len(pos)} pixels) ok, chunked encoder "
              f"identical, {sum(q.numel() for q in model.parameters()):,} parameters (small test model)")

    # predict() of 07 against one forward pass of the whole (untiled) field
    t = importlib.import_module("07_train_field_image")
    cfg = {"train": {"amp": False}, "predict": {"tile": 512, "overlap": 32, "batch_pixels": 333}}
    model = small_model(n_dyn, n_static).eval()
    pred = t.predict(model, (xd.half(), xs.half()), fs, cfg, "cpu")
    with torch.no_grad():
        f = 1
        a, b = fs.start[f], fs.end[f]
        mult = model.unet.multiple
        h, w = -(-fs.h[f] // mult) * mult, -(-fs.w[f] // mult) * mult
        direct = model(xd[a:b].half().float(), xs[a:b].half().float(), torch.zeros(b - a, dtype=torch.long),
                       torch.from_numpy(fs.lr[a:b]).long(), torch.from_numpy(fs.lc[a:b]).long(), 1, h, w)
    assert np.allclose(pred[a:b], direct.numpy(), atol=1e-4)
    cfg["predict"]["tile"] = 64  # forces tiling of fields 1 and 2
    pred_tiled = t.predict(model, (xd.half(), xs.half()), fs, cfg, "cpu")
    assert np.isfinite(pred_tiled).all()
    print(f"   predict(): equals a direct forward pass of the whole field; tiled prediction finite "
          f"(max |tiled - whole| {np.abs(pred_tiled - pred).max():.3f})")


def check_gpu(config: str) -> None:
    """Peak memory of one training step with the configured model, random inputs, fully labelled crops
    (worst case: a crop inside a large Brazilian field)."""
    cfg = load_yaml(config)
    t = importlib.import_module("07_train_field_image")
    tc = cfg["train"]
    n_dyn, n_static = 16, 7  # 12 S2 bands + 3 indices + observed; crop and country one-hot
    model = t.build_model(cfg, n_dyn, n_static, 24, n_dyn - 1).cuda()
    print(f"GPU check, {cfg['name']}: {sum(q.numel() for q in model.parameters()):,} parameters")
    opt = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=tc["amp"])
    for size in (128, 256):
        bs = tc["batch_size"]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rr, cc = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        r = torch.from_numpy(np.tile(rr.ravel(), bs)).cuda()
        c = torch.from_numpy(np.tile(cc.ravel(), bs)).cuda()
        img = torch.arange(bs).repeat_interleave(size * size).cuda()
        xd = torch.randn(len(r), 24, n_dyn, device="cuda")
        xs = torch.randn(len(r), n_static, device="cuda")
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=tc["amp"]):
                out = model(xd, xs, img, r, c, bs, size, size, tc["encoder_chunk"], tc["checkpoint_encoder"])
            scaler.scale(out.float().pow(2).mean()).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            print(f"  crop {size}, batch {bs} ({len(r):,} pixels): peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        except torch.OutOfMemoryError:
            print(f"  crop {size}, batch {bs} ({len(r):,} pixels): out of memory")
        del xd, xs, r, c, img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="also measure GPU memory of a training step")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "field_image_s2.yaml"))
    args = ap.parse_args()
    fs = check_real_fields()
    check_crops(fs)
    check_tiles(fs)
    check_model()
    if args.gpu:
        check_gpu(args.config)
    print("all checks passed")


if __name__ == "__main__":
    main()
