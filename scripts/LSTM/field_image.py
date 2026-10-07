"""Field-image model: every field-year is an image and the whole yield map is predicted in one pass
(plan: .claude/plans/field_image_model.md).

Two stages, so the expensive temporal part only runs on labelled pixels:
1. a temporal encoder (LSTM as in LSTMv2, or a small transformer) turns every pixel's time series + static
   inputs into an embedding;
2. the embeddings are scattered into a (E, H, W) image (zeros where there is no labelled pixel), together with
   the static inputs and a mask channel, and a small 2D U-Net predicts one yield value per pixel.

Images are built on the fly from the per-pixel inputs of FeatureSpec.transform (no new extraction): pixels of a
field are contiguous, their grid position is (row - row_min, col - col_min) of the field.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


# --------------------------------------------------------------------------- field images

class FieldSet:
    """The fields of one set (train, val or test). rows/cols: grid position of every pixel of the set, codes:
    field code per pixel (0..n_fields-1, pixels of a field contiguous, as pd.factorize gives them)."""

    def __init__(self, rows: np.ndarray, cols: np.ndarray, codes: np.ndarray):
        codes = np.asarray(codes)
        self.start = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        self.end = np.r_[self.start[1:], len(codes)]
        if len(self.start) != codes.max() + 1 or not (codes[self.start] == np.arange(len(self.start))).all():
            raise ValueError("pixels of a field must be contiguous and codes in order of appearance")
        rows, cols = np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)
        r0, c0 = np.minimum.reduceat(rows, self.start), np.minimum.reduceat(cols, self.start)
        self.lr, self.lc = rows - r0[codes], cols - c0[codes]   # position inside the field's bounding box
        self.h = np.maximum.reduceat(self.lr, self.start) + 1
        self.w = np.maximum.reduceat(self.lc, self.start) + 1

    def __len__(self) -> int:
        return len(self.start)

    def crop(self, f: int, size: int, rng: np.random.Generator, augment: bool = False):
        """Random size x size window of field f that contains a random labelled pixel (so it is never empty).
        Larger fields are cut, smaller ones placed at a random position (zero-padded). With augment, a random
        90 degree rotation and flip. Returns (set positions of the pixels inside, window row, window col)."""
        a, b = self.start[f], self.end[f]
        lr, lc = self.lr[a:b], self.lc[a:b]
        p = rng.integers(b - a)
        r0 = _offset(lr[p], self.h[f], size, rng)
        c0 = _offset(lc[p], self.w[f], size, rng)
        wr, wc = lr - r0, lc - c0
        keep = (wr >= 0) & (wr < size) & (wc >= 0) & (wc < size)
        pos, wr, wc = a + np.flatnonzero(keep), wr[keep], wc[keep]
        if augment:
            for _ in range(rng.integers(4)):     # rotate by 90 degrees
                wr, wc = wc, size - 1 - wr
            if rng.random() < 0.5:               # with the rotations: all 8 symmetries of the square
                wc = size - 1 - wc
        return pos, wr, wc

    def tiles(self, f: int, multiple: int, tile: int, overlap: int):
        """Windows covering field f for inference: the whole field padded to a multiple of `multiple`, or, if a
        side is longer than `tile`, overlapping tile x tile windows. Yields (set positions, row, col, H, W)."""
        a, b = self.start[f], self.end[f]
        lr, lc = self.lr[a:b], self.lc[a:b]
        for r0, hh in _windows(self.h[f], multiple, tile, overlap):
            for c0, ww in _windows(self.w[f], multiple, tile, overlap):
                wr, wc = lr - r0, lc - c0
                keep = (wr >= 0) & (wr < hh) & (wc >= 0) & (wc < ww)
                if keep.any():
                    yield a + np.flatnonzero(keep), wr[keep], wc[keep], hh, ww


def _offset(p: int, length: int, size: int, rng: np.random.Generator) -> int:
    """Window start along one axis: inside [min(0, L - S), max(0, L - S)] (covers as much of the field as
    possible) and in [p - S + 1, p] (contains position p). The intersection is never empty."""
    lo = max(min(0, length - size), p - size + 1)
    hi = min(max(0, length - size), p)
    return int(rng.integers(lo, hi + 1))


def _windows(length: int, multiple: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    padded = -(-length // multiple) * multiple
    if padded <= tile:
        return [(0, padded)]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride)) + [length - tile]
    return [(s, tile) for s in starts]


def collate(crops) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """List of (positions, row, col) -> concatenated positions, image index, row, col."""
    pos = np.concatenate([c[0] for c in crops])
    img = np.repeat(np.arange(len(crops)), [len(c[0]) for c in crops])
    return pos, img, np.concatenate([c[1] for c in crops]), np.concatenate([c[2] for c in crops])


# --------------------------------------------------------------------------- model

class TemporalEncoder(nn.Module):
    """Per-pixel time series (P, T, F) + static (P, S) -> embedding (P, E).

    lstm:        LSTM (static inputs repeated at every step as in LSTMv2), last hidden state -> linear.
    transformer: linear projection + learned position embedding, transformer layers, mean over observed steps
                 (obs_index: channel of the observed indicator, None = all steps)."""

    def __init__(self, n_dyn: int, n_static: int, n_steps: int, embed: int, type: str = "lstm",
                 hidden_size: int = 128, num_layers: int = 2, heads: int = 4, dropout: float = 0.0,
                 obs_index: int | None = None):
        super().__init__()
        self.type, self.obs_index = type, obs_index
        n_in = n_dyn + n_static
        if type == "lstm":
            self.lstm = nn.LSTM(n_in, hidden_size, num_layers=num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
        elif type == "transformer":
            self.proj = nn.Linear(n_in, hidden_size)
            self.pos = nn.Parameter(torch.zeros(n_steps, hidden_size))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(hidden_size, heads, 2 * hidden_size, dropout, batch_first=True,
                                               norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        else:
            raise ValueError(f"unknown encoder type {type}")
        self.out = nn.Linear(hidden_size, embed)

    def forward(self, xd: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([xd, xs[:, None, :].expand(-1, xd.shape[1], -1)], dim=-1) if xs.shape[-1] else xd
        if self.type == "lstm":
            _, (h, _) = self.lstm(x)
            return self.out(h[-1])
        z = self.transformer(self.proj(x) + self.pos)
        if self.obs_index is None:
            return self.out(z.mean(1))
        w = xd[..., self.obs_index:self.obs_index + 1]
        return self.out((z * w).sum(1) / w.sum(1).clamp(min=1))


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Plain U-Net, len(channels) levels; image sides must be multiples of 2 ** (len(channels) - 1).
    BatchNorm (not GroupNorm): in eval it uses running statistics, so the output does not depend on the image
    size, which differs between training crops and whole fields."""

    def __init__(self, c_in: int, channels: list[int], dropout: float = 0.0):
        super().__init__()
        self.down = nn.ModuleList()
        c = c_in
        for ch in channels:
            self.down.append(_conv_block(c, ch))
            c = ch
        self.drop = nn.Dropout(dropout)
        self.upsample = nn.ModuleList()
        self.up = nn.ModuleList()
        for ch in reversed(channels[:-1]):
            self.upsample.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.up.append(_conv_block(2 * ch, ch))
            c = ch
        self.head = nn.Conv2d(c, 1, 1)

    @property
    def multiple(self) -> int:
        return 2 ** (len(self.down) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            if i:
                x = nn.functional.max_pool2d(x, 2)
            x = block(x)
            skips.append(x)
        x = self.drop(x)
        for upsample, block, skip in zip(self.upsample, self.up, reversed(skips[:-1])):
            x = block(torch.cat([upsample(x), skip], dim=1))
        return self.head(x)[:, 0]


class FieldImageModel(nn.Module):
    """Temporal encoder per labelled pixel -> scatter into an image with the static inputs and a mask channel
    -> U-Net -> one value per pixel, read back at the labelled pixels."""

    def __init__(self, n_dyn: int, n_static: int, n_steps: int, encoder: dict, embed: int = 64,
                 unet_channels: list[int] = (32, 64, 128, 256), dropout: float = 0.0,
                 obs_index: int | None = None):
        super().__init__()
        self.encoder = TemporalEncoder(n_dyn, n_static, n_steps, embed, obs_index=obs_index, **encoder)
        self.unet = UNet(embed + n_static + 1, list(unet_channels), dropout)

    def encode(self, xd: torch.Tensor, xs: torch.Tensor, chunk: int | None = None,
               use_checkpoint: bool = False) -> torch.Tensor:
        """Embeddings of P pixels, in chunks of `chunk` pixels; with use_checkpoint the encoder's activations
        are recomputed in the backward pass instead of kept (a batch of crops can hold >100k pixels)."""
        chunk = chunk or len(xd)
        if len(xd) <= chunk and not use_checkpoint:
            return self.encoder(xd, xs)
        out = []
        for i in range(0, len(xd), chunk):
            a, b = xd[i:i + chunk], xs[i:i + chunk]
            out.append(checkpoint(self.encoder, a, b, use_reentrant=False) if use_checkpoint else self.encoder(a, b))
        return torch.cat(out)

    def spatial(self, emb: torch.Tensor, xs: torch.Tensor, img: torch.Tensor, r: torch.Tensor, c: torch.Tensor,
                n_img: int, h: int, w: int) -> torch.Tensor:
        """Pixels (embedding, static) at (img, r, c) -> images (n_img, C, h, w) -> U-Net -> value per pixel."""
        feats = torch.cat([emb, xs.to(emb.dtype), torch.ones_like(emb[:, :1])], dim=1)
        grid = feats.new_zeros((n_img, h, w, feats.shape[1]))
        grid[img, r, c] = feats
        out = self.unet(grid.permute(0, 3, 1, 2).contiguous())
        return out[img, r, c]

    def forward(self, xd, xs, img, r, c, n_img: int, h: int, w: int, chunk: int | None = None,
                use_checkpoint: bool = False) -> torch.Tensor:
        return self.spatial(self.encode(xd, xs, chunk, use_checkpoint), xs, img, r, c, n_img, h, w)
