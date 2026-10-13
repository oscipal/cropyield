"""Unit checks for lstm_v2.augment_batch; no data, no training.

    python scripts/LSTM/check_augment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lstm_v2 import augment_batch  # noqa: E402

B, K, T, F = 64, 25, 24, 8
OBS, JITTER = F - 1, [0, 1, 2, 3]  # columns 4-6 stand for weather / indices that must stay untouched


def make(patch: bool, fill: float = 0.0):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, K, T, F + 1, generator=g) + 5.0
    obs = (torch.rand(B, 1, T, generator=g) > 0.2).float().expand(B, K, T).clone()
    x[..., OBS] = obs
    x[..., F] = 1.0
    x[..., :OBS] *= obs[..., None]  # missing steps hold the fill value 0
    return x if patch else x[:, K // 2, :, :F].clone()


def check(patch: bool) -> None:
    gen = torch.Generator().manual_seed(1)
    x = make(patch)
    tag = "patch" if patch else "pixel"
    args = (OBS, JITTER, F, 0.0)
    assert torch.equal(augment_batch(x, {}, gen, *args), x), "empty aug must be identity"

    for aug in ({"time_shift": 2}, {"step_dropout": 0.3}, {"spectral_noise": 0.5},
                {"time_shift": 1, "step_dropout": 0.1, "spectral_noise": 0.1}):
        y = augment_batch(x, aug, gen, *args)
        assert y.shape == x.shape, (tag, aug)
        if patch:
            assert torch.equal(y[..., F:], x[..., F:]), "neighbour channel changed"
        obs_y = y[..., OBS]
        assert (y[..., :F][obs_y == 0] == 0).all() or "spectral_noise" in aug and len(aug) == 1, (tag, aug)
        if list(aug) == ["spectral_noise"]:
            assert torch.equal(y[..., 4:OBS], x[..., 4:OBS]), "non-spectral channels jittered"
            assert torch.equal(y[..., OBS], x[..., OBS]), "mask changed by noise"
            assert (y[..., :4][x[..., OBS] == 0] == 0).all(), "noise on missing steps"
            assert not torch.equal(y[..., :4], x[..., :4])
            # one noise value per sample and channel, shared over time and neighbours
            d = (y - x)[..., :4] * x[..., OBS:OBS + 1]
            cen = x[..., OBS] > 0
            for b in range(4):
                v = d[b][cen[b]] if patch else d[b][cen[b]]
                assert torch.allclose(v, v[0].expand_as(v), atol=1e-5)

    # dropout: only observed steps dropped, rate near p, same steps for all neighbours
    y = augment_batch(x, {"step_dropout": 0.3}, gen, *args)
    dropped = (x[..., OBS] == 1) & (y[..., OBS] == 0)
    assert ((x[..., OBS] == 0) <= (y[..., OBS] == 0)).all()
    rate = dropped.sum().item() / (x[..., OBS] == 1).sum().item()
    assert 0.2 < rate < 0.4, rate
    if patch:
        assert (dropped == dropped[:, :1]).all(), "neighbours dropped differently"

    # shift: result equals a roll of the original with empty steps shifted in
    y = augment_batch(x, {"time_shift": 2}, gen, *args)
    shifts = set()
    for b in range(B):
        for k in range(-2, 3):
            ref = torch.zeros_like(x[b][..., :F]) if not patch else torch.zeros_like(x[b][..., :F])
            lo, hi = max(0, k), min(T, T + k)
            ref[..., lo:hi, :] = x[b][..., lo - k:hi - k, :F]
            if torch.equal(y[b][..., :F], ref):
                shifts.add(k)
                break
        else:
            raise AssertionError(f"{tag}: sample {b} is not a shift")
    assert shifts == {-2, -1, 0, 1, 2}, shifts

    # no mask channel: shapes kept, nothing breaks
    z = x[..., :F - 1] if not patch else torch.cat([x[..., :F - 1], x[..., F:]], -1)
    out = augment_batch(z, {"time_shift": 1, "step_dropout": 0.1, "spectral_noise": 0.1}, gen, None, JITTER, F - 1)
    assert out.shape == z.shape
    print(f"{tag}: ok (dropout rate {rate:.2f}, shifts {sorted(shifts)})")


if __name__ == "__main__":
    check(False)
    check(True)
