"""Unit checks for convlstm.ConvLSTMCell / ConvLSTMModel; no data, no training.

    python scripts/LSTM/check_convlstm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from convlstm import ConvLSTMCell, ConvLSTMModel  # noqa: E402

B, T, F, S = 16, 24, 16, 9


def check_cell_equals_lstmcell() -> None:
    """kernel 1 on a 1 x 1 map = torch.nn.LSTMCell with the same weights (gate order i, f, g, o)."""
    torch.manual_seed(0)
    cin, ch = 7, 5
    cell, ref = ConvLSTMCell(cin, ch, kernel_size=1), torch.nn.LSTMCell(cin, ch)
    with torch.no_grad():
        cell.conv.weight.copy_(torch.cat([ref.weight_ih, ref.weight_hh], dim=1)[:, :, None, None])
        cell.conv.bias.copy_(ref.bias_ih + ref.bias_hh)
    x = torch.randn(T, B, cin)
    h, c = torch.zeros(B, ch, 1, 1), torch.zeros(B, ch, 1, 1)
    hr, cr = torch.zeros(B, ch), torch.zeros(B, ch)
    for t in range(T):
        h, c = cell(x[t][:, :, None, None], (h, c))
        hr, cr = ref(x[t], (hr, cr))
    assert torch.allclose(h[:, :, 0, 0], hr, atol=1e-5), (h[:, :, 0, 0] - hr).abs().max()
    assert torch.allclose(c[:, :, 0, 0], cr, atol=1e-5)
    print("cell == LSTMCell: ok")


def make(k: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(k)
    xd = torch.randn(B, k * k, T, F + 1, generator=g)
    present = (torch.rand(B, k * k, 1, 1, generator=g) > 0.2).float()
    present[:, k * k // 2] = 1.0
    xd = xd * present
    xd[..., F] = present[..., 0].expand(B, k * k, T)
    return xd, torch.randn(B, S, generator=g)


def check_model(k: int) -> None:
    torch.manual_seed(0)
    for layers in (1, 2):
        model = ConvLSTMModel(F, S, k=k, hidden_channels=32, num_layers=layers, dropout=0.2)
        xd, xs = make(k)
        model.train()
        y = model(xd, xs)
        assert y.shape == (B,), y.shape
        y.pow(2).mean().backward()
        for n, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
        # all neighbours absent: present channel 0, inputs 0 -> still a finite output
        model.eval()
        out = model(torch.zeros_like(xd), xs)
        assert out.shape == (B,) and torch.isfinite(out).all()
        n_par = sum(p.numel() for p in model.parameters())
        print(f"k={k} layers={layers}: ok, {n_par:,} parameters")
        if k == 5 and layers == 1:
            assert n_par < 200_000, n_par


if __name__ == "__main__":
    check_cell_equals_lstmcell()
    check_model(5)
    check_model(9)
