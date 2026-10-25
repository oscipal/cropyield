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


def check_late() -> None:
    """late_cols: the branch sees only the centre pixel's late channels, the ConvLSTM never sees them, gradients
    reach the GRU, modality dropout 1.0 removes the branch's effect in training mode."""
    torch.manual_seed(0)
    late = [12, 13, 14]
    model = ConvLSTMModel(F, S, k=5, late_cols=late, late_hidden=8, late_dropout=0.0)
    xd, xs = make(5)
    model.train()
    y = model(xd, xs)
    y.pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.cells[0].conv.in_channels == F + 1 - len(late) + 32
    model.eval()
    base = model(xd, xs)
    x2 = xd.clone()
    x2[:, :12, :, late] += 5.0                  # change the late channels of the non-centre pixels only
    assert torch.allclose(model(x2, xs), base, atol=1e-5), "late branch must read the centre pixel only"
    x3 = xd.clone()
    x3[:, 12, :, late] += 5.0                   # centre pixel's late channels: the output must change
    assert not torch.allclose(model(x3, xs), base, atol=1e-4)
    model.late_dropout = 1.0
    model.train()
    torch.manual_seed(1)
    a = model(xd, xs)
    torch.manual_seed(1)
    b = model(x3, xs)
    assert torch.allclose(a, b, atol=1e-5), "late_dropout 1 must switch the branch off"
    print(f"late fusion: ok, {sum(p.numel() for p in model.parameters()):,} parameters")


def check_augment_drop_cols() -> None:
    """step_dropout with drop_cols leaves the other channels alone; without drop_cols the old behaviour."""
    from lstm_v2 import augment_batch
    xd, _ = make(5)
    obs = 15
    xd[..., obs] = 1.0
    g = torch.Generator().manual_seed(0)
    out = augment_batch(xd.clone(), {"step_dropout": 0.5}, g, obs, [], F, 0.0, drop_cols=list(range(12)) + [obs])
    dropped = out[:, 12, :, obs] == 0                                  # centre pixel's dropped steps
    assert dropped.float().mean().item() > 0.3
    assert torch.equal(out[..., 12:15], xd[..., 12:15]), "channels outside drop_cols must be unchanged"
    assert (out[:, 12, :, :12][dropped] == 0).all()
    g = torch.Generator().manual_seed(0)
    old = augment_batch(xd.clone(), {"step_dropout": 0.5}, g, obs, [], F, 0.0)
    assert (old[:, 12, :, 12:15][old[:, 12, :, obs] == 0] == 0).all(), "default drops every channel"
    print("augment drop_cols: ok")


if __name__ == "__main__":
    check_cell_equals_lstmcell()
    check_model(5)
    check_model(9)
    check_late()
    check_augment_drop_cols()
