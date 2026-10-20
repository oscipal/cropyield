"""Convolutional LSTM on the k x k patch of a pixel (Shi et al. 2015): the LSTM gates are convolutions, so space
and time are mixed in the same layer. Same interface as lstm_v2.PatchLSTM, predicts the centre pixel only.

Used by scripts/LSTM/04_train_v2.py with model.type: convlstm (configs/convlstm_s2.yaml).
"""
from __future__ import annotations

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """One ConvLSTM step. Gates from one conv over [x_t, h_{t-1}], split in the order i, f, g, o (as
    torch.nn.LSTMCell, so a 1 x 1 kernel on a 1 x 1 map is exactly an LSTMCell)."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size,
                              padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        i, f, g, o = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMModel(nn.Module):
    """Pixel with its k x k neighbourhood through `num_layers` ConvLSTM layers; the last layer's hidden map at
    the last step is summarised as centre + mean over the neighbours present, plus the static inputs, and goes
    through the head of LSTMv2 (FC -> BN -> ReLU -> Dropout -> FC 1).

    Input xd (B, K=k*k, T, C) from gather_patches (last channel = neighbour present), xs (B, S)."""

    def __init__(self, n_dyn: int, n_static: int, k: int = 5, hidden_channels: int = 32, num_layers: int = 1,
                 kernel_size: int = 3, head_size: int = 128, dropout: float = 0.0):
        super().__init__()
        self.k = k
        c = n_dyn + 1  # + neighbour-present channel
        self.cells = nn.ModuleList(
            ConvLSTMCell(c if i == 0 else hidden_channels, hidden_channels, kernel_size) for i in range(num_layers))
        self.drop = nn.Dropout(dropout)
        n = 2 * hidden_channels + n_static
        self.head = nn.Sequential(nn.Linear(n, head_size), nn.BatchNorm1d(head_size), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(head_size, 1))

    def forward(self, xd: torch.Tensor, xs: torch.Tensor | None = None) -> torch.Tensor:
        B, K, T, C = xd.shape
        seq = xd.permute(0, 2, 3, 1).reshape(B, T, C, self.k, self.k)   # (B, T, C, k, k)
        present = seq[:, 0, -1:]                                         # (B, 1, k, k), same at every step
        for li, cell in enumerate(self.cells):
            if li > 0:
                seq = self.drop(seq)
            h = seq.new_zeros(B, cell.hidden_channels, self.k, self.k)
            c = torch.zeros_like(h)
            out = []
            for t in range(T):
                h, c = cell(seq[:, t], (h, c))
                out.append(h)
            seq = torch.stack(out, dim=1)
        centre = h[:, :, self.k // 2, self.k // 2]
        mean = (h * present).sum((2, 3)) / present.sum((2, 3)).clamp(min=1)
        feats = [centre, mean] + ([xs] if xs is not None and xs.shape[-1] > 0 else [])
        return self.head(torch.cat(feats, dim=-1)).squeeze(-1)
