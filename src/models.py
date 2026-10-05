"""LSTM regressor as in the official YieldSAT tutorial."""
from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Input (batch, time, bands) -> LSTM -> last hidden state -> linear -> yield."""

    def __init__(self, n_bands: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_bands, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)
