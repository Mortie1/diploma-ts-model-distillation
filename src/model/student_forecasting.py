from __future__ import annotations

import torch
from torch import nn


class StudentForecaster(nn.Module):
    """Light student forecaster with encoder backbone + linear horizon head."""

    def __init__(
        self,
        in_channels: int,
        horizon: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.in_channels = in_channels

        self.feature = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(hidden_dim, in_channels * horizon)

    def forward(self, inputs: torch.Tensor, **batch):
        # inputs: [B, C, Tc]
        feats = self.feature(inputs).transpose(1, 2)
        encoded = self.encoder(feats)
        pooled = encoded.mean(dim=1)

        forecast_flat = self.head(pooled)
        forecast = forecast_flat.view(-1, self.in_channels, self.horizon)
        return {
            "forecast": forecast,
            "student_hidden": pooled,
            "student_pred": forecast_flat,
        }
