from __future__ import annotations
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, horizons: list[int]):
        super().__init__()
        self.horizons = horizons
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict({str(h): nn.Linear(d_model, 3) for h in horizons})

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        pooled = self.norm(h.mean(dim=1))
        logits_by_horizon = {horizon: self.heads[str(horizon)](pooled) for horizon in self.horizons}
        return pooled, logits_by_horizon
