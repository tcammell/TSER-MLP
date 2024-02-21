from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    def __init__(
            self,
            ch_in: int,
            seq_len: int,
            activation: Optional[str] = "relu",
            dropout: Optional[float] = 0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=[seq_len, ch_in])
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features=seq_len, out_features=seq_len)
        self.activation = (
            None if activation is None else getattr(F, activation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        x = x.permute(0, 2, 1)
        x = res + x
        return x


class FeaturalBlock(nn.Module):
    def __init__(
            self,
            ch_in: int,
            seq_len: int,
            hidden_size: int,
            activation: Optional[str] = "relu",
            dropout: Optional[float] = 0,
    ):
        super().__init__()
        self.activation = (
            None if activation is None else getattr(F, activation)
        )
        self.norm = nn.LayerNorm(normalized_shape=[seq_len, ch_in])
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(in_features=ch_in, out_features=hidden_size)
        self.dropout2 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=ch_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        x = self.dropout1(x)
        x = self.linear1(x)
        if self.activation:
            x = self.activation(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        return res + x


class TSMixBlock(nn.Module):
    def __init__(
            self,
            ch_in: int,
            seq_len: int,
            hidden_size: int,
            activation: str,
            dropout: float,
    ):
        super().__init__()
        self.temporal_block = TemporalBlock(ch_in, seq_len, activation, dropout)
        self.feature_block = FeaturalBlock(ch_in, seq_len, hidden_size, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_block(x)
        x = self.feature_block(x)
        return x
