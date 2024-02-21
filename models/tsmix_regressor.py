from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .gmlp import gMLP
from .revin import RevIN
from .tsmixer import TemporalBlock, TSMixBlock

class TSMixRegressor(nn.Module):
    def __init__(
            self,
            n_block: int,
            ch_in: int,
            seq_len: int,
            hidden_size: int,
            activation: str = "relu",
            dropout: float = 0.2,
            revin: Optional[bool] = False,
            univariate: Optional[bool] = False,
            gmlp_proj: Optional[bool] = False,
            gmlp_blocks: Optional[int] = 2,
            gmlp_patch_size: Optional[int] = 1,
            gmlp_d_model: Optional[int] = 256,
            gmlp_d_ffn: Optional[int] = 256,
    ):
        super().__init__()
        self.n_block = n_block
        if n_block > 0:
            if univariate:
                self.mixerblocks = nn.Sequential(
                    *[TemporalBlock(ch_in, seq_len, activation=activation, dropout=dropout)
                      for _ in range(n_block)]
                )
            else:
                self.mixerblocks = nn.Sequential(
                    *[TSMixBlock(ch_in, seq_len, hidden_size, activation, dropout)
                      for _ in range(n_block)]
                )
        if revin:
            self.revin = RevIN(num_features=ch_in)
        else:
            self.revin = None

        self.gmlp_proj = gmlp_proj
        if gmlp_proj:
            self.projection = gMLP(
                ch_in=ch_in, ch_out=1, seq_len=seq_len, n_blocks=gmlp_blocks,
                patch_size=gmlp_patch_size, d_model=gmlp_d_model, d_ffn=gmlp_d_ffn)
        else:
            self.projection = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features=seq_len, out_features=seq_len),
                nn.ReLU(),
                nn.Linear(in_features=seq_len, out_features=1),
            )

    def forward(self, x) -> torch.Tensor:
        if self.n_block > 0:
            x = torch.permute(x, (0, 2, 1))
            if self.revin:
                x = self.revin(x, 'norm')
            x = self.mixerblocks(x)
            if self.revin:
                x = self.revin(x, 'denorm')
            x = torch.permute(x, (0, 2, 1))

        if self.gmlp_proj:
            x = self.projection(x)
        else:
            x = x.mean(dim=1)
            x = self.projection(x)
        return x


class TSMixRegressorModule(L.LightningModule):
    def __init__(
            self,
            n_block: int,
            ch_in: int,
            seq_len: int,
            hidden_size: int = 256,
            activation: str = "relu",
            dropout: float = 0.2,
            revin: Optional[bool] = False,
            univariate: Optional[bool] = False,
            gmlp_proj: Optional[bool] = False,
            gmlp_blocks: Optional[int] = 2,
            gmlp_patch_size: int = 1,
            gmlp_d_model: int = 256,
            gmlp_d_ffn: int = 256,
            lr: float = 1e-3,
            lr_patience: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSMixRegressor(
            n_block=n_block,
            ch_in=ch_in,
            seq_len=seq_len,
            hidden_size=hidden_size,
            activation=activation,
            dropout=dropout,
            revin=revin,
            univariate=univariate,
            gmlp_proj=gmlp_proj,
            gmlp_blocks=gmlp_blocks,
            gmlp_patch_size=gmlp_patch_size,
            gmlp_d_model=gmlp_d_model,
            gmlp_d_ffn=gmlp_d_ffn,
        ).float()

        self.lr = lr
        self.lr_patience = lr_patience

    def _compute_loss(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        mse = F.mse_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        rmse = torch.sqrt(mse)
        # rmsep = (torch.sqrt(torch.mean(torch.square((y - y_hat) / y)))) * 100
        return mse, rmse, mae

    def training_step(self, batch, batch_idx):
        mse, rmse, mae = self._compute_loss(batch, batch_idx)
        self.log_dict({"train_mse": mse, "train_rmse": rmse, "train_mae": mae}, prog_bar=True)
        return mse

    def validation_step(self, batch, batch_idx):
        mse, rmse, mae = self._compute_loss(batch, batch_idx)
        self.log_dict({"val_mse": mse, "val_rmse": rmse, "val_mae": mae}, prog_bar=True)
        return mse

    def test_step(self, batch, batch_idx):
        mse, rmse, mae = self._compute_loss(batch, batch_idx)
        self.log_dict({"test_mse": mse, "test_rmse": rmse, "test_mae": mae}, prog_bar=False)
        return mse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=self.lr_patience, factor=0.666),
                "monitor": "val_mse",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def forward(self, x):
        return self.model(x)
