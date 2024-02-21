import torch.nn as nn
import torch.nn.functional as F

# gMLP based on:
# H. Liu, Z. Dai, D. So, and Q. V. Le, ‘Pay Attention to MLPs’, in Advances in Neural Information Processing Systems, Curran Associates, Inc., 2021, pp. 9204–9215
# https://proceedings.neurips.cc/paper_files/paper/2021/file/4cc05b35c2f937c5bd9e7d41d3686fff-Paper.pdf

# Portions of code inspired by https://github.com/jaketae/g-mlp/tree/master/g_mlp and https://github.com/timeseriesAI/tsai/blob/main/nbs/040_models.gMLP.ipynb


class SpatialGatingUnit(nn.Module):
    def __init__(
            self,
            d_ffn: int,
            seq_len: int
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.proj.bias, val=1.0)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-6)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)  # split into 2
        v = self.norm(v)
        v = self.proj(v)
        return u * v


class gMLPBlock(nn.Module):

    def __init__(
            self,
            d_model: int,
            d_ffn: int,
            seq_len: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, 2 * d_ffn)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.proj2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.proj1(x)
        x = F.gelu(x)
        x = self.sgu(x)
        x = self.proj2(x)
        x = x + res
        return x


class gMLP(nn.Module):
    def __init__(
            self,
            ch_in,
            ch_out,
            seq_len,
            n_blocks: int = 2,
            patch_size: int = 1,
            d_model: int = 256,
            d_ffn: int = 256,
    ):
        assert seq_len % patch_size == 0
        super().__init__()
        self.patch = nn.Conv1d(in_channels=ch_in, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        self.gmlp_blocks = nn.Sequential(
            *[gMLPBlock(d_model=d_model, d_ffn=d_ffn, seq_len=(seq_len // patch_size))
              for _ in range(n_blocks)]
        )
        self.proj = nn.Linear(d_model, ch_out)

    def forward(self, x):
        x = self.patch(x)
        n_batch, n_ch, _ = x.shape
        x = x.permute(0, 2, 1)
        x = self.gmlp_blocks(x)
        x = x.mean(dim=1)
        x = self.proj(x)
        return x
