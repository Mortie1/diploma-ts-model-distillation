import torch
import torch.nn as nn


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000, batch_first: bool = False):
        super().__init__()
        if d % 2 != 0:
            raise ValueError("RoPE dimension `d` must be even.")
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None
        self.batch_first = batch_first

    def _build_cache(self, x: torch.Tensor):
        seq_len_idx = 1 if self.batch_first else 0

        if self.cos_cached is not None:
            cache_len = (
                self.cos_cached.shape[1]
                if self.batch_first
                else self.cos_cached.shape[0]
            )
            if (
                x.shape[seq_len_idx] <= cache_len
                and self.cos_cached.device == x.device
                and self.cos_cached.dtype == torch.float32
            ):
                return

        seq_len = x.shape[seq_len_idx]

        theta = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.d, 2, device=x.device, dtype=torch.float32)
                / self.d
            )
        )

        seq_idx = torch.arange(seq_len, device=x.device, dtype=torch.float32)

        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)

        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        if self.batch_first:
            self.cos_cached = idx_theta2.cos()[None, :, None, :].to(dtype=torch.float32)
            self.sin_cached = idx_theta2.sin()[None, :, None, :].to(dtype=torch.float32)
        else:
            self.cos_cached = idx_theta2.cos()[:, None, None, :].to(dtype=torch.float32)
            self.sin_cached = idx_theta2.sin()[:, None, None, :].to(dtype=torch.float32)

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2

        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        self._build_cache(x)

        x_fp32 = x.to(dtype=torch.float32)
        neg_half_x = self._neg_half(x_fp32)

        if self.batch_first:
            seq_len = x.shape[1]
            cos = self.cos_cached[:, :seq_len]
            sin = self.sin_cached[:, :seq_len]
        else:
            seq_len = x.shape[0]
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]

        x_rope = (x_fp32 * cos) + (neg_half_x * sin)

        return x_rope.to(dtype=x.dtype)
