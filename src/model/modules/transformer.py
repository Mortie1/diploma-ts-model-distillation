from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.model.modules.ff import FeedForward, SwiGLU
from src.model.modules.rope import RotaryPositionalEmbeddings


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        use_swiglu: bool = False,
        use_rope: bool = True,
        rope_base: int = 10_000,
    ):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}."
            )
        if ff_mult <= 0:
            raise ValueError("ff_mult must be > 0.")

        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        self.attn_dropout = float(dropout)

        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_dropout = nn.Dropout(float(dropout))

        if not use_swiglu:
            self.ff = FeedForward(
                hidden_dim=hidden_dim,
                hidden_dim_multiplier=ff_mult,
                activation="relu",
                dropout_p=dropout,
            )
        else:
            self.ff = SwiGLU(
                hidden_dim=hidden_dim,
                hidden_dim_multiplier=ff_mult,
                dropout_p=dropout,
            )

        self.rope = (
            RotaryPositionalEmbeddings(
                d=self.head_dim, base=int(rope_base), batch_first=True
            )
            if use_rope
            else None
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, T, H, Dh]
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, T, Dh] -> [B, T, D]
        b, h, t, dh = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(b, t, h * dh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x_norm = self.norm1(x)
        q = self._split_heads(self.q_proj(x_norm))
        k = self._split_heads(self.k_proj(x_norm))
        v = self._split_heads(self.v_proj(x_norm))

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # SDPA expects [..., T, Dh]
        q = q.permute(0, 2, 1, 3)  # [B, H, T, Dh]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = self._merge_heads(attn)
        x = x + self.out_dropout(self.out_proj(attn))
        x = x + self.ff(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        ff_mult: int = 4,
        use_swiglu: bool = False,
        use_rope: bool = True,
        rope_base: int = 10_000,
    ):
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be > 0.")
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    use_swiglu=use_swiglu,
                    use_rope=use_rope,
                    rope_base=rope_base,
                )
                for _ in range(int(n_layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
