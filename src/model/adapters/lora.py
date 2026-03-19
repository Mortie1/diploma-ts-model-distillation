from __future__ import annotations

import math
from fnmatch import fnmatch

import torch
from torch import nn


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()
        return base_out + self.scaling * lora_out


def _resolve_parent_module(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    if "." not in path:
        return root, path
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_patterns: tuple[str, ...] = ("*",),
) -> int:
    """
    Replace matched nn.Linear modules with LoRALinear wrappers.
    Returns number of replaced modules.
    """
    replaced = 0
    names = [name for name, m in module.named_modules() if isinstance(m, nn.Linear)]
    for name in names:
        if not any(fnmatch(name, pat) for pat in target_patterns):
            continue
        parent, attr = _resolve_parent_module(module, name)
        linear = getattr(parent, attr)
        if isinstance(linear, LoRALinear):
            continue
        setattr(parent, attr, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))
        replaced += 1
    return replaced

