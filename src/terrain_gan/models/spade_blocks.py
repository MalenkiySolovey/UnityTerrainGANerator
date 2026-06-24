"""SPADE-like block placeholder."""

from __future__ import annotations

import torch
from torch import nn


class SPADELikeBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(x)
