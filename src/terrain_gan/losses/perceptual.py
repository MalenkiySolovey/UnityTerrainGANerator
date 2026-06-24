"""Perceptual loss placeholder."""

from __future__ import annotations

import torch
from torch import nn


class VGGPerceptualLoss(nn.Module):
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(generated - target))
