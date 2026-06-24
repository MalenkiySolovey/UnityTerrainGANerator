"""Terrain-specific structural losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TerrainShapeLossParts:
    total: torch.Tensor


@dataclass
class StructureFeatureSupervisionLossParts:
    total: torch.Tensor


class TerrainShapeLoss(nn.Module):
    def forward(self, generated: torch.Tensor, target: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return torch.mean(torch.abs(generated - target))


class StructureFeatureSupervisionLoss(nn.Module):
    def forward(self, generated: torch.Tensor, target: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return torch.mean(torch.abs(generated - target))
