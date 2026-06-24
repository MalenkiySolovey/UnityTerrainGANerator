"""Multi-scale terrain discriminator stubs used by the training CLI."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DiscriminatorScaleOutput:
    logits: torch.Tensor
    features: list[torch.Tensor]


class MultiScaleTerrainDiscriminator(nn.Module):
    def __init__(self, input_channels: int = 18, base_channels: int = 64, scales: list[float] | None = None, **_: object) -> None:
        super().__init__()
        self.scales = scales or [1.0]
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, condition: torch.Tensor, terrain: torch.Tensor) -> list[DiscriminatorScaleOutput]:
        joined = _build_discriminator_input(condition, terrain)
        outputs: list[DiscriminatorScaleOutput] = []
        for scale in self.scales:
            item = joined if scale == 1.0 else F.interpolate(joined, scale_factor=scale, mode="bilinear", align_corners=False)
            logits = self.net(item)
            outputs.append(DiscriminatorScaleOutput(logits=logits, features=[item, logits]))
        return outputs


class MultiScaleHeightDetailDiscriminator(MultiScaleTerrainDiscriminator):
    pass


def _build_discriminator_input(condition: torch.Tensor, terrain: torch.Tensor, slope_mode: str = "standard") -> torch.Tensor:
    if terrain.shape[-2:] != condition.shape[-2:]:
        terrain = F.interpolate(terrain, size=condition.shape[-2:], mode="bilinear", align_corners=False)
    height = terrain[:, 3:4]
    dx = F.pad(height[..., :, 1:] - height[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(height[..., 1:, :] - height[..., :-1, :], (0, 0, 0, 1))
    return torch.cat([condition, terrain, dx, dy], dim=1)
