"""Texture rendering helpers for terrain output."""

from __future__ import annotations

import torch
from torch import nn


class OriginalLabShadowRenderer(nn.Module):
    def forward(self, terrain: torch.Tensor) -> torch.Tensor:
        return terrain


class OriginalLearnedShadowRenderer(OriginalLabShadowRenderer):
    pass


class OriginalLuminanceShadowRenderer(OriginalLabShadowRenderer):
    pass


class LearnedShadowRenderer(OriginalLabShadowRenderer):
    pass


def render_generator_terrain(terrain: torch.Tensor, shadow_renderer: nn.Module | None = None) -> torch.Tensor:
    return terrain if shadow_renderer is None else shadow_renderer(terrain)


def render_shadowed_terrain(terrain: torch.Tensor, shadow_renderer: nn.Module | None = None) -> torch.Tensor:
    return render_generator_terrain(terrain, shadow_renderer)


def lab_like_to_rgb(texture: torch.Tensor) -> torch.Tensor:
    return texture.clamp(0.0, 1.0)


def lab_like_gamut_loss(texture: torch.Tensor) -> torch.Tensor:
    return torch.relu(texture - 1.0).mean() + torch.relu(-texture).mean()


def height_slopes(height: torch.Tensor) -> torch.Tensor:
    dx = height[..., :, 1:] - height[..., :, :-1]
    dy = height[..., 1:, :] - height[..., :-1, :]
    dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
    dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
    return torch.cat([dx, dy], dim=1)
