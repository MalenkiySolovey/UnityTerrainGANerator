"""ONNX export wrappers."""

from __future__ import annotations

import torch
from torch import nn

from terrain_gan.models.shadow import render_generator_terrain


class HeightExportWrapper(nn.Module):
    def __init__(self, generator: nn.Module) -> None:
        super().__init__()
        self.generator = generator

    def forward(self, biome_onehot: torch.Tensor) -> torch.Tensor:
        return self.generator(biome_onehot)[:, 3:4]


class TerrainRgbHeightExportWrapper(nn.Module):
    def __init__(self, generator: nn.Module, shadow_renderer: nn.Module | None = None) -> None:
        super().__init__()
        self.generator = generator
        self.shadow_renderer = shadow_renderer

    def forward(self, biome_onehot: torch.Tensor) -> torch.Tensor:
        terrain = self.generator(biome_onehot)
        return render_generator_terrain(terrain, self.shadow_renderer)


class TerrainRgbHeightWithLatentExportWrapper(nn.Module):
    def __init__(self, generator: nn.Module, shadow_renderer: nn.Module | None = None) -> None:
        super().__init__()
        self.generator = generator
        self.shadow_renderer = shadow_renderer

    def forward(self, biome_onehot: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        terrain = self.generator(biome_onehot, noise=noise, noise_map=None)
        return render_generator_terrain(terrain, self.shadow_renderer)
