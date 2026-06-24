"""Lightweight SPADE-like terrain generator used for export and tests.

The historical training implementation was larger. This module preserves the
active public contract: an 11-channel condition and a 16-value latent vector
produce a four-channel terrain tensor with RGB and height channels.
"""

from __future__ import annotations

import torch
from torch import nn


class StructureCarverBranch(nn.Module):
    """Small residual branch that adjusts the generated height channel."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([features, height], dim=1))


class SPADETerrainGeneratorV9(nn.Module):
    """SPADE-like generator with the active terrain4 runtime contract."""

    def __init__(
        self,
        input_channels: int = 11,
        output_channels: int = 4,
        base_channels: int = 32,
        noise_dim: int = 16,
        **_: object,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.noise_dim = noise_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels + noise_dim, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.rgb_head = nn.Sequential(nn.Conv2d(base_channels, 3, kernel_size=1), nn.Sigmoid())
        self.height_head = nn.Sequential(nn.Conv2d(base_channels, 1, kernel_size=1), nn.Sigmoid())
        self.structure_detail_refiner = StructureCarverBranch(base_channels)

    def forward(
        self,
        biome_onehot: torch.Tensor,
        noise: torch.Tensor | None = None,
        noise_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if biome_onehot.ndim != 4:
            raise ValueError(f"Expected condition [B,C,H,W], got {tuple(biome_onehot.shape)}")
        batch, _, height, width = biome_onehot.shape
        if noise is None:
            noise = sample_generator_latent(batch, self.noise_dim, biome_onehot.device, biome_onehot.dtype)
        noise_tiled = noise.view(batch, self.noise_dim, 1, 1).expand(batch, self.noise_dim, height, width)
        features = self.encoder(torch.cat([biome_onehot, noise_tiled], dim=1))
        rgb = self.rgb_head(features)
        base_height = self.height_head(features)
        height_delta = 0.08 * self.structure_detail_refiner(features, base_height)
        terrain_height = (base_height + height_delta).clamp(0.0, 1.0)
        return torch.cat([rgb, terrain_height], dim=1)


def sample_generator_latent(
    batch_size: int,
    noise_dim: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.randn(batch_size, noise_dim, device=device, dtype=dtype)


def make_terrain_noise_map(
    batch_size: int,
    size: tuple[int, int],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    height, width = size
    return torch.randn(batch_size, 1, height, width, device=device, dtype=dtype)


def uses_internal_original_noise(config: dict | None = None) -> bool:
    return str((config or {}).get("noise_map_mode", "")).startswith("internal")


def uses_original_fresh_block_noise(config: dict | None = None) -> bool:
    return str((config or {}).get("noise_map_mode", "")) == "original_fresh_block"


def split_terrain_output(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if output.ndim != 4 or output.shape[1] != 4:
        raise ValueError(f"terrain output must have shape [B,4,H,W], got {tuple(output.shape)}")
    return output[:, :3], output[:, 3:4]
