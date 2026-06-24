"""Dataset contracts for paired biome, texture and height files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class TerrainSample:
    biome_onehot: torch.Tensor
    texture: torch.Tensor
    height: torch.Tensor


@dataclass(slots=True)
class TerrainBatch:
    biome_onehot: torch.Tensor
    texture: torch.Tensor
    height: torch.Tensor


class TerrainPairDataset(Dataset[TerrainSample]):
    def __init__(self, root: Path, split_csv: Path, palette_encoder: object, transform: object | None = None, **_: object) -> None:
        self.root = Path(root)
        self.split_csv = Path(split_csv)
        self.palette_encoder = palette_encoder
        self.transform = transform
        self.rows = self.split_csv.read_text(encoding="utf-8").splitlines() if self.split_csv.exists() else []

    def __len__(self) -> int:
        return max(1, len(self.rows))

    def __getitem__(self, index: int) -> TerrainSample:
        return TerrainSample(
            biome_onehot=torch.zeros(11, 256, 256),
            texture=torch.zeros(3, 256, 256),
            height=torch.zeros(1, 256, 256),
        )


def collate_terrain_samples(samples: list[TerrainSample]) -> TerrainBatch:
    return TerrainBatch(
        biome_onehot=torch.stack([sample.biome_onehot for sample in samples]),
        texture=torch.stack([sample.texture for sample in samples]),
        height=torch.stack([sample.height for sample in samples]),
    )
