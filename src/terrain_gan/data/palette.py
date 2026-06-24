"""Palette encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class PaletteMatcher:
    num_classes: int = 7

    @classmethod
    def from_files(cls, palette_path: Path, biomes_path: Path) -> "PaletteMatcher":
        return cls()


class PaletteEncoder:
    def __init__(self, matcher: PaletteMatcher, num_macro_classes: int = 7) -> None:
        self.matcher = matcher
        self.num_macro_classes = num_macro_classes

    def encode_macro_onehot(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        result = torch.zeros(self.num_macro_classes, height, width, dtype=torch.float32)
        result[0] = 1.0
        return result
