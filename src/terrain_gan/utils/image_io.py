"""Image IO helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def read_rgb_image(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def read_height_image(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)
