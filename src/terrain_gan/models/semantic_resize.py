"""Semantic resize helper."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_semantic(condition: torch.Tensor, size: tuple[int, int], mode: str = "nearest") -> torch.Tensor:
    return F.interpolate(condition, size=size, mode=mode)
