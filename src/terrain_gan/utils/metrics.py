"""Metric helpers."""

from __future__ import annotations

import torch


def compute_terrain_quality_metrics(*_: object, **__: object) -> dict[str, float]:
    return {}


def l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(a - b))
