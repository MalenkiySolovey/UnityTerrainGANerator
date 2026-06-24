"""RGB losses."""

from __future__ import annotations

import torch


def rgb_l1(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(generated - target))
