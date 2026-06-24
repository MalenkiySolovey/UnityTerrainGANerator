"""Reproducibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass(slots=True)
class SeedConfig:
    seed: int = 42


def seed_everything(config: SeedConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
