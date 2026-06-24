"""Checkpoint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CheckpointState:
    epoch: int = 0
    metrics: dict | None = None


class CheckpointManager:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, state: dict) -> Path:
        path = self.directory / name
        torch.save(state, path)
        return path


def capture_rng_state() -> dict:
    return {"torch": torch.get_rng_state()}


def restore_rng_state(state: dict | None) -> None:
    if state and "torch" in state:
        torch.set_rng_state(state["torch"])


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for group in optimizer.state.values():
        for key, value in group.items():
            if torch.is_tensor(value):
                group[key] = value.to(device)
