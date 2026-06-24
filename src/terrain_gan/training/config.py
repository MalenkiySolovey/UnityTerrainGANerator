"""Training config loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class TrainingConfig:
    raw: dict

    def section(self, name: str) -> dict:
        return dict(self.raw.get(name, {}))


def load_training_config(path: Path) -> TrainingConfig:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return TrainingConfig(value)
