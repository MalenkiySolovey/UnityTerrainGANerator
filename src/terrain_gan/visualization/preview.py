"""Preview writers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_terrain_gan_preview(*args: object, output_path: Path | None = None, **kwargs: object) -> None:
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()


def save_terrain_gan_preview_grid(*args: object, output_path: Path | None = None, **kwargs: object) -> None:
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
