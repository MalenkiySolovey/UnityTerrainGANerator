"""Generator configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GeneratorConfig:
    """Compact configuration used by the model factory."""

    input_channels: int = 11
    output_channels: int = 4
    base_channels: int = 32
    noise_dim: int = 16
    image_size: int = 256
