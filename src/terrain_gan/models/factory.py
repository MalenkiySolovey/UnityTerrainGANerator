"""Model factory."""

from __future__ import annotations

from terrain_gan.models.generator import GeneratorConfig
from terrain_gan.models.spade_terrain import SPADETerrainGeneratorV9


def build_generator(config: dict | GeneratorConfig | None = None) -> SPADETerrainGeneratorV9:
    if isinstance(config, GeneratorConfig):
        values = config.__dict__
    else:
        values = dict(config or {})
    return SPADETerrainGeneratorV9(
        input_channels=int(values.get("input_channels", 11)),
        output_channels=4,
        base_channels=int(values.get("base_channels", 32)),
        noise_dim=int(values.get("noise_dim", 16)),
    )
