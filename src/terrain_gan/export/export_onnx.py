"""Export a generator checkpoint to ONNX."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from terrain_gan.models.export_wrappers import (
    HeightExportWrapper,
    TerrainRgbHeightExportWrapper,
    TerrainRgbHeightWithLatentExportWrapper,
)
from terrain_gan.models.factory import build_generator
from terrain_gan.models.shadow import OriginalLabShadowRenderer
from terrain_gan.utils.paths import resolve_path
from terrain_gan.utils.state_dict import load_compatible_state_dict


def export_generator_to_onnx(
    checkpoint_path: Path,
    config_path: Path,
    output_path: Path,
    mode: str = "terrain4",
    dynamic_batch: bool = False,
    dynamic_hw: bool = False,
) -> Path:
    config = _load_yaml(config_path)
    model_config = config.get("model", {})
    export_config = config.get("export", {})
    generator = build_generator(model_config).eval()
    if checkpoint_path.exists():
        checkpoint = torch.load(resolve_path(checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("generator", checkpoint.get("model", checkpoint))
        if isinstance(state_dict, dict):
            load_compatible_state_dict(generator, state_dict, strict=False)
    shadow_renderer = OriginalLabShadowRenderer().eval()
    include_noise_inputs = bool(export_config.get("include_noise_inputs", False))
    include_noise_map_input = bool(export_config.get("include_noise_map_input", False))
    image_size = int(export_config.get("image_size", config.get("data", {}).get("image_size", 256)))
    input_channels = int(model_config.get("input_channels", 11))
    noise_dim = int(model_config.get("noise_dim", 16))

    if mode in {"terrain", "terrain4", "rgb_height"}:
        if include_noise_map_input:
            raise ValueError("active export forbids an external spatial noise map")
        if include_noise_inputs:
            wrapper = TerrainRgbHeightWithLatentExportWrapper(generator, shadow_renderer).eval()
            export_inputs = (
                torch.zeros(1, input_channels, image_size, image_size),
                torch.zeros(1, noise_dim),
            )
            input_names = ["biome_onehot", "noise"]
        else:
            wrapper = TerrainRgbHeightExportWrapper(generator, shadow_renderer).eval()
            export_inputs = (torch.zeros(1, input_channels, image_size, image_size),)
            input_names = ["biome_onehot"]
        output_name = "terrain"
    elif mode == "height":
        if include_noise_map_input:
            raise ValueError("active export forbids an external spatial noise map")
        wrapper = HeightExportWrapper(generator).eval()
        export_inputs = (torch.zeros(1, input_channels, image_size, image_size),)
        input_names = ["biome_onehot"]
        output_name = "height"
    else:
        raise ValueError(f"Unsupported export mode: {mode}")

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch or dynamic_hw:
        axes = {0: "batch"} if dynamic_batch else {}
        if dynamic_hw:
            axes.update({2: "height", 3: "width"})
        dynamic_axes = {name: axes for name in input_names}
        dynamic_axes[output_name] = axes

    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        export_inputs,
        output_path,
        input_names=input_names,
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        opset_version=int(export_config.get("opset", 17)),
    )
    return output_path


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value
