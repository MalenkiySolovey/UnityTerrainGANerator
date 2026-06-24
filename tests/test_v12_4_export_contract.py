"""v12.4 ONNX export contract checks."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TypeAlias

import yaml

from terrain_gan.models.export_wrappers import (
    TerrainRgbHeightExportWrapper,
    TerrainRgbHeightWithLatentExportWrapper,
)


ExportWrapper: TypeAlias = type[TerrainRgbHeightExportWrapper]

EXPORT_CONFIG = Path("configs/export_onnx.yaml")


def test_v12_4_export_config_pins_active_contract() -> None:
    config = _load_yaml(EXPORT_CONFIG)
    export = config["export"]
    model = config["model"]

    assert export["include_noise_inputs"] is True
    assert export["include_noise_map_input"] is False
    assert model["input_channels"] == 11
    assert model["output_mode"] == "terrain4"
    assert model["name"] == "terrain_gan_feature_supervised"
    assert model["noise_dim"] == 16


def test_v12_4_terrain4_export_selects_latent_wrapper_without_noise_map() -> None:
    config = _load_yaml(EXPORT_CONFIG)
    export = config["export"]
    model = config["model"]
    wrapper = _select_terrain_wrapper(
        mode=model["output_mode"],
        include_noise_inputs=export["include_noise_inputs"],
        include_noise_map_input=export["include_noise_map_input"],
    )

    assert wrapper is TerrainRgbHeightWithLatentExportWrapper

    forward_signature = inspect.signature(wrapper.forward)
    assert list(forward_signature.parameters) == ["self", "biome_onehot", "noise"]
    assert "noise_map" not in forward_signature.parameters
    assert "noise_map=None" in inspect.getsource(wrapper.forward)


def test_exporter_terrain_branch_has_no_external_noise_map_wrapper() -> None:
    source = Path("src/terrain_gan/export/export_onnx.py").read_text(encoding="utf-8")
    terrain_branch = source.split('if mode in {"terrain", "terrain4", "rgb_height"}:', maxsplit=1)[1]
    terrain_branch = terrain_branch.split('elif mode == "height":', maxsplit=1)[0]

    assert "TerrainRgbHeightWithNoiseExportWrapper" not in source
    assert "if include_noise_inputs and include_noise_map_input:" not in terrain_branch
    assert "if include_noise_inputs:" in terrain_branch
    assert "TerrainRgbHeightWithLatentExportWrapper(generator, shadow_renderer).eval()" in terrain_branch


def test_v12_4_onnx_input_names_do_not_include_noise_map() -> None:
    config = _load_yaml(EXPORT_CONFIG)
    export = config["export"]
    input_names = _export_input_names(
        include_noise_inputs=export["include_noise_inputs"],
        include_noise_map_input=export["include_noise_map_input"],
    )

    assert input_names == ["biome_onehot", "noise"]
    assert "noise_map" not in input_names


def _select_terrain_wrapper(
    *,
    mode: str,
    include_noise_inputs: bool,
    include_noise_map_input: bool,
) -> ExportWrapper:
    assert mode in {"terrain", "terrain4", "rgb_height"}
    if include_noise_inputs and include_noise_map_input:
        raise ValueError("external spatial noise_map export was removed")
    if include_noise_inputs:
        return TerrainRgbHeightWithLatentExportWrapper
    return TerrainRgbHeightExportWrapper


def _export_input_names(*, include_noise_inputs: bool, include_noise_map_input: bool) -> list[str]:
    if include_noise_inputs and include_noise_map_input:
        raise ValueError("external spatial noise_map export was removed")
    if include_noise_inputs:
        return ["biome_onehot", "noise"]
    return ["biome_onehot"]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    assert isinstance(value, dict)
    return value
