"""v12.4 ONNX validation contract checks."""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import yaml
from onnx import TensorProto, helper

from terrain_gan.export.validate_onnx import (
    build_onnx_validation_contract,
    resolve_validation_mode,
    validate_onnx_model,
)


EXPORT_CONFIG = Path("configs/export_onnx.yaml")


def test_v12_4_validation_contract_expects_latent_terrain4() -> None:
    config = _load_yaml(EXPORT_CONFIG)
    contract = build_onnx_validation_contract(config, mode="terrain4")

    assert contract.mode == "terrain4"
    assert [item.name for item in contract.inputs] == ["biome_onehot", "noise"]
    assert [item.shape for item in contract.inputs] == [(1, 11, 256, 256), (1, 16)]
    assert "noise_map" not in [item.name for item in contract.inputs]
    assert contract.output.name == "terrain"
    assert contract.output.shape == (1, 4, 256, 256)


def test_v12_4_validation_auto_detects_terrain4_from_config() -> None:
    config = _load_yaml(EXPORT_CONFIG)

    assert resolve_validation_mode(config) == "terrain4"


def test_v12_4_validation_rejects_noise_map_input_config() -> None:
    config = _load_yaml(EXPORT_CONFIG)
    config["export"]["include_noise_map_input"] = True

    with pytest.raises(ValueError, match="forbids an external noise_map"):
        build_onnx_validation_contract(config, mode="terrain4")


def test_height_contract_rejects_external_noise_map() -> None:
    config = {
        "export": {
            "image_size": 32,
            "include_noise_inputs": True,
            "include_noise_map_input": True,
        },
        "model": {
            "input_channels": 7,
            "noise_dim": 8,
            "output_mode": "height",
        },
    }

    with pytest.raises(ValueError, match="forbids an external noise_map"):
        build_onnx_validation_contract(config, mode="height")


def test_v12_4_terrain4_validation_accepts_matching_onnx(tmp_path: Path) -> None:
    onnx_path = tmp_path / "terrain4.onnx"
    _write_test_onnx(onnx_path, output_channels=4)

    result = validate_onnx_model(
        onnx_path=onnx_path,
        checkpoint_path=tmp_path / "missing_checkpoint.pt",
        config_path=EXPORT_CONFIG,
        mode="terrain4",
    )

    assert result.mode == "terrain4"
    assert result.input_names == ("biome_onehot", "noise")
    assert result.output_name == "terrain"
    assert result.output_shape == (1, 4, 256, 256)
    assert result.output_dtype == "float32"
    assert result.max_abs_err is None
    assert result.mean_abs_err is None


def test_v12_4_terrain4_validation_rejects_height_only_output(tmp_path: Path) -> None:
    onnx_path = tmp_path / "height_named_terrain.onnx"
    _write_test_onnx(onnx_path, output_channels=1)

    with pytest.raises(ValueError, match="Expected output terrain shape"):
        validate_onnx_model(
            onnx_path=onnx_path,
            checkpoint_path=tmp_path / "missing_checkpoint.pt",
            config_path=EXPORT_CONFIG,
            mode="terrain4",
        )


def _write_test_onnx(path: Path, *, output_channels: int) -> None:
    biome = helper.make_tensor_value_info("biome_onehot", TensorProto.FLOAT, [1, 11, 256, 256])
    noise = helper.make_tensor_value_info("noise", TensorProto.FLOAT, [1, 16])
    terrain = helper.make_tensor_value_info("terrain", TensorProto.FLOAT, [1, output_channels, 256, 256])
    starts = helper.make_tensor("starts", TensorProto.INT64, [4], [0, 0, 0, 0])
    ends = helper.make_tensor("ends", TensorProto.INT64, [4], [1, output_channels, 256, 256])
    axes = helper.make_tensor("axes", TensorProto.INT64, [4], [0, 1, 2, 3])
    steps = helper.make_tensor("steps", TensorProto.INT64, [4], [1, 1, 1, 1])
    reduce_axes = helper.make_tensor("reduce_axes", TensorProto.INT64, [2], [0, 1])
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    nodes = [
        helper.make_node(
            "Slice",
            ["biome_onehot", "starts", "ends", "axes", "steps"],
            ["sliced"],
        ),
        helper.make_node("ReduceSum", ["noise", "reduce_axes"], ["noise_sum"], keepdims=0),
        helper.make_node("Mul", ["noise_sum", "zero"], ["noise_zero"]),
        helper.make_node("Add", ["sliced", "noise_zero"], ["terrain"]),
    ]
    graph = helper.make_graph(
        nodes,
        "terrain4_contract_test",
        [biome, noise],
        [terrain],
        [starts, ends, axes, steps, reduce_axes, zero],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 15)])
    model.ir_version = 8
    onnx.save(model, path)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    assert isinstance(value, dict)
    return value
