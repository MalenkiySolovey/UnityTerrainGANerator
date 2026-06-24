"""Validate ONNX runtime contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import TensorProto
import yaml


@dataclass(slots=True)
class TensorContract:
    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"


@dataclass(slots=True)
class OnnxValidationContract:
    mode: str
    inputs: tuple[TensorContract, ...]
    output: TensorContract


@dataclass(slots=True)
class OnnxValidationResult:
    mode: str
    input_names: tuple[str, ...]
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: str
    max_abs_err: float | None = None
    mean_abs_err: float | None = None


def resolve_validation_mode(config: dict) -> str:
    output_mode = str(config.get("model", {}).get("output_mode", "height"))
    if output_mode in {"terrain", "terrain4", "rgb_height"}:
        return "terrain4"
    return "height"


def build_onnx_validation_contract(config: dict, mode: str | None = None) -> OnnxValidationContract:
    mode = mode or resolve_validation_mode(config)
    export = config.get("export", {})
    model = config.get("model", {})
    if bool(export.get("include_noise_map_input", False)):
        raise ValueError("active runtime contract forbids an external noise_map input")
    image_size = int(export.get("image_size", config.get("data", {}).get("image_size", 256)))
    input_channels = int(model.get("input_channels", 11))
    noise_dim = int(model.get("noise_dim", 16))
    include_noise = bool(export.get("include_noise_inputs", mode == "terrain4"))
    inputs = [TensorContract("biome_onehot", (1, input_channels, image_size, image_size))]
    if include_noise:
        inputs.append(TensorContract("noise", (1, noise_dim)))
    if mode in {"terrain", "terrain4", "rgb_height"}:
        output = TensorContract("terrain", (1, 4, image_size, image_size))
        normalized_mode = "terrain4"
    elif mode == "height":
        output = TensorContract("height", (1, 1, image_size, image_size))
        normalized_mode = "height"
    else:
        raise ValueError(f"Unsupported ONNX validation mode: {mode}")
    return OnnxValidationContract(normalized_mode, tuple(inputs), output)


def validate_onnx_model(
    onnx_path: Path,
    checkpoint_path: Path | None = None,
    config_path: Path | None = None,
    mode: str | None = None,
) -> OnnxValidationResult:
    config = _load_yaml(config_path) if config_path else {}
    contract = build_onnx_validation_contract(config, mode=mode)
    model = onnx.load(str(onnx_path))
    graph_inputs = [item for item in model.graph.input if not _is_initializer(model, item.name)]
    input_names = tuple(item.name for item in graph_inputs)
    expected_input_names = tuple(item.name for item in contract.inputs)
    if input_names != expected_input_names:
        raise ValueError(f"Expected ONNX inputs {expected_input_names}, got {input_names}")
    for graph_input, expected in zip(graph_inputs, contract.inputs):
        shape = _value_shape(graph_input)
        if not _shape_matches(shape, expected.shape):
            raise ValueError(f"Expected input {expected.name} shape {expected.shape}, got {shape}")
    if len(model.graph.output) != 1:
        raise ValueError(f"Expected one ONNX output, got {len(model.graph.output)}")
    output = model.graph.output[0]
    output_shape = _value_shape(output)
    output_dtype = _value_dtype(output)
    if output.name != contract.output.name:
        raise ValueError(f"Expected output {contract.output.name}, got {output.name}")
    if not _shape_matches(output_shape, contract.output.shape):
        raise ValueError(f"Expected output {contract.output.name} shape {contract.output.shape}, got {output_shape}")
    if output_dtype != contract.output.dtype:
        raise ValueError(f"Expected output dtype {contract.output.dtype}, got {output_dtype}")
    return OnnxValidationResult(
        mode=contract.mode,
        input_names=input_names,
        output_name=output.name,
        output_shape=_reported_shape(output_shape, contract.output.shape),
        output_dtype=output_dtype,
    )


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def _is_initializer(model: onnx.ModelProto, name: str) -> bool:
    return any(initializer.name == name for initializer in model.graph.initializer)


def _value_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    dims = value_info.type.tensor_type.shape.dim
    return tuple(int(dim.dim_value) for dim in dims)


def _shape_matches(actual: tuple[int, ...], expected: tuple[int, ...]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(actual_dim in (0, expected_dim) for actual_dim, expected_dim in zip(actual, expected))


def _reported_shape(actual: tuple[int, ...], expected: tuple[int, ...]) -> tuple[int, ...]:
    if _shape_matches(actual, expected):
        return tuple(expected_dim if actual_dim == 0 else actual_dim for actual_dim, expected_dim in zip(actual, expected))
    return actual


def _value_dtype(value_info: onnx.ValueInfoProto) -> str:
    elem_type = value_info.type.tensor_type.elem_type
    if elem_type == TensorProto.FLOAT:
        return "float32"
    return TensorProto.DataType.Name(elem_type).lower()
