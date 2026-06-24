"""ONNX validation CLI shell."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from terrain_gan.export.validate_onnx import validate_onnx_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate PyTorch checkpoint against exported ONNX.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/export_onnx.yaml"),
    )
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", choices=("height", "terrain", "terrain4", "rgb_height"), default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load_yaml(args.config)
    export_config = config.get("export", {})
    onnx_path = args.onnx or Path(export_config.get("output", "artifacts/onnx/terrain_gan.onnx"))
    checkpoint_path = args.checkpoint or Path(export_config.get("checkpoint", "artifacts/checkpoints/last.pt"))

    result = validate_onnx_model(
        onnx_path=onnx_path,
        checkpoint_path=checkpoint_path,
        config_path=args.config,
        mode=args.mode,
    )
    print(f"mode={result.mode}")
    print(f"inputs={','.join(result.input_names)}")
    print(f"output={result.output_name}")
    print(f"output_shape={result.output_shape}")
    print(f"output_dtype={result.output_dtype}")
    if result.max_abs_err is not None and result.mean_abs_err is not None:
        print(f"max_abs_err={result.max_abs_err:.8f}")
        print(f"mean_abs_err={result.mean_abs_err:.8f}")
    else:
        print("contract_valid=true")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


if __name__ == "__main__":
    main()
