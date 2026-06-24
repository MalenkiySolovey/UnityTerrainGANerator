"""ONNX export CLI shell."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from terrain_gan.export.export_onnx import export_generator_to_onnx
from terrain_gan.utils.paths import resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export generator checkpoint to ONNX.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/export_onnx.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=("height", "terrain", "terrain4", "rgb_height"), default="terrain4")
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--dynamic-hw", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = _load_yaml(args.config)
    export_config = config.get("export", {})
    checkpoint = args.checkpoint or Path(export_config.get("checkpoint", "artifacts/checkpoints/last.pt"))
    output = args.output or Path(export_config.get("output", "artifacts/onnx/terrain_gan.onnx"))
    dynamic_batch = bool(args.dynamic_batch or export_config.get("dynamic_batch", False))
    dynamic_hw = bool(args.dynamic_hw or export_config.get("dynamic_hw", False))

    export_generator_to_onnx(
        checkpoint_path=checkpoint,
        config_path=args.config,
        output_path=output,
        mode=args.mode,
        dynamic_batch=dynamic_batch,
        dynamic_hw=dynamic_hw,
    )
    print(f"exported={resolve_path(output)}")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


if __name__ == "__main__":
    main()
