"""Biome-derived control maps."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class ControlMapConfig:
    enabled: bool = True
    channels: tuple[str, ...] = ("height_prior", "detail_budget", "boundary_distance", "semantic_complexity")
    dropout_probability: float = 0.0
    smoothing_kernel_size: int = 5
    height_prior_by_class: tuple[float, ...] = (0.05, 0.22, 0.36, 0.48, 0.66, 0.78, 0.90)
    detail_budget_by_class: tuple[float, ...] = (0.10, 0.28, 0.38, 0.50, 0.58, 0.74, 0.88)
    boundary_center_by_class: tuple[float, ...] = (0.0, 0.30, 0.40, 0.50, 0.68, 0.82, 1.0)
    boundary_distance_steps: int = 16
    semantic_complexity_kernel_size: int = 9


def parse_control_map_config(value: dict | None) -> ControlMapConfig:
    if not value:
        return ControlMapConfig(enabled=False, channels=())
    channels = tuple(value.get("channels", ControlMapConfig.channels))
    return ControlMapConfig(
        enabled=bool(value.get("enabled", True)),
        channels=channels,
        dropout_probability=float(value.get("dropout_probability", 0.0)),
        smoothing_kernel_size=int(value.get("smoothing_kernel_size", 5)),
        height_prior_by_class=tuple(value.get("height_prior_by_class", ControlMapConfig.height_prior_by_class)),
        detail_budget_by_class=tuple(value.get("detail_budget_by_class", ControlMapConfig.detail_budget_by_class)),
        boundary_center_by_class=tuple(value.get("boundary_center_by_class", ControlMapConfig.boundary_center_by_class)),
        boundary_distance_steps=int(value.get("boundary_distance_steps", 16)),
        semantic_complexity_kernel_size=int(value.get("semantic_complexity_kernel_size", 9)),
    )


def expected_control_channels(config: ControlMapConfig) -> int:
    return len(config.channels) if config.enabled else 0


def with_control_maps(biome_onehot: torch.Tensor, config: ControlMapConfig) -> torch.Tensor:
    if not config.enabled:
        return biome_onehot
    added_batch = biome_onehot.ndim == 3
    batch = biome_onehot.unsqueeze(0) if added_batch else biome_onehot
    maps = build_control_maps(batch, config)
    result = torch.cat([batch, maps], dim=1)
    return result.squeeze(0) if added_batch else result


def build_control_maps(biome_onehot: torch.Tensor, config: ControlMapConfig) -> torch.Tensor:
    outputs = []
    for channel in config.channels:
        if channel == "height_prior":
            outputs.append(_weighted_class_map(biome_onehot, config.height_prior_by_class))
        elif channel == "detail_budget":
            outputs.append(_weighted_class_map(biome_onehot, config.detail_budget_by_class))
        elif channel == "boundary_distance":
            outputs.append(_boundary_map(biome_onehot))
        elif channel == "semantic_complexity":
            outputs.append(_semantic_complexity(biome_onehot))
        else:
            outputs.append(torch.zeros_like(biome_onehot[:, :1]))
    return torch.cat(outputs, dim=1)


def _weighted_class_map(biome_onehot: torch.Tensor, weights: tuple[float, ...]) -> torch.Tensor:
    weight_tensor = biome_onehot.new_tensor(weights[: biome_onehot.shape[1]]).view(1, -1, 1, 1)
    return (biome_onehot[:, : weight_tensor.shape[1]] * weight_tensor).sum(dim=1, keepdim=True)


def _boundary_map(biome_onehot: torch.Tensor) -> torch.Tensor:
    classes = biome_onehot.argmax(dim=1, keepdim=True).float()
    dx = F.pad((classes[..., :, 1:] != classes[..., :, :-1]).float(), (0, 1, 0, 0))
    dy = F.pad((classes[..., 1:, :] != classes[..., :-1, :]).float(), (0, 0, 0, 1))
    return (dx + dy).clamp(0.0, 1.0)


def _semantic_complexity(biome_onehot: torch.Tensor) -> torch.Tensor:
    kernel = torch.ones(1, 1, 5, 5, device=biome_onehot.device, dtype=biome_onehot.dtype) / 25.0
    class_map = biome_onehot.argmax(dim=1, keepdim=True).float() / max(1, biome_onehot.shape[1] - 1)
    blurred = F.conv2d(class_map, kernel, padding=2)
    return (class_map - blurred).abs().clamp(0.0, 1.0)
