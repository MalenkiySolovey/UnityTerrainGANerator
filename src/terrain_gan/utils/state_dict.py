"""State-dict helpers."""

from __future__ import annotations

from torch import nn


def load_compatible_state_dict(module: nn.Module, state_dict: dict, strict: bool = False) -> None:
    module.load_state_dict(state_dict, strict=strict)
