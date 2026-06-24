"""Paired transform placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TransformConfig:
    image_size: int = 256
    random_flip: bool = False
    random_rotate: bool = False
    random_crop: bool = False


class TrainPairedTransform:
    def __init__(self, config: TransformConfig) -> None:
        self.config = config

    def __call__(self, value: object) -> object:
        return value


class EvalPairedTransform(TrainPairedTransform):
    pass
