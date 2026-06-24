"""Split helpers."""

from __future__ import annotations

from pathlib import Path


def read_split(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []
