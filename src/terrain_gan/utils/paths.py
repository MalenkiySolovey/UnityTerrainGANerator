"""Path helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
