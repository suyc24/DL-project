from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def selected_layers_for_model(config: dict[str, Any]) -> list[int]:
    layers = config.get("model", {}).get("layers")
    if not layers:
        raise ValueError("config.model.layers must list transformer block indices to extract")
    return [int(x) for x in layers]
