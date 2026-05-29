from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def section(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    return cfg.get(name, {}) or {}


def required_path(value: str | None, name: str) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{name} must be set.")
    return Path(value)


def optional_path(value: str | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value)


def dataloader_kwargs(num_workers: int, *, train: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4 if train else 2
    return kwargs

