"""Small runtime helpers shared by training and evaluation."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed random generators while retaining fast, non-deterministic CUDA kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_first_param_device(model) -> torch.device:
    return next(model.parameters()).device


def supports_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def pick_vision_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    keys = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
    return {key: batch[key] for key in keys if key in batch}


def append_text_log(path: str, line: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")
