"""Dataset reader for E2E-GMNER/VGThink JSONL records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


def _without_image_marker(text: str) -> str:
    return (text or "").replace("<image> Text:", "<image>\nText:").replace("<image>", "").strip()


class TwitterGroundedMNERJsonl(Dataset):
    """Load the compact chat-style records used by all three VGThink stages."""

    def __init__(self, jsonl_path: str | Path, image_root: str | Path | None = None):
        self.path = Path(jsonl_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.items: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.path}:{line_number}") from exc
                self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        messages = item.get("messages") or []
        user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
        images = item.get("images") or []
        if user is None or assistant is None or len(images) != 1:
            raise ValueError(
                f"Record {item.get('id', index)!r} must contain one user message, "
                "one assistant message, and exactly one image"
            )

        image_path = Path(images[0])
        if not image_path.is_absolute() and self.image_root is not None:
            image_path = self.image_root / image_path
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        return {
            "record_id": item.get("id", index),
            "user_text": _without_image_marker(str(user.get("content") or "")),
            "assistant_text": str(assistant.get("content") or ""),
            "image": image,
            "image_path": str(image_path),
            "region_targets": item.get("region_targets") or [],
        }


__all__ = ["TwitterGroundedMNERJsonl"]
