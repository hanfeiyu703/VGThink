"""Geometry and Qwen image-resize coordinate conversion."""

from __future__ import annotations

import math
import re
from typing import Optional

from .parsing import (
    Box,
    EntityTriple,
    _structured_answer_parse_failed,
    extract_answer_text,
    parse_triples,
    replace_answer_text,
    triples_to_canon_text,
)


def iou(box1: Box, box2: Box) -> float:
    x1, y1, x2, y2 = box1
    gx1, gy1, gx2, gy2 = box2
    intersection = max(0.0, min(x2, gx2) - max(x1, gx1)) * max(0.0, min(y2, gy2) - max(y1, gy1))
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = area1 + area2 - intersection
    return intersection / union if union > 0.0 else 0.0


def max_iou(pred_boxes: list[Box], gold_boxes: list[Box]) -> float:
    return max((iou(pred, gold) for pred in pred_boxes for gold in gold_boxes), default=0.0)


def region_correct(
    prediction: EntityTriple, gold: EntityTriple, iou_threshold: float
) -> tuple[bool, float]:
    if not prediction.region_valid:
        return False, 0.0
    if prediction.regions is None and gold.regions is None:
        return True, 1.0
    if prediction.regions is None or gold.regions is None:
        return False, 0.0
    overlap = max_iou(prediction.regions, gold.regions)
    return overlap >= iou_threshold, overlap


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """Match Qwen2.5-VL's aspect-preserving, factor-aligned resize."""

    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(width, height)}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must be smaller than 200")
    resized_h = max(factor, round(height / factor) * factor)
    resized_w = max(factor, round(width / factor) * factor)
    if resized_h * resized_w > max_pixels:
        ratio = math.sqrt((height * width) / max_pixels)
        resized_h = max(factor, math.floor(height / ratio / factor) * factor)
        resized_w = max(factor, math.floor(width / ratio / factor) * factor)
    elif resized_h * resized_w < min_pixels:
        ratio = math.sqrt(min_pixels / (height * width))
        resized_h = math.ceil(height * ratio / factor) * factor
        resized_w = math.ceil(width * ratio / factor) * factor
    return resized_h, resized_w


def _ordered_clipped_box(
    box: Box, scale_w: float, scale_h: float, width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    values = [
        round(x1 * scale_w),
        round(y1 * scale_h),
        round(x2 * scale_w),
        round(y2 * scale_h),
    ]
    values[0] = min(width, max(0, values[0]))
    values[2] = min(width, max(0, values[2]))
    values[1] = min(height, max(0, values[1]))
    values[3] = min(height, max(0, values[3]))
    if values[0] > values[2]:
        values[0], values[2] = values[2], values[0]
    if values[1] > values[3]:
        values[1], values[3] = values[3], values[1]
    return tuple(int(value) for value in values)


def scale_box_xyxy(
    box: Box, scale_w: float, scale_h: float, new_w: int, new_h: int
) -> tuple[int, int, int, int]:
    return _ordered_clipped_box(box, scale_w, scale_h, new_w, new_h)


def inverse_scale_box_xyxy(
    box: Box, scale_w: float, scale_h: float, orig_w: int, orig_h: int
) -> tuple[int, int, int, int]:
    return _ordered_clipped_box(box, 1.0 / scale_w, 1.0 / scale_h, orig_w, orig_h)


_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
OBJ_BOX_RE = re.compile(
    rf"<obj>\s*(.*?)\s*\|\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,"
    rf"\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]\s*</obj>",
    flags=re.IGNORECASE | re.DOTALL,
)
OBJ_POINT_RE = re.compile(
    rf"<obj>\s*(.*?)\s*\|\s*\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]\s*</obj>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _format_coordinate(value: float) -> str:
    return (
        str(int(round(value)))
        if abs(value - round(value)) < 1e-6
        else f"{value:.3f}".rstrip("0").rstrip(".")
    )


def transform_obj_coordinates(text: str, transform_box) -> str:
    """Transform both four-coordinate evidence boxes and two-coordinate anchors."""

    def replace_box(match: re.Match[str]) -> str:
        transformed = transform_box(tuple(float(match.group(index)) for index in range(2, 6)))
        coords = ",".join(_format_coordinate(value) for value in transformed)
        return f"<obj>{match.group(1).strip()}|[{coords}]</obj>"

    def replace_point(match: re.Match[str]) -> str:
        x, y = float(match.group(2)), float(match.group(3))
        transformed = transform_box((x, y, x, y))
        return (
            f"<obj>{match.group(1).strip()}|"
            f"[{_format_coordinate(transformed[0])},{_format_coordinate(transformed[1])}]</obj>"
        )

    return OBJ_POINT_RE.sub(replace_point, OBJ_BOX_RE.sub(replace_box, text or ""))


def _convert_response(
    assistant_text: str,
    *,
    transform_box,
) -> Optional[str]:
    answer = extract_answer_text(assistant_text)
    if not answer or answer.casefold() == "none":
        return transform_obj_coordinates(assistant_text, transform_box)
    triples = parse_triples(answer, strict=False, where="coordinate_conversion")
    if not triples:
        return None if _structured_answer_parse_failed(answer) else assistant_text
    converted: list[EntityTriple] = []
    for triple in triples:
        if not triple.region_valid:
            return None
        regions = (
            None
            if triple.regions is None
            else [tuple(float(value) for value in transform_box(box)) for box in triple.regions]
        )
        converted.append(
            EntityTriple(
                text=triple.text,
                etype=triple.etype,
                regions=regions,
                region_valid=True,
            )
        )
    replaced = replace_answer_text(assistant_text.strip(), triples_to_canon_text(converted))
    return transform_obj_coordinates(replaced, transform_box)


def rescale_assistant_text_bboxes(
    assistant_text: str,
    orig_w: int,
    orig_h: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = 28,
    *,
    transform_obj_points: bool = True,
) -> Optional[str]:
    """Map original-pixel answer boxes and anchors into Qwen's resized image."""

    if not transform_obj_points:
        raise ValueError(
            "VGThink uses one pixel coordinate space for trace anchors and answer boxes"
        )
    new_h, new_w = smart_resize(orig_h, orig_w, factor, min_pixels, max_pixels)
    scale_w, scale_h = new_w / float(orig_w), new_h / float(orig_h)
    return _convert_response(
        assistant_text,
        transform_box=lambda box: scale_box_xyxy(box, scale_w, scale_h, new_w, new_h),
    )


def restore_assistant_text_bboxes_to_original(
    assistant_text: str,
    orig_w: int,
    orig_h: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = 28,
    *,
    transform_obj_points: bool = True,
) -> Optional[str]:
    """Restore generated resized-pixel boxes and anchors to original pixels."""

    if not transform_obj_points:
        raise ValueError(
            "VGThink uses one pixel coordinate space for trace anchors and answer boxes"
        )
    new_h, new_w = smart_resize(orig_h, orig_w, factor, min_pixels, max_pixels)
    scale_w, scale_h = new_w / float(orig_w), new_h / float(orig_h)
    return _convert_response(
        assistant_text,
        transform_box=lambda box: inverse_scale_box_xyxy(box, scale_w, scale_h, orig_w, orig_h),
    )


__all__ = [
    "inverse_scale_box_xyxy",
    "iou",
    "max_iou",
    "region_correct",
    "rescale_assistant_text_bboxes",
    "restore_assistant_text_bboxes_to_original",
    "scale_box_xyxy",
    "smart_resize",
    "transform_obj_coordinates",
]
