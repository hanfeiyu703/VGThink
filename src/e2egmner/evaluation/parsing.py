"""Parse and serialize the structured answer records used by VGThink."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Optional


BAR_VARIANTS = ("｜", "│", "丨", "￨", "∣")
ENTITY_SPLIT_RE = re.compile(r";|；|\n+")
HTML_ENTITY_PREFIX_RE = re.compile(r"&(?:#[0-9]+|#[xX][0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+)\s*$")
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class EntityTriple:
    text: str
    etype: str
    regions: Optional[list[Box]]
    region_valid: bool = True


def normalize_separators(text: str) -> str:
    text = text or ""
    for variant in BAR_VARIANTS:
        text = text.replace(variant, "|")
    return text


def split_entity_segments(text: str) -> list[str]:
    """Split records without treating an HTML-entity semicolon as a separator."""

    segments: list[str] = []
    start = 0
    for match in ENTITY_SPLIT_RE.finditer(text):
        if match.group(0) == ";" and HTML_ENTITY_PREFIX_RE.search(text[: match.start()]):
            continue
        segment = text[start : match.start()].strip()
        if segment:
            segments.append(segment)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        segments.append(tail)
    return segments


def extract_answer_text(text: Optional[str]) -> str:
    """Extract an answer body from a complete response or return bare input."""

    if not text:
        return ""
    text = text.strip()
    tagged = ANSWER_TAG_RE.search(text)
    if tagged:
        return (tagged.group(1) or "").strip()
    after_think = re.search(r"</think>\s*(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
    return (after_think.group(1) or "").strip() if after_think else text


def replace_answer_text(full_text: Optional[str], answer: str) -> str:
    """Replace an answer body while preserving the surrounding response."""

    answer = (answer or "").strip()
    if not full_text:
        return answer
    text = full_text.strip()
    if ANSWER_TAG_RE.search(text):
        return ANSWER_TAG_RE.sub(lambda _: f"<answer>{answer}</answer>", text, count=1)
    if re.search(r"</think>", text, flags=re.IGNORECASE):
        return re.sub(
            r"(</think>\s*).*$",
            lambda match: f"{match.group(1)}<answer>{answer}</answer>",
            text,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )
    return answer


def _invalid(message: str, *, strict: bool) -> tuple[None, bool]:
    if strict:
        raise ValueError(message)
    return None, False


def _box(values: object, *, strict: bool, where: str) -> Optional[Box]:
    if (
        not isinstance(values, list)
        or len(values) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        if strict:
            raise ValueError(f"[{where}] invalid box: {values!r}")
        return None
    x1, y1, x2, y2 = (float(value) for value in values)
    if x1 >= x2 or y1 >= y2:
        if strict:
            raise ValueError(f"[{where}] unordered or empty box: {values!r}")
        return None
    return x1, y1, x2, y2


def _boxes_from_payload(
    payload: object, *, strict: bool, where: str
) -> tuple[Optional[list[Box]], bool]:
    if (
        isinstance(payload, list)
        and len(payload) == 4
        and not any(isinstance(value, list) for value in payload)
    ):
        box = _box(payload, strict=strict, where=where)
        return ([box], True) if box is not None else (None, False)
    if isinstance(payload, list) and payload and all(isinstance(value, list) for value in payload):
        boxes = [_box(value, strict=strict, where=where) for value in payload]
        if any(box is None for box in boxes):
            return None, False
        return [box for box in boxes if box is not None], True
    return _invalid(f"[{where}] invalid region payload: {payload!r}", strict=strict)


def parse_bbox_regions(
    bbox_text: str, *, strict: bool, where: str
) -> tuple[Optional[list[Box]], bool]:
    """Parse one box, a non-empty box list, or the unsupported marker None."""

    text = (bbox_text or "").strip()
    if not text:
        return _invalid(f"[{where}] empty bbox string", strict=strict)
    if text.casefold() in {"none", "n/a"}:
        return None, True

    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not numbers or len(numbers) % 4:
            return _invalid(f"[{where}] cannot parse bbox: {bbox_text!r}", strict=strict)
        raw_boxes = [
            [float(value) for value in numbers[index : index + 4]]
            for index in range(0, len(numbers), 4)
        ]
        payload = raw_boxes[0] if len(raw_boxes) == 1 else raw_boxes
    return _boxes_from_payload(payload, strict=strict, where=where)


def parse_triple_segment(segment: str, *, strict: bool, where: str) -> Optional[EntityTriple]:
    segment = normalize_separators(segment.strip())
    if not segment or segment.casefold() == "none":
        return None
    parts = segment.rsplit("|", 2)
    if len(parts) != 3:
        if strict:
            raise ValueError(f"[{where}] cannot parse segment: {segment!r}")
        return None
    mention, entity_type, region_text = (part.strip() for part in parts)
    if not mention or not entity_type:
        if strict:
            raise ValueError(f"[{where}] empty mention/type: {segment!r}")
        return None
    regions, valid = parse_bbox_regions(region_text, strict=strict, where=where)
    return EntityTriple(mention, entity_type, regions, region_valid=valid)


def parse_triples(text: str, *, strict: bool, where: str) -> list[EntityTriple]:
    text = normalize_separators(text or "").strip()
    if not text or text.casefold() == "none":
        return []
    triples: list[EntityTriple] = []
    for segment in split_entity_segments(text):
        triple = parse_triple_segment(segment, strict=strict, where=where)
        if triple is not None:
            triples.append(triple)
    return triples


def _structured_answer_parse_failed(answer: str) -> bool:
    return bool((answer or "").strip()) and (answer or "").strip().casefold() != "none"


def triples_to_canon_text(triples: list[EntityTriple]) -> str:
    records: list[str] = []
    for triple in triples:
        if triple.regions is None:
            region_text = "None"
        else:
            boxes = [[int(round(value)) for value in box] for box in triple.regions]
            region_text = (
                json.dumps(boxes[0], separators=(",", ":"))
                if len(boxes) == 1
                else json.dumps(boxes, separators=(",", ":"))
            )
        records.append(f"{triple.text}|{triple.etype}|{region_text}")
    return "; ".join(records)


__all__ = [
    "Box",
    "EntityTriple",
    "extract_answer_text",
    "normalize_separators",
    "parse_bbox_regions",
    "parse_triple_segment",
    "parse_triples",
    "replace_answer_text",
    "split_entity_segments",
    "triples_to_canon_text",
]
