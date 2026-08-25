#!/usr/bin/env python3
"""Prepare the annotation-grounded VGThink targets from E2E-GMNER SFT JSONL.

The official answer is the only grounding source.  Its ordered
``mention|type|box/None`` records produce both the Stage-I point trace and the
normalized box metadata used by the Stage-II set-mass objective.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable, Sequence

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e2egmner.evaluation.parsing import (  # noqa: E402
    EntityTriple,
    parse_triples,
)


THINK_RE = re.compile(r"<think>(.*?)</think>", flags=re.I | re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.I | re.S)


def _source_sections(response: str) -> tuple[str, str]:
    """Return the teacher rationale and the exact official answer payload."""

    think = THINK_RE.search(response)
    answer = ANSWER_RE.search(response)
    teacher_think = think.group(1) if think else ""
    if answer:
        return teacher_think, answer.group(1)
    if think:
        tail = response[think.end() :].strip()
        if tail:
            return teacher_think, tail
        raise ValueError("assistant response has <think> but no answer")
    return "", response.strip()


def canonical_point(box: Sequence[float]) -> str:
    """Project an official box to the integer center used in the paper."""

    x1, y1, x2, y2 = (float(value) for value in box)
    return f"[{int(round((x1 + x2) / 2.0))},{int(round((y1 + y2) / 2.0))}]"


def canonical_grounded_suffix(triples: Sequence[EntityTriple]) -> str:
    """Serialize the ordered, entity-indexed annotation-grounded trace."""

    if not triples:
        return (
            "Entity-centric Grounded Reasoning:\n"
            "1. Text entity inventory: no licensed named entity is present.\n"
            "2. Text-primary entity typing: no type is assigned.\n"
            "3. Visual support before localization: no entity-specific region is licensed.\n"
            "4. Trace-answer consistency: the empty inventory agrees with the final answer."
        )

    inventory = ", ".join(f'"{triple.text}"' for triple in triples)
    typed = "; ".join(f'"{triple.text}" is {triple.etype}' for triple in triples)
    lines = [
        "Entity-centric Grounded Reasoning:",
        f"1. Text entity inventory: {inventory}.",
        f"2. Text-primary entity typing: {typed}.",
        "3. Visual support before localization:",
    ]
    for triple in triples:
        if triple.regions:
            anchors = " ".join(
                f"<obj>{triple.text}|{canonical_point(box)}</obj>" for box in triple.regions
            )
            lines.append(
                f'- "{triple.text}" has reliable entity-specific visual support; '
                f"localization is licensed: {anchors}"
            )
        else:
            lines.append(
                f'- "{triple.text}" lacks reliable task-valid visual attribution; '
                f"localization is rejected: <ungroundable>{triple.text}</ungroundable>"
            )
    lines.append(
        "4. Trace-answer consistency: the text inventory, types, support states, "
        "conditional locations, and final answer agree in the same entity order."
    )
    return "\n".join(lines)


def _parse_official_answer(answer: str, *, where: str) -> list[EntityTriple]:
    stripped = answer.strip()
    if not stripped:
        raise ValueError(f"[{where}] empty official answer")
    triples = parse_triples(stripped, strict=True, where=where)
    if stripped.casefold() != "none" and not triples:
        raise ValueError(f"[{where}] official answer contains no parseable records")
    for triple in triples:
        if not triple.region_valid:
            raise ValueError(f"[{where}] invalid region for {triple.text!r}")
        for box in triple.regions or []:
            if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
                raise ValueError(f"[{where}] non-finite box for {triple.text!r}: {box}")
    return triples


def _region_target(
    triple: EntityTriple,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    boxes = [
        [
            float(x1) / width,
            float(y1) / height,
            float(x2) / width,
            float(y2) / height,
        ]
        for x1, y1, x2, y2 in (triple.regions or [])
    ]
    points = [[(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0] for box in boxes]
    return {
        "text": triple.text,
        "etype": triple.etype,
        "visible": bool(triple.regions),
        "loss_enabled": True,
        "source": "official_answer",
        "boxes": boxes,
        "points": points,
    }


def convert_record(
    record: dict[str, Any],
    image_size: tuple[int, int],
    *,
    where: str = "record",
) -> dict[str, Any]:
    """Convert one record without mutating it."""

    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError(f"[{where}] invalid image size: {image_size}")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"[{where}] record has no messages")
    if messages[-1].get("role") != "assistant":
        raise ValueError(f"[{where}] record must end with an assistant message")

    response = str(messages[-1].get("content") or "")
    teacher_think, answer = _source_sections(response)
    triples = _parse_official_answer(answer, where=where)
    suffix = canonical_grounded_suffix(triples)
    reasoning = f"{teacher_think}\n\n{suffix}" if teacher_think else suffix

    converted = deepcopy(record)
    converted["messages"][-1]["content"] = f"<think>{reasoning}</think><answer>{answer}</answer>"
    converted["vgthink_variant"] = "annotation_grounded_cot"
    converted["region_targets"] = [
        _region_target(triple, width=width, height=height) for triple in triples
    ]
    converted["grounded_entity_count"] = sum(bool(triple.regions) for triple in triples)
    converted["region_schema_version"] = 1
    return converted


def _image_path(record: dict[str, Any], image_root: Path, *, where: str) -> Path:
    images = record.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"[{where}] record has no image path")
    path = Path(str(images[0]))
    return path if path.is_absolute() else image_root / path


def convert_file(
    input_path: Path,
    output_path: Path,
    image_root: Path,
) -> dict[str, int]:
    """Convert one JSONL file and replace its destination atomically."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    counts: Counter[str] = Counter()
    try:
        with (
            input_path.open("r", encoding="utf-8") as source,
            os.fdopen(fd, "w", encoding="utf-8") as sink,
        ):
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                where = f"{input_path}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {where}: {exc}") from exc
                image_path = _image_path(record, image_root, where=where)
                with Image.open(image_path) as image:
                    converted = convert_record(record, image.size, where=where)
                sink.write(json.dumps(converted, ensure_ascii=False) + "\n")
                counts["records"] += 1
                counts["entities"] += len(converted["region_targets"])
                counts["visible"] += converted["grounded_entity_count"]
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise
    counts["ungroundable"] = counts["entities"] - counts["visible"]
    return dict(counts)


def _splits(values: Iterable[str]) -> list[str]:
    splits = [part.strip() for value in values for part in value.split(",") if part.strip()]
    if not splits:
        raise ValueError("at least one split is required")
    return list(dict.fromkeys(splits))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/sft")
    parser.add_argument("--output-dir", default="data/vgthink")
    parser.add_argument("--image-root", default="data")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root)
    for split in _splits(args.splits):
        stats = convert_file(
            input_dir / f"{split}_sft.jsonl",
            output_dir / f"{split}.jsonl",
            image_root,
        )
        print(f"{split}: {json.dumps(stats, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
