"""Deterministic Stage-III rewards from Eqs. (27)--(32)."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from html import unescape
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from ..evaluation.geometry import max_iou
from ..evaluation.metrics import (
    count_correct_eeg,
    count_correct_gmner,
    count_correct_mner,
)
from ..evaluation.parsing import (
    EntityTriple,
    extract_answer_text,
    normalize_separators,
    split_entity_segments,
)

_NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_RESPONSE_RE = re.compile(
    r"\A\s*<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>\s*\Z",
    re.IGNORECASE | re.DOTALL,
)
_SECTION_TAG_RE = re.compile(r"<\s*/?\s*(?:think|answer)\b", re.IGNORECASE)
_TRACE_TAG_RE = re.compile(r"<\s*/?\s*(?:obj|ungroundable)\b", re.IGNORECASE)
_TRACE_RECORD_RE = re.compile(
    r"(?P<obj><obj>(?P<obj_body>.*?)</obj>)|"
    r"(?P<ung><ungroundable>(?P<ung_body>.*?)</ungroundable>)",
    re.IGNORECASE | re.DOTALL,
)
_POINT_RE = re.compile(
    rf"\A\s*(?P<mention>[^<>]+?)\s*\|\s*"
    rf"\[\s*(?P<x>{_NUMBER})\s*,\s*(?P<y>{_NUMBER})\s*\]\s*\Z"
)
_TOKEN_RE = re.compile(r"[#@]?[\w]+", re.UNICODE)


@dataclass(frozen=True)
class RewardBreakdown:
    """Auditable raw terms and weighted contributions for one response."""

    total: float
    valid: bool
    task_reward: float
    fine_reward: float
    consistency_reward: float
    format_reward: float
    parse_penalty: float
    gmner_f1: float
    mner_f1: float
    eeg_f1: float
    count_reward: float
    span_reward: float
    type_reward: float
    support_reward: float
    localization_reward: float
    pred_count: int
    gold_count: int
    trace_count: int

    @property
    def fine_contribution(self) -> float:
        return 0.20 * self.fine_reward

    @property
    def consistency_contribution(self) -> float:
        return 0.25 * self.consistency_reward

    @property
    def format_contribution(self) -> float:
        return 0.10 * self.format_reward

    @property
    def parse_penalty_contribution(self) -> float:
        return -0.20 * self.parse_penalty


@dataclass(frozen=True)
class _TraceRecord:
    mention: str
    support: str
    anchors: tuple[tuple[float, float], ...] = ()


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _box(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4 or not all(_is_number(v) for v in value):
        return None
    x1, y1, x2, y2 = (float(v) for v in value)
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def _parse_regions(text: str) -> list[tuple[float, float, float, float]] | None:
    if text.strip().casefold() == "none":
        return None
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        raise ValueError("invalid region literal") from None

    one = _box(value)
    if one is not None:
        return [one]
    if not isinstance(value, list) or not value:
        raise ValueError("a region must contain one or more valid boxes")
    boxes = [_box(item) for item in value]
    if any(item is None for item in boxes):
        raise ValueError("invalid box in region list")
    return [item for item in boxes if item is not None]


def _parse_answer_strict(answer: str, *, where: str) -> list[EntityTriple]:
    answer = normalize_separators(answer or "").strip()
    if answer.casefold() == "none":
        return []
    if not answer:
        raise ValueError(f"[{where}] an empty entity set must be written as None")

    triples: list[EntityTriple] = []
    segments = split_entity_segments(answer)
    if not segments:
        raise ValueError(f"[{where}] no entity records")
    for segment in segments:
        parts = [part.strip() for part in segment.rsplit("|", 2)]
        if len(parts) != 3 or not parts[0] or not parts[1]:
            raise ValueError(f"[{where}] malformed entity record: {segment!r}")
        mention, entity_type, region_text = parts
        if "<" in mention or ">" in mention or "<" in entity_type or ">" in entity_type:
            raise ValueError(f"[{where}] tags are not allowed inside an answer record")
        regions = _parse_regions(region_text)
        triples.append(EntityTriple(mention, entity_type, regions, region_valid=True))
    return triples


def _parse_trace_strict(think: str) -> list[_TraceRecord]:
    records: list[tuple[_TraceRecord, tuple[int, int]]] = []
    spans: list[tuple[int, int]] = []
    for match in _TRACE_RECORD_RE.finditer(think):
        spans.append(match.span())
        if match.group("obj") is not None:
            point = _POINT_RE.fullmatch(match.group("obj_body") or "")
            if point is None:
                raise ValueError("malformed <obj> record")
            mention = point.group("mention").strip()
            x, y = float(point.group("x")), float(point.group("y"))
            if not mention or not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("invalid <obj> record")
            records.append((_TraceRecord(mention, "obj", ((x, y),)), match.span()))
        else:
            mention = (match.group("ung_body") or "").strip()
            if not mention or "<" in mention or ">" in mention:
                raise ValueError("malformed <ungroundable> record")
            records.append((_TraceRecord(mention, "ung"), match.span()))

    remainder = list(think)
    for start, end in spans:
        remainder[start:end] = " " * (end - start)
    if _TRACE_TAG_RE.search("".join(remainder)):
        raise ValueError("unclosed, nested, or malformed trace tag")

    # A multi-region answer is serialized by the data builder as one <obj>
    # anchor per official box.  Those anchors still constitute one entity
    # record.  Unsupported markers remain separate so duplicates/conflicts
    # are penalized by Eq. (31)'s denominator.
    grouped: list[tuple[_TraceRecord, int]] = []
    for record, (start, end) in records:
        previous = grouped[-1][0] if grouped else None
        previous_end = grouped[-1][1] if grouped else None
        if (
            record.support == "obj"
            and previous is not None
            and previous.support == "obj"
            and _normalize_mention(previous.mention) == _normalize_mention(record.mention)
            and not think[previous_end:start].strip()
        ):
            grouped[-1] = (
                _TraceRecord(previous.mention, "obj", previous.anchors + record.anchors),
                end,
            )
        else:
            grouped.append((record, end))
    return [record for record, _ in grouped]


def _parse_response(text: str) -> tuple[list[_TraceRecord], list[EntityTriple]]:
    match = _RESPONSE_RE.fullmatch(text or "")
    if (
        match is None
        or not (match.group("think") or "").strip()
        or len(_SECTION_TAG_RE.findall(text or "")) != 4
    ):
        raise ValueError("response must contain exactly one ordered <think>-<answer> pair")
    traces = _parse_trace_strict(match.group("think") or "")
    triples = _parse_answer_strict(match.group("answer") or "", where="prediction")
    return traces, triples


def _sample_f1(correct: int, predicted: int, gold: int) -> float:
    if predicted == gold == 0:
        return 1.0
    return 2.0 * correct / (predicted + gold)


def _count_reward(predicted: int, gold: int) -> float:
    if predicted == gold:
        return 1.0
    if predicted > gold:
        weight = 0.4 if gold <= 2 else 0.2 if gold <= 4 else 0.1
        return max(0.0, 1.0 - (predicted - gold) * weight)
    weight = 0.5 if predicted <= 2 else 0.3 if predicted <= 4 else 0.2
    return max(0.0, 1.0 - (gold - predicted) * weight)


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", unescape(text or "")).casefold()
    return _TOKEN_RE.findall(normalized)


def _span_f1(prediction: str, gold: str) -> float:
    pred_tokens, gold_tokens = _tokens(prediction), _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    previous = [0] * (len(gold_tokens) + 1)
    longest = 0
    for pred_token in pred_tokens:
        current = [0] * (len(gold_tokens) + 1)
        for index, gold_token in enumerate(gold_tokens, 1):
            if pred_token == gold_token:
                current[index] = previous[index - 1] + 1
                longest = max(longest, current[index])
        previous = current
    return 2.0 * longest / (len(pred_tokens) + len(gold_tokens))


def _mention_pairs(
    predictions: Sequence[EntityTriple], golds: Sequence[EntityTriple]
) -> list[tuple[int, int, float]]:
    scores = {
        (i, j): _span_f1(pred.text, gold.text)
        for i, pred in enumerate(predictions)
        for j, gold in enumerate(golds)
    }

    @lru_cache(maxsize=None)
    def best(i: int, used: int) -> tuple[float, tuple[tuple[int, int, float], ...]]:
        if i == len(predictions):
            return 0.0, ()
        best_score, best_pairs = best(i + 1, used)
        for j in range(len(golds)):
            score = scores[i, j]
            if score <= 0.0 or used & (1 << j):
                continue
            tail_score, tail_pairs = best(i + 1, used | (1 << j))
            candidate = score + tail_score
            if candidate > best_score:
                best_score = candidate
                best_pairs = ((i, j, score),) + tail_pairs
        return best_score, best_pairs

    return list(best(0, 0)[1])


def _fine_rewards(
    predictions: list[EntityTriple],
    golds: list[EntityTriple],
    *,
    iou_threshold: float,
) -> tuple[float, float, float, float, float, float]:
    pairs = _mention_pairs(predictions, golds)
    denominator = max(len(predictions), len(golds), 1)
    count = _count_reward(len(predictions), len(golds))
    span = sum(score for _, _, score in pairs) / denominator
    entity_type = (
        sum(score * float(predictions[i].etype == golds[j].etype) for i, j, score in pairs)
        / denominator
    )
    support = (
        sum(
            score * float((predictions[i].regions is None) == (golds[j].regions is None))
            for i, j, score in pairs
        )
        / denominator
    )

    visible = sum(gold.regions is not None for gold in golds)
    if visible == 0:
        localization = 1.0
    else:
        localization_sum = 0.0
        for i, j, score in pairs:
            pred, gold = predictions[i], golds[j]
            overlap = 0.0
            if pred.regions is not None and gold.regions is not None:
                overlap = max_iou(pred.regions, gold.regions)
            mapped = max(0.0, (overlap - iou_threshold) / (1.0 - iou_threshold))
            if gold.regions is not None:
                localization_sum += score * mapped
        localization = localization_sum / visible

    fine = (count + span + entity_type + support + localization) / 5.0
    return count, span, entity_type, support, localization, fine


def _normalize_mention(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", unescape(text or "")).casefold()
    return " ".join(normalized.split())


def _point_in_regions(
    point: tuple[float, float], regions: list[tuple[float, float, float, float]]
) -> bool:
    x, y = point
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in regions)


def _consistent(trace: _TraceRecord, answer: EntityTriple) -> bool:
    if _normalize_mention(trace.mention) != _normalize_mention(answer.text):
        return False
    if trace.support == "ung":
        return answer.regions is None
    return (
        answer.regions is not None
        and bool(trace.anchors)
        and all(_point_in_regions(anchor, answer.regions) for anchor in trace.anchors)
    )


def _consistency_reward(traces: list[_TraceRecord], answers: list[EntityTriple]) -> float:
    if not traces and not answers:
        return 1.0

    edges = {
        i: [j for j, answer in enumerate(answers) if _consistent(trace, answer)]
        for i, trace in enumerate(traces)
    }
    answer_to_trace: dict[int, int] = {}

    def augment(trace_index: int, seen: set[int]) -> bool:
        for answer_index in edges[trace_index]:
            if answer_index in seen:
                continue
            seen.add(answer_index)
            owner = answer_to_trace.get(answer_index)
            if owner is None or augment(owner, seen):
                answer_to_trace[answer_index] = trace_index
                return True
        return False

    correct = sum(augment(i, set()) for i in range(len(traces)))
    return correct / max(len(traces), len(answers))


class GMNERReward:
    """Compute the complete Stage-III verifier reward for one rollout."""

    def __init__(self, *, iou_threshold: float = 0.5):
        if not 0.0 <= iou_threshold < 1.0:
            raise ValueError("iou_threshold must be in [0, 1)")
        self.iou_threshold = float(iou_threshold)

    def __call__(self, prediction_text: str, gold_text: str) -> RewardBreakdown:
        golds = _parse_answer_strict(extract_answer_text(gold_text), where="gold")
        try:
            traces, predictions = _parse_response(prediction_text)
        except ValueError:
            return RewardBreakdown(
                total=-0.20,
                valid=False,
                task_reward=0.0,
                fine_reward=0.0,
                consistency_reward=0.0,
                format_reward=0.0,
                parse_penalty=1.0,
                gmner_f1=0.0,
                mner_f1=0.0,
                eeg_f1=0.0,
                count_reward=0.0,
                span_reward=0.0,
                type_reward=0.0,
                support_reward=0.0,
                localization_reward=0.0,
                pred_count=0,
                gold_count=len(golds),
                trace_count=0,
            )

        predicted_count, gold_count = len(predictions), len(golds)
        gmner_f1 = _sample_f1(
            count_correct_gmner(predictions, golds, self.iou_threshold),
            predicted_count,
            gold_count,
        )
        mner_f1 = _sample_f1(count_correct_mner(predictions, golds), predicted_count, gold_count)
        eeg_f1 = _sample_f1(
            count_correct_eeg(predictions, golds, self.iou_threshold),
            predicted_count,
            gold_count,
        )
        task = 0.50 * gmner_f1 + 0.25 * mner_f1 + 0.25 * eeg_f1
        count, span, entity_type, support, localization, fine = _fine_rewards(
            predictions, golds, iou_threshold=self.iou_threshold
        )
        consistency = _consistency_reward(traces, predictions)
        total = task + 0.20 * fine + 0.25 * consistency + 0.10

        return RewardBreakdown(
            total=float(total),
            valid=True,
            task_reward=float(task),
            fine_reward=float(fine),
            consistency_reward=float(consistency),
            format_reward=1.0,
            parse_penalty=0.0,
            gmner_f1=float(gmner_f1),
            mner_f1=float(mner_f1),
            eeg_f1=float(eeg_f1),
            count_reward=float(count),
            span_reward=float(span),
            type_reward=float(entity_type),
            support_reward=float(support),
            localization_reward=float(localization),
            pred_count=predicted_count,
            gold_count=gold_count,
            trace_count=len(traces),
        )


def compute_reward(
    prediction_text: str, gold_text: str, *, iou_threshold: float = 0.5
) -> RewardBreakdown:
    """Functional interface for :class:`GMNERReward`."""

    return GMNERReward(iou_threshold=iou_threshold)(prediction_text, gold_text)
